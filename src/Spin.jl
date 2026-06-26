"""
    Spin.jl — Spin-weighted (spin-s) non-uniform spherical harmonic transforms.

Generalizes the scalar DFS + nuFFT pipeline (`N·F·D·S`) to spin-weighted fields. Only two
steps change relative to the scalar transform:

- **S**: FastTransforms spin-weighted synthesis/analysis (`plan_spinsph2fourier`,
  `plan_spinsph_synthesis`) — complex-valued, on the same equiangular grid.
- **D**: the Double-Fourier-Sphere doubling carries a spin factor. A spin-`s` field's
  azimuthal Fourier coefficients extend across the south pole as
  `ₛG_m(2π−θ) = (−1)^(m+s) ₛG_m(θ)` (McEwen & Wiaux 2011), which in the spatial domain is the
  scalar pole reflection with the `φ→φ+π` shift (the `(−1)^m` part) times an extra `(−1)^s`.

The `F` (FFT, half-pixel θ phase correction) and `N` (FINUFFT) stages are reused unchanged
(now complex). Spin fields and their coefficients are complex.

Spin-1 (`s = ±1`) is the case needed for tangent vector fields: a tangent field
`U = U_θ + i U_φ` (equivalently `u_east + i·u_north`) is spin-1, and `U = ð(scalar)` links it
to scalar potentials — enabling vector Helmholtz decomposition at scattered points.

# Coefficient layout
Same column convention as the scalar `FastSphericalHarmonics.sph_mode`, with the row offset by
the spin floor: for degree `ℓ ≥ max(|m|,|s|)` and order `m`,
`row = ℓ − max(|m|,|s|) + 1`, `col = 2|m| + (m ≥ 0 ? 1 : 0)`.
"""

using FastTransforms: FastTransforms
using FFTW: FFTW
using FINUFFT: FINUFFT
using LinearAlgebra: LinearAlgebra

export SpinNUSHTplan, make_spin_plan, spin_sph_mode
export nusht_type2_spin!, nusht_type1_spin!, nusht_solve_spin!

"""
    spin_sph_mode(ℓ, m, s) -> CartesianIndex

Index into a spin-`s` coefficient array `(lmax+1, 2lmax+1)` for degree `ℓ` (with
`ℓ ≥ max(|m|,|s|)`) and order `m`.
"""
@inline function spin_sph_mode(ℓ::Integer, m::Integer, s::Integer)
    return CartesianIndex(ℓ - max(abs(m), abs(s)) + 1, 2 * abs(m) + (m ≥ 0 ? 1 : 0))
end

"""
    SpinNUSHTplan{T}

Plan for spin-weighted non-uniform spherical harmonic transforms at `M` scattered points up to
degree `lmax`, spin `s`. Mirrors `NUSHTplan` but with complex buffers and the spin-weighted
FastTransforms plans.
"""
struct SpinNUSHTplan{T<:AbstractFloat}
    lmax::Int
    Nθ::Int
    Nφ::Int
    s::Int
    C::Matrix{Complex{T}}
    F::Matrix{Complex{T}}
    F̃::Matrix{Complex{T}}
    θ_nodes::Vector{T}
    φ_nodes::Vector{T}
    tol::Float64
    fft_plan::FFTW.Plan
    ifft_plan::FFTW.Plan
    phase_θ::Vector{Complex{T}}
    phase_θ_conj::Vector{Complex{T}}
    sph_plan::FastTransforms.FTPlan          # plan_spinsph2fourier
    sph_plan_synth::FastTransforms.FTPlan    # plan_spinsph_synthesis
end

"""
    make_spin_plan(θ_nodes, φ_nodes, lmax, s; tol=1e-8)

Construct a `SpinNUSHTplan` for spin `s` (real `Float64` pipeline; FastTransforms' spin
transforms are `Complex{Float64}`).
"""
function make_spin_plan(θ_nodes, φ_nodes, lmax, s; tol = 1e-8)
    T = Float64
    @assert length(θ_nodes) == length(φ_nodes)
    Nθ = lmax + 1
    Nφ = 2lmax + 1

    C = zeros(Complex{T}, Nθ, Nφ)
    F = zeros(Complex{T}, Nθ, Nφ)
    F̃ = zeros(Complex{T}, 2Nθ, Nφ)

    θ = Vector{T}(θ_nodes)
    φ = Vector{T}(φ_nodes)

    fft_plan = FFTW.plan_fft(F̃)
    ifft_plan = FFTW.plan_ifft(F̃)

    Nθ_dbl = 2Nθ
    k_θ = [k < Nθ_dbl ÷ 2 ? k : k - Nθ_dbl for k in 0:(Nθ_dbl - 1)]
    phase_θ = exp.(-im .* π .* T.(k_θ) ./ Nθ_dbl)
    phase_θ_conj = conj.(phase_θ)

    sph_plan = FastTransforms.plan_spinsph2fourier(C, s)
    sph_plan_synth = FastTransforms.plan_spinsph_synthesis(C, s)

    return SpinNUSHTplan{T}(lmax, Nθ, Nφ, s, C, F, F̃, θ, φ, Float64(tol),
        fft_plan, ifft_plan, phase_θ, phase_θ_conj, sph_plan, sph_plan_synth)
end

# Spin-weighted DFS doubling: scalar pole reflection (φ+π shift) × (−1)^s.
function dfs_double_spin!(F̃::AbstractMatrix, F::AbstractMatrix, s::Int)
    Nθ, Nφ = size(F)
    @assert size(F̃) == (2Nθ, Nφ)
    sgn = isodd(s) ? -1 : 1
    half = Nφ ÷ 2
    @inbounds F̃[1:Nθ, :] .= F
    @inbounds for i in 1:Nθ, j in 1:Nφ
        F̃[Nθ + i, j] = sgn * F[Nθ + 1 - i, mod1(j + half, Nφ)]
    end
    return F̃
end

# Adjoint of dfs_double_spin! ((−1)^s is real ⇒ same factor on the folded mirror term).
function dfs_fold_spin!(F::AbstractMatrix, F̃::AbstractMatrix, s::Int)
    Nθ, Nφ = size(F)
    @assert size(F̃) == (2Nθ, Nφ)
    sgn = isodd(s) ? -1 : 1
    half = Nφ ÷ 2
    @inbounds for i in 1:Nθ, j in 1:Nφ
        F[i, j] = F̃[i, j] + sgn * F̃[Nθ + (Nθ + 1 - i), mod1(j - half, Nφ)]
    end
    return F
end

# Complex analogues of fft2_to_coeffs / ifft2_from_coeffs (identical phase/shift logic).
function _fft2_to_coeffs_spin(F̃, plan::SpinNUSHTplan)
    Nθ_dbl = 2 * plan.Nθ
    Nφ = plan.Nφ
    Fhat_raw = plan.fft_plan * F̃
    Fhat_corrected = (Fhat_raw .* plan.phase_θ) ./ (Nθ_dbl * Nφ)
    return collect(FFTW.fftshift(Fhat_corrected)')
end

function _ifft2_from_coeffs_spin(Fhat_2d, plan::SpinNUSHTplan)
    Fhat_shifted = FFTW.ifftshift(collect(Fhat_2d'))
    return plan.ifft_plan * (Fhat_shifted .* plan.phase_θ_conj)
end

"""
    nusht_type2_spin!(f, C, plan) -> f

Spin-weighted synthesis: evaluate the spin-`s` field with coefficients `C` at the `M`
scattered points, writing complex values into `f`.
"""
function nusht_type2_spin!(f, C, plan::SpinNUSHTplan{T}) where {T}
    @assert length(f) == length(plan.θ_nodes)
    @assert size(C) == (plan.Nθ, plan.Nφ)

    F = plan.F
    F .= C
    LinearAlgebra.lmul!(plan.sph_plan, F)
    LinearAlgebra.lmul!(plan.sph_plan_synth, F)

    dfs_double_spin!(plan.F̃, F, plan.s)
    Fhat_2d = _fft2_to_coeffs_spin(plan.F̃, plan)
    f .= FINUFFT.nufft2d2(plan.φ_nodes, plan.θ_nodes, -1, plan.tol, Fhat_2d)
    return f
end

"""
    nusht_type1_spin!(C, f, plan) -> C

True Euclidean adjoint of `nusht_type2_spin!`: scattered complex values `f` → spin-`s`
coefficients `C`. Uses the transposes of the FastTransforms spin plans (`PS'·P'`), so
`(A†A)` is Hermitian PSD and CG converges (see `nusht_solve_spin!`).
"""
function nusht_type1_spin!(C, f, plan::SpinNUSHTplan{T}) where {T}
    @assert length(f) == length(plan.θ_nodes)
    @assert size(C) == (plan.Nθ, plan.Nφ)

    Fhat_vec = FINUFFT.nufft2d1(plan.φ_nodes, plan.θ_nodes, Complex{T}.(f), +1, plan.tol, plan.Nφ, 2plan.Nθ)
    Fhat_2d = dropdims(Fhat_vec; dims = 3)
    F̃ = _ifft2_from_coeffs_spin(Fhat_2d, plan)
    dfs_fold_spin!(plan.F, F̃, plan.s)

    C .= plan.F
    LinearAlgebra.lmul!(plan.sph_plan_synth', C)
    LinearAlgebra.lmul!(plan.sph_plan', C)
    return C
end

"""
    nusht_solve_spin!(C, f, plan; maxiter=500, rtol=1e-8, verbose=false)

Exact inversion of the spin-weighted synthesis at arbitrary scattered points: solve `A c = f`
for spin-`s` coefficients `C` by conjugate gradients on the normal equations
`(A†A) c = A† f`. Returns `(C, iters, rel_res)`.
"""
function nusht_solve_spin!(C, f, plan::SpinNUSHTplan{T}; maxiter::Int = 500, rtol::Real = 1e-8, verbose::Bool = false) where {T}
    Nθ, Nφ = plan.Nθ, plan.Nφ
    @assert size(C) == (Nθ, Nφ)
    @assert length(f) == length(plan.θ_nodes)

    bufC = zeros(Complex{T}, Nθ, Nφ)
    buff = zeros(Complex{T}, length(f))
    bufC2 = zeros(Complex{T}, Nθ, Nφ)

    function AHA!(y, x)
        bufC .= reshape(x, Nθ, Nφ)
        nusht_type2_spin!(buff, bufC, plan)
        nusht_type1_spin!(bufC2, buff, plan)
        y .= vec(bufC2)
    end

    nusht_type1_spin!(bufC, f, plan)
    rhs = vec(copy(bufC))
    rhs_norm = LinearAlgebra.norm(rhs)
    K = Nθ * Nφ
    x = zeros(Complex{T}, K)
    r = copy(rhs)
    p = copy(r)
    Ap = zeros(Complex{T}, K)
    rsold = real(LinearAlgebra.dot(r, r))

    rel_res = one(T)
    iters = 0
    rhs_norm == 0 && (C .= reshape(x, Nθ, Nφ); return C, 0, zero(T))
    for i in 1:maxiter
        iters = i
        AHA!(Ap, p)
        α = rsold / real(LinearAlgebra.dot(p, Ap))
        x .+= α .* p
        r .-= α .* Ap
        rsnew = real(LinearAlgebra.dot(r, r))
        rel_res = sqrt(rsnew) / rhs_norm
        verbose && @info "nusht_solve_spin! iter $i: rel_res=$rel_res"
        rel_res < rtol && break
        p .= r .+ (rsnew / rsold) .* p
        rsold = rsnew
    end
    C .= reshape(x, Nθ, Nφ)
    return C, iters, rel_res
end
