"""
    Spin.jl — Spin-weighted (spin-s) non-uniform spherical harmonic transforms.

Self-contained spin-weighted synthesis/analysis at arbitrary scattered points, built directly
from the Wigner-d "Fourier factorization" + a 2-D nonuniform FFT — independent of any
spin-weighted convention in FastTransforms.

A spin-`s` band-limited field is

    f(θ,φ) = Σ_{ℓ,m} ₛf_{ℓm} · ₛY_{ℓm}(θ,φ),
    ₛY_{ℓm}(θ,φ) = N_ℓ · d^ℓ_{m,−s}(θ) · e^{imφ},   N_ℓ = √((2ℓ+1)/4π),

where `d^ℓ` is the Wigner small-d. Using the Fourier factorization
`d^ℓ_{mn}(θ) = i^{m−n} Σ_{m'} Δ^ℓ_{m'm} Δ^ℓ_{m'n} e^{−im'θ}` with `Δ^ℓ ≡ d^ℓ(π/2)`, the field
becomes a bivariate Fourier series

    f(θ,φ) = Σ_{m',m} G_{m'm} e^{−im'θ} e^{imφ},
    G_{m'm} = i^{m+s} Σ_ℓ ₛf_{ℓm} N_ℓ Δ^ℓ_{m'm} Δ^ℓ_{m',−s},

evaluated at scattered points by a single 2-D NUFFT (FINUFFT type 2). Analysis is the exact
adjoint (FINUFFT type 1 + the transpose Δ-contraction); `nusht_solve_spin!` inverts at
arbitrary points by conjugate gradients on the normal equations.

Coefficients use a dense `(lmax+1) × (2lmax+1)` layout: `sf[ℓ+1, m+lmax+1]` for
`ℓ = 0..lmax`, `m = −ℓ..ℓ` (zero where `|m| > ℓ` or `ℓ < |s|`).

Spin `s = ±1` is the tangent-vector case: `U = u_θ + i u_φ` (≈ `u_east + i u_north`) is a
spin-1 field, enabling vector/Helmholtz operations at scattered points.
"""

using FINUFFT: FINUFFT
using LinearAlgebra: LinearAlgebra
using SpecialFunctions: loggamma

export SpinNUSHTplan, make_spin_plan
export nusht_type2_spin!, nusht_type1_spin!, nusht_solve_spin!
export spin_coeff_index, sYlm

"""
    wigner_d(ℓ, m, n, β)

Wigner small-d matrix element `d^ℓ_{mn}(β)` (Wikipedia/Varshalovich convention), computed by
the explicit alternating sum with log-gamma factorials for stability.
"""
function wigner_d(ℓ::Integer, m::Integer, n::Integer, β::Real)
    (abs(m) > ℓ || abs(n) > ℓ) && return 0.0
    c = cos(β / 2); s = sin(β / 2)
    kmin = max(0, n - m); kmax = min(ℓ + n, ℓ - m)
    pref = 0.5 * (loggamma(ℓ + m + 1) + loggamma(ℓ - m + 1) + loggamma(ℓ + n + 1) + loggamma(ℓ - n + 1))
    tot = 0.0
    for k in kmin:kmax
        den = loggamma(ℓ + n - k + 1) + loggamma(k + 1) + loggamma(ℓ - m - k + 1) + loggamma(k + m - n + 1)
        tot += ((-1)^k) * exp(pref - den) * (c^(2ℓ + n - m - 2k)) * (s^(2k + m - n))
    end
    return tot
end

@inline _Nℓ(ℓ) = sqrt((2ℓ + 1) / (4π))

"""
    sYlm(s, ℓ, m, θ, φ) -> Complex

Spin-weighted spherical harmonic `ₛY_{ℓm}(θ,φ) = N_ℓ d^ℓ_{m,−s}(θ) e^{imφ}` (this package's
convention). Provided for validation / direct evaluation.
"""
sYlm(s::Integer, ℓ::Integer, m::Integer, θ::Real, φ::Real) = _Nℓ(ℓ) * wigner_d(ℓ, m, -s, θ) * cis(m * φ)

"""
    spin_coeff_index(ℓ, m, lmax) -> CartesianIndex

Index into the dense `(lmax+1, 2lmax+1)` spin coefficient array for degree `ℓ`, order `m`.
"""
@inline spin_coeff_index(ℓ::Integer, m::Integer, lmax::Integer) = CartesianIndex(ℓ + 1, m + lmax + 1)

"""
    SpinNUSHTplan{T}

Plan for spin-`s` non-uniform spherical harmonic transforms at `M` scattered points up to
degree `lmax`. Precomputes the Wigner `Δ^ℓ = d^ℓ(π/2)` matrices and the per-`(ℓ,m,m')`
contraction weights.
"""
struct SpinNUSHTplan{T<:AbstractFloat}
    lmax::Int
    s::Int
    θ_nodes::Vector{T}
    φ_nodes::Vector{T}
    tol::Float64
    Δ::Vector{Matrix{T}}        # Δ[ℓ+1] = d^ℓ(π/2), size (2ℓ+1)×(2ℓ+1), indexed [m'+ℓ+1, m+ℓ+1]
end

"""
    make_spin_plan(θ_nodes, φ_nodes, lmax, s; tol=1e-10, T=Float64)

Build a `SpinNUSHTplan`. Colatitudes `θ ∈ [0,π]`, longitudes `φ ∈ [0,2π)`.
"""
function make_spin_plan(θ_nodes, φ_nodes, lmax::Integer, s::Integer; tol = 1e-10, T::Type{<:AbstractFloat} = Float64)
    @assert length(θ_nodes) == length(φ_nodes)
    Δ = Vector{Matrix{T}}(undef, lmax + 1)
    for ℓ in 0:lmax
        M = Matrix{T}(undef, 2ℓ + 1, 2ℓ + 1)
        for mp in -ℓ:ℓ, m in -ℓ:ℓ
            M[mp + ℓ + 1, m + ℓ + 1] = T(wigner_d(ℓ, mp, m, π / 2))
        end
        Δ[ℓ + 1] = M
    end
    return SpinNUSHTplan{T}(lmax, Int(s), Vector{T}(θ_nodes), Vector{T}(φ_nodes), Float64(tol), Δ)
end

# Assemble bivariate Fourier coefficients G_{m'm} (modes −lmax:lmax each) from coefficients sf.
function _assemble_G(sf, plan::SpinNUSHTplan{T}) where {T}
    lmax = plan.lmax; s = plan.s
    L = 2lmax + 1
    G = zeros(Complex{T}, L, L)            # [m'+lmax+1, m+lmax+1]
    @inbounds for ℓ in max(abs(s), 0):lmax
        Δℓ = plan.Δ[ℓ + 1]
        Nℓ = _Nℓ(ℓ)
        sidx = -s + ℓ + 1
        for m in -ℓ:ℓ
            val = sf[ℓ + 1, m + lmax + 1]
            val == 0 && continue
            ph = (Complex{T}(0, 1))^(m + s) * val * Nℓ
            for mp in -ℓ:ℓ
                G[mp + lmax + 1, m + lmax + 1] += ph * Δℓ[mp + ℓ + 1, m + ℓ + 1] * Δℓ[mp + ℓ + 1, sidx]
            end
        end
    end
    return G
end

# Adjoint of _assemble_G: contract NUFFT-type1 modes Ĝ back to coefficients.
function _assemble_G_adjoint!(sf, Ĝ, plan::SpinNUSHTplan{T}) where {T}
    lmax = plan.lmax; s = plan.s
    fill!(sf, zero(Complex{T}))
    @inbounds for ℓ in max(abs(s), 0):lmax
        Δℓ = plan.Δ[ℓ + 1]
        Nℓ = _Nℓ(ℓ)
        sidx = -s + ℓ + 1
        for m in -ℓ:ℓ
            phc = conj((Complex{T}(0, 1))^(m + s)) * Nℓ
            acc = zero(Complex{T})
            for mp in -ℓ:ℓ
                acc += Δℓ[mp + ℓ + 1, m + ℓ + 1] * Δℓ[mp + ℓ + 1, sidx] * Ĝ[mp + lmax + 1, m + lmax + 1]
            end
            sf[ℓ + 1, m + lmax + 1] = phc * acc
        end
    end
    return sf
end

# Map G_{m'm} into a FINUFFT 2-D mode array c[k1,k2] s.t. type2(iflag=+1) gives
# Σ G_{m'm} e^{−im'θ} e^{imφ}:  set k1 = −m' (reverse θ-mode axis), k2 = m.
function _G_to_finufft(G, lmax)
    L = 2lmax + 1
    c = zeros(eltype(G), L, L)
    @inbounds for a in 1:L, b in 1:L
        mp = a - lmax - 1
        c[(-mp) + lmax + 1, b] = G[a, b]
    end
    return c
end

# Inverse of _G_to_finufft (the same axis reversal).
function _finufft_to_G(c, lmax)
    L = 2lmax + 1
    G = zeros(eltype(c), L, L)
    @inbounds for a in 1:L, b in 1:L
        mp = a - lmax - 1
        G[(-mp) + lmax + 1, b] = c[a, b]
    end
    return G
end

"""
    nusht_type2_spin!(f, sf, plan) -> f

Spin-weighted synthesis: evaluate the spin-`s` field with coefficients `sf` at the `M`
scattered points, writing complex values into `f`.
"""
function nusht_type2_spin!(f, sf, plan::SpinNUSHTplan{T}) where {T}
    @assert length(f) == length(plan.θ_nodes)
    @assert size(sf) == (plan.lmax + 1, 2plan.lmax + 1)
    G = _assemble_G(sf, plan)
    c = _G_to_finufft(G, plan.lmax)
    f .= FINUFFT.nufft2d2(plan.θ_nodes, plan.φ_nodes, +1, plan.tol, c)
    return f
end

"""
    nusht_type1_spin!(sf, f, plan) -> sf

Exact Euclidean adjoint of `nusht_type2_spin!`: scattered values `f` → spin-`s` coefficients
`sf`.
"""
function nusht_type1_spin!(sf, f, plan::SpinNUSHTplan{T}) where {T}
    @assert length(f) == length(plan.θ_nodes)
    @assert size(sf) == (plan.lmax + 1, 2plan.lmax + 1)
    L = 2plan.lmax + 1
    cvec = FINUFFT.nufft2d1(plan.θ_nodes, plan.φ_nodes, Complex{T}.(f), -1, plan.tol, L, L)  # adjoint of type2(+1)
    c = dropdims(cvec; dims = 3)
    Ĝ = _finufft_to_G(c, plan.lmax)
    _assemble_G_adjoint!(sf, Ĝ, plan)
    return sf
end

"""
    nusht_solve_spin!(sf, f, plan; maxiter=500, rtol=1e-8, verbose=false)

Exact inversion of the spin-weighted synthesis at arbitrary scattered points: solve
`A sf = f` for the spin-`s` coefficients by conjugate gradients on the normal equations
`(A†A) sf = A† f`. Returns `(sf, iters, rel_res)`.
"""
function nusht_solve_spin!(sf, f, plan::SpinNUSHTplan{T}; maxiter::Int = 500, rtol::Real = 1e-8, verbose::Bool = false) where {T}
    lmax = plan.lmax
    shp = (lmax + 1, 2lmax + 1)
    @assert size(sf) == shp
    bufc = zeros(Complex{T}, shp)
    buff = zeros(Complex{T}, length(f))
    bufc2 = zeros(Complex{T}, shp)

    function AHA!(y, x)
        bufc .= reshape(x, shp)
        nusht_type2_spin!(buff, bufc, plan)
        nusht_type1_spin!(bufc2, buff, plan)
        y .= vec(bufc2)
    end

    nusht_type1_spin!(bufc, f, plan)
    rhs = vec(copy(bufc))
    rhs_norm = LinearAlgebra.norm(rhs)
    K = prod(shp)
    x = zeros(Complex{T}, K)
    rhs_norm == 0 && (sf .= reshape(x, shp); return sf, 0, zero(T))
    r = copy(rhs); p = copy(r); Ap = zeros(Complex{T}, K)
    rsold = real(LinearAlgebra.dot(r, r))
    rel_res = one(T); iters = 0
    for i in 1:maxiter
        iters = i
        AHA!(Ap, p)
        α = rsold / real(LinearAlgebra.dot(p, Ap))
        x .+= α .* p
        r .-= α .* Ap
        rsnew = real(LinearAlgebra.dot(r, r))
        rel_res = sqrt(rsnew) / rhs_norm
        verbose && @info "nusht_solve_spin! iter $i rel_res=$rel_res"
        rel_res < rtol && break
        p .= r .+ (rsnew / rsold) .* p
        rsold = rsnew
    end
    sf .= reshape(x, shp)
    return sf, iters, rel_res
end
