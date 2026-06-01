"""
    Plan.jl — Pre-allocated plan struct for NUFSHT transforms.

A NUSHTplan pre-allocates all intermediate arrays so that repeated transforms
(e.g. filtering multiple fields at the same grid) minimise allocation.
"""

using FFTW: FFTW
using FastTransforms: FastTransforms

export NUSHTplan, make_plan

"""
    NUSHTplan{T}

Pre-computed plan for non-uniform spherical harmonic transforms.

Fields:
- `lmax`: Maximum spherical harmonic degree
- `Nθ`, `Nφ`: Size of the equiangular CC grid (Nθ = lmax+1, Nφ = 2lmax+1)
- `C`: Coefficient array (lmax+1) × (2lmax+1) — reused across calls
- `F`: Real map on equiangular CC grid (Nθ × Nφ)
- `F̃`: Doubled real map on torus (2Nθ × Nφ)
- `Fhat`: Complex Fourier coefficients of doubled map (2Nθ × Nφ)
- `θ_nodes`: Colatitudes θ ∈ [0,π] of scattered points (passed directly to FINUFFT)
- `φ_nodes`: Longitudes φ ∈ [0,2π) of scattered points (passed directly to FINUFFT)
- `tol`: FINUFFT accuracy tolerance
- `fft_plan`: pre-computed FFTW forward plan for F̃ (avoids per-call planning in `fft2_to_coeffs`)
- `ifft_plan`: pre-computed FFTW inverse plan for Fhat (avoids per-call planning in `ifft2_from_coeffs`)
- `phase_θ`: per-mode θ phase correction exp(-πi kθ/Nθ_dbl) for the CC half-pixel offset, size 2Nθ
- `phase_θ_conj`: conjugate phase exp(+πi kθ/Nθ_dbl), used in `ifft2_from_coeffs`
- `sph_plan`: FastTransforms `plan_sph2fourier` plan (P), for `sph_evaluate!` and its adjoint
- `sph_plan_synth`: FastTransforms `plan_sph_synthesis` plan (PS), for `sph_evaluate!` and its adjoint
"""
struct NUSHTplan{T<:AbstractFloat}
    lmax::Int
    Nθ::Int
    Nφ::Int
    C::Matrix{T}
    F::Matrix{T}
    F̃::Matrix{T}
    Fhat::Matrix{Complex{T}}
    θ_nodes::Vector{T}
    φ_nodes::Vector{T}
    tol::Float64
    fft_plan::FFTW.Plan
    ifft_plan::FFTW.Plan
    phase_θ::Vector{Complex{T}}
    phase_θ_conj::Vector{Complex{T}}
    sph_plan::FastTransforms.FTPlan
    sph_plan_synth::FastTransforms.FTPlan
end

"""
    make_plan(θ_nodes, φ_nodes, lmax; tol=1e-8, T=Float64)

Construct a NUSHTplan for M scattered points at colatitudes θ_nodes ∈ [0,π]
and longitudes φ_nodes ∈ [0,2π), up to spherical harmonic degree lmax.

FINUFFT accepts coordinates in [-3π, 3π], so natural [0,π] and [0,2π) coordinates
are passed directly without remapping.
"""
function make_plan(
    θ_nodes,
    φ_nodes,
    lmax;
    tol = 1e-8,
    T::Type{<:AbstractFloat} = Float64,
)
    @assert length(θ_nodes) == length(φ_nodes)

    Nθ = lmax + 1
    Nφ = 2lmax + 1

    C    = zeros(T, Nθ, Nφ)
    F    = zeros(T, Nθ, Nφ)
    F̃    = zeros(T, 2Nθ, Nφ)
    Fhat = zeros(Complex{T}, 2Nθ, Nφ)

    θ = Vector{T}(θ_nodes)
    φ = Vector{T}(φ_nodes)

    fft_plan  = FFTW.plan_fft(F̃)
    ifft_plan = FFTW.plan_ifft(Fhat)

    Nθ_dbl = 2Nθ
    k_θ = [k < Nθ_dbl ÷ 2 ? k : k - Nθ_dbl for k in 0:(Nθ_dbl - 1)]
    phase_θ      = exp.(-im .* π .* T.(k_θ) ./ Nθ_dbl)
    phase_θ_conj = conj.(phase_θ)

    sph_plan       = FastTransforms.plan_sph2fourier(C)
    sph_plan_synth = FastTransforms.plan_sph_synthesis(C)

    return NUSHTplan{T}(lmax, Nθ, Nφ, C, F, F̃, Fhat, θ, φ, Float64(tol), fft_plan, ifft_plan, phase_θ, phase_θ_conj, sph_plan, sph_plan_synth)
end
