"""
    Kernels.jl — Spectral filter transfer functions H(ℓ) for spherical harmonic filtering.

A spectral filter on the sphere multiplies each harmonic coefficient c_ℓm by H(ℓ),
where ℓ is the angular degree. The physical-space interpretation is convolution with
a kernel whose Legendre spectrum is H(ℓ).

References:
- Aluie et al. (2018): Coarse-graining as a measurement operator on fields over the sphere
- Lilly & Gascard (2006): Wavelet ridge diagnosis of time-varying elliptical signals
"""

export TopHatTransfer, GaussianTransfer, SharpSpectralTransfer
export kernel_transfer, cutoff_degree, gaussian_from_scale


abstract type AbstractSpectralTransfer end

"""
    cutoff_degree(scale_m, R_m)

Compute the spherical harmonic cutoff degree L corresponding to a physical
filter scale in meters on a sphere of radius R_m.

The relationship is L ≈ π * R_m / scale_m, analogous to Nyquist for a
circle of circumference 2π R_m.
"""
function cutoff_degree(scale_m, R_m=6.371e6)
    return round(Int, π * R_m / scale_m)
end

"""
    TopHatTransfer

Ideal low-pass (sharp spectral cutoff) filter: H(ℓ) = 1 for ℓ ≤ L, else 0.
Physical-space equivalent is a convolution with a zonal kernel whose Legendre
spectrum is a boxcar. Note: this is spectrally sharp but spatially oscillatory
(Gibbs phenomenon).
"""
struct TopHatTransfer <: AbstractSpectralTransfer
    L::Int
end

"""
    GaussianTransfer

Gaussian spectral filter: H(ℓ) = exp(-ℓ(ℓ+1) σ²/2)

where σ = scale_m / R_m is the dimensionless filter width.
In physical space this corresponds to convolution with a Gaussian-like kernel
on the sphere. At degree ℓ the spatial scale is approximately R/ℓ.

Reference: Eq. (3) of Aluie et al. (2018), analogous to Gaussian in Fourier space.
"""
struct GaussianTransfer <: AbstractSpectralTransfer
    σ²::Float64
end

"""
    gaussian_from_scale(scale_m, R_m=6.371e6)

Construct a GaussianTransfer from a physical filter scale in meters.
Use this instead of `GaussianTransfer(scale_m)` to avoid ambiguity
with the struct constructor which takes σ² directly.
"""
function gaussian_from_scale(scale_m, R_m=6.371e6)
    σ = Float64(scale_m) / Float64(R_m)
    return GaussianTransfer(σ^2)
end

"""
    SharpSpectralTransfer

Alias for TopHatTransfer — identical sharp low-pass cutoff at degree L.
"""
const SharpSpectralTransfer = TopHatTransfer

"""
    kernel_transfer(filter, ℓ) -> H

Evaluate the spectral transfer function H(ℓ) for a given filter type and degree ℓ.
Returns a real scalar in [0, 1].
"""
function kernel_transfer(f::TopHatTransfer, ℓ)
    return ℓ <= f.L ? 1.0 : 0.0
end

function kernel_transfer(f::GaussianTransfer, ℓ)
    return exp(-ℓ * (ℓ + 1) * f.σ² / 2)
end

"""
    apply_transfer!(C, filter, lmax)

Multiply a FastSphericalHarmonics coefficient array C (size (lmax+1)×(2lmax+1))
in-place by the transfer function H(ℓ) for each degree ℓ.
"""
function apply_transfer!(C, filter, lmax)
    for ℓ in 0:lmax
        h = kernel_transfer(filter, ℓ)
        for m in -ℓ:ℓ
            idx = FastSphericalHarmonics.sph_mode(ℓ, m)
            C[idx] *= h
        end
    end
    return C
end
