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

Multiply a FastSphericalHarmonics coefficient array `C` — size `(lmax+1, 2lmax+1)` or batched
`(lmax+1, 2lmax+1, B)` — in-place by the transfer function `H(ℓ)` for each degree `ℓ` (broadcast
across the batch dimension).
"""
# Degree held by scalar slot `(i, j)`. Inverting `sph_mode(ℓ,m) = (ℓ-|m|+1, 2|m|+(m≥0))` gives
# `|m| = j÷2`, `ℓ = i + |m| - 1`. Every slot carries a real mode — the array is square and invertible,
# so rows past `ℓ = lmax` hold degrees `lmax < ℓ ≤ lmax+|m|` rather than padding.
@inline _slot_degree(i::Int, j::Int) = i + (j ÷ 2) - 1

function apply_transfer!(C, filter, lmax)
    RT = real(eltype(C))
    n1, n2 = size(C, 1), size(C, 2)
    @inbounds for j in 1:n2
        mabs = j ÷ 2
        for i in 1:n1
            h = RT(kernel_transfer(filter, i + mabs - 1))
            for b in axes(C, 3)
                C[i, j, b] *= h
            end
        end
    end
    return C
end

# Host-built `(lmax+1, 2lmax+1)` transfer matrix (element type `RT`): `H(ℓ)` at every slot, at that
# slot's own degree. Used by the device `apply_transfer!` (one broadcast); the CPU path uses the
# zero-alloc loop above, and the two must agree everywhere.
function _transfer_matrix(filter, lmax, ::Type{RT}) where {RT}
    n1, n2 = lmax + 1, 2lmax + 1
    H = Matrix{RT}(undef, n1, n2)
    for j in 1:n2, i in 1:n1
        H[i, j] = RT(kernel_transfer(filter, _slot_degree(i, j)))
    end
    return H
end
