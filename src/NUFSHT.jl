"""
    NUFSHT.jl — Non-Uniform Fast Spherical Harmonic Transform (native Julia)

Implements the Double Fourier Sphere (DFS) + nuFFT algorithm for computing
spherical harmonic transforms at arbitrary scattered (colatitude, longitude) points.

## Algorithm

The algorithm decomposes the non-uniform SHT (nuSHT) into four operations
(following Reinecke & Seljebotn 2013 and Belkner et al. 2024):

    Y = N · F · D · S

where:
- **S** (`sph_transform` / `sph_evaluate`): iso-latitude rSHT between
  spherical harmonic coefficients c_ℓm and an equiangular Clenshaw-Curtis grid.
- **D** (`dfs_double` / `dfs_fold`): "doubling" that extends the colatitude range
  from [0,π] to [0,2π) across the south pole, making the field doubly-periodic.
- **F** (implicit 2D FFT embedded in FINUFFT): standard 2D DFT on the doubled torus.
- **N** (`nufft2d2` / `nufft2d1`): non-uniform FFT evaluating the 2D Fourier series
  at the arbitrary scattered points.

## Usage

```julia
using NUFSHT

# Create a plan for M scattered points up to degree lmax
lmax = 100
θ = rand(1000) .* π          # colatitudes in [0,π]
φ = rand(1000) .* 2π         # longitudes in [0,2π)
plan = make_plan(θ, φ, lmax)

# Compute spherical harmonic coefficients from scattered field values
f = randn(1000)               # field values at (θ,φ)
C = similar(plan.C)           # coefficient array (lmax+1)×(2lmax+1)
nusht_type1!(C, f, plan)

# Synthesise back to scattered points (should recover f for lmax large enough)
f_out = similar(f)
nusht_type2!(f_out, C, plan)

# Filter a field: apply low-pass Gaussian filter with scale 200 km
filter = GaussianTransfer(200e3)
f_filt = similar(f)
nusht_filter!(f_filt, f, filter, plan)
```

## References

- Merilees, P.E. (1973): The pseudospectral approximation applied to the shallow
  water equations on a sphere. Atmosphere, 11(1), 13–20.
- Townsend, A. & Olver, S. (2015): The automatic solution of partial differential
  equations using a global spectral method. J. Comput. Phys., 299, 106–123.
- Reinecke, M. & Seljebotn, D.S. (2013): Libsharp – spherical harmonic transforms
  revisited. A&A, 554, A112. https://doi.org/10.1051/0004-6361/201220728
- Keiner, J., Kunis, S. & Potts, D. (2009): Using NFFT3 – a software library for
  various nonequispaced fast Fourier transforms. ACM Trans. Math. Softw., 36, 19.
- Belkner, S. et al. (2024): cunuSHT – GPU Accelerated Spherical Harmonic Transforms
  on Arbitrary Pixelizations. arXiv:2406.14542.
- FastSphericalHarmonics.jl: https://github.com/eschnett/FastSphericalHarmonics.jl
- FINUFFT.jl: https://github.com/ludvigak/FINUFFT.jl
"""
module NUFSHT

using FFTW: FFTW
using FINUFFT: FINUFFT
using FastSphericalHarmonics: FastSphericalHarmonics
using LinearAlgebra: LinearAlgebra

include("DFS.jl")
include("Plan.jl")
include("Kernels.jl")

export make_plan, NUSHTplan
export nusht_type1!, nusht_type2!, nusht_filter!
export TopHatTransfer, GaussianTransfer, SharpSpectralTransfer
export kernel_transfer, cutoff_degree, gaussian_from_scale

# ─────────────────────────────────────────────────────────────────────────────
# Type 1: scattered map → spherical harmonic coefficients
# ─────────────────────────────────────────────────────────────────────────────

"""
    nusht_type1!(C, f, plan)

**Type 1 (analysis / adjoint synthesis):** Given real field values `f` at
M scattered points (θᵢ, φᵢ) defined in `plan`, compute spherical harmonic
coefficients `C` up to degree `plan.lmax`.

This is the adjoint of `nusht_type2!`.

Algorithm:
1. FINUFFT Type 1: scattered f → 2D Fourier coefficients F̂ on doubled torus
2. DFS fold (adjoint D†): collapse doubled torus to equiangular CC half-sphere
3. FastSphericalHarmonics adjoint rSHT (S†): equiangular map → c_ℓm in C
"""
function nusht_type1!(C, f, plan::NUSHTplan{T}) where {T}
    (; lmax, Nθ, Nφ, F̃, Fhat, tol) = plan

    M = length(f)
    @assert M == length(plan.θ_nodes) == length(plan.φ_nodes)
    @assert size(C) == (Nθ, Nφ)

    f_cmplx = Complex{T}.(f)

    Fhat_vec = FINUFFT.nufft2d1(
        plan.φ_nodes, plan.θ_nodes,
        f_cmplx,
        +1, tol,
        Nφ, 2Nθ,
    )

    Fhat_2d = dropdims(Fhat_vec, dims=3)

    F̃_real = ifft2_from_coeffs(Fhat_2d, 2Nθ, Nφ)

    F_folded = dfs_fold(F̃_real)

    C .= F_folded
    FastSphericalHarmonics.sph_transform!(C)

    return C
end

# ─────────────────────────────────────────────────────────────────────────────
# Type 2: spherical harmonic coefficients → scattered map
# ─────────────────────────────────────────────────────────────────────────────

"""
    nusht_type2!(f, C, plan)

**Type 2 (synthesis):** Given spherical harmonic coefficients `C` up to degree
`plan.lmax`, evaluate the field at the M scattered points (θᵢ, φᵢ), writing
results into `f`.

This is the adjoint of `nusht_type1!`.

Algorithm:
1. FastSphericalHarmonics forward rSHT (S): c_ℓm in C → equiangular CC map F
2. DFS double (D): extend F on [0,π] to doubled torus F̃ on [0,2π)
3. FINUFFT Type 2: F̃ Fourier coefficients → scattered point values f
"""
function nusht_type2!(f, C, plan::NUSHTplan{T}) where {T}
    (; lmax, Nθ, Nφ, F, F̃, tol) = plan

    M = length(f)
    @assert M == length(plan.θ_nodes) == length(plan.φ_nodes)
    @assert size(C) == (Nθ, Nφ)

    F .= C
    FastSphericalHarmonics.sph_evaluate!(F)

    F̃ .= zero(T)
    dfs_double!(F̃, F)

    Fhat_2d = fft2_to_coeffs(F̃, 2Nθ, Nφ)

    f_cmplx = FINUFFT.nufft2d2(plan.φ_nodes, plan.θ_nodes, -1, tol, Fhat_2d)

    @. f = real(f_cmplx)

    return f
end

# ─────────────────────────────────────────────────────────────────────────────
# Filter: apply spectral filter in harmonic space
# ─────────────────────────────────────────────────────────────────────────────

"""
    nusht_filter!(f_out, f_in, filter, plan)

Apply a spectral filter to `f_in` at scattered points, writing the filtered
field to `f_out`. Both must be length-M vectors matching `plan`.

The filter is applied by:
1. `nusht_type1!`: f_in → c_ℓm
2. `apply_transfer!`: c_ℓm × H(ℓ) in-place
3. `nusht_type2!`: filtered c_ℓm → f_out

For land masking: set masked values to 0 in `f_in` before calling, then
renormalise `f_out` by the local kernel mass over wet points (see `nusht_filter_renorm!`).
"""
function nusht_filter!(f_out, f_in, filter, plan::NUSHTplan)
    C = copy(plan.C)
    nusht_type1!(C, f_in, plan)
    apply_transfer!(C, filter, plan.lmax)
    nusht_type2!(f_out, C, plan)
    return f_out
end

# ─────────────────────────────────────────────────────────────────────────────
# Internal FFT helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    fft2_to_coeffs(F̃, Nθ_dbl, Nφ) -> Fhat

2D FFT of the doubled map F̃ (size Nθ_dbl × Nφ), returning complex Fourier
coefficients in the layout expected by FINUFFT Type 2 (row-major Nφ × Nθ_dbl,
with FINUFFT-centered mode order).

The DFS doubled grid places data at cell-center positions θ̃ᵢ = 2π/Nθ_dbl*(i-0.5),
which are offset by half a pixel from FFTW's assumed integer-index positions.
A per-mode θ phase correction exp(-πi kθ/Nθ_dbl) compensates for this offset so
that nufft2d2 evaluated at natural θ ∈ [0,π] coordinates gives the correct values.
The φ grid φⱼ = 2π/Nφ*(j-1) is already zero-based (no φ phase correction needed).
"""
function fft2_to_coeffs(F̃, Nθ_dbl, Nφ)
    @assert size(F̃) == (Nθ_dbl, Nφ)
    Fhat_raw = FFTW.fft(F̃)  # (Nθ_dbl, Nφ), FFTW natural mode order
    # θ phase correction for half-pixel cell-center offset
    k_θ = [k < Nθ_dbl ÷ 2 ? k : k - Nθ_dbl for k in 0:(Nθ_dbl - 1)]
    phase_θ = exp.(-im .* π .* k_θ ./ Nθ_dbl)
    Fhat_corrected = (Fhat_raw .* phase_θ) ./ (Nθ_dbl * Nφ)
    return collect(FFTW.fftshift(Fhat_corrected)')
end

"""
    ifft2_from_coeffs(Fhat_2d, Nθ_dbl, Nφ) -> F̃

Adjoint of `fft2_to_coeffs`: maps FINUFFT type-1 mode coefficients back to a
spatial map on the doubled cell-center grid (Nθ_dbl × Nφ).

Applies the conjugate θ phase correction exp(+πi kθ/Nθ_dbl) before the inverse
FFT to correctly invert the half-pixel offset compensation in `fft2_to_coeffs`.
"""
function ifft2_from_coeffs(Fhat_2d, Nθ_dbl, Nφ)
    Fhat_shifted = FFTW.ifftshift(collect(Fhat_2d'))  # back to FFTW natural order
    k_θ = [k < Nθ_dbl ÷ 2 ? k : k - Nθ_dbl for k in 0:(Nθ_dbl - 1)]
    phase_θ_conj = exp.(+im .* π .* k_θ ./ Nθ_dbl)
    return real.(FFTW.ifft(Fhat_shifted .* phase_θ_conj))
end

end # module NUFSHT
