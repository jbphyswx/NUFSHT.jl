# NUFSHT.jl

**Non-Uniform Fast Spherical Harmonic Transforms** — native Julia implementation of the
Double Fourier Sphere (DFS) + NUFFT algorithm for computing spherical harmonic transforms
at arbitrary scattered (colatitude, longitude) points on the sphere.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/jbphyswx/NUFSHT.jl")
```

## Algorithm

The transform decomposes the non-uniform SHT (nuSHT) into four operations
(following Reinecke & Seljebotn 2013 and Belkner et al. 2024):

```
Type 2 (synthesis):   Y  = N · F · D · S
Type 1 (adjoint):     Y† = S† · D† · F† · N†
```

| Step | Operation | Function |
|------|-----------|----------|
| **S** | iso-latitude rSHT on a Clenshaw-Curtis (CC) grid | `FastSphericalHarmonics.sph_evaluate!` |
| **D** | DFS doubling: extend colatitude from [0,π] to [0,2π) | `dfs_double!` |
| **F** | 2D FFT of the doubled torus (with CC half-pixel phase correction) | `fft2_to_coeffs` |
| **N** | NUFFT type 2: evaluate Fourier series at scattered points | `FINUFFT.nufft2d2` |

The DFS method ensures the doubled field is doubly-periodic on the torus, so its
Fourier band-limit equals the spherical harmonic band-limit lmax.

## Accuracy

| Operation | Points | Max error | Notes |
|-----------|--------|-----------|-------|
| `nusht_type2!` | CC grid (lmax=20) | ~3 × 10⁻¹¹ | Machine precision vs `sph_evaluate` |
| `nusht_type1!` | CC grid (lmax=20) | ~7 × 10⁻¹¹ | Round-trip `type2(type1(f)) = f` |
| `nusht_type1!` | Scattered (non-CC) | O(ε_NUFFT) | Adjoint only; approximate analysis |

**Note:** `nusht_type1!` is the adjoint (`Y†`) of `nusht_type2!`, not its inverse.
On the Clenshaw-Curtis grid it coincides with the exact inverse. For general scattered
non-uniform points, it is an approximate analysis step.

## Usage

### Basic synthesis and adjoint analysis

```julia
using NUFSHT

lmax = 50
θ = rand(2000) .* π      # colatitudes in [0,π]
φ = rand(2000) .* 2π     # longitudes in [0,2π)
plan = make_plan(θ, φ, lmax; tol=1e-8)

# Type 2 (synthesis): coefficients → scattered field values
C = zeros(lmax+1, 2lmax+1)
C[sph_mode(2, 0)] = 1.0   # set Y_2^0 mode (requires FastSphericalHarmonics)
f = zeros(length(θ))
nusht_type2!(f, C, plan)

# Type 1 (adjoint analysis): scattered values → coefficients (exact on CC grid)
C_out = similar(plan.C)
nusht_type1!(C_out, f, plan)
```

### Exact use on the Clenshaw-Curtis grid

```julia
using NUFSHT, FastSphericalHarmonics

lmax = 30
pts = sph_points(lmax + 1)          # CC grid coordinates
θ = vec([θ for θ in pts[1], φ in pts[2]])
φ = vec([φ for θ in pts[1], φ in pts[2]])
plan = make_plan(θ, φ, lmax; tol=1e-10)

# C → f → C round-trip is machine-precision on CC grid
C_true = randn(lmax+1, 2lmax+1)
f = zeros(length(θ)); nusht_type2!(f, C_true, plan)
C_rec = similar(plan.C); nusht_type1!(C_rec, f, plan)
# maximum(abs.(C_rec .- C_true)) ≈ 1e-10
```

### Spectral filtering

```julia
using NUFSHT

lmax = 100
plan = make_plan(θ, φ, lmax)

# Gaussian low-pass filter at 500 km scale
filter = gaussian_from_scale(500e3)
f_filtered = similar(f)
nusht_filter!(f_filtered, f, filter, plan)

# Sharp spectral cutoff at degree 50
filter = TopHatTransfer(50)
nusht_filter!(f_filtered, f, filter, plan)
```

### Masking and renormalisation

```julia
# Ocean-only filtering: zero out land points, then renormalise
mask = Float64.(is_ocean_point)     # 1 = ocean, 0 = land
f_masked = f .* mask

f_out = similar(f)
filter = gaussian_from_scale(200e3)
nusht_filter!(f_out, f_masked, filter, plan)
nusht_filter_renorm!(f_out, mask, filter, plan)  # correct for mask bias
```

## API Reference

| Function | Description |
|----------|-------------|
| `make_plan(θ, φ, lmax; tol, T)` | Construct pre-allocated plan |
| `nusht_type2!(f, C, plan)` | Synthesis: SH coefficients → scattered values |
| `nusht_type1!(C, f, plan)` | Adjoint analysis: scattered values → SH coefficients |
| `nusht_filter!(f_out, f_in, filter, plan)` | Apply spectral filter via type1→multiply→type2 |
| `nusht_filter_renorm!(f_out, mask, filter, plan)` | Correct mask bias after `nusht_filter!` |
| `GaussianTransfer(σ²)` | Gaussian filter H(ℓ) = exp(-ℓ(ℓ+1)σ²/2) |
| `gaussian_from_scale(scale_m)` | Construct `GaussianTransfer` from physical scale in metres |
| `TopHatTransfer(L)` | Sharp spectral cutoff at degree L |
| `cutoff_degree(scale_m)` | Convert physical scale (m) to SH degree |

## References

- Merilees, P.E. (1973): The pseudospectral approximation applied to the shallow
  water equations on a sphere. *Atmosphere*, 11(1), 13–20.
- Reinecke, M. & Seljebotn, D.S. (2013): Libsharp – spherical harmonic transforms
  revisited. *A&A*, 554, A112. https://doi.org/10.1051/0004-6361/201220728
- Belkner, S. et al. (2024): cunuSHT – GPU Accelerated Spherical Harmonic Transforms
  on Arbitrary Pixelizations. *arXiv:2406.14542*.
- [FastSphericalHarmonics.jl](https://github.com/eschnett/FastSphericalHarmonics.jl)
- [FINUFFT.jl](https://github.com/ludvigak/FINUFFT.jl)
- [cunuSHT](https://github.com/Sebastian-Belkner/cunuSHT)
- [FastTransforms.jl](https://github.com/JuliaApproximation/FastTransforms.jl)
