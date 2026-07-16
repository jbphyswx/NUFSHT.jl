# NUFSHT.jl

**Non-Uniform Fast Spherical Harmonic Transforms** — native Julia implementation of the
Double Fourier Sphere (DFS) + NUFFT algorithm for computing spherical harmonic transforms
at arbitrary scattered (colatitude, longitude) points on the sphere.

## Results

### Synthesis at Scattered Points + Round-Trip Accuracy

![Synthesis and Accuracy](docs/assets/synthesis_and_accuracy.png)

### CG Inversion at Arbitrary Scattered Points

![CG Inversion](docs/assets/cg_inversion.png)

### Spectral Filtering (Gaussian and Sharp Cutoff)

![Spectral Filtering](docs/assets/spectral_filtering.png)

### Ocean Mask + Renormalization

![Mask Renormalization](docs/assets/mask_renorm.png)

### Spin-weighted synthesis + spin-1 Hodge decomposition

![Spin-weighted transforms](docs/assets/spin_synthesis.png)

## Spin-weighted transforms (spin-`s`)

NUFSHT also synthesizes/analyzes **spin-weighted** fields at arbitrary scattered points —
built directly from the Wigner-`d` Fourier factorization plus a 2-D NUFFT, independent of any
spin convention in FastTransforms. Spin-1 is the tangent-vector case (a velocity
`U = u_θ + i u_φ` is a spin-1 field), which enables vector/Helmholtz decomposition on
scattered spherical data.

```julia
using NUFSHT
lmax, s = 32, 1
θ = π .* rand(5000); φ = 2π .* rand(5000)          # scattered colatitude/longitude
plan = make_spin_plan(θ, φ, lmax, s)                # add ntrans=B for batches; T=Float32 for f32

# synthesis: spin-s coefficients -> complex field values at the points
sf = zeros(ComplexF64, lmax+1, 2lmax+1)             # spin-s coefficients (set some modes)
f  = zeros(ComplexF64, length(θ))
nusht_type2_spin!(f, sf, plan)

# exact inversion at arbitrary scattered points (CG on the normal equations)
sf_rec = zeros(ComplexF64, lmax+1, 2lmax+1)
nusht_solve_spin!(sf_rec, f, plan; rtol=1e-9)
```

`make_spin_plan` / `nusht_type2_spin!` / `nusht_type1_spin!` (exact adjoint) /
`nusht_solve_spin!`; coefficient indices via `spin_coeff_index(ℓ,m,lmax)`, direct harmonic
values via `sYlm(s,ℓ,m,θ,φ)`. Validated to ~1e-12 (synthesis), ~1e-15 (adjoint), and exact
scattered inversion. See `dev/spin_hodge_validation.jl` for the spin-1 Helmholtz/Hodge use.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/jbphyswx/NUFSHT.jl")
```

## Performance, batching, parallelism & GPU

Everything below is dependency-light by default — the accelerators are **package extensions**, loaded
only when you load their trigger package, so a plain `using NUFSHT` never pulls in MPI/CUDA/etc.

- **Persistent guru plans, zero allocation.** A plan builds the FINUFFT guru plans once (points set
  once) and pre-allocates every buffer, so warmed-up `nusht_type2!`/`nusht_type1!` and the hundreds of
  CG matvecs inside `nusht_solve!` allocate **nothing** and never re-plan. (Single-threaded FINUFFT is
  exactly zero-alloc; the all-cores default adds only FINUFFT's own small planner-lock allocation.)
- **Batching (`ntrans = B`).** Transform `B` co-located fields (same points) in one call —
  `make_plan(θ, φ, lmax; ntrans = B)`, coefficients/fields carry a trailing batch axis. FFTW and
  FINUFFT parallelize *across the batch* internally (measured ~4.5× / ~2.8× on small transforms), which
  is the recommended way to use many cores on one node.
- **Mixed precision.** `T = Float32` is supported on the spin/recurrence path (the scalar DFS path is
  Float64-only, a FastTransforms limitation).
- **Parallel extensions** (each keyed on its trigger package):
  - `using OhMyThreads` — thread-parallel over independent problems (one plan per task).
  - `using Distributed` — farm independent problems across worker **processes** (`addprocs`); falls
    back to serial when there are none.
  - `using MPI` — point-decomposition: partition the `M` points across ranks; `A` needs no
    communication, `A†` and the CG inner products are `Allreduce`d.
  > Note: FastTransforms is *not* safe to call from a Julia task/thread (its OpenMP corrupts results);
  > the thread/process extensions force it single-threaded, which is why batching + processes are the
  > recommended scaling paths. See `dev/fasttransforms_task_safety.md`.
- **GPU** (`using CUDA`, with `using KernelAbstractions`). The array-indexed steps (DFS doubling/folding
  and the spin Wigner-`d` recurrence + bivariate-Fourier assembly) are KernelAbstractions `@kernel`s —
  **written once, run on any backend** — and the NUFFT is bound to cuFINUFFT. A device node set yields a
  device-resident plan (buffers `similar` to the nodes). The device kernels are validated bit-for-bit
  against the CPU path on `JLArrays`; end-to-end GPU parity (incl. cuFINUFFT) is in `test/gpu_cuda.jl`,
  to run on NVIDIA hardware.

## Algorithm

The transform decomposes the non-uniform SHT (nuSHT) into four operations
(following Reinecke & Seljebotn 2013 and Belkner et al. 2024):

```
Type 2 (synthesis):   A  = N · F · D · S
Type 1 (adjoint):     A† = S† · D† · F† · N†
```

| Step | Operation | Forward | Adjoint |
|------|-----------|---------|---------|
| **S** | Iso-latitude rSHT: SH coefficients ↔ CC grid | `sph_evaluate!` (PS·P) | `sph_transform!` (S⁻¹, exact on CC) |
| **D** | DFS doubling: extend [0,π] → [0,2π) torus | `dfs_double!` | `dfs_fold!` |
| **F** | 2D FFT + fused half-pixel CC phase (in-place, `modeord=1`) | stored FFTW plan + phase broadcast | conj-phase + inverse FFT |
| **N** | NUFFT: evaluate Fourier series at scattered points | guru type-2 plan | guru type-1 plan |

(The scalar path above uses FastTransforms for **S**; the **spin** path replaces **S** with an
on-the-fly Trapani–Navaza Wigner-`d` recurrence — O(lmax²) memory and numerically stable to
lmax ≈ 1024, whereas the previous dense-`Δ` / explicit-factorial evaluation used O(lmax³) memory and
was silently inaccurate above lmax ≈ 40. The recurrence is also the device S-engine.)

The DFS doubling extends the sphere to a doubly-periodic torus by reflecting across
the south pole with a φ+π shift. The shift uses `mod1(j + Nφ÷2, Nφ)` to ensure a
proper bijection for all `Nφ` (including odd, which is always the case since
`Nφ = 2*lmax+1`).

### Adjoint vs inverse

- **`nusht_type1!`** computes `A†f` — the adjoint/analysis. On the Clenshaw-Curtis
  (CC) quadrature grid (`sph_points(lmax+1)`), the CC quadrature makes `A†A = I`
  (exact round-trip). At arbitrary scattered non-CC points it is only the adjoint.

- **`nusht_solve!`** uses the true Euclidean adjoint (`PS'·P'` for the S step,
  correct `dfs_fold!` for the D step) to solve `(A†A)c = A†f` via Conjugate
  Gradients. This gives exact inversion at any set of scattered points.

> **Use `nusht_type1!` for CC-grid analysis and filtering.**
> **Use `nusht_solve!` for exact inversion at arbitrary scattered points.**

## Accuracy

| Operation | Points | Error | Notes |
|-----------|--------|-------|-------|
| `nusht_type2!` | CC grid (lmax=20) | ~3 × 10⁻¹¹ | vs `sph_evaluate` |
| `nusht_type1!` → `nusht_type2!` | CC grid (lmax=20) | ~7 × 10⁻¹¹ | Round-trip |
| `nusht_solve!` | Scattered, 4× overdetermined (lmax=10) | field ~2.5 × 10⁻⁴ | After ~200 CG iters |

`nusht_solve!` convergence rate depends on the condition number of `A†A`, which
scales with point distribution quality. Jittered-uniform points are well-conditioned;
clustered or gapped distributions require more iterations.

## Usage

### Synthesis at scattered points

```julia
using NUFSHT, FastSphericalHarmonics

lmax = 50
θ = rand(2000) .* π      # colatitudes in [0,π]
φ = rand(2000) .* 2π     # longitudes in [0,2π)
plan = make_plan(θ, φ, lmax; tol=1e-8)

# Set some SH coefficients
C = zeros(lmax+1, 2lmax+1)
C[sph_mode(2, 0)] = 1.0   # Y_2^0
C[sph_mode(3, 1)] = 0.5   # Y_3^1

# Type 2 (synthesis): coefficients → field values at scattered points
f = zeros(length(θ))
nusht_type2!(f, C, plan)
```

### Adjoint analysis (CC grid only — exact round-trip)

```julia
using NUFSHT, FastSphericalHarmonics

lmax = 30
pts = sph_points(lmax + 1)          # Clenshaw-Curtis quadrature grid
θ = vec([θ for θ in pts[1], φ in pts[2]])
φ = vec([φ for θ in pts[1], φ in pts[2]])
plan = make_plan(θ, φ, lmax; tol=1e-10)

C_true = randn(lmax+1, 2lmax+1)
f = zeros(length(θ))
nusht_type2!(f, C_true, plan)     # synthesise

C_rec = similar(plan.C)
nusht_type1!(C_rec, f, plan)      # analyse (exact inverse on CC grid)
# maximum(abs.(C_rec .- C_true)) ≈ 7e-11
```

### Exact inversion at arbitrary scattered points

```julia
using NUFSHT

lmax = 20
M = 4 * (lmax+1)^2    # 4× overdetermined — well-conditioned
θ = ...               # arbitrary scattered colatitudes ∈ (0,π)
φ = ...               # arbitrary scattered longitudes ∈ [0,2π)
plan = make_plan(θ, φ, lmax; tol=1e-10)

f = ...               # observed field values at (θ,φ)

C = similar(plan.C)
C, iters, rel_res = nusht_solve!(C, f, plan; rtol=1e-6, maxiter=500)
# Returns (coefficients, number_of_CG_iterations, final_relative_residual)
```

### Spectral filtering

```julia
using NUFSHT

lmax = 100
plan = make_plan(θ, φ, lmax)

# Gaussian low-pass filter at 500 km scale
filt = gaussian_from_scale(500e3)
f_filtered = similar(f)
nusht_filter!(f_filtered, f, filt, plan)

# Sharp spectral cutoff at degree 50
nusht_filter!(f_filtered, f, TopHatTransfer(50), plan)
```

### Ocean masking and renormalisation

```julia
# Zero out land points, filter, then correct for mask bias
mask = Float64.(is_ocean_point)     # 1.0 = ocean, 0.0 = land
f_masked = f .* mask

f_out = similar(f)
filt = gaussian_from_scale(200e3)
nusht_filter!(f_out, f_masked, filt, plan)
nusht_filter_renorm!(f_out, mask, filt, plan)   # divide by filtered mask
```

## API Reference

| Function | Description |
|----------|-------------|
| `make_plan(θ, φ, lmax; tol, T)` | Construct pre-allocated plan for M scattered points |
| `nusht_type2!(f, C, plan)` | Synthesis: SH coefficients → scattered field values |
| `nusht_type1!(C, f, plan)` | Adjoint analysis (exact inverse on CC grid; adjoint elsewhere) |
| `nusht_solve!(C, f, plan; maxiter, rtol, verbose)` | Exact CG inversion at any scattered points |
| `nusht_filter!(f_out, f_in, filter, plan)` | Spectral filter: type1 → multiply → type2 |
| `nusht_filter_renorm!(f_out, mask, filter, plan)` | Correct land-mask bias after `nusht_filter!` |
| `GaussianTransfer(σ²)` | Gaussian filter `H(ℓ) = exp(-ℓ(ℓ+1)σ²/2)` |
| `gaussian_from_scale(scale_m)` | `GaussianTransfer` from physical scale in metres |
| `TopHatTransfer(L)` | Sharp spectral cutoff at degree `L` |
| `SharpSpectralTransfer(L)` | Alias for `TopHatTransfer` |
| `cutoff_degree(scale_m)` | Convert physical scale (m) to SH degree |

### `nusht_solve!` return value

```julia
C, iters, rel_res = nusht_solve!(C, f, plan; rtol=1e-6, maxiter=500)
```

- `C`: output coefficient array (overwritten in-place)
- `iters`: number of CG iterations performed
- `rel_res`: final relative residual `‖r‖/‖A†f‖`; will be `< rtol` if converged

## Implementation notes

### DFS shift for odd Nφ

`Nφ = 2*lmax+1` is always odd. The φ+π column shift in `dfs_double!` uses
`mod1(j + Nφ÷2, Nφ)` (a proper cyclic permutation). The older conditional
`j <= half ? j+half : j-half` is **not a bijection** for odd Nφ — two input
columns map to the same output column — and was silently wrong, causing `dfs_fold!`
to fail as the algebraic adjoint of `dfs_double!`. The fix uses the inverse shift
`mod1(j - Nφ÷2, Nφ)` in `dfs_fold!`.

### True adjoint vs CC-grid inverse

`nusht_type1!` uses `sph_transform!` (the CC-grid analysis = `S⁻¹`). This gives
an exact round-trip on the CC grid but is NOT the Euclidean matrix adjoint of
`nusht_type2!` at non-CC points. `nusht_solve!` uses the private
`_nusht_true_adjoint!` which applies `PS'·P'` (the true `S†`), making `A†A`
symmetric positive definite and CG convergent for arbitrary point distributions.

## References

- Merilees, P.E. (1973): The pseudospectral approximation applied to the shallow
  water equations on a sphere. *Atmosphere*, 11(1), 13–20.
- Townsend, A. & Olver, S. (2015): The automatic solution of partial differential
  equations using a global spectral method. *J. Comput. Phys.*, 299, 106–123.
- Reinecke, M. & Seljebotn, D.S. (2013): Libsharp – spherical harmonic transforms
  revisited. *A&A*, 554, A112. https://doi.org/10.1051/0004-6361/201220728
- Belkner, S. et al. (2024): cunuSHT – GPU Accelerated Spherical Harmonic Transforms
  on Arbitrary Pixelizations. *arXiv:2406.14542*.
- [FastSphericalHarmonics.jl](https://github.com/eschnett/FastSphericalHarmonics.jl)
- [FINUFFT.jl](https://github.com/ludvigak/FINUFFT.jl)
- [FastTransforms.jl](https://github.com/JuliaApproximation/FastTransforms.jl)
