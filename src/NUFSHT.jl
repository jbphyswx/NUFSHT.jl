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

# Type 1 (adjoint analysis): scattered field values → harmonic coefficients
# Note: nusht_type1! is the adjoint (not inverse) of nusht_type2!.
# It is exact at Clenshaw-Curtis grid points (sph_points); approximate elsewhere.
f = randn(1000)               # field values at (θ,φ)
C = similar(plan.C)           # coefficient array (lmax+1)×(2lmax+1)
nusht_type1!(C, f, plan)

# Type 2 (synthesis): harmonic coefficients → scattered field values
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
export nusht_type1!, nusht_type2!, nusht_filter!, nusht_filter_renorm!, nusht_solve!
export TopHatTransfer, GaussianTransfer, SharpSpectralTransfer
export kernel_transfer, cutoff_degree, gaussian_from_scale

# ─────────────────────────────────────────────────────────────────────────────
# Type 1: scattered map → spherical harmonic coefficients
# ─────────────────────────────────────────────────────────────────────────────

"""
    nusht_type1!(C, f, plan)

**Type 1 (adjoint synthesis):** Given real field values `f` at M scattered
points (θᵢ, φᵢ) defined in `plan`, compute the adjoint spherical harmonic
projection `C = Y† f` up to degree `plan.lmax`.

This is the **adjoint** (not the inverse) of `nusht_type2!`. On the Clenshaw-
Curtis grid (as returned by `FastSphericalHarmonics.sph_points`), the adjoint
coincides with the inverse and gives machine-precision coefficients. For
general scattered non-uniform points, use `nusht_solve!` for exact inversion.

Algorithm (Y† = S† · D† · F† · N†):
1. N†: FINUFFT type 1 at the M scattered points → 2D Fourier modes on doubled torus
2. F†: adjoint 2D FFT (with θ phase correction for CC cell-center offset)
3. D†: DFS fold → equiangular CC half-sphere map (using mod-correct φ+π shift)
4. S†: `sph_transform!` = S⁻¹ on the CC grid (exact analysis/inverse, equivalent to the
   Euclidean adjoint when input points lie on the Clenshaw-Curtis quadrature grid)
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

    F̃_real = ifft2_from_coeffs(Fhat_2d, plan)

    F_folded = dfs_fold(F̃_real)

    C .= F_folded
    FastSphericalHarmonics.sph_transform!(C)

    return C
end

"""
    _nusht_true_adjoint!(C, f, plan)

Internal: compute the **true Euclidean adjoint** of `nusht_type2!` at arbitrary
scattered points. Unlike `nusht_type1!`, this uses `PS’·P’` as the S† step (the
exact matrix-transpose adjoint of `sph_evaluate!` = PS·P), making the composite
operator `A† = S†·D†·F†·N†` the true adjoint of `A = N·F·D·S`.

Used internally by `nusht_solve!` to build the normal equations `A†Ac = A†f`.
At CC grid points `nusht_type1!` (using `sph_transform!`) and this function are
equivalent (they differ only when input points are off the CC grid).
"""
function _nusht_true_adjoint!(C, f, plan::NUSHTplan{T}) where {T}
    (; Nθ, Nφ, tol) = plan

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

    Fhat_2d  = dropdims(Fhat_vec, dims=3)
    F̃_real  = ifft2_from_coeffs(Fhat_2d, plan)
    F_folded = dfs_fold(F̃_real)

    C .= F_folded
    LinearAlgebra.lmul!(plan.sph_plan_synth', C)
    LinearAlgebra.lmul!(plan.sph_plan', C)

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
    LinearAlgebra.lmul!(plan.sph_plan, F)
    LinearAlgebra.lmul!(plan.sph_plan_synth, F)

    F̃ .= zero(T)
    dfs_double!(F̃, F)

    Fhat_2d = fft2_to_coeffs(F̃, plan)

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

"""
    nusht_filter_renorm!(f_out, mask, filter, plan)

Renormalise the output of `nusht_filter!` to correct for land/ocean masking.

When a mask is applied by zeroing out land points in `f_in` before calling
`nusht_filter!`, the filtered result at each ocean point is biased toward zero
because the kernel integrates over land area where the field was forced to 0.
This function corrects that bias by dividing the filtered field by the
locally-filtered mask (the fraction of kernel weight falling over ocean).

Arguments:
- `f_out`: filtered field (overwritten in-place); must have been computed via
  `nusht_filter!(f_out, f_masked, filter, plan)` where `f_masked = f .* mask`.
- `mask`: binary (0/1) or fractional land-sea mask at the M scattered points;
  1 = ocean (valid), 0 = land (masked).
- `filter`: the same filter used in `nusht_filter!`.
- `plan`: the same NUSHTplan.

Points where the filtered mask is below a small threshold (< 0.01) are set to
0 to avoid division by near-zero values near coasts.
"""
function nusht_filter_renorm!(f_out, mask, filter, plan::NUSHTplan{T}) where {T}
    mask_filt = similar(f_out)
    mask_T = T.(mask)
    nusht_filter!(mask_filt, mask_T, filter, plan)
    threshold = T(0.01)
    for i in eachindex(f_out)
        w = mask_filt[i]
        f_out[i] = abs(w) >= threshold ? f_out[i] / w : zero(T)
    end
    return f_out
end

# ─────────────────────────────────────────────────────────────────────────────
# Exact inversion: Conjugate Gradient solver
# ─────────────────────────────────────────────────────────────────────────────

"""
    nusht_solve!(C, f, plan; maxiter=500, rtol=1e-6, verbose=false)

**Exact inversion:** Solve `A c = f` for spherical harmonic coefficients `C`
given field values `f` at the M scattered points in `plan`, using Conjugate
Gradients on the normal equations `(A†A) c = A† f`.

This gives the minimum-norm least-squares solution and is exact (to tolerance
`rtol`) for any distribution of scattered points with M ≥ (lmax+1)², provided
the points are reasonably well-distributed (no large gaps at scale 1/lmax).

Unlike `nusht_type1!` (which uses `sph_transform!`, the exact inverse only on the
Clenshaw-Curtis grid), this function uses the **true Euclidean adjoint** `A†`
(`_nusht_true_adjoint!` via `PS'·P'`), making `A†A` symmetric positive definite
and CG guaranteed to converge.

Arguments:
- `C`:       output coefficient array (lmax+1)×(2lmax+1), overwritten in-place
- `f`:       input field values at M scattered points
- `plan`:    NUSHTplan with pre-computed NUFFT and FFTW plans
- `maxiter`: maximum CG iterations (default 500)
- `rtol`:    relative residual tolerance for convergence (default 1e-6)
- `verbose`: print residual at each iteration if true

Returns `(C, iters, rel_res)` where `iters` is the number of CG iterations
performed and `rel_res` is the final relative residual `‖r‖/‖A†f‖`.
"""
function nusht_solve!(
    C, f, plan::NUSHTplan{T};
    maxiter::Int  = 500,
    rtol::Real    = 1e-6,
    verbose::Bool = false,
) where {T}
    (; Nθ, Nφ) = plan
    K = Nθ * Nφ

    @assert size(C) == (Nθ, Nφ)
    @assert length(f) == length(plan.θ_nodes)

    buf_C  = zeros(T, Nθ, Nφ)
    buf_f  = zeros(T, length(f))
    buf_C2 = zeros(T, Nθ, Nφ)

    function matvec!(y, x)
        buf_C .= reshape(x, Nθ, Nφ)
        nusht_type2!(buf_f, buf_C, plan)
        _nusht_true_adjoint!(buf_C2, buf_f, plan)
        y .= vec(buf_C2)
    end

    rhs = zeros(T, K)
    _nusht_true_adjoint!(buf_C, f, plan)
    rhs .= vec(buf_C)
    rhs_norm = LinearAlgebra.norm(rhs)

    x = zeros(T, K)
    r = copy(rhs)
    p = copy(r)
    rsold = LinearAlgebra.dot(r, r)
    Ap = zeros(T, K)

    rel_res = one(T)
    iters = 0
    for i in 1:maxiter
        iters = i
        matvec!(Ap, p)
        α = rsold / LinearAlgebra.dot(p, Ap)
        x .+= α .* p
        r .-= α .* Ap
        rsnew = LinearAlgebra.dot(r, r)
        rel_res = sqrt(rsnew) / rhs_norm
        verbose && @info "nusht_solve! iter $i: rel_res=$rel_res"
        if rel_res < rtol
            break
        end
        p .= r .+ (rsnew / rsold) .* p
        rsold = rsnew
    end

    C .= reshape(x, Nθ, Nφ)
    return C, iters, rel_res
end

# ─────────────────────────────────────────────────────────────────────────────
# Internal FFT helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    fft2_to_coeffs(F̃, plan) -> Fhat

2D FFT of the doubled map F̃ (size Nθ_dbl × Nφ), returning complex Fourier
coefficients in the layout expected by FINUFFT Type 2 (row-major Nφ × Nθ_dbl,
with FINUFFT-centered mode order).

The DFS doubled grid places data at cell-center positions θ̃ᵢ = 2π/Nθ_dbl*(i-0.5),
which are offset by half a pixel from FFTW's assumed integer-index positions.
A per-mode θ phase correction exp(-πi kθ/Nθ_dbl) compensates for this offset so
that nufft2d2 evaluated at natural θ ∈ [0,π] coordinates gives the correct values.
The φ grid φⱼ = 2π/Nφ*(j-1) is already zero-based (no φ phase correction needed).
"""
function fft2_to_coeffs(F̃, plan::NUSHTplan)
    Nθ_dbl = 2 * plan.Nθ; Nφ = plan.Nφ
    @assert size(F̃) == (Nθ_dbl, Nφ)
    Fhat_raw = plan.fft_plan * F̃  # pre-planned FFT, no per-call FFTW planning
    Fhat_corrected = (Fhat_raw .* plan.phase_θ) ./ (Nθ_dbl * Nφ)
    return collect(FFTW.fftshift(Fhat_corrected)')
end

"""
    ifft2_from_coeffs(Fhat_2d, plan) -> F̃

Adjoint of `fft2_to_coeffs`: maps FINUFFT type-1 mode coefficients back to a
spatial map on the doubled cell-center grid (Nθ_dbl × Nφ).

Applies the conjugate θ phase correction exp(+πi kθ/Nθ_dbl) before the inverse
FFT to correctly invert the half-pixel offset compensation in `fft2_to_coeffs`.
"""
function ifft2_from_coeffs(Fhat_2d, plan::NUSHTplan)
    Fhat_shifted = FFTW.ifftshift(collect(Fhat_2d'))  # back to FFTW natural order
    return real.(plan.ifft_plan * (Fhat_shifted .* plan.phase_θ_conj))
end

end # module NUFSHT
