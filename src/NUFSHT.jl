"""
    NUFSHT.jl — Non-Uniform Fast Spherical Harmonic Transform (native Julia)

Double Fourier Sphere (DFS) + nuFFT spherical harmonic transforms at arbitrary scattered
`(colatitude, longitude)` points. The synthesis operator factors as `A = N·F·D·S`:

- **S** (`plan_sph2fourier`/`plan_sph_synthesis`): iso-latitude Legendre step between SH
  coefficients and an equiangular Clenshaw-Curtis grid.
- **D** (`dfs_double!`/`dfs_fold!`): "doubling" that extends colatitude [0,π] → [0,2π) across the
  south pole, making the field doubly-periodic.
- **F** (`fft_plan`/`ifft_plan` + half-pixel phase): 2D FFT on the doubled torus.
- **N** (FINUFFT guru type 2 / type 1): non-uniform FFT evaluating the 2D Fourier series at the
  scattered points.

A [`NUSHTplan`](@ref) owns persistent FINUFFT guru plans (built once, points set once) and every
work buffer, so repeated transforms — filtering, or the hundreds of matvecs in [`nusht_solve!`](@ref)
— allocate nothing and never re-plan. All calls transform a batch of `B = plan.B` co-located fields
(`ntrans`); `B = 1` methods accept plain vectors/matrices.

## References
- Merilees (1973); Townsend & Olver (2015); Reinecke & Seljebotn (2013, A&A 554 A112);
  Keiner, Kunis & Potts (2009); Belkner et al. (2024, arXiv:2406.14542).
- FastSphericalHarmonics.jl, FINUFFT.jl, FastTransforms.jl.
"""
module NUFSHT

using FFTW: FFTW
using FINUFFT: FINUFFT
using FastSphericalHarmonics: FastSphericalHarmonics
using LinearAlgebra: LinearAlgebra

include("DFS.jl")
include("Plan.jl")
include("Kernels.jl")
include("Spin.jl")

export make_plan, NUSHTplan, close!, CGWorkspace
export nusht_type1!, nusht_type2!, nusht_filter!, nusht_filter_renorm!, nusht_solve!
export TopHatTransfer, GaussianTransfer, SharpSpectralTransfer
export kernel_transfer, cutoff_degree, gaussian_from_scale
export nusht_type2_threaded!, nusht_type1_threaded!, nusht_solve_threaded!
export nusht_type2_distributed, nusht_solve_distributed
export nusht_adjoint_mpi!, nusht_solve_mpi!
export plot_field

"""
    plot_field(θ, φ, f; colormap=:RdBu, markersize=8, title="", colorbarlabel="Field value") -> Figure

Scatter-plot a scalar field `f` sampled at scattered colatitude/longitude points
(`θ ∈ [0,π]`, `φ ∈ [0,2π)`), coloured by `real(f)` (so a complex/spin field plots its real part).
Longitude on x, colatitude on y (poles top/bottom). Method supplied by `NUFSHTCairoMakieExt` — load
it with `using CairoMakie`.
"""
function plot_field end

# ── Parallel-execution API (methods supplied by the accelerator extensions) ────
# Two shapes, reflecting how each backend shares work:
#
#  • In-process (`NUFSHTOhMyThreadsExt`, `using OhMyThreads`): a single FINUFFT plan is not safe to
#    `exec!` concurrently, so the *_threaded! methods parallelize over independent
#    (output, input, plan) triples — one plan per task — mutating the outputs in place.
#
#  • Cross-process (`NUFSHTDistributedExt`, `using Distributed`): FINUFFT plans hold C pointers and
#    cannot be serialized to workers, so the *_distributed methods take *node sets* and each worker
#    builds its own plan (farm of independent problems), returning fresh results.
#
#  • `NUFSHTMPIExt` (`using MPI`) point-decomposes a *single* transform across ranks (see the ext).

"""
    nusht_type2_threaded!(fs, Cs, plans)

Synthesize a collection of independent problems in parallel: for each `i`, `nusht_type2!(fs[i],
Cs[i], plans[i])`. One plan per task (FINUFFT plans are not concurrency-safe). Requires an
extension — e.g. `using OhMyThreads` or `using Distributed`.
"""
function nusht_type2_threaded! end

"""
    nusht_type1_threaded!(Cs, fs, plans)

Parallel adjoint analysis over independent problems; see [`nusht_type2_threaded!`](@ref).
"""
function nusht_type1_threaded! end

"""
    nusht_solve_threaded!(Cs, fs, plans; kwargs...)

Parallel exact inversion over independent problems: for each `i`, `nusht_solve!(Cs[i], fs[i],
plans[i]; kwargs...)`. See [`nusht_type2_threaded!`](@ref).
"""
function nusht_solve_threaded! end

"""
    nusht_type2_distributed(θs, φs, Cs, lmax; kwargs...) -> fs

Farm `N` independent synthesis problems across `Distributed` workers: for each `i`, a plan is built
on a worker from `(θs[i], φs[i])`, `nusht_type2!` evaluates `Cs[i]`, and the field is returned as
`fs[i]`. `kwargs` are forwarded to `make_plan` (`tol`, `T`, `ntrans`, `nthreads`). Requires
`using Distributed` (and `@everywhere using NUFSHT`).
"""
function nusht_type2_distributed end

"""
    nusht_solve_distributed(θs, φs, fs, lmax; kwargs...) -> Cs

Farm `N` independent inversions across `Distributed` workers (one plan built per problem on a
worker). Returns the recovered coefficient arrays. See [`nusht_type2_distributed`](@ref).
"""
function nusht_solve_distributed end

"""
    nusht_adjoint_mpi!(C, f_local, plan_local, comm)

MPI point-decomposed **adjoint** `A†f`. Each rank owns a disjoint subset of the `M` points with a
local plan; since `A†` is a sum over points, each rank computes its local contribution and the
result is `MPI.Allreduce!`-summed into `C` (replicated on every rank; communication is O(lmax²),
independent of `M`). Requires `using MPI`.
"""
function nusht_adjoint_mpi! end

"""
    nusht_solve_mpi!(C, f_local, plan_local, comm; maxiter, rtol, verbose) -> (C, iters, rel_res)

MPI point-decomposed **exact inversion**: conjugate gradients on `A†A c = A†f` where the points are
partitioned across ranks. `A` (synthesis) needs no communication; `A†` and the CG inner products are
`Allreduce`d. Solves the *global* least-squares system with `C` replicated on every rank. Requires
`using MPI`.
"""
function nusht_solve_mpi! end

# FastTransforms' `__init__` starts its bundled OpenMP FFTW with `ceil(CPU_THREADS/2)` threads
# (FastTransforms/src/libfasttransforms.jl `__init__`). Its butterfly transforms and sphere FFTs then
# run inside OpenMP parallel regions. Executing such a transform from a **non-root Julia task**
# (`@async`/`Threads.@spawn`/a `Distributed` worker's message-handler task) silently corrupts the
# result — it reproduces even at `-t1` (one OS thread), so it is the OpenMP runtime being entered from
# a task context, not thread migration or oversubscription. Forcing FastTransforms to a single thread
# takes its serial code path (no OpenMP parallel region) and is exact in a task (verified round-trip
# 3e-16 in `@async` vs ~0.5 multi-threaded). `ft_set_num_threads(1)` covers the butterfly step and
# `ft_fftw_plan_with_nthreads(1)` the FFTW plans built afterward. MPI is unaffected: ranks are separate
# processes running on their main task. There is no FastTransforms thread-count getter, so the
# `__init__` default `cld(CPU_THREADS, 2)` is the value to restore.
_fasttransforms_default_nthreads() = max(1, cld(Sys.CPU_THREADS, 2))

"""
    _fasttransforms_single!()

Force FastTransforms single-threaded **without restoring** (set-only). Idempotent and race-free to
call concurrently — every caller writes the same value `1` — so it is the safe primitive to invoke
inside each farmed worker task, where the coordinator's barrier-protected restore cannot reach the
worker's process. See [`_with_fasttransforms_single`](@ref) for why single-threading is required.
"""
function _fasttransforms_single!()
    FastTransforms.ft_set_num_threads(1)
    FastTransforms.ft_fftw_plan_with_nthreads(1)
    return nothing
end

"""
    _with_fasttransforms_single(f)

Run `f()` with FastTransforms single-threaded, restoring the `__init__` default afterward. Wrap the
**entire** task-parallel section in this from the coordinating (root) task — the set happens before
any task is spawned and the restore after the join barrier, so concurrent worker tasks never touch
the global thread count and cannot race on it. (Wrapping each task's body individually instead would
let one task's restore corrupt another's in-flight transform.) Remote `Distributed` workers live in
other processes that this restore cannot reach; they call [`_fasttransforms_single!`](@ref)
themselves. See its comment above for the underlying FastTransforms OpenMP-in-task hazard.
"""
function _with_fasttransforms_single(f)
    default_nt = _fasttransforms_default_nthreads()
    try
        _fasttransforms_single!()
        return f()
    finally
        FastTransforms.ft_set_num_threads(default_nt)
        FastTransforms.ft_fftw_plan_with_nthreads(default_nt)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Shape helpers (B=1 ergonomics): user passes vectors/matrices; cores work on (…, B).
# ─────────────────────────────────────────────────────────────────────────────

@inline _npts(plan::NUSHTplan) = length(plan.θ_nodes)
@inline _slicelen(plan::NUSHTplan) = plan.Nθ * plan.Nφ

@inline function _assert_coeffs(C, plan::NUSHTplan)
    @assert length(C) == plan.Nθ * plan.Nφ * plan.B "coefficient array has $(length(C)) entries, expected $(plan.Nθ*plan.Nφ*plan.B) = Nθ·Nφ·B"
end
@inline function _assert_field(f, plan::NUSHTplan)
    @assert length(f) == _npts(plan) * plan.B "field array has $(length(f)) entries, expected $(_npts(plan)*plan.B) = M·B"
end

# Copy batch slice `b` of a linear `(Nθ·Nφ·B)` array `A` ↔ the dense `(Nθ,Nφ)` `Fslice` scratch,
# without views or reshapes (both allocate small headers per call) — keeps the hot path zero-alloc.
@inline _load_slice!(Fslice, A, plan::NUSHTplan, b) =
    copyto!(Fslice, 1, A, (b - 1) * _slicelen(plan) + 1, _slicelen(plan))
@inline _store_slice!(A, Fslice, plan::NUSHTplan, b) =
    copyto!(A, (b - 1) * _slicelen(plan) + 1, Fslice, 1, _slicelen(plan))

# Real-part extraction `fbuf → f` and field load `f → fbuf` (real↔complex), shape-agnostic (`f` may be
# `(M,)` or `(M,B)`; `fbuf` is `(M,B)` — equal length). Host: zero-alloc scalar loop. Device methods
# (a `reshape`d broadcast) live in the KA extension, dispatched on the plan buffer `fbuf`.
function _copy_real!(f, fbuf)
    @inbounds for i in eachindex(f)
        f[i] = real(fbuf[i])
    end
    return f
end
function _copy_field!(fbuf, f)
    @inbounds for i in eachindex(fbuf)
        fbuf[i] = f[i]
    end
    return fbuf
end

# ─────────────────────────────────────────────────────────────────────────────
# Internal S / D·F·N cores (operate entirely on plan buffers).
# ─────────────────────────────────────────────────────────────────────────────

# S (forward): sph_evaluate! = PS·P, applied per batch slice through the dense `Fslice`.
function _sph_evaluate!(plan::NUSHTplan)
    @inbounds for b in 1:plan.B
        _load_slice!(plan.Fslice, plan.F, plan, b)
        LinearAlgebra.lmul!(plan.sph_plan, plan.Fslice)
        LinearAlgebra.lmul!(plan.sph_plan_synth, plan.Fslice)
        _store_slice!(plan.F, plan.Fslice, plan, b)
    end
    return plan
end

# D·F·N (forward): doubled map → torus FFT → half-pixel phase → type-2 NUFFT into `fbuf`.
# `phase_scaled` is a length-2Nθ vector; broadcasting it against `Fhat` (2Nθ,Nφ,B) aligns dim 1 and
# spreads over the (φ, batch) dims — no reshape needed.
function _dfn_synthesis!(plan::NUSHTplan)
    dfs_double!(plan.F̃, plan.F)
    LinearAlgebra.mul!(plan.Fhat, plan.fft_plan, plan.F̃)
    plan.Fhat .*= plan.phase_scaled
    _nufft_exec!(plan.nufft_type2, plan.Fhat, plan.fbuf)
    return plan
end

# N†·F†·D† (adjoint): type-1 NUFFT (from `fbuf`) → conj phase → inverse FFT → fold into real `F`.
function _dfn_analysis!(plan::NUSHTplan)
    _nufft_exec!(plan.nufft_type1, plan.fbuf, plan.Fhat)
    plan.Fhat .*= plan.phase_conj
    LinearAlgebra.mul!(plan.F̃, plan.ifft_plan, plan.Fhat)
    dfs_fold!(plan.F, plan.F̃)
    return plan
end

# ─────────────────────────────────────────────────────────────────────────────
# Type 2: spherical harmonic coefficients → scattered map
# ─────────────────────────────────────────────────────────────────────────────

"""
    nusht_type2!(f, C, plan)

**Type 2 (synthesis):** evaluate the field with spherical harmonic coefficients `C` at the `M`
scattered points, writing values into `f`. Batched: `C` is `(Nθ, Nφ)` / `(Nθ, Nφ, B)` and `f` is
length-`M` / `(M, B)`.

Algorithm `A = N·F·D·S`: forward rSHT (S) → DFS double (D) → torus FFT + half-pixel phase (F) →
FINUFFT type 2 (N).
"""
function nusht_type2!(f, C, plan::NUSHTplan{T}) where {T}
    _assert_coeffs(C, plan)
    _assert_field(f, plan)
    copyto!(plan.F, C)
    _sph_evaluate!(plan)
    _dfn_synthesis!(plan)
    _copy_real!(f, plan.fbuf)
    return f
end

# ─────────────────────────────────────────────────────────────────────────────
# Type 1: scattered map → spherical harmonic coefficients
# ─────────────────────────────────────────────────────────────────────────────

"""
    nusht_type1!(C, f, plan)

**Type 1 (adjoint analysis):** given field values `f` at the `M` scattered points, compute
coefficients `C`. On the Clenshaw-Curtis grid this is the exact inverse of `nusht_type2!`
(machine-precision round-trip); at general scattered points it is only the adjoint — use
[`nusht_solve!`](@ref) for exact inversion there.

The S† step uses the CC-grid analysis `S⁻¹ = P⁻¹·PA` via the plan's persistent `plan_sph_analysis`
(PA) and `plan_sph2fourier` (P), replicating `FastSphericalHarmonics.sph_transform!` without its
per-call plan rebuild.
"""
function nusht_type1!(C, f, plan::NUSHTplan{T}) where {T}
    _assert_field(f, plan)
    _assert_coeffs(C, plan)
    _copy_field!(plan.fbuf, f)
    _dfn_analysis!(plan)
    @inbounds for b in 1:plan.B
        _load_slice!(plan.Fslice, plan.F, plan, b)
        LinearAlgebra.lmul!(plan.sph_plan_analysis, plan.Fslice)
        LinearAlgebra.ldiv!(plan.sph_plan, plan.Fslice)
        _store_slice!(C, plan.Fslice, plan, b)
    end
    return C
end

"""
    _nusht_true_adjoint!(C, f, plan)

Internal: the **exact Euclidean adjoint** of `nusht_type2!`. Identical to `nusht_type1!` except the
S† step applies `PS'·P'` (the matrix transpose of `PS·P`) via the conjugate FastTransforms plans,
making `A†A` symmetric positive definite for CG. At CC-grid points it coincides with `nusht_type1!`.
"""
function _nusht_true_adjoint!(C, f, plan::NUSHTplan{T}) where {T}
    _assert_field(f, plan)
    _assert_coeffs(C, plan)
    _copy_field!(plan.fbuf, f)
    _dfn_analysis!(plan)
    @inbounds for b in 1:plan.B
        _load_slice!(plan.Fslice, plan.F, plan, b)
        LinearAlgebra.lmul!(plan.sph_plan_synth_adj, plan.Fslice)
        LinearAlgebra.lmul!(plan.sph_plan_adj, plan.Fslice)
        _store_slice!(C, plan.Fslice, plan, b)
    end
    return C
end

# ─────────────────────────────────────────────────────────────────────────────
# Filtering
# ─────────────────────────────────────────────────────────────────────────────

"""
    nusht_filter!(f_out, f_in, filter, plan)

Apply a spectral filter to `f_in` at the scattered points, writing to `f_out` (both length-`M` /
`(M, B)`): `nusht_type1!` → `apply_transfer!` (× H(ℓ)) → `nusht_type2!`. Uses the plan's coefficient
scratch, allocation-free.
"""
function nusht_filter!(f_out, f_in, filter, plan::NUSHTplan)
    nusht_type1!(plan.C, f_in, plan)
    apply_transfer!(plan.C, filter, plan.lmax)
    nusht_type2!(f_out, plan.C, plan)
    return f_out
end

"""
    nusht_filter_renorm!(f_out, mask, filter, plan; mask_filt=similar(f_out))

Renormalise the output of `nusht_filter!` to correct for land/ocean masking: divide by the
filtered mask (the fraction of kernel weight over ocean). `f_out` must have been produced by
`nusht_filter!(f_out, f .* mask, filter, plan)`. Points where the filtered mask is below `0.01`
are set to 0. Pass a reusable `mask_filt` scratch (shaped like `f_out`) to run allocation-free.
"""
function nusht_filter_renorm!(f_out, mask, filter, plan::NUSHTplan{T}; mask_filt = similar(f_out)) where {T}
    nusht_filter!(mask_filt, mask, filter, plan)   # `mask` is read straight into `plan.fbuf` — no copy needed
    threshold = T(0.01)
    # `mask_filt` is shaped like `f_out` → a single fused broadcast, zero-alloc + device-safe.
    f_out .= ifelse.(abs.(mask_filt) .>= threshold, f_out ./ mask_filt, zero(T))
    return f_out
end

# ─────────────────────────────────────────────────────────────────────────────
# Exact inversion: batched Conjugate Gradients on the normal equations A†A c = A†f
# ─────────────────────────────────────────────────────────────────────────────

"""
    CGWorkspace(plan)

Reusable scratch for [`nusht_solve!`](@ref). Holding one across solves makes CG allocation-free.
Per-column scalars are length-`B` vectors so the `B` columns run as `B` independent single-column
CGs (batched result == looping single transforms).
"""
struct CGWorkspace{AT3<:AbstractArray, FT2<:AbstractMatrix, VT<:AbstractVector}
    x::AT3
    r::AT3
    p::AT3
    Ap::AT3
    rhs::AT3
    f::FT2
    rsold::VT
    rsnew::VT
    pAp::VT
    α::VT
    β::VT
    rel::VT
    rhsnorm::VT
end

function CGWorkspace(plan::NUSHTplan{T}) where {T}
    Nθ, Nφ, B = plan.Nθ, plan.Nφ, plan.B
    M = _npts(plan)
    # Buffers shaped like the plan's (so a device plan gets a device workspace).
    z3() = _zeros_like(plan.F, T, Nθ, Nφ, B)
    vB() = _zeros_like(plan.θ_nodes, T, B)
    return CGWorkspace(z3(), z3(), z3(), z3(), z3(), _zeros_like(plan.fbuf, T, M, B),
                       vB(), vB(), vB(), vB(), vB(), vB(), vB())
end

# Per-column reductions / updates over a (Nθ, Nφ, B) array — zero-allocation.
function _col_dot!(dst, a, b)
    @inbounds for k in axes(a, 3)
        s = zero(eltype(a))
        for j in axes(a, 2), i in axes(a, 1)
            s += a[i, j, k] * b[i, j, k]
        end
        dst[k] = s
    end
    return dst
end

function _col_axpy!(y, α, x, σ)   # y[:,:,k] += σ·α[k]·x[:,:,k]
    @inbounds for k in axes(y, 3)
        c = σ * α[k]
        for j in axes(y, 2), i in axes(y, 1)
            y[i, j, k] += c * x[i, j, k]
        end
    end
    return y
end

function _col_pbp!(p, r, β)       # p[:,:,k] = r[:,:,k] + β[k]·p[:,:,k]
    @inbounds for k in axes(p, 3)
        c = β[k]
        for j in axes(p, 2), i in axes(p, 1)
            p[i, j, k] = r[i, j, k] + c * p[i, j, k]
        end
    end
    return p
end

_AtA!(Ap, p, ws::CGWorkspace, plan::NUSHTplan) = (nusht_type2!(ws.f, p, plan); _nusht_true_adjoint!(Ap, ws.f, plan))

"""
    nusht_solve!(C, f, plan; ws=CGWorkspace(plan), maxiter=500, rtol=1e-6, verbose=false)

**Exact inversion:** solve `A c = f` for coefficients `C` via Conjugate Gradients on the normal
equations `(A†A) c = A† f`, using the true Euclidean adjoint (`_nusht_true_adjoint!`), so `A†A` is
SPD and CG converges for any well-distributed scattered point set. Batched (`B > 1`) runs the columns
as independent single-column CGs. Returns `(C, iters, rel_res)` with `rel_res = max_k ‖r_k‖/‖A†f_k‖`.
"""
function nusht_solve!(
    C, f, plan::NUSHTplan{T};
    ws::CGWorkspace = CGWorkspace(plan),
    maxiter::Int = 500,
    rtol::Real = 1e-6,
    verbose::Bool = false,
) where {T}
    _assert_coeffs(C, plan)
    _assert_field(f, plan)
    _nusht_true_adjoint!(ws.rhs, f, plan)
    _col_dot!(ws.rhsnorm, ws.rhs, ws.rhs)
    ws.rhsnorm .= sqrt.(ws.rhsnorm)

    fill!(ws.x, zero(T))
    copyto!(ws.r, ws.rhs)
    copyto!(ws.p, ws.r)
    _col_dot!(ws.rsold, ws.r, ws.r)
    fill!(ws.rel, one(T))

    iters = 0
    for i in 1:maxiter
        iters = i
        _AtA!(ws.Ap, ws.p, ws, plan)
        _col_dot!(ws.pAp, ws.p, ws.Ap)
        @. ws.α = ifelse(ws.pAp == 0, zero(T), ws.rsold / ws.pAp)
        _col_axpy!(ws.x, ws.α, ws.p, one(T))
        _col_axpy!(ws.r, ws.α, ws.Ap, -one(T))
        _col_dot!(ws.rsnew, ws.r, ws.r)
        @. ws.rel = ifelse(ws.rhsnorm == 0, zero(T), sqrt(ws.rsnew) / ws.rhsnorm)
        verbose && @info "nusht_solve! iter $i: rel_res=$(maximum(ws.rel))"
        maximum(ws.rel) < rtol && break
        @. ws.β = ifelse(ws.rsold == 0, zero(T), ws.rsnew / ws.rsold)
        _col_pbp!(ws.p, ws.r, ws.β)
        copyto!(ws.rsold, ws.rsnew)
    end

    copyto!(C, ws.x)
    return C, iters, maximum(ws.rel)
end

end # module NUFSHT
