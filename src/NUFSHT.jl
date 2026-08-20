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

using ComputationalBackends: ComputationalBackends
using FFTW: FFTW
using FastSphericalHarmonics: FastSphericalHarmonics
using LinearAlgebra: LinearAlgebra

include("DFS.jl")
include("NUFFT.jl")
include("Plan.jl")
include("Kernels.jl")
include("Spin.jl")

export make_plan, NUSHTplan, close!, CGWorkspace
export nusht_type1!, nusht_type2!, nusht_synthesize!, nusht_filter!, nusht_filter_renorm!, nusht_solve!
export TopHatTransfer, GaussianTransfer, SharpSpectralTransfer
export kernel_transfer, cutoff_degree, gaussian_from_scale
export nusht_type2_spin!, nusht_type1_spin!, nusht_solve_spin!
export nusht_type2, nusht_solve
export plot_field

"""
    plot_field(θ, φ, f; colormap=:RdBu, markersize=8, title="", colorbarlabel="Field value") -> Figure

Scatter-plot a scalar field `f` sampled at scattered colatitude/longitude points
(`θ ∈ [0,π]`, `φ ∈ [0,2π)`), coloured by `real(f)` (so a complex/spin field plots its real part).
Longitude on x, colatitude on y (poles top/bottom). Method supplied by `NUFSHTCairoMakieExt` — load
it with `using CairoMakie`.
"""
function plot_field end

# ── Parallel execution: backend dispatch ──────────────────────────────────────
# Parallelism is a `ComputationalBackends.AbstractExecutionBackend` argument, in two families:
#
#  • **Farm over independent problems** (collection methods below). A FINUFFT plan is not safe to
#    `exec!` concurrently, so each problem carries its own. `DistributedBackend` needs the node-set
#    form — plans hold C pointers and cannot be serialized, so each worker builds its own.
#  • **Decompose one transform** — `MPIBackend`, partitioning the `M` points across ranks.
#
# GPU is not a backend argument: it follows from the plan's array types.

@inline _omt_loaded() = Base.get_extension(@__MODULE__, :NUFSHTOhMyThreadsExt) !== nothing

"""
    _resolve_backend(backend) -> AbstractExecutionBackend

Concrete backends pass through untouched. `AutoBackend` is the only thing allowed to choose, and it
chooses on **real capability**: a `ThreadedBackend` only when Julia has more than one thread *and* the
OhMyThreads extension is loaded, otherwise `SerialBackend`. A backend the caller named explicitly is
either honoured exactly or refused — never silently downgraded.

This is deliberately NUFSHT's own function rather than a method on `ComputationalBackends.resolve_backend`,
which would be type piracy.
"""
@inline _resolve_backend(backend::ComputationalBackends.AbstractExecutionBackend) = backend
@inline _resolve_backend(::ComputationalBackends.AbstractAutoBackend) =
    (Threads.nthreads() > 1 && _omt_loaded()) ? ComputationalBackends.ThreadedBackend() :
                                                ComputationalBackends.SerialBackend()

_backend_unavailable(backend, what) = throw(ArgumentError(
    "$(nameof(typeof(backend))) cannot run $what here — the extension providing it is not loaded. " *
    "ThreadedBackend needs `using OhMyThreads`, DistributedBackend `using Distributed`, " *
    "MPIBackend `using MPI`."))

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
    # FastTransforms and FFTW.jl share one libfftw3, so the planner count has a real getter and can be
    # restored exactly; the butterfly count has none, so that one falls back to the `__init__` default.
    prev_planner = FFTW.get_num_threads()
    try
        _fasttransforms_single!()
        return f()
    finally
        FastTransforms.ft_set_num_threads(_fasttransforms_default_nthreads())
        FastTransforms.ft_fftw_plan_with_nthreads(prev_planner)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Shape helpers (B=1 ergonomics): user passes vectors/matrices; cores work on (…, B).
# ─────────────────────────────────────────────────────────────────────────────

@inline _npts(plan::NUSHTplan) = length(_θnodes(plan))
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

# The NUFFT always returns complex strengths; a real field drops the imaginary residue.
_copy_out!(f, fbuf, ::AbstractNUSHTplan) = _copy_field!(f, fbuf)
_copy_out!(f, fbuf, ::NUSHTplan{T,FE}) where {T,FE<:Real} = _copy_real!(f, fbuf)
_copy_out!(f, fbuf, ::NUSHTplan{T,FE}) where {T,FE<:Complex} = _copy_field!(f, fbuf)

# ─────────────────────────────────────────────────────────────────────────────
# Internal S / D·F·N cores (operate entirely on plan buffers).
# ─────────────────────────────────────────────────────────────────────────────

# Walk the active columns applying a per-column sphere operation. FastTransforms has no batched `lmul!`
# for these, so the column loop is the only parallelism available; each spawned task takes its own
# slice buffer and its own plans from the pool, keyed by chunk (see `_sph_pool`). The whole region runs
# under `_with_fasttransforms_single` — FastTransforms returns WRONG RESULTS from a non-root task while
# its OpenMP regions are live, so that pin is a correctness requirement, not a tuning choice.
function _sph_columns!(op!, plan::NUSHTplan, k::Integer)
    pool = plan.sph_pool
    nt = min(length(pool), k)
    if nt <= 1
        @inbounds for b in 1:k
            op!(plan.Fslice, plan.sph_plan, plan.sph_plan_synth,
                plan.sph_plan_adj, plan.sph_plan_synth_adj, b)
        end
        return plan
    end
    _with_fasttransforms_single() do
        @sync for c in 1:nt
            Threads.@spawn begin
                sl, P, PS, _PA, Padj, PSadj = pool[c]
                for b in c:nt:k
                    op!(sl, P, PS, Padj, PSadj, b)
                end
            end
        end
    end
    return plan
end

# S (forward): sph_evaluate! = PS·P, applied per batch slice through a dense slice buffer.
function _sph_evaluate!(plan::NUSHTplan, k::Integer = plan.B)
    _sph_columns!(plan, k) do sl, P, PS, _Padj, _PSadj, b
        _load_slice!(sl, plan.F, plan, b)
        LinearAlgebra.lmul!(P, sl)
        LinearAlgebra.lmul!(PS, sl)
        _store_slice!(plan.F, sl, plan, b)
    end
    return plan
end

# D·F·N (forward): doubled map → torus FFT → half-pixel phase → type-2 NUFFT into `fbuf`.
# `phase_scaled` is a length-2Nθ vector; broadcasting it against `Fhat` (2Nθ,Nφ,B) aligns dim 1 and
# spreads over the (φ, batch) dims — no reshape needed.
# A real field's spectrum is Hermitian, so `rfft`'s `k₁ ≥ 0` half, weighted by `phase_scaled`'s
# `{1,2,…,2,1}`, reproduces the whole sum. Which of the two mode layouts is in use is carried by the
# plan's `θ_shift` type: without one, the half goes into the head of a full-length array whose tail
# stays zero; with one, the array is the half itself, in centered order on both axes.
function _pack_modes!(Fhat, Fhalf, phase, ::Nothing)
    n1, n2 = size(Fhat, 1), size(Fhat, 2)
    nh = size(Fhalf, 1)
    @inbounds for b in axes(Fhalf, 3), j in 1:n2
        d = (b - 1) * n1 * n2 + (j - 1) * n1
        s = (b - 1) * nh * n2 + (j - 1) * nh
        @simd for i in 1:nh
            Fhat[d + i] = Fhalf[s + i] * phase[i]
        end
    end
    return Fhat
end

function _unpack_modes!(Fhalf, Fhat, phase, ::Nothing)
    n1, n2 = size(Fhat, 1), size(Fhat, 2)
    nh = size(Fhalf, 1)
    z = zero(eltype(Fhat))
    @inbounds for b in axes(Fhalf, 3), j in 1:n2
        d = (b - 1) * n1 * n2 + (j - 1) * n1
        s = (b - 1) * nh * n2 + (j - 1) * nh
        @simd for i in 1:nh
            Fhalf[s + i] = Fhat[d + i] * phase[i]
        end
        # Type-1 writes the whole array; synthesis needs the negative half zero.
        @simd for i in (nh + 1):n1
            Fhat[d + i] = z
        end
    end
    return Fhalf
end

function _pack_modes!(Fhat, Fhalf, phase, ::AbstractVector)
    n2 = size(Fhalf, 2)
    h = n2 ÷ 2
    @inbounds for b in axes(Fhalf, 3), j in 1:n2
        @views Fhat[:, mod1(j + h, n2), b] .= Fhalf[:, j, b] .* phase
    end
    return Fhat
end

function _unpack_modes!(Fhalf, Fhat, phase, ::AbstractVector)
    n2 = size(Fhalf, 2)
    h = n2 ÷ 2
    @inbounds for b in axes(Fhalf, 3), j in 1:n2
        @views Fhalf[:, j, b] .= Fhat[:, mod1(j + h, n2), b] .* phase
    end
    return Fhalf
end

# Undo the constant wavenumber offset centered ordering imposes; `conj` keeps the pair an exact transpose.
_rephase!(fbuf, ::Nothing) = fbuf
_rephase!(fbuf, s::AbstractVector) = (fbuf .*= s; fbuf)
_rephase_conj!(fbuf, ::Nothing) = fbuf
_rephase_conj!(fbuf, s::AbstractVector) = (fbuf .*= conj.(s); fbuf)

# Leading `k` columns of a batch buffer. Contiguous, so the width-`k` plans apply to it exactly.
@inline _pfx(A::AbstractArray{<:Any,3}, k::Integer) = view(A, :, :, 1:k)
@inline _pfx(A::AbstractMatrix, k::Integer) = view(A, :, 1:k)

# Size-dependent plans for a working width. The full width uses the plan's own, so that path is
# untouched by the pool's existence; narrower widths come from `size_pool`.
# Both branches return the pool's element type, so this stays statically dispatched. A miss builds and
# caches the width, which is why the full-width branch comes first: it is the common case and never
# touches the cache.
@inline function _width_plans(plan::NUSHTplan, k::Integer)
    k == plan.B && return (k = Int(k), fft_plan = plan.fft_plan, ifft_plan = plan.ifft_plan,
                           nufft_type2 = _nufft2(plan), nufft_type1 = _nufft1(plan))
    @inbounds for e in plan.size_pool
        e.k == k && return e
    end
    return _build_width!(plan, k)
end

function _dfn_synthesis!(plan::NUSHTplan{T,FE}, k::Integer = plan.B) where {T,FE<:Real}
    e = _width_plans(plan, k)
    if k == plan.B
        _dfn_synth_real!(plan, plan.F, plan.F̃, plan.Fhalf, plan.Fhat, _fbuf(plan),
                         e.fft_plan, e.nufft_type2)
    else
        _dfn_synth_real!(plan, _pfx(plan.F, k), _pfx(plan.F̃, k), _pfx(plan.Fhalf, k),
                         _pfx(plan.Fhat, k), _pfx(_fbuf(plan), k), e.fft_plan, e.nufft_type2)
    end
    return plan
end

function _dfn_synth_real!(plan, F, F̃, Fhalf, Fhat, fbuf, fp, nu2)
    dfs_double!(F̃, F)
    LinearAlgebra.mul!(Fhalf, fp, F̃)
    _pack_modes!(Fhat, Fhalf, plan.phase_scaled, _θshift(plan))
    _nufft_exec!(nu2, Fhat, fbuf)
    _rephase!(fbuf, _θshift(plan))
    return nothing
end

function _dfn_synthesis!(plan::NUSHTplan{T,FE}, k::Integer = plan.B) where {T,FE<:Complex}
    e = _width_plans(plan, k)
    if k == plan.B
        _dfn_synth_cplx!(plan, plan.F, plan.F̃, plan.Fhat, _fbuf(plan), e.fft_plan, e.nufft_type2)
    else
        _dfn_synth_cplx!(plan, _pfx(plan.F, k), _pfx(plan.F̃, k), _pfx(plan.Fhat, k),
                         _pfx(_fbuf(plan), k), e.fft_plan, e.nufft_type2)
    end
    return plan
end

function _dfn_synth_cplx!(plan, F, F̃, Fhat, fbuf, fp, nu2)
    dfs_double!(F̃, F)
    LinearAlgebra.mul!(Fhat, fp, F̃)
    Fhat .*= plan.phase_scaled
    _nufft_exec!(nu2, Fhat, fbuf)
    return nothing
end

# N†·F†·D† (adjoint): type-1 NUFFT (from `fbuf`) → conj phase → inverse FFT → fold into `F`.
# `phase_conj` carries no Hermitian weight — `brfft` already applies that doubling, and applying it on
# both sides breaks the adjoint identity `nusht_solve!` depends on.
function _dfn_analysis!(plan::NUSHTplan{T,FE}, k::Integer = plan.B) where {T,FE<:Real}
    e = _width_plans(plan, k)
    if k == plan.B
        _dfn_ana_real!(plan, plan.F, plan.F̃, plan.Fhalf, plan.Fhat, _fbuf(plan),
                       e.ifft_plan, e.nufft_type1)
    else
        _dfn_ana_real!(plan, _pfx(plan.F, k), _pfx(plan.F̃, k), _pfx(plan.Fhalf, k),
                       _pfx(plan.Fhat, k), _pfx(_fbuf(plan), k), e.ifft_plan, e.nufft_type1)
    end
    return plan
end

function _dfn_ana_real!(plan, F, F̃, Fhalf, Fhat, fbuf, ip, nu1)
    _rephase_conj!(fbuf, _θshift(plan))
    _nufft_exec!(nu1, fbuf, Fhat)
    _unpack_modes!(Fhalf, Fhat, plan.phase_conj, _θshift(plan))
    LinearAlgebra.mul!(F̃, ip, Fhalf)
    dfs_fold!(F, F̃)
    return nothing
end

function _dfn_analysis!(plan::NUSHTplan{T,FE}, k::Integer = plan.B) where {T,FE<:Complex}
    e = _width_plans(plan, k)
    if k == plan.B
        _dfn_ana_cplx!(plan, plan.F, plan.F̃, plan.Fhat, _fbuf(plan), e.ifft_plan, e.nufft_type1)
    else
        _dfn_ana_cplx!(plan, _pfx(plan.F, k), _pfx(plan.F̃, k), _pfx(plan.Fhat, k),
                       _pfx(_fbuf(plan), k), e.ifft_plan, e.nufft_type1)
    end
    return plan
end

function _dfn_ana_cplx!(plan, F, F̃, Fhat, fbuf, ip, nu1)
    _nufft_exec!(nu1, fbuf, Fhat)
    Fhat .*= plan.phase_conj
    LinearAlgebra.mul!(F̃, ip, Fhat)
    dfs_fold!(F, F̃)
    return nothing
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
function nusht_type2!(f, C, plan::NUSHTplan{T}, k::Integer = plan.B,
                      kdfn::Integer = k) where {T}
    _assert_coeffs(C, plan)
    _assert_field(f, plan)
    copyto!(plan.F, C)
    _sph_evaluate!(plan, k)
    _dfn_synthesis!(plan, kdfn)
    _copy_out!(f, _fbuf(plan), plan)
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
    _copy_field!(_fbuf(plan), f)
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
function _nusht_true_adjoint!(C, f, plan::NUSHTplan{T}, k::Integer = plan.B,
                             kdfn::Integer = k) where {T}
    _assert_field(f, plan)
    _assert_coeffs(C, plan)
    _copy_field!(_fbuf(plan), f)
    _dfn_analysis!(plan, kdfn)
    _sph_columns!(plan, k) do sl, _P, _PS, Padj, PSadj, b
        _load_slice!(sl, plan.F, plan, b)
        LinearAlgebra.lmul!(PSadj, sl)
        LinearAlgebra.lmul!(Padj, sl)
        _store_slice!(C, sl, plan, b)
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
    nusht_synthesize!(f_out, plan.C, filter, plan)
    return f_out
end

"""
    nusht_synthesize!(f_out, C, filter, plan) -> f_out

Scale coefficients `C` by `filter`'s transfer function and synthesize at the plan's points. `C` is not
modified — the scaling runs in the plan's own scratch — so one analysis feeds any number of filters:

```julia
nusht_type1!(C, f, plan)
for (out, filt) in zip(outs, filters)
    nusht_synthesize!(out, C, filt, plan)
end
```

[`nusht_filter!`](@ref) is `nusht_type1!` followed by this, and so re-analyses per filter; use the pair
when filtering one field at several scales.
"""
function nusht_synthesize!(f_out, C, filter, plan::NUSHTplan)
    _assert_coeffs(C, plan)
    C === plan.C || copyto!(plan.C, C)
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
function nusht_filter_renorm!(f_out, mask, filter, plan::NUSHTplan{T};
                              mask_filt = similar(f_out),
                              C_mask = nothing) where {T}
    # `C_mask` lets a multi-scale caller analyse the scale-independent mask once and pass its
    # coefficients to every call; without it the mask is analysed here, as before.
    if C_mask === nothing
        nusht_type1!(plan.C, mask, plan)
        nusht_synthesize!(mask_filt, plan.C, filter, plan)
    else
        nusht_synthesize!(mask_filt, C_mask, filter, plan)
    end
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
struct CGWorkspace{AT3<:AbstractArray, FT2<:AbstractMatrix, VT<:AbstractVector, VM<:AbstractArray,
                   PV<:AbstractVector{<:Integer}}
    x::AT3
    r::AT3
    p::AT3
    Ap::AT3
    rhs::AT3
    f::FT2
    # `1` on the (lmax+1)^2 slots holding degrees l ≤ lmax, `0` on the supernumerary ones. The array is
    # a square, invertible representation carrying degrees up to `lmax+|m|`, so the transform needs all
    # of it — but a least-squares fit must name its space, and only `l ≤ lmax` is SO(3)-invariant.
    # Fitting the ragged set instead makes the answer depend on the coordinate frame.
    valid::VM
    # Slot -> original column. Compaction moves live columns to the front, so a slot's identity is
    # only recoverable through this; it is what keeps a result from being written to the wrong column.
    perm::PV
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
    vB() = _zeros_like(_θnodes(plan), T, B)
    return CGWorkspace(z3(), z3(), z3(), z3(), z3(), _zeros_like(_fbuf(plan), T, M, B),
                       _valid_mask(plan.F, T, plan.lmax),
                       collect(1:B),
                       vB(), vB(), vB(), vB(), vB(), vB(), vB())
end

# Per-column reductions / updates over a (Nθ, Nφ, B) array — zero-allocation.
# `n` bounds the columns visited, so a compacted solve reduces only over its live prefix. Taken as a
# count rather than a view: a `SubArray` per call would put allocation back on the CG hot path.
function _col_dot!(dst, a, b, n::Integer = size(a, 3))
    @inbounds for k in 1:n
        s = zero(eltype(a))
        for j in axes(a, 2), i in axes(a, 1)
            s += a[i, j, k] * b[i, j, k]
        end
        dst[k] = s
    end
    return dst
end

function _col_axpy!(y, α, x, σ, n::Integer = size(y, 3))   # y[:,:,k] += σ·α[k]·x[:,:,k]
    @inbounds for k in 1:n
        c = σ * α[k]
        for j in axes(y, 2), i in axes(y, 1)
            y[i, j, k] += c * x[i, j, k]
        end
    end
    return y
end

function _col_pbp!(p, r, β, n::Integer = size(p, 3))       # p[:,:,k] = r[:,:,k] + β[k]·p[:,:,k]
    @inbounds for k in 1:n
        c = β[k]
        for j in axes(p, 2), i in axes(p, 1)
            p[i, j, k] = r[i, j, k] + c * p[i, j, k]
        end
    end
    return p
end

# `P A†A P`. Projecting after the adjoint is enough: `p` is already in the valid subspace by induction
# (x₀ = 0, r₀ = P A†f, and every later direction is a combination of projected vectors), so `A P p = A p`.
# Converged columns are skipped in the sphere loops, so their `Ap` slices go stale. That is safe
# because `pAp` contracts `Ap` against `p`, and `nusht_solve!` holds `p` at zero for exactly those
# columns — nothing stale ever reaches `x`. Asserted in the solve tests.
# Largest of the first `n` entries, without a view (which would allocate on the CG hot path).
@inline function _max_prefix(v, n::Integer)
    m = zero(eltype(v))
    @inbounds for i in 1:n
        v[i] > m && (m = v[i])
    end
    return m
end

# Working width for `n` live columns: the next power of two, capped at `B`. Computed rather than looked
# up in a ladder, so it allocates nothing on the CG hot path — `_build_width!` builds any width on
# demand, so the powers of two exist only to bound how many distinct plan sets a solve can create.
@inline _fit_width(n::Integer, B::Integer) = min(Int(B), Int(nextpow(2, max(n, 1))))

@inline function _copy_col!(dst, dsl::Integer, src, ssl::Integer, len::Integer)
    d0 = (dsl - 1) * len; s0 = (ssl - 1) * len
    @inbounds @simd for i in 1:len
        dst[d0 + i] = src[s0 + i]
    end
    return dst
end

"""
    _retire_and_compact!(C, ws, rtol, nlive, len) -> nlive′

Write out any live column that has reached `rtol` — into `C` at its *original* index, taken from
`ws.perm`, since its answer is final and must not be carried further — then close the gaps so the
remaining live columns occupy slots `1:nlive′`. Every per-column quantity that survives an iteration
moves with its column; `pAp`, `rsnew`, `α` and `β` are recomputed each iteration and so are not moved.
"""
function _retire_and_compact!(C, ws::CGWorkspace, rtol, nlive::Integer, len::Integer)
    w = 0
    @inbounds for s in 1:nlive
        if ws.rel[s] < rtol
            _copy_col!(C, ws.perm[s], ws.x, s, len)      # final answer, to its own column
            continue
        end
        w += 1
        if w != s
            _copy_col!(ws.x, w, ws.x, s, len)
            _copy_col!(ws.r, w, ws.r, s, len)
            _copy_col!(ws.p, w, ws.p, s, len)
            _copy_col!(ws.rhs, w, ws.rhs, s, len)
            ws.rsold[w] = ws.rsold[s]
            ws.rel[w] = ws.rel[s]
            ws.rhsnorm[w] = ws.rhsnorm[s]
            # Swap, not assign: the retired originals stay somewhere in the tail, so `perm` remains a
            # bijection over 1:B and can be asserted as one. Assigning leaves duplicates behind, which
            # still scatters correctly today (only the live prefix is read) but is one refactor away
            # from writing two columns to the same destination.
            ws.perm[w], ws.perm[s] = ws.perm[s], ws.perm[w]
        end
    end
    return w
end

# `k` is the live column count, which the per-column sphere loops always narrow to. `kdfn` is the
# width of the FFT/NUFFT plan set, which can only narrow where the backend's plans are
# width-polymorphic (`_width_polymorphic`); elsewhere it stays at `B` and those transforms run wide.
_AtA!(Ap, p, ws::CGWorkspace, plan::NUSHTplan, k::Integer = plan.B, kdfn::Integer = k) =
    (nusht_type2!(ws.f, p, plan, k, kdfn);
     _nusht_true_adjoint!(Ap, ws.f, plan, k, kdfn); Ap .*= ws.valid; Ap)

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
    ws.rhs .*= ws.valid
    _col_dot!(ws.rhsnorm, ws.rhs, ws.rhs)
    ws.rhsnorm .= sqrt.(ws.rhsnorm)

    fill!(ws.x, zero(T))
    copyto!(ws.r, ws.rhs)
    copyto!(ws.p, ws.r)
    _col_dot!(ws.rsold, ws.r, ws.r)
    fill!(ws.rel, one(T))
    @inbounds for b in 1:plan.B
        ws.perm[b] = b
    end

    # A column that reaches `rtol` is written out and dropped, and the survivors are packed into the
    # front slots, so every transform runs at the live width rather than at `B`. `sizes` is the ladder
    # of widths a plan set exists (or can be built) for.
    len = _slicelen(plan)
    nlive = plan.B
    width = plan.B
    iters = 0
    worst = one(T)
    for i in 1:maxiter
        iters = i
        _AtA!(ws.Ap, ws.p, ws, plan, nlive, width)
        _col_dot!(ws.pAp, ws.p, ws.Ap, nlive)
        @. ws.α = ifelse(ws.pAp == 0, zero(T), ws.rsold / ws.pAp)
        _col_axpy!(ws.x, ws.α, ws.p, one(T), nlive)
        _col_axpy!(ws.r, ws.α, ws.Ap, -one(T), nlive)
        _col_dot!(ws.rsnew, ws.r, ws.r, nlive)
        @. ws.rel = ifelse(ws.rhsnorm == 0, zero(T), sqrt(ws.rsnew) / ws.rhsnorm)
        worst = _max_prefix(ws.rel, nlive)
        verbose && @info "nusht_solve! iter $i: rel_res=$worst live=$nlive"
        worst < rtol && break
        @. ws.β = ifelse(ws.rsold == 0, zero(T), ws.rsnew / ws.rsold)
        _col_pbp!(ws.p, ws.r, ws.β, nlive)
        copyto!(ws.rsold, ws.rsnew)
        n2 = _retire_and_compact!(C, ws, rtol, nlive, len)
        if n2 != nlive
            nlive = n2
            nlive == 0 && break
            width = plan.pool_recipe.narrowable ? _fit_width(nlive, plan.B) : plan.B
        end
    end

    # Whatever is still live at exit has not been written out yet.
    @inbounds for s in 1:nlive
        _copy_col!(C, ws.perm[s], ws.x, s, len)
    end
    return C, iters, worst
end

# ─────────────────────────────────────────────────────────────────────────────
# Collections of independent problems, split across a backend
# ─────────────────────────────────────────────────────────────────────────────
# One plan per problem; `SerialBackend` loops here, `ThreadedBackend` in NUFSHTOhMyThreadsExt.

@inline function _check_farm(outs, ins, plans)
    @assert length(outs) == length(ins) == length(plans) "outs, ins and plans must have equal length"
    return nothing
end

"""
    nusht_type2!(fs, Cs, plans, backend = AutoBackend()) -> fs

Synthesize a collection of independent problems: for each `i`, `nusht_type2!(fs[i], Cs[i], plans[i])`.
`ThreadedBackend` requires `using OhMyThreads`.
"""
nusht_type2!(fs, Cs, plans::AbstractVector) = nusht_type2!(fs, Cs, plans, ComputationalBackends.AutoBackend())
nusht_type2!(fs, Cs, plans::AbstractVector, b::ComputationalBackends.AbstractAutoBackend) =
    nusht_type2!(fs, Cs, plans, _resolve_backend(b))
function nusht_type2!(fs, Cs, plans::AbstractVector, ::ComputationalBackends.AbstractSerialBackend)
    _check_farm(fs, Cs, plans)
    for i in eachindex(plans)
        nusht_type2!(fs[i], Cs[i], plans[i])
    end
    return fs
end

"""
    nusht_type1!(Cs, fs, plans, backend = AutoBackend()) -> Cs

Adjoint analysis over a collection of independent problems; see [`nusht_type2!`](@ref).
"""
nusht_type1!(Cs, fs, plans::AbstractVector) = nusht_type1!(Cs, fs, plans, ComputationalBackends.AutoBackend())
nusht_type1!(Cs, fs, plans::AbstractVector, b::ComputationalBackends.AbstractAutoBackend) =
    nusht_type1!(Cs, fs, plans, _resolve_backend(b))
function nusht_type1!(Cs, fs, plans::AbstractVector, ::ComputationalBackends.AbstractSerialBackend)
    _check_farm(Cs, fs, plans)
    for i in eachindex(plans)
        nusht_type1!(Cs[i], fs[i], plans[i])
    end
    return Cs
end

"""
    nusht_solve!(Cs, fs, plans, backend = AutoBackend(); kwargs...) -> Cs

Exact inversion over a collection of independent problems; `kwargs` go to the single-problem method.
"""
nusht_solve!(Cs, fs, plans::AbstractVector; kwargs...) = nusht_solve!(Cs, fs, plans, ComputationalBackends.AutoBackend(); kwargs...)
nusht_solve!(Cs, fs, plans::AbstractVector, b::ComputationalBackends.AbstractAutoBackend; kwargs...) =
    nusht_solve!(Cs, fs, plans, _resolve_backend(b); kwargs...)
function nusht_solve!(Cs, fs, plans::AbstractVector,
                      ::ComputationalBackends.AbstractSerialBackend; kwargs...)
    _check_farm(Cs, fs, plans)
    for i in eachindex(plans)
        nusht_solve!(Cs[i], fs[i], plans[i]; kwargs...)
    end
    return Cs
end

"""
    nusht_filter!(outs, ins, filter, plans, backend = AutoBackend()) -> outs

Spectral filtering over a collection of independent problems; see [`nusht_type2!`](@ref).
"""
nusht_filter!(outs, ins, filter, plans::AbstractVector) =
    nusht_filter!(outs, ins, filter, plans, ComputationalBackends.AutoBackend())
nusht_filter!(outs, ins, filter, plans::AbstractVector, b::ComputationalBackends.AbstractAutoBackend) =
    nusht_filter!(outs, ins, filter, plans, _resolve_backend(b))
function nusht_filter!(outs, ins, filter, plans::AbstractVector,
                       ::ComputationalBackends.AbstractSerialBackend)
    _check_farm(outs, ins, plans)
    for i in eachindex(plans)
        nusht_filter!(outs[i], ins[i], filter, plans[i])
    end
    return outs
end

"""
    nusht_type2_spin!(fs, sfs, plans, backend = AutoBackend()) -> fs

Spin-weighted synthesis over a collection of independent problems. This path touches no FastTransforms
state, so it carries none of the in-task hazard the scalar path works around.
"""
nusht_type2_spin!(fs, sfs, plans::AbstractVector) = nusht_type2_spin!(fs, sfs, plans, ComputationalBackends.AutoBackend())
nusht_type2_spin!(fs, sfs, plans::AbstractVector, b::ComputationalBackends.AbstractAutoBackend) =
    nusht_type2_spin!(fs, sfs, plans, _resolve_backend(b))
function nusht_type2_spin!(fs, sfs, plans::AbstractVector,
                           ::ComputationalBackends.AbstractSerialBackend)
    _check_farm(fs, sfs, plans)
    for i in eachindex(plans)
        nusht_type2_spin!(fs[i], sfs[i], plans[i])
    end
    return fs
end

"""
    nusht_type1_spin!(sfs, fs, plans, backend = AutoBackend()) -> sfs

Spin-weighted adjoint analysis over a collection; see [`nusht_type2_spin!`](@ref).
"""
nusht_type1_spin!(sfs, fs, plans::AbstractVector) = nusht_type1_spin!(sfs, fs, plans, ComputationalBackends.AutoBackend())
nusht_type1_spin!(sfs, fs, plans::AbstractVector, b::ComputationalBackends.AbstractAutoBackend) =
    nusht_type1_spin!(sfs, fs, plans, _resolve_backend(b))
function nusht_type1_spin!(sfs, fs, plans::AbstractVector,
                           ::ComputationalBackends.AbstractSerialBackend)
    _check_farm(sfs, fs, plans)
    for i in eachindex(plans)
        nusht_type1_spin!(sfs[i], fs[i], plans[i])
    end
    return sfs
end

"""
    nusht_solve_spin!(sfs, fs, plans, backend = AutoBackend(); kwargs...) -> sfs

Spin-weighted exact inversion over a collection; see [`nusht_type2_spin!`](@ref).
"""
nusht_solve_spin!(sfs, fs, plans::AbstractVector; kwargs...) =
    nusht_solve_spin!(sfs, fs, plans, ComputationalBackends.AutoBackend(); kwargs...)
nusht_solve_spin!(sfs, fs, plans::AbstractVector, b::ComputationalBackends.AbstractAutoBackend; kwargs...) =
    nusht_solve_spin!(sfs, fs, plans, _resolve_backend(b); kwargs...)
function nusht_solve_spin!(sfs, fs, plans::AbstractVector,
                           ::ComputationalBackends.AbstractSerialBackend; kwargs...)
    _check_farm(sfs, fs, plans)
    for i in eachindex(plans)
        nusht_solve_spin!(sfs[i], fs[i], plans[i]; kwargs...)
    end
    return sfs
end

# Any backend with no method above is one whose extension is not loaded — refuse, never downgrade.
nusht_type2!(fs, Cs, plans::AbstractVector, b::ComputationalBackends.AbstractExecutionBackend) =
    _backend_unavailable(b, "a collection of syntheses")
nusht_type1!(Cs, fs, plans::AbstractVector, b::ComputationalBackends.AbstractExecutionBackend) =
    _backend_unavailable(b, "a collection of analyses")
nusht_solve!(Cs, fs, plans::AbstractVector, b::ComputationalBackends.AbstractExecutionBackend; kwargs...) =
    _backend_unavailable(b, "a collection of solves")
nusht_filter!(outs, ins, filter, plans::AbstractVector, b::ComputationalBackends.AbstractExecutionBackend) =
    _backend_unavailable(b, "a collection of filters")
nusht_type2_spin!(fs, sfs, plans::AbstractVector, b::ComputationalBackends.AbstractExecutionBackend) =
    _backend_unavailable(b, "a collection of spin syntheses")
nusht_type1_spin!(sfs, fs, plans::AbstractVector, b::ComputationalBackends.AbstractExecutionBackend) =
    _backend_unavailable(b, "a collection of spin analyses")
nusht_solve_spin!(sfs, fs, plans::AbstractVector, b::ComputationalBackends.AbstractExecutionBackend; kwargs...) =
    _backend_unavailable(b, "a collection of spin solves")

# ── Node-set form: build the plans internally ─────────────────────────────────
# Takes node sets rather than plans, so it serves backends a plan cannot reach — see the header note.

"""
    nusht_type2(θs, φs, Cs, lmax, backend = AutoBackend(); tol, ntrans, tuning) -> fs

Synthesize `N` independent problems given their node sets: for each `i` a plan is built from
`(θs[i], φs[i])`, `Cs[i]` is evaluated, and the field is returned as `fs[i]`.

Use this instead of the plan-collection [`nusht_type2!`](@ref) when the backend is a
`DistributedBackend`. When you already hold plans and are on a local backend, prefer the in-place
form — it reuses them.
"""
nusht_type2(θs, φs, Cs, lmax; kwargs...) = nusht_type2(θs, φs, Cs, lmax, ComputationalBackends.AutoBackend(); kwargs...)
nusht_type2(θs, φs, Cs, lmax, b::ComputationalBackends.AbstractAutoBackend; kwargs...) =
    nusht_type2(θs, φs, Cs, lmax, _resolve_backend(b); kwargs...)

function nusht_type2(θs, φs, Cs, lmax, ::ComputationalBackends.AbstractLocalBackend; kwargs...)
    @assert length(θs) == length(φs) == length(Cs) "θs, φs and Cs must have equal length"
    return map(eachindex(θs)) do i
        plan = make_plan(θs[i], φs[i], lmax; kwargs...)
        try
            f = zeros(eltype(plan.F), length(θs[i]))
            nusht_type2!(f, Cs[i], plan)
            return f
        finally
            close!(plan)
        end
    end
end

nusht_type2(θs, φs, Cs, lmax, b::ComputationalBackends.AbstractExecutionBackend; kwargs...) =
    _backend_unavailable(b, "a node-set synthesis farm")

"""
    nusht_solve(θs, φs, fs, lmax, backend = AutoBackend(); tol, rtol, maxiter, …) -> Cs

Exact inversion of `N` independent problems given their node sets; see [`nusht_type2`](@ref).
"""
nusht_solve(θs, φs, fs, lmax; kwargs...) = nusht_solve(θs, φs, fs, lmax, ComputationalBackends.AutoBackend(); kwargs...)
nusht_solve(θs, φs, fs, lmax, b::ComputationalBackends.AbstractAutoBackend; kwargs...) =
    nusht_solve(θs, φs, fs, lmax, _resolve_backend(b); kwargs...)

function nusht_solve(θs, φs, fs, lmax, ::ComputationalBackends.AbstractLocalBackend;
                     rtol = 1e-6, maxiter = 500, kwargs...)
    @assert length(θs) == length(φs) == length(fs) "θs, φs and fs must have equal length"
    return map(eachindex(fs)) do i
        plan = make_plan(θs[i], φs[i], lmax; kwargs...)
        try
            C = zeros(eltype(plan.C), lmax + 1, 2lmax + 1)
            nusht_solve!(C, fs[i], plan; rtol = rtol, maxiter = maxiter)
            return C
        finally
            close!(plan)
        end
    end
end

nusht_solve(θs, φs, fs, lmax, b::ComputationalBackends.AbstractExecutionBackend; kwargs...) =
    _backend_unavailable(b, "a node-set solve farm")

# ── Decomposing a single transform ────────────────────────────────────────────
# A local backend on one transform is just the ordinary call; `MPIBackend` partitions the M points
# across ranks and is supplied by NUFSHTMPIExt.

nusht_type1!(C, f, plan::NUSHTplan, ::ComputationalBackends.AbstractLocalBackend) =
    nusht_type1!(C, f, plan)
nusht_type1!(C, f, plan::NUSHTplan, b::ComputationalBackends.AbstractAutoBackend) =
    nusht_type1!(C, f, plan, _resolve_backend(b))
nusht_type1!(C, f, plan::NUSHTplan, b::ComputationalBackends.AbstractExecutionBackend) =
    _backend_unavailable(b, "a point-decomposed adjoint")

nusht_solve!(C, f, plan::NUSHTplan, ::ComputationalBackends.AbstractLocalBackend; kwargs...) =
    nusht_solve!(C, f, plan; kwargs...)
nusht_solve!(C, f, plan::NUSHTplan, b::ComputationalBackends.AbstractAutoBackend; kwargs...) =
    nusht_solve!(C, f, plan, _resolve_backend(b); kwargs...)
nusht_solve!(C, f, plan::NUSHTplan, b::ComputationalBackends.AbstractExecutionBackend; kwargs...) =
    _backend_unavailable(b, "a point-decomposed solve")

end # module NUFSHT
