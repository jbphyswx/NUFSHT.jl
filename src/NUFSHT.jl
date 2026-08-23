"""
    NUFSHT.jl — Non-Uniform Fast Spherical Harmonic Transform (native Julia)

Double Fourier Sphere (DFS) + nuFFT spherical harmonic transforms at arbitrary scattered
`(colatitude, longitude)` points. The synthesis operator factors as `A = N·F·S`:

- **S** (`plan_sph2fourier`): the Legendre step, taking SH coefficients to the Double-Fourier-Sphere
  bivariate Fourier series. No equiangular grid is formed.
- **F** (`_assemble_modes!`): the cos/sin bivariate Fourier basis `S` produces, rewritten as the
  complex exponentials a NUFFT evaluates. Both mode axes carry wavenumbers `-lmax…lmax`.
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

include("Modes.jl")
include("NUFFT.jl")
include("Plan.jl")
include("Kernels.jl")

export make_plan, NUSHTplan, close!, LSMRWorkspace
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

@inline _npts(plan::AbstractNUSHTplan) = length(_θnodes(plan))
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

# Accumulating forms of the pair above: `f ← f + Re(fbuf)` / `f ← f + fbuf`. LSMR's `u ← A v − α u`
# scales `u` first and then adds the synthesis, so it needs no second point-space buffer.
function _add_real!(f, fbuf, n::Integer = size(f, ndims(f)))
    len = _colstride(f)
    @inbounds for k in 1:n, i in ((k - 1) * len + 1):(k * len)
        f[i] += real(fbuf[i])
    end
    return f
end
function _add_field!(f, fbuf, n::Integer = size(f, ndims(f)))
    len = _colstride(f)
    @inbounds for k in 1:n, i in ((k - 1) * len + 1):(k * len)
        f[i] += fbuf[i]
    end
    return f
end

# The NUFFT always returns complex strengths; a real field drops the imaginary residue.
_copy_out!(f, fbuf, ::AbstractNUSHTplan) = _copy_field!(f, fbuf)
_copy_out!(f, fbuf, ::NUSHTplan{T,FE}) where {T,FE<:Real} = _copy_real!(f, fbuf)
_copy_out!(f, fbuf, ::NUSHTplan{T,FE}) where {T,FE<:Complex} = _copy_field!(f, fbuf)
_add_out!(f, fbuf, ::NUSHTplan{T,FE}, n) where {T,FE<:Real} = _add_real!(f, fbuf, n)
_add_out!(f, fbuf, ::NUSHTplan{T,FE}, n) where {T,FE<:Complex} = _add_field!(f, fbuf, n)

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
            op!(plan.Fslice, plan.sph_plan, plan.sph_plan_adj, b)
        end
        return plan
    end
    _with_fasttransforms_single() do
        @sync for c in 1:nt
            Threads.@spawn begin
                sl, P, Padj = pool[c]
                for b in c:nt:k
                    op!(sl, P, Padj, b)
                end
            end
        end
    end
    return plan
end

# S (forward): `plan_sph2fourier` alone, per batch slice through a dense slice buffer. Its output IS
# the DFS bivariate Fourier series, which `_assemble_modes!` hands straight to the NUFFT — evaluating
# it onto the equiangular grid (`plan_sph_synthesis`) only to double it and transform back would be a
# round trip, and the doubling step is not even exact for the odd `Nφ = 2lmax+1` this package uses.
function _sph_evaluate!(plan::NUSHTplan, k::Integer = plan.B)
    _sph_columns!(plan, k) do sl, P, _Padj, b
        _load_slice!(sl, plan.F, plan, b)
        LinearAlgebra.lmul!(P, sl)
        _store_slice!(plan.F, sl, plan, b)
    end
    return plan
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
    k == plan.B && return (k = Int(k), nufft_type2 = _nufft2(plan), nufft_type1 = _nufft1(plan))
    @inbounds for e in plan.size_pool
        e.k == k && return e
    end
    return _build_width!(plan, k)
end

# F·N (forward): bivariate Fourier coefficients → complex mode array → type-2 NUFFT into `fbuf`.
# No doubling and no FFT: `_assemble_modes!` produces the modes the NUFFT evaluates directly.
function _dfn_synthesis!(plan::NUSHTplan, k::Integer = plan.B)
    e = _width_plans(plan, k)
    if k == plan.B
        _assemble_modes!(plan.Fhat, plan.F, plan.lmax)
        _nufft_exec!(e.nufft_type2, plan.Fhat, _fbuf(plan))
    else
        _assemble_modes!(_pfx(plan.Fhat, k), _pfx(plan.F, k), plan.lmax)
        _nufft_exec!(e.nufft_type2, _pfx(plan.Fhat, k), _pfx(_fbuf(plan), k))
    end
    return plan
end

# N†·F† (adjoint): type-1 NUFFT from `fbuf` → mode array → the exact transpose of the assembly.
function _dfn_analysis!(plan::NUSHTplan, k::Integer = plan.B)
    e = _width_plans(plan, k)
    if k == plan.B
        _nufft_exec!(e.nufft_type1, _fbuf(plan), plan.Fhat)
        _assemble_modes_adjoint!(plan.F, plan.Fhat, plan.lmax)
    else
        _nufft_exec!(e.nufft_type1, _pfx(_fbuf(plan), k), _pfx(plan.Fhat, k))
        _assemble_modes_adjoint!(_pfx(plan.F, k), _pfx(plan.Fhat, k), plan.lmax)
    end
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

Algorithm `A = N·F·S`: `plan_sph2fourier` to the bivariate Fourier series (S) → assemble the complex
mode array (F) → FINUFFT type 2 (N).
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

**Type 1 (adjoint):** given field values `f` at the `M` scattered points, apply `A†` and write the
result to `C`. This is the exact Euclidean adjoint of [`nusht_type2!`](@ref) — the transpose, not an
inverse — so `A†A` is symmetric positive definite and usable by an iterative solver. Use
[`nusht_solve!`](@ref) to invert.

For exact analysis of a field already sampled on the Clenshaw-Curtis grid, use
`FastSphericalHarmonics.sph_transform`: that is quadrature, needs no scattered-point machinery, and
is a different operation from the adjoint.
"""
nusht_type1!(C, f, plan::NUSHTplan) = _nusht_true_adjoint!(C, f, plan)

"""
    _nusht_true_adjoint!(C, f, plan, k = plan.B, kdfn = k)

The exact Euclidean adjoint of `nusht_type2!`: type-1 NUFFT, the transpose of the mode assembly, then
`P'` per column. `k` bounds the columns the sphere loop visits and `kdfn` the NUFFT plan width.
"""
function _nusht_true_adjoint!(C, f, plan::NUSHTplan{T}, k::Integer = plan.B,
                             kdfn::Integer = k) where {T}
    _assert_field(f, plan)
    _assert_coeffs(C, plan)
    _copy_field!(_fbuf(plan), f)
    _dfn_analysis!(plan, kdfn)
    _sph_columns!(plan, k) do sl, _P, Padj, b
        _load_slice!(sl, plan.F, plan, b)
        LinearAlgebra.lmul!(Padj, sl)
        _store_slice!(C, sl, plan, b)
    end
    return C
end

# ─────────────────────────────────────────────────────────────────────────────
# Filtering
# ─────────────────────────────────────────────────────────────────────────────
# `nusht_filter!` and `nusht_filter_renorm!` need `LSMRWorkspace` in a signature, so they live below
# the solver; the two that do not are here.

"""
    nusht_synthesize!(f_out, C, filter, plan) -> f_out

Scale coefficients `C` by `filter`'s transfer function and synthesize at the plan's points. `C` is not
modified — the scaling runs in the plan's own scratch — so one analysis feeds any number of filters:

```julia
nusht_solve!(C, f, plan; ws = ws)
for (out, filt) in zip(outs, filters)
    nusht_synthesize!(out, C, filt, plan)
end
```

[`nusht_filter!`](@ref) is a solve followed by this, and so re-fits per filter; use the pair when
filtering one field at several scales, since the fit is the expensive half.
"""
function nusht_synthesize!(f_out, C, filter, plan::NUSHTplan)
    _assert_coeffs(C, plan)
    C === plan.C || copyto!(plan.C, C)
    apply_transfer!(plan.C, filter, plan.lmax)
    nusht_type2!(f_out, plan.C, plan)
    return f_out
end

# ─────────────────────────────────────────────────────────────────────────────
# Exact inversion: batched LSMR on min ‖A c − f‖
# ─────────────────────────────────────────────────────────────────────────────
# LSMR (Fong & Saunders, SISC 33(5), 2011) applies Golub–Kahan bidiagonalization to `A` itself rather
# than solving the normal equations, so it works at cond(A) where CG on `A†A` works at cond(A)² — for
# the same one `A` and one `A†` per iteration, and the same buffer count. Measured on this operator:
# identical iteration counts to a given tolerance, 25-250x better coefficient accuracy, and `‖A†r‖`
# (the quantity this package reports and stops on) monotone by construction where CG's rises to 7.9x
# its running best on a square point set and diverges outright on a rank-deficient one.

# Host view of a per-column scalar vector. Retirement and the bidiagonalization's scalar recurrences
# are host control flow, so the values they read cannot be scalar-indexed out of a device array. On a
# host plan the mirror IS the vector, so `_mirror!` copies nothing.
_host_mirror(v::Array) = v
_host_mirror(v::AbstractVector) = Array(v)
@inline _mirror!(h, v) = h === v ? h : copyto!(h, v)

"""
    LSMRWorkspace(plan)

Reusable scratch for [`nusht_solve!`](@ref). Holding one across solves makes the solve
allocation-free. Per-column scalars are length-`B` vectors so the `B` columns run as `B` independent
single-column solves (batched result == looping single transforms).

After a solve, `colres[b]` is the relative residual `‖A†r‖/‖A†f‖` of the coefficients delivered for
column `b`, in the caller's column order — the per-column form of the scalar `rel_res` returned.

Only `nrm` and `cf` live on the plan's device: everything else the bidiagonalization tracks is scalar
recurrence, which is cheaper and clearer on the host than as a chain of length-`B` kernel launches.
"""
struct LSMRWorkspace{AT3<:AbstractArray, FT2<:AbstractMatrix, VT<:AbstractVector,
                     HV<:AbstractVector, VM, PV<:AbstractVector{<:Integer}}
    x::AT3                # iterate
    v::AT3                # right bidiagonalization vector
    h::AT3
    hbar::AT3
    w::AT3                # A†u, before it is folded into v
    u::FT2                # left bidiagonalization vector, in point space
    # `1` on the (lmax+1)^2 slots holding degrees l ≤ lmax, `0` on the supernumerary ones. The array is
    # a square, invertible representation carrying degrees up to `lmax+|m|`, so the transform needs all
    # of it — but a least-squares fit must name its space, and only `l ≤ lmax` is SO(3)-invariant.
    # Fitting the ragged set instead makes the answer depend on the coordinate frame.
    valid::VM
    # Slot -> original column. Compaction moves live columns to the front, so a slot's identity is
    # only recoverable through this; it is what keeps a result from being written to the wrong column.
    perm::PV
    nrm::VT               # device scratch: per-column ⟨·,·⟩
    cf::VT                # device scratch: the per-column coefficient of the moment
    nrm_h::HV
    cf_h::HV
    α::HV
    β::HV
    αbar::HV
    ζbar::HV              # |ζbar| is LSMR's ‖A†r‖, available with no extra transform
    ρ::HV
    ρbar::HV
    cbar::HV
    sbar::HV
    normA2::HV            # Σ(α² + β²): ‖A‖_F², which bounds ‖A‖
    maxrbar::HV
    minrbar::HV           # extreme ρbar, giving LSMR's cond(A) estimate
    atb::HV               # ‖A†f‖ per column, the residual's denominator
    rel::HV
    done::HV
    colres::HV
end

# `valid` is `nothing` for a spin plan: that coefficient array is dense, with no supernumerary slots
# to project away.
@inline _lsmr_project!(w, ::Nothing) = w
@inline _lsmr_project!(w, valid) = (w .*= valid; w)

@inline _coefflen(plan::NUSHTplan) = plan.Nθ * plan.Nφ

# `k` is the live column count the per-column sphere loop narrows to; `kdfn` the NUFFT plan width,
# which can only narrow where the backend's plans are width-polymorphic.
@inline _lsmr_widths(plan::NUSHTplan, nlive::Integer) =
    (nlive, plan.pool_recipe.narrowable ? _fit_width(max(nlive, 1), plan.B) : plan.B)

@inline _assert_solve(C, f, plan::NUSHTplan) = (_assert_coeffs(C, plan); _assert_field(f, plan))

function LSMRWorkspace(plan::NUSHTplan{T,FE}) where {T,FE}
    Nθ, Nφ, B = plan.Nθ, plan.Nφ, plan.B
    M = _npts(plan)
    z3() = _zeros_like(plan.F, FE, Nθ, Nφ, B)
    vB() = _zeros_like(_θnodes(plan), T, B)
    hB() = zeros(T, B)
    nrm, cf = vB(), vB()
    return LSMRWorkspace(z3(), z3(), z3(), z3(), z3(), _zeros_like(_fbuf(plan), FE, M, B),
                         _valid_mask(plan.F, T, plan.lmax), collect(1:B),
                         nrm, cf, _host_mirror(nrm), _host_mirror(cf),
                         hB(), hB(), hB(), hB(), hB(), hB(), hB(), hB(),
                         hB(), hB(), hB(), hB(), hB(), hB(), hB())
end

# Per-column primitives address a batch buffer by its linear column stride, so one method serves a
# point-space `(M, B)` buffer and a coefficient-space `(Nθ, Nφ, B)` one alike. `n` bounds the columns
# visited, so a compacted solve touches only its live prefix; taken as a count rather than a view,
# since a `SubArray` per call would put allocation back on the hot path.
@inline _colstride(A) = length(A) ÷ size(A, ndims(A))

# dst[k] = Re Σ conj(a)·b over column k. For real arrays `conj` and `real` are identities, so this is
# the plain dot product and the real and complex paths need only this one method.
function _col_hdot!(dst, a, b, n::Integer = size(a, ndims(a)))
    len = _colstride(a)
    @inbounds for k in 1:n
        s = zero(real(eltype(a)))
        o = (k - 1) * len
        @simd for i in 1:len
            s += real(conj(a[o + i]) * b[o + i])
        end
        dst[k] = s
    end
    return dst
end

function _col_axpy!(y, α, x, σ, n::Integer = size(y, ndims(y)))   # y[:,k] += σ·α[k]·x[:,k]
    len = _colstride(y)
    @inbounds for k in 1:n
        c = σ * α[k]
        o = (k - 1) * len
        @simd for i in 1:len
            y[o + i] += c * x[o + i]
        end
    end
    return y
end

function _col_pbp!(p, r, β, n::Integer = size(p, ndims(p)))       # p[:,k] = r[:,k] + β[k]·p[:,k]
    len = _colstride(p)
    @inbounds for k in 1:n
        c = β[k]
        o = (k - 1) * len
        @simd for i in 1:len
            p[o + i] = r[o + i] + c * p[o + i]
        end
    end
    return p
end

function _col_scale!(y, s, n::Integer = size(y, ndims(y)))        # y[:,k] *= s[k]
    len = _colstride(y)
    @inbounds for k in 1:n
        c = s[k]
        o = (k - 1) * len
        @simd for i in 1:len
            y[o + i] *= c
        end
    end
    return y
end

# Largest of the first `n` entries of a host vector, without a view (which would allocate).
@inline function _max_prefix(v, n::Integer)
    m = zero(eltype(v))
    @inbounds for i in 1:n
        v[i] > m && (m = v[i])
    end
    return m
end

# Working width for `n` live columns: the next power of two, capped at `B`. Computed rather than looked
# up in a ladder, so it allocates nothing on the hot path — `_build_width!` builds any width on demand,
# so the powers of two exist only to bound how many distinct plan sets a solve can create.
@inline _fit_width(n::Integer, B::Integer) = min(Int(B), Int(nextpow(2, max(n, 1))))

# Column `ssl` of `src` → column `dsl` of `dst`, addressed by linear offset so it works on a 2-D or 3-D
# batch buffer alike. `copyto!` rather than an elementwise loop: it is a `memmove` on a host array and a
# device-to-device copy on a GPU one, where the loop would be a scalar-indexing error.
@inline _copy_col!(dst, dsl::Integer, src, ssl::Integer, len::Integer) =
    copyto!(dst, (dsl - 1) * len + 1, src, (ssl - 1) * len + 1, len)

# Load `cf_h[1:n]` onto the device coefficient vector. On a host plan the two alias, so filling `cf_h`
# already filled `cf` and this copies nothing.
@inline _push_cf!(ws::LSMRWorkspace) = _mirror!(ws.cf, ws.cf_h)

"""
    _retire_and_compact!(C, ws, rtol, nlive, len, mlen) -> nlive′

Write out every live slot that is finished — converged (`rel < rtol`) or stopped by LSMR's own
condition/exact-termination tests — into `C` at its *original* column, then close the gaps so the
survivors occupy slots `1:nlive′`. Every per-slot quantity that survives an iteration moves with its
slot: the four coefficient buffers, the point-space `u`, and the bidiagonalization scalars.
"""
function _retire_and_compact!(C, ws::LSMRWorkspace, rtol, nlive::Integer, len::Integer, mlen::Integer)
    w = 0
    @inbounds for s in 1:nlive
        if ws.done[s] != 0
            _copy_col!(C, ws.perm[s], ws.x, s, len)      # final answer, to its own column
            ws.colres[ws.perm[s]] = ws.rel[s]
            continue
        end
        w += 1
        if w != s
            _copy_col!(ws.x, w, ws.x, s, len)
            _copy_col!(ws.v, w, ws.v, s, len)
            _copy_col!(ws.h, w, ws.h, s, len)
            _copy_col!(ws.hbar, w, ws.hbar, s, len)
            _copy_col!(ws.u, w, ws.u, s, mlen)
            for f in (ws.α, ws.β, ws.αbar, ws.ζbar, ws.ρ, ws.ρbar, ws.cbar, ws.sbar,
                      ws.normA2, ws.maxrbar, ws.minrbar, ws.atb, ws.rel)
                f[w] = f[s]
            end
            # Swap rather than assign: retired originals stay in the tail, keeping `perm` a bijection
            # over 1:B. Assigning would leave duplicates, and two slots sharing a destination.
            ws.perm[w], ws.perm[s] = ws.perm[s], ws.perm[w]
        end
    end
    return w
end

# One `A` and one `A†`, at the live width. `k` is the live column count the per-column sphere loop
# narrows to; `kdfn` is the NUFFT plan width, which can only narrow where the backend's plans are
# width-polymorphic (`_width_polymorphic`) and otherwise stays at `B`.
#
# `A v` lands in the plan's own strengths buffer, so `u ← A v − α u` needs no second point-space array:
# scale `u` first, then accumulate the synthesis onto it.
function _lsmr_Av_axpy!(ws::LSMRWorkspace, plan::NUSHTplan, k::Integer, kdfn::Integer, n::Integer)
    _col_scale!(ws.u, ws.cf, n)
    copyto!(plan.F, ws.v)
    _sph_evaluate!(plan, k)
    _dfn_synthesis!(plan, kdfn)
    return _add_out!(ws.u, _fbuf(plan), plan, n)
end

# `P A† u` — the projection is what restricts the fit to the SO(3)-invariant `l ≤ lmax` subspace.
function _lsmr_Atu!(ws::LSMRWorkspace, plan::NUSHTplan, k::Integer, kdfn::Integer)
    _nusht_true_adjoint!(ws.w, ws.u, plan, k, kdfn)
    return _lsmr_project!(ws.w, ws.valid)
end

"""
    nusht_solve!(C, f, plan; ws=LSMRWorkspace(plan), maxiter=500, rtol=1e-6, conlim=0, verbose=false)

**Exact inversion:** solve `min ‖A c − f‖` for coefficients `C` by LSMR on the Golub–Kahan
bidiagonalization of `A`. Batched (`B > 1`) runs the columns as independent single-column solves.

Returns `(C, iters, rel_res, converged)` with `rel_res = max_k ‖A†r_k‖/‖A†f_k‖` and
`converged = rel_res < rtol`; `ws.colres` carries the same residual per column. `rel_res` is LSMR's
own recurrence value for `‖A†r‖`, floored at `eps(T)` since a relative residual is not resolvable
below that — so an `rtol` under machine precision never reports convergence.

A column also stops when LSMR's condition estimate exceeds `conlim` (default `1/eps(T)`), or when the
bidiagonalization terminates exactly. That matters when the points do not determine the coefficients —
`M` below `(lmax+1)²`, or clustered so that they effectively do not — where `A` is rank deficient and
the iteration has nothing left to resolve. `converged == false` is the signal that the point set, not
the budget, was the limit.
"""
nusht_solve!(C, f, plan::NUSHTplan; ws::LSMRWorkspace = LSMRWorkspace(plan), kwargs...) =
    _lsmr!(C, f, plan, ws; kwargs...)

# The solve itself, shared by the scalar and spin paths: they differ only in `_lsmr_Av_axpy!`,
# `_lsmr_Atu!`, `_lsmr_widths` and `_coefflen`, all of which dispatch on the plan.
function _lsmr!(
    C, f, plan::AbstractNUSHTplan, ws::LSMRWorkspace;
    maxiter::Int = 500,
    rtol::Real = 1e-6,
    conlim::Real = 0,
    verbose::Bool = false,
)
    T = real(eltype(ws.x))
    FE = eltype(ws.x)
    _assert_solve(C, f, plan)
    B = plan.B
    len = _coefflen(plan)
    mlen = _npts(plan)
    clim = conlim > 0 ? T(conlim) : one(T) / eps(T)

    # β₁u₁ = f ; α₁v₁ = P A†u₁
    _copy_field!(ws.u, f)
    _col_hdot!(ws.nrm, ws.u, ws.u, B)
    _mirror!(ws.nrm_h, ws.nrm)
    @inbounds for k in 1:B
        ws.β[k] = sqrt(ws.nrm_h[k])
        ws.cf_h[k] = ws.β[k] > 0 ? inv(ws.β[k]) : zero(T)
    end
    _push_cf!(ws)
    _col_scale!(ws.u, ws.cf, B)

    kB, kdfnB = _lsmr_widths(plan, B)
    _lsmr_Atu!(ws, plan, kB, kdfnB)
    copyto!(ws.v, ws.w)
    _col_hdot!(ws.nrm, ws.v, ws.v, B)
    _mirror!(ws.nrm_h, ws.nrm)
    @inbounds for k in 1:B
        ws.α[k] = sqrt(ws.nrm_h[k])
        ws.cf_h[k] = ws.α[k] > 0 ? inv(ws.α[k]) : zero(T)
    end
    _push_cf!(ws)
    _col_scale!(ws.v, ws.cf, B)

    fill!(ws.x, zero(FE))
    fill!(ws.hbar, zero(FE))
    copyto!(ws.h, ws.v)
    @inbounds for k in 1:B
        ws.αbar[k] = ws.α[k]
        ws.ζbar[k] = ws.α[k] * ws.β[k]
        ws.atb[k] = ws.α[k] * ws.β[k]
        ws.ρ[k] = one(T); ws.ρbar[k] = one(T); ws.cbar[k] = one(T); ws.sbar[k] = zero(T)
        ws.normA2[k] = ws.α[k]^2
        ws.maxrbar[k] = zero(T); ws.minrbar[k] = T(Inf)
        ws.rel[k] = ws.atb[k] > 0 ? one(T) : zero(T)
        ws.done[k] = ws.atb[k] > 0 ? zero(T) : one(T)   # A†f = 0 ⟹ c = 0 already solves it
        ws.colres[k] = ws.rel[k]
        ws.perm[k] = k
    end

    nlive = B
    iters = 0
    # Columns that were already finished at setup never enter the loop; retire them first.
    nlive = _retire_and_compact!(C, ws, rtol, nlive, len, mlen)
    for i in 1:maxiter
        nlive == 0 && break
        iters = i

        # Bidiagonalization: u ← A v − α u, β = ‖u‖, u /= β
        @inbounds for k in 1:nlive
            ws.cf_h[k] = -ws.α[k]
        end
        k, kdfn = _lsmr_widths(plan, nlive)
        _push_cf!(ws)
        _lsmr_Av_axpy!(ws, plan, k, kdfn, nlive)
        _col_hdot!(ws.nrm, ws.u, ws.u, nlive)
        _mirror!(ws.nrm_h, ws.nrm)
        @inbounds for k in 1:nlive
            ws.β[k] = sqrt(max(ws.nrm_h[k], zero(T)))
            ws.cf_h[k] = ws.β[k] > 0 ? inv(ws.β[k]) : zero(T)
        end
        _push_cf!(ws)
        _col_scale!(ws.u, ws.cf, nlive)

        # v ← P A†u − β v, α = ‖v‖, v /= α
        @inbounds for k in 1:nlive
            ws.cf_h[k] = -ws.β[k]
        end
        _push_cf!(ws)
        _lsmr_Atu!(ws, plan, k, kdfn)
        _col_pbp!(ws.v, ws.w, ws.cf, nlive)
        _col_hdot!(ws.nrm, ws.v, ws.v, nlive)
        _mirror!(ws.nrm_h, ws.nrm)
        @inbounds for k in 1:nlive
            ws.α[k] = sqrt(max(ws.nrm_h[k], zero(T)))
            ws.cf_h[k] = ws.α[k] > 0 ? inv(ws.α[k]) : zero(T)
        end
        _push_cf!(ws)
        _col_scale!(ws.v, ws.cf, nlive)

        # Plane rotations (scalar per column) and the update coefficients they produce.
        @inbounds for k in 1:nlive
            ρold = ws.ρ[k]
            ρbarold = ws.ρbar[k]
            r = hypot(ws.αbar[k], ws.β[k])
            c = r > 0 ? ws.αbar[k] / r : one(T)
            s = r > 0 ? ws.β[k] / r : zero(T)
            θnew = s * ws.α[k]
            ws.αbar[k] = c * ws.α[k]
            ws.ρ[k] = r

            θbar = ws.sbar[k] * r
            ρtemp = ws.cbar[k] * r
            rb = hypot(ρtemp, θnew)
            ws.cbar[k] = rb > 0 ? ρtemp / rb : one(T)
            ws.sbar[k] = rb > 0 ? θnew / rb : zero(T)
            ws.ρbar[k] = rb
            ζ = ws.cbar[k] * ws.ζbar[k]
            ws.ζbar[k] = -ws.sbar[k] * ws.ζbar[k]

            # ‖A‖ and cond(A) estimates from the bidiagonal entries (Fong & Saunders §5.2).
            ws.normA2[k] += ws.β[k]^2
            ws.maxrbar[k] = max(ws.maxrbar[k], ρbarold)
            i > 1 && (ws.minrbar[k] = min(ws.minrbar[k], ρbarold))
            condA = max(ws.maxrbar[k], ρtemp) / max(min(ws.minrbar[k], ρtemp), eps(T))
            ws.normA2[k] += ws.α[k]^2

            # Reported floored at `eps(T)`: a relative residual is not resolvable below it, and `ζbar`
            # — a running product of factors `|s̄| ≤ 1` — underflows to exactly zero long before the
            # true residual does, which would otherwise report convergence at any `rtol` whatsoever.
            raw = ws.atb[k] > 0 ? abs(ws.ζbar[k]) / ws.atb[k] : zero(T)
            ws.rel[k] = ws.atb[k] > 0 ? max(raw, eps(T)) : zero(T)
            # Reaching that floor is itself a stopping condition: there is nothing further to extract,
            # whatever `rtol` asked for. Without it an `rtol` below `eps(T)` runs the whole budget.
            degenerate = !(r > 0) || !(rb > 0) || ws.α[k] == 0 || ws.β[k] == 0 || raw <= eps(T)
            ws.done[k] = (ws.rel[k] < rtol || condA >= clim || degenerate) ? one(T) : zero(T)

            # hbar and x update coefficients; h's is θnew/ρ = β·α/ρ², recoverable from stored state.
            ws.cf_h[k] = ρold * ρbarold > 0 ? -(θbar * r / (ρold * ρbarold)) : zero(T)
            ws.nrm_h[k] = r * rb > 0 ? ζ / (r * rb) : zero(T)
        end

        # hbar ← h − c₁·hbar ; x ← x + c₂·hbar ; h ← v − c₃·h
        _push_cf!(ws)
        _col_pbp!(ws.hbar, ws.h, ws.cf, nlive)
        @inbounds for k in 1:nlive
            ws.cf_h[k] = ws.nrm_h[k]
        end
        _push_cf!(ws)
        _col_axpy!(ws.x, ws.cf, ws.hbar, one(FE), nlive)
        @inbounds for k in 1:nlive
            ws.cf_h[k] = ws.ρ[k] > 0 ? -(ws.β[k] * ws.α[k] / ws.ρ[k]^2) : zero(T)
        end
        _push_cf!(ws)
        _col_pbp!(ws.h, ws.v, ws.cf, nlive)

        verbose && @info "nusht_solve! iter $i: rel_res=$(_max_prefix(ws.rel, nlive)) live=$nlive"
        nlive = _retire_and_compact!(C, ws, rtol, nlive, len, mlen)
    end

    # Whatever is still live at exit ran out of budget rather than finishing.
    @inbounds for s in 1:nlive
        _copy_col!(C, ws.perm[s], ws.x, s, len)
        ws.colres[ws.perm[s]] = ws.rel[s]
    end
    worst = maximum(ws.colres)
    return C, iters, worst, worst < rtol
end

# ─────────────────────────────────────────────────────────────────────────────
# Filtering that fits coefficients first (see the note above `nusht_synthesize!`)
# ─────────────────────────────────────────────────────────────────────────────

"""
    nusht_filter!(f_out, f_in, filter, plan; ws=LSMRWorkspace(plan), kwargs...)

Apply a spectral filter to `f_in` at the scattered points, writing to `f_out` (both length-`M` /
`(M, B)`): fit coefficients with [`nusht_solve!`](@ref) → `apply_transfer!` (× H(ℓ)) → `nusht_type2!`.
`kwargs` (`rtol`, `maxiter`, …) go to the solve. Uses the plan's coefficient scratch, and is
allocation-free when a `ws` is supplied.

The fit is what makes this a filter. `A H A†` — the adjoint in place of a fit — is a smoothing
operator, not `A H A⁺`: at scattered points `A†` is the transpose of the synthesis, not its inverse,
and only on a quadrature grid do the two coincide. So filtering scattered data is iterative; hold a
`ws` and reuse it across calls.
"""
function nusht_filter!(f_out, f_in, filter, plan::NUSHTplan;
                       ws::LSMRWorkspace = LSMRWorkspace(plan), kwargs...)
    nusht_solve!(plan.C, f_in, plan; ws = ws, kwargs...)
    nusht_synthesize!(f_out, plan.C, filter, plan)
    return f_out
end

"""
    nusht_filter_renorm!(f_out, mask, filter, plan; mask_filt=similar(f_out), ws, C_mask=nothing)

Renormalise the output of `nusht_filter!` to correct for land/ocean masking: divide by the
filtered mask (the fraction of kernel weight over ocean). `f_out` must have been produced by
`nusht_filter!(f_out, f .* mask, filter, plan)`. Points where the filtered mask is below `0.01`
are set to 0. Pass a reusable `mask_filt` scratch (shaped like `f_out`) to run allocation-free.
"""
function nusht_filter_renorm!(f_out, mask, filter, plan::NUSHTplan{T};
                              mask_filt = similar(f_out),
                              ws::LSMRWorkspace = LSMRWorkspace(plan),
                              C_mask = nothing) where {T}
    # `C_mask` lets a multi-scale caller fit the scale-independent mask once and pass its coefficients
    # to every call; without it the mask is fitted here, which is the expensive half.
    if C_mask === nothing
        nusht_solve!(plan.C, mask, plan; ws = ws)
        nusht_synthesize!(mask_filt, plan.C, filter, plan)
    else
        nusht_synthesize!(mask_filt, C_mask, filter, plan)
    end
    threshold = T(0.01)
    # `mask_filt` is shaped like `f_out` → a single fused broadcast, zero-alloc + device-safe.
    f_out .= ifelse.(abs.(mask_filt) .>= threshold, f_out ./ mask_filt, zero(T))
    return f_out
end

include("Spin.jl")

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
