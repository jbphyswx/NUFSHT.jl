"""
    NUFSHT.jl — Non-Uniform Fast Spherical Harmonic Transform (native Julia)

Double Fourier Sphere (DFS) + nuFFT spherical harmonic transforms at arbitrary scattered
`(colatitude, longitude)` points. The synthesis operator factors as `A = N·F·S`:

- **S** (`plan_sph2fourier`): the Legendre step, taking SH coefficients to the Double-Fourier-Sphere
  bivariate Fourier series. No equiangular grid is formed.
- **F** (`_assemble_modes!`): the cos/sin bivariate Fourier basis `S` produces, rewritten as the
  complex exponentials a NUFFT evaluates. The φ axis carries wavenumbers `-lmax…lmax`, the θ axis
  `-(lmax+1)…lmax+1` — one further out, because the coefficient array's supernumerary slots hold
  degrees up to `lmax+|m|` and an odd-order column reaches θ-frequency `lmax+1`.
- **N** (NUFFT type 2 / type 1): non-uniform FFT evaluating the 2D Fourier series at the scattered
  points. A real-eltype plan on a backend with a real-data transform stores only `kθ ≥ 0` and lets
  that backend supply the Hermitian half, halving both the mode array and the spreading.

A [`NUSHTplan`](@ref) owns persistent NUFFT plans (built once over the nodes) and every work buffer, so
repeated transforms — filtering, or the hundreds of matvecs in [`nusht_solve!`](@ref) — allocate
nothing and never re-plan. All calls transform a batch of `B = plan.B` co-located fields (`ntrans`);
`B = 1` methods accept plain vectors/matrices. The NUFFT is FlowTransformBindings' (NonuniformFFTs or
FINUFFT), or the in-core direct sum.

## References
- Merilees (1973); Townsend & Olver (2015); Reinecke & Seljebotn (2013, A&A 554 A112);
  Keiner, Kunis & Potts (2009); Belkner et al. (2024, arXiv:2406.14542).
- FastSphericalHarmonics.jl, FastTransforms.jl, FINUFFT.jl, NonuniformFFTs.jl.
"""
module NUFSHT

# First, so its `__init__` selects the OpenMP runtime's thread-local mode before FastTransforms loads
# that runtime; see `FlowTransformBindings.with_fasttransforms_threads`.
using FlowTransformBindings: FlowTransformBindings, FlowTransformBindings as FTB
using ComputationalBackends: ComputationalBackends
using FFTW: FFTW
using FastSphericalHarmonics: FastSphericalHarmonics
using LinearAlgebra: LinearAlgebra

public FlowTransformBindings, FTB
public ComputationalBackends

include("Modes.jl")
include("NUFFT.jl")
include("Plan.jl")
include("Kernels.jl")

export make_plan, NUSHTplan, close!, plan_memory
export nusht_type1!, nusht_type2!, nusht_synthesize!, nusht_filter!, nusht_filter_renorm!, nusht_solve!
export nusht_analyze!
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

# ── Parallel execution ────────────────────────────────────────────────────────
# An execution backend is a `ComputationalBackends.AbstractExecutionBackend`, in three roles:
#
#  • **Farm over independent problems** (collection methods below), the `execution` keyword. A NUFFT
#    plan is not safe to `exec!` concurrently, so each problem carries its own. Under
#    `DistributedBackend` the node-set form builds each problem's plan on its worker, since a plan
#    holding C pointers cannot be serialized.
#  • **One transform** — `make_plan(FE, θ, φ, lmax; execution)`: a thread count, a device, or the points
#    divided among `DistributedBackend` workers or `MPIBackend` ranks.
#  • **One process's share of a transform's points** — the fourth positional argument of
#    `nusht_type1!`, `nusht_analyze!`, `nusht_solve!` and `nusht_solve_spin!`: the adjoint, the analysis
#    or the fit over the points of every process the backend names, `plan` and `f` being this process's.
#
# An entry hands the backend to an internal function that dispatches on it; the extensions add their
# methods to those internal functions.

@inline _omt_loaded() = Base.get_extension(@__MODULE__, :NUFSHTOhMyThreadsExt) !== nothing

"""
    _resolve_execution(execution) -> AbstractExecutionBackend

NUFSHT's resolution of `AutoBackend` for a farm: `ThreadedBackend` when Julia has more than one thread
and the OhMyThreads extension is loaded, otherwise `SerialBackend`. Every other backend is returned
as given.
"""
@inline _resolve_execution(execution::ComputationalBackends.AbstractExecutionBackend) = execution
@inline _resolve_execution(::ComputationalBackends.AbstractAutoBackend) =
    (Threads.nthreads() > 1 && _omt_loaded()) ? ComputationalBackends.ThreadedBackend() :
                                                ComputationalBackends.SerialBackend()

# Backends under which a plan over one process's points is applied as built.
const _OneProcess = Union{ComputationalBackends.AbstractLocalBackend, ComputationalBackends.AbstractAutoBackend}

_execution_unavailable(execution, what) = throw(ArgumentError(
    "$(nameof(typeof(execution))) does not run $what here. Supported: ThreadedBackend farms " *
    "(`using OhMyThreads`); GPUBackend plans (`using KernelAbstractions, GPUArraysCore`); " *
    "DistributedBackend node-set farms and scalar plans (`using Distributed`); MPIBackend scalar plans " *
    "and per-rank adjoints, fits and analyses (`using MPI`)."))

# Every FastTransforms call of a plan goes through `FTB.with_fasttransforms_threads`, which sets the
# OpenMP count on the OS thread that makes the call and restores the count it found. The count is the
# plan's own, `n`, unless a task farm or a column pool has set `FTB.FASTTRANSFORMS_THREADS` for the tasks
# it spawns.
@inline _ft_threads(n::Integer) = something(FTB.FASTTRANSFORMS_THREADS[], Int(n))
@inline _ft_call(f::F, n::Integer) where {F} = FTB.with_fasttransforms_threads(f, _ft_threads(n))

# `lmul!(P, x)` at the thread count of `plan`.
@inline _ft_lmul!(P, x, plan::AbstractNUSHTplan) = _ft_call(() -> LinearAlgebra.lmul!(P, x), plan.ntasks)

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

# Copy batch slice `b` of a linear `(Nθ·Nφ·B)` array `A` to or from the dense `(Nθ,Nφ)` scratch `Fslice`
# by offset, allocating nothing; through the host slice `stage` when there is one (`_slice_stage`).
@inline _load_slice!(Fslice, ::Nothing, A, plan::NUSHTplan, b) =
    copyto!(Fslice, 1, A, (b - 1) * _slicelen(plan) + 1, _slicelen(plan))
@inline function _load_slice!(Fslice, stage::AbstractArray, A, plan::NUSHTplan, b)
    copyto!(stage, 1, A, (b - 1) * _slicelen(plan) + 1, _slicelen(plan))
    return copyto!(Fslice, stage)
end
@inline _store_slice!(A, Fslice, ::Nothing, plan::NUSHTplan, b) =
    copyto!(A, (b - 1) * _slicelen(plan) + 1, Fslice, 1, _slicelen(plan))
@inline function _store_slice!(A, Fslice, stage::AbstractArray, plan::NUSHTplan, b)
    copyto!(stage, Fslice)
    return copyto!(A, (b - 1) * _slicelen(plan) + 1, stage, 1, _slicelen(plan))
end

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

# `op!(entry..., b)` for the columns `b in 1:k` of a FastTransforms step. FastTransforms has no batched
# `lmul!`, so the column loop is the only parallelism available: with a pool of more than one entry the
# columns run on one task per entry, task `c` taking `c, c + nt, …` with `pool[c]`, its own slices and
# plans (keyed by chunk, see `_sph_pool`), and FastTransforms on one OpenMP thread inside the tasks.
# Otherwise they run on the calling task with `own`.
function _pool_columns!(op!::F, own::Tuple, pool::AbstractVector, k::Integer) where {F}
    nt = min(length(pool), k)
    if nt <= 1
        for b in 1:k
            op!(own..., b)
        end
        return nothing
    end
    Base.ScopedValues.with(FTB.FASTTRANSFORMS_THREADS => 1) do
        @sync for c in 1:nt
            Threads.@spawn begin
                e = pool[c]
                for b in c:nt:k
                    op!(e..., b)
                end
            end
        end
    end
    return nothing
end

# A scalar plan's per-column sphere operation, `op!(slice, stage, P, P', b)`, over its first `k` columns.
function _sph_columns!(op!::F, plan::NUSHTplan, k::Integer) where {F}
    _pool_columns!(op!, (plan.Fslice, plan.Fstage, plan.sph_plan, plan.sph_plan_adj), plan.sph_pool, k)
    return plan
end

# S (forward): `plan_sph2fourier` alone, per batch slice through a dense slice buffer. Its output is
# the DFS bivariate Fourier series, which `_assemble_modes!` hands to the NUFFT; no equiangular grid is
# formed.
function _sph_evaluate!(plan::NUSHTplan, k::Integer = plan.B)
    _sph_columns!(plan, k) do sl, st, P, _Padj, b
        _load_slice!(sl, st, plan.F, plan, b)
        _ft_lmul!(P, sl, plan)
        _store_slice!(plan.F, sl, st, plan, b)
    end
    return plan
end

# Undo the constant wavenumber offset centered ordering imposes; `conj` keeps the pair an exact transpose.
_rephase!(fbuf, ::Nothing) = fbuf
_rephase!(fbuf, s::AbstractVector) = (fbuf .*= s; fbuf)
_rephase_conj!(fbuf, ::Nothing) = fbuf
_rephase_conj!(fbuf, s::AbstractVector) = (fbuf .*= conj.(s); fbuf)

# A half-height complex mode array carries no `kθ < 0` rows, so each retained `kθ > 0` row stands for
# itself and its conjugate partner and counts twice; `kθ = 0` (row 1 of the folded layout) is its own
# partner. A real-data transform supplies the conjugate half itself, and the plans without one are
# exactly those carrying a `θ_shift`.
_fold_weights!(Z, plan::NUSHTplan) = _fold_weights!(Z, plan, _θshift(plan))
_fold_weights!(Z, ::NUSHTplan, ::Nothing) = Z
function _fold_weights!(Z, ::NUSHTplan{T}, ::AbstractVector) where {T}
    Z .*= T(2)
    @views Z[1, :, :] ./= T(2)
    return Z
end

# Leading `k` columns of a batch buffer. Contiguous, so the width-`k` plans apply to it exactly.
@inline _pfx(A::AbstractArray{<:Any,3}, k::Integer) = view(A, :, :, 1:k)
@inline _pfx(A::AbstractMatrix, k::Integer) = view(A, :, 1:k)

# The reduced-width plan pair for `k`, built and stored on a miss (see `_pool_lookup!`). A backend
# whose plan type carries the transform count (`_width_polymorphic` false) has one plan type per width,
# and a binding holding either width is a `Union` that boxes a large immutable on every transform, so
# each width is read only inside the branch that selected it, where its type is concrete.
@inline _with_narrowed(f::F, plan::NUSHTplan, k::Integer) where {F} =
    _with_pool_entry(f, plan.size_pool, k, kk -> _build_width!(plan, kk))

# F·N (forward): bivariate Fourier coefficients → complex mode array → type-2 NUFFT into `fbuf`.
# `_assemble_modes!` produces the modes the NUFFT evaluates. The full width never reads the pool.
function _dfn_synthesis!(plan::NUSHTplan, k::Integer = plan.B)
    if k == plan.B
        _fold_weights!(_assemble_modes!(plan.Fhat, plan.F, plan.lmax), plan)
        _nufft_exec!(_nufft2(plan), plan.Fhat, _fbuf(plan))
        _rephase!(_fbuf(plan), _θshift(plan))
    else
        _fold_weights!(_assemble_modes!(_pfx(plan.Fhat, k), _pfx(plan.F, k), plan.lmax), plan)
        _with_narrowed(plan, k) do e
            _nufft_exec!(e.nufft_type2, _pfx(plan.Fhat, k), _pfx(_fbuf(plan), k))
        end
        _rephase!(_pfx(_fbuf(plan), k), _θshift(plan))
    end
    return plan
end

# N†·F† (adjoint): type-1 NUFFT from `fbuf` → mode array → the exact transpose of the assembly.
# A synthesis-only plan holds no analysis handle; this is the one place that requires it.
function _dfn_analysis!(plan::NUSHTplan, k::Integer = plan.B)
    if k == plan.B
        _rephase_conj!(_fbuf(plan), _θshift(plan))
        _nufft_exec!(_require_analysis(plan.nodes.nufft_type1), _fbuf(plan), plan.Fhat)
        _assemble_modes_adjoint!(plan.F, plan.Fhat, plan.lmax)
    else
        _rephase_conj!(_pfx(_fbuf(plan), k), _θshift(plan))
        _with_narrowed(plan, k) do e
            _nufft_exec!(_require_analysis(e.nufft_type1), _pfx(_fbuf(plan), k), _pfx(plan.Fhat, k))
        end
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
mode array (F) → NUFFT type 2 (N).
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
result to `C`. This is the exact Euclidean adjoint of [`nusht_type2!`](@ref), its transpose, so `A†A`
is Hermitian positive semidefinite and usable by an iterative solver. It carries no quadrature weights
and fills every slot of `C`, the supernumerary ones included; [`nusht_analyze!`](@ref) gives the
coefficients of `f`.
"""
nusht_type1!(C, f, plan::NUSHTplan) = _nusht_true_adjoint!(C, f, plan)

"""
    nusht_type1!(C, f_local, plan, execution) -> C

`A†f` over the points of every process `execution` names, `plan` and `f_local` being this process's:
under `MPIBackend(; comm)` (with MPI loaded) each rank's share, summed over the ranks of `comm` into
`C` on every rank. A local or `AutoBackend()` backend names this process alone, and gives
`nusht_type1!(C, f_local, plan)`.
"""
nusht_type1!(C, f_local, plan::NUSHTplan, execution::ComputationalBackends.AbstractExecutionBackend) =
    _point_share_adjoint!(C, f_local, plan, execution)

# `A†f` over the points of every process `execution` names; NUFSHTMPIExt adds `MPIBackend`.
_point_share_adjoint!(C, f, plan::NUSHTplan, ::_OneProcess) = _nusht_true_adjoint!(C, f, plan)
_point_share_adjoint!(C, f, plan::NUSHTplan, e::ComputationalBackends.AbstractExecutionBackend) =
    _execution_unavailable(e, "a point-decomposed adjoint")

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
    return _strengths_adjoint!(C, plan, k, kdfn)
end

# `A†` applied to the plan's strengths buffer, into `C`.
function _strengths_adjoint!(C, plan::NUSHTplan, k::Integer, kdfn::Integer)
    _dfn_analysis!(plan, kdfn)
    _sph_columns!(plan, k) do sl, st, _P, Padj, b
        _load_slice!(sl, st, plan.F, plan, b)
        _ft_lmul!(Padj, sl, plan)
        _store_slice!(C, sl, st, plan, b)
    end
    return C
end

# ─────────────────────────────────────────────────────────────────────────────
# Synthesis of filtered coefficients
# ─────────────────────────────────────────────────────────────────────────────

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
    scratch = _filter_scratch(plan)
    C === scratch || copyto!(scratch, C)
    apply_transfer!(scratch, filter, plan.lmax)
    nusht_type2!(f_out, scratch, plan)
    return f_out
end

# ─────────────────────────────────────────────────────────────────────────────
# Exact inversion: least squares on min ‖A c − f‖ by FlowTransformBindings' LSMR
# ─────────────────────────────────────────────────────────────────────────────
# A plan is the operator of its own fit: synthesis is `A`, the exact adjoint `A†`, through the methods
# `FTB.lsmr!` calls. A scalar plan's fit runs over the `(lmax+1)²` slots holding degrees `l ≤ lmax`,
# packed (see `_valid_indices`): the array's supernumerary slots carry degrees up to `lmax+|m|`, and the
# degrees `l ≤ lmax` span the subspace invariant under rotation, so the fit is independent of the
# coordinate frame. The adjoint lands in the plan's full layout, `plan.F`, and gathering it into the
# packed vector is the projection onto those slots. `plan.F` is free from that gather to the next forward
# step, which refills it.

# Length of one packed solver column, and of one column of the plan's full layout.
@inline _coefflen(plan::NUSHTplan) = (plan.lmax + 1)^2
@inline _fulllen(plan::NUSHTplan) = plan.Nθ * plan.Nφ

# `k` is the live column count the per-column sphere loop narrows to; `kdfn` the NUFFT plan width,
# which can only narrow where the backend's plans are width-polymorphic.
@inline _lsmr_widths(plan::NUSHTplan, nlive::Integer) =
    (nlive, plan.pool_recipe.narrowable ? _fit_width(max(nlive, 1), plan.B) : plan.B)

# The packed slots, built by the first solve and kept on the plan.
@inline function _valid(plan::NUSHTplan)
    v = plan.valid[]
    v === nothing || return v
    return _build_valid!(plan)
end
@noinline _build_valid!(plan::NUSHTplan) = (plan.valid[] = _valid_indices(plan.F, plan.lmax))

FTB.lsmr_ncolumns(plan::AbstractNUSHTplan) = plan.B
FTB.lsmr_allocate_domain(plan::NUSHTplan{T,FE}) where {T,FE} =
    (_valid(plan); _zeros_like(plan.F, FE, _coefflen(plan), plan.B))
FTB.lsmr_allocate_range(plan::NUSHTplan{T,FE}) where {T,FE} = _zeros_like(_fbuf(plan), FE, _npts(plan), plan.B)
FTB.lsmr_check_solution(C, plan::NUSHTplan, ws) = _assert_coeffs(C, plan)

# `u ← A v + c u`: `u` is scaled first and the synthesis, which lands in the plan's strengths buffer,
# is added to it.
function FTB.lsmr_forward!(u, plan::NUSHTplan, v, c, n)
    k, kdfn = _lsmr_widths(plan, n)
    FTB.colscale!(u, c, n, plan.B)
    _unpack_coeffs!(plan.F, v, _valid(plan), _fulllen(plan), plan.B)
    _sph_evaluate!(plan, k)
    _dfn_synthesis!(plan, kdfn)
    return _add_out!(u, _fbuf(plan), plan, n)
end

FTB.lsmr_adjoint!(v, plan::AbstractNUSHTplan, u, c, n) = _fold!(v, plan, _adjoint_full!(plan, u, n), c, n)

# `A†u` in the plan's full coefficient layout.
function _adjoint_full!(plan::NUSHTplan, u, n::Integer)
    k, kdfn = _lsmr_widths(plan, n)
    return _nusht_true_adjoint!(plan.F, u, plan, k, kdfn)
end

# `v ← P w + c v` for the full-layout `w`, `P` the gather onto the packed slots.
_fold!(v, plan::NUSHTplan, w, ::Nothing, n::Integer) = _pack_coeffs!(v, w, _valid(plan), _fulllen(plan), n)
_fold!(v, plan::NUSHTplan, w, c, n::Integer) = _col_pbp_pack!(v, w, c, _valid(plan), _fulllen(plan), n)

FTB.lsmr_write!(C, plan::NUSHTplan, x, k, j) = _write_solution!(C, x, _valid(plan), _fulllen(plan), k, j)

"""
    _PointShare(plan, reduce!)

`plan` over one share of the points, as the operator of the fit over every share: `reduce!(A)` sums
`A` in place across the shares, and is applied to the two sums over points LSMR takes, the per-column
`‖u‖²` and `A†u`. The coefficients are the same on every share, so every share runs the same recurrence
and stops on the same iteration.
"""
struct _PointShare{P<:AbstractNUSHTplan, R}
    plan::P
    reduce!::R
end

FTB.lsmr_ncolumns(s::_PointShare) = FTB.lsmr_ncolumns(s.plan)
FTB.lsmr_allocate_domain(s::_PointShare) = FTB.lsmr_allocate_domain(s.plan)
FTB.lsmr_allocate_range(s::_PointShare) = FTB.lsmr_allocate_range(s.plan)
FTB.lsmr_check_solution(C, s::_PointShare, ws) = FTB.lsmr_check_solution(C, s.plan, ws)
FTB.lsmr_write!(C, s::_PointShare, x, k, j) = FTB.lsmr_write!(C, s.plan, x, k, j)
FTB.lsmr_forward!(u, s::_PointShare, v, c, n) = FTB.lsmr_forward!(u, s.plan, v, c, n)

function FTB.lsmr_range_norm2!(out, s::_PointShare, u, n)
    FTB.lsmr_range_norm2!(out, s.plan, u, n)
    s.reduce!(view(out, 1:n))
    return out
end

function FTB.lsmr_adjoint!(v, s::_PointShare, u, c, n)
    w = _adjoint_full!(s.plan, u, n)
    s.reduce!(w)
    return _fold!(v, s.plan, w, c, n)
end

# A batch buffer's linear column stride: one column of a point-space `(M, B)` buffer or of a
# coefficient-space `(Nθ, Nφ, B)` one.
@inline _colstride(A) = length(A) ÷ size(A, ndims(A))

# `v[:,k] = gather(wfull[:,k]) + β[k]·v[:,k]` — the packed counterpart of `FTB.colxpby!`, for the step
# where the adjoint has written the plan's full layout and the iterate is packed. The gather is the
# projection onto `l ≤ lmax`.
function _col_pbp_pack!(v, wfull, β, idx, fulllen::Integer, n::Integer)
    K = length(idx)
    @inbounds for k in 1:n
        c = β[k]
        vo = (k - 1) * K
        wo = (k - 1) * fulllen
        @simd for t in 1:K
            v[vo + t] = wfull[wo + idx[t]] + c * v[vo + t]
        end
    end
    return v
end

# Write column `slot` of the packed iterate `x` into column `dstcol` of the caller's full-layout `C`,
# the supernumerary slots zeroed.
function _write_solution!(C, x, idx, full::Integer, slot::Integer, dstcol::Integer)
    K = length(idx)
    so = (slot - 1) * K
    do_ = (dstcol - 1) * full
    @inbounds begin
        @simd for i in 1:full
            C[do_ + i] = zero(eltype(C))
        end
        @simd for t in 1:K
            C[do_ + idx[t]] = x[so + t]
        end
    end
    return C
end

# Working width for `n` live columns: the next power of two, capped at `B`. `_build_width!` builds any
# width on demand, and the powers of two bound how many distinct plan sets a solve can create.
@inline _fit_width(n::Integer, B::Integer) = min(Int(B), Int(nextpow(2, max(n, 1))))

"""
    nusht_solve!(C, f, plan; ws = FlowTransformBindings.LSMRWorkspace(plan), weights = nothing,
                 maxiter = 500, rtol = 1e-6, conlim = 0) -> (; C, iterations, residual, converged)
    nusht_solve!(C, f, FlowTransformBindings.RangeWeighted(plan, weights); ws, …)

**Exact inversion:** solve `min ‖A c − f‖` for the coefficients `C` of each of the `B` columns by LSMR
(`FlowTransformBindings.lsmr!`), `ws` holding the solver's arrays so a solve that reuses it allocates
nothing.

`weights`, nonnegative and one per point, makes it `min Σⱼ wⱼ |(A c − f)ⱼ|²`; a zero weight leaves its
point out, whatever `f` holds there (a masked `NaN`, say). The points' quadrature weights
or their areas bring `A^H W A` close to a multiple of the identity, and the iteration count down with it;
for data the degree-`lmax` harmonics represent, the solution is the unweighted one. The second form takes
the weighted operator built once, so a repeated weighted solve allocates nothing either.

A column stops once `‖A†r‖ ≤ rtol ‖A†f‖`, when LSMR's estimate of `cond(A)` reaches `conlim` (`1/eps(T)`
at `0`), or when its Krylov process ends exactly; `residual` is the largest `‖A†r‖/‖A†f‖`, at least
`eps(T)`, and `ws.residual` and `ws.status` hold each column's. Points that do not determine the
coefficients (`M` below `(lmax+1)²`, or clustered) make `A` rank deficient, and a column then stops on
`conlim` with `converged == false`.
"""
nusht_solve!(C, f, plan::NUSHTplan; ws::FTB.LSMRWorkspace = FTB.LSMRWorkspace(plan), weights = nothing,
             kwargs...) =
    _solve!(C, f, _weighted(plan, weights), ws; kwargs...)

"""
    nusht_solve!(C, f_local, plan, execution; ws, weights = nothing, …) -> (; C, iterations, residual, converged)
    nusht_solve!(C, f_local, FlowTransformBindings.RangeWeighted(plan, weights), execution; ws, …)

The fit over the points of every process `execution` names, `plan`, `f_local` and `weights` being this
process's: under `MPIBackend(; comm)` (with MPI loaded) each rank's share, the coefficients replicated
on every rank of `comm`. A local or `AutoBackend()` backend names this process alone, and gives
`nusht_solve!(C, f_local, plan)`. The keywords are those of `nusht_solve!(C, f, plan)`; the second form
takes this process's weighted operator built once.
"""
nusht_solve!(C, f_local, plan::NUSHTplan, execution::ComputationalBackends.AbstractExecutionBackend;
             ws::FTB.LSMRWorkspace = FTB.LSMRWorkspace(plan), weights = nothing, kwargs...) =
    _solve!(C, f_local, _weighted(_point_share(plan, execution), weights), ws; kwargs...)

nusht_solve!(C, f_local, op::FTB.RangeWeighted{<:NUSHTplan}, execution::ComputationalBackends.AbstractExecutionBackend;
             ws::FTB.LSMRWorkspace = FTB.LSMRWorkspace(op), kwargs...) =
    _solve!(C, f_local, _weighted_share(op, execution), ws; kwargs...)

_weighted(plan, ::Nothing) = plan
_weighted(plan, weights) = FTB.RangeWeighted(plan, weights)

# The operator of a fit over the points of every process `execution` names; NUFSHTMPIExt adds
# `MPIBackend`.
_point_share(op, ::_OneProcess) = op
_point_share(op, e::ComputationalBackends.AbstractExecutionBackend) =
    _execution_unavailable(e, "a point-decomposed solve")

# A weighted operator's weights on the share of its plan's points `execution` names.
_weighted_share(op::FTB.RangeWeighted, ::_OneProcess) = op
_weighted_share(op::FTB.RangeWeighted, execution) = FTB.RangeWeighted(_point_share(op.op, execution), op)

nusht_solve!(C, f, op::FTB.RangeWeighted{<:NUSHTplan}; ws::FTB.LSMRWorkspace = FTB.LSMRWorkspace(op),
             kwargs...) =
    _solve!(C, f, op, ws; kwargs...)

# The fit through `op`: a plan, or a plan's share of the points (`_PointShare`).
function _solve!(C, f, op, ws::FTB.LSMRWorkspace{T}; maxiter::Integer = 500, rtol::Real = 1e-6,
                 conlim::Real = 0) where {T}
    info = FTB.lsmr!(C, op, f, ws; maxiter, rtol, conlim = conlim > 0 ? conlim : inv(eps(T)))
    return (; C, iterations = info.iterations, residual = info.residual, converged = info.converged)
end

# ─────────────────────────────────────────────────────────────────────────────
# Analysis: the coefficients of a sampled field, by a quadrature or by the fit
# ─────────────────────────────────────────────────────────────────────────────

"""
    nusht_analyze!(C, f, plan; weights = nothing, ws, maxiter, rtol, conlim) -> (; C, iterations, residual, converged)

The coefficients of the field `f` sampled at the points of a scalar or spin plan, written into `C`. With
`weights`, one per point in the plan's array type, the quadrature `Cₖ = Σⱼ wⱼ fⱼ conj(Yₖ(xⱼ))`: one
adjoint transform, `iterations = 0`, `residual = 0`, `converged = true`, and the expansion of a
degree-`lmax` field when the rule integrates the product of two degree-`lmax` harmonics exactly. Without,
the least-squares fit [`nusht_solve!`](@ref) (spin: [`nusht_solve_spin!`](@ref)), which takes the
remaining keywords; the quadrature does not use them.

A point of zero weight contributes nothing to the quadrature, whatever `f` holds there (a NaN on masked
land). A scalar plan's slots above degree `lmax` are zero, and a real-field spin plan returns the
Hermitian coefficients of the real field. The quadrature allocates nothing, the fit nothing once `ws` is
held.
"""
nusht_analyze!(C, f, plan::AbstractNUSHTplan; weights = nothing, kwargs...) =
    _analyze!(C, f, plan, weights; kwargs...)

"""
    nusht_analyze!(C, f_local, plan, execution; weights = nothing, …) -> (; C, iterations, residual, converged)

The analysis over the points of every process `execution` names, `plan`, `f_local` and `weights` being
this process's: under `MPIBackend(; comm)` (with MPI loaded) the quadrature summed over the ranks of
`comm`, or the fit `nusht_solve!(C, f_local, plan, execution)` (spin: `nusht_solve_spin!`), with the
coefficients on every rank. A local or `AutoBackend()` backend names this process alone, and gives
`nusht_analyze!(C, f_local, plan)`.
"""
nusht_analyze!(C, f_local, plan::AbstractNUSHTplan, execution::ComputationalBackends.AbstractExecutionBackend;
               weights = nothing, kwargs...) =
    _analyze!(C, f_local, plan, weights, execution; kwargs...)

# The fit without weights (the spin plan's methods are in Spin.jl), the quadrature with them.
_analyze!(C, f, plan::NUSHTplan, ::Nothing; kwargs...) = nusht_solve!(C, f, plan; kwargs...)
_analyze!(C, f, plan::NUSHTplan, ::Nothing, execution; kwargs...) =
    nusht_solve!(C, f, plan, execution; kwargs...)

function _analyze!(C, f, plan::AbstractNUSHTplan, w::AbstractArray; ws = nothing, maxiter = nothing,
                   rtol = nothing, conlim = nothing)
    _quadrature!(C, f, w, plan)
    return _quadrature_result(C, plan)
end

function _analyze!(C, f, plan::AbstractNUSHTplan, w::AbstractArray, execution; ws = nothing,
                   maxiter = nothing, rtol = nothing, conlim = nothing)
    _point_share_quadrature!(C, f, w, plan, execution)
    return _quadrature_result(C, plan)
end

_quadrature_result(C, plan) =
    (; C, iterations = 0, residual = zero(real(eltype(_fbuf(plan)))), converged = true)

# The quadrature over the points of every process `execution` names; NUFSHTMPIExt adds `MPIBackend`.
_point_share_quadrature!(C, f, w, plan::AbstractNUSHTplan, ::_OneProcess) = _quadrature!(C, f, w, plan)
_point_share_quadrature!(C, f, w, plan::AbstractNUSHTplan, e::ComputationalBackends.AbstractExecutionBackend) =
    _execution_unavailable(e, "a point-decomposed analysis")

@inline function _assert_weights(w, plan::AbstractNUSHTplan)
    length(w) == _npts(plan) || throw(DimensionMismatch(
        "weights has $(length(w)) entries for the plan's $(_npts(plan)) points"))
    return nothing
end

# `w f`, zero where `w` is zero whatever `f` holds there (a NaN or Inf on a masked point).
@inline _weighted_value(w, f) = (p = w * f; ifelse(iszero(w), zero(p), p))

# `fbuf[j, b] = w[j] f[j, b]` by `_weighted_value`, with `f` shaped `(M,)` or `(M, B)`; the device method
# is in the KernelAbstractions extension.
function _copy_weighted!(fbuf, f, w)
    M = length(w)
    @inbounds for b in 1:(length(fbuf) ÷ M), j in 1:M
        i = (b - 1) * M + j
        fbuf[i] = _weighted_value(w[j], f[i])
    end
    return fbuf
end

# `Σⱼ wⱼ fⱼ Yₖ(xⱼ)` over the real harmonics `Yₖ` of degree `l ≤ lmax`, the supernumerary slots zeroed:
# the adjoint of the weighted strengths, then the projection onto those degrees, the top hat at `lmax`.
function _quadrature!(C, f, w, plan::NUSHTplan)
    _assert_field(f, plan)
    _assert_coeffs(C, plan)
    _assert_weights(w, plan)
    _copy_weighted!(_fbuf(plan), f, w)
    _strengths_adjoint!(C, plan, plan.B, plan.B)
    return apply_transfer!(C, TopHatTransfer(plan.lmax), plan.lmax)
end

# ─────────────────────────────────────────────────────────────────────────────
# Filtering scattered data: a fit, then the synthesis of the filtered coefficients
# ─────────────────────────────────────────────────────────────────────────────

"""
    nusht_filter!(f_out, f_in, filter, plan; ws = FlowTransformBindings.LSMRWorkspace(plan), kwargs...)

Apply a spectral filter to `f_in` at the scattered points, writing to `f_out` (both length-`M` /
`(M, B)`): fit coefficients with [`nusht_solve!`](@ref) → `apply_transfer!` (× H(ℓ)) → `nusht_type2!`.
`kwargs` (`rtol`, `maxiter`, …) go to the solve. Uses the plan's coefficient scratch, and is
allocation-free when a `ws` is supplied.

The filter is `A H A⁺`, with `A⁺` the least-squares fit. At scattered points the adjoint `A†` is the
transpose of the synthesis, and `A H A†` is a smoothing; the two agree only on a quadrature grid. The
fit is iterative, so hold a `ws` and reuse it across calls.
"""
function nusht_filter!(f_out, f_in, filter, plan::NUSHTplan;
                       ws::FTB.LSMRWorkspace = FTB.LSMRWorkspace(plan), kwargs...)
    scratch = _filter_scratch(plan)
    nusht_solve!(scratch, f_in, plan; ws = ws, kwargs...)
    nusht_synthesize!(f_out, scratch, filter, plan)
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
                              ws::FTB.LSMRWorkspace = FTB.LSMRWorkspace(plan),
                              C_mask = nothing) where {T}
    # `C_mask` holds the mask's coefficients, which do not depend on the filter's scale, so a multi-scale
    # caller fits them once; `nothing` fits the mask here.
    if C_mask === nothing
        scratch = _filter_scratch(plan)
        nusht_solve!(scratch, mask, plan; ws = ws)
        nusht_synthesize!(mask_filt, scratch, filter, plan)
    else
        nusht_synthesize!(mask_filt, C_mask, filter, plan)
    end
    threshold = T(0.01)
    # `mask_filt` is shaped like `f_out` → a single fused broadcast, zero-alloc + device-safe.
    f_out .= ifelse.(abs.(mask_filt) .>= threshold, f_out ./ mask_filt, zero(T))
    return f_out
end

include("Spin.jl")
include("RingEngine.jl")

# ─────────────────────────────────────────────────────────────────────────────
# Collections of independent problems, split across an execution backend
# ─────────────────────────────────────────────────────────────────────────────
# One plan per problem. Each entry resolves `execution` and hands it to its `_farm_*` function:
# `SerialBackend` loops here, `ThreadedBackend` in NUFSHTOhMyThreadsExt.

@inline function _check_farm(outs, ins, plans)
    @assert length(outs) == length(ins) == length(plans) "outs, ins and plans must have equal length"
    return nothing
end

"""
    nusht_type2!(fs, Cs, plans; execution = AutoBackend()) -> fs

Synthesize a collection of independent problems: for each `i`, `nusht_type2!(fs[i], Cs[i], plans[i])`.
`execution = ThreadedBackend()` runs the problems on concurrent tasks and requires `using OhMyThreads`;
`AutoBackend()` takes it when Julia has more than one thread and OhMyThreads is loaded, and
`SerialBackend()` otherwise.
"""
nusht_type2!(fs, Cs, plans::AbstractVector;
             execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend()) =
    _farm_type2!(fs, Cs, plans, _resolve_execution(execution))
function _farm_type2!(fs, Cs, plans::AbstractVector, ::ComputationalBackends.AbstractSerialBackend)
    _check_farm(fs, Cs, plans)
    for i in eachindex(plans)
        nusht_type2!(fs[i], Cs[i], plans[i])
    end
    return fs
end

"""
    nusht_type1!(Cs, fs, plans; execution = AutoBackend()) -> Cs

Adjoint analysis over a collection of independent problems; see [`nusht_type2!`](@ref).
"""
nusht_type1!(Cs, fs, plans::AbstractVector;
             execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend()) =
    _farm_type1!(Cs, fs, plans, _resolve_execution(execution))
function _farm_type1!(Cs, fs, plans::AbstractVector, ::ComputationalBackends.AbstractSerialBackend)
    _check_farm(Cs, fs, plans)
    for i in eachindex(plans)
        nusht_type1!(Cs[i], fs[i], plans[i])
    end
    return Cs
end

"""
    nusht_solve!(Cs, fs, plans; execution = AutoBackend(), kwargs...) -> Cs

Exact inversion over a collection of independent problems; `kwargs` go to the single-problem method.
`execution` is as in [`nusht_type2!`](@ref).
"""
nusht_solve!(Cs, fs, plans::AbstractVector;
             execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
             kwargs...) =
    _farm_solve!(Cs, fs, plans, _resolve_execution(execution); kwargs...)
function _farm_solve!(Cs, fs, plans::AbstractVector, ::ComputationalBackends.AbstractSerialBackend; kwargs...)
    _check_farm(Cs, fs, plans)
    for i in eachindex(plans)
        nusht_solve!(Cs[i], fs[i], plans[i]; kwargs...)
    end
    return Cs
end

"""
    nusht_analyze!(Cs, fs, plans; execution = AutoBackend(), weights = nothing, kwargs...) -> Cs

Analysis over a collection of independent problems, scalar or spin plans: for each `i`,
`nusht_analyze!(Cs[i], fs[i], plans[i]; weights = weights[i], kwargs...)`, the fit when `weights` is
`nothing`. `execution` is as in [`nusht_type2!`](@ref).
"""
nusht_analyze!(Cs, fs, plans::AbstractVector;
               execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
               weights = nothing, kwargs...) =
    _farm_analyze!(Cs, fs, plans, weights, _resolve_execution(execution); kwargs...)
function _farm_analyze!(Cs, fs, plans::AbstractVector, weights, ::ComputationalBackends.AbstractSerialBackend;
                        kwargs...)
    _check_farm(Cs, fs, plans)
    _check_farm_weights(weights, plans)
    for i in eachindex(plans)
        nusht_analyze!(Cs[i], fs[i], plans[i]; weights = _problem_weights(weights, i), kwargs...)
    end
    return Cs
end

# One weight vector per problem, or none.
_problem_weights(::Nothing, i) = nothing
_problem_weights(weights, i) = weights[i]
_check_farm_weights(::Nothing, plans) = nothing
function _check_farm_weights(weights, plans)
    length(weights) == length(plans) || throw(DimensionMismatch(
        "weights holds $(length(weights)) vectors for $(length(plans)) plans; pass one per plan"))
    return nothing
end

"""
    nusht_filter!(outs, ins, filter, plans; execution = AutoBackend()) -> outs

Spectral filtering over a collection of independent problems; see [`nusht_type2!`](@ref).
"""
nusht_filter!(outs, ins, filter, plans::AbstractVector;
              execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend()) =
    _farm_filter!(outs, ins, filter, plans, _resolve_execution(execution))
function _farm_filter!(outs, ins, filter, plans::AbstractVector, ::ComputationalBackends.AbstractSerialBackend)
    _check_farm(outs, ins, plans)
    for i in eachindex(plans)
        nusht_filter!(outs[i], ins[i], filter, plans[i])
    end
    return outs
end

"""
    nusht_type2_spin!(fs, sfs, plans; execution = AutoBackend()) -> fs

Spin-weighted synthesis over a collection of independent problems, `execution` as in
[`nusht_type2!`](@ref).
"""
nusht_type2_spin!(fs, sfs, plans::AbstractVector;
                  execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend()) =
    _farm_type2_spin!(fs, sfs, plans, _resolve_execution(execution))
function _farm_type2_spin!(fs, sfs, plans::AbstractVector, ::ComputationalBackends.AbstractSerialBackend)
    _check_farm(fs, sfs, plans)
    for i in eachindex(plans)
        nusht_type2_spin!(fs[i], sfs[i], plans[i])
    end
    return fs
end

"""
    nusht_type1_spin!(sfs, fs, plans; execution = AutoBackend()) -> sfs

Spin-weighted adjoint analysis over a collection; see [`nusht_type2_spin!`](@ref).
"""
nusht_type1_spin!(sfs, fs, plans::AbstractVector;
                  execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend()) =
    _farm_type1_spin!(sfs, fs, plans, _resolve_execution(execution))
function _farm_type1_spin!(sfs, fs, plans::AbstractVector, ::ComputationalBackends.AbstractSerialBackend)
    _check_farm(sfs, fs, plans)
    for i in eachindex(plans)
        nusht_type1_spin!(sfs[i], fs[i], plans[i])
    end
    return sfs
end

"""
    nusht_solve_spin!(sfs, fs, plans; execution = AutoBackend(), kwargs...) -> sfs

Spin-weighted exact inversion over a collection; see [`nusht_type2_spin!`](@ref).
"""
nusht_solve_spin!(sfs, fs, plans::AbstractVector;
                  execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
                  kwargs...) =
    _farm_solve_spin!(sfs, fs, plans, _resolve_execution(execution); kwargs...)
function _farm_solve_spin!(sfs, fs, plans::AbstractVector, ::ComputationalBackends.AbstractSerialBackend;
                           kwargs...)
    _check_farm(sfs, fs, plans)
    for i in eachindex(plans)
        nusht_solve_spin!(sfs[i], fs[i], plans[i]; kwargs...)
    end
    return sfs
end

# A backend with no method above raises.
_farm_type2!(fs, Cs, plans::AbstractVector, e::ComputationalBackends.AbstractExecutionBackend) =
    _execution_unavailable(e, "a collection of syntheses")
_farm_type1!(Cs, fs, plans::AbstractVector, e::ComputationalBackends.AbstractExecutionBackend) =
    _execution_unavailable(e, "a collection of analyses")
_farm_solve!(Cs, fs, plans::AbstractVector, e::ComputationalBackends.AbstractExecutionBackend; kwargs...) =
    _execution_unavailable(e, "a collection of solves")
_farm_analyze!(Cs, fs, plans::AbstractVector, weights, e::ComputationalBackends.AbstractExecutionBackend;
               kwargs...) =
    _execution_unavailable(e, "a collection of analyses")
_farm_filter!(outs, ins, filter, plans::AbstractVector, e::ComputationalBackends.AbstractExecutionBackend) =
    _execution_unavailable(e, "a collection of filters")
_farm_type2_spin!(fs, sfs, plans::AbstractVector, e::ComputationalBackends.AbstractExecutionBackend) =
    _execution_unavailable(e, "a collection of spin syntheses")
_farm_type1_spin!(sfs, fs, plans::AbstractVector, e::ComputationalBackends.AbstractExecutionBackend) =
    _execution_unavailable(e, "a collection of spin analyses")
_farm_solve_spin!(sfs, fs, plans::AbstractVector, e::ComputationalBackends.AbstractExecutionBackend; kwargs...) =
    _execution_unavailable(e, "a collection of spin solves")

# ── Node-set form: build the plans internally ─────────────────────────────────
# Each problem's plan is built where it runs, so a `DistributedBackend` worker builds its own: a plan
# holding C pointers cannot be serialized.

"""
    nusht_type2(θs, φs, Cs, lmax; execution = AutoBackend(), tol, ntrans, …) -> fs

Synthesize `N` independent problems given their node sets: for each `i` a plan is built from
`(θs[i], φs[i])` with the remaining keywords, `Cs[i]` is evaluated, and the field is returned as `fs[i]`.

`execution = DistributedBackend()` (with Distributed loaded) takes this form, each worker building its
own plans. A local backend runs the problems in turn on the calling process, where the plan-collection
[`nusht_type2!`](@ref) reuses plans already built; `AutoBackend()` resolves as it does there.
"""
nusht_type2(θs, φs, Cs, lmax;
            execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
            kwargs...) =
    _nodeset_type2(θs, φs, Cs, lmax, _resolve_execution(execution); kwargs...)

function _nodeset_type2(θs, φs, Cs, lmax, ::ComputationalBackends.AbstractLocalBackend; kwargs...)
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

_nodeset_type2(θs, φs, Cs, lmax, e::ComputationalBackends.AbstractExecutionBackend; kwargs...) =
    _execution_unavailable(e, "a node-set synthesis farm")

"""
    nusht_solve(θs, φs, fs, lmax; execution = AutoBackend(), tol, rtol, maxiter, …) -> Cs

Exact inversion of `N` independent problems given their node sets; see [`nusht_type2`](@ref).
"""
nusht_solve(θs, φs, fs, lmax;
            execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
            kwargs...) =
    _nodeset_solve(θs, φs, fs, lmax, _resolve_execution(execution); kwargs...)

function _nodeset_solve(θs, φs, fs, lmax, ::ComputationalBackends.AbstractLocalBackend;
                        rtol = 1e-6, maxiter = 500, kwargs...)
    @assert length(θs) == length(φs) == length(fs) "θs, φs and fs must have equal length"
    return map(eachindex(fs)) do i
        plan = make_plan(θs[i], φs[i], lmax; kwargs...)
        try
            C = zeros(eltype(plan.F), lmax + 1, 2lmax + 1)
            nusht_solve!(C, fs[i], plan; rtol = rtol, maxiter = maxiter)
            return C
        finally
            close!(plan)
        end
    end
end

_nodeset_solve(θs, φs, fs, lmax, e::ComputationalBackends.AbstractExecutionBackend; kwargs...) =
    _execution_unavailable(e, "a node-set solve farm")

end # module NUFSHT
