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
using FlowTransformBindings: FlowTransformBindings as FTB
using ComputationalBackends: ComputationalBackends
using FFTW: FFTW
using FastSphericalHarmonics: FastSphericalHarmonics
using LinearAlgebra: LinearAlgebra

include("Modes.jl")
include("NUFFT.jl")
include("Plan.jl")
include("Kernels.jl")

export make_plan, NUSHTplan, close!, plan_memory
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
#  • **Farm over independent problems** (collection methods below). A NUFFT plan is not safe to
#    `exec!` concurrently, so each problem carries its own. Under `DistributedBackend` the node-set form
#    builds each problem's plan on its worker, since a plan holding C pointers cannot be serialized.
#  • **One transform** — `make_plan(FE, θ, φ, lmax, backend)`: a thread count, a device, or the points
#    divided among `DistributedBackend` workers or `MPIBackend` ranks.

@inline _omt_loaded() = Base.get_extension(@__MODULE__, :NUFSHTOhMyThreadsExt) !== nothing

"""
    _resolve_backend(backend) -> AbstractExecutionBackend

NUFSHT's resolution of `AutoBackend` for a farm: `ThreadedBackend` when Julia has more than one thread
and the OhMyThreads extension is loaded, otherwise `SerialBackend`. Every other backend is returned
as given.
"""
@inline _resolve_backend(backend::ComputationalBackends.AbstractExecutionBackend) = backend
@inline _resolve_backend(::ComputationalBackends.AbstractAutoBackend) =
    (Threads.nthreads() > 1 && _omt_loaded()) ? ComputationalBackends.ThreadedBackend() :
                                                ComputationalBackends.SerialBackend()

_backend_unavailable(backend, what) = throw(ArgumentError(
    "$(nameof(typeof(backend))) cannot run $what here: its extension is not loaded. ThreadedBackend " *
    "farms need `using OhMyThreads`, GPUBackend plans `using KernelAbstractions`, DistributedBackend " *
    "`using Distributed`, MPIBackend `using MPI`."))

# Every FastTransforms call goes through `FTB.with_fasttransforms_threads`, which sets the OpenMP count
# on the OS thread that makes the call: FastTransforms' own count by default, one thread inside a task
# farm, which sets `FTB.FASTTRANSFORMS_THREADS` for the tasks it spawns.
@inline _ft_lmul!(P, x) = FTB.with_fasttransforms_threads(() -> LinearAlgebra.lmul!(P, x))

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
# slice buffer and its own plans from the pool, keyed by chunk (see `_sph_pool`). The tasks carry the
# parallelism, so FastTransforms runs on one OpenMP thread inside them.
function _sph_columns!(op!, plan::NUSHTplan, k::Integer)
    pool = plan.sph_pool
    nt = min(length(pool), k)
    if nt <= 1
        @inbounds for b in 1:k
            op!(plan.Fslice, plan.sph_plan, plan.sph_plan_adj, b)
        end
        return plan
    end
    Base.ScopedValues.with(FTB.FASTTRANSFORMS_THREADS => 1) do
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
# the DFS bivariate Fourier series, which `_assemble_modes!` hands straight to the NUFFT. No
# equiangular grid is formed; the doubling that route needs is not exact at the odd `Nφ = 2lmax+1`.
function _sph_evaluate!(plan::NUSHTplan, k::Integer = plan.B)
    _sph_columns!(plan, k) do sl, P, _Padj, b
        _load_slice!(sl, plan.F, plan, b)
        _ft_lmul!(P, sl)
        _store_slice!(plan.F, sl, plan, b)
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
# partner. A real-data transform needs none of this — it supplies the conjugate half itself — and it is
# exactly the plans that do not have one that carry a `θ_shift`.
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

# The reduced-width plan pair for `k`, built and stored on a miss (see `_pool_lookup!`). Kept out of
# the full-width path entirely: a backend that carries the transform count in its plan's *type* — which
# is why [`_width_polymorphic`](@ref) exists — has no one type spanning widths, so a binding that could
# hold either the full-width handle or a narrowed one is a `Union`, and a `Union` of a large immutable
# is boxed on the heap once per transform. Each width is therefore only ever read where its own type is
# concrete, which is inside the branch that selected it.
@inline _with_narrowed(f::F, plan::NUSHTplan, k::Integer) where {F} =
    _with_pool_entry(f, plan.size_pool, k, kk -> _build_width!(plan, kk))

# F·N (forward): bivariate Fourier coefficients → complex mode array → type-2 NUFFT into `fbuf`.
# No doubling and no FFT: `_assemble_modes!` produces the modes the NUFFT evaluates directly.
# Full width first: it is the common case and never reads the pool.
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
        _ft_lmul!(Padj, sl)
        _store_slice!(C, sl, plan, b)
    end
    return C
end

# ─────────────────────────────────────────────────────────────────────────────
# Filtering
# ─────────────────────────────────────────────────────────────────────────────
# `nusht_filter!` and `nusht_filter_renorm!` fit coefficients first, and follow the solver.

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
    nusht_solve!(C, f, plan; ws = FlowTransformBindings.LSMRWorkspace(plan), maxiter = 500, rtol = 1e-6,
                 conlim = 0) -> (; C, iterations, residual, converged)

**Exact inversion:** solve `min ‖A c − f‖` for the coefficients `C` of each of the `B` columns by LSMR
(`FlowTransformBindings.lsmr!`), `ws` holding the solver's arrays so a solve that reuses it allocates
nothing.

A column stops once `‖A†r‖ ≤ rtol ‖A†f‖`, when LSMR's estimate of `cond(A)` reaches `conlim` (`1/eps(T)`
at `0`), or when its Krylov process ends exactly; `residual` is the largest `‖A†r‖/‖A†f‖`, at least
`eps(T)`, and `ws.residual` and `ws.status` hold each column's. Points that do not determine the
coefficients (`M` below `(lmax+1)²`, or clustered) make `A` rank deficient, and a column then stops on
`conlim` with `converged == false`.
"""
nusht_solve!(C, f, plan::NUSHTplan; ws::FTB.LSMRWorkspace = FTB.LSMRWorkspace(plan), kwargs...) =
    _solve!(C, f, plan, ws; kwargs...)

# The fit through `op`: a plan, or a plan's share of the points (`_PointShare`).
function _solve!(C, f, op, ws::FTB.LSMRWorkspace{T}; maxiter::Integer = 500, rtol::Real = 1e-6,
                 conlim::Real = 0) where {T}
    info = FTB.lsmr!(C, op, f, ws; maxiter, rtol, conlim = conlim > 0 ? conlim : inv(eps(T)))
    return (; C, iterations = info.iterations, residual = info.residual, converged = info.converged)
end

# ─────────────────────────────────────────────────────────────────────────────
# Filtering that fits coefficients first (see the note above `nusht_synthesize!`)
# ─────────────────────────────────────────────────────────────────────────────

"""
    nusht_filter!(f_out, f_in, filter, plan; ws = FlowTransformBindings.LSMRWorkspace(plan), kwargs...)

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
    # `C_mask` lets a multi-scale caller fit the scale-independent mask once and pass its coefficients
    # to every call; without it the mask is fitted here, which is the expensive half.
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
    nusht_type2(θs, φs, Cs, lmax, backend = AutoBackend(); tol, ntrans, …) -> fs

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
            C = zeros(eltype(plan.F), lmax + 1, 2lmax + 1)
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
