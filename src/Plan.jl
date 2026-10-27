# The plan: every intermediate buffer and the NUFFT plans over the nodes, built once, so repeated
# transforms on one node set (filtering many fields, the iterations of `nusht_solve!`) allocate nothing
# and never re-plan. Every array and plan field is a type parameter, so one struct serves host and
# device arrays.

using FFTW: FFTW
using FastTransforms: FastTransforms

export AbstractNUSHTplan, AbstractScalarNUSHTplan, NUSHTplan, make_plan, close!, set_nodes!, plan_memory
export coefficient_size, allocate_coefficients
export AbstractNodeSet, FixedCountNodes, VariableCountNodes
export AbstractPointCount, FixedCount, VariableCount
export AbstractPlanDirections, SynthesisOnly, SynthesisAndAnalysis

# A NUFFT plan over `nodes = (x, y)` with its points set, through the bindings in NUFFT.jl; an
# `upsampfac` of `nothing` takes the library's own oversampling.
#
# `T` is the precision; `Z` is the element type of the non-uniform data, and a real one selects a
# real-data transform on a backend that has one (`_real_capable`). A backend without one ignores it.
@inline _make_nufft(backend, nodes::NTuple{2,AbstractVector}, type, n_modes, iflag, B, tol, ::Type{T},
                    nthreads, upsampfac, ::Type{Z} = Complex{T}) where {T,Z} =
    _nufft_makeplan(backend, nodes, type, n_modes, iflag, B, tol, T, Z, nthreads, upsampfac)

# The oversampling factor a NUFFT plan is built with: `nothing` for the library's own.
@inline _upsampfac(::Nothing) = nothing
@inline _upsampfac(σ::Real) = Float64(σ)

function _check_nodes(θ_nodes, φ_nodes)
    length(θ_nodes) == length(φ_nodes) || throw(DimensionMismatch(
        "θ_nodes has $(length(θ_nodes)) points and φ_nodes $(length(φ_nodes))"))
    return nothing
end

function _check_ntrans(ntrans::Integer)
    ntrans ≥ 1 || throw(ArgumentError("ntrans is the number of fields per call, at least 1; got $ntrans"))
    return Int(ntrans)
end

# Backend-generic zeroed buffer shaped like `ref` (host `Array` for CPU nodes, device array for GPU
# nodes) — used so a device node set yields device-resident plan buffers.
@inline _zeros_like(ref::AbstractArray, ::Type{S}, dims::Integer...) where {S} =
    fill!(similar(ref, S, dims...), zero(S))

# A copy of host array `a` on `ref`'s backend, for the small precomputed vectors a device plan
# broadcasts against its device buffers.
@inline _to_like(ref::AbstractArray, a::AbstractArray) = copyto!(similar(ref, eltype(a), size(a)...), a)

# A length-`n` array of `ref`'s type and element type, filled from `src` (converting if needed). Used
# when a node set's point count changes, so the new buffers keep the plan's declared types.
@inline _resized_like(ref::AbstractVector, src, n::Integer) =
    copyto!(similar(ref, eltype(ref), n), src)

"""
    AbstractPlanDirections

Which transform directions a plan is built for. A plan built for synthesis alone holds no type-1 NUFFT,
and anything needing the adjoint throws, naming the keyword to change. A backend allocates most of its
working memory when a transform executes, so the saving is the type-1 plan's own storage.
"""
abstract type AbstractPlanDirections end

"""
    SynthesisAndAnalysis <: AbstractPlanDirections

Build both directions — the default. Required by [`nusht_type1!`](@ref), [`nusht_analyze!`](@ref),
[`nusht_solve!`](@ref) and [`nusht_filter!`](@ref).
"""
struct SynthesisAndAnalysis <: AbstractPlanDirections end

"""
    SynthesisOnly <: AbstractPlanDirections

Build the synthesis (type-2) plan only. [`nusht_type2!`](@ref) works; anything needing the adjoint
throws and names the keyword to change.
"""
struct SynthesisOnly <: AbstractPlanDirections end

# The S-step is `plan_sph2fourier` and its adjoint, and its output is already the DFS bivariate Fourier
# series, so no grid synthesis or analysis plan is needed. FastTransforms plans its FFTs on the
# libfftw3 FFTW.jl loads, whose planner is process-global, so the plans are built under FFTW.jl's
# planner lock, the planner at one thread, and FastTransforms at the plan's thread count `n`.
function _build_sph_plans(Fslice, n::Integer)
    return FFTW.set_num_threads(1) do
        P = _ft_call(() -> FastTransforms.plan_sph2fourier(Fslice), n)
        # `P'` leaves `AdjointFTPlan.adjoint` undefined, which FastTransforms then resolves through an
        # `UndefRefError` on every `lmul!`. Name the parent explicitly.
        return (P, FastTransforms.AdjointFTPlan(P, P))
    end
end

"""
    _pool_sizes(B) -> Vector{Int}

Working batch sizes a solve may shrink to: powers of two up to `B`, plus `B` itself. Powers of two
bound the number of plan sets at `log2(B)+1` while never wasting more than a factor of two of transform
width, so a column retiring early costs at most one halving of unused work.
"""
function _pool_sizes(B::Integer)
    B <= 1 && return Int[]
    s = [1 << i for i in 0:floor(Int, log2(B))]
    s[end] == B || push!(s, Int(B))
    return s
end

"""
    _nufft_size_pool(backend, nufft_type2, nufft_type1, widths) -> pool

The store of reduced-width plan sets a batched solve narrows into, filled on demand through
[`_with_pool_entry`](@ref). Where the backend's plans share one type at every width
([`_width_polymorphic`](@ref)) it is an empty `Vector` typed from the plan's full-width pair. Otherwise
it holds one slot per width, a `Ref` whose element type `FlowTransformBindings.plan_type` derives
without building anything, so a width nobody uses is never built and never specialised.

Nothing is built here: only a solve that narrows pays for a width. An empty `widths` disables narrowing
and nothing is ever stored.
"""
function _nufft_size_pool(backend, nufft_type2, nufft_type1, widths::AbstractVector{Int})
    (isempty(widths) || _width_polymorphic(backend)) ||
        return _typed_width_slots(nufft_type2, nufft_type1, widths)
    E = @NamedTuple{k::Int, nufft_type2::typeof(nufft_type2), nufft_type1::typeof(nufft_type1)}
    pool = E[]
    sizehint!(pool, length(widths))   # final capacity is known, so filling it never regrows
    return pool
end

# One empty, concretely typed slot per width, the handle type read off `FTB.plan_type`.
function _typed_width_slots(nufft_type2::_FTBNUFFT, nufft_type1, widths::AbstractVector{Int})
    return ntuple(length(widths)) do i
        k = widths[i]
        H = _FTBNUFFT{FTB.plan_type(typeof(nufft_type2.plan), k)}
        E = NamedTuple{(:k, :nufft_type2, :nufft_type1),
                       Tuple{Int, H, nufft_type1 === nothing ? Nothing : H}}
        (k = k, pair = Ref{Union{Nothing,E}}(nothing))
    end
end

"""
    _pool_recipe(backend, nt, uf)

The build inputs a narrower plan set needs that cannot be recovered from a plan: the resolved NUFFT
backend, its thread count and its oversampling factor. Everything else — mode counts, tolerance, nodes,
realness — is read back off the plan when a width is built. `narrowable` is
[`_width_narrowable`](@ref) of the backend; how the widths are stored is [`_nufft_size_pool`](@ref)'s.
"""
_pool_recipe(backend, nt, uf) =
    (backend = backend, nt = Int(nt), uf = uf, narrowable = _width_narrowable(backend))

"""
    _build_width!(plan, k, E) -> E

Build the NUFFT plan pair for working width `k` as the pool entry type `E`, whose handle types the
library plan is asserted to (`_make_nufft_as`).

Called on a pool miss, which stores the entry in `plan.size_pool`, so it is not safe to call
concurrently on a shared plan — the same restriction a plan already carries, since a NUFFT plan cannot
be executed concurrently either.
"""
function _build_width!(plan, k::Integer, ::Type{E}) where {E<:NamedTuple}
    r = plan.pool_recipe
    T = real(eltype(plan.Fhat))
    θn, φn = _θnufft(plan), _φnodes(plan)
    # The strengths type comes off the plan's own buffer, so a narrower set is the same transform as the
    # full-width one, and it fixes the θ count: a real transform covers all `2lmax+3` wavenumbers and
    # stores half, a complex half-height one is built at the stored size.
    Z = eltype(_fbuf(plan))
    n_modes = Int64[Z <: Real ? 2plan.lmax + 3 : size(plan.Fhat, 1), plan.Nφ]
    n2 = _make_nufft_as(fieldtype(E, :nufft_type2), r.backend, (θn, φn), 2, n_modes, +1, k, plan.tol, T,
                        r.nt, r.uf, Z)
    n1 = if plan.nodes.nufft_type1 === nothing
        nothing                                   # mirror the plan's own directions
    elseif _nufft_share_directions(r.backend)
        _nufft_as_type1(n2)                       # same plan, opposite direction
    else
        _make_nufft(r.backend, (θn, φn), 1, n_modes, -1, k, plan.tol, T, r.nt, r.uf, Z)
    end::fieldtype(E, :nufft_type1)
    return E((Int(k), n2, n1))
end

"""
    _pool_lookup!(pool, k, build) -> entry

The plan pair for working width `k` in a `Vector` pool, built by `build(k, E)` as the pool's element
type `E` and stored on a miss.
"""
function _pool_lookup!(pool::AbstractVector{E}, k::Integer, build) where {E}
    @inbounds for e in pool
        e.k == k && return e
    end
    e = build(k, E)::E
    push!(pool, e)
    return e
end

# A pool of one typed slot per width (`_typed_width_slots`): the width that is used lands in its slot
# and stays there for the plan's life. The walk is over a tuple, so it unrolls. Slots of different
# widths hold different concrete types, so the entry is passed to `f`, which each arm then calls on one
# concrete type: the slot's on a hit, the type `_fill_slot!` builds as on a miss.
@inline function _with_pool_entry(f::F, pool::Tuple, k::Integer, build) where {F}
    slot = first(pool)
    slot.k == k || return _with_pool_entry(f, Base.tail(pool), k, build)
    e = slot.pair[]
    e === nothing || return f(e)
    return f(_fill_slot!(slot.pair, k, build))
end

# A solve narrows only to the pool's widths (`_fit_width`), so running off the end is a caller's width.
@noinline _with_pool_entry(f::F, ::Tuple{}, k::Integer, build) where {F} = throw(ArgumentError(
    "width $k is not one of the plan's NUFFT widths, the powers of two below its batch size and the " *
    "batch size"))

function _fill_slot!(slot::Base.RefValue{Union{Nothing,E}}, k::Integer, build) where {E}
    e = build(k, E)::E
    slot[] = e
    return e
end

# One element type, so there is no union to split and the plain lookup serves. `build` only passes
# through here, so it takes a type parameter to be specialized on.
_with_pool_entry(f::F, pool::AbstractVector, k::Integer, build::B) where {F,B} =
    f(_pool_lookup!(pool, k, build))

"""
    _pool_built(pool) -> Int

How many reduced widths currently hold plans, for either pool shape: an on-demand `Vector` holds only
the built widths, and a per-width slot holds `nothing` until its width is built.
"""
_pool_built(pool::AbstractVector) = length(pool)
_pool_built(pool::Tuple) = count(s -> s.pair[] !== nothing, pool)

# Free one slot's plans, if it ever held any, and empty it so `close!` stays idempotent.
function _release_width!(slot)
    e = slot.pair[]
    e === nothing && return nothing
    _nufft_destroy!(e.nufft_type2)
    _nufft_destroy!(e.nufft_type1)
    slot.pair[] = nothing
    return nothing
end

"""
    _sph_pool(Fslice, P, Padj, nslots, n)

One `Fslice` and one set of sphere plans per task for `nslots` tasks, for threading the per-column
sphere loops, the first entry the plan's own `(Fslice, P, Padj)` and the rest built at the plan's thread
count `n`. A FastTransforms plan is not safe to apply concurrently, and the loop's slice buffer is
shared, so both have to be replicated. Empty when there is nothing to thread, so a single-threaded plan
pays nothing.

Index the result by task chunk. A spawned task owns its chunk index for its whole life, while tasks
migrate between threads, so two tasks keyed by `Threads.threadid()` can share an entry.
"""
function _sph_pool(Fslice, P, Padj, nslots::Integer, n::Integer)
    pool = Tuple{typeof(Fslice), typeof(P), typeof(Padj)}[]
    nslots <= 1 && return pool
    push!(pool, (Fslice, P, Padj))
    for _ in 2:nslots
        push!(pool, (similar(Fslice), _build_sph_plans(Fslice, n)...))
    end
    return pool
end

"""
    AbstractNUSHTplan

Supertype of the plans over one process's points: the scalar plans ([`AbstractScalarNUSHTplan`](@ref))
and [`SpinNUSHTplan`](@ref).
"""
abstract type AbstractNUSHTplan end

"""
    AbstractScalarNUSHTplan{T,FE} <: AbstractNUSHTplan

Supertype of the scalar plans, [`NUSHTplan`](@ref) and [`RingNUSHTplan`](@ref), of precision `T` and
field element type `FE`, whose coefficients are in FastTransforms' `sph_mode` layout. The scalar
transforms, fits, analyses and filters take either.
"""
abstract type AbstractScalarNUSHTplan{T<:AbstractFloat,FE<:Number} <: AbstractNUSHTplan end

"""
    AbstractNodeSet

The point-dependent half of a plan: the `M` scattered nodes, the `(M, B)` strengths buffer, and the
NUFFT handles, which own the loaded point tables and so belong with the points.

Both concrete forms let the nodes **move** freely (that only rewrites array contents). They differ in
whether the *count* may change, which is the only thing that requires rebinding a field:
[`FixedCountNodes`](@ref) is immutable, [`VariableCountNodes`](@ref) is not. `make_plan`'s
`point_count` keyword picks one; the immutable form is the default.
"""
abstract type AbstractNodeSet end

"""
    AbstractPointCount

The `point_count` of [`make_plan`](@ref) and [`make_spin_plan`](@ref): [`FixedCount`](@ref) or
[`VariableCount`](@ref).
"""
abstract type AbstractPointCount end

"""
    FixedCount() <: AbstractPointCount

A plan over a [`FixedCountNodes`](@ref) set, the default.
"""
struct FixedCount <: AbstractPointCount end

"""
    VariableCount() <: AbstractPointCount

A plan over a [`VariableCountNodes`](@ref) set, whose count [`set_nodes!`](@ref) may change.
"""
struct VariableCount <: AbstractPointCount end

"""
    FixedCountNodes <: AbstractNodeSet

Immutable node set — the default. The nodes may move anywhere via [`set_nodes!`](@ref); only their
count is fixed, because it sizes the point-indexed buffers.
"""
struct FixedCountNodes{RV,SV,CT2,N1,N2} <: AbstractNodeSet
    θ_nodes::RV
    φ_nodes::RV
    θ_nufft::RV
    θ_shift::SV
    fbuf::CT2
    nufft_type2::N2
    nufft_type1::N1                     # `Nothing` for a synthesis-only plan; see `directions`
end

"""
    VariableCountNodes <: AbstractNodeSet

Node set whose three point-sized fields are assignable, so [`set_nodes!`](@ref) also accepts a
different number of points. The NUFFT handles stay `const`; setting their points updates the count in
place. Request one with `make_plan(…; point_count = VariableCount())`.
"""
mutable struct VariableCountNodes{RV,SV,CT2,N1,N2} <: AbstractNodeSet
    θ_nodes::RV
    φ_nodes::RV
    θ_nufft::RV
    θ_shift::SV
    fbuf::CT2
    const nufft_type2::N2
    const nufft_type1::N1               # `Nothing` for a synthesis-only plan; see `directions`
end

_node_set(::FixedCount, θ, φ, θn, θs, fbuf, p2, p1) = FixedCountNodes(θ, φ, θn, θs, fbuf, p2, p1)
_node_set(::VariableCount, θ, φ, θn, θs, fbuf, p2, p1) = VariableCountNodes(θ, φ, θn, θs, fbuf, p2, p1)

# Re-point the analysis plan with the nodes. A derived handle shares the plan just pointed, and a
# synthesis-only plan has none.
@inline _repoint_analysis!(::Nothing, θn, φn) = nothing
@inline _repoint_analysis!(p1, θn, φn) = _nufft_derived(p1) ? nothing : _nufft_setpts!(p1, θn, φn)

# The analysis plan, or `nothing` when the caller declined it — on every backend, so which calls a plan
# answers does not depend on which library is loaded.
_build_analysis(::SynthesisOnly, backend, θ, φ, n_modes, B, tol, ::Type{T}, nt, uf,
                ::Type{Z}, p2) where {T,Z} = nothing

function _build_analysis(::SynthesisAndAnalysis, backend, θ, φ, n_modes, B, tol, ::Type{T}, nt, uf,
                         ::Type{Z}, p2) where {T,Z}
    _nufft_share_directions(backend) && return _nufft_as_type1(p2)
    return _make_nufft(backend, (θ, φ), 1, n_modes, -1, B, tol, T, nt, uf, Z)
end

# Internal accessors for the point-dependent fields, so the rest of the package is written against
# one spelling regardless of which node-set form a plan carries.
@inline _θnodes(p) = p.nodes.θ_nodes
@inline _φnodes(p) = p.nodes.φ_nodes
@inline _θnufft(p) = p.nodes.θ_nufft
@inline _θshift(p) = p.nodes.θ_shift
@inline _fbuf(p) = p.nodes.fbuf
@inline _nufft2(p) = p.nodes.nufft_type2
@inline _nufft1(p) = _require_analysis(p.nodes.nufft_type1)

# A synthesis-only plan's missing analysis handle, reported with the keyword to change.
@inline _require_analysis(p1) = p1
_require_analysis(::Nothing) = throw(ArgumentError(
    "this plan was built with `directions = SynthesisOnly()` and has no analysis plan. Rebuild with " *
    "`directions = SynthesisAndAnalysis()` to use nusht_type1!, nusht_analyze!, nusht_solve! or " *
    "nusht_filter!."))

"""
    NUSHTplan{T}

Pre-computed plan for non-uniform spherical harmonic transforms at `M` scattered points, up to
degree `lmax`, transforming `B` co-located fields per call (`ntrans = B`).

Fields:
- `lmax`, `Nθ = lmax+1`, `Nφ = 2lmax+1`, `B` (batch size / NUFFT `ntrans`)
- `tol`: NUFFT accuracy tolerance
- `ntasks`: the plan's thread count, which its FastTransforms calls run at and its sphere-column loop
  divides among tasks
- `nodes`: the [`AbstractNodeSet`](@ref) holding `θ_nodes`, `φ_nodes` (colatitudes ∈ [0,π] and
  longitudes ∈ [0,2π) of the `M` points), the `(M, B)` strengths buffer `fbuf`, and the two NUFFT
  handles. [`set_nodes!`](@ref) re-points it.
- `C`: filter scratch, **empty until the first filter call** (`_filter_scratch`).
  `F`: the bivariate Fourier coefficients `P·C`. Both `(Nθ, Nφ, B)` with
  eltype `FE`
- `Fhat`: the complex mode array the NUFFT evaluates, assembled from `F` by
  [`_assemble_modes!`](@ref). The φ axis carries wavenumbers `-lmax…lmax` (`Nφ = 2lmax+1`), the θ axis
  `-(lmax+1)…lmax+1` (`2lmax+3`) — one further out, since an odd-order column of the coefficient array
  reaches θ-frequency `lmax+1`. Where the field is real *and* the NUFFT backend has a real-data
  transform, only the `kθ ≥ 0` half is stored (`lmax+2` rows) and that backend supplies the rest.
- `Fslice`: `(Nθ, Nφ)` scratch `Matrix` each batch slice is `copyto!`-ed through for the S-step.
  FastTransforms has no `Float32` sphere plan, so it stays double precision and the copy converts.
- `sph_plan`, `sph_plan_adj`: FastTransforms `plan_sph2fourier` (P) and its adjoint on a `(Nθ, Nφ)`
  slice. `P` alone is the whole S-step — its output is already the bivariate Fourier series, so
  synthesis is `P·C` → assemble → one NUFFT, with no equiangular grid in between.

`nodes.nufft_type2` is the type-2 handle (`iflag = +1`, synthesis N) and `nodes.nufft_type1` the
type-1 handle (`iflag = -1`, adjoint N†). `iflag = +1` supplies the reconstruction sign directly, so the
modes need no conjugate-transpose. The axis convention is `x = θ`, `y = φ`, and both mode axes are in
`FlowTransformBindings.CenteredModes()` order, the signed wavenumber ranges above, so nothing is shifted
per point.

FastTransforms runs on the host, so a `NUSHTplan` holds host arrays; [`make_plan`](@ref) builds a
[`RingNUSHTplan`](@ref) for device nodes.
"""
struct NUSHTplan{T<:AbstractFloat, FE<:Number, AT3<:AbstractArray{FE,3},
                 AT2<:AbstractMatrix, CT3<:AbstractArray{Complex{T},3},
                 ND<:AbstractNodeSet, SP, SPADJ, SPL, SZP, RCP,
                 FT, IV<:AbstractVector{Int}} <: AbstractScalarNUSHTplan{T,FE}
    lmax::Int
    Nθ::Int
    Nφ::Int
    B::Int
    tol::FT
    ntasks::Int                   # the plan's thread count; see _ft_threads
    nodes::ND                     # the point-dependent half; see AbstractNodeSet
    C::Base.RefValue{Union{Nothing,AT3}}   # filter scratch, empty until used; see _filter_scratch
    F::AT3
    Fhat::CT3
    Fslice::AT2
    sph_plan::SP
    sph_plan_adj::SPADJ           # sph_plan' (P'), stored to keep the adjoint alloc-free
    sph_pool::SPL                 # per-task slice + plans for the threaded column loops; see _sph_pool
    size_pool::SZP                # narrower plan sets, filled on demand; see _nufft_size_pool
    pool_recipe::RCP              # what building one needs that the plan cannot supply
    valid::Base.RefValue{Union{Nothing,IV}}   # the fit's packed slots, built by the first solve; see _valid_indices
end

"""
    _filter_scratch(plan) -> AbstractArray

The coefficient scratch filtering scales in, so the caller's array is left unmodified. Allocated on the
first filter call and reused after; a plan that never filters holds none.
"""
@inline function _filter_scratch(plan::AbstractScalarNUSHTplan)
    c = plan.C[]
    c === nothing || return c
    fresh = allocate_coefficients(plan)
    plan.C[] = fresh
    return fresh
end

# The element type of the fields a scalar plan synthesizes and analyses.
@inline _field_eltype(::AbstractScalarNUSHTplan{T,FE}) where {T,FE} = FE

"""
    coefficient_size(plan) -> NTuple{3,Int}

The shape of the coefficient array `plan` transforms: `(Nθ, Nφ, B)` for a scalar plan in
FastTransforms' layout, `(lmax+1, 2lmax+1, B)` for a spin plan.
"""
coefficient_size(plan::NUSHTplan) = (plan.Nθ, plan.Nφ, plan.B)

"""
    allocate_coefficients(plan) -> AbstractArray

A zeroed coefficient array for `plan`, of [`coefficient_size`](@ref)`(plan)`, in the plan's coefficient
element type and array type, so a device plan returns a device array.
"""
allocate_coefficients(plan::NUSHTplan) =
    _zeros_like(plan.F, eltype(plan.F), coefficient_size(plan)...)

"""
    plan_memory(plan) -> NamedTuple

Bytes held by each of the plan's own buffers, plus their `total`, so a caller can see where a plan's
memory goes and what `upsampfac`, `ntrans` or the element type do to it.

This counts what the plan allocates in Julia. A backend holding its oversampled grid in C (FINUFFT) is
invisible to any Julia-side accounting, `Base.summarysize` included — measure that as the resident set
of a process that builds one plan. An empty slot counts as zero.
"""
function plan_memory(plan::NUSHTplan)
    slot(r) = r[] === nothing ? 0 : Base.summarysize(r[])
    C      = slot(plan.C)
    F      = Base.summarysize(plan.F)
    Fhat   = Base.summarysize(plan.Fhat)
    Fslice = Base.summarysize(plan.Fslice)
    fbuf   = Base.summarysize(_fbuf(plan))
    nodes  = Base.summarysize(_θnodes(plan)) + Base.summarysize(_φnodes(plan))
    pool   = Base.summarysize(plan.size_pool)
    # The first entry of a nonempty pool is the plan's own slice and plans, counted above.
    sph    = sum(Base.summarysize, view(plan.sph_pool, 2:length(plan.sph_pool)); init = 0)
    valid  = slot(plan.valid)
    return (; C, F, Fhat, Fslice, fbuf, nodes, size_pool = pool, sph_pool = sph, valid,
            total = C + F + Fhat + Fslice + fbuf + nodes + pool + sph + valid)
end

"""
    make_plan([FE = float(eltype(θ_nodes)),] θ_nodes, φ_nodes, lmax; execution = AutoBackend(), tol=1e-8, ntrans=1, …)

Construct a scalar plan for `M` scattered points at colatitudes `θ_nodes ∈ [0,π]` and longitudes
`φ_nodes ∈ [0,2π)`, up to spherical harmonic degree `lmax`, transforming `ntrans` co-located fields
per call: a [`NUSHTplan`](@ref) for host nodes, a [`RingNUSHTplan`](@ref) for device nodes. Builds the
NUFFT plans over the nodes once; [`close!`](@ref) frees them.

`FE` is the field element type, positional as it is for `zeros(T, …)`. A real `FE` asserts the field
values are real, so the mode array is conjugate-symmetric in `kθ`; on a NUFFT backend with a real-data
transform (either FlowTransformBindings library) only the `kθ ≥ 0` half is built and real strengths
come back. A complex `FE` builds the full array. Both are the same spherical harmonic transform.

Keyword arguments:
- `execution`: where the plan runs, a `ComputationalBackends` backend. `SerialBackend()` runs it on one
  thread and `ThreadedBackend()` on `Threads.nthreads()`; `AutoBackend()`, the default, runs it on the
  host at `nthreads`. `GPUBackend(b)` puts the nodes, and so the plan, in `b`'s memory (with
  KernelAbstractions and GPUArraysCore), as device nodes passed under any other local backend do.
  `DistributedBackend(inner)` divides the points among the worker
  processes and `MPIBackend(inner)` among the ranks of its communicator (with Distributed, MPI), each
  share's plan built under `inner` with the other keywords; both plans take and return the whole field.
- `tol`: NUFFT accuracy tolerance, raised to `eps(T)`.
- `nufft`: `SpectralBackends.AutoSpectralBackend()` (default),
  `FlowTransformBindings.NonuniformFFTsBackend()`, `FlowTransformBindings.FINUFFTBackend()` or
  `SpectralBackends.DirectSumSpectralBackend()`.
- `ntrans`: batch size `B` — transform `B` co-located fields (same nodes) per call.
- `nthreads`: the thread count the NUFFT and FastTransforms' OpenMP regions run at, and the per-column
  sphere steps divide among tasks at; `nothing` or `0` takes `Threads.nthreads()`. Inside an
  OhMyThreads farm, and in the tasks of the per-column loop, FastTransforms runs on one thread. An
  explicit count takes precedence over the one
  `execution` sets, except under `SerialBackend()`, which runs on one thread and raises an
  `ArgumentError` for any count other than `1`. FINUFFT takes any count, while NonuniformFFTs
  parallelises over Julia's own threads and so reaches only `1` and `Threads.nthreads()` — a count it
  cannot deliver is an error.
- `upsampfac`: the NUFFT's oversampling factor; `nothing` takes the library's own.
- `point_count`: [`FixedCount`](@ref) (default) or [`VariableCount`](@ref), which lets
  [`set_nodes!`](@ref) change the point count.
- `directions`: [`SynthesisAndAnalysis`](@ref) (default) or [`SynthesisOnly`](@ref).
"""
make_plan(::Type{FE}, θ_nodes, φ_nodes, lmax;
          execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
          kwargs...) where {FE<:Number} =
    _make_plan(FE, θ_nodes, φ_nodes, lmax, execution; kwargs...)

# The plan on this process over the nodes as given: host nodes build a host plan, device nodes a
# device-resident one.
function _build_plan(
    ::Type{FE},
    θ_nodes,
    φ_nodes,
    lmax;
    tol = 1e-8,
    ntrans::Integer = 1,
    nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    point_count::AbstractPointCount = FixedCount(),
    directions::AbstractPlanDirections = SynthesisAndAnalysis(),
    nthreads::Union{Nothing,Integer} = nothing,
    upsampfac::Union{Nothing,Real} = nothing,
) where {FE<:Number}
    _check_nodes(θ_nodes, φ_nodes)
    B = _check_ntrans(ntrans)
    T = real(FE)
    T <: AbstractFloat ||
        throw(ArgumentError("field element type must be a float or complex float, got $FE"))
    realfield = FE <: Real

    Nθ = lmax + 1
    Nφ = 2lmax + 1
    M = length(θ_nodes)

    # Nodes keep their input array type (host `Vector` or device array), eltype coerced to `T`; every
    # buffer is allocated `similar` to the nodes, so a device node set yields a device-resident plan.
    θ = T.(θ_nodes)
    φ = T.(φ_nodes)

    # `plan_sph2fourier` already yields the DFS bivariate Fourier series, so the NUFFT's mode array is
    # assembled straight from it (`_assemble_modes!`). Both axes are signed and centered,
    # `FTB.CenteredModes()`, with no offset to undo per point. θ reaches lmax+1 where φ reaches lmax:
    # the supernumerary slots of the square coefficient array carry degrees up to `lmax+|m|`, and an
    # odd-order column's last row is the sine at frequency `lmax+1`.
    nub = _resolve_nufft(nufft, FE)
    # A real field's mode array is Hermitian, so only `kθ ≥ 0` is stored, `lmax+2` of the `2lmax+3` rows:
    # the deconvolution and the FFT halve, and the interpolation is unchanged.
    #
    # A backend with a real-data transform (`_real_capable`) is handed the half of a transform built for
    # the full θ axis and supplies the rest itself, with real strengths, so the spreading halves too and
    # the forward needs no weights. Otherwise the transform is half-height and complex, its centered rows
    # labelled `kθ - Nkstore÷2`, and the conjugate half enters explicitly: `θ_shift` undoes the labelling
    # per point and `_fold_weights!` doubles every `kθ > 0` row.
    Nk = 2lmax + 3
    r2c  = realfield && _real_capable(nub)
    fold = realfield
    _warn_if_directsum(nufft, nub, M, Nk * Nφ)
    ZS = r2c ? T : Complex{T}                 # element type of the non-uniform data
    Nkstore = fold ? Nk ÷ 2 + 1 : Nk          # `lmax+2` folded
    F    = _zeros_like(θ, FE, Nθ, Nφ, B)
    C    = Base.RefValue{Union{Nothing,typeof(F)}}(nothing)   # allocated on first use; see `_filter_scratch`
    Fhat = _zeros_like(θ, Complex{T}, Nkstore, Nφ, B)
    fbuf = _zeros_like(θ, ZS, M, B)
    θ_shift = (fold && !r2c) ?
        _to_like(θ, Complex{T}.(cis.(T(Nkstore ÷ 2) .* θ))) : nothing

    # FastTransforms runs on one dense `(Nθ, Nφ)` `Matrix` of Float64 or ComplexF64 (it has no Float32
    # sphere plans). Each batch slice is copied through `Fslice`, converting precision; every other
    # buffer runs at `FE`.
    Fslice = zeros(realfield ? Float64 : ComplexF64, Nθ, Nφ)
    ntasks = _task_count(nthreads)
    sph_plan, sph_plan_adj = _build_sph_plans(Fslice, ntasks)
    # Replicated only when there is something to thread, so a single-threaded plan pays no build cost
    # and no memory. Capped at `B`: more tasks than columns cannot help.
    sph_pool = _sph_pool(Fslice, sph_plan, sph_plan_adj, min(ntasks, B), ntasks)

    # iflag +1 for synthesis (type 2): reconstruction uses the +i (inverse-DFT) sign, so the raw
    # FFT modes need no conjugation — an axis swap takes the place of a conjugate-transpose
    # (conj(c)·e^{-ikx} = c·e^{+ikx}). type 1 (−1) is the exact adjoint.
    tol64 = Float64(tol)
    # A real transform is built for the full θ axis and stores its half; a complex half-height one is
    # built at the stored size.
    n_modes = Int64[r2c ? Nk : Nkstore, Nφ]
    nt = nthreads === nothing ? 0 : Int(nthreads)
    uf = _upsampfac(upsampfac)
    # Where a backend's plan carries no direction, one object serves both and the second is a handle
    # onto it — that halves the oversampled grid and the sorted point copy the plan owns, and its
    # points are already set.
    nufft_type2 = _make_nufft(nub, (θ, φ), 2, n_modes, +1, B, tol64, T, nt, uf, ZS)
    # The type-1 plan: the type-2 plan itself on a backend whose plan serves both directions, and none
    # for a synthesis-only plan.
    nufft_type1 = _build_analysis(directions, nub, θ, φ, n_modes, B, tol64, T, nt, uf, ZS, nufft_type2)
    # A scalar plan hands the NUFFT the colatitudes unchanged, so `θ_nufft` aliases `θ_nodes` and costs
    # no extra storage; a spin plan negates them and owns a separate array.
    pool_recipe = _pool_recipe(nub, nt, uf)
    size_pool = _nufft_size_pool(nub, nufft_type2, nufft_type1,
                                 pool_recipe.narrowable ? _pool_sizes(B) : Int[])

    nodes = _node_set(point_count, θ, φ, θ, θ_shift, fbuf, nufft_type2, nufft_type1)

    IV = typeof(similar(θ, Int, 0))
    return NUSHTplan{T, FE, typeof(F), typeof(Fslice), typeof(Fhat), typeof(nodes),
                     typeof(sph_plan), typeof(sph_plan_adj),
                     typeof(sph_pool), typeof(size_pool), typeof(pool_recipe), typeof(tol64), IV}(
        lmax, Nθ, Nφ, B, tol64, ntasks, nodes, C, F, Fhat, Fslice,
        sph_plan, sph_plan_adj, sph_pool, size_pool, pool_recipe,
        Base.RefValue{Union{Nothing,IV}}(nothing),
    )
end

# Element type positional, as for `zeros(T, …)`: it carries precision and realness together. Omitted,
# precision comes from the nodes and the field is real.
make_plan(θ_nodes, φ_nodes, lmax; kwargs...) =
    make_plan(float(eltype(θ_nodes)), θ_nodes, φ_nodes, lmax; kwargs...)

# Backends whose plan is built on this process from the nodes as given.
const _HostExecution = Union{ComputationalBackends.AbstractSerialBackend,
                             ComputationalBackends.AbstractThreadedBackend,
                             ComputationalBackends.AbstractAutoBackend}

"""
    _plan_threads(execution, nthreads) -> Union{Nothing,Integer}

The thread count a plan built under `execution` runs at, given the caller's `nthreads`: `1` under
`SerialBackend`, which refuses any other count; `nthreads` under `ThreadedBackend` when given and
`Threads.nthreads()` when not; `nthreads` itself under any other backend.
"""
_plan_threads(::ComputationalBackends.AbstractSerialBackend, nthreads) =
    (nthreads === nothing || nthreads == 1) ? 1 : throw(ArgumentError(
        "execution = SerialBackend() runs a plan on one thread; got nthreads = $nthreads"))
_plan_threads(::ComputationalBackends.AbstractThreadedBackend, nthreads) =
    nthreads === nothing ? Threads.nthreads() : nthreads
_plan_threads(::ComputationalBackends.AbstractExecutionBackend, nthreads) = nthreads

# A thread count, `nothing` and `0` taking every Julia thread.
@inline _task_count(nthreads) = (nthreads === nothing || nthreads == 0) ? Threads.nthreads() : Int(nthreads)

# `make_plan` under `execution`. NUFSHTKernelAbstractionsExt adds `GPUBackend`, NUFSHTDistributedExt
# `DistributedBackend` and NUFSHTMPIExt `MPIBackend`.
_make_plan(::Type{FE}, θ, φ, lmax, execution::_HostExecution; nthreads = nothing, kwargs...) where {FE} =
    _build_plan(FE, θ, φ, lmax; nthreads = _plan_threads(execution, nthreads), kwargs...)
_make_plan(::Type{FE}, θ, φ, lmax, e::ComputationalBackends.AbstractExecutionBackend; kwargs...) where {FE} =
    _execution_unavailable(e, "a plan")

"""
    set_nodes!(plan, θ_nodes, φ_nodes) -> plan

Move a plan's nodes to new positions, reusing every structure fixed by `(lmax, B, T)` — the
FastTransforms sphere plans, the FFTW plans and all coefficient buffers. The NUFFT plans' point tables
are rebuilt, and any reduced-width plan pair a batched solve cached is released, to be rebuilt at the
new nodes if a later solve narrows again.

The points may move anywhere. Changing how *many* there are additionally needs the plan's
point-indexed buffers to be replaced, which requires `make_plan(…; point_count = VariableCount())`; a
plan built with the default [`FixedCountNodes`](@ref) throws a `DimensionMismatch`.

With the count unchanged and no reduced-width plans cached, this rewrites array contents only.
"""
function set_nodes!(plan::AbstractNUSHTplan, θ_nodes, φ_nodes)
    _check_nodes(θ_nodes, φ_nodes)
    _set_nodes!(plan.nodes, θ_nodes, φ_nodes)
    _sync_θshift!(_θshift(plan), _θnufft(plan), _shift_offset(plan))
    _nufft_setpts!(_nufft2(plan), _θnufft(plan), _φnodes(plan))
    # A derived handle shares the plan just pointed, which already holds these points.
    _repoint_analysis!(plan.nodes.nufft_type1, _θnufft(plan), _φnodes(plan))
    _close_pool!(plan)
    return plan
end

# `θ_nufft` either aliases `θ_nodes` (scalar plan, nothing to do) or is its negation (spin plan).
@inline function _sync_θnufft!(nd)
    nd.θ_nufft === nd.θ_nodes || (nd.θ_nufft .= .-nd.θ_nodes)
    return nd
end

# `θ_shift` moves with the nodes, at an offset fixed by the bandlimit, which the plan holds and the node
# set does not.
@inline _sync_θshift!(::Nothing, _θ, _N0) = nothing
@inline _sync_θshift!(s::AbstractVector, θ, N0) = (s .= cis.(N0 .* θ); s)

"""
    _valid_indices(proto, lmax) -> AbstractVector{Int}

Linear positions, within one coefficient column, of the `(lmax+1)^2` slots holding degrees
`l ≤ lmax`, in the plan's array type. The index rule is FastTransforms' own (`sphones`): column 1
carries `m = 0` for every degree, and the column pair `2j, 2j+1` carries `m = ∓j` for `l = j … lmax`, so
it occupies only its first `lmax+1-j` rows; the other `lmax(lmax+1)` slots hold the supernumerary
degrees `lmax < l ≤ lmax+|m|`.

The solver's vectors are packed to these, `(K, B)` with `K = (lmax+1)^2`, so every reduction and update
of the iteration runs over the packed length.
"""
function _valid_indices(proto, lmax::Integer)
    Nθ = lmax + 1
    idx = Vector{Int}(undef, (lmax + 1)^2)
    k = 0
    @inbounds for i in 1:Nθ                       # column 1 is m = 0, every degree
        idx[k += 1] = i
    end
    @inbounds for j in 1:lmax, c in (2j, 2j + 1)  # column pair 2j, 2j+1 is m = ∓j, degrees j…lmax
        base = (c - 1) * Nθ
        for i in 1:(Nθ - j)
            idx[k += 1] = base + i
        end
    end
    k == length(idx) || throw(AssertionError("valid-slot count $k ≠ $(length(idx))"))
    return _to_like(proto, idx)
end

# Gather a packed `(K, B)` view out of a full `(Nθ, Nφ, B)` array, and scatter it back with the
# supernumerary slots zeroed. `n` bounds the columns, matching the solver's compaction.
function _pack_coeffs!(dst, src, idx, srclen::Integer, n::Integer)
    K = length(idx)
    @inbounds for b in 1:n
        so = (b - 1) * srclen
        do_ = (b - 1) * K
        @simd for t in 1:K
            dst[do_ + t] = src[so + idx[t]]
        end
    end
    return dst
end

function _unpack_coeffs!(dst, src, idx, dstlen::Integer, n::Integer)
    K = length(idx)
    @inbounds for b in 1:n
        do_ = (b - 1) * dstlen
        so = (b - 1) * K
        @simd for i in 1:dstlen
            dst[do_ + i] = zero(eltype(dst))
        end
        @simd for t in 1:K
            dst[do_ + idx[t]] = src[so + t]
        end
    end
    return dst
end

# Centered order labels a length-`n` axis from `-(n÷2)`, the offset to undo, read off the mode buffer so
# it matches the buffer.
@inline _shift_offset(plan::NUSHTplan{T}) where {T} = T(size(plan.Fhat, 1) ÷ 2)

# Whether the plan's mode array holds the `kθ ≥ 0` half (`lmax+2` rows) of the centered
# `-(lmax+1) … lmax+1` (`2lmax+3` rows): a real field's array is Hermitian, so its plan stores the half on
# every backend, and supplies the other.
@inline _folded(::NUSHTplan{T,FE}) where {T,FE} = FE <: Real

# Same count: rewrite contents, no allocation, works on either node-set form.
function _set_nodes_inplace!(nd, θ_nodes, φ_nodes)
    copyto!(nd.θ_nodes, θ_nodes)
    copyto!(nd.φ_nodes, φ_nodes)
    return _sync_θnufft!(nd)
end

function _set_nodes!(nd::FixedCountNodes, θ_nodes, φ_nodes)
    M = length(nd.θ_nodes)
    length(θ_nodes) == M || throw(DimensionMismatch(
        "this plan holds $M points and its node set has a fixed count, but got $(length(θ_nodes)). " *
        "Build it with point_count = VariableCount() to allow the count to change."))
    return _set_nodes_inplace!(nd, θ_nodes, φ_nodes)
end

function _set_nodes!(nd::VariableCountNodes, θ_nodes, φ_nodes)
    M = length(θ_nodes)
    M == length(nd.θ_nodes) && return _set_nodes_inplace!(nd, θ_nodes, φ_nodes)
    aliased = nd.θ_nufft === nd.θ_nodes
    B = size(nd.fbuf, 2)
    nd.θ_nodes = _resized_like(nd.θ_nodes, θ_nodes, M)
    nd.φ_nodes = _resized_like(nd.φ_nodes, φ_nodes, M)
    nd.θ_nufft = aliased ? nd.θ_nodes : similar(nd.θ_nodes, eltype(nd.θ_nodes), M)
    nd.fbuf = _zeros_like(nd.θ_nodes, eltype(nd.fbuf), M, B)
    return _sync_θnufft!(nd)
end

# The narrower plan sets `_build_width!` caches own NUFFT plans of their own, so `close!` reaches them;
# a spin plan has no pool.
_close_pool!(::AbstractNUSHTplan) = nothing
_close_pool!(plan::NUSHTplan) = _close_pool!(plan.size_pool)

# A `Vector` pool is emptied; a `Tuple` of typed slots has its slots cleared, which keeps `close!`
# idempotent there too.
function _close_pool!(pool::AbstractVector)
    for e in pool
        _nufft_destroy!(e.nufft_type2)
        _nufft_destroy!(e.nufft_type1)
    end
    empty!(pool)
    return nothing
end

function _close_pool!(pool::Tuple)
    for slot in pool
        _release_width!(slot)
    end
    return nothing
end

"""
    close!(plan::NUSHTplan)

Free every NUFFT plan the plan owns, now: its own and any narrower set `_build_width!` cached for a
compacted solve. A plan left unreachable has its host NUFFT plans freed by collection, and its
cuFINUFFT plans only by this. Safe to call more than once.
"""
function close!(plan::AbstractNUSHTplan)
    _nufft_destroy!(_nufft2(plan))
    _nufft_destroy!(plan.nodes.nufft_type1)
    _close_pool!(plan)
    return nothing
end

# One line: the default display recurses into the stored FFTW plan, whose printer (`fftw_sprint_plan`)
# can segfault on a plan whose C state is gone (after `close!`, or across a `Distributed` worker).
Base.show(io::IO, plan::NUSHTplan{T}) where {T} =
    print(io, "NUSHTplan{", T, "}(lmax=", plan.lmax, ", M=", length(_θnodes(plan)), ", B=", plan.B, ", tol=", plan.tol, ")")
