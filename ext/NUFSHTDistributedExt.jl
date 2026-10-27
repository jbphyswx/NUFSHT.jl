"""
    NUFSHTDistributedExt

Coarse-grained farming of *independent* NUFSHT problems across `Distributed` **worker processes**,
selected by `execution = ComputationalBackends.DistributedBackend()`. NUFFT plans hold library state that
cannot be serialized, so this extension implements the **node-set** entry points (`nusht_type2`,
`nusht_solve`): each worker builds its own plan from the node set it is given, runs the transform,
frees the plan, and returns the result. Loaded by `using Distributed`; workers need NUFSHT and the NUFFT
library their plans use (`@everywhere using NUFSHT, NonuniformFFTs`), or `Auto` gives them direct
summation.

A worker runs one problem at a time, its plan built with `execution = local_backend(execution)`: one
thread under `DistributedBackend()` (inner `SerialBackend`), the worker's `Threads.nthreads()` under
`DistributedBackend(ThreadedBackend())` — so `addprocs(n; exeflags = "-t k")` gives each problem `k`
threads — or an explicit `nthreads`. FastTransforms runs at the same count
(`FlowTransformBindings.FASTTRANSFORMS_THREADS`). Each worker is sent only its own problem's nodes and
data. With no worker processes the problems run one after another on the calling process.

A *single* transform divides among the workers through
`make_plan(FE, θ, φ, lmax; execution = DistributedBackend())` or `make_spin_plan(…)`, a
[`DistributedNUSHTplan`](@ref): a batch of at least as many fields as workers by its fields, each worker
holding a plan over every point for a block of them, and otherwise by its points.
"""
module NUFSHTDistributedExt

using NUFSHT: NUFSHT
using ComputationalBackends: ComputationalBackends
using Distributed: Distributed
using FlowTransformBindings: FlowTransformBindings as FTB

# `work(item)` for each item on the worker processes when there are any, else one after another here.
# `pmap` sends each call its own item.
_farm(work, items) = Distributed.nprocs() == 1 ? map(work, items) : Distributed.pmap(work, items)

# A plan over one problem's nodes under `inner`, built and run at the thread count `inner` gives on the
# process that runs it.
function _problem(run, θ, φ, lmax, inner, kw)
    n = _part_threads(inner, get(kw, :nthreads, nothing))
    plan = _at_part_threads(() -> NUFSHT.make_plan(θ, φ, lmax; kw..., execution = inner, nthreads = n), n)
    try
        return _at_part_threads(() -> run(plan), n)
    finally
        NUFSHT.close!(plan)
    end
end

function NUFSHT._nodeset_type2(θs, φs, Cs, lmax, execution::ComputationalBackends.AbstractDistributedBackend;
                               tol = 1e-8, kwargs...)
    length(θs) == length(φs) == length(Cs) || NUFSHT._unequal_lengths("θs, φs and Cs")
    inner = ComputationalBackends.local_backend(execution)
    kw = (; tol, kwargs...)
    NUFSHT._plan_threads(inner, get(kw, :nthreads, nothing))   # a count `inner` refuses raises here
    return _farm(collect(zip(θs, φs, Cs))) do (θ, φ, C)
        _problem(θ, φ, lmax, inner, kw) do plan
            f = zeros(NUFSHT._field_eltype(plan), length(θ))
            NUFSHT.nusht_type2!(f, C, plan)
            f
        end
    end
end

function NUFSHT._nodeset_solve(θs, φs, fs, lmax, execution::ComputationalBackends.AbstractDistributedBackend;
                               tol = 1e-8, rtol = 1e-6, maxiter = 500, kwargs...)
    length(θs) == length(φs) == length(fs) || NUFSHT._unequal_lengths("θs, φs and fs")
    inner = ComputationalBackends.local_backend(execution)
    kw = (; tol, kwargs...)
    NUFSHT._plan_threads(inner, get(kw, :nthreads, nothing))
    return _farm(collect(zip(θs, φs, fs))) do (θ, φ, f)
        _problem(θ, φ, lmax, inner, kw) do plan
            C = zeros(NUFSHT._field_eltype(plan), lmax + 1, 2lmax + 1)
            NUFSHT.nusht_solve!(C, f, plan; rtol = rtol, maxiter = maxiter)
            C
        end
    end
end

# ── One transform divided among the worker processes ─────────────────────────────────────────────────
# By fields, each worker transforms, analyses and fits its own block of the batch alone. By points, the
# synthesis and the adjoint of a block need no other block, and the fit runs `nusht_solve!`'s recurrence on
# every worker at once, its two sums over points sent to the caller, which adds the workers' parts in
# worker order and returns the same total to each, so every worker runs the same recurrence.

# Which transforms a plan's workers hold: `nothing` for a scalar plan, the spin for a spin plan.
struct _Scalar end
struct _Spin
    s::Int
end

"""
    DistributedNUSHTplan

A scalar or spin NUSHT plan divided among the worker processes, each worker's plan and least-squares
workspace held on that worker: with `columns` each worker's plan covers every point for the block of the
`B` fields in `blocks`, otherwise the block of the `npts` points. The coefficients and fields are the
caller's, whole.
"""
struct DistributedNUSHTplan{FE<:Number, CE<:Number, K, VD<:AbstractVector{Distributed.Future},
                            VI<:AbstractVector{<:Integer}, VU<:AbstractVector{UnitRange{Int}}, TL<:Real}
    kind::K
    parts::VD
    workers::VI
    blocks::VU
    columns::Bool
    npts::Int
    lmax::Int
    B::Int
    coefficient_size::NTuple{3,Int}
    tol::TL
end

Base.show(io::IO, p::DistributedNUSHTplan{FE}) where {FE} =
    print(io, "DistributedNUSHTplan{", FE, "}(lmax=", p.lmax, ", M=", p.npts, ", ntrans=", p.B, " on ",
          length(p.workers), " workers by ", p.columns ? "field" : "point", ")")

_block(k::Int, nw::Int, n::Int) = (div((k - 1) * n, nw) + 1):div(k * n, nw)

# The thread count a worker runs its share at under the inner backend and the caller's `nthreads`, read
# on that worker.
_part_threads(inner, nthreads) = NUFSHT._task_count(NUFSHT._plan_threads(inner, nthreads))

# `f()` at the part's FastTransforms thread count.
_at_part_threads(f, nthreads::Int) = Base.ScopedValues.with(f, FTB.FASTTRANSFORMS_THREADS => nthreads)

_build(::_Scalar, ::Type{FE}, θ, φ, lmax; kw...) where {FE} = NUFSHT.make_plan(FE, θ, φ, lmax; kw...)
_build(k::_Spin, ::Type{FE}, θ, φ, lmax; kw...) where {FE} = NUFSHT.make_spin_plan(FE, θ, φ, lmax, k.s; kw...)

function _part_plan(kind, ::Type{FE}, θ, φ, lmax, inner, kwargs) where {FE}
    n = _part_threads(inner, get(kwargs, :nthreads, nothing))
    plan = _at_part_threads(n) do
        _build(kind, FE, θ, φ, lmax; kwargs..., execution = inner, nthreads = n)
    end
    return (; plan, ws = FTB.LSMRWorkspace(plan), nthreads = n)
end
_part_info(f) = (p = fetch(f).plan; (NUFSHT.coefficient_size(p), p.tol))

# The workers and their blocks: by field when the batch has at least as many fields as there are workers,
# by point otherwise. With no worker processes, the caller holds every point.
function _divide(M::Int, B::Int)
    nw = min(Distributed.nworkers(), M)
    ws = Distributed.workers()[1:nw]
    columns = Distributed.nprocs() > 1 && B >= nw
    n = columns ? B : M
    return ws, [_block(k, nw, n) for k in 1:nw], columns
end

function _distributed_plan(kind, ::Type{FE}, ::Type{CE}, θ_nodes, φ_nodes, lmax, execution; kwargs...) where {FE,CE}
    length(θ_nodes) == length(φ_nodes) || throw(DimensionMismatch("θ and φ must have equal length"))
    M = length(θ_nodes)
    kw = NamedTuple(kwargs)
    B = Int(get(kw, :ntrans, 1))
    inner = ComputationalBackends.local_backend(execution)
    # A count `inner` refuses raises here, before any worker builds.
    NUFSHT._plan_threads(inner, get(kw, :nthreads, nothing))
    ws, blocks, columns = _divide(M, B)
    parts = map(enumerate(ws)) do (k, w)
        b = blocks[k]
        θ, φ = columns ? (collect(θ_nodes), collect(φ_nodes)) : (collect(θ_nodes[b]), collect(φ_nodes[b]))
        Distributed.remotecall(_part_plan, w, kind, FE, θ, φ, lmax, inner,
                               columns ? merge(kw, (; ntrans = length(b))) : kw)
    end
    infos = Vector{Tuple{NTuple{3,Int},Float64}}(undef, length(ws))
    @sync for (k, w) in enumerate(ws)
        @async infos[k] = Distributed.remotecall_fetch(_part_info, w, parts[k])
    end
    csize, tol = first(infos)
    return DistributedNUSHTplan{FE,CE,typeof(kind),typeof(parts),typeof(ws),typeof(blocks),typeof(tol)}(
        kind, parts, ws, blocks, columns, M, Int(lmax), B, (csize[1], csize[2], B), tol)
end

# `make_plan(FE, θ, φ, lmax; execution::DistributedBackend, kwargs...)` and its spin counterpart: each
# worker's plan built with `kwargs` under `local_backend(execution)`. A worker runs its NUFFT and
# FastTransforms at that backend's thread count there, as `make_plan` sets it: 1 under `SerialBackend()`,
# else `nthreads` when given and the worker's `Threads.nthreads()` when not. The workers need NUFSHT and
# the NUFFT library their plans use loaded.
NUFSHT._make_plan(::Type{FE}, θ, φ, lmax, execution::ComputationalBackends.AbstractDistributedBackend;
                  kwargs...) where {FE<:Number} =
    _distributed_plan(_Scalar(), FE, FE, θ, φ, lmax, execution; kwargs...)

NUFSHT._make_spin_plan(::Type{FE}, θ, φ, lmax, s, execution::ComputationalBackends.AbstractDistributedBackend;
                       kwargs...) where {FE<:Number} =
    _distributed_plan(_Spin(Int(s)), FE, Complex{real(FE)}, θ, φ, lmax, execution; kwargs...)

NUFSHT.allocate_coefficients(p::DistributedNUSHTplan{FE,CE}) where {FE,CE} = zeros(CE, p.coefficient_size...)
NUFSHT.coefficient_size(p::DistributedNUSHTplan) = p.coefficient_size

_part_close(f) = (NUFSHT.close!(fetch(f).plan); nothing)
function NUFSHT.close!(p::DistributedNUSHTplan)
    @sync for (k, w) in enumerate(p.workers)
        @async Distributed.remotecall_wait(_part_close, w, p.parts[k])
    end
    return nothing
end

# The part of a whole-field or whole-coefficient array a worker takes: rows `block` of every column, or
# with `columns` the fields `block` along the last axis; and the reverse.
_take(A::AbstractVector, block, columns::Bool) = A[block]
_take(A::AbstractArray{<:Any,N}, block, columns::Bool) where {N} =
    columns ? A[ntuple(_ -> Colon(), N - 1)..., block] : A[block, ntuple(_ -> Colon(), N - 1)...]
_put!(A::AbstractVector, block, v, columns::Bool) = (A[block] .= v; A)
_put!(A::AbstractArray{<:Any,N}, block, v, columns::Bool) where {N} =
    columns ? (A[ntuple(_ -> Colon(), N - 1)..., block] .= v; A) : (A[block, ntuple(_ -> Colon(), N - 1)...] .= v; A)

_to_host(A::Array) = A
_to_host(A::AbstractArray) = Array(A)

# The transforms a worker's plan runs, scalar or spin by its type.
_synth!(out, C, plan::NUFSHT.AbstractScalarNUSHTplan, ::Nothing) = NUFSHT.nusht_type2!(out, C, plan)
_synth!(out, C, plan::NUFSHT.AbstractScalarNUSHTplan, filter) = NUFSHT.nusht_synthesize!(out, C, filter, plan)
_synth!(out, sf, plan::NUFSHT.SpinNUSHTplan, ::Nothing) = NUFSHT.nusht_type2_spin!(out, sf, plan)
_adjoint!(C, f, plan::NUFSHT.AbstractScalarNUSHTplan) = NUFSHT.nusht_type1!(C, f, plan)
_adjoint!(sf, f, plan::NUFSHT.SpinNUSHTplan) = NUFSHT.nusht_type1_spin!(sf, f, plan)

# A worker's field: its points' rows of every field, or every point of its fields, shaped like the
# caller's whole field.
function _part_field(p, nrows::Int, ncols::Int, batched::Bool)
    E = _field_eltype(p)
    return batched ? zeros(E, nrows, ncols) : zeros(E, nrows)
end
_field_eltype(p::NUFSHT.AbstractScalarNUSHTplan) = NUFSHT._field_eltype(p)
_field_eltype(p::NUFSHT.SpinNUSHTplan{T}) where {T} = NUFSHT._hermitian_domain(p) ? T : Complex{T}

function _part_synthesis(f, C, filter, nrows::Int, ncols::Int, batched::Bool)
    part = fetch(f)
    out = _part_field(part.plan, nrows, ncols, batched)
    _at_part_threads(() -> _synth!(out, C, part.plan, filter), part.nthreads)
    return _to_host(out)
end

function _synthesize!(out::AbstractArray, C, filter, p::DistributedNUSHTplan)
    size(out, 1) == p.npts || throw(DimensionMismatch("the plan covers $(p.npts) points; got $(size(out, 1))"))
    parts = Vector{Any}(undef, length(p.workers))
    @sync for (k, w) in enumerate(p.workers)
        b = p.blocks[k]
        Ck = p.columns ? _take(C, b, true) : C
        nrows, ncols = p.columns ? (p.npts, length(b)) : (length(b), p.B)
        @async parts[k] = Distributed.remotecall_fetch(_part_synthesis, w, p.parts[k], Ck, filter, nrows,
                                                       ncols, out isa AbstractMatrix)
    end
    for (k, v) in enumerate(parts)
        _put!(out, p.blocks[k], v, p.columns)
    end
    return out
end

NUFSHT.nusht_type2!(f, C, p::DistributedNUSHTplan{FE,CE,_Scalar}) where {FE,CE} = _synthesize!(f, C, nothing, p)
NUFSHT.nusht_synthesize!(f_out, C, filter, p::DistributedNUSHTplan{FE,CE,_Scalar}) where {FE,CE} =
    _synthesize!(f_out, C, filter, p)
NUFSHT.nusht_type2_spin!(f, sf, p::DistributedNUSHTplan{FE,CE,_Spin}) where {FE,CE} = _synthesize!(f, sf, nothing, p)

# The adjoint of a worker's part, or with its weights its quadrature.
function _part_adjoint(f, f_part, w_part)
    part = fetch(f)
    C = NUFSHT.allocate_coefficients(part.plan)
    w = _part_weights(part.plan, w_part)
    _at_part_threads(part.nthreads) do
        w === nothing ? _adjoint!(C, f_part, part.plan) : NUFSHT._quadrature!(C, f_part, w, part.plan)
    end
    return _to_host(C)
end

_part_weights(plan, ::Nothing) = nothing
_part_weights(plan, w::AbstractVector) =
    copyto!(NUFSHT._zeros_like(NUFSHT._fbuf(plan), real(eltype(NUFSHT._fbuf(plan))), length(w)), w)

_check_weights(::Nothing, p) = nothing
_check_weights(w::AbstractVector, p) = length(w) == p.npts ||
    throw(DimensionMismatch("the plan covers $(p.npts) points; got $(length(w)) weights"))

# A worker's weights: its points', or every point's for its fields.
_part_w(::Nothing, block, columns::Bool) = nothing
_part_w(w::AbstractVector, block, columns::Bool) = columns ? w : w[block]

# The adjoint and the quadrature: by field, each worker's fields' coefficients; by point, the sum of the
# workers' parts over their blocks, added in worker order.
function _adjoint_blocks!(C, f, w, p::DistributedNUSHTplan{FE,CE}) where {FE,CE}
    size(f, 1) == p.npts || throw(DimensionMismatch("the plan covers $(p.npts) points; got $(size(f, 1))"))
    _check_weights(w, p)
    parts = Vector{Array{CE,3}}(undef, length(p.workers))
    @sync for (k, wk) in enumerate(p.workers)
        b = p.blocks[k]
        @async parts[k] = Distributed.remotecall_fetch(_part_adjoint, wk, p.parts[k], _take(f, b, p.columns),
                                                       _part_w(w, b, p.columns))
    end
    if p.columns
        for (k, v) in enumerate(parts)
            _put!(C, p.blocks[k], reshape(v, size(v, 1), size(v, 2), :), true)
        end
        return C
    end
    fill!(C, zero(eltype(C)))
    for v in parts, i in eachindex(v)
        C[i] += v[i]
    end
    return C
end

NUFSHT.nusht_type1!(C, f, p::DistributedNUSHTplan{FE,CE,_Scalar}) where {FE,CE} = _adjoint_blocks!(C, f, nothing, p)
NUFSHT.nusht_type1_spin!(sf, f, p::DistributedNUSHTplan{FE,CE,_Spin}) where {FE,CE} =
    _adjoint_blocks!(sf, f, nothing, p)

# `nusht_analyze!` with whole-field `f` and `weights`: the quadrature, or the fit.
NUFSHT.nusht_analyze!(C, f, p::DistributedNUSHTplan; weights = nothing, kwargs...) =
    _analyze!(C, f, weights, p; kwargs...)

_analyze!(C, f, ::Nothing, p::DistributedNUSHTplan{FE,CE,_Scalar}; kwargs...) where {FE,CE} =
    NUFSHT.nusht_solve!(C, f, p; kwargs...)
_analyze!(C, f, ::Nothing, p::DistributedNUSHTplan{FE,CE,_Spin}; kwargs...) where {FE,CE} =
    NUFSHT.nusht_solve_spin!(C, f, p; kwargs...)

function _analyze!(C, f, w::AbstractVector, p::DistributedNUSHTplan{FE,CE}; ws = nothing, maxiter = nothing,
                   rtol = nothing, conlim = nothing) where {FE,CE}
    _adjoint_blocks!(C, f, w, p)
    return (; C, iterations = 0, residual = zero(real(CE)), converged = true)
end

function NUFSHT.nusht_filter!(f_out, f_in, filter, p::DistributedNUSHTplan{FE,CE,_Scalar}; ws::Nothing = nothing,
                              kwargs...) where {FE,CE}
    C = NUFSHT.allocate_coefficients(p)
    NUFSHT.nusht_solve!(C, f_in, p; kwargs...)
    return _synthesize!(f_out, C, filter, p)
end

# ── The fit ──────────────────────────────────────────────────────────────────────────────────────────

# A worker's fields fitted alone, with its plan's own workspace.
function _part_solve_fields(f, f_part, w_part, kwargs)
    part = fetch(f)
    C = NUFSHT.allocate_coefficients(part.plan)
    w = _part_weights(part.plan, w_part)
    r = _at_part_threads(part.nthreads) do
        NUFSHT._solve!(C, f_part, NUFSHT._weighted(part.plan, w), part.ws; kwargs...)
    end
    return (C = _to_host(C), iterations = r.iterations, residual = r.residual, converged = r.converged)
end

# The caller's reply once a worker's solve has failed. A worker sends its part of each sum, `nothing`
# when its solve has finished, and its exception if it fails. The sums are the per-column `‖u‖²`, a host
# `Vector{T}`, and `A†u`, the plan's coefficient buffer.
struct _Abort end
_Part{T,CE} = Union{Nothing, Exception, Vector{T}, Array{CE,3}}
_Total{T,CE} = Union{_Abort, Vector{T}, Array{CE,3}}

function _part_solve(f, f_block, w_block, inbox, outbox, k::Int, kwargs)
    p = fetch(f)
    C = NUFSHT.allocate_coefficients(p.plan)
    reduce!(A) = begin
        put!(inbox, (k, _to_host(A)))
        total = take!(outbox)
        total isa _Abort && throw(ErrorException("another worker's solve failed"))
        copyto!(A, total)
    end
    result = try
        op = NUFSHT._weighted(NUFSHT._PointShare(p.plan, reduce!), _part_weights(p.plan, w_block))
        _at_part_threads(() -> NUFSHT._solve!(C, f_block, op, p.ws; kwargs...), p.nthreads)
    catch err
        put!(inbox, (k, err isa Exception ? err : ErrorException(sprint(showerror, err))))
        rethrow()
    end
    put!(inbox, (k, nothing))
    _, iters, rel, converged = result
    return (C = k == 1 ? _to_host(C) : nothing, iterations = iters, residual = rel, converged = converged)
end

# Sum each round of the workers' parts in worker order and return the total to every worker, until they
# report the end of the solve. Returns the first failed worker's index, or `nothing`.
function _coordinate(inbox, outboxes, ::Type{T}, ::Type{CE}, nw::Int) where {T,CE}
    got = Vector{_Part{T,CE}}(undef, nw)
    while true
        for _ in 1:nw
            k, A = take!(inbox)
            got[k] = A
        end
        failed = findfirst(a -> a isa Exception, got)
        if failed !== nothing
            foreach(o -> isready(o) || put!(o, _Abort()), outboxes)
            return failed
        end
        all(isnothing, got) && return nothing
        total = copy(got[1])
        for k in 2:nw
            total .+= got[k]
        end
        foreach(o -> put!(o, total), outboxes)
    end
end

function _solve_points!(C, f, weights, p::DistributedNUSHTplan{FE,CE}, kw) where {FE,CE}
    T = real(CE)
    nw = length(p.workers)
    inbox = Distributed.RemoteChannel(() -> Channel{Tuple{Int,_Part{T,CE}}}(nw))
    outboxes = [Distributed.RemoteChannel(() -> Channel{_Total{T,CE}}(1), w) for w in p.workers]
    solves = [Distributed.remotecall(_part_solve, w, p.parts[k], _take(f, p.blocks[k], false),
                                     _part_w(weights, p.blocks[k], false), inbox, outboxes[k], k, kw)
              for (k, w) in enumerate(p.workers)]
    failed = _coordinate(inbox, outboxes, T, CE, nw)
    failed === nothing || fetch(solves[failed])   # raises that worker's own exception
    foreach(wait, solves)
    Cw, iters, rel, converged = fetch(solves[1])
    copyto!(C, Cw)
    return (; C = C, iterations = iters, residual = rel, converged = converged)
end

function _solve_fields!(C, f, weights, p::DistributedNUSHTplan, kw)
    parts = Vector{Any}(undef, length(p.workers))
    @sync for (k, w) in enumerate(p.workers)
        b = p.blocks[k]
        @async parts[k] = Distributed.remotecall_fetch(_part_solve_fields, w, p.parts[k], _take(f, b, true),
                                                       weights, kw)
    end
    for (k, r) in enumerate(parts)
        _put!(C, p.blocks[k], reshape(r.C, size(r.C, 1), size(r.C, 2), :), true)
    end
    return (; C = C, iterations = maximum(r -> r.iterations, parts), residual = maximum(r -> r.residual, parts),
            converged = all(r -> r.converged, parts))
end

# A real spin field exists only at `s = 0`, as `nusht_solve_spin!` checks on one process.
_check_solvable(::DistributedNUSHTplan{FE,CE,_Scalar}) where {FE,CE} = nothing
_check_solvable(p::DistributedNUSHTplan{FE,CE,_Spin}) where {FE,CE} =
    (FE <: Real && p.kind.s != 0) ? throw(ArgumentError(
        "a spin-$(p.kind.s) field cannot be real, so a real-field plan at s ≠ 0 has no coefficient subspace " *
        "on which a fit is determined; build the plan with a complex element type to fit")) : nothing

function _solve!(C, f, p::DistributedNUSHTplan; weights = nothing, kwargs...)
    _check_solvable(p)
    size(f, 1) == p.npts || throw(DimensionMismatch("the plan covers $(p.npts) points; got $(size(f, 1))"))
    _check_weights(weights, p)
    kw = NamedTuple(kwargs)
    return p.columns ? _solve_fields!(C, f, weights, p, kw) : _solve_points!(C, f, weights, p, kw)
end

"""
    nusht_solve!(C, f, plan::DistributedNUSHTplan; weights = nothing, maxiter, rtol, conlim)
        -> (; C, iterations, residual, converged)

The least-squares fit over every point: by field each worker fits its own fields, by point
[`nusht_solve!`](@ref)'s recurrence runs on every worker at once, the sums over points added across the
workers. `f` and `weights` hold every point, in the plan's order; with `weights` the fit is
`min Σⱼ wⱼ |(A c − f)ⱼ|²`. `iterations` and `residual` are the largest over the fields, `converged` whether
every field converged.
"""
NUFSHT.nusht_solve!(C, f, p::DistributedNUSHTplan{FE,CE,_Scalar}; ws::Nothing = nothing, kwargs...) where {FE,CE} =
    _solve!(C, f, p; kwargs...)

NUFSHT.nusht_solve_spin!(sf, f, p::DistributedNUSHTplan{FE,CE,_Spin}; ws::Nothing = nothing,
                         kwargs...) where {FE,CE} =
    _solve!(sf, f, p; kwargs...)

end # module NUFSHTDistributedExt
