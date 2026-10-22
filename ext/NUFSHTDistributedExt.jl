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
`make_plan(FE, θ, φ, lmax; execution = DistributedBackend())`, a [`DistributedNUSHTplan`](@ref): each
worker holds a plan over its block of the points.
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
    @assert length(θs) == length(φs) == length(Cs) "θs, φs and Cs must have equal length"
    inner = ComputationalBackends.local_backend(execution)
    kw = (; tol, kwargs...)
    NUFSHT._plan_threads(inner, get(kw, :nthreads, nothing))   # a count `inner` refuses raises here
    return _farm(collect(zip(θs, φs, Cs))) do (θ, φ, C)
        _problem(θ, φ, lmax, inner, kw) do plan
            f = zeros(eltype(plan.F), length(θ))
            NUFSHT.nusht_type2!(f, C, plan)
            f
        end
    end
end

function NUFSHT._nodeset_solve(θs, φs, fs, lmax, execution::ComputationalBackends.AbstractDistributedBackend;
                               tol = 1e-8, rtol = 1e-6, maxiter = 500, kwargs...)
    @assert length(θs) == length(φs) == length(fs) "θs, φs and fs must have equal length"
    inner = ComputationalBackends.local_backend(execution)
    kw = (; tol, kwargs...)
    NUFSHT._plan_threads(inner, get(kw, :nthreads, nothing))
    return _farm(collect(zip(θs, φs, fs))) do (θ, φ, f)
        _problem(θ, φ, lmax, inner, kw) do plan
            C = zeros(eltype(plan.F), lmax + 1, 2lmax + 1)
            NUFSHT.nusht_solve!(C, f, plan; rtol = rtol, maxiter = maxiter)
            C
        end
    end
end

# ── One transform divided among the worker processes ─────────────────────────────────────────────────
# Synthesis and the adjoint of a block need no other block. The solve runs `nusht_solve!`'s recurrence
# on every worker at once; its two sums over points go to the caller, which adds the workers' parts in
# worker order and returns the same total to each, so every worker runs the same recurrence.

"""
    DistributedNUSHTplan

A NUSHT plan whose points are divided among the worker processes: each worker's plan over its block and
its least-squares workspace, held on that worker, with the block of the points each covers and the NUFFT
tolerance the plans were built at. The coefficients are the caller's.
"""
struct DistributedNUSHTplan{FE<:Number, CE<:Number, VD<:AbstractVector{Distributed.Future},
                            VI<:AbstractVector{<:Integer}, VU<:AbstractVector{UnitRange{Int}}, TL<:Real}
    parts::VD
    workers::VI
    blocks::VU
    npts::Int
    lmax::Int
    B::Int
    coefficient_size::NTuple{3,Int}
    tol::TL
end

Base.show(io::IO, p::DistributedNUSHTplan{FE}) where {FE} =
    print(io, "DistributedNUSHTplan{", FE, "}(lmax=", p.lmax, ", M=", p.npts, ", ntrans=", p.B, " on ",
          length(p.workers), " workers)")

_block(k::Int, nw::Int, n::Int) = (div((k - 1) * n, nw) + 1):div(k * n, nw)

# The thread count a worker runs its share at under the inner backend and the caller's `nthreads`, read
# on that worker.
_part_threads(inner, nthreads) = NUFSHT._task_count(NUFSHT._plan_threads(inner, nthreads))

# `f()` at the part's FastTransforms thread count.
_at_part_threads(f, nthreads::Int) = Base.ScopedValues.with(f, FTB.FASTTRANSFORMS_THREADS => nthreads)

function _part_plan(::Type{FE}, θ, φ, lmax, inner, kwargs) where {FE}
    n = _part_threads(inner, get(kwargs, :nthreads, nothing))
    plan = _at_part_threads(n) do
        NUFSHT.make_plan(FE, θ, φ, lmax; kwargs..., execution = inner, nthreads = n)
    end
    return (; plan, ws = FTB.LSMRWorkspace(plan), nthreads = n)
end
_part_info(f) = (p = fetch(f).plan; (NUFSHT.coefficient_size(p), p.B, eltype(p.F), p.tol))

# `make_plan(FE, θ, φ, lmax; execution::DistributedBackend, kwargs...)`: the `M` points in contiguous
# blocks, one per worker process (with none, the caller's over all of them), each worker's plan built
# with `kwargs` over its block under `local_backend(execution)`. A worker runs its NUFFT and
# FastTransforms at that backend's thread count there, as `make_plan` sets it: 1 under `SerialBackend()`,
# else `nthreads` when given and the worker's `Threads.nthreads()` when not. The workers need NUFSHT and
# the NUFFT library their plans use loaded.
function NUFSHT._make_plan(::Type{FE}, θ_nodes, φ_nodes, lmax,
                           execution::ComputationalBackends.AbstractDistributedBackend; kwargs...) where {FE<:Number}
    length(θ_nodes) == length(φ_nodes) || throw(DimensionMismatch("θ and φ must have equal length"))
    M = length(θ_nodes)
    ws = Distributed.workers()[1:min(Distributed.nworkers(), M)]
    nw = length(ws)
    blocks = [_block(k, nw, M) for k in 1:nw]
    kw = NamedTuple(kwargs)
    inner = ComputationalBackends.local_backend(execution)
    # A count `inner` refuses raises here, before any worker builds.
    NUFSHT._plan_threads(inner, get(kw, :nthreads, nothing))
    parts = [Distributed.remotecall(_part_plan, w, FE, collect(θ_nodes[blocks[k]]), collect(φ_nodes[blocks[k]]),
                                    lmax, inner, kw) for (k, w) in enumerate(ws)]
    infos = Vector{Tuple{NTuple{3,Int},Int,DataType,Float64}}(undef, nw)
    @sync for (k, w) in enumerate(ws)
        @async infos[k] = Distributed.remotecall_fetch(_part_info, w, parts[k])
    end
    csize, B, CE, tol = first(infos)
    return DistributedNUSHTplan{FE,CE,typeof(parts),typeof(ws),typeof(blocks),typeof(tol)}(
        parts, ws, blocks, M, Int(lmax), B, csize, tol)
end

NUFSHT.allocate_coefficients(p::DistributedNUSHTplan{FE,CE}) where {FE,CE} = zeros(CE, p.coefficient_size...)
NUFSHT.coefficient_size(p::DistributedNUSHTplan) = p.coefficient_size

_part_close(f) = (NUFSHT.close!(fetch(f).plan); nothing)
function NUFSHT.close!(p::DistributedNUSHTplan)
    @sync for (k, w) in enumerate(p.workers)
        @async Distributed.remotecall_wait(_part_close, w, p.parts[k])
    end
    return nothing
end

# Rows `block` of every column of `f`, and the reverse.
_rows(f::AbstractVector, block) = f[block]
_rows(f::AbstractMatrix, block) = f[block, :]
_put_rows!(f::AbstractVector, block, v) = (f[block] .= v; f)
_put_rows!(f::AbstractMatrix, block, v) = (f[block, :] .= v; f)

# The field at this worker's points, a vector or a column per transform like the caller's.
_block_field(p, nrows::Int, batched::Bool) =
    batched ? zeros(eltype(p.F), nrows, p.B) : zeros(eltype(p.F), nrows)

function _part_synthesis(f, C, filter, nrows::Int, batched::Bool)
    part = fetch(f)
    p = part.plan
    out = _block_field(p, nrows, batched)
    _at_part_threads(part.nthreads) do
        filter === nothing ? NUFSHT.nusht_type2!(out, C, p) : NUFSHT.nusht_synthesize!(out, C, filter, p)
    end
    return _to_host(out)
end

_to_host(A::Array) = A
_to_host(A::AbstractArray) = Array(A)

function _synthesize!(out::AbstractArray{<:Any,N}, C, filter, p::DistributedNUSHTplan{FE,CE}) where {N,FE,CE}
    size(out, 1) == p.npts || throw(DimensionMismatch("the plan covers $(p.npts) points; got $(size(out, 1))"))
    parts = Vector{Array{CE,N}}(undef, length(p.workers))
    @sync for (k, w) in enumerate(p.workers)
        @async parts[k] = Distributed.remotecall_fetch(_part_synthesis, w, p.parts[k], C, filter,
                                                       length(p.blocks[k]), out isa AbstractMatrix)
    end
    for (k, v) in enumerate(parts)
        _put_rows!(out, p.blocks[k], v)
    end
    return out
end

NUFSHT.nusht_type2!(f, C, p::DistributedNUSHTplan) = _synthesize!(f, C, nothing, p)
NUFSHT.nusht_synthesize!(f_out, C, filter, p::DistributedNUSHTplan) = _synthesize!(f_out, C, filter, p)

# The adjoint of a worker's block, or with the block's weights its quadrature.
function _part_adjoint(f, f_block, w_block)
    part = fetch(f)
    C = NUFSHT.allocate_coefficients(part.plan)
    _at_part_threads(part.nthreads) do
        w_block === nothing ? NUFSHT.nusht_type1!(C, f_block, part.plan) :
                              NUFSHT._quadrature!(C, f_block, w_block, part.plan)
    end
    return _to_host(C)
end

_block_weights(::Nothing, block) = nothing
_block_weights(w::AbstractVector, block) = w[block]

# The adjoint and the quadrature are sums over points: each worker's over its block, added in worker
# order.
function _sum_blocks!(C, f, w, p::DistributedNUSHTplan{FE,CE}) where {FE,CE}
    size(f, 1) == p.npts || throw(DimensionMismatch("the plan covers $(p.npts) points; got $(size(f, 1))"))
    (w === nothing || length(w) == p.npts) ||
        throw(DimensionMismatch("the plan covers $(p.npts) points; got $(length(w)) weights"))
    parts = Vector{Array{CE,3}}(undef, length(p.workers))
    @sync for (k, wk) in enumerate(p.workers)
        @async parts[k] = Distributed.remotecall_fetch(_part_adjoint, wk, p.parts[k], _rows(f, p.blocks[k]),
                                                       _block_weights(w, p.blocks[k]))
    end
    fill!(C, zero(eltype(C)))
    for v in parts, i in eachindex(v)
        C[i] += v[i]
    end
    return C
end

NUFSHT.nusht_type1!(C, f, p::DistributedNUSHTplan) = _sum_blocks!(C, f, nothing, p)

# `nusht_analyze!` with whole-field `f` and `weights`: the quadrature summed over the workers' blocks,
# or the fit.
NUFSHT.nusht_analyze!(C, f, p::DistributedNUSHTplan; weights = nothing, kwargs...) =
    _analyze!(C, f, weights, p; kwargs...)

_analyze!(C, f, ::Nothing, p::DistributedNUSHTplan; kwargs...) = NUFSHT.nusht_solve!(C, f, p; kwargs...)

function _analyze!(C, f, w::AbstractVector, p::DistributedNUSHTplan{FE,CE}; ws = nothing, maxiter = nothing,
                   rtol = nothing, conlim = nothing) where {FE,CE}
    _sum_blocks!(C, f, w, p)
    return (; C, iterations = 0, residual = zero(real(CE)), converged = true)
end

function NUFSHT.nusht_filter!(f_out, f_in, filter, p::DistributedNUSHTplan; ws::Nothing = nothing, kwargs...)
    C = NUFSHT.allocate_coefficients(p)
    NUFSHT.nusht_solve!(C, f_in, p; kwargs...)
    return _synthesize!(f_out, C, filter, p)
end

# ── The solve ────────────────────────────────────────────────────────────────────────────────────────

# The caller's reply once a worker's solve has failed. A worker sends its part of each sum, `nothing`
# when its solve has finished, and its exception if it fails. The sums are the per-column `‖u‖²`, a host
# `Vector{T}`, and `A†u`, the plan's coefficient buffer.
struct _Abort end
_Part{T,CE} = Union{Nothing, Exception, Vector{T}, Array{CE,3}}
_Total{T,CE} = Union{_Abort, Vector{T}, Array{CE,3}}

function _part_solve(f, f_block, inbox, outbox, k::Int, kwargs)
    p = fetch(f)
    C = NUFSHT.allocate_coefficients(p.plan)
    reduce!(A) = begin
        put!(inbox, (k, _to_host(A)))
        total = take!(outbox)
        total isa _Abort && throw(ErrorException("another worker's solve failed"))
        copyto!(A, total)
    end
    result = try
        _at_part_threads(() -> NUFSHT._solve!(C, f_block, NUFSHT._PointShare(p.plan, reduce!), p.ws; kwargs...),
                         p.nthreads)
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

"""
    nusht_solve!(C, f, plan::DistributedNUSHTplan; maxiter, rtol, conlim) -> (; C, iterations, residual, converged)

The least-squares fit over every worker's points: [`nusht_solve!`](@ref)'s recurrence on every worker at
once, the sums over points added across the workers. `f` holds every point, in the plan's order.
"""
function NUFSHT.nusht_solve!(C, f, p::DistributedNUSHTplan{FE,CE}; ws::Nothing = nothing, kwargs...) where {FE,CE}
    size(f, 1) == p.npts || throw(DimensionMismatch("the plan covers $(p.npts) points; got $(size(f, 1))"))
    T = real(CE)
    nw = length(p.workers)
    kw = NamedTuple(kwargs)
    inbox = Distributed.RemoteChannel(() -> Channel{Tuple{Int,_Part{T,CE}}}(nw))
    outboxes = [Distributed.RemoteChannel(() -> Channel{_Total{T,CE}}(1), w) for w in p.workers]
    solves = [Distributed.remotecall(_part_solve, w, p.parts[k], _rows(f, p.blocks[k]), inbox, outboxes[k], k, kw)
              for (k, w) in enumerate(p.workers)]
    failed = _coordinate(inbox, outboxes, T, CE, nw)
    failed === nothing || fetch(solves[failed])   # raises that worker's own exception
    foreach(wait, solves)
    Cw, iters, rel, converged = fetch(solves[1])
    copyto!(C, Cw)
    return (; C = C, iterations = iters, residual = rel, converged = converged)
end

end # module NUFSHTDistributedExt
