"""
    NUFSHTMPIExt

MPI decomposition of a *single* transform, selected by `ComputationalBackends.MPIBackend(; comm)`, whose
`comm` field names the communicator (`nothing` → `MPI.COMM_WORLD`).

A rank passes its own points' plan and field to `nusht_type1!(C, f_local, plan, MPIBackend(; comm))`, and
likewise `nusht_solve!`, `nusht_solve_spin!` and `nusht_analyze!`:

- **Synthesis** `A` (coeffs → local field) needs no communication.
- **Adjoint** `A†` is a sum over points, so each rank computes its local contribution and the result
  is `MPI.Allreduce!`-summed — communication O(lmax²), independent of `M`.
- **Solve** runs [`nusht_solve!`](@ref)'s recurrence on the global least-squares problem `min ‖Ac − f‖`,
  with its two sums over points, `‖u‖` and `A†u`, `Allreduce`d; every rank ends with the same
  replicated solution.

`make_plan(FE, θ, φ, lmax; execution = MPIBackend())` and `make_spin_plan(…)` instead take every point and
return an [`MPINUSHTplan`](@ref), whose transforms take and return the whole field on every rank: a batch
of at least as many fields as ranks divides by its fields, each rank's plan over every point for a block
of them, and otherwise by its points.

Loaded by `using MPI`.
"""
module NUFSHTMPIExt

using NUFSHT: NUFSHT
using ComputationalBackends: ComputationalBackends
using FlowTransformBindings: FlowTransformBindings as FTB
using MPI: MPI

# A custom MPI backend must carry the communicator the same way `ComputationalBackends.MPIBackend`
# does; `nothing` means the world communicator.
@inline _comm(execution::ComputationalBackends.AbstractMPIBackend) = something(execution.comm, MPI.COMM_WORLD)

# `nusht_type1!(C, f_local, plan, MPIBackend(; comm))`: `A†` is a sum over points, so each rank adds
# its own points' contribution and the sum lands in `C` on every rank.
function NUFSHT._point_share_adjoint!(C, f_local, plan::NUFSHT.AbstractScalarNUSHTplan,
                                      execution::ComputationalBackends.AbstractMPIBackend)
    NUFSHT._nusht_true_adjoint!(C, f_local, plan)   # local A† (sum over this rank's points)
    MPI.Allreduce!(C, +, _comm(execution))          # sum contributions across ranks → A†f
    return C
end

# `nusht_solve!` and `nusht_solve_spin!` with `MPIBackend(; comm)` fourth: the least-squares fit over
# the points of every rank, each rank's plan over its own points.
NUFSHT._point_share(plan::NUFSHT.AbstractNUSHTplan, execution::ComputationalBackends.AbstractMPIBackend) =
    _ranks(plan, _comm(execution))

# `nusht_analyze!(C, f_local, plan, MPIBackend(; comm); weights)`: the quadrature is a sum over points,
# each rank's own added across the ranks.
NUFSHT._point_share_quadrature!(C, f_local, w_local, plan::NUFSHT.AbstractNUSHTplan,
                                execution::ComputationalBackends.AbstractMPIBackend) =
    MPI.Allreduce!(NUFSHT._quadrature!(C, f_local, w_local, plan), +, _comm(execution))

# `plan` over this rank's points, its sums over points added across the ranks of `comm`.
_ranks(plan, comm) = NUFSHT._PointShare(plan, A -> MPI.Allreduce!(A, +, comm))

# ── A plan over every point, divided among the ranks ─────────────────────────────────────────────────

"""
    MPINUSHTplan

A scalar or spin NUSHT plan divided among the ranks of a communicator: with `columns`, this rank's plan
over every point for the contiguous block `own` of the `B` fields, the ranks' field and coefficient
blocks `fcounts` and `ccounts` values long; otherwise its plan over the points `own = (rank+1):nranks:M`.
Every rank holds the whole field and the coefficients, and the transforms take and return them whole.
"""
struct MPINUSHTplan{P<:NUFSHT.AbstractNUSHTplan, O<:AbstractRange{Int}, L<:AbstractMatrix, C}
    plan::P
    own::O
    columns::Bool
    npts::Int
    B::Int
    local_out::L
    fcounts::Vector{Int}
    ccounts::Vector{Int}
    comm::C
end

Base.show(io::IO, p::MPINUSHTplan) =
    print(io, "MPINUSHTplan(lmax=", p.plan.lmax, ", M=", p.npts, ", ntrans=", p.B, ", ",
          p.columns ? "fields " : "points ", p.own, " on rank ", MPI.Comm_rank(p.comm), " of ",
          MPI.Comm_size(p.comm), ")")

_block(k::Int, n::Int, total::Int) = (div((k - 1) * total, n) + 1):div(k * total, n)

_build(::Nothing, ::Type{FE}, θ, φ, lmax, execution; kw...) where {FE} =
    NUFSHT._make_plan(FE, θ, φ, lmax, execution; kw...)
_build(s::Int, ::Type{FE}, θ, φ, lmax, execution; kw...) where {FE} =
    NUFSHT._make_spin_plan(FE, θ, φ, lmax, s, execution; kw...)

# This rank's plan under `local_backend(execution)`: over every point for a block of the fields when the
# batch has at least as many fields as there are ranks, over its share of the points otherwise.
function _mpi_plan(s, ::Type{FE}, θ_nodes, φ_nodes, lmax, execution; kwargs...) where {FE}
    length(θ_nodes) == length(φ_nodes) || throw(DimensionMismatch("θ and φ must have equal length"))
    comm = _comm(execution)
    nranks = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    M = length(θ_nodes)
    B = Int(get(kwargs, :ntrans, 1))
    inner = ComputationalBackends.local_backend(execution)
    lmax = Int(lmax)
    K = (lmax + 1) * (2lmax + 1)
    if B >= nranks
        own = _block(rank + 1, nranks, B)
        plan = _build(s, FE, θ_nodes, φ_nodes, lmax, inner; kwargs..., ntrans = length(own))
        nb = [length(_block(k, nranks, B)) for k in 1:nranks]
        local_out = NUFSHT._zeros_like(NUFSHT._fbuf(plan), _field_eltype(plan), M, length(own))
        return MPINUSHTplan(plan, own, true, M, B, local_out, M .* nb, K .* nb, comm)
    end
    M ≥ nranks || throw(ArgumentError("an MPI plan needs a point per rank: $M points on $nranks ranks"))
    own = (rank + 1):nranks:M
    plan = _build(s, FE, θ_nodes[own], φ_nodes[own], lmax, inner; kwargs...)
    local_out = NUFSHT._zeros_like(NUFSHT._fbuf(plan), _field_eltype(plan), length(own), B)
    return MPINUSHTplan(plan, own, false, M, B, local_out, Int[], Int[], comm)
end

NUFSHT._make_plan(::Type{FE}, θ, φ, lmax, execution::ComputationalBackends.AbstractMPIBackend;
                  kwargs...) where {FE<:Number} =
    _mpi_plan(nothing, FE, θ, φ, lmax, execution; kwargs...)
NUFSHT._make_spin_plan(::Type{FE}, θ, φ, lmax, s, execution::ComputationalBackends.AbstractMPIBackend;
                       kwargs...) where {FE<:Number} =
    _mpi_plan(Int(s), FE, θ, φ, lmax, execution; kwargs...)

_field_eltype(p::NUFSHT.AbstractScalarNUSHTplan) = NUFSHT._field_eltype(p)
_field_eltype(p::NUFSHT.SpinNUSHTplan{T}) where {T} = NUFSHT._hermitian_domain(p) ? T : Complex{T}

NUFSHT.allocate_coefficients(p::MPINUSHTplan) =
    NUFSHT._zeros_like(NUFSHT._fbuf(p.plan), _coeff_eltype(p.plan), NUFSHT.coefficient_size(p)...)
_coeff_eltype(p::NUFSHT.AbstractScalarNUSHTplan) = NUFSHT._field_eltype(p)
_coeff_eltype(::NUFSHT.SpinNUSHTplan{T}) where {T} = Complex{T}
NUFSHT.coefficient_size(p::MPINUSHTplan) = (Base.front(NUFSHT.coefficient_size(p.plan))..., p.B)
FTB.LSMRWorkspace(p::MPINUSHTplan) = FTB.LSMRWorkspace(p.plan)
NUFSHT.close!(p::MPINUSHTplan) = NUFSHT.close!(p.plan)

function _checked(f, p::MPINUSHTplan)
    size(f, 1) == p.npts || throw(DimensionMismatch("the plan covers $(p.npts) points; got $(size(f, 1))"))
    return f
end

# This rank's part of a whole field (its points' rows, or its fields) and of the coefficients (its
# fields, or all of them).
_own(f, p::MPINUSHTplan) = p.columns ? view(reshape(f, p.npts, :), :, p.own) : view(reshape(f, p.npts, :), p.own, :)
_own_coeffs(C, p::MPINUSHTplan) = p.columns ? view(reshape(C, size(C, 1), size(C, 2), :), :, :, p.own) : C

# The whole field from every rank's synthesis in `local_out`: by fields, gathered; by points, this rank's
# rows written into zeros and summed across the ranks.
function _assemble!(f, p::MPINUSHTplan)
    if p.columns
        copyto!(view(reshape(f, p.npts, :), :, p.own), p.local_out)
        return MPI.Allgatherv!(MPI.VBuffer(f, p.fcounts), p.comm)
    end
    fill!(f, zero(eltype(f)))
    view(reshape(f, p.npts, :), p.own, :) .= p.local_out
    return MPI.Allreduce!(f, +, p.comm)
end

# The whole coefficients from every rank's adjoint in place: by fields, gathered; by points, summed.
_combine!(C, p::MPINUSHTplan) =
    p.columns ? MPI.Allgatherv!(MPI.VBuffer(C, p.ccounts), p.comm) : MPI.Allreduce!(C, +, p.comm)

_synth!(out, C, plan::NUFSHT.AbstractScalarNUSHTplan) = NUFSHT.nusht_type2!(out, C, plan)
_synth!(out, sf, plan::NUFSHT.SpinNUSHTplan) = NUFSHT.nusht_type2_spin!(out, sf, plan)
_adjoint!(C, f, plan::NUFSHT.AbstractScalarNUSHTplan) = NUFSHT._nusht_true_adjoint!(C, f, plan)
_adjoint!(sf, f, plan::NUFSHT.SpinNUSHTplan) = NUFSHT.nusht_type1_spin!(sf, f, plan)

function _synthesize!(f, C, p::MPINUSHTplan)
    _synth!(p.local_out, _own_coeffs(C, p), p.plan)
    return _assemble!(_checked(f, p), p)
end

function _adjoint_all!(C, f, p::MPINUSHTplan)
    _adjoint!(_own_coeffs(C, p), _own(_checked(f, p), p), p.plan)
    return _combine!(C, p)
end

NUFSHT.nusht_type2!(f, C, p::MPINUSHTplan{<:NUFSHT.AbstractScalarNUSHTplan}) = _synthesize!(f, C, p)
NUFSHT.nusht_type2_spin!(f, sf, p::MPINUSHTplan{<:NUFSHT.SpinNUSHTplan}) = _synthesize!(f, sf, p)
NUFSHT.nusht_type1!(C, f, p::MPINUSHTplan{<:NUFSHT.AbstractScalarNUSHTplan}) = _adjoint_all!(C, f, p)
NUFSHT.nusht_type1_spin!(sf, f, p::MPINUSHTplan{<:NUFSHT.SpinNUSHTplan}) = _adjoint_all!(sf, f, p)

function NUFSHT.nusht_synthesize!(f, C, filter, p::MPINUSHTplan{<:NUFSHT.AbstractScalarNUSHTplan})
    NUFSHT.nusht_synthesize!(p.local_out, _own_coeffs(C, p), filter, p.plan)
    return _assemble!(_checked(f, p), p)
end

# The fit: by fields, each rank fits its own and the coefficients are gathered, the outcome the worst
# over the ranks; by points, the recurrence over every rank's points.
function _solve!(C, f, weights, p::MPINUSHTplan, ws; kwargs...)
    if p.columns
        r = NUFSHT._solve!(_own_coeffs(C, p), _own(_checked(f, p), p), NUFSHT._weighted(p.plan, weights), ws;
                           kwargs...)
        _combine!(C, p)
        return (; C, iterations = MPI.Allreduce(r.iterations, max, p.comm),
                residual = MPI.Allreduce(r.residual, max, p.comm),
                converged = MPI.Allreduce(Int(r.converged), min, p.comm) == 1)
    end
    w = weights === nothing ? nothing : view(weights, p.own)
    return NUFSHT._solve!(C, _own(_checked(f, p), p), NUFSHT._weighted(_ranks(p.plan, p.comm), w), ws; kwargs...)
end

_check_length(::Nothing, p) = nothing
_check_length(w::AbstractVector, p) = length(w) == p.npts ||
    throw(DimensionMismatch("the plan covers $(p.npts) points; got $(length(w)) weights"))

function NUFSHT.nusht_solve!(C, f, p::MPINUSHTplan{<:NUFSHT.AbstractScalarNUSHTplan};
                             ws::FTB.LSMRWorkspace = FTB.LSMRWorkspace(p), weights = nothing, kwargs...)
    _check_length(weights, p)
    return _solve!(C, f, weights, p, ws; kwargs...)
end

function NUFSHT.nusht_solve_spin!(sf, f, p::MPINUSHTplan{<:NUFSHT.SpinNUSHTplan};
                                  ws::FTB.LSMRWorkspace = FTB.LSMRWorkspace(p), weights = nothing, kwargs...)
    NUFSHT._check_spin_solvable(p.plan)
    _check_length(weights, p)
    return _solve!(sf, f, weights, p, ws; kwargs...)
end

function NUFSHT.nusht_filter!(f_out, f_in, filter, p::MPINUSHTplan{<:NUFSHT.AbstractScalarNUSHTplan};
                              ws::FTB.LSMRWorkspace = FTB.LSMRWorkspace(p), kwargs...)
    if p.columns
        NUFSHT.nusht_filter!(p.local_out, _own(_checked(f_in, p), p), filter, p.plan; ws, kwargs...)
        return _assemble!(_checked(f_out, p), p)
    end
    C = NUFSHT._filter_scratch(p.plan)
    NUFSHT.nusht_solve!(C, f_in, p; ws = ws, kwargs...)
    return NUFSHT.nusht_synthesize!(f_out, C, filter, p)
end

# `nusht_analyze!` with whole-field `f` and `weights`: the quadrature over this rank's part, combined
# across the ranks, or the fit.
NUFSHT.nusht_analyze!(C, f, p::MPINUSHTplan; weights = nothing, kwargs...) = _analyze!(C, f, weights, p; kwargs...)

_analyze!(C, f, ::Nothing, p::MPINUSHTplan{<:NUFSHT.AbstractScalarNUSHTplan}; kwargs...) =
    NUFSHT.nusht_solve!(C, f, p; kwargs...)
_analyze!(C, f, ::Nothing, p::MPINUSHTplan{<:NUFSHT.SpinNUSHTplan}; kwargs...) =
    NUFSHT.nusht_solve_spin!(C, f, p; kwargs...)

function _analyze!(C, f, w::AbstractArray, p::MPINUSHTplan; ws = nothing, maxiter = nothing,
                   rtol = nothing, conlim = nothing)
    _check_length(w, p)
    NUFSHT._quadrature!(_own_coeffs(C, p), _own(_checked(f, p), p), p.columns ? w : view(w, p.own), p.plan)
    _combine!(C, p)
    return NUFSHT._quadrature_result(C, p.plan)
end

end # module NUFSHTMPIExt
