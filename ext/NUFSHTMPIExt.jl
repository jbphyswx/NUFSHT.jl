"""
    NUFSHTMPIExt

MPI point-decomposition of a *single* transform: the `M` scattered points are partitioned across
ranks, each holding a local plan for its subset, with the spherical harmonic coefficients replicated
on every rank. Selected by passing a `ComputationalBackends.MPIBackend`, whose `comm` field names the
communicator (`nothing` → `MPI.COMM_WORLD`).

- **Synthesis** `A` (coeffs → local field) needs no communication.
- **Adjoint** `A†` is a sum over points, so each rank computes its local contribution and the result
  is `MPI.Allreduce!`-summed — communication O(lmax²), independent of `M`.
- **Solve** runs [`nusht_solve!`](@ref)'s recurrence on the global least-squares problem `min ‖Ac − f‖`,
  with its two sums over points, `‖u‖` and `A†u`, `Allreduce`d; every rank ends with the same
  replicated solution.

A rank passes its own points' plan and field to these methods. `make_plan(FE, θ, φ, lmax,
MPIBackend())` instead takes every point and returns an [`MPINUSHTplan`](@ref) over this rank's share,
whose transforms take and return the whole field.

Loaded by `using MPI`.
"""
module NUFSHTMPIExt

using NUFSHT: NUFSHT
using ComputationalBackends: ComputationalBackends
using FlowTransformBindings: FlowTransformBindings as FTB
using MPI: MPI

# A custom MPI backend must carry the communicator the same way `ComputationalBackends.MPIBackend`
# does; `nothing` means the world communicator.
@inline _comm(backend::ComputationalBackends.AbstractMPIBackend) = something(backend.comm, MPI.COMM_WORLD)

"""
    nusht_type1!(C, f_local, plan, MPIBackend(; comm)) -> C

MPI point-decomposed **adjoint** `A†f`. Each rank owns a disjoint subset of the `M` points with a
local plan; since `A†` is a sum over points, each rank computes its local contribution and the result
is summed into `C`, replicated on every rank.
"""
function NUFSHT.nusht_type1!(C, f_local, plan::NUFSHT.NUSHTplan, backend::ComputationalBackends.AbstractMPIBackend)
    NUFSHT._nusht_true_adjoint!(C, f_local, plan)   # local A† (sum over this rank's points)
    MPI.Allreduce!(C, +, _comm(backend))            # sum contributions across ranks → A†f
    return C
end

"""
    nusht_solve!(C, f_local, plan, MPIBackend(; comm); ws = FlowTransformBindings.LSMRWorkspace(plan), maxiter, rtol, conlim)
    nusht_solve_spin!(sf, f_local, plan, MPIBackend(; comm); …)

MPI point-decomposed **exact inversion**: the least-squares fit over the points of every rank, each
rank's `plan` over its own points. Same contract and return as the single-process
[`nusht_solve!`](@ref) and [`nusht_solve_spin!`](@ref), with the coefficients replicated on every rank.
"""
function NUFSHT.nusht_solve!(C, f_local, plan::NUFSHT.NUSHTplan, backend::ComputationalBackends.AbstractMPIBackend;
                             ws::FTB.LSMRWorkspace = FTB.LSMRWorkspace(plan), kwargs...)
    return NUFSHT._solve!(C, f_local, _ranks(plan, _comm(backend)), ws; kwargs...)
end

function NUFSHT.nusht_solve_spin!(sf, f_local, plan::NUFSHT.SpinNUSHTplan,
                                  backend::ComputationalBackends.AbstractMPIBackend;
                                  ws::FTB.LSMRWorkspace = FTB.LSMRWorkspace(plan), kwargs...)
    NUFSHT._check_spin_solvable(plan)
    return NUFSHT._solve!(sf, f_local, _ranks(plan, _comm(backend)), ws; kwargs...)
end

# `plan` over this rank's points, its sums over points added across the ranks of `comm`.
_ranks(plan, comm) = NUFSHT._PointShare(plan, A -> MPI.Allreduce!(A, +, comm))

# ── A plan over every point, divided among the ranks ─────────────────────────────────────────────────

"""
    MPINUSHTplan

A NUSHT plan whose points are divided among the ranks of a communicator: this rank's plan over its
points `own = (rank+1):nranks:M`, and the synthesis of those points. Every rank holds the whole field
and the coefficients, and the transforms take and return the whole field.
"""
struct MPINUSHTplan{P<:NUFSHT.NUSHTplan, O<:AbstractRange{Int}, L<:AbstractMatrix, C}
    plan::P
    own::O
    npts::Int
    local_out::L
    comm::C
end

Base.show(io::IO, p::MPINUSHTplan) =
    print(io, "MPINUSHTplan(lmax=", p.plan.lmax, ", M=", p.npts, ", ntrans=", p.plan.B, ", ", length(p.own),
          " points on rank ", MPI.Comm_rank(p.comm), " of ", MPI.Comm_size(p.comm), ")")

# `make_plan(FE, θ, φ, lmax, backend::MPIBackend; kwargs...)`: this rank's plan, on
# `local_backend(backend)`, over its share of the `M` points.
function NUFSHT._backend_plan(::Type{FE}, θ_nodes, φ_nodes, lmax, backend::ComputationalBackends.AbstractMPIBackend;
                              kwargs...) where {FE<:Number}
    length(θ_nodes) == length(φ_nodes) || throw(DimensionMismatch("θ and φ must have equal length"))
    comm = _comm(backend)
    nranks = MPI.Comm_size(comm)
    M = length(θ_nodes)
    M ≥ nranks || throw(ArgumentError("an MPI plan needs a point per rank: $M points on $nranks ranks"))
    own = (MPI.Comm_rank(comm) + 1):nranks:M
    plan = NUFSHT._backend_plan(FE, θ_nodes[own], φ_nodes[own], lmax,
                                ComputationalBackends.local_backend(backend); kwargs...)
    local_out = NUFSHT._zeros_like(plan.F, eltype(plan.F), length(own), plan.B)
    return MPINUSHTplan(plan, own, M, local_out, comm)
end

NUFSHT.allocate_coefficients(p::MPINUSHTplan) = NUFSHT.allocate_coefficients(p.plan)
NUFSHT.coefficient_size(p::MPINUSHTplan) = NUFSHT.coefficient_size(p.plan)
FTB.LSMRWorkspace(p::MPINUSHTplan) = FTB.LSMRWorkspace(p.plan)
NUFSHT.close!(p::MPINUSHTplan) = NUFSHT.close!(p.plan)

# This rank's rows of a whole field, a vector or a column per transform.
_own(f, p::MPINUSHTplan) = view(reshape(f, p.npts, :), p.own, :)

function _checked(f, p::MPINUSHTplan)
    size(f, 1) == p.npts || throw(DimensionMismatch("the plan covers $(p.npts) points; got $(size(f, 1))"))
    return f
end

# This rank's synthesis written into the whole field, which the sum across ranks completes.
function _assemble!(f, p::MPINUSHTplan)
    fill!(f, zero(eltype(f)))
    _own(f, p) .= p.local_out
    return MPI.Allreduce!(f, +, p.comm)
end

function NUFSHT.nusht_type2!(f, C, p::MPINUSHTplan)
    NUFSHT.nusht_type2!(p.local_out, C, p.plan)
    return _assemble!(_checked(f, p), p)
end

function NUFSHT.nusht_synthesize!(f, C, filter, p::MPINUSHTplan)
    NUFSHT.nusht_synthesize!(p.local_out, C, filter, p.plan)
    return _assemble!(_checked(f, p), p)
end

function NUFSHT.nusht_type1!(C, f, p::MPINUSHTplan)
    NUFSHT._nusht_true_adjoint!(C, _own(_checked(f, p), p), p.plan)
    return MPI.Allreduce!(C, +, p.comm)
end

function NUFSHT.nusht_solve!(C, f, p::MPINUSHTplan; ws::FTB.LSMRWorkspace = FTB.LSMRWorkspace(p), kwargs...)
    return NUFSHT._solve!(C, _own(_checked(f, p), p), _ranks(p.plan, p.comm), ws; kwargs...)
end

function NUFSHT.nusht_filter!(f_out, f_in, filter, p::MPINUSHTplan;
                              ws::FTB.LSMRWorkspace = FTB.LSMRWorkspace(p), kwargs...)
    C = NUFSHT._filter_scratch(p.plan)
    NUFSHT.nusht_solve!(C, f_in, p; ws = ws, kwargs...)
    return NUFSHT.nusht_synthesize!(f_out, C, filter, p)
end

end # module NUFSHTMPIExt
