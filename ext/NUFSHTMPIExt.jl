"""
    NUFSHTMPIExt

MPI point-decomposition of a *single* transform: the `M` scattered points are partitioned across
ranks, each holding a local plan for its subset, with the spherical harmonic coefficients replicated
on every rank. Selected by passing a `ComputationalBackends.MPIBackend`, whose `comm` field names the
communicator (`nothing` → `MPI.COMM_WORLD`).

- **Synthesis** `A` (coeffs → local field) needs no communication.
- **Adjoint** `A†` is a sum over points, so each rank computes its local contribution and the result
  is `MPI.Allreduce!`-summed — communication O(lmax²), independent of `M`.
- **Solve** runs conjugate gradients on the global normal equations `A†A c = A†f` with the adjoint
  and inner products `Allreduce`d; every rank ends with the same replicated solution.

Loaded by `using MPI`.
"""
module NUFSHTMPIExt

using NUFSHT: NUFSHT
using ComputationalBackends: ComputationalBackends
using MPI: MPI
using LinearAlgebra: LinearAlgebra

# A custom MPI backend must carry the communicator the same way `ComputationalBackends.MPIBackend`
# does; `nothing` means the world communicator.
@inline _comm(backend::ComputationalBackends.AbstractMPIBackend) = something(backend.comm, MPI.COMM_WORLD)

# Global real inner product across ranks. `dot` on equal-shaped arrays needs no `vec`, which would
# allocate a reshape header on both operands every CG iteration.
_global_dot(a, b, comm) = MPI.Allreduce(LinearAlgebra.dot(a, b), +, comm)

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
    MPICGWorkspace(C, f_local)

Reusable scratch for the MPI solve, mirroring `CGWorkspace` on the shared-memory path: holding one
across solves makes the solver allocation-free instead of allocating five coefficient arrays and a
field array per call.
"""
struct MPICGWorkspace{A, F}
    rhs::A
    r::A
    p::A
    Ap::A
    x::A
    fbuf::F
end
MPICGWorkspace(C, f_local) =
    MPICGWorkspace(similar(C), similar(C), similar(C), similar(C), similar(C), similar(f_local))

"""
    nusht_solve!(C, f_local, plan, MPIBackend(; comm); ws, maxiter, rtol, verbose) -> (C, iters, rel_res)

MPI point-decomposed **exact inversion**: conjugate gradients on `A†A c = A†f` where the points are
partitioned across ranks. `A` (synthesis) needs no communication; `A†` and the CG inner products are
`Allreduce`d. Solves the *global* least-squares system with `C` replicated on every rank.
"""
function NUFSHT.nusht_solve!(C, f_local, plan::NUFSHT.NUSHTplan, backend::ComputationalBackends.AbstractMPIBackend;
                             ws::MPICGWorkspace = MPICGWorkspace(C, f_local),
                             maxiter::Int = 500, rtol::Real = 1e-6, verbose::Bool = false)
    T = eltype(C)
    comm = _comm(backend)
    rhs = ws.rhs; r = ws.r; p = ws.p; Ap = ws.Ap; x = ws.x; fbuf = ws.fbuf
    isroot = MPI.Comm_rank(comm) == 0        # hoisted: was an MPI call per iteration

    # Restrict the fit to degrees `l ≤ lmax`, as the serial solver does: the coefficient array is a
    # square representation carrying degrees up to `lmax+|m|`, and that ragged set is not
    # SO(3)-invariant, so fitting it gives a frame-dependent answer.
    valid = NUFSHT._valid_mask(C, T, plan.lmax)

    NUFSHT._nusht_true_adjoint!(rhs, f_local, plan)
    MPI.Allreduce!(rhs, +, comm)                    # rhs = A†f (replicated)
    rhs .*= valid
    rhsnorm = sqrt(_global_dot(rhs, rhs, comm))

    fill!(x, zero(T)); copyto!(r, rhs); copyto!(p, r)
    rsold = _global_dot(r, r, comm)
    iters = 0
    rel = one(T)
    rhsnorm == 0 && (copyto!(C, x); return C, 0, zero(T))
    for i in 1:maxiter
        iters = i
        NUFSHT.nusht_type2!(fbuf, p, plan)          # A_local p (no communication)
        NUFSHT._nusht_true_adjoint!(Ap, fbuf, plan) # A†_local (A_local p)
        MPI.Allreduce!(Ap, +, comm)                 # Ap = A†A p (replicated)
        Ap .*= valid                                # P A†A P; `p` is already in the subspace
        α = rsold / _global_dot(p, Ap, comm)
        x .+= α .* p
        r .-= α .* Ap
        rsnew = _global_dot(r, r, comm)
        rel = sqrt(rsnew) / rhsnorm
        (verbose && isroot) && @info "nusht_solve! (MPI) iter $i rel_res=$rel"
        rel < rtol && break
        p .= r .+ (rsnew / rsold) .* p
        rsold = rsnew
    end
    copyto!(C, x)
    return C, iters, rel
end

end # module NUFSHTMPIExt