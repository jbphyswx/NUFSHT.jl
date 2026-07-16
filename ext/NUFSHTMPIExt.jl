"""
    NUFSHTMPIExt

MPI point-decomposition of a *single* transform: the `M` scattered points are partitioned across
ranks, each holding a local plan for its subset, with the spherical harmonic coefficients replicated
on every rank.

- **Synthesis** `A` (coeffs → local field) needs no communication.
- **Adjoint** `A†` is a sum over points, so each rank computes its local contribution and the result
  is `MPI.Allreduce!`-summed — communication O(lmax²), independent of `M`.
- **Solve** runs conjugate gradients on the global normal equations `A†A c = A†f` with the adjoint
  and inner products `Allreduce`d; every rank ends with the same replicated solution.

Loaded by `using MPI`.
"""
module NUFSHTMPIExt

using NUFSHT: NUFSHT
using MPI: MPI
using LinearAlgebra: LinearAlgebra

# Global real inner product across ranks.
_global_dot(a, b, comm) = MPI.Allreduce(LinearAlgebra.dot(vec(a), vec(b)), +, comm)

function NUFSHT.nusht_adjoint_mpi!(C, f_local, plan, comm)
    NUFSHT._nusht_true_adjoint!(C, f_local, plan)   # local A† (sum over this rank's points)
    MPI.Allreduce!(C, +, comm)                      # sum contributions across ranks → A†f
    return C
end

function NUFSHT.nusht_solve_mpi!(C, f_local, plan, comm;
                                 maxiter::Int = 500, rtol::Real = 1e-6, verbose::Bool = false)
    T = eltype(C)
    rhs = similar(C); r = similar(C); p = similar(C); Ap = similar(C); x = similar(C)
    fbuf = similar(f_local)

    NUFSHT._nusht_true_adjoint!(rhs, f_local, plan)
    MPI.Allreduce!(rhs, +, comm)                    # rhs = A†f (replicated)
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
        α = rsold / _global_dot(p, Ap, comm)
        x .+= α .* p
        r .-= α .* Ap
        rsnew = _global_dot(r, r, comm)
        rel = sqrt(rsnew) / rhsnorm
        (verbose && MPI.Comm_rank(comm) == 0) && @info "nusht_solve_mpi! iter $i rel_res=$rel"
        rel < rtol && break
        p .= r .+ (rsnew / rsold) .* p
        rsold = rsnew
    end
    copyto!(C, x)
    return C, iters, rel
end

end # module NUFSHTMPIExt
