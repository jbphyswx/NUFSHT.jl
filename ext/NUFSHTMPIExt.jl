"""
    NUFSHTMPIExt

MPI point-decomposition of a *single* transform: the `M` scattered points are partitioned across
ranks, each holding a local plan for its subset, with the spherical harmonic coefficients replicated
on every rank. Selected by passing a `ComputationalBackends.MPIBackend`, whose `comm` field names the
communicator (`nothing` → `MPI.COMM_WORLD`).

- **Synthesis** `A` (coeffs → local field) needs no communication.
- **Adjoint** `A†` is a sum over points, so each rank computes its local contribution and the result
  is `MPI.Allreduce!`-summed — communication O(lmax²), independent of `M`.
- **Solve** runs LSMR on the global least-squares problem `min ‖Ac − f‖` with the adjoint
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

# `u` lives in POINT space and is partitioned, so its norm is a sum over ranks. The coefficient-space
# vectors are replicated — every rank holds the same array — so their norms are already global and
# reducing them would multiply by the rank count. Only this one collective per iteration is real.
# `dot` on equal-shaped arrays needs no `vec`, which would allocate a reshape header on both operands.
_global_sq(u, comm) = MPI.Allreduce(LinearAlgebra.dot(u, u), +, comm)
_local_sq(v) = real(LinearAlgebra.dot(v, v))

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
    MPILSMRWorkspace(C, f_local)

Reusable scratch for the MPI solve, mirroring [`LSMRWorkspace`](@ref) on the shared-memory path:
holding one across solves makes the solver allocation-free instead of allocating five coefficient
arrays and two field arrays per call.
"""
struct MPILSMRWorkspace{A, F}
    x::A
    v::A
    h::A
    hbar::A
    w::A                                        # A†u before it is folded into v
    u::F                                        # left bidiagonalization vector, this rank's points
    Av::F
end
MPILSMRWorkspace(C, f_local) =
    MPILSMRWorkspace(similar(C), similar(C), similar(C), similar(C), similar(C),
                     similar(f_local), similar(f_local))

"""
    nusht_solve!(C, f_local, plan, MPIBackend(; comm); ws, maxiter, rtol, conlim, verbose) -> (C, iters, rel_res, converged)

MPI point-decomposed **exact inversion**: LSMR on the Golub–Kahan bidiagonalization of `A`, where the
points are partitioned across ranks. `A` (synthesis) needs no communication; `A†` is `Allreduce`d, as
is `‖u‖` — the one bidiagonalization scalar that lives in the partitioned point space. Solves the
*global* least-squares problem with `C` replicated on every rank.

Same contract and return as the shared-memory [`nusht_solve!`](@ref). Every rank runs the identical
scalar recurrence on identical reduced values, so all ranks stop on the same iteration.
"""
function NUFSHT.nusht_solve!(C, f_local, plan::NUFSHT.NUSHTplan, backend::ComputationalBackends.AbstractMPIBackend;
                             ws::MPILSMRWorkspace = MPILSMRWorkspace(C, f_local),
                             maxiter::Int = 500, rtol::Real = 1e-6, conlim::Real = 0,
                             verbose::Bool = false)
    T = real(eltype(C))
    comm = _comm(backend)
    x = ws.x; v = ws.v; h = ws.h; hbar = ws.hbar; w = ws.w; u = ws.u; Av = ws.Av
    isroot = MPI.Comm_rank(comm) == 0        # hoisted: was an MPI call per iteration
    clim = conlim > 0 ? T(conlim) : one(T) / eps(T)

    # Restrict the fit to degrees `l ≤ lmax`, as the serial solver does: the coefficient array is a
    # square representation carrying degrees up to `lmax+|m|`, and that ragged set is not
    # SO(3)-invariant, so fitting it gives a frame-dependent answer.
    valid = NUFSHT._valid_mask(C, T, plan.lmax)

    # A†u, summed across ranks and projected — the operator's adjoint half.
    atu!(dst, src) = (NUFSHT._nusht_true_adjoint!(dst, src, plan);
                      MPI.Allreduce!(dst, +, comm); dst .*= valid; dst)

    copyto!(u, f_local)
    β = sqrt(_global_sq(u, comm))
    β > 0 && (u ./= β)
    atu!(w, u); copyto!(v, w)
    α = sqrt(_local_sq(v))
    α > 0 && (v ./= α)

    fill!(x, zero(eltype(C))); fill!(hbar, zero(eltype(C))); copyto!(h, v)
    αbar = α; ζbar = α * β; atb = α * β
    ρ = one(T); ρbar = one(T); cbar = one(T); sbar = zero(T)
    maxrbar = zero(T); minrbar = T(Inf)
    rel = atb > 0 ? one(T) : zero(T)
    iters = 0
    atb == 0 && (fill!(C, zero(eltype(C))); return C, 0, zero(T), true)

    for i in 1:maxiter
        iters = i
        NUFSHT.nusht_type2!(Av, v, plan)            # A_local v (no communication)
        @. u = Av - α * u
        β = sqrt(_global_sq(u, comm))
        β > 0 && (u ./= β)
        atu!(w, u)
        @. v = w - β * v
        α = sqrt(_local_sq(v))
        α > 0 && (v ./= α)

        ρold = ρ; ρbarold = ρbar
        r = hypot(αbar, β)
        c = r > 0 ? αbar / r : one(T)
        s = r > 0 ? β / r : zero(T)
        θnew = s * α
        αbar = c * α
        ρ = r

        θbar = sbar * r
        ρtemp = cbar * r
        rb = hypot(ρtemp, θnew)
        cbar = rb > 0 ? ρtemp / rb : one(T)
        sbar = rb > 0 ? θnew / rb : zero(T)
        ρbar = rb
        ζ = cbar * ζbar
        ζbar = -sbar * ζbar

        maxrbar = max(maxrbar, ρbarold)
        i > 1 && (minrbar = min(minrbar, ρbarold))
        condA = max(maxrbar, ρtemp) / max(min(minrbar, ρtemp), eps(T))

        c1 = ρold * ρbarold > 0 ? -(θbar * r / (ρold * ρbarold)) : zero(T)
        @. hbar = h + c1 * hbar
        r * rb > 0 && (@. x += (ζ / (r * rb)) * hbar)
        c3 = r > 0 ? -(θnew / r) : zero(T)
        @. h = v + c3 * h

        rel = abs(ζbar) / atb
        (verbose && isroot) && @info "nusht_solve! (MPI) iter $i rel_res=$rel"
        (rel < rtol || condA >= clim || !(r > 0) || !(rb > 0) || α == 0 || β == 0) && break
    end
    copyto!(C, x)
    return C, iters, rel, rel < rtol
end

end # module NUFSHTMPIExt