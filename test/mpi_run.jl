# MPI point-decomposition validation. Run with:
#   mpiexec -n <R> julia --project=test test/mpi_run.jl
# Each rank owns a strided subset of the global scattered points, builds a local plan, and the ranks
# cooperatively invert the GLOBAL least-squares system via `nusht_solve!` under `MPIBackend`. The
# recovered coefficients are compared to the known band-limited truth (well-conditioned jittered
# points → robust recovery, no tuned threshold / seed hunting).

using MPI: MPI
using NUFSHT: NUFSHT
using ComputationalBackends: ComputationalBackends
using FastSphericalHarmonics: FastSphericalHarmonics
using Random: Random

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

Random.seed!(2024)                       # identical global problem on every rank
lmax = 8
Nθ, Nφ = lmax + 1, 2lmax + 1
K = Nθ * Nφ
M = 6 * K                                # overdetermined

# Area-uniform random points, identical on all ranks. An index-linked spiral (φ advancing 2π/M per
# point while θ sweeps pole to pole) winds only once and leaves the design matrix near-degenerate, so
# the coefficients are not identifiable from the field even though the field itself is fit.
φ_all = 2π .* rand(M)
θ_all = clamp.(acos.(2 .* rand(M) .- 1), 1e-10, π - 1e-10)

Ctrue = zeros(Nθ, Nφ)
for ℓ in 1:min(5, lmax), m in -ℓ:ℓ
    Ctrue[FastSphericalHarmonics.sph_mode(ℓ, m)] = randn()
end

# Global field on all points (nthreads=1 for a deterministic reference), then partition.
planfull = NUFSHT.make_plan(collect(θ_all), collect(φ_all), lmax; tol = 1e-11, nthreads = 1)
f_all = zeros(M); NUFSHT.nusht_type2!(f_all, Ctrue, planfull)

idx = (rank + 1):nranks:M                # disjoint strided partition
plan_loc = NUFSHT.make_plan(θ_all[idx], φ_all[idx], lmax; tol = 1e-11, nthreads = 1)

C_mpi = zeros(Nθ, Nφ)
_, iters, rel, conv = NUFSHT.nusht_solve!(C_mpi, f_all[idx], plan_loc,
                                          ComputationalBackends.MPIBackend(; comm = comm);
                                          rtol = 1e-8, maxiter = 500)

# `Ctrue` is band-limited and the solve fits the same `l ≤ lmax` space, so the coefficients themselves
# must be recovered — not merely a field that matches. A rank solving only its local subset would fail
# this, as would a solve that spent energy on the supernumerary `l > lmax` slots.
relc = sqrt(sum(abs2, C_mpi .- Ctrue) / sum(abs2, Ctrue))
f_rec = zeros(M); NUFSHT.nusht_type2!(f_rec, C_mpi, planfull)
relf = sqrt(sum(abs2, f_rec .- f_all) / sum(abs2, f_all))
if rank == 0
    println("MPI point-decomposition: nranks=$nranks  iters=$iters  rel_res=$rel  converged=$conv  " *
            "coeff_rel_err=$relc  field_rel_err=$relf")
    conv || error("MPI solve reported not converged (rel=$rel)")
    relc < 1e-6 || error("MPI coefficient recovery failed (relc=$relc)")
    relf < 1e-3 || error("MPI global field recovery failed (relf=$relf)")
    println("MPI OK")
end
MPI.Finalize()
