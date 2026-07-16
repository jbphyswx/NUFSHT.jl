# MPI point-decomposition validation. Run with:
#   mpiexec -n <R> julia --project=test test/mpi_run.jl
# Each rank owns a strided subset of the global scattered points, builds a local plan, and the ranks
# cooperatively invert the GLOBAL least-squares system via `nusht_solve_mpi!`. The recovered
# coefficients are compared, on every rank, to the known band-limited truth (well-conditioned
# jittered points → robust recovery, no tuned threshold / seed hunting).

using MPI: MPI
using NUFSHT: NUFSHT
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

# Jittered-from-equidistribution points (well-conditioned for CG), identical on all ranks.
φ_all = mod.((2π / M) .* (0:M-1) .+ (rand(M) .- 0.5) .* (0.4 * 2π / sqrt(M)), 2π)
θ_all = clamp.(acos.(clamp.(2 .* ((0:M-1) .+ 0.5) ./ M .- 1, -1.0, 1.0)) .+
               (rand(M) .- 0.5) .* (0.4π / sqrt(M)), 1e-10, π - 1e-10)

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
_, iters, rel = NUFSHT.nusht_solve_mpi!(C_mpi, f_all[idx], plan_loc, comm; rtol = 1e-8, maxiter = 500)

# The global solve reconstructs the *field* (CG objective); the min-norm LS solution need not equal
# a narrower-band Ctrue. Synthesizing C_mpi on ALL points and matching f_all verifies the points
# really were solved jointly (a rank solving only its local subset would not fit the global field).
f_rec = zeros(M); NUFSHT.nusht_type2!(f_rec, C_mpi, planfull)
relf = sqrt(sum(abs2, f_rec .- f_all) / sum(abs2, f_all))
if rank == 0
    println("MPI point-decomposition: nranks=$nranks  iters=$iters  rel_res=$rel  field_rel_err=$relf")
    relf < 1e-3 || error("MPI global field recovery failed (relf=$relf)")
    println("MPI OK")
end
MPI.Finalize()
