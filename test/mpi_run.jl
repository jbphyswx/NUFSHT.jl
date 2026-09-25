# MPI point decomposition. Run with MPI.jl's launcher, `MPI.mpiexec()`:
#
#      julia --project=test using MPI: MPI; MPI.mpiexec(<test/mpi_run.jl>)
#
# Each rank holds a plan over a strided share of the points, and `nusht_solve!` under `MPIBackend` solves
# the least-squares problem over all of them. The coefficients are compared with the band-limited field's
# own.

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

# Area-uniform random points, identical on all ranks.
φ_all = 2π .* rand(M)
θ_all = clamp.(acos.(2 .* rand(M) .- 1), 1e-10, π - 1e-10)

Ctrue = zeros(Nθ, Nφ)
for ℓ in 1:min(5, lmax), m in -ℓ:ℓ
    Ctrue[FastSphericalHarmonics.sph_mode(ℓ, m)] = randn()
end

rel(a, b) = sqrt(sum(abs2, a .- b) / sum(abs2, b))
solve(C, f, plan, args...) = NUFSHT.nusht_solve!(C, f, plan, args...; rtol = 1e-10, maxiter = 1000)

planfull = NUFSHT.make_plan(collect(θ_all), collect(φ_all), lmax; tol = 1e-11, nthreads = 1)
f_band = zeros(M); NUFSHT.nusht_type2!(f_band, Ctrue, planfull)
idx = (rank + 1):nranks:M
plan_loc = NUFSHT.make_plan(θ_all[idx], φ_all[idx], lmax; tol = 1e-11, nthreads = 1)
mpi = ComputationalBackends.MPIBackend(; comm = comm)
failures = String[]
check(ok, what) = ok || push!(failures, what)

# `Ctrue` is band-limited and the solve fits the same `l ≤ lmax` space, so the fit recovers the
# coefficients themselves.
C_mpi = zeros(Nθ, Nφ)
_, iters, res, conv = solve(C_mpi, f_band[idx], plan_loc, mpi)
check(conv, "band-limited solve converged")
check(rel(C_mpi, Ctrue) < 1e-8, "band-limited coefficients recovered")

# A noisy field's fit depends on every point: the MPI solve equals the solve over all of them, and a
# rank's own points fit other coefficients.
f_noisy = f_band .+ 0.05 .* randn(M)
C_all = zeros(Nθ, Nφ); C_loc = zeros(Nθ, Nφ)
check(solve(C_all, f_noisy, planfull)[4], "all-points solve converged")
check(solve(C_mpi, f_noisy[idx], plan_loc, mpi)[4], "noisy MPI solve converged")
check(rel(C_mpi, C_all) < 1e-8, "MPI solve equals the all-points solve")
solve(C_loc, f_noisy[idx], plan_loc)
nranks > 1 && check(rel(C_loc, C_all) > 1e-3, "one rank's points fit other coefficients")

# `make_plan(…, MPIBackend)` takes every point, holds this rank's, and takes and returns whole fields.
mp = NUFSHT.make_plan(Float64, θ_all, φ_all, lmax, mpi; tol = 1e-11)
check(mp.own == idx, "the MPI plan holds this rank's points")
f_mp = zeros(M); NUFSHT.nusht_type2!(f_mp, Ctrue, mp)
check(rel(f_mp, f_band) < 1e-12, "MPI plan synthesis equals the all-points synthesis")
A_mp = zeros(Nθ, Nφ); A_all = zeros(Nθ, Nφ)
NUFSHT.nusht_type1!(A_mp, f_noisy, mp); NUFSHT.nusht_type1!(A_all, f_noisy, planfull)
check(rel(A_mp, A_all) < 1e-12, "MPI plan adjoint equals the all-points adjoint")
C_mp = zeros(Nθ, Nφ)
check(solve(C_mp, f_noisy, mp)[4], "MPI plan solve converged")
check(rel(C_mp, C_all) < 1e-8, "MPI plan solve equals the all-points solve")
NUFSHT.close!(mp)

rank == 0 && println("MPI point decomposition: nranks=$nranks iters=$iters rel_res=$res; failures: ", failures)
isempty(failures) || error("rank $rank: MPI checks failed: $(join(failures, "; "))")
rank == 0 && println("MPI OK")
MPI.Finalize()
