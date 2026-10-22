# MPI point decomposition. Run with MPI.jl's launcher, `MPI.mpiexec()`:
#
#      julia --project=test using MPI: MPI; MPI.mpiexec(<test/mpi_run.jl>)
#
# Each rank holds a plan over a strided share of the points, and `nusht_solve!(C, f_local, plan,
# MPIBackend(; comm))` solves the least-squares problem over all of them. The coefficients are compared
# with the band-limited field's own.

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
solve(C, f, plan, e...) = NUFSHT.nusht_solve!(C, f, plan, e...; rtol = 1e-10, maxiter = 1000)

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

# Point weights: the quadrature is a sum over every rank's points, and the weighted fit the weighted
# least squares over all of them.
w_all = (4π / M) .* (0.5 .+ rand(M))
Q_mpi = zeros(Nθ, Nφ); Q_all = zeros(Nθ, Nφ)
NUFSHT.nusht_analyze!(Q_mpi, f_noisy[idx], plan_loc, mpi; weights = w_all[idx])
NUFSHT.nusht_analyze!(Q_all, f_noisy, planfull; weights = w_all)
check(rel(Q_mpi, Q_all) < 1e-12, "MPI quadrature equals the all-points quadrature")
wsolve(C, f, plan, w, e...) = NUFSHT.nusht_solve!(C, f, plan, e...; weights = w, rtol = 1e-10, maxiter = 1000)
Cw_mpi = zeros(Nθ, Nφ); Cw_all = zeros(Nθ, Nφ)
check(wsolve(Cw_all, f_noisy, planfull, w_all)[4], "all-points weighted solve converged")
check(wsolve(Cw_mpi, f_noisy[idx], plan_loc, w_all[idx], mpi)[4], "MPI weighted solve converged")
check(rel(Cw_mpi, Cw_all) < 1e-8, "MPI weighted solve equals the all-points weighted solve")

# The same for a complex spin-1 field, each rank's spin plan over its own points.
u = complex.(f_noisy, f_band)
sp_all = NUFSHT.make_spin_plan(ComplexF64, θ_all, φ_all, lmax, 1; tol = 1e-11, nthreads = 1)
sp_loc = NUFSHT.make_spin_plan(ComplexF64, θ_all[idx], φ_all[idx], lmax, 1; tol = 1e-11, nthreads = 1)
Sw_all = NUFSHT.allocate_coefficients(sp_all); Sw_mpi = NUFSHT.allocate_coefficients(sp_loc)
check(NUFSHT.nusht_solve_spin!(Sw_all, u, sp_all; weights = w_all, rtol = 1e-10, maxiter = 1000)[4],
      "all-points weighted spin solve converged")
check(NUFSHT.nusht_solve_spin!(Sw_mpi, u[idx], sp_loc, mpi; weights = w_all[idx], rtol = 1e-10,
                               maxiter = 1000)[4], "MPI weighted spin solve converged")
check(rel(Sw_mpi, Sw_all) < 1e-8, "MPI weighted spin solve equals the all-points weighted spin solve")
NUFSHT.close!(sp_all); NUFSHT.close!(sp_loc)

# `make_plan(…; execution = MPIBackend())` takes every point and takes and returns whole fields.
mp = NUFSHT.make_plan(Float64, θ_all, φ_all, lmax; execution = mpi, tol = 1e-11)
f_mp = zeros(M); NUFSHT.nusht_type2!(f_mp, Ctrue, mp)
check(rel(f_mp, f_band) < 1e-12, "MPI plan synthesis equals the all-points synthesis")
A_mp = zeros(Nθ, Nφ); A_all = zeros(Nθ, Nφ)
NUFSHT.nusht_type1!(A_mp, f_noisy, mp); NUFSHT.nusht_type1!(A_all, f_noisy, planfull)
check(rel(A_mp, A_all) < 1e-12, "MPI plan adjoint equals the all-points adjoint")
C_mp = zeros(Nθ, Nφ)
check(solve(C_mp, f_noisy, mp)[4], "MPI plan solve converged")
check(rel(C_mp, C_all) < 1e-8, "MPI plan solve equals the all-points solve")
Q_mp = zeros(Nθ, Nφ)
NUFSHT.nusht_analyze!(Q_mp, f_noisy, mp; weights = w_all)
check(rel(Q_mp, Q_all) < 1e-12, "MPI plan quadrature equals the all-points quadrature")
NUFSHT.close!(mp)

rank == 0 && println("MPI point decomposition: nranks=$nranks iters=$iters rel_res=$res; failures: ", failures)
isempty(failures) || error("rank $rank: MPI checks failed: $(join(failures, "; "))")
rank == 0 && println("MPI OK")
MPI.Finalize()
