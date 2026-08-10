using OhMyThreads: OhMyThreads   # triggers NUFSHTOhMyThreadsExt
using ComputationalBackends: ComputationalBackends

# Node-local thread parallelism over independent problems. FastTransforms is unsafe to *drive* from a
# Julia task, so the extension forces it single-threaded around the whole threaded section; the plans
# are built with `nthreads = 1` (single-threaded FINUFFT) so each threaded transform is deterministic
# and cores are not oversubscribed across tasks. With those, a threaded transform is bit-for-bit the
# serial one (verified below), and the main task is uncorrupted afterward.
Test.@testset "OhMyThreads extension: threaded == serial, main task intact" begin
    Random.seed!(11)
    lmax = 10
    N = lmax + 1
    Nφ = 2lmax + 1
    N_modes = N^2
    M = 4 * N_modes                       # 4× overdetermined, well-conditioned (per test_solve.jl)
    P = 4                                 # independent problems (distinct point sets → distinct plans)

    # Jittered-from-equidistribution points: well-conditioned for CG at any seed.
    function jittered(m)
        φ = mod.((2π / m) .* (0:m-1) .+ (rand(m) .- 0.5) .* (0.4 * 2π / sqrt(m)), 2π)
        θ = clamp.(acos.(clamp.(2 .* ((0:m-1) .+ 0.5) ./ m .- 1, -1.0, 1.0)) .+
                   (rand(m) .- 0.5) .* (0.4 * π / sqrt(m)), 1e-10, π - 1e-10)
        return collect(θ), collect(φ)
    end
    probs = [jittered(M) for _ in 1:P]
    plans = [NUFSHT.make_plan(probs[i][1], probs[i][2], lmax; tol = 1e-10, nthreads = 1) for i in 1:P]
    Cs = [zeros(N, Nφ) for _ in 1:P]
    for i in 1:P, ℓ in 1:4, m in -ℓ:ℓ
        Cs[i][FastSphericalHarmonics.sph_mode(ℓ, m)] = randn()
    end

    # Serial references (main task) computed BEFORE the threaded section, so a corrupted threaded run
    # cannot hide by also corrupting the reference.
    fref = [zeros(M) for _ in 1:P]
    for i in 1:P
        NUFSHT.nusht_type2!(fref[i], Cs[i], plans[i])
    end
    health_before = zeros(M); NUFSHT.nusht_type2!(health_before, Cs[1], plans[1])

    fs = [zeros(M) for _ in 1:P]
    NUFSHT.nusht_type2!(fs, Cs, plans, ComputationalBackends.ThreadedBackend())
    for i in 1:P
        Test.@test fs[i] ≈ fref[i]
    end

    Cs_out = [zeros(N, Nφ) for _ in 1:P]
    NUFSHT.nusht_type1!(Cs_out, fs, plans, ComputationalBackends.ThreadedBackend())
    for i in 1:P
        Cref = zeros(N, Nφ); NUFSHT.nusht_type1!(Cref, fs[i], plans[i])
        Test.@test Cs_out[i] ≈ Cref
    end

    # Threaded solve: each recovers its field to CG tolerance.
    Cs_sol = [zeros(N, Nφ) for _ in 1:P]
    NUFSHT.nusht_solve!(Cs_sol, fs, plans, ComputationalBackends.ThreadedBackend(); rtol = 1e-6, maxiter = 1000)
    for i in 1:P
        frec = zeros(M); NUFSHT.nusht_type2!(frec, Cs_sol[i], plans[i])
        Test.@test sqrt(sum(abs2, frec .- fs[i]) / sum(abs2, fs[i])) < 1e-3
    end

    # Main task must be uncorrupted by the threaded work (FastTransforms-in-task hazard).
    health_after = zeros(M); NUFSHT.nusht_type2!(health_after, Cs[1], plans[1])
    Test.@test health_after ≈ health_before

    @info "OhMyThreads ext validated with $(Threads.nthreads()) thread(s), P=$P problems (incl. solve)"
end
