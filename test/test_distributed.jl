using Distributed: Distributed   # triggers NUFSHTDistributedExt
using ComputationalBackends: ComputationalBackends

# Independent problems farmed over worker processes started with two threads each, every result scored
# against the same problem run on the calling process. The NUFFT is named, so a worker and the
# calling process run the same library.
Test.@testset "Distributed extension: farm over worker processes" begin
    ext = Base.get_extension(NUFSHT, :NUFSHTDistributedExt)
    added = Distributed.addprocs(2; exeflags = ["--project=$(Base.active_project())", "--threads=2"])
    try
        Distributed.@everywhere added begin
            using NUFSHT: NUFSHT
            using FINUFFT: FINUFFT
            using Distributed: Distributed
            using FlowTransformBindings: FlowTransformBindings as FTB
        end

        # A problem runs on a worker, with FastTransforms at the worker's own thread count.
        runs = ext._farm(i -> (Distributed.myid(), Threads.nthreads(), FTB.FASTTRANSFORMS_THREADS[]), 4)
        Test.@test all(r -> r[1] in added && r[2] == 2 && r[3] == 2, runs)

        Random.seed!(21)
        lmax = 10
        N = lmax + 1
        Nφ = 2lmax + 1
        M = 4 * N^2                # 4× overdetermined
        P = 4                      # distinct point sets, so distinct plans
        nufft = FTB.FINUFFTBackend()
        probs = [iid_points(M, 30 + i) for i in 1:P]
        θs = [p[1] for p in probs]
        φs = [p[2] for p in probs]

        Ctrue = [zeros(N, Nφ) for _ in 1:P]
        for i in 1:P, ℓ in 1:4, m in -ℓ:ℓ
            Ctrue[i][FastSphericalHarmonics.sph_mode(ℓ, m)] = randn()
        end

        fs = NUFSHT.nusht_type2(θs, φs, Ctrue, lmax; execution = ComputationalBackends.DistributedBackend(),
                                tol = 1e-10, nthreads = 1, nufft)
        for i in 1:P
            Test.@test relerr(fs[i], synth_ref(Ctrue[i], lmax, θs[i], φs[i])) < 1e-9
        end

        Cs = NUFSHT.nusht_solve(θs, φs, fs, lmax; execution = ComputationalBackends.DistributedBackend(),
                                tol = 1e-10, nthreads = 1, nufft, rtol = 1e-10, maxiter = 1000)
        for i in 1:P
            plan = NUFSHT.make_plan(θs[i], φs[i], lmax; tol = 1e-10, nthreads = 1, nufft)
            C = zeros(N, Nφ)
            _, _, _, conv = NUFSHT.nusht_solve!(C, fs[i], plan; rtol = 1e-10, maxiter = 1000)
            Test.@test conv
            Test.@test Cs[i] == C
            Test.@test relerr(C, Ctrue[i]) < 1e-8
            NUFSHT.close!(plan)
        end

        # One transform divided among the workers, each holding a plan over a block of the points, against
        # the plan over all of them on the calling process.
        CB = ComputationalBackends
        θ, φ = iid_points(M, 41)
        kw = (; tol = 1e-10, nufft)
        serial = NUFSHT.make_plan(Float64, θ, φ, lmax; nthreads = 1, kw...)
        dp = NUFSHT.make_plan(Float64, θ, φ, lmax; execution = CB.DistributedBackend(), kw...)
        Test.@test dp.workers == added

        fs = zeros(M); fd = zeros(M)
        NUFSHT.nusht_type2!(fs, Ctrue[1], serial)
        NUFSHT.nusht_type2!(fd, Ctrue[1], dp)
        Test.@test relerr(fd, synth_ref(Ctrue[1], lmax, θ, φ)) < 1e-9
        Test.@test isapprox(fd, fs; rtol = 1e-12)
        filt = NUFSHT.gaussian_from_scale(2e6)
        NUFSHT.nusht_synthesize!(fs, Ctrue[1], filt, serial)
        NUFSHT.nusht_synthesize!(fd, Ctrue[1], filt, dp)
        Test.@test isapprox(fd, fs; rtol = 1e-12)

        # A noisy field has a least-squares fit that depends on every point, so a block's points alone fit
        # other coefficients.
        NUFSHT.nusht_type2!(fs, Ctrue[1], serial)
        g = fs .+ 0.05 .* randn(M)
        solve(p, f) = (C = NUFSHT.allocate_coefficients(p);
                       (C, NUFSHT.nusht_solve!(C, f, p; rtol = 1e-10, maxiter = 1000)[4]))
        Cs, conv_s = solve(serial, g)
        Cd, conv_d = solve(dp, g)
        Test.@test conv_s && conv_d
        Test.@test relerr(Cd, Cs) < 1e-8
        block = NUFSHT.make_plan(Float64, θ[dp.blocks[1]], φ[dp.blocks[1]], lmax; nthreads = 1, kw...)
        Test.@test relerr(solve(block, g[dp.blocks[1]])[1], Cs) > 1e-3
        NUFSHT.close!(block)

        # A worker's failure reaches the caller, and the plan solves again afterwards.
        Test.@test_throws Distributed.RemoteException NUFSHT.nusht_solve!(copy(Cd), g, dp; unknown_keyword = 1)
        Test.@test relerr(solve(dp, g)[1], Cs) < 1e-8

        # The adjoint and the quadrature are sums over every block, and the filter a fit and a synthesis.
        As = NUFSHT.allocate_coefficients(serial); Ad = NUFSHT.allocate_coefficients(dp)
        NUFSHT.nusht_type1!(As, g, serial)
        NUFSHT.nusht_type1!(Ad, g, dp)
        Test.@test isapprox(Ad, As; rtol = 1e-12)
        wq = (4π / M) .* (0.5 .+ rand(M))
        rq = NUFSHT.nusht_analyze!(Ad, g, dp; weights = wq)
        Test.@test rq.iterations == 0 && rq.converged
        Test.@test relerr(Ad[:, :, 1], quadrature_ref(g, wq, lmax, θ, φ)) < 1e-9
        gs = zeros(M); gd = zeros(M)
        NUFSHT.nusht_filter!(gs, g, filt, serial; rtol = 1e-10, maxiter = 1000)
        NUFSHT.nusht_filter!(gd, g, filt, dp; rtol = 1e-10, maxiter = 1000)
        Test.@test relerr(gd, gs) < 1e-8

        # A batch, on workers running their parts threaded.
        G = hcat(g, g .^ 2)
        serial2 = NUFSHT.make_plan(Float64, θ, φ, lmax; nthreads = 1, ntrans = 2, kw...)
        dp2 = NUFSHT.make_plan(Float64, θ, φ, lmax; execution = CB.DistributedBackend(CB.ThreadedBackend()),
                               ntrans = 2, kw...)
        C2 = solve(serial2, G)[1]
        Test.@test relerr(solve(dp2, G)[1], C2) < 1e-8
        Fs = zeros(M, 2); Fd = zeros(M, 2)
        NUFSHT.nusht_synthesize!(Fs, C2, filt, serial2)
        NUFSHT.nusht_synthesize!(Fd, C2, filt, dp2)
        Test.@test isapprox(Fd, Fs; rtol = 1e-12)
        foreach(NUFSHT.close!, (serial, dp, serial2, dp2))
    finally
        Distributed.rmprocs(intersect(added, Distributed.workers()))
    end
end
