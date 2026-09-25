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

        fs = NUFSHT.nusht_type2(θs, φs, Ctrue, lmax, ComputationalBackends.DistributedBackend();
                                tol = 1e-10, nthreads = 1, nufft)
        for i in 1:P
            Test.@test relerr(fs[i], synth_ref(Ctrue[i], lmax, θs[i], φs[i])) < 1e-9
        end

        Cs = NUFSHT.nusht_solve(θs, φs, fs, lmax, ComputationalBackends.DistributedBackend();
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

        # One worker process takes the problems too.
        Distributed.rmprocs(added[2])
        Test.@test ext._farm(i -> Distributed.myid(), 3) == fill(added[1], 3)
    finally
        Distributed.rmprocs(intersect(added, Distributed.workers()))
    end
end
