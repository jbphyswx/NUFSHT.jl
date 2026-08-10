using Distributed: Distributed   # triggers NUFSHTDistributedExt
using ComputationalBackends: ComputationalBackends

# Farming independent problems across Distributed *worker processes* (process isolation makes
# FastTransforms safe — each worker has its own OpenMP/FFTW state). We add real workers so the test
# exercises the actual cross-process path, not the local serial fallback. Well-conditioned canonical
# problem (matches test_solve.jl); nthreads=1 farm plans (deterministic, no core oversubscription).
Test.@testset "Distributed extension: farm over worker processes" begin
    nadd = 2
    added = Distributed.addprocs(nadd; exeflags = "--project=$(Base.active_project())")
    try
        Distributed.@everywhere added begin
            using NUFSHT: NUFSHT
            using FastSphericalHarmonics: FastSphericalHarmonics
        end

        Random.seed!(21)
        lmax = 10
        N = lmax + 1
        Nφ = 2lmax + 1
        N_modes = N^2
        M = 4 * N_modes            # 4× overdetermined, well-conditioned
        P = 4                      # independent problems (distinct point sets → distinct plans)

        function jittered(m)
            φ = mod.((2π / m) .* (0:m-1) .+ (rand(m) .- 0.5) .* (0.4 * 2π / sqrt(m)), 2π)
            θ = clamp.(acos.(clamp.(2 .* ((0:m-1) .+ 0.5) ./ m .- 1, -1.0, 1.0)) .+
                       (rand(m) .- 0.5) .* (0.4 * π / sqrt(m)), 1e-10, π - 1e-10)
            return collect(θ), collect(φ)
        end
        probs = [jittered(M) for _ in 1:P]
        θs = [p[1] for p in probs]
        φs = [p[2] for p in probs]

        Ctrue = [zeros(N, Nφ) for _ in 1:P]
        for i in 1:P, ℓ in 1:4, m in -ℓ:ℓ
            Ctrue[i][FastSphericalHarmonics.sph_mode(ℓ, m)] = randn()
        end

        # Synthesis farm: type-2 is a deterministic gather (nthreads=1), so the farm must match the
        # serial transform bit-for-bit — validates the pmap-to-workers dispatch + plan-per-worker.
        fs = NUFSHT.nusht_type2(θs, φs, Ctrue, lmax, ComputationalBackends.DistributedBackend(); tol = 1e-10, nthreads = 1)
        for i in 1:P
            plan = NUFSHT.make_plan(θs[i], φs[i], lmax; tol = 1e-10, nthreads = 1)
            fref = zeros(M); NUFSHT.nusht_type2!(fref, Ctrue[i], plan)
            Test.@test fs[i] ≈ fref
            NUFSHT.close!(plan)
        end

        # Inversion farm: each solve reconstructs its field to CG tolerance.
        Cs = NUFSHT.nusht_solve(θs, φs, fs, lmax, ComputationalBackends.DistributedBackend();
                                            tol = 1e-10, nthreads = 1, rtol = 1e-6, maxiter = 1000)
        for i in 1:P
            plan = NUFSHT.make_plan(θs[i], φs[i], lmax; tol = 1e-10, nthreads = 1)
            frec = zeros(M); NUFSHT.nusht_type2!(frec, Cs[i], plan)
            Test.@test sqrt(sum(abs2, frec .- fs[i]) / sum(abs2, fs[i])) < 1e-3
            NUFSHT.close!(plan)
        end
        @info "Distributed farm validated over $P problems ($(Distributed.nworkers()) worker process(es))"
    finally
        Distributed.rmprocs(added)
    end
end
