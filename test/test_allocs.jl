# Hot-path methods allocate nothing once warm. Each is measured inside a barrier of fixed arity whose
# parameters carry the concrete buffer types, after a plain call (one-time per-plan setup) and an
# `@allocated` call (compiles the measured specialisation). Every plan runs on one task.
_a_type2(f, C, p)        = (NUFSHT.nusht_type2!(f, C, p); @allocated NUFSHT.nusht_type2!(f, C, p); @allocated NUFSHT.nusht_type2!(f, C, p))
_a_type1(C, f, p)        = (NUFSHT.nusht_type1!(C, f, p); @allocated NUFSHT.nusht_type1!(C, f, p); @allocated NUFSHT.nusht_type1!(C, f, p))
_a_filter(o, i, ft, p, w) = (NUFSHT.nusht_filter!(o, i, ft, p; ws = w); @allocated NUFSHT.nusht_filter!(o, i, ft, p; ws = w); @allocated NUFSHT.nusht_filter!(o, i, ft, p; ws = w))
_a_renorm(o, m, ft, p, s, w) = (NUFSHT.nusht_filter_renorm!(o, m, ft, p; mask_filt = s, ws = w); @allocated NUFSHT.nusht_filter_renorm!(o, m, ft, p; mask_filt = s, ws = w); @allocated NUFSHT.nusht_filter_renorm!(o, m, ft, p; mask_filt = s, ws = w))
_a_solve(C, f, p, ws)    = (NUFSHT.nusht_solve!(C, f, p; ws = ws, maxiter = 60, rtol = 1e-8); @allocated NUFSHT.nusht_solve!(C, f, p; ws = ws, maxiter = 60, rtol = 1e-8); @allocated NUFSHT.nusht_solve!(C, f, p; ws = ws, maxiter = 60, rtol = 1e-8))
_a_type2s(f, sf, p)      = (NUFSHT.nusht_type2_spin!(f, sf, p); @allocated NUFSHT.nusht_type2_spin!(f, sf, p); @allocated NUFSHT.nusht_type2_spin!(f, sf, p))
_a_type1s(sf, f, p)      = (NUFSHT.nusht_type1_spin!(sf, f, p); @allocated NUFSHT.nusht_type1_spin!(sf, f, p); @allocated NUFSHT.nusht_type1_spin!(sf, f, p))
_a_solves(sf, f, p, ws)  = (NUFSHT.nusht_solve_spin!(sf, f, p; ws = ws, maxiter = 60, rtol = 1e-8); @allocated NUFSHT.nusht_solve_spin!(sf, f, p; ws = ws, maxiter = 60, rtol = 1e-8); @allocated NUFSHT.nusht_solve_spin!(sf, f, p; ws = ws, maxiter = 60, rtol = 1e-8))
_a_quad(C, f, p, w)      = (NUFSHT.nusht_analyze!(C, f, p; weights = w); @allocated NUFSHT.nusht_analyze!(C, f, p; weights = w); @allocated NUFSHT.nusht_analyze!(C, f, p; weights = w))
_a_fit(C, f, p, ws)      = (NUFSHT.nusht_analyze!(C, f, p; ws = ws, maxiter = 60, rtol = 1e-8); @allocated NUFSHT.nusht_analyze!(C, f, p; ws = ws, maxiter = 60, rtol = 1e-8); @allocated NUFSHT.nusht_analyze!(C, f, p; ws = ws, maxiter = 60, rtol = 1e-8))

Test.@testset "allocation: full hot-path surface is allocation-free" begin
    Random.seed!(9)
    lmax = 8
    Nθ, Nφ = lmax + 1, 2lmax + 1
    M = 4 * (lmax + 1)^2
    θ, φ = fib_points(M)          # cond(A) ≈ 1.04, so the solve converges in a few iterations
    w = fill(4π / M, M)
    filt = NUFSHT.gaussian_from_scale(2000e3)
    serial = NUFSHT.ComputationalBackends.SerialBackend()

    # FINUFFT's exec allocates nothing, so on it a whole-transform count is NUFSHT's own.
    for B in (1, 2)
        Test.@testset "scalar transform + solve (B=$B)" begin
            plan = NUFSHT.make_plan(θ, φ, lmax; tol = 1e-10, ntrans = B, execution = serial,
                                    nufft = FTB.FINUFFTBackend())
            C = randn(Nθ, Nφ, B); f = zeros(M, B); Cout = zeros(Nθ, Nφ, B); out = zeros(M, B)
            NUFSHT.nusht_type2!(f, C, plan)
            mask = abs.(randn(M, B)) .+ 0.5; scratch = similar(out)
            C_true = zeros(Nθ, Nφ, B)
            for b in 1:B, ℓ in 1:min(4, lmax), m in -ℓ:ℓ
                C_true[FastSphericalHarmonics.sph_mode(ℓ, m), b] = randn()
            end
            ftrue = zeros(M, B); NUFSHT.nusht_type2!(ftrue, C_true, plan)
            Csol = NUFSHT.allocate_coefficients(plan); ws = FTB.LSMRWorkspace(plan)
            for _ in 1:3
                NUFSHT.nusht_type1!(Cout, f, plan)
                NUFSHT.nusht_filter!(out, f, filt, plan; ws = ws)
                NUFSHT.nusht_solve!(Csol, ftrue, plan; ws = ws, maxiter = 60, rtol = 1e-8)
            end
            Test.@test _a_type2(f, C, plan)           == 0
            Test.@test _a_type1(Cout, f, plan)        == 0
            Test.@test _a_filter(out, f, filt, plan, ws) == 0
            Test.@test _a_renorm(out, mask, filt, plan, scratch, ws) == 0
            Test.@test _a_solve(Csol, ftrue, plan, ws) == 0
            Test.@test _a_quad(Cout, f, plan, w) == 0
            Test.@test _a_fit(Csol, ftrue, plan, ws) == 0
            if B == 2
                # Column 2 is zero and retires at once, so the solve narrows to width 1.
                fz = copy(ftrue); fz[:, 2] .= 0
                Test.@test _a_solve(Csol, fz, plan, ws) == 0
                Test.@test NUFSHT._pool_built(plan.size_pool) == 1
            end
            NUFSHT.close!(plan)
        end

        Test.@testset "spin transform + solve (B=$B)" begin
            s = 1
            plan = NUFSHT.make_spin_plan(θ, φ, lmax, s; tol = 1e-10, ntrans = B, execution = serial,
                                         nufft = FTB.FINUFFTBackend())
            Test.@test plan.engine isa NUFSHT._RingEngine && length(plan.engine.work) == 1
            sf = zeros(ComplexF64, Nθ, Nφ, B); fs = zeros(ComplexF64, M, B); sfo = zeros(ComplexF64, Nθ, Nφ, B)
            for b in 1:B, ℓ in abs(s):min(4, lmax), m in -ℓ:ℓ
                sf[NUFSHT.spin_coeff_index(ℓ, m, lmax), b] = randn(ComplexF64)
            end
            NUFSHT.nusht_type2_spin!(fs, sf, plan)
            sws = FTB.LSMRWorkspace(plan)
            for _ in 1:3
                NUFSHT.nusht_type1_spin!(sfo, fs, plan)
                NUFSHT.nusht_solve_spin!(sfo, fs, plan; ws = sws, maxiter = 60, rtol = 1e-8)
            end
            Test.@test _a_type2s(fs, sf, plan)          == 0
            Test.@test _a_type1s(sfo, fs, plan)         == 0
            Test.@test _a_solves(sfo, fs, plan, sws)    == 0
            Test.@test _a_quad(sfo, fs, plan, w)        == 0
            Test.@test _a_fit(sfo, fs, plan, sws)       == 0
            NUFSHT.close!(plan)
        end

        # Single precision, here with the direct sum on one thread.
        Test.@testset "spin transform (ComplexF32, B=$B)" begin
            s = 1
            plan = NUFSHT.make_spin_plan(ComplexF32, Float32.(θ), Float32.(φ), lmax, s; tol = 1e-5,
                                         ntrans = B, execution = serial,
                                         nufft = NUFSHT.SpectralBackends.DirectSumSpectralBackend())
            Test.@test plan.engine isa NUFSHT._RingEngine
            sf = zeros(ComplexF32, Nθ, Nφ, B); fs = zeros(ComplexF32, M, B); sfo = zeros(ComplexF32, Nθ, Nφ, B)
            for b in 1:B, ℓ in abs(s):min(4, lmax), m in -ℓ:ℓ
                sf[NUFSHT.spin_coeff_index(ℓ, m, lmax), b] = randn(ComplexF32)
            end
            NUFSHT.nusht_type2_spin!(fs, sf, plan)
            NUFSHT.nusht_type1_spin!(sfo, fs, plan)
            Test.@test _a_type2s(fs, sf, plan)  == 0
            Test.@test _a_type1s(sfo, fs, plan) == 0
            NUFSHT.close!(plan)
        end
    end
end
