# Dispatch gates: the scalar (FastTransforms and ring) and spin transforms, fits, analyses and filters on
# both NUFFT libraries make no runtime dispatch in this package's code, nor does a FINUFFT plan build. A
# NonuniformFFTs build types its width pool from `ntrans` (`_nufft_size_pool`), one dispatch at that
# barrier. `JET` refuses pre-release Julia.
using JET: JET

Test.@testset "Entry points make no runtime dispatch" begin
    if isempty(VERSION.prerelease)
        M, lmax, B = 400, 10, 2
        θ, φ = iid_points(M, 90)
        w = fill(4π / M, M)
        filt = NUFSHT.gaussian_from_scale(2000e3)
        ser = NUFSHT.ComputationalBackends.SerialBackend()
        JET.@test_opt target_modules = (NUFSHT,) NUFSHT.make_plan(Float64, θ, φ, lmax; execution = ser,
                                                                   nufft = FTB.FINUFFTBackend(), ntrans = B)
        for nufft in (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend()), FE in (Float64, ComplexF64)
            for p in (NUFSHT.make_plan(FE, θ, φ, lmax; nufft, ntrans = B),
                      NUFSHT._build_ring_plan(FE, θ, φ, lmax; nufft, ntrans = B))
                C = NUFSHT.allocate_coefficients(p); f = zeros(FE, M, B); ws = FTB.LSMRWorkspace(p)
                JET.@test_opt target_modules = (NUFSHT,) NUFSHT.nusht_type2!(f, C, p)
                JET.@test_opt target_modules = (NUFSHT,) NUFSHT.nusht_type1!(C, f, p)
                JET.@test_opt target_modules = (NUFSHT,) NUFSHT.nusht_solve!(C, f, p; ws, maxiter = 3)
                JET.@test_opt target_modules = (NUFSHT,) NUFSHT.nusht_analyze!(C, f, p; weights = w)
                JET.@test_opt target_modules = (NUFSHT,) NUFSHT.nusht_filter!(f, f, filt, p; ws, maxiter = 3)
                NUFSHT.close!(p)
            end
            s = FE <: Real ? 0 : 1
            sp = NUFSHT.make_spin_plan(FE, θ, φ, lmax, s; nufft, ntrans = B)
            sf = NUFSHT.allocate_coefficients(sp); g = zeros(FE, M, B); sws = FTB.LSMRWorkspace(sp)
            JET.@test_opt target_modules = (NUFSHT,) NUFSHT.nusht_type2_spin!(g, sf, sp)
            JET.@test_opt target_modules = (NUFSHT,) NUFSHT.nusht_type1_spin!(sf, g, sp)
            JET.@test_opt target_modules = (NUFSHT,) NUFSHT.nusht_solve_spin!(sf, g, sp; ws = sws, maxiter = 3)
            NUFSHT.close!(sp)
        end
    end
end
