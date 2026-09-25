# GPU parity on an NVIDIA machine; not part of `runtests.jl`:
#
#   julia --project=test test/gpu_cuda.jl
#
# Device plans (KernelAbstractions kernels for the mode assembly, the spin recurrence and the solver's
# column primitives; FlowTransformBindings' NUFFT on the nodes' device) are scored against host plans
# on the same points, for both NUFFT libraries, and each device solve against the coefficients its
# field was synthesised from.
using Test: Test
using Random: Random
using CUDA: CUDA
using KernelAbstractions: KernelAbstractions
using GPUArraysCore: GPUArraysCore
using NUFSHT: NUFSHT          # before FastSphericalHarmonics, as runtests.jl loads it
using FlowTransformBindings: FlowTransformBindings as FTB
using FINUFFT: FINUFFT
using NonuniformFFTs: NonuniformFFTs
using FastSphericalHarmonics: FastSphericalHarmonics

const LIBRARIES = (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend())
relerr(a, b) = sqrt(sum(abs2, a .- b) / sum(abs2, b))

CUDA.functional() || error("CUDA is not functional on this machine")

Test.@testset "CUDA spin transforms equal host: $(nameof(typeof(nufft)))" for nufft in LIBRARIES
    Random.seed!(7)
    for (lmax, s, B) in ((16, 0, 1), (16, 2, 1), (24, 1, 2))
        N = lmax + 1; Nφ = 2lmax + 1; M = 4 * N^2
        θ = acos.(2 .* rand(M) .- 1); φ = 2π .* rand(M)
        sf = zeros(ComplexF64, N, Nφ, B)
        for b in 1:B, ℓ in abs(s):lmax, m in -ℓ:ℓ
            sf[NUFSHT.spin_coeff_index(ℓ, m, lmax), b] = randn(ComplexF64)
        end

        pc = NUFSHT.make_spin_plan(θ, φ, lmax, s; tol = 1e-10, ntrans = B, nufft)
        pg = NUFSHT.make_spin_plan(CUDA.CuArray(θ), CUDA.CuArray(φ), lmax, s; tol = 1e-10, ntrans = B, nufft)
        Test.@test pg.G isa CUDA.CuArray && pg.nodes.fbuf isa CUDA.CuArray

        fc = zeros(ComplexF64, M, B); NUFSHT.nusht_type2_spin!(fc, sf, pc)
        fg = CUDA.zeros(ComplexF64, M, B); NUFSHT.nusht_type2_spin!(fg, CUDA.CuArray(sf), pg)
        Test.@test relerr(Array(fg), fc) < 1e-8

        g = randn(ComplexF64, M, B)
        sfc = zeros(ComplexF64, N, Nφ, B); NUFSHT.nusht_type1_spin!(sfc, g, pc)
        sfg = CUDA.zeros(ComplexF64, N, Nφ, B); NUFSHT.nusht_type1_spin!(sfg, CUDA.CuArray(g), pg)
        Test.@test relerr(Array(sfg), sfc) < 1e-8

        solg = CUDA.zeros(ComplexF64, N, Nφ, B)
        res = NUFSHT.nusht_solve_spin!(solg, fg, pg; rtol = 1e-10, maxiter = 500)
        Test.@test res[4]
        Test.@test relerr(Array(solg), sf) < 1e-6
        NUFSHT.close!(pc); NUFSHT.close!(pg)
    end
end

Test.@testset "CUDA scalar transforms equal host: $(nameof(typeof(nufft)))" for nufft in LIBRARIES
    Random.seed!(11)
    for (lmax, B) in ((16, 1), (24, 2))
        N = lmax + 1; Nφ = 2lmax + 1; M = 4 * N^2
        θ = acos.(2 .* rand(M) .- 1); φ = 2π .* rand(M)
        C = zeros(N, Nφ, B)
        for b in 1:B, ℓ in 0:min(6, lmax), m in -ℓ:ℓ
            C[FastSphericalHarmonics.sph_mode(ℓ, m), b] = randn()
        end

        pc = NUFSHT.make_plan(θ, φ, lmax; tol = 1e-10, ntrans = B, nufft)
        pg = NUFSHT.make_plan(CUDA.CuArray(θ), CUDA.CuArray(φ), lmax; tol = 1e-10, ntrans = B, nufft)
        Test.@test pg.F isa CUDA.CuArray && pg.nodes.fbuf isa CUDA.CuArray

        fc = zeros(M, B); NUFSHT.nusht_type2!(fc, C, pc)
        fg = CUDA.zeros(M, B); NUFSHT.nusht_type2!(fg, CUDA.CuArray(C), pg)
        Test.@test relerr(Array(fg), fc) < 1e-8

        g = randn(M, B)
        Cc = zeros(N, Nφ, B); NUFSHT.nusht_type1!(Cc, g, pc)
        Cg = CUDA.zeros(N, Nφ, B); NUFSHT.nusht_type1!(Cg, CUDA.CuArray(g), pg)
        Test.@test relerr(Array(Cg), Cc) < 1e-8

        filt = NUFSHT.gaussian_from_scale(2000e3)
        foc = zeros(M, B); NUFSHT.nusht_filter!(foc, fc, filt, pc)
        fog = CUDA.zeros(M, B); NUFSHT.nusht_filter!(fog, fg, filt, pg)
        Test.@test relerr(Array(fog), foc) < 1e-6

        solg = NUFSHT.allocate_coefficients(pg)
        res = NUFSHT.nusht_solve!(solg, fg, pg; rtol = 1e-10, maxiter = 500)
        Test.@test res[4]
        Test.@test relerr(Array(solg), C) < 1e-6
        NUFSHT.close!(pc); NUFSHT.close!(pg)
    end
end
