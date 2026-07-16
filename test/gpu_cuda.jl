# Standalone GPU parity test — RUN ON A CUDA MACHINE. Not part of `runtests.jl` (no GPU on CI here).
#
#   julia --project=test test/gpu_cuda.jl
#
# It exercises the FULL device spin transform: the Trapani–Navaza recurrence + bivariate-Fourier
# contraction run as KernelAbstractions kernels on the GPU, and the NUFFT via cuFINUFFT. The device
# result must match the CPU transform to the NUFFT tolerance. On this repo's dev box (macOS, no CUDA)
# the recurrence/assembly kernels are already validated bit-for-bit on JLArrays (see test_ka.jl); this
# script is the remaining piece — the cuFINUFFT runtime — which can only run on NVIDIA hardware.
using Test, Random, LinearAlgebra
using CUDA
using KernelAbstractions, GPUArraysCore        # loads NUFSHTKernelAbstractionsExt (device kernels)
using NUFSHT

if !CUDA.functional()
    @warn "CUDA not functional — skipping GPU parity test."
else
    @testset "CUDA device spin transform == CPU" begin
        Random.seed!(7)
        for (lmax, s, B) in ((16, 0, 1), (16, 2, 1), (24, 1, 2))
            N = lmax + 1; Nφ = 2lmax + 1; M = 4 * N^2
            θ = clamp.(π .* rand(M), 1e-9, π - 1e-9); φ = 2π .* rand(M)
            sf = zeros(ComplexF64, N, Nφ, B)
            for b in 1:B, ℓ in abs(s):lmax, m in -ℓ:ℓ
                sf[NUFSHT.spin_coeff_index(ℓ, m, lmax), b] = randn(ComplexF64)
            end

            # CPU reference
            pc = NUFSHT.make_spin_plan(θ, φ, lmax, s; tol = 1e-9, ntrans = B)
            fc = zeros(ComplexF64, M, B); NUFSHT.nusht_type2_spin!(fc, sf, pc)

            # GPU: device nodes ⇒ device buffers (similar) + cuFINUFFT (creation seam).
            pg = NUFSHT.make_spin_plan(CuArray(θ), CuArray(φ), lmax, s; tol = 1e-9, ntrans = B)
            fg = CUDA.zeros(ComplexF64, M, B); NUFSHT.nusht_type2_spin!(fg, CuArray(sf), pg)
            @test Array(fg) ≈ fc rtol = 1e-6

            # adjoint on device
            g = randn(ComplexF64, M, B)
            sfc = zeros(ComplexF64, N, Nφ, B); NUFSHT.nusht_type1_spin!(sfc, g, pc)
            sfg = CUDA.zeros(ComplexF64, N, Nφ, B); NUFSHT.nusht_type1_spin!(sfg, CuArray(g), pg)
            @test Array(sfg) ≈ sfc rtol = 1e-6

            # exact inversion entirely on device (CG reductions are device-generic; assembly is KA
            # kernels; NUFFT is cuFINUFFT). Recovers the field to CG tolerance.
            solg = CUDA.zeros(ComplexF64, N, Nφ, B)
            NUFSHT.nusht_solve_spin!(solg, fg, pg; rtol = 1e-7, maxiter = 500)
            frecg = CUDA.zeros(ComplexF64, M, B); NUFSHT.nusht_type2_spin!(frecg, solg, pg)
            @test sqrt(sum(abs2, Array(frecg) .- Array(fg)) / sum(abs2, Array(fg))) < 1e-3

            @info "GPU parity lmax=$lmax s=$s B=$B: ‖f_gpu-f_cpu‖/‖f‖=$(norm(Array(fg).-fc)/norm(fc))"
            NUFSHT.close!(pc); NUFSHT.close!(pg)
        end
    end
end
