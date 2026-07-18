using KernelAbstractions: KernelAbstractions   # loads NUFSHTKernelAbstractionsExt (with GPUArraysCore)
using GPUArraysCore: GPUArraysCore
using JLArrays: JLArrays                        # reference GPU-array backend for GPU-less testing

# Device-genericity of the KA-kernel steps, exercised on `JLArray` (a CPU-backed `AbstractGPUArray`
# with a KA backend) — it takes the exact `::AbstractGPUArray` dispatch + kernel-launch path a real
# CuArray/ROCArray does, so a real GPU should reproduce these results. Kernels must match the plain
# CPU-loop `src` methods bit-for-bit and preserve the doubling↔folding adjoint on device.
Test.@testset "KernelAbstractions extension: device-generic DFS kernels (JLArray)" begin
    Test.@test JLArrays.JLArray <: GPUArraysCore.AbstractGPUArray

    Random.seed!(101)
    for (Nθ, Nφ, B) in ((11, 21, 1), (11, 21, 4), (7, 13, 3))
        F = randn(Nθ, Nφ, B)

        # doubling: JLArray path == CPU-loop path
        F̃_cpu = zeros(ComplexF64, 2Nθ, Nφ, B); NUFSHT.dfs_double!(F̃_cpu, F)
        Fj = JLArrays.JLArray(F)
        F̃j = JLArrays.JLArray(zeros(ComplexF64, 2Nθ, Nφ, B))
        NUFSHT.dfs_double!(F̃j, Fj)
        Test.@test Array(F̃j) == F̃_cpu

        # folding: JLArray path == CPU-loop path
        G_cpu = zeros(Nθ, Nφ, B); NUFSHT.dfs_fold!(G_cpu, F̃_cpu)
        Gj = JLArrays.JLArray(zeros(Nθ, Nφ, B)); NUFSHT.dfs_fold!(Gj, F̃j)
        Test.@test Array(Gj) == G_cpu

        # device adjoint: ⟨double(x), y⟩ == ⟨x, fold(y)⟩ (real embedding), exact
        x = JLArrays.JLArray(randn(Nθ, Nφ, B))
        y = JLArrays.JLArray(randn(ComplexF64, 2Nθ, Nφ, B))
        dx = JLArrays.JLArray(zeros(ComplexF64, 2Nθ, Nφ, B)); NUFSHT.dfs_double!(dx, x)
        fy = JLArrays.JLArray(zeros(Nθ, Nφ, B)); NUFSHT.dfs_fold!(fy, y)
        lhs = sum(real.(Array(dx)) .* real.(Array(y)))
        rhs = sum(Array(x) .* Array(fy))
        Test.@test isapprox(lhs, rhs; atol = 1e-12, rtol = 1e-12)
    end
    @info "KernelAbstractions ext: DFS double/fold + adjoint validated on JLArray (device-generic path)"
end

# The spin S-engine (Trapani–Navaza recurrence + bivariate-Fourier contraction) as KA kernels — the
# device path for the whole spin transform (only the NUFFT is vendor-specific, via cuFINUFFT). Must
# reproduce the CPU `_assemble_G!`/`_assemble_G_adjoint!` bit-for-bit on JLArray.
Test.@testset "KernelAbstractions extension: device spin assembly (recurrence) on JLArray" begin
    Random.seed!(202)
    for (lmax, s, B) in ((8, 0, 1), (8, 2, 1), (12, 1, 3), (6, -1, 2))
        N = lmax + 1; Nφ = 2lmax + 1; M = 50
        θ = clamp.(π .* rand(M), 1e-9, π - 1e-9); φ = 2π .* rand(M)
        pc = NUFSHT.make_spin_plan(θ, φ, lmax, s; tol = 1e-10, ntrans = B)   # CPU (Array buffers)
        sf = zeros(ComplexF64, N, Nφ, B)
        for b in 1:B, ℓ in abs(s):lmax, m in -ℓ:ℓ
            sf[NUFSHT.spin_coeff_index(ℓ, m, lmax), b] = randn(ComplexF64)
        end
        # Device plan: same scalar fields, JLArray recurrence/mode buffers (NUFFT plans are unused here).
        pj = NUFSHT.SpinNUSHTplan(lmax, s, B, pc.tol, JLArrays.JLArray(pc.θ_nodes), JLArrays.JLArray(pc.φ_nodes),
                                  JLArrays.JLArray(pc.dl_curr), JLArrays.JLArray(pc.dl_prev),
                                  JLArrays.JLArray(zeros(ComplexF64, size(pc.G))), JLArrays.JLArray(pc.fbuf),
                                  pc.nufft_type2, pc.nufft_type1)

        Gc = copy(pc.G); NUFSHT._assemble_G!(Gc, sf, pc)                     # CPU forward
        NUFSHT._assemble_G!(pj.G, JLArrays.JLArray(sf), pj)                  # device forward
        Test.@test Array(pj.G) ≈ Gc

        Ĝ = randn(ComplexF64, Nφ, Nφ, B)
        sfc = zeros(ComplexF64, N, Nφ, B); NUFSHT._assemble_G_adjoint!(sfc, Ĝ, pc)          # CPU adjoint
        sfj = JLArrays.JLArray(zeros(ComplexF64, N, Nφ, B))
        NUFSHT._assemble_G_adjoint!(sfj, JLArrays.JLArray(Ĝ), pj)                            # device adjoint
        Test.@test Array(sfj) ≈ sfc
        NUFSHT.close!(pc)
    end
    @info "KernelAbstractions ext: device spin recurrence+assembly matches CPU on JLArray (s=0,±1,2; batched)"
end

# Device-generic CG workspace reductions (so nusht_solve_spin! runs on GPU) — must match the CPU
# scalar-loop versions bit-for-bit on JLArray.
Test.@testset "KernelAbstractions extension: device spin CG reductions (JLArray)" begin
    Random.seed!(303)
    N = 8; Nφ = 15; B = 4
    a = randn(ComplexF64, N, Nφ, B); b = randn(ComplexF64, N, Nφ, B); x = randn(ComplexF64, N, Nφ, B)
    α = randn(B); β = randn(B); σ = -1.0

    dc = zeros(Float64, B); NUFSHT._col_hdot!(dc, a, b)
    dj = JLArrays.JLArray(zeros(Float64, B)); NUFSHT._col_hdot!(dj, JLArrays.JLArray(a), JLArrays.JLArray(b))
    Test.@test Array(dj) ≈ dc

    yc = copy(a); NUFSHT._col_axpy_c!(yc, α, x, σ)
    yj = JLArrays.JLArray(copy(a)); NUFSHT._col_axpy_c!(yj, JLArrays.JLArray(α), JLArrays.JLArray(x), σ)
    Test.@test Array(yj) ≈ yc

    pc = copy(a); NUFSHT._col_pbp_c!(pc, b, β)
    pj = JLArrays.JLArray(copy(a)); NUFSHT._col_pbp_c!(pj, JLArrays.JLArray(b), JLArrays.JLArray(β))
    Test.@test Array(pj) ≈ pc
end

# Structural device-type propagation: building a plan from device node arrays must yield a plan (and CG
# workspace) whose buffers are ALL device-resident. This is the regression guard for issue #6 — the
# scalar `make_plan` previously built host buffers for device nodes (silently, since JLArray coords copy
# to host for FINUFFT), shipping a plan that could not run on GPU. (The FFT plan on a JLArray falls back
# to host FFTW — JLArrays are CPU-backed strided memory — so the device *FFT* itself is only validated on
# real CUDA, in test/gpu_cuda.jl; here we assert the buffer array types, which is what regressed.)
Test.@testset "KernelAbstractions extension: device plan buffers are device-resident (JLArray)" begin
    isdev(x) = x isa GPUArraysCore.AbstractGPUArray
    Random.seed!(404)
    lmax = 6; M = 60; B = 2
    θ = JLArrays.JLArray(clamp.(π .* rand(M), 1e-9, π - 1e-9))
    φ = JLArrays.JLArray(2π .* rand(M))

    plan = NUFSHT.make_plan(θ, φ, lmax; tol = 1e-8, ntrans = B)
    for f in (:θ_nodes, :φ_nodes, :C, :F, :F̃, :Fhat, :fbuf)
        Test.@test isdev(getfield(plan, f))
    end
    ws = NUFSHT.CGWorkspace(plan)
    for f in (:x, :r, :p, :Ap, :rhs, :f, :rsold, :α)
        Test.@test isdev(getfield(ws, f))
    end
    NUFSHT.close!(plan)

    splan = NUFSHT.make_spin_plan(θ, φ, lmax, 2; tol = 1e-8, ntrans = B)
    for f in (:θ_nodes, :φ_nodes, :dl_curr, :dl_prev, :G, :fbuf)
        Test.@test isdev(getfield(splan, f))
    end
    sws = NUFSHT.SpinCGWorkspace(splan)
    for f in (:x, :r, :p, :Ap, :rhs, :f)
        Test.@test isdev(getfield(sws, f))
    end
    NUFSHT.close!(splan)
    @info "KernelAbstractions ext: scalar+spin device plans/workspaces are device-resident (issue #6)"
end

# Device-generic scalar CG reductions + real↔complex field copy (so the *scalar* nusht_solve!/type-2/1
# run on GPU) — must match the CPU scalar-loop `src` methods bit-for-bit on JLArray.
Test.@testset "KernelAbstractions extension: device scalar reductions + field copy (JLArray)" begin
    Random.seed!(505)
    N = 7; Nφ = 13; B = 4
    a = randn(N, Nφ, B); b = randn(N, Nφ, B); x = randn(N, Nφ, B)
    α = randn(B); β = randn(B); σ = -1.0

    dc = zeros(B); NUFSHT._col_dot!(dc, a, b)
    dj = JLArrays.JLArray(zeros(B)); NUFSHT._col_dot!(dj, JLArrays.JLArray(a), JLArrays.JLArray(b))
    Test.@test Array(dj) ≈ dc

    yc = copy(a); NUFSHT._col_axpy!(yc, α, x, σ)
    yj = JLArrays.JLArray(copy(a)); NUFSHT._col_axpy!(yj, JLArrays.JLArray(α), JLArrays.JLArray(x), σ)
    Test.@test Array(yj) ≈ yc

    pc = copy(a); NUFSHT._col_pbp!(pc, b, β)
    pj = JLArrays.JLArray(copy(a)); NUFSHT._col_pbp!(pj, JLArrays.JLArray(b), JLArrays.JLArray(β))
    Test.@test Array(pj) ≈ pc

    # real↔complex field copy, both `f` shapes: (M, B) and (M,) with B=1.
    M = 20
    for (fsz, bufsz) in (((M, B), (M, B)), ((M,), (M, 1)))
        fbuf = randn(ComplexF64, bufsz...)
        fc = zeros(fsz...); NUFSHT._copy_real!(fc, fbuf)
        fj = JLArrays.JLArray(zeros(fsz...)); NUFSHT._copy_real!(fj, JLArrays.JLArray(fbuf))
        Test.@test Array(fj) == fc

        fsrc = randn(fsz...)
        bc = zeros(ComplexF64, bufsz...); NUFSHT._copy_field!(bc, fsrc)
        bj = JLArrays.JLArray(zeros(ComplexF64, bufsz...)); NUFSHT._copy_field!(bj, JLArrays.JLArray(fsrc))
        Test.@test Array(bj) == bc
    end
end

# Device-generic spectral filter (`apply_transfer!`, × H(ℓ)) — must match the CPU scalar mode loop
# bit-for-bit on JLArray, for both a smooth (Gaussian) and a sharp (top-hat) transfer.
Test.@testset "KernelAbstractions extension: device apply_transfer! (JLArray)" begin
    Random.seed!(606)
    lmax = 8; N = lmax + 1; Nφ = 2lmax + 1; B = 3
    for filt in (NUFSHT.gaussian_from_scale(2000e3), NUFSHT.TopHatTransfer(4))
        C = randn(N, Nφ, B)
        Cc = copy(C); NUFSHT.apply_transfer!(Cc, filt, lmax)
        Cj = JLArrays.JLArray(copy(C)); NUFSHT.apply_transfer!(Cj, filt, lmax)
        Test.@test Array(Cj) == Cc
    end
end
