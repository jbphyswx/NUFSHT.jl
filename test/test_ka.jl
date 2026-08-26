using KernelAbstractions: KernelAbstractions   # loads NUFSHTKernelAbstractionsExt (with GPUArraysCore)
using GPUArraysCore: GPUArraysCore
using JLArrays: JLArrays                        # reference GPU-array backend for GPU-less testing

# Device-genericity of the KA-kernel steps, exercised on `JLArray` (a CPU-backed `AbstractGPUArray`
# with a KA backend) — it takes the exact `::AbstractGPUArray` dispatch + kernel-launch path a real
# CuArray/ROCArray does, so a real GPU should reproduce these results. Kernels must match the plain
# CPU-loop `src` methods bit-for-bit.
Test.@testset "KernelAbstractions extension: JLArray is a device array" begin
    Test.@test JLArrays.JLArray <: GPUArraysCore.AbstractGPUArray
end

# The scalar F step (`P·C` → the complex exponential mode array) as KA kernels. Both directions are
# per-element gathers sharing `_mode_entry`/`_mode_adjoint_entry` with the host loops, so device and
# host must agree bit-for-bit — this is the whole scalar mode assembly, exercised in both layouts (the
# full `2lmax+3`-row array and the folded `lmax+2`-row real one) and at both coefficient element types.
Test.@testset "KernelAbstractions extension: device scalar mode assembly on JLArray" begin
    Random.seed!(101)
    for (lmax, B) in ((8, 1), (11, 3)), FE in (Float64, ComplexF64)
        N = lmax + 1; Nφ = 2lmax + 1
        G = FE <: Real ? randn(N, Nφ, B) : randn(ComplexF64, N, Nφ, B)
        for rows in (2lmax + 3, lmax + 2)          # full and folded
            Zc = zeros(ComplexF64, rows, Nφ, B)
            NUFSHT._assemble_modes!(Zc, G, lmax)
            Zj = JLArrays.JLArray(zeros(ComplexF64, rows, Nφ, B))
            NUFSHT._assemble_modes!(Zj, JLArrays.JLArray(G), lmax)
            Test.@test Array(Zj) == Zc
            # The device method must be the one dispatch actually picks, or this compares host to host.
            Test.@test which(NUFSHT._assemble_modes!,
                             (typeof(Zj), typeof(JLArrays.JLArray(G)), Int)).module ===
                       Base.get_extension(NUFSHT, :NUFSHTKernelAbstractionsExt)

            Z = randn(ComplexF64, rows, Nφ, B)
            Gc = zeros(FE, N, Nφ, B); NUFSHT._assemble_modes_adjoint!(Gc, Z, lmax)
            Gj = JLArrays.JLArray(zeros(FE, N, Nφ, B))
            NUFSHT._assemble_modes_adjoint!(Gj, JLArrays.JLArray(Z), lmax)
            Test.@test Array(Gj) == Gc
        end
    end
    @info "KernelAbstractions ext: device scalar mode assembly matches CPU on JLArray (full + folded, real + complex)"
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
        ndj = NUFSHT.FixedCountNodes(JLArrays.JLArray(pc.nodes.θ_nodes), JLArrays.JLArray(pc.nodes.φ_nodes),
                                     JLArrays.JLArray(pc.nodes.θ_nufft), nothing, JLArrays.JLArray(pc.nodes.fbuf),
                                     pc.nodes.nufft_type2, pc.nodes.nufft_type1)
        pj = NUFSHT.SpinNUSHTplan(lmax, s, B, pc.tol, ndj,
                                  JLArrays.JLArray(pc.dl_curr), JLArrays.JLArray(pc.dl_prev),
                                  JLArrays.JLArray(zeros(ComplexF64, size(pc.G))), pc.wigner)

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

# Device-generic solver column primitives on complex data (so `nusht_solve_spin!` runs on GPU) — must
# match the CPU scalar-loop versions on JLArray. One set serves the real and complex paths.
Test.@testset "KernelAbstractions extension: device column primitives, complex (JLArray)" begin
    Random.seed!(303)
    N = 8; Nφ = 15; B = 4
    a = randn(ComplexF64, N, Nφ, B); b = randn(ComplexF64, N, Nφ, B); x = randn(ComplexF64, N, Nφ, B)
    α = randn(B); β = randn(B); s = randn(B); σ = -1.0

    dc = zeros(Float64, B); NUFSHT._col_hdot!(dc, a, b)
    dj = JLArrays.JLArray(zeros(Float64, B)); NUFSHT._col_hdot!(dj, JLArrays.JLArray(a), JLArrays.JLArray(b))
    Test.@test Array(dj) ≈ dc

    yc = copy(a); NUFSHT._col_axpy!(yc, α, x, σ)
    yj = JLArrays.JLArray(copy(a)); NUFSHT._col_axpy!(yj, JLArrays.JLArray(α), JLArrays.JLArray(x), σ)
    Test.@test Array(yj) ≈ yc

    pc = copy(a); NUFSHT._col_pbp!(pc, b, β)
    pj = JLArrays.JLArray(copy(a)); NUFSHT._col_pbp!(pj, JLArrays.JLArray(b), JLArrays.JLArray(β))
    Test.@test Array(pj) ≈ pc

    qc = copy(a); NUFSHT._col_scale!(qc, s)
    qj = JLArrays.JLArray(copy(a)); NUFSHT._col_scale!(qj, JLArrays.JLArray(s))
    Test.@test Array(qj) ≈ qc
end

# A plan built from device node arrays must have ALL buffers device-resident. Only the spin path can
# be checked here: building a scalar plan needs a NUFFT for the array type, and the only device NUFFT
# is cuFINUFFT, which is `CuArray`-only. The scalar path's own device kernels are covered above
# without a plan; end to end it is exercised on real hardware in test/gpu_cuda.jl.
Test.@testset "KernelAbstractions extension: device plan buffers are device-resident (JLArray)" begin
    isdev(x) = x isa GPUArraysCore.AbstractGPUArray
    Random.seed!(404)
    lmax = 6; M = 60; B = 2
    θ = JLArrays.JLArray(clamp.(π .* rand(M), 1e-9, π - 1e-9))
    φ = JLArrays.JLArray(2π .* rand(M))

    splan = NUFSHT.make_spin_plan(θ, φ, lmax, 2; tol = 1e-8, ntrans = B)
    for f in (:dl_curr, :dl_prev, :G)
        Test.@test isdev(getfield(splan, f))
    end
    for f in (:θ_nodes, :φ_nodes, :θ_nufft, :fbuf)
        Test.@test isdev(getfield(splan.nodes, f))
    end
    sws = NUFSHT.LSMRWorkspace(splan)
    for f in (:x, :v, :h, :hbar, :w, :u, :nrm, :cf)
        Test.@test isdev(getfield(sws, f))
    end
    # The bidiagonalization's scalar recurrences run on the host, so those stay host vectors.
    for f in (:α, :β, :ζbar, :rel, :colres)
        Test.@test getfield(sws, f) isa Array
    end
    NUFSHT.close!(splan)
    @info "KernelAbstractions ext: spin device plan/workspace are device-resident"
end

# Device-generic solver column primitives on real data + the real↔complex field copy (so the *scalar*
# nusht_solve!/type-2/1 run on GPU) — must match the CPU scalar-loop `src` methods on JLArray.
Test.@testset "KernelAbstractions extension: device column primitives, real (JLArray)" begin
    Random.seed!(505)
    N = 7; Nφ = 13; B = 4
    a = randn(N, Nφ, B); b = randn(N, Nφ, B); x = randn(N, Nφ, B)
    α = randn(B); β = randn(B); σ = -1.0

    dc = zeros(B); NUFSHT._col_hdot!(dc, a, b)
    dj = JLArrays.JLArray(zeros(B)); NUFSHT._col_hdot!(dj, JLArrays.JLArray(a), JLArrays.JLArray(b))
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

# `nusht_solve!` narrows to the live column prefix as columns retire, so every call it makes to the
# column kernels carries a count. A device override without that argument is simply not the method
# those calls select: the loop lands on the `src` scalar loop instead and indexes the device array
# element by element. Assert the arity the solver actually uses, not the bare one.
Test.@testset "KernelAbstractions extension: device column kernels take the live-column count" begin
    JA3 = JLArrays.JLArray{Float64,3}; JA1 = JLArrays.JLArray{Float64,1}
    ext = Base.get_extension(NUFSHT, :NUFSHTKernelAbstractionsExt)
    Test.@test ext !== nothing
    for (fn, sig) in ((NUFSHT._col_hdot!, (JA1, JA3, JA3, Int)),
                      (NUFSHT._col_axpy!, (JA3, JA1, JA3, Float64, Int)),
                      (NUFSHT._col_pbp!, (JA3, JA3, JA1, Int)),
                      (NUFSHT._col_scale!, (JA3, JA1, Int)))
        Test.@test which(fn, sig).module === ext
    end
end

# Everything the solver does to its column buffers, on device arrays with scalar indexing turned into
# an error: the reductions at the live width, the column copies and the retire/compact pass. The
# transforms themselves cannot run here (nothing implements AbstractFFTs for `JLArray`), so this
# covers the bookkeeping — the part that reaches into individual columns and rows.
Test.@testset "KernelAbstractions extension: device solver bookkeeping never scalar-indexes (JLArray)" begin
    Random.seed!(707)
    N, Nf, B, n = 7, 13, 4, 3
    len = N * Nf
    mlen = 5
    J(v) = JLArrays.JLArray(v)
    no_scalar(f) = task_local_storage(f, :ScalarIndexing, GPUArraysCore.ScalarDisallowed)
    # The guard must actually bite, or everything below it passes vacuously.
    Test.@test_throws ErrorException no_scalar(() -> J(zeros(3))[1])

    a = randn(N, Nf, B); b = randn(N, Nf, B); x = randn(N, Nf, B); s = randn(B)
    α = randn(B); β = randn(B); σ = -1.0
    dc = zeros(B); NUFSHT._col_hdot!(dc, a, b, n)
    yc = copy(a); NUFSHT._col_axpy!(yc, α, x, σ, n)
    pc = copy(a); NUFSHT._col_pbp!(pc, b, β, n)
    qc = copy(a); NUFSHT._col_scale!(qc, s, n)

    # The scalar solver's vectors are packed to the `l ≤ lmax` slots, so its iterate is `(K, B)` and
    # retirement scatters a column back into the plan's `(Nθ, Nφ, B)` layout. Build a real plan for the
    # index vector and the shapes — its transforms are never run here.
    θp, φp = fib_points(64)
    splan = NUFSHT.make_plan(Float64, θp, φp, N - 1; tol = 1e-8)
    idx = NUFSHT._valid_indices(zeros(0), N - 1)
    K = length(idx)
    xp = randn(K, B)

    dj = J(zeros(B)); yj = J(copy(a)); pj = J(copy(a)); qj = J(copy(a)); Cj = J(zeros(N, Nf, B))
    ws = NUFSHT.LSMRWorkspace(J(copy(xp)), J(zeros(K, B)), J(zeros(K, B)), J(zeros(K, B)),
                              J(copy(a)), J(zeros(mlen, B)), J(idx), collect(1:B),
                              J(zeros(B)), J(zeros(B)),
                              ntuple(_ -> zeros(B), 17)...)
    nlive = no_scalar() do
        NUFSHT._col_hdot!(dj, J(a), J(b), n)
        NUFSHT._col_axpy!(yj, J(α), J(x), σ, n)
        NUFSHT._col_pbp!(pj, J(b), J(β), n)
        NUFSHT._col_scale!(qj, J(s), n)
        ws.rel .= (1e-9, 0.5, 0.4, 0.3)
        ws.done .= (1.0, 1.0, 0.0, 0.0)            # slots 1 and 2 finished, 3 and 4 live
        NUFSHT._retire_and_compact!(Cj, ws, splan, 1e-6, B, K, mlen)
    end
    NUFSHT.close!(splan)

    Test.@test Array(dj)[1:n] ≈ dc[1:n]
    Test.@test all(Array(dj)[(n + 1):end] .== 0)   # the count really did bound the reduction
    Test.@test Array(yj) ≈ yc
    Test.@test Array(pj) ≈ pc
    Test.@test Array(qj) ≈ qc
    # Retirement scatters slot `s` to column `perm[s]` — the identity before compaction — putting the
    # packed values at the valid slots and zero everywhere else.
    for c in 1:2
        Test.@test Array(Cj)[:, :, c][idx] == xp[:, c]
        Test.@test sum(abs, Array(Cj)[:, :, c]) == sum(abs, xp[:, c])   # nothing outside them
    end
    Test.@test nlive == 2
    Test.@test sort(ws.perm) == collect(1:B)
    Test.@test ws.colres[1] == 1e-9 && ws.colres[2] == 0.5
    Test.@test ws.rel[1:nlive] == [0.4, 0.3]       # survivors' state moved with them
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
