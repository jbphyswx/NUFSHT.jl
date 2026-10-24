using OhMyThreads: OhMyThreads   # triggers NUFSHTOhMyThreadsExt
using ComputationalBackends: ComputationalBackends

# Node-local thread parallelism over independent problems. The farm runs FastTransforms on one OpenMP
# thread per task and the plans are built with `nthreads = 1`, so a threaded transform equals the serial
# one, and the main task's transforms are unchanged afterwards.
Test.@testset "OhMyThreads extension: threaded == serial, main task intact" begin
    Random.seed!(11)
    lmax = 10
    N = lmax + 1
    Nφ = 2lmax + 1
    N_modes = N^2
    M = 4 * N_modes                       # 4× overdetermined
    P = 4                                 # independent problems (distinct point sets → distinct plans)

    # Jittered equidistributed points.
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

    # Serial references on the main task, computed before the threaded section.
    fref = [zeros(M) for _ in 1:P]
    for i in 1:P
        NUFSHT.nusht_type2!(fref[i], Cs[i], plans[i])
    end
    health_before = zeros(M); NUFSHT.nusht_type2!(health_before, Cs[1], plans[1])

    fs = [zeros(M) for _ in 1:P]
    NUFSHT.nusht_type2!(fs, Cs, plans; execution = ComputationalBackends.ThreadedBackend())
    for i in 1:P
        Test.@test fs[i] ≈ fref[i]
    end

    Cs_out = [zeros(N, Nφ) for _ in 1:P]
    NUFSHT.nusht_type1!(Cs_out, fs, plans; execution = ComputationalBackends.ThreadedBackend())
    for i in 1:P
        Cref = zeros(N, Nφ); NUFSHT.nusht_type1!(Cref, fs[i], plans[i])
        Test.@test Cs_out[i] ≈ Cref
    end

    Cs_sol = [zeros(N, Nφ) for _ in 1:P]
    NUFSHT.nusht_solve!(Cs_sol, fs, plans; execution = ComputationalBackends.ThreadedBackend(), rtol = 1e-6,
                        maxiter = 1000)
    for i in 1:P
        Cref = zeros(N, Nφ); NUFSHT.nusht_solve!(Cref, fs[i], plans[i]; rtol = 1e-6, maxiter = 1000)
        Test.@test Cs_sol[i] ≈ Cref
    end

    # Analyses: the quadrature with one weight vector per problem, and the fit without weights.
    wq = [(4π / M) .* (0.5 .+ rand(M)) for _ in 1:P]
    Cs_q = [zeros(N, Nφ) for _ in 1:P]
    NUFSHT.nusht_analyze!(Cs_q, fs, plans; execution = ComputationalBackends.ThreadedBackend(), weights = wq)
    Cs_a = [zeros(N, Nφ) for _ in 1:P]
    NUFSHT.nusht_analyze!(Cs_a, fs, plans; execution = ComputationalBackends.ThreadedBackend(), rtol = 1e-6,
                          maxiter = 1000)
    for i in 1:P
        Cq = zeros(N, Nφ); NUFSHT.nusht_analyze!(Cq, fs[i], plans[i]; weights = wq[i])
        Test.@test Cs_q[i] ≈ Cq
        Ca = zeros(N, Nφ); NUFSHT.nusht_analyze!(Ca, fs[i], plans[i]; rtol = 1e-6, maxiter = 1000)
        Test.@test Cs_a[i] ≈ Ca
    end

    splans = [NUFSHT.make_spin_plan(probs[i][1], probs[i][2], lmax, 1; tol = 1e-10, nthreads = 1) for i in 1:P]
    sfs = [randn(ComplexF64, N, Nφ) for _ in 1:P]
    gref = [zeros(ComplexF64, M) for _ in 1:P]
    for i in 1:P
        NUFSHT.nusht_type2_spin!(gref[i], sfs[i], splans[i])
    end
    gs = [zeros(ComplexF64, M) for _ in 1:P]
    NUFSHT.nusht_type2_spin!(gs, sfs, splans; execution = ComputationalBackends.ThreadedBackend())
    sas = [zeros(ComplexF64, N, Nφ) for _ in 1:P]
    NUFSHT.nusht_type1_spin!(sas, gref, splans; execution = ComputationalBackends.ThreadedBackend())
    for i in 1:P
        Test.@test gs[i] ≈ gref[i]
        sa = zeros(ComplexF64, N, Nφ); NUFSHT.nusht_type1_spin!(sa, gref[i], splans[i])
        Test.@test sas[i] ≈ sa
    end
    foreach(NUFSHT.close!, splans)

    health_after = zeros(M); NUFSHT.nusht_type2!(health_after, Cs[1], plans[1])
    Test.@test health_after ≈ health_before

    @info "OhMyThreads ext validated with $(Threads.nthreads()) thread(s), P=$P problems (incl. solve)"
end

# The per-column sphere loops are threaded across a per-task pool of slice buffers and plans. A
# `ntrans = 1` plan has an empty pool, so it is the serial reference at any thread count.
Test.@testset "threaded sphere loops agree with serial" begin
    Random.seed!(404)
    lmax, M, B = 24, 900, 4
    N, Nf = lmax + 1, 2lmax + 1
    θ = acos.(2 .* rand(M) .- 1); φ = 2π .* rand(M)

    pB = NUFSHT.make_plan(Float64, θ, φ, lmax; ntrans = B)
    p1 = NUFSHT.make_plan(Float64, θ, φ, lmax)
    Test.@test isempty(p1.sph_pool)                              # serial reference
    if Threads.nthreads() > 1
        Test.@test !isempty(pB.sph_pool)                         # threaded path really is exercised
    end

    C = randn(N, Nf, B)
    fB = zeros(M, B); NUFSHT.nusht_type2!(fB, C, pB)
    CaB = zeros(N, Nf, B); NUFSHT.nusht_type1!(CaB, fB, pB)

    # The runs differ by NUFFT width and the order the sphere loop visits columns, a reassociation; a
    # column in the wrong slot, a stale per-task buffer or a corrupted task is an O(1) error.
    for b in 1:B
        f1 = zeros(M); NUFSHT.nusht_type2!(f1, C[:, :, b], p1)
        Test.@test maximum(abs, fB[:, b] .- f1) / maximum(abs, f1) < 1e-10

        Ca1 = zeros(N, Nf); NUFSHT.nusht_type1!(Ca1, f1, p1)
        Test.@test maximum(abs, CaB[:, :, b] .- Ca1) / maximum(abs, Ca1) < 1e-10
    end

    @info "sphere loops: pool=$(length(pB.sph_pool)) at $(Threads.nthreads()) thread(s), matches serial"
    NUFSHT.close!(pB); NUFSHT.close!(p1)

    # `execution` sets the thread count, an explicit `nthreads` overrides it, and `SerialBackend` refuses
    # any count but one.
    CB = ComputationalBackends
    pS = NUFSHT.make_plan(Float64, θ, φ, lmax; execution = CB.SerialBackend(), ntrans = B)
    pT = NUFSHT.make_plan(Float64, θ, φ, lmax; execution = CB.ThreadedBackend(), ntrans = B)
    pT1 = NUFSHT.make_plan(Float64, θ, φ, lmax; execution = CB.ThreadedBackend(), nthreads = 1, ntrans = B)
    Test.@test isempty(pS.sph_pool) && isempty(pT1.sph_pool)
    nT = min(Threads.nthreads(), B)
    Test.@test length(pT.sph_pool) == (nT > 1 ? nT : 0)
    Test.@test_throws ArgumentError NUFSHT.make_plan(Float64, θ, φ, lmax; execution = CB.SerialBackend(), nthreads = 2)
    Test.@test_throws ArgumentError NUFSHT.make_plan(Float64, θ, φ, lmax; execution = CB.MPIBackend())
    fS = zeros(M, B); NUFSHT.nusht_type2!(fS, C, pS)
    fT = zeros(M, B); NUFSHT.nusht_type2!(fT, C, pT)
    Test.@test maximum(abs, fT .- fS) / maximum(abs, fS) < 1e-10
    foreach(NUFSHT.close!, (pS, pT, pT1))

    # A spin plan takes the same `execution`: its S-step divides the orders among the plan's tasks, one
    # workspace each, and each column of a batch equals the one-column plan's.
    s = 1
    sS = NUFSHT.make_spin_plan(θ, φ, lmax, s; execution = CB.SerialBackend())
    sT = NUFSHT.make_spin_plan(θ, φ, lmax, s; execution = CB.ThreadedBackend())
    Test.@test sS.ntasks == 1 && sT.ntasks == Threads.nthreads()
    Test.@test_throws ArgumentError NUFSHT.make_spin_plan(θ, φ, lmax, s; execution = CB.SerialBackend(), nthreads = 2)
    Test.@test_throws ArgumentError NUFSHT.make_spin_plan(θ, φ, lmax, s; execution = CB.DistributedBackend())
    sf = randn(ComplexF64, lmax + 1, 2lmax + 1)
    gS = zeros(ComplexF64, M); NUFSHT.nusht_type2_spin!(gS, sf, sS)
    gT = zeros(ComplexF64, M); NUFSHT.nusht_type2_spin!(gT, sf, sT)
    Test.@test maximum(abs, gT .- gS) / maximum(abs, gS) < 1e-10

    sB = NUFSHT.make_spin_plan(θ, φ, lmax, s; execution = CB.ThreadedBackend(), ntrans = B)
    Test.@test sB.engine isa NUFSHT._RingEngine
    Test.@test length(sB.engine.work) == Threads.nthreads()
    sfB = randn(ComplexF64, lmax + 1, 2lmax + 1, B)
    gB = zeros(ComplexF64, M, B); NUFSHT.nusht_type2_spin!(gB, sfB, sB)
    aB = zeros(ComplexF64, size(sfB)); NUFSHT.nusht_type1_spin!(aB, gB, sB)
    for b in 1:B
        g1 = zeros(ComplexF64, M); NUFSHT.nusht_type2_spin!(g1, sfB[:, :, b], sS)
        Test.@test maximum(abs, gB[:, b] .- g1) / maximum(abs, g1) < 1e-10
        a1 = zeros(ComplexF64, lmax + 1, 2lmax + 1); NUFSHT.nusht_type1_spin!(a1, g1, sS)
        Test.@test maximum(abs, aB[:, :, b] .- a1) / maximum(abs, a1) < 1e-10
    end
    foreach(NUFSHT.close!, (sS, sT, sB))
end

# FastTransforms plans on the libfftw3 FFTW.jl loads, so the sphere plans are built under FFTW.jl's own
# planner lock, and a plan build leaves the planner thread count as it found it.
Test.@testset "sphere plans are built on FFTW.jl's planner and leave its count as found" begin
    FFTW = NUFSHT.FFTW
    prev = FFTW.get_num_threads()
    NUFSHT.FastTransforms.ft_fftw_plan_with_nthreads(5)
    Test.@test FFTW.get_num_threads() == 5                 # one planner count, both libraries
    FFTW.set_num_threads(3)
    Random.seed!(5)
    θ = acos.(2 .* rand(200) .- 1); φ = 2π .* rand(200)
    p = NUFSHT.make_plan(Float64, θ, φ, 8; ntrans = 2)
    Test.@test FFTW.get_num_threads() == 3
    Test.@test !islocked(FFTW.fftwlock)
    sp = NUFSHT.make_spin_plan(θ, φ, 8, 1; ntrans = 2)
    Test.@test sp.engine isa NUFSHT._RingEngine
    Test.@test FFTW.get_num_threads() == 3
    Test.@test !islocked(FFTW.fftwlock)
    NUFSHT.close!(p); NUFSHT.close!(sp)
    FFTW.set_num_threads(prev)
end

# `_OMPProbe` stands in for a FastTransforms plan in `_ft_lmul!`, the call every plan's FastTransforms
# step goes through, and records the OpenMP count its `lmul!` runs at.
struct _OMPProbe end
_omp_threads() = Int(ccall((:omp_get_max_threads, NUFSHT.FastTransforms.libfasttransforms), Cint, ()))
LinearAlgebra.lmul!(::_OMPProbe, r::Base.RefValue{Int}) = (r[] = _omp_threads(); r)

# A plan's FastTransforms calls run at the plan's thread count: one under `SerialBackend`, `nthreads`
# under `ThreadedBackend`, and one inside a farm's scope or the tasks of the per-column loop. Each call,
# and each plan build and transform, leaves the count it found.
Test.@testset "a plan's FastTransforms calls run at its thread count" begin
    CB = ComputationalBackends
    prev = _omp_threads()
    NUFSHT.FastTransforms.ft_set_num_threads(5)
    Random.seed!(17)
    lmax, M, B = 8, 300, 2
    θ = acos.(2 .* rand(M) .- 1); φ = 2π .* rand(M)
    kw = (; nufft = FTB.FINUFFTBackend(), ntrans = B)
    serial = NUFSHT.make_plan(Float64, θ, φ, lmax; kw..., execution = CB.SerialBackend())
    threaded = NUFSHT.make_plan(Float64, θ, φ, lmax; kw..., execution = CB.ThreadedBackend(), nthreads = 3)
    Test.@test _omp_threads() == 5

    r = Ref(0)
    Test.@test serial.ntasks == 1
    NUFSHT._ft_lmul!(_OMPProbe(), r, serial)
    Test.@test r[] == 1
    Test.@test threaded.ntasks == 3
    NUFSHT._ft_lmul!(_OMPProbe(), r, threaded)
    Test.@test r[] == 3
    Base.ScopedValues.with(() -> NUFSHT._ft_lmul!(_OMPProbe(), r, threaded), FTB.FASTTRANSFORMS_THREADS => 1)
    Test.@test r[] == 1
    rs = (Ref(0), Ref(0))
    NUFSHT._pool_columns!((q, b) -> NUFSHT._ft_lmul!(_OMPProbe(), q, threaded), (Ref(0),),
                          [(rs[1],), (rs[2],)], 2)
    Test.@test rs[1][] == 1 && rs[2][] == 1
    Test.@test _omp_threads() == 5

    # The transforms, through the single column and the column pool.
    C = randn(lmax + 1, 2lmax + 1, B); f = zeros(M, B)
    for p in (serial, threaded)
        NUFSHT.nusht_type2!(f, C, p); NUFSHT.nusht_type1!(C, f, p)
        Test.@test _omp_threads() == 5
    end
    foreach(NUFSHT.close!, (serial, threaded))
    NUFSHT.FastTransforms.ft_set_num_threads(prev)
end
