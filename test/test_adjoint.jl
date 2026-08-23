# `@allocated` measures the harness as well as the call. A keyword call written inline in testset
# scope boxes its arguments on Julia 1.11 (464 B for the `nusht_filter!` line below, while the same
# call through a function barrier in `test_allocs.jl` reports 0). Every count here therefore goes
# through a barrier of fixed arity, so what is measured is the transform.
_alloc_t2(f, C, p) = @allocated NUFSHT.nusht_type2!(f, C, p)
_alloc_t1(C, f, p) = @allocated NUFSHT.nusht_type1!(C, f, p)
_alloc_adj(C, f, p) = @allocated NUFSHT._nusht_true_adjoint!(C, f, p)
_alloc_filt(o, f, ft, p, w) = @allocated NUFSHT.nusht_filter!(o, f, ft, p; ws = w)

Test.@testset "Euclidean adjoint, zero-allocation, and batching" begin
    Random.seed!(7)
    lmax = 8
    Nθ, Nφ = lmax + 1, 2lmax + 1
    M = 400
    θ = π .* rand(M)
    φ = 2π .* rand(M)
    # `nthreads = 1`: the zero-allocation guarantee is FINUFFT-single-threaded (see the alloc testset
    # below). With the all-cores default FINUFFT takes its FFTW-planner lock per exec and allocates a
    # little outside our control, which `nusht_filter!` multiplies since it now runs a solve.
    plan = NUFSHT.make_plan(θ, φ, lmax; tol = 1e-12, nthreads = 1)

    Test.@testset "scalar Euclidean adjoint ⟨Ax,y⟩ = ⟨x,A†y⟩ (scattered)" begin
        x = randn(Nθ, Nφ)
        y = randn(M)
        Ax = zeros(M); NUFSHT.nusht_type2!(Ax, x, plan)
        Aty = zeros(Nθ, Nφ); NUFSHT._nusht_true_adjoint!(Aty, y, plan)
        lhs = sum(Ax .* y)
        rhs = sum(x .* Aty)
        rel = abs(lhs - rhs) / abs(lhs)
        @info "scalar adjoint: |⟨Ax,y⟩-⟨x,A†y⟩|/|⟨Ax,y⟩| = $rel"
        Test.@test rel < 1e-11
    end

    Test.@testset "zero-allocation on warmed-up calls" begin
        C = randn(Nθ, Nφ); f = randn(M); out = zeros(M)
        Cout = zeros(Nθ, Nφ); Aty = zeros(Nθ, Nφ)
        filt = NUFSHT.gaussian_from_scale(2000e3)
        # Filtering fits coefficients, so it needs a solver workspace; supplying one is what makes it
        # allocation-free, exactly as the docstring says.
        ws = NUFSHT.LSMRWorkspace(plan)
        NUFSHT.nusht_type2!(out, C, plan)          # warmup
        NUFSHT.nusht_type1!(Cout, f, plan)
        NUFSHT._nusht_true_adjoint!(Aty, f, plan)
        NUFSHT.nusht_filter!(out, f, filt, plan; ws = ws)
        # The zero-alloc guarantee holds single-threaded. Under a multithreaded Julia,
        # FFTW/FINUFFT/FastTransforms internal threading (e.g. FINUFFT's FFTW-planner lock) adds small
        # external per-call allocations outside our control — there we only report the count.
        if Threads.nthreads() == 1
            Test.@test _alloc_t2(out, C, plan) == 0
            Test.@test _alloc_t1(Cout, f, plan) == 0
            Test.@test _alloc_adj(Aty, f, plan) == 0
            Test.@test _alloc_filt(out, f, filt, plan, ws) == 0
        else
            a = _alloc_filt(out, f, filt, plan, ws)
            @info "zero-alloc is a single-threaded guarantee. With $(Threads.nthreads()) Julia threads and nthreads=0 (all cores), nusht_filter! allocates $a B externally via FINUFFT/FFTW's thread-safe planner lock — build the plan with nthreads=1 (or run single-threaded) for zero allocation."
        end
    end

    Test.@testset "batched ntrans == looped single transforms (scalar)" begin
        # nthreads=1 so the batched (one B-way plan) and looped (B single plans) FINUFFT adjoints use
        # the same deterministic reduction — a mechanism test, exact by construction.
        B = 3
        planB = NUFSHT.make_plan(θ, φ, lmax; tol = 1e-12, ntrans = B, nthreads = 1)
        plan1 = NUFSHT.make_plan(θ, φ, lmax; tol = 1e-12, nthreads = 1)
        CB = randn(Nθ, Nφ, B); fB = zeros(M, B)
        NUFSHT.nusht_type2!(fB, CB, planB)
        fBr = randn(M, B); CBout = zeros(Nθ, Nφ, B)
        NUFSHT.nusht_type1!(CBout, fBr, planB)
        for b in 1:B
            fb = zeros(M); NUFSHT.nusht_type2!(fb, CB[:, :, b], plan1)
            Test.@test maximum(abs.(fB[:, b] .- fb)) < 1e-12
            Cb = zeros(Nθ, Nφ); NUFSHT.nusht_type1!(Cb, fBr[:, b], plan1)
            Test.@test maximum(abs.(CBout[:, :, b] .- Cb)) < 1e-12
        end
    end

    Test.@testset "spin: zero-allocation + batched == looped" begin
        s = 1
        splan = NUFSHT.make_spin_plan(θ, φ, lmax, s; tol = 1e-12)
        sf = zeros(ComplexF64, Nθ, Nφ)
        for ℓ in abs(s):lmax, m in -ℓ:ℓ
            sf[NUFSHT.spin_coeff_index(ℓ, m, lmax)] = randn(ComplexF64)
        end
        fs = zeros(ComplexF64, M); sfout = zeros(ComplexF64, Nθ, Nφ)
        NUFSHT.nusht_type2_spin!(fs, sf, splan)     # warmup
        NUFSHT.nusht_type1_spin!(sfout, fs, splan)
        if Threads.nthreads() == 1   # see note above re: multithreaded external overhead
            Test.@test (@allocated NUFSHT.nusht_type2_spin!(fs, sf, splan)) == 0
            Test.@test (@allocated NUFSHT.nusht_type1_spin!(sfout, fs, splan)) == 0
        end

        B = 2
        splanB = NUFSHT.make_spin_plan(θ, φ, lmax, s; tol = 1e-12, ntrans = B)
        sfB = zeros(ComplexF64, Nθ, Nφ, B)
        for b in 1:B, ℓ in abs(s):lmax, m in -ℓ:ℓ
            sfB[NUFSHT.spin_coeff_index(ℓ, m, lmax), b] = randn(ComplexF64)
        end
        fsB = zeros(ComplexF64, M, B)
        NUFSHT.nusht_type2_spin!(fsB, sfB, splanB)
        for b in 1:B
            fb = zeros(ComplexF64, M); NUFSHT.nusht_type2_spin!(fb, sfB[:, :, b], splan)
            Test.@test maximum(abs.(fsB[:, b] .- fb)) < 1e-12
        end
    end
end
