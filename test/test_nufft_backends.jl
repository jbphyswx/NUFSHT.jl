using SpectralBackends: SpectralBackends

const FTB_LIBRARIES = (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend())

# Direct summation shares no code with either library, so it is the reference both are scored against,
# in both directions: `nusht_solve!` converges only when the pair is an exact transpose.
Test.@testset "NUFFT backends agree with direct summation" begin
    Random.seed!(77)
    for (lmax, M) in ((8, 200), (12, 300))
        N = lmax + 1; Nφ = 2lmax + 1
        θ = clamp.(π .* rand(M), 1e-9, π - 1e-9)
        φ = 2π .* rand(M)
        C = randn(N, Nφ)

        vals = Dict{String,Vector{Float64}}()
        coef = Dict{String,Matrix{Float64}}()
        for (name, backend) in ("directsum" => SpectralBackends.DirectSumSpectralBackend(),
                                "finufft" => FTB.FINUFFTBackend(),
                                "nonuniformffts" => FTB.NonuniformFFTsBackend())
            plan = NUFSHT.make_plan(θ, φ, lmax; tol = 1e-12, nufft = backend)
            f = zeros(M); NUFSHT.nusht_type2!(f, C, plan); vals[name] = f
            Cb = zeros(N, Nφ); NUFSHT.nusht_type1!(Cb, f, plan); coef[name] = Cb
            NUFSHT.close!(plan)
        end

        for name in ("finufft", "nonuniformffts")
            Test.@test relerr(vals[name], vals["directsum"]) < 1e-10
            Test.@test relerr(coef[name], coef["directsum"]) < 1e-10
        end
    end
end

# `AutoSpectralBackend` takes the first loaded library in NUFSHT's order; a named backend passes
# through for every field type, and one NUFSHT cannot run is refused.
Test.@testset "NUFFT backend selection" begin
    Auto = SpectralBackends.AutoSpectralBackend()
    for FE in (Float64, Float32, ComplexF64)
        Test.@test NUFSHT._resolve_nufft(Auto, FE) === FTB.NonuniformFFTsBackend()
    end
    for FE in (Float64, ComplexF64),
        b in (FTB_LIBRARIES..., SpectralBackends.DirectSumSpectralBackend())
        Test.@test NUFSHT._resolve_nufft(b, FE) === b
    end
    θ, φ = iid_points(40, 5)
    Test.@test_throws ArgumentError NUFSHT.make_plan(θ, φ, 4; nufft = SpectralBackends.NUFFTSpectralBackend())
end

# A real field's mode array is Hermitian in `kθ`, so every plan stores its `kθ ≥ 0` half. Both
# FlowTransformBindings libraries take real strengths for it; direct summation runs a half-height
# complex transform with a per-point phase and doubled `kθ > 0` rows. Each is scored against the complex
# direct sum, which folds nothing, and against the analytic series. The adjoint identity pins the
# transpose's factor 2 on every `kθ > 0` row, which the forward direction does not carry.
Test.@testset "real fields: folded mode array" begin
    for b in FTB_LIBRARIES
        Test.@test NUFSHT._real_capable(b)
    end
    Test.@test !NUFSHT._real_capable(SpectralBackends.DirectSumSpectralBackend())

    lmax, M, B = 10, 500, 2
    N, Nf = lmax + 1, 2lmax + 1
    θ, φ = iid_points(M, 91)
    C = rand_coeffs(lmax, 92, B)
    ds = SpectralBackends.DirectSumSpectralBackend()

    for (backend, FE, folded, ZS) in
            ((FTB.NonuniformFFTsBackend(), Float64,    true,  Float64),
             (FTB.NonuniformFFTsBackend(), ComplexF64, false, ComplexF64),
             (FTB.FINUFFTBackend(),        Float64,    true,  Float64),
             (FTB.FINUFFTBackend(),        ComplexF64, false, ComplexF64),
             (ds,                          Float64,    true,  ComplexF64),
             (ds,                          ComplexF64, false, ComplexF64))
        p = NUFSHT.make_plan(FE, θ, φ, lmax; tol = 1e-12, ntrans = B, nufft = backend)
        Test.@test size(p.Fhat, 1) == (folded ? lmax + 2 : 2lmax + 3)
        Test.@test eltype(NUFSHT._fbuf(p)) == ZS
        # A half-height complex transform labels its rows from `-(lmax+2)÷2`, which a per-point phase
        # undoes; the real transform's half-spectrum starts at `kθ = 0`.
        Test.@test (NUFSHT._θshift(p) !== nothing) == (folded && ZS === ComplexF64)
        NUFSHT.close!(p)
    end

    pr = NUFSHT.make_plan(ComplexF64, θ, φ, lmax; tol = 1e-12, ntrans = B, nufft = ds)
    fr = zeros(ComplexF64, M, B); NUFSHT.nusht_type2!(fr, ComplexF64.(C), pr)
    Y = randn(M, B)
    Cr = zeros(ComplexF64, N, Nf, B); NUFSHT.nusht_type1!(Cr, ComplexF64.(Y), pr)
    for backend in (FTB_LIBRARIES..., ds)
        pf = NUFSHT.make_plan(Float64, θ, φ, lmax; tol = 1e-12, ntrans = B, nufft = backend)
        Test.@test size(pf.Fhat, 1) == lmax + 2

        ff = zeros(M, B); NUFSHT.nusht_type2!(ff, C, pf)
        Test.@test relerr(ff, real.(fr)) < 1e-11
        for b in 1:B
            Test.@test relerr(ff[:, b], synth_ref(C[:, :, b], lmax, θ, φ)) < 1e-11
        end

        Cf = zeros(N, Nf, B); NUFSHT.nusht_type1!(Cf, Y, pf)
        Test.@test relerr(Cf, real.(Cr)) < 1e-11
        Test.@test abs(sum(ff .* Y) - sum(C .* Cf)) / abs(sum(ff .* Y)) < 1e-11

        Sf = zeros(N, Nf, B)
        _, _, _, conv = NUFSHT.nusht_solve!(Sf, ff, pf; rtol = 1e-10, maxiter = 200)
        Test.@test conv
        Test.@test relerr(Sf, C) < 1e-8
        NUFSHT.close!(pf)
    end
    NUFSHT.close!(pr)
end

# `maxlog = 1` caps a log statement per logger instance, and `@test_logs` installs a fresh one, so
# each assertion sees the warning when it fires.
Test.@testset "direct summation warns only when Auto chose it" begin
    Auto = SpectralBackends.AutoSpectralBackend()
    ds = SpectralBackends.DirectSumSpectralBackend()
    Test.@test_logs (:warn,) NUFSHT._warn_if_directsum(Auto, ds, 484, 483)
    Test.@test_logs NUFSHT._warn_if_directsum(ds, ds, 484, 483)
    Test.@test_logs NUFSHT._warn_if_directsum(Auto, FTB.FINUFFTBackend(), 484, 483)
end

# `plan_memory` sums every buffer the plan holds, and an empty slot counts zero.
Test.@testset "plan_memory accounts for every buffer, empty ones included" begin
    Random.seed!(71)
    lmax, M = 8, 324
    θ, φ = iid_points(M, 72)
    p = NUFSHT.make_plan(Float64, θ, φ, lmax; tol = 1e-10, nthreads = 1)
    pm = NUFSHT.plan_memory(p)
    Test.@test pm.total == pm.C + pm.F + pm.Fhat + pm.Fslice + pm.fbuf + pm.nodes +
                           pm.size_pool + pm.sph_pool
    # Only filtering needs a coefficient scratch, so a plan that has never filtered holds none.
    Test.@test p.C[] === nothing && pm.C == 0
    Test.@test pm.F > 0 && pm.Fhat > 0 && pm.fbuf > 0

    C = rand_coeffs(lmax, 73)
    f = zeros(M); NUFSHT.nusht_type2!(f, C, p)
    Test.@test p.C[] === nothing
    out = zeros(M); ws = NUFSHT.LSMRWorkspace(p)
    NUFSHT.nusht_filter!(out, f, NUFSHT.gaussian_from_scale(2000e3), p; ws = ws)
    Test.@test p.C[] !== nothing
    Test.@test NUFSHT.plan_memory(p).C > 0
    Test.@test NUFSHT.plan_memory(p).total > pm.total
    NUFSHT.close!(p)
end

# `directions` means the same on every backend: a `SynthesisOnly()` plan answers `nusht_type2!` and
# refuses the adjoint, including on a library whose one plan serves both directions.
Test.@testset "plan directions are honoured on every backend" begin
    Random.seed!(61)
    lmax, M = 8, 324
    N, Nf = lmax + 1, 2lmax + 1
    θ, φ = iid_points(M, 62)
    C = rand_coeffs(lmax, 63)
    ref = synth_ref(C, lmax, θ, φ)
    ds = SpectralBackends.DirectSumSpectralBackend()
    gref = let p = NUFSHT.make_plan(Float64, θ, φ, lmax; nufft = ds)
        g = zeros(N, Nf); NUFSHT.nusht_type1!(g, ref, p); NUFSHT.close!(p); g
    end

    for backend in (FTB_LIBRARIES..., ds)
        so = NUFSHT.make_plan(Float64, θ, φ, lmax; tol = 1e-12, nufft = backend,
                              directions = NUFSHT.SynthesisOnly())
        Test.@test so.nodes.nufft_type1 === nothing
        f = zeros(M); NUFSHT.nusht_type2!(f, C, so)
        Test.@test relerr(f, ref) < 1e-11
        Test.@test_throws ArgumentError NUFSHT.nusht_type1!(zeros(N, Nf), f, so)
        Test.@test_throws ArgumentError NUFSHT.nusht_solve!(zeros(N, Nf), f, so; maxiter = 5)
        Test.@test NUFSHT.plan_memory(so).total > 0
        NUFSHT.close!(so)

        both = NUFSHT.make_plan(Float64, θ, φ, lmax; tol = 1e-12, nufft = backend)
        g = zeros(N, Nf); NUFSHT.nusht_type1!(g, ref, both)
        Test.@test relerr(g, gref) < 1e-10
        # Where one plan serves both directions the analysis handle reads the synthesis handle's plan.
        shared = NUFSHT._nufft_share_directions(backend)
        Test.@test NUFSHT._nufft_derived(both.nodes.nufft_type1) == shared
        shared && Test.@test both.nodes.nufft_type1.plan === both.nodes.nufft_type2.plan
        NUFSHT.close!(both)
    end
end

# A backend meets the `nthreads` it is given or refuses it. Direct summation splits the axis each
# direction writes, so every count gives the same bits. The FlowTransformBindings plans receive the
# count NUFSHT resolved: `nothing` and `0` take `Threads.nthreads()`. A plan run twice returns the same
# bits in type 2 at any count and in type 1 on one thread; a multi-threaded type 1 can differ in the
# last bit between runs on both libraries.
Test.@testset "NUFFT backends honour or refuse nthreads" begin
    Random.seed!(51)
    lmax, M, B = 8, 324, 2
    N, Nf = lmax + 1, 2lmax + 1
    n1, n2 = 2lmax + 3, Nf
    θ = clamp.(π .* rand(M), 1e-9, π - 1e-9); φ = 2π .* rand(M)
    ds = SpectralBackends.DirectSumSpectralBackend()

    modes = randn(ComplexF64, n1, n2, B); vals = randn(ComplexF64, M, B)
    ref = Dict{Int,Array{ComplexF64}}()
    for nt in (1, 0, Threads.nthreads()), ty in (2, 1)
        p = NUFSHT._nufft_makeplan(ds, (θ, φ), ty, [n1, n2], ty == 2 ? +1 : -1, B, 1e-12;
                                   dtype = Float64, modeord = 0, nthreads = nt)
        Test.@test p.nthreads == (nt == 0 ? Threads.nthreads() : nt)
        out = ty == 2 ? zeros(ComplexF64, M, B) : zeros(ComplexF64, n1, n2, B)
        NUFSHT._nufft_exec!(p, ty == 2 ? modes : vals, out)
        if nt == 1
            ref[ty] = out
        else
            Test.@test out == ref[ty]
        end
    end
    Test.@test_throws ArgumentError NUFSHT._nufft_makeplan(ds, (θ, φ), 2, [n1, n2], +1, B, 1e-12;
                                                           dtype = Float64, nthreads = -1)

    C = rand_coeffs(lmax, 52)
    spec_threads(p) = p.nodes.nufft_type2.plan.spec.nthreads
    for backend in FTB_LIBRARIES, nt in unique((nothing, 0, 1, Threads.nthreads()))
        p = NUFSHT.make_plan(Float64, θ, φ, lmax; tol = 1e-12, nthreads = nt, nufft = backend)
        Test.@test spec_threads(p) == (something(nt, 0) == 0 ? Threads.nthreads() : nt)
        f1 = zeros(M); f2 = zeros(M)
        NUFSHT.nusht_type2!(f1, C, p); NUFSHT.nusht_type2!(f2, C, p)
        g1 = zeros(N, Nf); g2 = zeros(N, Nf)
        NUFSHT.nusht_type1!(g1, f1, p); NUFSHT.nusht_type1!(g2, f1, p)
        Test.@test f1 == f2
        if spec_threads(p) == 1
            Test.@test g1 == g2
        else
            Test.@test relerr(g1, g2) < 8eps()
        end
        Test.@test relerr(f1, synth_ref(C, lmax, θ, φ)) < 1e-11
        NUFSHT.close!(p)
    end
    # NonuniformFFTs spreads on one thread or on `Threads.nthreads()`, and refuses any other count.
    Test.@test_throws ArgumentError NUFSHT.make_plan(Float64, θ, φ, lmax;
        nthreads = Threads.nthreads() + 1, nufft = FTB.NonuniformFFTsBackend())
    pf = NUFSHT.make_plan(Float64, θ, φ, lmax; tol = 1e-12, nthreads = 2, nufft = FTB.FINUFFTBackend())
    Test.@test spec_threads(pf) == 2
    NUFSHT.close!(pf)
end

# A batched solve retires columns as they converge and transforms only the live ones, through plans of
# narrower widths. FINUFFT's plans share one type at every width, so they share one `Vector` pool;
# NonuniformFFTs' carry the width in their type, so each width has a slot typed by
# `FlowTransformBindings.plan_type`, and a plan of another type cannot be stored in it.
Test.@testset "batched solve retires correctly on every backend" begin
    Random.seed!(20)
    lmax, M, B = 8, 150, 4
    N, Nf = lmax + 1, 2lmax + 1
    θ = acos.(2 .* rand(M) .- 1); φ = 2π .* rand(M)
    # Column `b` is band-limited to degree `b`, a subspace of dimension `(b+1)²`, so the columns
    # converge at different iterations.
    F = zeros(M, B)
    let p0 = NUFSHT.make_plan(Float64, θ, φ, lmax; nthreads = 1)
        for b in 1:B
            Cb = zeros(N, Nf)
            for ℓ in 0:b, m in -ℓ:ℓ
                Cb[FastSphericalHarmonics.sph_mode(ℓ, m)] = randn()
            end
            NUFSHT.nusht_type2!(view(F, :, b), Cb, p0)
        end
        NUFSHT.close!(p0)
    end

    for backend in FTB_LIBRARIES
        pB = NUFSHT.make_plan(Float64, θ, φ, lmax; ntrans = B, nufft = backend)
        p1 = NUFSHT.make_plan(Float64, θ, φ, lmax; nufft = backend)

        Test.@test pB.pool_recipe.narrowable
        Test.@test (pB.size_pool isa Tuple) == !NUFSHT._width_polymorphic(backend)
        Test.@test NUFSHT._pool_built(pB.size_pool) == 0

        CB = zeros(N, Nf, B)
        _, itB, relB, convB = NUFSHT.nusht_solve!(CB, F, pB; rtol = 1e-8, maxiter = 500)
        Test.@test itB < 500
        Test.@test convB
        Test.@test relB < 1e-8

        for b in 1:B
            c = zeros(N, Nf)
            NUFSHT.nusht_solve!(c, F[:, b], p1; rtol = 1e-8, maxiter = 500)
            Test.@test maximum(abs, CB[:, :, b] .- c) / maximum(abs, c) < 1e-4
        end

        # When a solve's columns cross `rtol` together it never narrows, so the narrowed transform is
        # driven directly: width `k < B` builds a plan and agrees with the full width on its columns.
        k = 2
        Ck = zeros(N, Nf, B)
        for b in 1:B, ℓ in 0:3, m in -ℓ:ℓ
            Ck[FastSphericalHarmonics.sph_mode(ℓ, m), b] = randn()
        end
        ffull = zeros(M, B); NUFSHT.nusht_type2!(ffull, Ck, pB)
        Test.@test NUFSHT._pool_built(pB.size_pool) == 0        # full width never touches the pool
        fnarrow = zeros(M, B); NUFSHT.nusht_type2!(fnarrow, Ck, pB, k, k)
        Test.@test NUFSHT._pool_built(pB.size_pool) == 1
        Test.@test maximum(abs, fnarrow[:, 1:k] .- ffull[:, 1:k]) /
                   maximum(abs, ffull[:, 1:k]) < 1e-12

        NUFSHT.close!(pB)
        Test.@test NUFSHT._pool_built(pB.size_pool) == 0
        NUFSHT.close!(pB)                        # idempotent
        NUFSHT.close!(p1)
    end
end
