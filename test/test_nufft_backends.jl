using SpectralBackends: SpectralBackends

const FTB_LIBRARIES = (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend())

# Both libraries against direct summation, in both directions, over the whole coefficient array.
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

Test.@testset "NUFFT backend selection" begin
    θ, φ = iid_points(40, 5)
    Test.@test_throws ArgumentError NUFSHT.make_plan(θ, φ, 4; nufft = SpectralBackends.NUFFTSpectralBackend())
end

# A real field's mode array is Hermitian in `kθ`, so a real-`FE` plan stores its `kθ ≥ 0` half. Each
# backend's real plan is scored against the complex direct sum, which folds nothing, and against the
# analytic series; the adjoint identity pins the transpose's factor 2 on every `kθ > 0` row.
Test.@testset "real fields: folded mode array" begin
    lmax, M, B = 10, 500, 2
    N, Nf = lmax + 1, 2lmax + 1
    θ, φ = iid_points(M, 91)
    C = rand_coeffs(lmax, 92, B)
    ds = SpectralBackends.DirectSumSpectralBackend()

    pr = NUFSHT.make_plan(ComplexF64, θ, φ, lmax; tol = 1e-12, ntrans = B, nufft = ds)
    fr = zeros(ComplexF64, M, B); NUFSHT.nusht_type2!(fr, ComplexF64.(C), pr)
    Y = randn(M, B)
    Cr = zeros(ComplexF64, N, Nf, B); NUFSHT.nusht_type1!(Cr, ComplexF64.(Y), pr)
    for backend in (FTB_LIBRARIES..., ds)
        pf = NUFSHT.make_plan(Float64, θ, φ, lmax; tol = 1e-12, ntrans = B, nufft = backend)

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

# A `SynthesisOnly()` plan synthesises and refuses the adjoint on every backend.
Test.@testset "plan directions are honoured on every backend" begin
    Random.seed!(61)
    lmax, M = 8, 324
    N, Nf = lmax + 1, 2lmax + 1
    θ, φ = iid_points(M, 62)
    C = rand_coeffs(lmax, 63)
    ref = synth_ref(C, lmax, θ, φ)

    for backend in (FTB_LIBRARIES..., SpectralBackends.DirectSumSpectralBackend())
        so = NUFSHT.make_plan(Float64, θ, φ, lmax; tol = 1e-12, nufft = backend,
                              directions = NUFSHT.SynthesisOnly())
        f = zeros(M); NUFSHT.nusht_type2!(f, C, so)
        Test.@test relerr(f, ref) < 1e-11
        Test.@test_throws ArgumentError NUFSHT.nusht_type1!(zeros(N, Nf), f, so)
        NUFSHT.close!(so)
    end
end

# Every accepted `nthreads` gives the right transform, and a backend refuses a count it cannot deliver.
# Direct summation splits the axis each direction writes, so every count gives the same bits.
Test.@testset "NUFFT backends honour or refuse nthreads" begin
    Random.seed!(51)
    lmax, M, B = 8, 324, 2
    n1, n2 = 2lmax + 3, 2lmax + 1
    θ = clamp.(π .* rand(M), 1e-9, π - 1e-9); φ = 2π .* rand(M)
    ds = SpectralBackends.DirectSumSpectralBackend()

    modes = randn(ComplexF64, n1, n2, B); vals = randn(ComplexF64, M, B)
    ref = Dict{Int,Array{ComplexF64}}()
    for nt in (1, 0, Threads.nthreads()), ty in (2, 1)
        p = NUFSHT._nufft_makeplan(ds, (θ, φ), ty, [n1, n2], ty == 2 ? +1 : -1, B, 1e-12;
                                   dtype = Float64, nthreads = nt)
        out = ty == 2 ? zeros(ComplexF64, M, B) : zeros(ComplexF64, n1, n2, B)
        NUFSHT._nufft_exec!(p, ty == 2 ? modes : vals, out)
        if nt == 1
            ref[ty] = out
        else
            Test.@test out == ref[ty]
        end
    end
    Test.@test_throws ArgumentError NUFSHT.make_plan(Float64, θ, φ, lmax; nufft = ds, nthreads = -1)

    C = rand_coeffs(lmax, 52)
    for backend in FTB_LIBRARIES, nt in unique((nothing, 0, 1, Threads.nthreads()))
        p = NUFSHT.make_plan(Float64, θ, φ, lmax; tol = 1e-12, nthreads = nt, nufft = backend)
        f = zeros(M); NUFSHT.nusht_type2!(f, C, p)
        Test.@test relerr(f, synth_ref(C, lmax, θ, φ)) < 1e-11
        NUFSHT.close!(p)
    end
    # NonuniformFFTs spreads on one thread or on `Threads.nthreads()`, and refuses any other count.
    Test.@test_throws ArgumentError NUFSHT.make_plan(Float64, θ, φ, lmax;
        nthreads = Threads.nthreads() + 1, nufft = FTB.NonuniformFFTsBackend())
end

# Column `b` is band-limited to degree `b`, a subspace of dimension `(b+1)²`, so the columns converge at
# different iterations; on both libraries the batched solve equals the single-column one.
Test.@testset "batched solve retires correctly on every backend" begin
    Random.seed!(20)
    lmax, M, B = 8, 150, 4
    N, Nf = lmax + 1, 2lmax + 1
    θ = acos.(2 .* rand(M) .- 1); φ = 2π .* rand(M)
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

        # The NUFFT width a solve narrows to once at most `k` columns remain, driven directly: width
        # `k < B` builds a plan and agrees with the full width on its columns.
        k = 2
        Ck = zeros(N, Nf, B)
        for b in 1:B, ℓ in 0:3, m in -ℓ:ℓ
            Ck[FastSphericalHarmonics.sph_mode(ℓ, m), b] = randn()
        end
        ffull = zeros(M, B); NUFSHT.nusht_type2!(ffull, Ck, pB)
        fnarrow = zeros(M, B); NUFSHT.nusht_type2!(fnarrow, Ck, pB, k, k)
        Test.@test NUFSHT._pool_built(pB.size_pool) ≥ 1
        Test.@test maximum(abs, fnarrow[:, 1:k] .- ffull[:, 1:k]) /
                   maximum(abs, ffull[:, 1:k]) < 1e-12

        NUFSHT.close!(pB)
        NUFSHT.close!(pB)                        # idempotent
        NUFSHT.close!(p1)
    end
end
