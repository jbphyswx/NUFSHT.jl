using FINUFFT: FINUFFT                     # triggers NUFSHTFINUFFTExt
using NonuniformFFTs: NonuniformFFTs        # triggers NUFSHTNonuniformFFTsExt
using SpectralBackends: SpectralBackends

# The `N` step is pluggable: direct summation (no dependencies, always available) plus one extension
# per library. Direct summation is the independent reference — it shares no code with either library —
# so both fast backends are checked against it rather than against each other, and the adjoint is
# checked too, since `nusht_solve!`'s convergence depends on the pair being an exact transpose.
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
                                "finufft" => NUFSHT.FINUFFTBackend(),
                                "nonuniformffts" => NUFSHT.NonuniformFFTsBackend())
            plan = NUFSHT.make_plan(θ, φ, lmax; tol = 1e-12, nufft = backend)
            f = zeros(M); NUFSHT.nusht_type2!(f, C, plan); vals[name] = f
            Cb = zeros(N, Nφ); NUFSHT.nusht_type1!(Cb, f, plan); coef[name] = Cb
            NUFSHT.close!(plan)
        end

        relerr(a, b) = sqrt(sum(abs2, a .- b) / sum(abs2, b))
        for name in ("finufft", "nonuniformffts")
            Test.@test relerr(vals[name], vals["directsum"]) < 1e-10
            Test.@test relerr(coef[name], coef["directsum"]) < 1e-10
        end
    end
    @info "NUFFT backends: FINUFFT and NonuniformFFTs match direct summation (synthesis and adjoint)"
end

# `AutoSpectralBackend` must choose a loaded fast backend, and a backend whose extension is missing
# must be refused rather than quietly swapped for a slower one.
Test.@testset "NUFFT backend selection" begin
    Test.@test NUFSHT._resolve_nufft(SpectralBackends.AutoSpectralBackend()) isa NUFSHT.FINUFFTBackend
    Test.@test NUFSHT._resolve_nufft(NUFSHT.NonuniformFFTsBackend()) isa NUFSHT.NonuniformFFTsBackend
    Test.@test NUFSHT._resolve_nufft(SpectralBackends.DirectSumSpectralBackend()) isa
               SpectralBackends.DirectSumSpectralBackend
end

# A real field's mode array is Hermitian in `kθ`, so a backend with a genuine real-data transform is
# handed only the `kθ ≥ 0` half and returns real strengths — halving the mode array, the upsampled FFT
# and the spreading. It has to be the SAME operator, so this scores the folded path against direct
# summation (which never folds) rather than against itself, and pins the adjoint identity: the
# embedding `Z[-k] = conj(Z[k])` is R-linear but not C-linear, and its transpose carries a factor 2 on
# every `kθ > 0` row that the forward direction must NOT have.
Test.@testset "real fast path: folded mode array" begin
    Test.@test NUFSHT._real_capable(NUFSHT.NonuniformFFTsBackend())
    Test.@test !NUFSHT._real_capable(NUFSHT.FINUFFTBackend())
    Test.@test !NUFSHT._real_capable(SpectralBackends.DirectSumSpectralBackend())

    lmax, M, B = 10, 500, 2
    N, Nf = lmax + 1, 2lmax + 1
    θ, φ = iid_points(M, 91)
    C = rand_coeffs(lmax, 92, B)

    # Only a real field on a real-capable backend folds; the other three combinations must not.
    for (backend, FE, folded) in ((NUFSHT.NonuniformFFTsBackend(), Float64,    true),
                                  (NUFSHT.NonuniformFFTsBackend(), ComplexF64, false),
                                  (NUFSHT.FINUFFTBackend(),        Float64,    false),
                                  (SpectralBackends.DirectSumSpectralBackend(), Float64, false))
        p = NUFSHT.make_plan(FE, θ, φ, lmax; tol = 1e-12, ntrans = B, nufft = backend)
        Test.@test size(p.Fhat, 1) == (folded ? lmax + 2 : 2lmax + 3)
        Test.@test eltype(NUFSHT._fbuf(p)) == (folded ? Float64 : ComplexF64)
        NUFSHT.close!(p)
    end

    pf = NUFSHT.make_plan(Float64, θ, φ, lmax; tol = 1e-12, ntrans = B,
                          nufft = NUFSHT.NonuniformFFTsBackend())
    pr = NUFSHT.make_plan(Float64, θ, φ, lmax; tol = 1e-12, ntrans = B,
                          nufft = SpectralBackends.DirectSumSpectralBackend())
    # The fold has to be live for anything below to mean what it says.
    Test.@test size(pf.Fhat, 1) == lmax + 2 && size(pr.Fhat, 1) == 2lmax + 3

    ff = zeros(M, B); NUFSHT.nusht_type2!(ff, C, pf)
    fr = zeros(M, B); NUFSHT.nusht_type2!(fr, C, pr)
    Test.@test relerr(ff, fr) < 1e-11
    for b in 1:B
        Test.@test relerr(ff[:, b], synth_ref(C[:, :, b], lmax, θ, φ)) < 1e-11
    end

    Y = randn(M, B)
    Cf = zeros(N, Nf, B); NUFSHT.nusht_type1!(Cf, Y, pf)
    Cr = zeros(N, Nf, B); NUFSHT.nusht_type1!(Cr, Y, pr)
    Test.@test relerr(Cf, Cr) < 1e-11
    # Without the transpose's factor 2 this lands at ~0.5, not at round-off.
    idf = abs(sum(ff .* Y) - sum(C .* Cf)) / abs(sum(ff .* Y))
    Test.@test idf < 1e-12
    @info "real fast path: folded rows=$(size(pf.Fhat,1)) (full $(size(pr.Fhat,1))), fbuf=$(eltype(NUFSHT._fbuf(pf))), adjoint id=$idf"

    # And the solve the fold exists to speed up must land on the same coefficients.
    fS = zeros(M, B); NUFSHT.nusht_type2!(fS, C, pf)
    Sf = zeros(N, Nf, B); NUFSHT.nusht_solve!(Sf, fS, pf; rtol = 1e-10, maxiter = 200)
    Sr = zeros(N, Nf, B); NUFSHT.nusht_solve!(Sr, fS, pr; rtol = 1e-10, maxiter = 200)
    Test.@test relerr(Sf, C) < 1e-8
    Test.@test relerr(Sf, Sr) < 1e-8
    NUFSHT.close!(pf); NUFSHT.close!(pr)
end

# So that a batched solve can narrow on NonuniformFFTs too, its extension *derives* a reduced-width
# plan's type instead of constructing one — `PlanNUFFT` lifts the transform count into its type via
# `to_static`, so widths are distinct types and a store grown on demand could not name its element type
# in advance. Naming a type builds nothing, which is what keeps the width slots lazy.
#
# That derivation reaches into the library's parameter layout, so it is pinned here against freshly
# built plans: an upstream reordering must fail loudly rather than yield a slot that cannot hold what
# is put into it. `base = 2` is included deliberately — the transform count then equals the
# dimensionality, which is what defeats a rule that matches `NTuple{n,…}` by shape instead of position.
Test.@testset "NonuniformFFTs reduced-width plan types are derived, not built" begin
    ext = Base.get_extension(NUFSHT, :NUFSHTNonuniformFFTsExt)
    Test.@test ext !== nothing
    mk(Z, k) = NonuniformFFTs.PlanNUFFT(Z, (131, 129); ntransforms = k,
                                        m = NonuniformFFTs.HalfSupport(5))
    widths = (1, 2, 3, 4, 8)
    for Z in (Float64, ComplexF64)
        actual = Dict(k => typeof(mk(Z, k)) for k in widths)
        for base in widths, k in widths
            Test.@test ext._plan_at_width(actual[base], k) === actual[k]
        end
    end
    @info "NonuniformFFTs: reduced-width plan types derived exactly over $(2 * length(widths)^2) (strengths, base, target) combinations"
end

# A batched solve retires columns as they converge and then transforms only the live ones. Narrowing
# the FFT/NUFFT needs plans of several widths to share one type, which is a per-backend property —
# FINUFFT keeps the transform count as a runtime field, NonuniformFFTs as a type parameter. Every
# backend must reach the same answer regardless, so this runs the retiring path on each of them; the
# solve tests alone cannot, since they resolve to whichever backend is preferred.
Test.@testset "batched solve retires correctly on every backend" begin
    Random.seed!(20)
    lmax, M, B = 8, 150, 4                       # barely overdetermined ⇒ columns retire apart
    N, Nf = lmax + 1, 2lmax + 1
    θ = acos.(2 .* rand(M) .- 1); φ = 2π .* rand(M)
    F = randn(M, B)

    for backend in (NUFSHT.FINUFFTBackend(), NUFSHT.NonuniformFFTsBackend())
        pB = NUFSHT.make_plan(Float64, θ, φ, lmax; ntrans = B, nufft = backend)
        p1 = NUFSHT.make_plan(Float64, θ, φ, lmax; nufft = backend)

        # Narrowing is a property of the solver, not of which NUFFT library was picked: both back ends
        # must offer it. How the reduced-width plans are stored differs — a `Vector` grown on demand
        # where widths share a type, per-width typed slots where they do not — which is precisely what
        # `_nufft_size_pool` absorbs, and nothing above it should be able to tell.
        Test.@test pB.pool_recipe.narrowable
        # Whatever the shape, no width is built before a solve asks for one.
        Test.@test NUFSHT._pool_built(pB.size_pool) == 0

        CB = zeros(N, Nf, B)
        _, itB, relB, convB = NUFSHT.nusht_solve!(CB, F, pB; rtol = 1e-8, maxiter = 500)
        Test.@test itB < 500                     # did not diverge or stall
        Test.@test convB
        Test.@test relB < 1e-8

        for b in 1:B
            c = zeros(N, Nf)
            NUFSHT.nusht_solve!(c, F[:, b], p1; rtol = 1e-8, maxiter = 500)
            Test.@test maximum(abs, CB[:, :, b] .- c) / maximum(abs, c) < 1e-4
        end

        # The solve must actually have narrowed, or everything below it passes vacuously.
        nbuilt = NUFSHT._pool_built(pB.size_pool)
        @info "$(nameof(typeof(backend))): iters=$itB widths built=$nbuilt of $(length(pB.size_pool))"
        Test.@test nbuilt > 0

        # Those cached plans own NUFFT plans too, and `close!` must destroy them rather than leave
        # them to their finalizers: FINUFFT's destructor re-enters Julia for its FFTW lock, and a
        # contended acquire yields, which a GC finalizer cannot do.
        NUFSHT.close!(pB)
        Test.@test NUFSHT._pool_built(pB.size_pool) == 0
        NUFSHT.close!(pB)                        # idempotent
        NUFSHT.close!(p1)
    end
end
