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
    Auto = SpectralBackends.AutoSpectralBackend()
    # The field element type is part of the choice: a real field prefers a backend with a real-data
    # transform, since its mode array is Hermitian in kθ and its non-uniform data is real.
    Test.@test NUFSHT._resolve_nufft(Auto, Float64) isa NUFSHT.NonuniformFFTsBackend
    Test.@test NUFSHT._resolve_nufft(Auto, Float32) isa NUFSHT.NonuniformFFTsBackend
    Test.@test NUFSHT._resolve_nufft(Auto, ComplexF64) isa NUFSHT.FINUFFTBackend
    # An explicitly named backend is honoured for either element type, never swapped.
    for FE in (Float64, ComplexF64)
        Test.@test NUFSHT._resolve_nufft(NUFSHT.NonuniformFFTsBackend(), FE) isa NUFSHT.NonuniformFFTsBackend
        Test.@test NUFSHT._resolve_nufft(NUFSHT.FINUFFTBackend(), FE) isa NUFSHT.FINUFFTBackend
        Test.@test NUFSHT._resolve_nufft(SpectralBackends.DirectSumSpectralBackend(), FE) isa
                   SpectralBackends.DirectSumSpectralBackend
    end
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

    # Every real field folds its mode array, on every backend — halving the θ axis halves the
    # deconvolution and the FFT and leaves the interpolation alone, so it is never more work. Real
    # *strengths* are the separate question, and only a real-capable backend gets them.
    for (backend, FE, folded, ZS) in
            ((NUFSHT.NonuniformFFTsBackend(), Float64,    true,  Float64),
             (NUFSHT.NonuniformFFTsBackend(), ComplexF64, false, ComplexF64),
             (NUFSHT.FINUFFTBackend(),        Float64,    true,  ComplexF64),
             (NUFSHT.FINUFFTBackend(),        ComplexF64, false, ComplexF64),
             (SpectralBackends.DirectSumSpectralBackend(), Float64,    true,  ComplexF64),
             (SpectralBackends.DirectSumSpectralBackend(), ComplexF64, false, ComplexF64))
        p = NUFSHT.make_plan(FE, θ, φ, lmax; tol = 1e-12, ntrans = B, nufft = backend)
        Test.@test size(p.Fhat, 1) == (folded ? lmax + 2 : 2lmax + 3)
        Test.@test eltype(NUFSHT._fbuf(p)) == ZS
        # A half-height *complex* transform is centered, so its rows are offset and a per-point phase
        # undoes it; a real transform's half-spectrum is not, and carries none.
        Test.@test (NUFSHT._θshift(p) !== nothing) == (folded && ZS === ComplexF64)
        NUFSHT.close!(p)
    end

    pf = NUFSHT.make_plan(Float64, θ, φ, lmax; tol = 1e-12, ntrans = B,
                          nufft = NUFSHT.NonuniformFFTsBackend())
    # The unfolded reference is a *complex* field on the dependency-free backend — the same operator
    # with no Hermitian shortcut anywhere in it.
    pr = NUFSHT.make_plan(ComplexF64, θ, φ, lmax; tol = 1e-12, ntrans = B,
                          nufft = SpectralBackends.DirectSumSpectralBackend())
    # The fold has to be live for anything below to mean what it says.
    Test.@test size(pf.Fhat, 1) == lmax + 2 && size(pr.Fhat, 1) == 2lmax + 3

    ff = zeros(M, B);             NUFSHT.nusht_type2!(ff, C, pf)
    fr = zeros(ComplexF64, M, B); NUFSHT.nusht_type2!(fr, ComplexF64.(C), pr)
    Test.@test relerr(ff, real.(fr)) < 1e-11
    for b in 1:B
        Test.@test relerr(ff[:, b], synth_ref(C[:, :, b], lmax, θ, φ)) < 1e-11
    end

    Y = randn(M, B)
    Cf = zeros(N, Nf, B);             NUFSHT.nusht_type1!(Cf, Y, pf)
    Cr = zeros(ComplexF64, N, Nf, B); NUFSHT.nusht_type1!(Cr, ComplexF64.(Y), pr)
    Test.@test relerr(Cf, real.(Cr)) < 1e-11
    # Without the transpose's factor 2 this lands at ~0.5, not at round-off.
    idf = abs(sum(ff .* Y) - sum(C .* Cf)) / abs(sum(ff .* Y))
    Test.@test idf < 1e-12
    @info "real fast path: folded rows=$(size(pf.Fhat,1)) (full $(size(pr.Fhat,1))), fbuf=$(eltype(NUFSHT._fbuf(pf))), adjoint id=$idf"

    # And the solve the fold exists to speed up must land on the same coefficients.
    fS = zeros(M, B); NUFSHT.nusht_type2!(fS, C, pf)
    Sf = zeros(N, Nf, B); NUFSHT.nusht_solve!(Sf, fS, pf; rtol = 1e-10, maxiter = 200)
    Sr = zeros(ComplexF64, N, Nf, B)
    NUFSHT.nusht_solve!(Sr, ComplexF64.(fS), pr; rtol = 1e-10, maxiter = 200)
    Test.@test relerr(Sf, C) < 1e-8
    Test.@test relerr(Sf, real.(Sr)) < 1e-8
    NUFSHT.close!(pf); NUFSHT.close!(pr)
end

# Falling back to direct summation is a decision the caller did not make, so `Auto` landing there says
# so — once per session, not per call. Asking for it by name is a decision they *did* make and stays
# silent forever. `maxlog = 1` caps a log statement per logger instance, and `@test_logs` installs a
# fresh one, so this sees the warning without any hand-rolled guard.
Test.@testset "direct summation announces itself only when Auto chose it" begin
    Auto = SpectralBackends.AutoSpectralBackend()
    ds = SpectralBackends.DirectSumSpectralBackend()
    Test.@test_logs (:warn,) NUFSHT._warn_if_directsum(Auto, ds, 484, 483)
    Test.@test_logs NUFSHT._warn_if_directsum(ds, ds, 484, 483)               # named: silent
    Test.@test_logs NUFSHT._warn_if_directsum(Auto, NUFSHT.FINUFFTBackend(), 484, 483)
end

# `plan_memory` has to account for what the plan actually holds, including the parts that are absent:
# an empty slot counts zero, which is the whole point of reporting it.
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
    Test.@test p.C[] === nothing                      # synthesis still allocates none
    out = zeros(M); ws = NUFSHT.LSMRWorkspace(p)
    NUFSHT.nusht_filter!(out, f, NUFSHT.gaussian_from_scale(2000e3), p; ws = ws)
    Test.@test p.C[] !== nothing                      # filtering fills it once
    Test.@test NUFSHT.plan_memory(p).C > 0
    Test.@test NUFSHT.plan_memory(p).total > pm.total
    NUFSHT.close!(p)
end

# `directions` says which calls a plan answers. It has to mean the same thing on every backend: on one
# where a single plan object serves both directions the analysis handle would cost nothing, and keeping
# it there would make `nusht_type1!` throw on FINUFFT and quietly succeed on NonuniformFFTs for the
# same `SynthesisOnly()` plan.
Test.@testset "plan directions are honoured on every backend" begin
    Random.seed!(61)
    lmax, M = 8, 324
    N, Nf = lmax + 1, 2lmax + 1
    θ, φ = iid_points(M, 62)
    C = rand_coeffs(lmax, 63)
    ref = synth_ref(C, lmax, θ, φ)

    for backend in (NUFSHT.FINUFFTBackend(), NUFSHT.NonuniformFFTsBackend(),
                    SpectralBackends.DirectSumSpectralBackend())
        so = NUFSHT.make_plan(Float64, θ, φ, lmax; tol = 1e-12, nufft = backend,
                              directions = NUFSHT.SynthesisOnly())
        Test.@test so.nodes.nufft_type1 === nothing
        f = zeros(M); NUFSHT.nusht_type2!(f, C, so)          # synthesis still works
        Test.@test relerr(f, ref) < 1e-11
        Test.@test_throws ArgumentError NUFSHT.nusht_type1!(zeros(N, Nf), f, so)
        Test.@test_throws ArgumentError NUFSHT.nusht_solve!(zeros(N, Nf), f, so; maxiter = 5)
        Test.@test NUFSHT.plan_memory(so).total > 0
        NUFSHT.close!(so)

        both = NUFSHT.make_plan(Float64, θ, φ, lmax; tol = 1e-12, nufft = backend)
        Test.@test both.nodes.nufft_type1 !== nothing
        g = zeros(N, Nf); NUFSHT.nusht_type1!(g, f, both)     # and analysis does on the default
        Test.@test all(isfinite, g)
        # Where one plan serves both directions the second handle is derived, not a second plan. `===`
        # is not usable here — a `PlanNUFFT` compares unequal to itself — so the flag recorded at
        # construction is what carries it.
        Test.@test NUFSHT._nufft_derived(both.nodes.nufft_type1) ==
                   NUFSHT._nufft_share_directions(backend)
        NUFSHT.close!(both)
    end
end

# `nthreads` is a request, so a backend either meets it or refuses it — it is never dropped on the
# floor. What each can reach differs: FINUFFT takes any count, direct summation splits the axis each
# direction writes, and NonuniformFFTs parallelises over Julia's own threads and so reaches exactly
# two counts. Splitting a transform must not change its value, which is what lets a solve be compared
# against another run of itself.
Test.@testset "NUFFT backends honour or refuse nthreads" begin
    Random.seed!(51)
    lmax, M, B = 8, 324, 2
    N, Nf = lmax + 1, 2lmax + 1
    n1, n2 = 2lmax + 3, Nf
    θ = clamp.(π .* rand(M), 1e-9, π - 1e-9); φ = 2π .* rand(M)
    ds = SpectralBackends.DirectSumSpectralBackend()

    # Direct summation: the split is over the axis each direction writes, so every thread count gives
    # bit-identical output. Compared against `nthreads = 1`, which spawns nothing at all.
    modes = randn(ComplexF64, n1, n2, B); vals = randn(ComplexF64, M, B)
    ref = Dict{Int,Array{ComplexF64}}()
    for nt in (1, 0, Threads.nthreads()), ty in (2, 1)
        p = NUFSHT._nufft_makeplan(ds, θ, ty, [n1, n2], ty == 2 ? +1 : -1, B, 1e-12;
                                   dtype = Float64, modeord = 0, nthreads = nt)
        NUFSHT._nufft_setpts!(p, θ, φ)
        Test.@test p.nthreads == (nt == 0 ? Threads.nthreads() : nt)
        out = ty == 2 ? zeros(ComplexF64, M, B) : zeros(ComplexF64, n1, n2, B)
        NUFSHT._nufft_exec!(p, ty == 2 ? modes : vals, out)
        if nt == 1
            ref[ty] = out
        else
            Test.@test out == ref[ty]
        end
    end
    Test.@test_throws ArgumentError NUFSHT._nufft_makeplan(ds, θ, 2, [n1, n2], +1, B, 1e-12;
                                                           dtype = Float64, nthreads = -1)

    # NonuniformFFTs: `1` selects the unblocked (serial) spreading path, the running count selects the
    # blocked one, anything else is an error rather than a silent 8 threads.
    Test.@test_throws ArgumentError NUFSHT.make_plan(Float64, θ, φ, lmax;
        nthreads = Threads.nthreads() + 1, nufft = NUFSHT.NonuniformFFTsBackend())
    C = rand_coeffs(lmax, 52)
    for nt in (1, Threads.nthreads())
        p = NUFSHT.make_plan(Float64, θ, φ, lmax; tol = 1e-12, nthreads = nt,
                             nufft = NUFSHT.NonuniformFFTsBackend())
        f1 = zeros(M); f2 = zeros(M)
        NUFSHT.nusht_type2!(f1, C, p); NUFSHT.nusht_type2!(f2, C, p)
        g1 = zeros(N, Nf); g2 = zeros(N, Nf)
        NUFSHT.nusht_type1!(g1, f1, p); NUFSHT.nusht_type1!(g2, f1, p)
        Test.@test f1 == f2 && g1 == g2                # a plan executed twice returns the same bits
        Test.@test relerr(f1, synth_ref(C, lmax, θ, φ)) < 1e-11
        NUFSHT.close!(p)
    end
    # A count of 1 must actually reach the serial path, or the reproducibility above is not what makes
    # the test pass. `BlockDataCPU` is the blocked layout and sits at the same type position either way.
    unblocked(nt) = nameof(typeof(NUFSHT.make_plan(Float64, θ, φ, lmax; nthreads = nt,
        nufft = NUFSHT.NonuniformFFTsBackend()).nodes.nufft_type2.plan).parameters[11]) ===
        :NullBlockData
    Test.@test unblocked(1) == (Threads.nthreads() > 1)
    Test.@test !unblocked(Threads.nthreads())

    # FINUFFT takes an arbitrary count, so nothing there is refused.
    pf = NUFSHT.make_plan(Float64, θ, φ, lmax; tol = 1e-12, nthreads = 2,
                          nufft = NUFSHT.FINUFFTBackend())
    ff = zeros(M); NUFSHT.nusht_type2!(ff, C, pf)
    Test.@test relerr(ff, synth_ref(C, lmax, θ, φ)) < 1e-11
    NUFSHT.close!(pf)
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
    # The unblocked layout is included because `nthreads = 1` selects it and it carries no transform
    # count of its own — a rule that rewrote parameter 3 unconditionally would fail on it.
    mk0(Z, k) = NonuniformFFTs.PlanNUFFT(Z, (131, 129); ntransforms = k,
                                         m = NonuniformFFTs.HalfSupport(5), block_size = nothing)
    widths = (1, 2, 3, 4, 8)
    for Z in (Float64, ComplexF64), build in (mk, mk0)
        actual = Dict(k => typeof(build(Z, k)) for k in widths)
        for base in widths, k in widths
            Test.@test ext._plan_at_width(actual[base], k) === actual[k]
        end
    end
    @info "NonuniformFFTs: reduced-width plan types derived exactly over $(4 * length(widths)^2) (strengths, blocking, base, target) combinations"
end

# A batched solve retires columns as they converge and then transforms only the live ones. Narrowing
# the FFT/NUFFT needs plans of several widths to share one type, which is a per-backend property —
# FINUFFT keeps the transform count as a runtime field, NonuniformFFTs as a type parameter. Every
# backend must reach the same answer regardless, so this runs the retiring path on each of them; the
# solve tests alone cannot, since they resolve to whichever backend is preferred.
Test.@testset "batched solve retires correctly on every backend" begin
    Random.seed!(20)
    lmax, M, B = 8, 150, 4
    N, Nf = lmax + 1, 2lmax + 1
    θ = acos.(2 .* rand(M) .- 1); φ = 2π .* rand(M)
    # Column `b` is a field band-limited to degree `b`, so it lives in a subspace of dimension
    # `(b+1)²` and needs correspondingly fewer iterations. The columns therefore retire in order by
    # construction rather than by whatever a random right-hand side happens to do — which is what the
    # narrowing assertion below needs to be non-vacuous.
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

        # Whether a *solve* narrows depends on the columns' residuals crossing `rtol` at different
        # iterations, which is emergent numerics and not something to assert on — with a shared
        # operator they usually cross together. So the narrowed path is driven directly: transform at
        # width `k < B` and require both that a plan was built for it and that it agrees with the
        # full-width transform on those columns.
        k = 2
        Ck = zeros(N, Nf, B)
        for b in 1:B, ℓ in 0:3, m in -ℓ:ℓ
            Ck[FastSphericalHarmonics.sph_mode(ℓ, m), b] = randn()
        end
        ffull = zeros(M, B); NUFSHT.nusht_type2!(ffull, Ck, pB)
        Test.@test NUFSHT._pool_built(pB.size_pool) == 0        # full width never touches the pool
        fnarrow = zeros(M, B); NUFSHT.nusht_type2!(fnarrow, Ck, pB, k, k)
        nbuilt = NUFSHT._pool_built(pB.size_pool)
        @info "$(nameof(typeof(backend))): iters=$itB, width $k built=$nbuilt"
        Test.@test nbuilt > 0
        Test.@test maximum(abs, fnarrow[:, 1:k] .- ffull[:, 1:k]) /
                   maximum(abs, ffull[:, 1:k]) < 1e-12

        # Those cached plans own NUFFT plans too, and `close!` must destroy them rather than leave
        # them to their finalizers: FINUFFT's destructor re-enters Julia for its FFTW lock, and a
        # contended acquire yields, which a GC finalizer cannot do.
        NUFSHT.close!(pB)
        Test.@test NUFSHT._pool_built(pB.size_pool) == 0
        NUFSHT.close!(pB)                        # idempotent
        NUFSHT.close!(p1)
    end
end
