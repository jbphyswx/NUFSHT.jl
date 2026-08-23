# The reference is pinned first, on the one grid where an external one already exists, and only then
# used where none does. Without this ordering the scattered-point gates below would be checking the
# transform against an unvalidated formula.
Test.@testset "ylm_ref matches sph_evaluate on the CC grid (pins the convention)" begin
    lmax = 8
    N = lmax + 1
    pts = FastSphericalHarmonics.sph_points(N)
    θ = vec([t for t in pts[1], p in pts[2]])
    φ = vec([p for t in pts[1], p in pts[2]])

    # Random coefficients over every mode at once: a convention error in any single (l,m) shows up.
    C = rand_coeffs(lmax, 1)
    Test.@test relerr(synth_ref(C, lmax, θ, φ), vec(FastSphericalHarmonics.sph_evaluate(C))) < 1e-13

    # And mode by mode, so a compensating pair of sign errors cannot hide.
    worst = 0.0
    for l in 0:lmax, m in -l:l
        C1 = zeros(N, 2lmax + 1); C1[FastSphericalHarmonics.sph_mode(l, m)] = 1.0
        r = vec(FastSphericalHarmonics.sph_evaluate(C1))
        worst = max(worst, maximum(abs, synth_ref(C1, lmax, θ, φ) .- r) / maximum(abs, r))
    end
    @info "ylm_ref vs sph_evaluate, worst single mode (lmax=$lmax): $worst"
    Test.@test worst < 1e-13
end

Test.@testset "nusht_type2! accuracy: synthesis vs sph_evaluate at CC grid" begin
    lmax = 20
    N = lmax + 1
    pts = FastSphericalHarmonics.sph_points(N)
    θ_nodes = vec([θ for θ in pts[1], φ in pts[2]])
    φ_nodes = vec([φ for θ in pts[1], φ in pts[2]])

    plan = NUFSHT.make_plan(θ_nodes, φ_nodes, lmax; tol = 1e-12)
    C_true = rand_coeffs(lmax, 123)
    f_exact = vec(FastSphericalHarmonics.sph_evaluate(C_true))
    f_synth = zeros(length(θ_nodes))
    NUFSHT.nusht_type2!(f_synth, C_true, plan)

    err = relerr(f_synth, f_exact)
    @info "nusht_type2! vs sph_evaluate at CC grid (lmax=$lmax): rel err $err"
    Test.@test err < 1e-10
    NUFSHT.close!(plan)
end

# The gate the suite did not have. Synthesis on the CC grid is an identity: whatever mode array the
# pipeline builds, evaluating it back at the sample points returns the samples it was built from, so
# a CC-grid test cannot see an error in the θ/φ representation at all. Only evaluation BETWEEN the
# sample points can, and that is the entire purpose of a non-uniform transform.
Test.@testset "nusht_type2! vs direct summation AT SCATTERED POINTS" begin
    for (name, gen, lmax, M) in (("fibonacci", (M, s) -> fib_points(M), 8, 324),
                                 ("iid", iid_points, 8, 324),
                                 ("iid", iid_points, 12, 676),
                                 ("clustered", clustered_points, 10, 484))
        θ, φ = gen(M, 7)
        plan = NUFSHT.make_plan(θ, φ, lmax; tol = 1e-12, nthreads = 1)
        C = rand_coeffs(lmax, 3)
        f = zeros(M); NUFSHT.nusht_type2!(f, C, plan)
        err = relerr(f, synth_ref(C, lmax, θ, φ))
        @info "nusht_type2! vs direct sum ($name, lmax=$lmax, M=$M): rel err $err"
        Test.@test err < 1e-11
        NUFSHT.close!(plan)
    end
end

# Per mode, not just in aggregate: an error confined to one order or growing with degree averages
# away in a random-coefficient comparison. This is the shape the DFS glide-reflection error had —
# exact at m = 0, growing with l, invisible on the grid.
Test.@testset "nusht_type2! is exact mode by mode at scattered points" begin
    lmax, M = 8, 300
    θ, φ = iid_points(M, 11)
    plan = NUFSHT.make_plan(θ, φ, lmax; tol = 1e-12, nthreads = 1)
    C = zeros(lmax + 1, 2lmax + 1)
    f = zeros(M)
    worst = (0.0, 0, 0)
    for l in 0:lmax, m in -l:l
        fill!(C, 0.0); C[FastSphericalHarmonics.sph_mode(l, m)] = 1.0
        NUFSHT.nusht_type2!(f, C, plan)
        g = [ylm_ref(l, m, θ[i], φ[i]) for i in 1:M]
        e = maximum(abs, f .- g) / maximum(abs, g)
        e > worst[1] && (worst = (e, l, m))
    end
    @info "worst single-mode scattered error (lmax=$lmax, M=$M): $(worst[1]) at (l=$(worst[2]), m=$(worst[3]))"
    Test.@test worst[1] < 1e-11
    NUFSHT.close!(plan)
end

# A fixture's conditioning is a property the tests depend on, so it is asserted rather than assumed.
Test.@testset "point-set fixtures have the conditioning the suite assumes" begin
    lmax = 8
    M = 4 * (lmax + 1)^2
    for (name, gen, lo, hi) in (("fibonacci", (M, s) -> fib_points(M), 1.0, 1.2),
                                ("iid", iid_points, 1.5, 30.0),
                                ("clustered", clustered_points, 10.0, 500.0),
                                ("spiral", spiral_points, 1e4, 1e9))
        θ, φ = gen(M, 2)
        plan = NUFSHT.make_plan(θ, φ, lmax; tol = 1e-12, nthreads = 1)
        κ = cond_A(first(design_matrix(plan, lmax, M)))
        @info "fixture $name (lmax=$lmax, M=$M): cond(A) = $κ"
        Test.@test lo <= κ <= hi
        NUFSHT.close!(plan)
    end
end
