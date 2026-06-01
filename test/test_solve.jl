Test.@testset "nusht_solve!: exact CG inverse at non-CC scattered points" begin
    Random.seed!(42)

    lmax = 10
    N = lmax + 1
    Nφ = 2lmax + 1
    N_modes = N^2  # = (lmax+1)^2 total modes

    # Jittered-from-uniform scattered points on the sphere (M = 4x overdetermined).
    # Per the FINUFFT tutorial, jittered points are well-conditioned; iid random
    # points can have condition number ~900 and require many more CG iterations.
    M = 4 * N_modes
    # Generate near-equidistributed points via latitude-band jitter
    φ_base = (2π / M) .* (0:M-1)
    θ_base = acos.(clamp.(2 .* ((0:M-1) .+ 0.5) ./ M .- 1, -1.0, 1.0))
    θ_nodes = θ_base .+ (rand(M) .- 0.5) .* (0.4 * π / sqrt(M))
    φ_nodes = mod.(φ_base .+ (rand(M) .- 0.5) .* (0.4 * 2π / sqrt(M)), 2π)
    θ_nodes = clamp.(θ_nodes, 1e-10, π - 1e-10)

    plan = NUFSHT.make_plan(θ_nodes, φ_nodes, lmax; tol=1e-10)

    # True band-limited coefficients (only a few modes set)
    C_true = zeros(N, Nφ)
    for ℓ in 1:min(4, lmax), m in -ℓ:ℓ
        C_true[FastSphericalHarmonics.sph_mode(ℓ, m)] = randn()
    end

    # Synthesise exact scattered field values from C_true
    f_true = zeros(M)
    NUFSHT.nusht_type2!(f_true, C_true, plan)

    # Solve for coefficients via CG
    C_solved = similar(plan.C)
    _, iters, rel_res = NUFSHT.nusht_solve!(C_solved, f_true, plan; rtol=1e-6, maxiter=1000)

    # Recovered field should match f_true to NUFFT tolerance
    f_recovered = zeros(M)
    NUFSHT.nusht_type2!(f_recovered, C_solved, plan)
    rms_f = sqrt(Statistics.mean(abs2.(f_true)) + 1e-30)
    rms_err = sqrt(Statistics.mean(abs2.(f_recovered .- f_true)))
    @info "nusht_solve! (lmax=$lmax, M=$M): iters=$iters rel_res=$(round(rel_res,sigdigits=3)) field_rms_err/rms_f=$(round(rms_err/rms_f,sigdigits=3))"
    Test.@test rms_err < 1e-3 * rms_f
end

Test.@testset "nusht_type2! accuracy: synthesis from true coefficients" begin
    Random.seed!(123)

    lmax = 20
    N = lmax + 1
    Nφ = 2lmax + 1

    pts = FastSphericalHarmonics.sph_points(N)
    θ_nodes = vec([θ for θ in pts[1], φ in pts[2]])
    φ_nodes = vec([φ for θ in pts[1], φ in pts[2]])

    plan = NUFSHT.make_plan(θ_nodes, φ_nodes, lmax; tol=1e-10)

    C_true = zeros(N, Nφ)
    for ℓ in 0:min(5, lmax), m in -ℓ:ℓ
        C_true[FastSphericalHarmonics.sph_mode(ℓ, m)] = randn()
    end

    # Exact field on CC grid via sph_evaluate
    f_exact = vec(FastSphericalHarmonics.sph_evaluate(C_true))

    # type2 synthesis at same CC grid points
    f_synth = zeros(length(θ_nodes))
    NUFSHT.nusht_type2!(f_synth, C_true, plan)

    rms_err = sqrt(Statistics.mean(abs2.(f_synth .- f_exact)))
    @info "nusht_type2! vs sph_evaluate at CC grid (lmax=$lmax): rms_err=$rms_err"
    Test.@test rms_err < 1e-9 * sqrt(Statistics.mean(abs2.(f_exact)))
end
