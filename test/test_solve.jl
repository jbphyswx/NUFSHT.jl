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

# The coefficient array is a square, invertible representation: `(lmax+1)(2lmax+1)` slots carrying
# every degree `l ≤ lmax` plus supernumerary ones with `lmax < l ≤ lmax+|m|`. The transform needs all
# of them. A least-squares fit does not — it must name its space, and only `l ≤ lmax` is
# SO(3)-invariant (the ragged set holds fragments of `H_l`, never whole irreps, so it is not closed
# under rotation). Fitting the ragged set gives a frame-dependent answer and, because it has
# `lmax(lmax+1)` extra parameters, a residual *below* the true harmonic one.
Test.@testset "nusht_solve! fits the l ≤ lmax subspace" begin
    Random.seed!(6)
    lmax, M = 8, 500
    θ = acos.(2 .* rand(M) .- 1); φ = 2π .* rand(M); v = randn(M)
    plan = NUFSHT.make_plan(Float64, θ, φ, lmax)

    valid = falses(lmax + 1, 2lmax + 1)
    for l in 0:lmax, m in -l:l
        valid[FastSphericalHarmonics.sph_mode(l, m)] = true
    end
    Test.@test sum(valid) == (lmax + 1)^2                    # 81 of 153

    # The supernumerary slots are NOT a null space — they carry real degrees and synthesise.
    Cx = zeros(lmax + 1, 2lmax + 1); Cx[.!valid] .= randn(sum(.!valid))
    fx = zeros(M); NUFSHT.nusht_type2!(fx, Cx, plan)
    Test.@test sqrt(sum(abs2, fx)) > 1                       # documents why masking must be in the solve

    Cs = zeros(lmax + 1, 2lmax + 1)
    NUFSHT.nusht_solve!(Cs, v, plan; rtol = 1e-12, maxiter = 800)

    # T2 — the answer is supported on l ≤ lmax, exactly.
    Test.@test sum(abs2, Cs[.!valid]) == 0

    # T3 — against a dense QR least-squares reference over those modes. Independent of the solver.
    idx = findall(valid)
    A = zeros(M, length(idx))
    for (k, I) in enumerate(idx)
        C1 = zeros(lmax + 1, 2lmax + 1); C1[I] = 1.0
        f1 = zeros(M); NUFSHT.nusht_type2!(f1, C1, plan); A[:, k] = f1
    end
    cqr = A \ v
    Test.@test sqrt(sum(abs2, Cs[idx] .- cqr)) / sqrt(sum(abs2, cqr)) < 1e-6

    # T5 — a residual BELOW the harmonic least-squares one is the over-parameterisation signature.
    fs = zeros(M); NUFSHT.nusht_type2!(fs, Cs, plan)
    res_solve = sqrt(sum(abs2, fs .- v)); res_qr = sqrt(sum(abs2, A * cqr .- v))
    Test.@test res_solve ≥ res_qr * (1 - 1e-8)
    Test.@test res_solve ≤ res_qr * (1 + 1e-6)
    @info "solve residual $res_solve vs QR $res_qr; invalid-slot energy $(sum(abs2, Cs[.!valid]))"
    NUFSHT.close!(plan)
end

# T1 — SO(3) equivariance. Degree power spectra are rotation invariants, so solving the same values at
# a rotated point set must reproduce them. This is the property the ragged mode set violates.
Test.@testset "nusht_solve! is rotation equivariant" begin
    Random.seed!(6)
    lmax, M = 8, 500
    θ0 = acos.(2 .* rand(M) .- 1); φ0 = 2π .* rand(M); v = randn(M)
    a, b = 0.7, 1.1
    R = [cos(b) -sin(b) 0; sin(b) cos(b) 0; 0 0 1] * [cos(a) 0 sin(a); 0 1 0; -sin(a) 0 cos(a)]
    P = R * permutedims(hcat(sin.(θ0) .* cos.(φ0), sin.(θ0) .* sin.(φ0), cos.(θ0)))
    θ1 = acos.(clamp.(P[3, :], -1, 1)); φ1 = mod.(atan.(P[2, :], P[1, :]), 2π)

    function spectrum(θ, φ)
        p = NUFSHT.make_plan(Float64, θ, φ, lmax)
        C = zeros(lmax + 1, 2lmax + 1)
        NUFSHT.nusht_solve!(C, v, p; rtol = 1e-12, maxiter = 800)
        NUFSHT.close!(p)
        return [sum(C[FastSphericalHarmonics.sph_mode(l, m)]^2 for m in -l:l) for l in 0:lmax]
    end
    s0, s1 = spectrum(θ0, φ0), spectrum(θ1, φ1)
    rel = sqrt(sum(abs2, s0 .- s1)) / sqrt(sum(abs2, s0))
    @info "degree-spectrum change under rotation: $rel"
    Test.@test rel < 0.02
end
