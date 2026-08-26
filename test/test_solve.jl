# A field check alone cannot tell a good solve from a bad one: on a near-degenerate point set the
# field is reproduced while the coefficients are meaningless. So the fixture's conditioning is
# asserted, and the coefficients themselves are what is checked.
Test.@testset "nusht_solve!: exact inverse at non-CC scattered points" begin
    lmax = 10
    M = 4 * (lmax + 1)^2
    θ_nodes, φ_nodes = fib_points(M)
    plan = NUFSHT.make_plan(θ_nodes, φ_nodes, lmax; tol = 1e-12, nthreads = 1)
    Test.@test cond_A(first(design_matrix(plan, lmax, M))) < 1.2   # the fixture really is benign

    C_true = rand_coeffs(lmax, 42)
    f_true = zeros(M); NUFSHT.nusht_type2!(f_true, C_true, plan)

    C_solved = similar(plan.F)
    _, iters, rel_res, converged = NUFSHT.nusht_solve!(C_solved, f_true, plan; rtol = 1e-10, maxiter = 200)
    Test.@test converged
    Test.@test converged == (rel_res < 1e-10)
    Test.@test iters < 30                                          # cond(A) ≈ 1 ⇒ a few iterations

    f_recovered = zeros(M); NUFSHT.nusht_type2!(f_recovered, C_solved, plan)
    @info "nusht_solve! (lmax=$lmax, M=$M): iters=$iters rel_res=$(round(rel_res, sigdigits = 3)) " *
          "coeff=$(relerr(C_solved, C_true)) field=$(relerr(f_recovered, f_true))"
    Test.@test relerr(C_solved, C_true) < 1e-8
    Test.@test relerr(f_recovered, f_true) < 1e-9
    NUFSHT.close!(plan)
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

# The batched solve retires each column at its own tolerance, dropping it from the live prefix and
# packing the survivors forward so the transforms narrow. The columns are independent, so the batched
# answer must equal solving them one at a time. Iteration counts are NOT comparable across runs —
# multithreaded FINUFFT spreading is not bit-deterministic and the iteration amplifies it — so this
# asserts the mechanism's invariants, not a count.
Test.@testset "batched solve retires columns independently" begin
    Random.seed!(11)
    lmax, M, B = 12, 676, 4
    N, Nf = lmax + 1, 2lmax + 1
    θ = acos.(2 .* rand(M) .- 1); φ = 2π .* rand(M)

    # Heterogeneous columns: differing bandlimit and amplitude ⇒ differing convergence rates.
    Ct = zeros(N, Nf, B)
    for (b, (L, a)) in enumerate(((2, 1.0), (5, 1e2), (9, 1e-2), (12, 3.0)))
        for l in 1:L, m in -l:l
            Ct[FastSphericalHarmonics.sph_mode(l, m), b] = a * randn()
        end
    end

    pB = NUFSHT.make_plan(Float64, θ, φ, lmax; ntrans = B)
    fB = zeros(M, B); NUFSHT.nusht_type2!(fB, Ct, pB)
    ws = NUFSHT.LSMRWorkspace(pB)
    CB = zeros(N, Nf, B)
    _, _, relB, convB = NUFSHT.nusht_solve!(CB, fB, pB; ws = ws, rtol = 1e-6, maxiter = 500)
    Test.@test convB

    p1 = NUFSHT.make_plan(Float64, θ, φ, lmax)
    C1 = zeros(N, Nf, B)
    for b in 1:B
        c = zeros(N, Nf)
        NUFSHT.nusht_solve!(c, fB[:, b], p1; rtol = 1e-6, maxiter = 500)
        C1[:, :, b] = c
    end

    Test.@test sqrt(sum(abs2, CB .- C1)) / sqrt(sum(abs2, C1)) < 1e-9
    Test.@test relB < 1e-6

    # `perm` maps slot -> original column. Compaction permutes it, and a column's best iterate is
    # written out through it, so it must remain a permutation of 1:B — if it ever repeats an entry, two
    # columns share a destination and one result is silently lost.
    Test.@test sort(collect(ws.perm)) == collect(1:B)
    # `colres` is per column, in the caller's order — the scalar return is its worst entry.
    Test.@test relB == maximum(ws.colres)
    Test.@test all(<(1e-6), ws.colres)
    NUFSHT.close!(pB); NUFSHT.close!(p1)
end

# Column identity. Norm-based assertions cannot see a permutation: if column b's answer is written to
# slot b', every magnitude still looks right. Each column here carries a distinct signature mode and is
# checked to come back in its own slot, with the other columns' signatures absent from it.
Test.@testset "batched solve keeps columns in their own slots" begin
    Random.seed!(99)
    lmax, M, B = 10, 900, 5
    N, Nf = lmax + 1, 2lmax + 1
    θ = acos.(2 .* rand(M) .- 1); φ = 2π .* rand(M)

    # Column b is a single harmonic of degree `deg[b]` — a signature no other column carries. Distinct
    # degrees also make the columns converge at different rates, so retirement is exercised.
    deg = (2, 4, 6, 8, 10)
    sig = [FastSphericalHarmonics.sph_mode(deg[b], deg[b] - 1) for b in 1:B]
    Ct = zeros(N, Nf, B)
    for b in 1:B
        Ct[sig[b], b] = 1.0 + b            # distinct amplitude too, so a swap is unambiguous
    end

    p = NUFSHT.make_plan(Float64, θ, φ, lmax; ntrans = B)
    f = zeros(M, B); NUFSHT.nusht_type2!(f, Ct, p)
    C = zeros(N, Nf, B)
    NUFSHT.nusht_solve!(C, f, p; rtol = 1e-10, maxiter = 600)

    for b in 1:B
        own = C[sig[b], b]
        Test.@test isapprox(own, 1.0 + b; rtol = 1e-6)          # its own signature, right amplitude
        for b2 in 1:B
            b2 == b && continue
            # No other column's signature may appear in slot b, at its amplitude or any other.
            Test.@test abs(C[sig[b2], b]) < 1e-6 * (1.0 + b2)
        end
    end
    @info "column identity held for $B columns with distinct signature degrees $deg"
    NUFSHT.close!(p)
end

# Batching must not change the answer at ANY tolerance. A retired column leaves the live prefix and the
# survivors are packed forward; every piece of per-slot state has to move with its column, or a
# survivor resumes from another column's residual and search direction. Sweeping rtol matters: at some
# tolerances the loop exits before a mismatched column can grow, so a single tolerance can pass while
# the mechanism is broken.
Test.@testset "batched solve matches single-column at every tolerance" begin
    Random.seed!(7)
    lmax, M, B = 8, 150, 4                       # M/modes = 1.85, barely overdetermined
    N, Nf = lmax + 1, 2lmax + 1
    θ, φ = iid_points(M, 7)
    F = randn(M, B)

    pB = NUFSHT.make_plan(Float64, θ, φ, lmax; ntrans = B, nthreads = 1)
    p1 = NUFSHT.make_plan(Float64, θ, φ, lmax; nthreads = 1)
    # The two runs use different FINUFFT `ntrans`, so they are not bit-identical, and a solve
    # propagates that difference by `cond(A)`: two iterates both at residual `rtol` can differ in the
    # coefficients by `~rtol·cond(A)`. Bound the comparison by what the conditioning actually allows
    # rather than by a fixed constant.
    κ = cond_A(first(design_matrix(p1, lmax, M)))
    @info "batched-vs-single fixture: cond(A) = $κ"
    for rtol in (1e-4, 1e-6, 1e-8)
        tol = 20 * rtol * κ
        CB = zeros(N, Nf, B)
        _, itB, relB, convB = NUFSHT.nusht_solve!(CB, F, pB; rtol = rtol, maxiter = 500)
        Test.@test itB < 500                     # did not run away to maxiter
        Test.@test convB
        Test.@test relB < rtol
        for b in 1:B
            c = zeros(N, Nf)
            NUFSHT.nusht_solve!(c, F[:, b], p1; rtol = rtol, maxiter = 500)
            Test.@test maximum(abs, CB[:, :, b] .- c) / maximum(abs, c) < tol
        end
    end
    NUFSHT.close!(pB); NUFSHT.close!(p1)
end

# On a point set that does not determine the coefficients (`M` below `(lmax+1)^2`) `A†A` is singular and
# CG is semiconvergent: the residual falls to a minimum, then rounding in the near-null directions
# amplifies without bound. So the delivered coefficients must be the least-residual iterate, not the
# last one, and the run must stop once the curvature setting the step length is no longer resolvable —
# otherwise a more generous `maxiter` returns a worse answer than a smaller one.
Test.@testset "nusht_solve! does not degrade with a larger iteration budget" begin
    for T in (Float64, Float32)
        M, lmax = 60, 12
        Test.@test M < (lmax + 1)^2                       # rank deficient by construction
        Random.seed!(4)
        θ = T.(acos.(2 .* rand(M) .- 1)); φ = T.(rand(M) .* 2π)
        f = T[abs(sin(3θ[k])) + T(0.1) for k in 1:M]

        # `nthreads = 1` so the runs are bit-reproducible: a longer run then shares its whole
        # prefix with a shorter one, and the two can be compared exactly rather than approximately.
        res = map((50, 200, 1000)) do mi
            p = NUFSHT.make_plan(T, θ, φ, lmax; ntrans = 1, nthreads = 1)
            C = zeros(T, lmax + 1, 2lmax + 1)
            _, iters, rel, conv = NUFSHT.nusht_solve!(C, f, p; rtol = T(1e-12), maxiter = mi)
            fr = zeros(T, M); NUFSHT.nusht_type2!(fr, C, p)
            NUFSHT.close!(p)
            (iters = iters, rel = rel, conv = conv, C = C,
             fieldres = sqrt(sum(abs2, fr .- f) / sum(abs2, f)))
        end

        for r in res
            Test.@test all(isfinite, r.C)
            Test.@test r.rel ≤ 1                          # never worse than the x₀ = 0 iterate
            Test.@test r.fieldres < 1                     # and the coefficients do fit the data
            Test.@test r.conv == (r.rel < T(1e-12))
        end
        # A longer run sees a superset of the iterates a shorter one did, so what it delivers cannot be
        # worse — and once the stopping rule has fired, the extra budget changes nothing at all.
        Test.@test res[2].rel ≤ res[1].rel
        Test.@test res[3].rel ≤ res[2].rel
        Test.@test res[2].iters == res[3].iters
        Test.@test res[2].C == res[3].C
        Test.@test res[3].iters < 1000                    # stopped on its own, not on the budget
        @info "semiconvergence ($T): iters=$(getfield.(res, :iters)) rel_res=$(getfield.(res, :rel))"
    end
end

# `rel_res` describes the coefficients handed back and `converged` says whether they met `rtol`, so a
# caller cannot mistake a stalled solve for a solved one without ignoring an explicit flag.
Test.@testset "nusht_solve! reports whether it converged" begin
    Random.seed!(31)
    lmax = 8; M = 4 * (lmax + 1)^2
    θ = acos.(2 .* rand(M) .- 1); φ = 2π .* rand(M)
    p = NUFSHT.make_plan(Float64, θ, φ, lmax; nthreads = 1)
    C = zeros(lmax + 1, 2lmax + 1)

    _, _, rel_ok, conv_ok = NUFSHT.nusht_solve!(C, randn(M), p; rtol = 1e-6, maxiter = 500)
    Test.@test conv_ok
    Test.@test rel_ok < 1e-6

    # Same well-conditioned points, a tolerance no Float64 CG can reach.
    _, it_no, rel_no, conv_no = NUFSHT.nusht_solve!(C, randn(M), p; rtol = 1e-30, maxiter = 200)
    Test.@test !conv_no
    Test.@test conv_no == (rel_no < 1e-30)
    Test.@test it_no ≤ 200
    NUFSHT.close!(p)
end

# Curvature retirement through the compaction path: a stalled column has to leave the live set with its
# own best iterate already in its own column, and the slot bookkeeping has to survive it. Batched
# against single-column results is NOT comparable here — the widths differ, so FINUFFT rounds
# differently, and a semiconvergent system amplifies that — so this asserts the invariants.
Test.@testset "batched solve retires a stalled column cleanly" begin
    Random.seed!(17)
    lmax, M, B = 12, 60, 3
    N, Nf = lmax + 1, 2lmax + 1
    θ = acos.(2 .* rand(M) .- 1); φ = 2π .* rand(M)
    F = randn(M, B)
    pB = NUFSHT.make_plan(Float64, θ, φ, lmax; ntrans = B, nthreads = 1)
    ws = NUFSHT.LSMRWorkspace(pB)
    CB = zeros(N, Nf, B)
    _, itB, relB, _ = NUFSHT.nusht_solve!(CB, F, pB; ws = ws, rtol = 1e-14, maxiter = 400)

    Test.@test all(isfinite, CB)
    Test.@test itB < 400                                  # every column stopped on its own
    Test.@test sort(collect(ws.perm)) == collect(1:B)
    Test.@test relB == maximum(ws.colres)
    Test.@test all(<(1), ws.colres)                       # every column beat the x₀ = 0 iterate
    p1 = NUFSHT.make_plan(Float64, θ, φ, lmax; nthreads = 1)
    for b in 1:B
        fr = zeros(M); NUFSHT.nusht_type2!(fr, CB[:, :, b], p1)
        Test.@test sqrt(sum(abs2, fr .- F[:, b]) / sum(abs2, F[:, b])) < 1
    end
    NUFSHT.close!(pB); NUFSHT.close!(p1)
end
