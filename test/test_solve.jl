# Coefficient recovery needs a well-conditioned fixture: on a near-degenerate one the field is reproduced
# while the coefficients are not.
Test.@testset "nusht_solve!: exact inverse at scattered points" begin
    lmax = 10
    M = 4 * (lmax + 1)^2
    θ_nodes, φ_nodes = fib_points(M)
    plan = NUFSHT.make_plan(θ_nodes, φ_nodes, lmax; tol = 1e-12, nthreads = 1)
    Test.@test cond_A(first(design_matrix(plan, lmax, M))) < 1.2

    C_true = rand_coeffs(lmax, 42)
    f_true = zeros(M); NUFSHT.nusht_type2!(f_true, C_true, plan)

    C_solved = NUFSHT.allocate_coefficients(plan)
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

# The coefficient array holds every degree `l ≤ lmax` plus supernumerary slots with
# `lmax < l ≤ lmax+|m|`. The fit runs over `l ≤ lmax`, the rotation-invariant subspace; a fit over the
# `lmax(lmax+1)` supernumerary slots as well reaches a residual below the harmonic least-squares one.
Test.@testset "nusht_solve! fits the l ≤ lmax subspace" begin
    Random.seed!(6)
    lmax, M = 8, 500
    θ = acos.(2 .* rand(M) .- 1); φ = 2π .* rand(M); v = randn(M)
    plan = NUFSHT.make_plan(Float64, θ, φ, lmax)
    valid = valid_mask_ref(lmax)

    # The supernumerary slots synthesise a nonzero field, so the support check below is not vacuous.
    Cx = zeros(lmax + 1, 2lmax + 1); Cx[.!valid] .= randn(sum(.!valid))
    fx = zeros(M); NUFSHT.nusht_type2!(fx, Cx, plan)
    Test.@test sqrt(sum(abs2, fx)) > 1

    Cs = zeros(lmax + 1, 2lmax + 1)
    NUFSHT.nusht_solve!(Cs, v, plan; rtol = 1e-12, maxiter = 800)
    Test.@test sum(abs2, Cs[.!valid]) == 0

    # Against dense least squares over the `l ≤ lmax` modes.
    A, idx = design_matrix(plan, lmax, M)
    cqr = A \ v
    Test.@test sqrt(sum(abs2, Cs[idx] .- cqr)) / sqrt(sum(abs2, cqr)) < 1e-6

    # The residual matches the harmonic least-squares one, which bounds it from below.
    fs = zeros(M); NUFSHT.nusht_type2!(fs, Cs, plan)
    res_solve = sqrt(sum(abs2, fs .- v)); res_qr = sqrt(sum(abs2, A * cqr .- v))
    Test.@test res_solve ≥ res_qr * (1 - 1e-8)
    Test.@test res_solve ≤ res_qr * (1 + 1e-6)
    @info "solve residual $res_solve vs QR $res_qr; invalid-slot energy $(sum(abs2, Cs[.!valid]))"
    NUFSHT.close!(plan)
end

# Degree power spectra are rotation invariants and `l ≤ lmax` is closed under rotation, so solving the same
# values at a rotated point set reproduces them. The difference is the least-squares sensitivity,
# `O(cond(A)²)`, times the NUFFT and solver tolerances.
Test.@testset "nusht_solve! is rotation equivariant" begin
    Random.seed!(6)
    lmax, M = 8, 500
    θ0 = acos.(2 .* rand(M) .- 1); φ0 = 2π .* rand(M); v = randn(M)
    a, b = 0.7, 1.1
    R = [cos(b) -sin(b) 0; sin(b) cos(b) 0; 0 0 1] * [cos(a) 0 sin(a); 0 1 0; -sin(a) 0 cos(a)]
    P = R * permutedims(hcat(sin.(θ0) .* cos.(φ0), sin.(θ0) .* sin.(φ0), cos.(θ0)))
    θ1 = acos.(clamp.(P[3, :], -1, 1)); φ1 = mod.(atan.(P[2, :], P[1, :]), 2π)

    function spectrum(θ, φ)
        p = NUFSHT.make_plan(Float64, θ, φ, lmax; tol = 1e-12)
        C = zeros(lmax + 1, 2lmax + 1)
        NUFSHT.nusht_solve!(C, v, p; rtol = 1e-12, maxiter = 800)
        NUFSHT.close!(p)
        return [sum(C[FastSphericalHarmonics.sph_mode(l, m)]^2 for m in -l:l) for l in 0:lmax]
    end
    s0, s1 = spectrum(θ0, φ0), spectrum(θ1, φ1)
    rel = sqrt(sum(abs2, s0 .- s1)) / sqrt(sum(abs2, s0))
    @info "degree-spectrum change under rotation: $rel"
    Test.@test rel < 1e-8
end

# Columns retire at their own tolerance and the survivors are packed forward. The columns are
# independent, so the batched answer equals solving them one at a time; differing bandlimits and
# amplitudes make them converge at different iterations.
Test.@testset "batched solve retires columns independently" begin
    Random.seed!(11)
    lmax, M, B = 12, 676, 4
    N, Nf = lmax + 1, 2lmax + 1
    θ = acos.(2 .* rand(M) .- 1); φ = 2π .* rand(M)

    Ct = zeros(N, Nf, B)
    for (b, (L, a)) in enumerate(((2, 1.0), (5, 1e2), (9, 1e-2), (12, 3.0)))
        for l in 1:L, m in -l:l
            Ct[FastSphericalHarmonics.sph_mode(l, m), b] = a * randn()
        end
    end

    pB = NUFSHT.make_plan(Float64, θ, φ, lmax; ntrans = B)
    fB = zeros(M, B); NUFSHT.nusht_type2!(fB, Ct, pB)
    ws = FTB.LSMRWorkspace(pB)
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
    # `residual` is the largest of the per-column `ws.residual`.
    Test.@test relB == maximum(ws.residual)
    Test.@test all(<(1e-6), ws.residual)
    NUFSHT.close!(pB); NUFSHT.close!(p1)
end

# Per-slot solver state moves with its column when the survivors are packed forward. A mismatch grows
# only over later iterations, so the comparison sweeps `rtol`.
Test.@testset "batched solve matches single-column at every tolerance" begin
    Random.seed!(7)
    lmax, M, B = 8, 150, 4                       # M/modes = 1.85, barely overdetermined
    N, Nf = lmax + 1, 2lmax + 1
    θ, φ = iid_points(M, 7)
    F = randn(M, B)

    pB = NUFSHT.make_plan(Float64, θ, φ, lmax; ntrans = B, nthreads = 1)
    p1 = NUFSHT.make_plan(Float64, θ, φ, lmax; nthreads = 1)
    # The batched and single-column runs differ in NUFFT width, so they are not bit-identical; two
    # iterates at residual `rtol` differ in the coefficients by `~rtol·cond(A)`.
    κ = cond_A(first(design_matrix(p1, lmax, M)))
    @info "batched-vs-single fixture: cond(A) = $κ"
    for rtol in (1e-4, 1e-6, 1e-8)
        tol = 20 * rtol * κ
        CB = zeros(N, Nf, B)
        _, itB, relB, convB = NUFSHT.nusht_solve!(CB, F, pB; rtol = rtol, maxiter = 500)
        Test.@test itB < 500
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

# With `M` below `(lmax+1)^2` the system is rank deficient. The solve stops on its own, and a larger
# `maxiter` returns the same coefficients.
Test.@testset "nusht_solve! does not degrade with a larger iteration budget" begin
    for T in (Float64, Float32)
        M, lmax = 60, 12
        Test.@test M < (lmax + 1)^2                       # rank deficient by construction
        Random.seed!(4)
        θ = T.(acos.(2 .* rand(M) .- 1)); φ = T.(rand(M) .* 2π)
        f = T[abs(sin(3θ[k])) + T(0.1) for k in 1:M]

        # `nthreads = 1` makes the runs bit-reproducible, so a longer run shares a shorter one's prefix
        # exactly.
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
            Test.@test r.fieldres < 1
            Test.@test r.conv == (r.rel < T(1e-12))
        end
        # A longer run's iterates extend a shorter run's; once the stopping rule fires, extra budget
        # changes nothing.
        Test.@test res[2].rel ≤ res[1].rel
        Test.@test res[3].rel ≤ res[2].rel
        Test.@test res[2].iters == res[3].iters
        Test.@test res[2].C == res[3].C
        Test.@test res[3].iters < 1000                    # stopped on its own
        @info "semiconvergence ($T): iters=$(getfield.(res, :iters)) rel_res=$(getfield.(res, :rel))"
    end
end

# An unreachable `rtol` returns `converged == false` within the iteration budget.
Test.@testset "nusht_solve! reports whether it converged" begin
    Random.seed!(31)
    lmax = 8; M = 4 * (lmax + 1)^2
    θ = acos.(2 .* rand(M) .- 1); φ = 2π .* rand(M)
    p = NUFSHT.make_plan(Float64, θ, φ, lmax; nthreads = 1)
    C = zeros(lmax + 1, 2lmax + 1)
    _, it_no, _, conv_no = NUFSHT.nusht_solve!(C, randn(M), p; rtol = 1e-30, maxiter = 200)
    Test.@test !conv_no
    Test.@test it_no ≤ 200
    NUFSHT.close!(p)
end

# A batched solve whose columns retire narrows its NUFFT to the live width and keeps that plan pair for
# later solves. Moving the nodes moves that pair too, so a solve after `set_nodes!` recovers the
# coefficients of fields evaluated at the new nodes, on each backend that narrows.
Test.@testset "set_nodes! moves every plan a solve narrows into" begin
    lmax, B = 8, 4
    M = 4 * (lmax + 1)^2
    θa, φa = iid_points(M, 21)
    θb, φb = iid_points(M, 22)
    C_true = rand_coeffs(lmax, 23)
    for nufft in (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend())
        p = NUFSHT.make_plan(Float64, θa, φa, lmax; ntrans = B, nufft = nufft, nthreads = 1)
        # Columns 2:B are zero, so they retire before the first iteration and the solve runs at
        # width 1 throughout.
        fa = zeros(M, B); fa[:, 1] .= synth_ref(C_true, lmax, θa, φa)
        C = NUFSHT.allocate_coefficients(p)
        NUFSHT.nusht_solve!(C, fa, p; rtol = 1e-10)
        Test.@test NUFSHT._pool_built(p.size_pool) ≥ 1
        NUFSHT.set_nodes!(p, θb, φb)
        fb = zeros(M, B); fb[:, 1] .= synth_ref(C_true, lmax, θb, φb)
        _, _, _, conv = NUFSHT.nusht_solve!(C, fb, p; rtol = 1e-10)
        Test.@test conv
        Test.@test relerr(C[:, :, 1], C_true) < 1e-8
        NUFSHT.close!(p)
    end
end

# Weights `w` make the fit `min Σⱼ wⱼ |(A c − f)ⱼ|²`, which the dense least squares of `√W A` checks. LSMR
# is MINRES on the normal equations, so after `k` iterations `‖A^H W r‖ ≤ 2((κ−1)/(κ+1))ᵏ ‖A^H W f‖` with
# `κ` the condition number of `√W A`: a quadrature exact for products of two degree-`lmax` harmonics has
# `A^H W A = I`, `κ = 1`, and one iteration solves it. Data in the span of those harmonics have the same
# solution under any weighting.
Test.@testset "nusht_solve! with point weights" begin
    lmax = 10
    C_true = rand_coeffs(lmax, 51)
    θg, φg, wg = gauss_legendre_grid(lmax)

    # Points drawn with density `∝ 1 + 8cos²θ`, weighted by its inverse.
    Random.seed!(52)
    M = 4 * (lmax + 1)^2
    θc = Float64[]
    while length(θc) < M
        t = acos(2rand() - 1)
        rand() * 9 < 1 + 8cos(t)^2 && push!(θc, t)
    end
    φc = 2π .* rand(M)
    wc = (4π / M) .* (1 + 8 / 3) ./ (1 .+ 8 .* cos.(θc) .^ 2)

    rtol = 1e-11
    for (name, θ, φ, w) in (("Gauss–Legendre", θg, φg, wg), ("clustered", θc, φc, wc))
        p = NUFSHT.make_plan(Float64, θ, φ, lmax; tol = 1e-13, nthreads = 1)
        f = zeros(length(θ)); NUFSHT.nusht_type2!(f, C_true, p)
        C0 = NUFSHT.allocate_coefficients(p); Cw = NUFSHT.allocate_coefficients(p)
        _, it0, _, conv0 = NUFSHT.nusht_solve!(C0, f, p; rtol = rtol, maxiter = 2000)
        _, itw, _, convw = NUFSHT.nusht_solve!(Cw, f, p; weights = w, rtol = rtol, maxiter = 2000)
        @info "weighted fit, $name: $it0 iterations unweighted, $itw weighted"
        Test.@test conv0 && convw
        Test.@test relerr(C0[:, :, 1], C_true) < 1e-8 && relerr(Cw[:, :, 1], C_true) < 1e-8
        A, idx = design_matrix(p, lmax, length(θ))
        κ = cond_A(sqrt.(w) .* A)
        Test.@test itw ≤ (κ < 1 + 1e-6 ? 2 : ceil(Int, log(2 / rtol) / log((κ + 1) / (κ - 1))))
        # Data off the span: the fit is the weighted least squares.
        Random.seed!(53)
        fn = f .+ 0.1 .* randn(length(f))
        Cn = NUFSHT.allocate_coefficients(p)
        NUFSHT.nusht_solve!(Cn, fn, p; weights = w, rtol = rtol, maxiter = 2000)
        Test.@test relerr(Cn[:, :, 1][idx], (sqrt.(w) .* A) \ (sqrt.(w) .* fn)) < 1e-8
        # The weighted operator built once and passed in.
        Cop = NUFSHT.allocate_coefficients(p)
        NUFSHT.nusht_solve!(Cop, f, FTB.RangeWeighted(p, w); rtol = rtol, maxiter = 2000)
        Test.@test relerr(Cop[:, :, 1], C_true) < 1e-8
        NUFSHT.close!(p)
    end
end

# With weights the analysis is the quadrature `Σⱼ wⱼ fⱼ Y_lm(xⱼ)` over the degrees `l ≤ lmax`, the other
# slots zero: on a rule exact for the product of two degree-`lmax` harmonics, the coefficients the field
# was synthesized from by the direct sum; on any points and weights, that sum evaluated directly.
# Without weights it is the least-squares fit, scored against dense least squares.
Test.@testset "nusht_analyze!: quadrature and fit" begin
    lmax, B = 8, 2
    valid = valid_mask_ref(lmax)
    θg, φg, wg = gauss_legendre_grid(lmax)
    Mg = length(θg)
    for FE in (Float64, ComplexF64)
        Ct = rand_coeffs(lmax, 61, B)
        FE <: Complex && (Ct = Ct .+ im .* rand_coeffs(lmax, 62, B))
        f = zeros(FE, Mg, B)
        for b in 1:B
            f[:, b] .= synth_ref(real.(Ct[:, :, b]), lmax, θg, φg)
            FE <: Complex && (f[:, b] .+= im .* synth_ref(imag.(Ct[:, :, b]), lmax, θg, φg))
        end
        p = NUFSHT.make_plan(FE, θg, φg, lmax; ntrans = B, tol = 1e-13, nthreads = 1)
        C = NUFSHT.allocate_coefficients(p)
        res = NUFSHT.nusht_analyze!(C, f, p; weights = wg)
        Test.@test res.iterations == 0 && res.converged
        Test.@test relerr(C, Ct) < 1e-11
        Test.@test all(b -> all(iszero, view(C, :, :, b)[.!valid]), 1:B)
        NUFSHT.close!(p)
    end

    Random.seed!(63)
    M = 500
    θ, φ = iid_points(M, 63)
    w = (4π / M) .* (0.5 .+ rand(M))
    g = randn(M)
    p = NUFSHT.make_plan(Float64, θ, φ, lmax; tol = 1e-13, nthreads = 1)
    C = NUFSHT.allocate_coefficients(p)
    NUFSHT.nusht_analyze!(C, g, p; weights = w)
    Test.@test relerr(C[:, :, 1], quadrature_ref(g, w, lmax, θ, φ)) < 1e-11

    # A zero weight drops its point whatever the field holds there: NaN and Inf at the masked points give
    # the coefficients zeros there give.
    wm = copy(w); wm[1:7:M] .= 0
    gz = copy(g); gz[1:7:M] .= 0
    gn = copy(g); gn[1:14:M] .= NaN; gn[8:14:M] .= Inf
    Cz = NUFSHT.allocate_coefficients(p); Cn = NUFSHT.allocate_coefficients(p)
    NUFSHT.nusht_analyze!(Cz, gz, p; weights = wm)
    NUFSHT.nusht_analyze!(Cn, gn, p; weights = wm)
    Test.@test Cn == Cz
    Test.@test relerr(Cn[:, :, 1], quadrature_ref(gz, wm, lmax, θ, φ)) < 1e-11

    _, _, _, conv = NUFSHT.nusht_analyze!(C, g, p; rtol = 1e-12, maxiter = 800)
    Test.@test conv
    A, idx = design_matrix(p, lmax, M)
    Test.@test relerr(C[:, :, 1][idx], A \ g) < 1e-6
    NUFSHT.close!(p)
end

# `nusht_filter_renorm!` divides by the filtered mask where it is at least `threshold` and writes zero
# below, so a higher threshold zeroes a superset of the points; a literal threshold serves a Float32 plan.
Test.@testset "nusht_filter_renorm! zeroes the points below threshold" begin
    lmax = 8; M = 4 * (lmax + 1)^2
    θ, φ = fib_points(M)
    mask = cos.(θ) .< 0.4
    filt = NUFSHT.gaussian_from_scale(2500e3)
    for T in (Float64, Float32)
        p = NUFSHT.make_plan(T, T.(θ), T.(φ), lmax; nthreads = 1)
        m = T.(mask)
        f = T.(synth_ref(rand_coeffs(lmax, 71), lmax, θ, φ)) .* m
        ws = FTB.LSMRWorkspace(p)
        num = zeros(T, M); NUFSHT.nusht_filter!(num, f, filt, p; ws)
        zeroed = map((0.05, 0.6)) do t
            out = copy(num); mf = zeros(T, M)
            NUFSHT.nusht_filter_renorm!(out, m, filt, p; mask_filt = mf, ws, threshold = t)
            below = abs.(mf) .< T(t)
            Test.@test any(below) && !all(below)
            Test.@test all(iszero, out[below])
            Test.@test out[.!below] == num[.!below] ./ mf[.!below]
            below
        end
        Test.@test all(zeroed[1] .<= zeroed[2]) && count(zeroed[2]) > count(zeroed[1])
        NUFSHT.close!(p)
    end
end
