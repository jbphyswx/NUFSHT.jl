# `exp(-iβ Jy)` in the |ℓm⟩ basis, built from the angular-momentum ladder operators. This is the
# definition of the Wigner small-d matrix, independent of any explicit sum, recurrence, or convention
# choice in this package — which is what makes it a usable oracle. `sYlm` is NOT: it and the transform
# share the same `d`, so a sign error in `d` cancels in that comparison and stays invisible.
function _jy_rotation(ℓ::Int, β::Float64)
    ms = collect(-ℓ:ℓ); n = length(ms)
    Jp = zeros(ComplexF64, n, n); Jm = zeros(ComplexF64, n, n)
    for (j, m) in enumerate(ms), (i, mp) in enumerate(ms)
        mp == m + 1 && (Jp[i, j] = sqrt(ℓ * (ℓ + 1) - m * (m + 1)))
        mp == m - 1 && (Jm[i, j] = sqrt(ℓ * (ℓ + 1) - m * (m - 1)))
    end
    return exp(-im * β * (Jp .- Jm) ./ (2im))
end
_true_d(ℓ, m, n, β) = real(_jy_rotation(ℓ, β)[m + ℓ + 1, n + ℓ + 1])
# Goldberg / Newman–Penrose: ₛY_ℓm = N_ℓ d^ℓ_{m,−s}(θ) e^{imφ}
_goldberg(s, ℓ, m, θ, φ) = sqrt((2ℓ + 1) / (4π)) * _true_d(ℓ, m, -s, θ) * cis(m * φ)

Test.@testset "wigner_d is the rotation matrix ⟨ℓm|exp(-iβ Jy)|ℓn⟩" begin
    for β in (0.37, 1.1, π / 2), ℓ in 1:4
        R = _jy_rotation(ℓ, β)
        for m in -ℓ:ℓ, n in -ℓ:ℓ
            # Odd `m-n` is the discriminating case: a `(-1)^k` sum flips exactly those.
            Test.@test NUFSHT.wigner_d(ℓ, m, n, β) ≈ real(R[m + ℓ + 1, n + ℓ + 1]) atol = 1e-12
        end
    end

    # ℓ=1 closed forms, written out so a regression names itself.
    β = 0.37
    Test.@test NUFSHT.wigner_d(1, 0, 0, β) ≈ cos(β)
    Test.@test NUFSHT.wigner_d(1, 1, 1, β) ≈ (1 + cos(β)) / 2
    Test.@test NUFSHT.wigner_d(1, 1, -1, β) ≈ (1 - cos(β)) / 2
    Test.@test NUFSHT.wigner_d(1, 1, 0, β) ≈ -sin(β) / sqrt(2)
    Test.@test NUFSHT.wigner_d(1, 0, 1, β) ≈ +sin(β) / sqrt(2)
    Test.@test NUFSHT.wigner_d(1, -1, 0, β) ≈ +sin(β) / sqrt(2)
    Test.@test NUFSHT.wigner_d(1, 0, -1, β) ≈ -sin(β) / sqrt(2)
end

# The Trapani recurrence and `wigner_d` are separate implementations of the same object, so each must
# be checked against `Jy` rather than against the other — they agreed for a long time only because
# both carried the same `(-1)^(m-n)` error. Also pins the transposed storage relation.
Test.@testset "Trapani Δ is d(π/2)" begin
    lmax = 5; off = lmax + 1; L = 2lmax + 1
    dl = zeros(Float64, L, L); dlp = zeros(Float64, L, L)
    NUFSHT._wigner_d_halfpi_step!(dl, dlp, 0, off)
    for ℓ in 1:lmax
        dl, dlp = dlp, dl
        NUFSHT._wigner_d_halfpi_step!(dl, dlp, ℓ, off)
    end
    R = _jy_rotation(lmax, π / 2)
    for m in -lmax:lmax, n in -lmax:lmax
        Test.@test dl[n + off, m + off] ≈ real(R[m + lmax + 1, n + lmax + 1]) atol = 1e-12
        Test.@test dl[n + off, m + off] ≈ NUFSHT.wigner_d(lmax, m, n, π / 2) atol = 1e-12
    end
end

# Synthesis against Goldberg ₛY_ℓm computed from `Jy` — the check `sYlm` cannot provide. Odd `m+s` is
# the discriminating case: the `i^(m+s)` prefactor error flipped exactly those.
Test.@testset "spin synthesis matches Goldberg ₛY_ℓm" begin
    Random.seed!(4)
    lmax, s, M = 6, 1, 400
    θ = π .* rand(M) .* 0.9 .+ 0.05; φ = 2π .* rand(M)
    plan = NUFSHT.make_spin_plan(ComplexF64, θ, φ, lmax, s)
    for (ℓ, m) in ((1, 0), (2, 1), (3, 2), (4, 0), (5, 3), (6, -2))
        sf = zeros(ComplexF64, lmax + 1, 2lmax + 1)
        sf[NUFSHT.spin_coeff_index(ℓ, m, lmax)] = 1.0
        f = zeros(ComplexF64, M); NUFSHT.nusht_type2_spin!(f, sf, plan)
        g = [_goldberg(s, ℓ, m, θ[j], φ[j]) for j in 1:M]
        Test.@test sqrt(sum(abs2, f .- g) / sum(abs2, g)) < 1e-9
    end
    # The single value quoted in the issue, spelled out.
    Test.@test NUFSHT.sYlm(1, 1, 0, 0.4, 0.0) ≈ sqrt(3 / (4π)) * (-sin(0.4) / sqrt(2))
    NUFSHT.close!(plan)
end

# The spin coefficient array is rectangular with triangular content (`ℓ ≥ max(|m|,|s|)`), but unlike
# the scalar path the assembly reads and writes only that triangle, so the remaining slots really are
# a null space and the solver never enters them. Pin that, since it is what keeps the spin solve
# frame-independent for free.
Test.@testset "spin solve stays in the ℓ ≥ max(|m|,|s|) triangle" begin
    Random.seed!(6)
    lmax, s, M = 8, 2, 500
    θ = acos.(2 .* rand(M) .- 1); φ = 2π .* rand(M)
    plan = NUFSHT.make_spin_plan(θ, φ, lmax, s)
    valid = falses(lmax + 1, 2lmax + 1)
    for l in abs(s):lmax, m in -l:l
        valid[NUFSHT.spin_coeff_index(l, m, lmax)] = true
    end

    # Forward operator annihilates the complement exactly — the scalar path does not.
    bad = zeros(ComplexF64, lmax + 1, 2lmax + 1)
    bad[.!valid] .= randn(ComplexF64, sum(.!valid))
    fb = zeros(ComplexF64, M); NUFSHT.nusht_type2_spin!(fb, bad, plan)
    Test.@test sum(abs2, fb) == 0

    sf = zeros(ComplexF64, lmax + 1, 2lmax + 1)
    NUFSHT.nusht_solve_spin!(sf, randn(ComplexF64, M), plan; rtol = 1e-10, maxiter = 400)
    Test.@test sum(abs2, sf[.!valid]) == 0
    NUFSHT.close!(plan)
end

Test.@testset "spin-weighted scattered transforms" begin
    Random.seed!(202)

    for s in (1, -1, 2)
        lmax = 10
        shp = (lmax + 1, 2lmax + 1)
        M = 800
        θ = π .* rand(M)
        φ = 2π .* rand(M)
        plan = NUFSHT.make_spin_plan(θ, φ, lmax, s; tol = 1e-12)

        # random coefficients (ℓ ≥ max(|m|,|s|))
        sf = zeros(ComplexF64, shp)
        for ℓ in abs(s):lmax, m in -ℓ:ℓ
            sf[NUFSHT.spin_coeff_index(ℓ, m, lmax)] = randn(ComplexF64)
        end

        # synthesis must match direct evaluation of ₛY_ℓm
        f = zeros(ComplexF64, M)
        NUFSHT.nusht_type2_spin!(f, sf, plan)
        ref = [sum(sf[NUFSHT.spin_coeff_index(ℓ, m, lmax)] * NUFSHT.sYlm(s, ℓ, m, θ[j], φ[j])
                   for ℓ in abs(s):lmax for m in -ℓ:ℓ) for j in 1:M]
        synth_err = maximum(abs.(f .- ref)) / maximum(abs.(ref))
        @info "spin $s synthesis vs direct ₛY: rel_err=$synth_err"
        Test.@test synth_err < 1e-10

        # type1 is the exact Euclidean adjoint of type2
        x = randn(ComplexF64, shp)
        y = randn(ComplexF64, M)
        Ax = zeros(ComplexF64, M); NUFSHT.nusht_type2_spin!(Ax, x, plan)
        Aty = zeros(ComplexF64, shp); NUFSHT.nusht_type1_spin!(Aty, y, plan)
        adj_err = abs(sum(Ax .* conj.(y)) - sum(vec(x) .* conj.(vec(Aty)))) / abs(sum(Ax .* conj.(y)))
        Test.@test adj_err < 1e-12

        # exact inversion at scattered points (overdetermined)
        Msolve = 4 * (lmax + 1)^2
        θr = π .* rand(Msolve); φr = 2π .* rand(Msolve)
        planr = NUFSHT.make_spin_plan(θr, φr, lmax, s; tol = 1e-12)
        fr = zeros(ComplexF64, Msolve); NUFSHT.nusht_type2_spin!(fr, sf, planr)
        sol = zeros(ComplexF64, shp)
        _, iters, relres, conv = NUFSHT.nusht_solve_spin!(sol, fr, planr; rtol = 1e-10, maxiter = 500)
        coeff_err = maximum(abs.(sol .- sf)) / maximum(abs.(sf))
        @info "spin $s solve: iters=$iters rel_res=$relres coeff_err=$coeff_err"
        Test.@test coeff_err < 1e-7
        Test.@test conv
        Test.@test conv == (relres < 1e-10)
    end
end

# The Trapani–Navaza recurrence is stable to ℓ≈1024; the explicit-factorial `wigner_d`/`sYlm`
# it replaced loses all accuracy above ℓ≈40 (single unit mode blows up to max|f|~1e9 at lmax=64).
# So this large-lmax testset cannot use `sYlm` as a reference. Anchors: (i) an INDEPENDENT stable
# reference for s=0,m=0 via ₀Y_{ℓ0}(θ)=N_ℓ Pℓ(cosθ) (Legendre 3-term recurrence); (ii) boundedness
# (catches the blow-up); (iii) exact Euclidean adjoint; (iv) round-trip inversion.
Test.@testset "spin recurrence: correctness at large lmax (stable where wigner_d fails)" begin
    Random.seed!(77)
    lmax = 128
    shp = (lmax + 1, 2lmax + 1)

    # Stable Legendre Pℓ(x) via (ℓ+1)P_{ℓ+1}=(2ℓ+1)xPℓ-ℓP_{ℓ-1}.
    function legendreP(ℓ, x)
        ℓ == 0 && return one(x)
        p0 = one(x); p1 = x
        for k in 1:(ℓ - 1)
            p0, p1 = p1, ((2k + 1) * x * p1 - k * p0) / (k + 1)
        end
        return p1
    end
    Nℓ(ℓ) = sqrt((2ℓ + 1) / (4π))

    # (i) s=0, single mode (ℓ=lmax, m=0): field = N_ℓ Pℓ(cosθ), independent & stable.
    M = 500; θ = clamp.(π .* rand(M), 1e-10, π - 1e-10); φ = 2π .* rand(M)
    plan0 = NUFSHT.make_spin_plan(θ, φ, lmax, 0; tol = 1e-11)
    sf0 = zeros(ComplexF64, shp); sf0[NUFSHT.spin_coeff_index(lmax, 0, lmax)] = 1.0
    f0 = zeros(ComplexF64, M); NUFSHT.nusht_type2_spin!(f0, sf0, plan0)
    ref0 = [Nℓ(lmax) * legendreP(lmax, cos(θ[k])) for k in 1:M]
    err0 = maximum(abs.(real.(f0) .- ref0)) / maximum(abs.(ref0))
    @info "large-lmax s=0 mode(ℓ=$lmax,m=0) vs N_ℓ·Pℓ(cosθ): rel_err=$(err0)  max|f|=$(maximum(abs,f0))"
    Test.@test err0 < 1e-8
    Test.@test maximum(abs, f0) < 10          # bounded (dense-Δ gave ~1e9 here)
    NUFSHT.close!(plan0)

    # (ii)–(iv) general spin, random band-limited field.
    s = 2; Ms = 4 * (lmax + 1)^2
    θs = clamp.(π .* rand(Ms), 1e-10, π - 1e-10); φs = 2π .* rand(Ms)
    plan = NUFSHT.make_spin_plan(θs, φs, lmax, s; tol = 1e-10)
    sf = zeros(ComplexF64, shp)
    for ℓ in abs(s):lmax, m in -ℓ:ℓ
        sf[NUFSHT.spin_coeff_index(ℓ, m, lmax)] = randn(ComplexF64)
    end
    f = zeros(ComplexF64, Ms); NUFSHT.nusht_type2_spin!(f, sf, plan)
    Test.@test isfinite(maximum(abs, f)) && maximum(abs, f) < 1e4     # bounded, not 1e9+

    x = randn(ComplexF64, shp); y = randn(ComplexF64, Ms)
    Ax = zeros(ComplexF64, Ms); NUFSHT.nusht_type2_spin!(Ax, x, plan)
    Aty = zeros(ComplexF64, shp); NUFSHT.nusht_type1_spin!(Aty, y, plan)
    adj_err = abs(sum(Ax .* conj.(y)) - sum(vec(x) .* conj.(vec(Aty)))) / abs(sum(Ax .* conj.(y)))
    Test.@test adj_err < 1e-11

    sol = zeros(ComplexF64, shp)
    _, iters, relres, conv = NUFSHT.nusht_solve_spin!(sol, f, plan; rtol = 1e-8, maxiter = 1500)
    frec = zeros(ComplexF64, Ms); NUFSHT.nusht_type2_spin!(frec, sol, plan)
    fieldrec = sqrt(sum(abs2, frec .- f) / sum(abs2, f))
    @info "large-lmax s=$s: adj_err=$adj_err  solve iters=$iters rel_res=$relres field-recovery=$fieldrec"
    Test.@test fieldrec < 1e-3
    Test.@test conv == (relres < 1e-8)
    NUFSHT.close!(plan)
end
