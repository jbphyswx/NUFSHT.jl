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
        _, iters, relres = NUFSHT.nusht_solve_spin!(sol, fr, planr; rtol = 1e-10, maxiter = 500)
        coeff_err = maximum(abs.(sol .- sf)) / maximum(abs.(sf))
        @info "spin $s solve: iters=$iters rel_res=$relres coeff_err=$coeff_err"
        Test.@test coeff_err < 1e-7
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
    _, iters, relres = NUFSHT.nusht_solve_spin!(sol, f, plan; rtol = 1e-8, maxiter = 1500)
    frec = zeros(ComplexF64, Ms); NUFSHT.nusht_type2_spin!(frec, sol, plan)
    fieldrec = sqrt(sum(abs2, frec .- f) / sum(abs2, f))
    @info "large-lmax s=$s: adj_err=$adj_err  solve iters=$iters field-recovery=$fieldrec"
    Test.@test fieldrec < 1e-3
    NUFSHT.close!(plan)
end
