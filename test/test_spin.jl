# `exp(-iβ Jy)` in the |ℓm⟩ basis, built from the angular-momentum ladder operators: the definition of the
# Wigner small-d matrix, independent of this package's sums and recurrences.
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
end

# Each order is one task's whole computation, so the tasks a transform runs on change no sum.
Test.@testset "ring S-step on several tasks equals one task" begin
    Random.seed!(9)
    lmax, M = 72, 2000
    θ = acos.(2 .* rand(M) .- 1); φ = 2π .* rand(M)
    sf = randn(ComplexF64, lmax + 1, 2lmax + 1, 3)
    DS = NUFSHT.SpectralBackends.DirectSumSpectralBackend()
    one_task, four = (NUFSHT.make_spin_plan(θ, φ, lmax, 1; nthreads = n, ntrans = 3, nufft = DS) for n in (1, 4))
    e1, e4 = one_task.engine, four.engine
    Test.@test NUFSHT._ring_tasks(four, e4) == 4
    G1 = copy(NUFSHT._assemble_G_impl!(one_task.G, sf, one_task, e1))
    Test.@test NUFSHT._assemble_G_impl!(four.G, sf, four, e4) == G1
    a1 = NUFSHT._assemble_G_adjoint_impl!(zeros(ComplexF64, size(sf)), G1, one_task, e1)
    a4 = NUFSHT._assemble_G_adjoint_impl!(zeros(ComplexF64, size(sf)), G1, four, e4)
    Test.@test a4 == a1
    NUFSHT.close!(one_task); NUFSHT.close!(four)
end

# Synthesis against Goldberg ₛY_ℓm computed from `Jy`. Odd `m+s` is the discriminating case for the
# `i^(m+s)` prefactor.
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
    NUFSHT.close!(plan)
end

# Coefficients exist only for `ℓ ≥ max(|m|,|s|)`: the forward operator annihilates the rest of the array
# and the solve returns zero there.
Test.@testset "spin solve stays in the ℓ ≥ max(|m|,|s|) triangle" begin
    Random.seed!(6)
    lmax, s, M = 8, 2, 500
    θ = acos.(2 .* rand(M) .- 1); φ = 2π .* rand(M)
    plan = NUFSHT.make_spin_plan(θ, φ, lmax, s)
    valid = falses(lmax + 1, 2lmax + 1)
    for l in abs(s):lmax, m in -l:l
        valid[NUFSHT.spin_coeff_index(l, m, lmax)] = true
    end

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

        sf = zeros(ComplexF64, shp)
        for ℓ in abs(s):lmax, m in -ℓ:ℓ
            sf[NUFSHT.spin_coeff_index(ℓ, m, lmax)] = randn(ComplexF64)
        end

        f = zeros(ComplexF64, M)
        NUFSHT.nusht_type2_spin!(f, sf, plan)
        ref = [sum(sf[NUFSHT.spin_coeff_index(ℓ, m, lmax)] * NUFSHT.sYlm(s, ℓ, m, θ[j], φ[j])
                   for ℓ in abs(s):lmax for m in -ℓ:ℓ) for j in 1:M]
        synth_err = maximum(abs.(f .- ref)) / maximum(abs.(ref))
        @info "spin $s synthesis vs direct ₛY: rel_err=$synth_err"
        Test.@test synth_err < 1e-10

        x = randn(ComplexF64, shp)
        y = randn(ComplexF64, M)
        Ax = zeros(ComplexF64, M); NUFSHT.nusht_type2_spin!(Ax, x, plan)
        Aty = zeros(ComplexF64, shp); NUFSHT.nusht_type1_spin!(Aty, y, plan)
        adj_err = abs(sum(Ax .* conj.(y)) - sum(vec(x) .* conj.(vec(Aty)))) / abs(sum(Ax .* conj.(y)))
        Test.@test adj_err < 1e-12

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

# `wigner_d` loses accuracy above ℓ ≈ 40, so at lmax = 128 the references are `₀Y_ℓ0(θ) = N_ℓ P_ℓ(cosθ)`
# from the Legendre three-term recurrence and the adjoint identity.
Test.@testset "spin transforms at large lmax (stable where wigner_d fails)" begin
    Random.seed!(77)
    lmax = 128
    shp = (lmax + 1, 2lmax + 1)

    # (ℓ+1)P_{ℓ+1} = (2ℓ+1)xP_ℓ - ℓP_{ℓ-1}
    function legendreP(ℓ, x)
        ℓ == 0 && return one(x)
        p0 = one(x); p1 = x
        for k in 1:(ℓ - 1)
            p0, p1 = p1, ((2k + 1) * x * p1 - k * p0) / (k + 1)
        end
        return p1
    end
    Nℓ(ℓ) = sqrt((2ℓ + 1) / (4π))

    M = 500; θ = clamp.(π .* rand(M), 1e-10, π - 1e-10); φ = 2π .* rand(M)
    plan0 = NUFSHT.make_spin_plan(θ, φ, lmax, 0; tol = 1e-11)
    sf0 = zeros(ComplexF64, shp); sf0[NUFSHT.spin_coeff_index(lmax, 0, lmax)] = 1.0
    f0 = zeros(ComplexF64, M); NUFSHT.nusht_type2_spin!(f0, sf0, plan0)
    ref0 = [Nℓ(lmax) * legendreP(lmax, cos(θ[k])) for k in 1:M]
    err0 = maximum(abs.(real.(f0) .- ref0)) / maximum(abs.(ref0))
    @info "large-lmax s=0 mode(ℓ=$lmax,m=0) vs N_ℓ·Pℓ(cosθ): rel_err=$(err0)"
    Test.@test err0 < 1e-8
    NUFSHT.close!(plan0)

    plan = NUFSHT.make_spin_plan(θ, φ, lmax, 2; tol = 1e-10)
    x = randn(ComplexF64, shp); y = randn(ComplexF64, M)
    Ax = zeros(ComplexF64, M); NUFSHT.nusht_type2_spin!(Ax, x, plan)
    Aty = zeros(ComplexF64, shp); NUFSHT.nusht_type1_spin!(Aty, y, plan)
    adj_err = abs(sum(Ax .* conj.(y)) - sum(vec(x) .* conj.(vec(Aty)))) / abs(sum(Ax .* conj.(y)))
    @info "large-lmax s=2: adj_err=$adj_err"
    Test.@test adj_err < 1e-11
    NUFSHT.close!(plan)
end

# A real-`FE` spin plan folds its mode array, and on a backend with a real-data transform its strengths
# are real too. Both folds equal the unfolded operator on the real spin fields: `s = 0` with
# `conj(sf[l,m]) = (-1)^m sf[l,-m]`, from `conj(Y_lm) = (-1)^m Y_l,-m`.
Test.@testset "real-field spin plans: both folds are one operator" begin
    Random.seed!(313)
    lmax, M = 8, 324
    N, Nf = lmax + 1, 2lmax + 1
    θ = clamp.(π .* rand(M), 1e-9, π - 1e-9); φ = 2π .* rand(M)

    sf = zeros(ComplexF64, N, Nf)
    for ℓ in 0:lmax
        sf[NUFSHT.spin_coeff_index(ℓ, 0, lmax)] = randn()          # m = 0 must be real
        for m in 1:ℓ
            a = randn(ComplexF64)
            sf[NUFSHT.spin_coeff_index(ℓ,  m, lmax)] = a
            sf[NUFSHT.spin_coeff_index(ℓ, -m, lmax)] = (-1)^m * conj(a)
        end
    end

    # The complex plan's field is the reference, and it is real on these coefficients.
    full = NUFSHT.make_spin_plan(ComplexF64, θ, φ, lmax, 0; tol = 1e-12, nthreads = 1)
    gc = zeros(ComplexF64, M); NUFSHT.nusht_type2_spin!(gc, sf, full)
    Test.@test sqrt(sum(abs2, imag.(gc)) / sum(abs2, real.(gc))) < 1e-12
    ref = real.(gc)

    for be in (FTB.NonuniformFFTsBackend(), FTB.FINUFFTBackend(),
               NUFSHT.SpectralBackends.DirectSumSpectralBackend())
        p = NUFSHT.make_spin_plan(Float64, θ, φ, lmax, 0; tol = 1e-12, nthreads = 1, nufft = be)
        f = zeros(M); NUFSHT.nusht_type2_spin!(f, sf, p)
        Test.@test sqrt(sum(abs2, f .- ref) / sum(abs2, ref)) < 1e-10

        # Exact adjoint over the reals: the field is real, the coefficients complex.
        x = zeros(ComplexF64, N, Nf); x .= sf
        y = randn(M)
        Ax = zeros(M);                    NUFSHT.nusht_type2_spin!(Ax, x, p)
        Aty = zeros(ComplexF64, N, Nf);   NUFSHT.nusht_type1_spin!(Aty, y, p)
        adj = abs(sum(Ax .* y) - real(sum(conj.(x) .* Aty))) / abs(sum(Ax .* y))
        Test.@test adj < 1e-11
        NUFSHT.close!(p)
    end
    NUFSHT.close!(full)
end

# A real field's coefficients are Hermitian, `(lmax+1)²` real numbers of the `(lmax+1)(2lmax+1)` complex
# ones the array holds, and a real-field solve runs over those packed real degrees. The restriction
# exists only at `s = 0`: conjugation maps spin `s` to spin `-s`, so a spin-`s ≠ 0` field is never real.
Test.@testset "real-field spin solve fits the Hermitian degrees" begin
    Random.seed!(414)
    lmax, M = 8, 324
    N, Nf = lmax + 1, 2lmax + 1
    K = (lmax + 1)^2
    θ = clamp.(π .* rand(M), 1e-9, π - 1e-9); φ = 2π .* rand(M)

    # The packed map is an isometry onto the Hermitian arrays, and `_pack_herm!` is its exact adjoint.
    p1 = randn(K, 1)
    sfu = zeros(ComplexF64, N, Nf, 1); NUFSHT._unpack_herm!(sfu, p1, lmax, 1)
    Test.@test sqrt(sum(abs2, sfu)) ≈ sqrt(sum(abs2, p1))
    g = randn(ComplexF64, N, Nf, 1); Utg = zeros(K, 1)
    NUFSHT._pack_herm!(Utg, g, lmax, 1)
    Test.@test real(sum(conj.(sfu) .* g)) ≈ sum(p1 .* Utg)
    for ℓ in 0:lmax, m in 1:ℓ
        Test.@test sfu[NUFSHT.spin_coeff_index(ℓ, -m, lmax), 1] ≈
                   (-1)^m * conj(sfu[NUFSHT.spin_coeff_index(ℓ, m, lmax), 1])
    end

    sf = zeros(ComplexF64, N, Nf)
    for ℓ in 0:lmax
        sf[NUFSHT.spin_coeff_index(ℓ, 0, lmax)] = randn()
        for m in 1:ℓ
            a = randn(ComplexF64)
            sf[NUFSHT.spin_coeff_index(ℓ,  m, lmax)] = a
            sf[NUFSHT.spin_coeff_index(ℓ, -m, lmax)] = (-1)^m * conj(a)
        end
    end

    for be in (FTB.NonuniformFFTsBackend(), FTB.FINUFFTBackend())
        p = NUFSHT.make_spin_plan(Float64, θ, φ, lmax, 0; tol = 1e-13, nthreads = 1, nufft = be)
        f = zeros(M); NUFSHT.nusht_type2_spin!(f, sf, p)
        S = zeros(ComplexF64, N, Nf)
        _, it, relres, conv = NUFSHT.nusht_solve_spin!(S, f, p; rtol = 1e-11, maxiter = 500)
        Test.@test conv && it < 200
        Test.@test relerr(S, sf) < 1e-9
        NUFSHT.close!(p)
    end

    pc = NUFSHT.make_spin_plan(ComplexF64, θ, φ, lmax, 0; tol = 1e-13, nthreads = 1)
    fc = zeros(ComplexF64, M); NUFSHT.nusht_type2_spin!(fc, sf, pc)
    Sc = zeros(ComplexF64, N, Nf)
    _, itc, _, convc = NUFSHT.nusht_solve_spin!(Sc, fc, pc; rtol = 1e-11, maxiter = 500)
    Test.@test convc && relerr(Sc, sf) < 1e-9
    NUFSHT.close!(pc)

    ps = NUFSHT.make_spin_plan(Float64, θ, φ, lmax, 1; tol = 1e-10, nthreads = 1)
    Test.@test_throws ArgumentError NUFSHT.nusht_solve_spin!(zeros(ComplexF64, N, Nf), zeros(M), ps;
                                                             maxiter = 2)
    NUFSHT.close!(ps)
end

# Random coefficients for `ℓ ≥ max(|m|, |s|)`, Hermitian when `hermitian`: the coefficients of a real
# spin-0 field.
function _spin_coeffs(lmax, s, seed; hermitian = false)
    Random.seed!(seed)
    sf = zeros(ComplexF64, lmax + 1, 2lmax + 1)
    for ℓ in abs(s):lmax
        sf[NUFSHT.spin_coeff_index(ℓ, 0, lmax)] = hermitian ? randn() : randn(ComplexF64)
        for m in 1:ℓ
            a = randn(ComplexF64)
            sf[NUFSHT.spin_coeff_index(ℓ, m, lmax)] = a
            sf[NUFSHT.spin_coeff_index(ℓ, -m, lmax)] = hermitian ? (-1)^m * conj(a) : randn(ComplexF64)
        end
    end
    return sf
end

# The direct sum `Σ sf ₛY_ℓm` at the points.
_spin_field(sf, s, lmax, θ, φ) =
    [sum(sf[NUFSHT.spin_coeff_index(ℓ, m, lmax)] * NUFSHT.sYlm(s, ℓ, m, θ[j], φ[j])
         for ℓ in abs(s):lmax for m in -ℓ:ℓ) for j in eachindex(θ)]

# On a rule exact for the product of two degree-`lmax` spin harmonics, the quadrature returns the
# coefficients the field was synthesized from by the direct sum: a complex spin-1 field, and a real
# spin-0 field on each fold of the real-field plan, whose coefficients are Hermitian. Without weights the
# analysis is the fit, which recovers the same coefficients from the same exact data.
Test.@testset "nusht_analyze!: spin quadrature and fit" begin
    lmax = 6
    θ, φ, w = gauss_legendre_grid(lmax)
    M = length(θ)

    sf = _spin_coeffs(lmax, 1, 71)
    f = _spin_field(sf, 1, lmax, θ, φ)
    p = NUFSHT.make_spin_plan(ComplexF64, θ, φ, lmax, 1; tol = 1e-13, nthreads = 1)
    S = NUFSHT.allocate_coefficients(p)
    res = NUFSHT.nusht_analyze!(S, f, p; weights = w)
    Test.@test res.iterations == 0 && res.converged
    Test.@test relerr(S[:, :, 1], sf) < 1e-10
    _, _, _, conv = NUFSHT.nusht_analyze!(S, f, p; rtol = 1e-12, maxiter = 400)
    Test.@test conv
    Test.@test relerr(S[:, :, 1], sf) < 1e-9
    NUFSHT.close!(p)

    sh = _spin_coeffs(lmax, 0, 72; hermitian = true)
    f0 = real.(_spin_field(sh, 0, lmax, θ, φ))
    for be in (FTB.NonuniformFFTsBackend(), FTB.FINUFFTBackend(),
               NUFSHT.SpectralBackends.DirectSumSpectralBackend())
        p0 = NUFSHT.make_spin_plan(Float64, θ, φ, lmax, 0; tol = 1e-13, nthreads = 1, nufft = be)
        S0 = NUFSHT.allocate_coefficients(p0)
        NUFSHT.nusht_analyze!(S0, f0, p0; weights = w)
        Test.@test relerr(S0[:, :, 1], sh) < 1e-10
        NUFSHT.close!(p0)
    end
end

# Weights `w` make the fit `min Σⱼ wⱼ |(A sf − f)ⱼ|²`, checked against the dense least squares of `√W A`,
# `A` the spin harmonics evaluated directly, on data off their span: a complex spin-1 field, and a real
# spin-0 one. For real data the complex fit is Hermitian (the minimizer is unique and conjugation maps
# minimizers to minimizers), so it is also the real-field plan's fit over the Hermitian degrees.
Test.@testset "nusht_solve_spin! with point weights" begin
    lmax = 6
    M = 4 * (lmax + 1)^2
    θ, φ = iid_points(M, 91)
    w = (4π / M) .* (0.5 .+ rand(M))
    for (FE, s, seed) in ((ComplexF64, 1, 92), (Float64, 0, 93))
        lm = [(ℓ, m) for ℓ in abs(s):lmax for m in -ℓ:ℓ]
        A = [NUFSHT.sYlm(s, ℓ, m, θ[j], φ[j]) for j in 1:M, (ℓ, m) in lm]
        sf = _spin_coeffs(lmax, s, seed; hermitian = FE <: Real)
        fc = _spin_field(sf, s, lmax, θ, φ)
        f = FE <: Real ? real.(fc) .+ 0.1 .* randn(M) : fc .+ 0.1 .* randn(ComplexF64, M)
        ref = (sqrt.(w) .* A) \ (sqrt.(w) .* f)
        packed(S) = [S[NUFSHT.spin_coeff_index(ℓ, m, lmax), 1] for (ℓ, m) in lm]

        p = NUFSHT.make_spin_plan(FE, θ, φ, lmax, s; tol = 1e-13, nthreads = 1)
        S = NUFSHT.allocate_coefficients(p)
        _, _, _, conv = NUFSHT.nusht_solve_spin!(S, f, p; weights = w, rtol = 1e-11, maxiter = 2000)
        Test.@test conv
        Test.@test relerr(packed(S), ref) < 1e-8
        # The weighted operator built once and passed in.
        Sop = NUFSHT.allocate_coefficients(p)
        NUFSHT.nusht_solve_spin!(Sop, f, FTB.RangeWeighted(p, w); rtol = 1e-11, maxiter = 2000)
        Test.@test relerr(packed(Sop), ref) < 1e-8
        NUFSHT.close!(p)
    end
end

# A scalar plan synthesizes `Σ C_k Y_k` over FastTransforms' real harmonics, a spin-0 plan `Σ sf ₀Y_ℓm`
# over complex ones; the conversions map one coefficient array to the other, so both plans give the same
# field at the same scattered points, a real field with real `C` and Hermitian `sf`, a complex one with
# complex arrays, in both directions.
Test.@testset "scalar ↔ spin-0 coefficients synthesize the same field" begin
    lmax, M = 8, 400
    N, Nf = lmax + 1, 2lmax + 1
    θ, φ = iid_points(M, 81)
    kw = (; tol = 1e-12, nthreads = 1)
    scalar(FE, C) = (p = NUFSHT.make_plan(FE, θ, φ, lmax; kw...); f = zeros(FE, M);
                     NUFSHT.nusht_type2!(f, C, p); NUFSHT.close!(p); f)
    spin0(FE, sf) = (p = NUFSHT.make_spin_plan(FE, θ, φ, lmax, 0; kw...); f = zeros(FE, M);
                     NUFSHT.nusht_type2_spin!(f, sf, p); NUFSHT.close!(p); f)

    C = rand_coeffs(lmax, 82)
    sf = NUFSHT.scalar_to_spin0!(zeros(ComplexF64, N, Nf), C)
    Test.@test relerr(spin0(Float64, sf), scalar(Float64, C)) < 1e-10
    Test.@test relerr(NUFSHT.spin0_to_scalar!(zeros(N, Nf), sf), C) < 1e-14

    Cc = rand_coeffs(lmax, 83) .+ im .* rand_coeffs(lmax, 84)
    sfc = NUFSHT.scalar_to_spin0!(zeros(ComplexF64, N, Nf), Cc)
    Test.@test relerr(spin0(ComplexF64, sfc), scalar(ComplexF64, Cc)) < 1e-10
    Test.@test relerr(NUFSHT.spin0_to_scalar!(zeros(ComplexF64, N, Nf), sfc), Cc) < 1e-14

    sh = _spin_coeffs(lmax, 0, 85; hermitian = true)
    Ch = NUFSHT.spin0_to_scalar!(zeros(N, Nf), sh)
    Test.@test relerr(scalar(Float64, Ch), spin0(Float64, sh)) < 1e-10
    sg = _spin_coeffs(lmax, 0, 86)
    Cg = NUFSHT.spin0_to_scalar!(zeros(ComplexF64, N, Nf), sg)
    Test.@test relerr(scalar(ComplexF64, Cg), spin0(ComplexF64, sg)) < 1e-10
end

# The Wigner small-d in Jacobi form, `d^ℓ_{mn}(β) = (−1)^λ √(k!(k+a+b)!/((k+a)!(k+b)!)) sin^a(β/2)
# cos^b(β/2) P_k^{(a,b)}(cos β)` with `k = ℓ − max(|m|,|n|)`: `(a, λ) = (m−n, m−n)` when the smallest of
# `ℓ ± m`, `ℓ ± n` is `ℓ + n` or `ℓ − m`, `(n−m, 0)` when it is `ℓ − n` or `ℓ + m`, and
# `b = 2max(|m|,|n|) − a`. `P_k^{(a,b)}` comes from its three-term recurrence (DLMF §18.9), which holds at
# every degree these tests reach; the form is pinned against `Jy` below.
function _jacobi_ab(m::Int, n::Int)
    abs(n) >= abs(m) && return n <= 0 ? (m - n, m - n) : (n - m, 0)
    return m > 0 ? (m - n, m - n) : (n - m, 0)
end

function _jacobi_column(K::Int, a::Int, b::Int, x::Float64)
    P = zeros(K + 1)
    P[1] = 1.0
    K >= 1 && (P[2] = (a + 1) + (a + b + 2) * (x - 1) / 2)
    for n in 2:K
        c = 2n + a + b
        P[n + 1] = ((c - 1) * (c * (c - 2) * x + a^2 - b^2) * P[n] -
                    2 * (n + a - 1) * (n + b - 1) * c * P[n - 1]) / (2n * (n + a + b) * (c - 2))
    end
    return P
end

# `(a, b, λ)` of `d^ℓ_{mn}` and its factorial ratios `√Π_{i=1}^{b} (k+a+i)/(k+i)` for
# `k = 0 … lmax − max(|m|,|n|)`.
function _jacobi_form(lmax::Int, m::Int, n::Int)
    a, λ = _jacobi_ab(m, n)
    μ = max(abs(m), abs(n)); b = 2μ - a
    norms = [sqrt(prod((k + a + i) / (k + i) for i in 1:b; init = 1.0)) for k in 0:(lmax - μ)]
    return (; a, b, λ, norms)
end

# `d^ℓ_{mn}(β)` for `ℓ = max(|m|,|n|) … lmax`.
function _jacobi_d(form, β::Float64)
    P = _jacobi_column(length(form.norms) - 1, form.a, form.b, cos(β))
    w = (iseven(form.λ) ? 1.0 : -1.0) * sin(β / 2)^form.a * cos(β / 2)^form.b
    return w .* form.norms .* P
end

Test.@testset "Jacobi-form d^ℓ_{mn} is the rotation matrix" begin
    for β in (0.37, 1.1, 2.6), ℓ in 0:5, m in -ℓ:ℓ, n in -ℓ:ℓ
        d = _jacobi_d(_jacobi_form(ℓ, m, n), β)
        Test.@test d[ℓ - max(abs(m), abs(n)) + 1] ≈ _true_d(ℓ, m, n, β) atol = 1e-12
    end
end

# `Σ sf ₛY_ℓm` at the points, `ₛY_ℓm = N_ℓ d^ℓ_{m,−s}(θ) e^{imφ}` from the Jacobi form, one recurrence per
# order and point.
function _spin_field_jacobi(sf, s, lmax, θ, φ)
    f = zeros(ComplexF64, length(θ))
    for m in -lmax:lmax
        μ = max(abs(m), abs(s))
        μ > lmax && continue
        form = _jacobi_form(lmax, m, -s)
        c = [sf[NUFSHT.spin_coeff_index(ℓ, m, lmax)] * sqrt((2ℓ + 1) / (4π)) for ℓ in μ:lmax]
        for j in eachindex(θ)
            f[j] += sum(c .* _jacobi_d(form, θ[j])) * cis(m * φ[j])
        end
    end
    return f
end

# Synthesis at scattered points against the closed-form harmonics, every degree and order populated, at
# spins 0, ±1, ±2.
Test.@testset "spin synthesis matches the Jacobi-form ₛY_ℓm" begin
    for lmax in (8, 64), s in (0, 1, -1, 2, -2)
        θ, φ = iid_points(500, 950 + lmax + s)
        sf = _spin_coeffs(lmax, s, 960 + lmax + s)
        p = NUFSHT.make_spin_plan(ComplexF64, θ, φ, lmax, s; tol = 1e-12, nthreads = 1)
        f = zeros(ComplexF64, length(θ)); NUFSHT.nusht_type2_spin!(f, sf, p)
        Test.@test relerr(f, _spin_field_jacobi(sf, s, lmax, θ, φ)) < 1e-10
        NUFSHT.close!(p)
    end
end

# At lmax = 512 the starting values of high orders near the poles lie far below `floatmin(Float32)`.
# Single spin-0 modes there against `₀Y_ℓm = Λ_ℓm(θ) e^{imφ}` and `₀Y_ℓ,−m = (−1)^m Λ_ℓm(θ) e^{−imφ}`,
# `Λ` the normalized associated Legendre function with the Condon–Shortley phase (`plm_column_ref`), in
# double precision; the single-precision S-step's Fourier coefficients against the double-precision
# ones within `lmax eps(Float32)`.
Test.@testset "spin-0 synthesis at lmax = 512 against the associated Legendre functions" begin
    lmax, M = 512, 64
    θ, φ = iid_points(M, 980)
    DS = NUFSHT.SpectralBackends.DirectSumSpectralBackend()
    p64 = NUFSHT.make_spin_plan(ComplexF64, θ, φ, lmax, 0; nthreads = 1, nufft = DS)
    p32 = NUFSHT.make_spin_plan(ComplexF32, Float32.(θ), Float32.(φ), lmax, 0; nthreads = 1, nufft = DS)
    G64 = similar(p64.G); G32 = similar(p32.G)
    for (ℓ, m) in ((512, 0), (512, 3), (450, 400), (512, 511), (400, -390))
        sf = zeros(ComplexF64, lmax + 1, 2lmax + 1)
        sf[NUFSHT.spin_coeff_index(ℓ, m, lmax)] = 1
        f = zeros(ComplexF64, M); NUFSHT.nusht_type2_spin!(f, sf, p64)
        σ = m < 0 && isodd(m) ? -1 : 1
        ref = [σ * plm_column_ref(lmax, abs(m), θ[j])[ℓ + 1] * cis(m * φ[j]) for j in 1:M]
        Test.@test relerr(f, ref) < 1e-11
        NUFSHT._assemble_G_impl!(G64, sf, p64, p64.engine)
        NUFSHT._assemble_G_impl!(G32, ComplexF32.(sf), p32, p32.engine)
        Test.@test relerr(ComplexF64.(G32), G64) < lmax * eps(Float32)
    end
    NUFSHT.close!(p64); NUFSHT.close!(p32)
end

# `⟨A x, y⟩ = ⟨x, A† y⟩` with a batch of two: complex fields, and a real spin-0 field with Hermitian
# coefficients, whose adjoint is taken over the reals.
Test.@testset "spin adjoint identity" begin
    lmax, B, M = 32, 2, 900
    θ, φ = iid_points(M, 990)
    for (FE, s) in ((ComplexF64, 1), (ComplexF64, -2), (Float64, 0))
        p = NUFSHT.make_spin_plan(FE, θ, φ, lmax, s; tol = 1e-12, ntrans = B, nthreads = 1)
        x = cat((_spin_coeffs(lmax, s, 991 + b; hermitian = FE <: Real) for b in 1:B)...; dims = 3)
        Random.seed!(993)
        y = FE <: Real ? randn(M, B) : randn(ComplexF64, M, B)
        Ax = zeros(FE, M, B); NUFSHT.nusht_type2_spin!(Ax, x, p)
        Aty = zeros(ComplexF64, size(x)); NUFSHT.nusht_type1_spin!(Aty, y, p)
        lhs = sum(conj.(Ax) .* y)
        rhs = sum(conj.(x) .* Aty)
        err = FE <: Real ? abs(lhs - real(rhs)) : abs(lhs - rhs)
        Test.@test err / abs(lhs) < 1e-12
        NUFSHT.close!(p)
    end
end
