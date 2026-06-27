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
