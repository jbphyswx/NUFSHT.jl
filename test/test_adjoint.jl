Test.@testset "Euclidean adjoint and batching" begin
    Random.seed!(7)
    lmax = 8
    Nθ, Nφ = lmax + 1, 2lmax + 1
    M = 400
    θ = π .* rand(M)
    φ = 2π .* rand(M)

    # On the default backend a real field takes the folded mode array, whose transpose doubles every
    # `kθ > 0` row. `x` fills the whole coefficient array, the supernumerary slots included.
    Test.@testset "scalar Euclidean adjoint ⟨Ax,y⟩ = ⟨x,A†y⟩ (scattered)" begin
        plan = NUFSHT.make_plan(θ, φ, lmax; tol = 1e-12, nthreads = 1)
        x = randn(Nθ, Nφ)
        y = randn(M)
        Ax = zeros(M); NUFSHT.nusht_type2!(Ax, x, plan)
        Aty = zeros(Nθ, Nφ); NUFSHT.nusht_type1!(Aty, y, plan)
        lhs = sum(Ax .* y)
        rhs = sum(x .* Aty)
        rel = abs(lhs - rhs) / abs(lhs)
        @info "scalar adjoint: |⟨Ax,y⟩-⟨x,A†y⟩|/|⟨Ax,y⟩| = $rel"
        Test.@test rel < 1e-11
        NUFSHT.close!(plan)
    end

    Test.@testset "spin: batched == looped" begin
        s = 1
        splan = NUFSHT.make_spin_plan(θ, φ, lmax, s; tol = 1e-12, nthreads = 1,
                                      nufft = FTB.FINUFFTBackend())
        B = 2
        splanB = NUFSHT.make_spin_plan(θ, φ, lmax, s; tol = 1e-12, ntrans = B, nthreads = 1,
                                       nufft = FTB.FINUFFTBackend())
        sfB = zeros(ComplexF64, Nθ, Nφ, B)
        for b in 1:B, ℓ in abs(s):lmax, m in -ℓ:ℓ
            sfB[NUFSHT.spin_coeff_index(ℓ, m, lmax), b] = randn(ComplexF64)
        end
        fsB = zeros(ComplexF64, M, B)
        NUFSHT.nusht_type2_spin!(fsB, sfB, splanB)
        for b in 1:B
            fb = zeros(ComplexF64, M); NUFSHT.nusht_type2_spin!(fb, sfB[:, :, b], splan)
            Test.@test maximum(abs.(fsB[:, b] .- fb)) < 1e-12
        end
        NUFSHT.close!(splan); NUFSHT.close!(splanB)
    end
end
