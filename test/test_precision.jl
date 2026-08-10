# Mixed precision. The spin/recurrence path is fully type-generic in `T` (the Trapani–Navaza buffers
# and all transforms run at `T`), so Float32 works end-to-end. (The scalar DFS path is Float64-locked
# by FastTransforms, whose `plan_sph2fourier` has no Float32 method — a dependency limitation.)
Test.@testset "mixed precision (Float32) — spin/recurrence path" begin
    Random.seed!(31)
    lmax = 12; N = lmax + 1; Nφ = 2lmax + 1; M = 4 * N^2
    for (T, tol, synth_tol, adj_tol) in ((Float64, 1e-11, 1e-9, 1e-12), (Float32, 1.0f-5, 1e-4, 1e-4))
        θ = T.(clamp.(π .* rand(M), 1.0f-5, Float32(π) - 1.0f-5))
        φ = T.(2π .* rand(M))
        sp = NUFSHT.make_spin_plan(Complex{T}, θ, φ, lmax, 1; tol = tol)
        Test.@test eltype(sp.dl_curr) === T                       # recurrence runs at T
        Test.@test eltype(sp.G) === Complex{T}

        sf = zeros(Complex{T}, N, Nφ)
        for ℓ in 1:6, m in -ℓ:ℓ
            sf[NUFSHT.spin_coeff_index(ℓ, m, lmax)] = randn(Complex{T})
        end
        fs = zeros(Complex{T}, M); NUFSHT.nusht_type2_spin!(fs, sf, sp)
        Test.@test eltype(fs) === Complex{T}
        ref = [sum(sf[NUFSHT.spin_coeff_index(ℓ, m, lmax)] *
                   NUFSHT.sYlm(1, ℓ, m, Float64(θ[k]), Float64(φ[k])) for ℓ in 1:6 for m in -ℓ:ℓ)
               for k in 1:M]
        synth_err = sqrt(sum(abs2, fs .- ref)) / sqrt(sum(abs2, ref))
        Test.@test synth_err < synth_tol

        # exact Euclidean adjoint holds at T
        x = randn(Complex{T}, N, Nφ); y = randn(Complex{T}, M)
        Ax = zeros(Complex{T}, M); NUFSHT.nusht_type2_spin!(Ax, x, sp)
        Aty = zeros(Complex{T}, N, Nφ); NUFSHT.nusht_type1_spin!(Aty, y, sp)
        adj_err = abs(sum(Ax .* conj.(y)) - sum(vec(x) .* conj.(vec(Aty)))) / abs(sum(Ax .* conj.(y)))
        Test.@test adj_err < adj_tol
        @info "precision $T: synth_err=$synth_err adj_err=$adj_err"
        NUFSHT.close!(sp)
    end
end
