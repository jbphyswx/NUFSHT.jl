Test.@testset "mixed precision (Float32) — spin/recurrence path" begin
    Random.seed!(31)
    lmax = 12; N = lmax + 1; Nφ = 2lmax + 1; M = 4 * N^2
    θ = Float32.(clamp.(π .* rand(M), 1.0f-5, Float32(π) - 1.0f-5))
    φ = Float32.(2π .* rand(M))
    sp = NUFSHT.make_spin_plan(ComplexF32, θ, φ, lmax, 1; tol = 1.0f-5)

    sf = zeros(ComplexF32, N, Nφ)
    for ℓ in 1:6, m in -ℓ:ℓ
        sf[NUFSHT.spin_coeff_index(ℓ, m, lmax)] = randn(ComplexF32)
    end
    fs = zeros(ComplexF32, M); NUFSHT.nusht_type2_spin!(fs, sf, sp)
    ref = [sum(sf[NUFSHT.spin_coeff_index(ℓ, m, lmax)] *
               NUFSHT.sYlm(1, ℓ, m, Float64(θ[k]), Float64(φ[k])) for ℓ in 1:6 for m in -ℓ:ℓ)
           for k in 1:M]
    synth_err = sqrt(sum(abs2, fs .- ref)) / sqrt(sum(abs2, ref))
    Test.@test synth_err < 1e-4

    x = randn(ComplexF32, N, Nφ); y = randn(ComplexF32, M)
    Ax = zeros(ComplexF32, M); NUFSHT.nusht_type2_spin!(Ax, x, sp)
    Aty = zeros(ComplexF32, N, Nφ); NUFSHT.nusht_type1_spin!(Aty, y, sp)
    adj_err = abs(sum(Ax .* conj.(y)) - sum(vec(x) .* conj.(vec(Aty)))) / abs(sum(Ax .* conj.(y)))
    Test.@test adj_err < 1e-4
    @info "precision Float32: synth_err=$synth_err adj_err=$adj_err"
    NUFSHT.close!(sp)
end

Test.@testset "mixed precision (Float32) — scalar path" begin
    lmax = 10; N = lmax + 1; Nφ = 2lmax + 1; M = 4 * N^2
    θ64, φ64 = iid_points(M, 32)
    C64 = rand_coeffs(lmax, 33)
    ref = synth_ref(C64, lmax, θ64, φ64)

    p = NUFSHT.make_plan(Float32, Float32.(θ64), Float32.(φ64), lmax; tol = 1.0f-6)
    f = zeros(Float32, M); NUFSHT.nusht_type2!(f, Float32.(C64), p)
    synth_err = relerr(Float64.(f), ref)
    Test.@test synth_err < 2e-5

    x = randn(Float32, N, Nφ); y = randn(Float32, M)
    Ax = zeros(Float32, M); NUFSHT.nusht_type2!(Ax, x, p)
    Aty = zeros(Float32, N, Nφ); NUFSHT.nusht_type1!(Aty, y, p)
    adj_err = abs(sum(Float64.(Ax) .* Float64.(y)) - sum(Float64.(x) .* Float64.(Aty))) /
              abs(sum(Float64.(Ax) .* Float64.(y)))
    Test.@test adj_err < 1e-4
    @info "scalar precision Float32: synth_err=$synth_err adj_err=$adj_err"
    NUFSHT.close!(p)
end
