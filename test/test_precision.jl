# Mixed precision. The spin/recurrence path is fully type-generic in `T` (the Trapani–Navaza buffers
# and all transforms run at `T`). The scalar path is too, with one buffer pinned: FastTransforms has
# no `Float32` sphere plan, so `Fslice` — the dense slice every column is already copied through — stays
# `Float64` and that copy does the conversion. Every other buffer, and the NUFFT, runs at `T`.
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

# The scalar path at `Float32`: `Fslice` is the one buffer that stays `Float64` (FastTransforms has no
# single-precision sphere plan) and everything else — coefficients, the mode array, the strengths and
# the NUFFT itself — must be at `T`. Scored against the independent `Y_lm` sum, not against the
# `Float64` plan, so it measures the transform rather than the agreement of two runs of it.
Test.@testset "mixed precision (Float32) — scalar path" begin
    lmax = 10; N = lmax + 1; Nφ = 2lmax + 1; M = 4 * N^2
    θ64, φ64 = iid_points(M, 32)
    C64 = rand_coeffs(lmax, 33)
    ref = synth_ref(C64, lmax, θ64, φ64)

    for (T, tol, synth_tol, adj_tol) in ((Float64, 1e-12, 1e-11, 1e-11),
                                         (Float32, 1.0f-6, 2e-5, 1e-4))
        p = NUFSHT.make_plan(T, T.(θ64), T.(φ64), lmax; tol = tol)
        Test.@test eltype(p.C) === T && eltype(p.F) === T
        Test.@test eltype(p.Fhat) === Complex{T}
        # Real strengths and a half-height mode array are the same decision, whichever backend the
        # `Auto` resolution picked here.
        folded = size(p.Fhat, 1) == lmax + 2
        Test.@test eltype(NUFSHT._fbuf(p)) === (folded ? T : Complex{T})
        Test.@test eltype(p.Fslice) === Float64          # the one FastTransforms pins

        f = zeros(T, M); NUFSHT.nusht_type2!(f, T.(C64), p)
        Test.@test eltype(f) === T
        synth_err = relerr(Float64.(f), ref)
        Test.@test synth_err < synth_tol

        x = randn(T, N, Nφ); y = randn(T, M)
        Ax = zeros(T, M); NUFSHT.nusht_type2!(Ax, x, p)
        Aty = zeros(T, N, Nφ); NUFSHT.nusht_type1!(Aty, y, p)
        adj_err = abs(sum(Float64.(Ax) .* Float64.(y)) - sum(Float64.(x) .* Float64.(Aty))) /
                  abs(sum(Float64.(Ax) .* Float64.(y)))
        Test.@test adj_err < adj_tol
        @info "scalar precision $T: synth_err=$synth_err adj_err=$adj_err"
        NUFSHT.close!(p)
    end
end
