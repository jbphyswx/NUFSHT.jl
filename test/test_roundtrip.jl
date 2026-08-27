# `nusht_type1!` is the adjoint `A†`, not an inverse: at scattered points `A†A ≠ I`, and the package
# no longer pretends otherwise. Inversion is `nusht_solve!`. What has to round-trip is therefore
# solve∘synthesize, at scattered points — which is the operation the package is for — and the
# CC-grid claim belongs to `FastSphericalHarmonics.sph_transform`, not here.
Test.@testset "Round-trip: solve then synthesize recovers the field at scattered points" begin
    for (name, gen, lmax, M) in (("fibonacci", (M, s) -> fib_points(M), 12, 4 * 13^2),
                                 ("iid", iid_points, 10, 4 * 11^2))
        θ, φ = gen(M, 123)
        plan = NUFSHT.make_plan(θ, φ, lmax; tol = 1e-12, nthreads = 1)
        C_true = rand_coeffs(lmax, 5)
        f_in = zeros(M); NUFSHT.nusht_type2!(f_in, C_true, plan)

        C_out = similar(plan.F)
        _, iters, rel, conv = NUFSHT.nusht_solve!(C_out, f_in, plan; rtol = 1e-11, maxiter = 400)
        Test.@test conv
        # Overdetermined and well conditioned, so the coefficients themselves come back, not merely a
        # field that matches.
        Test.@test relerr(C_out, C_true) < 1e-8

        f_out = similar(f_in); NUFSHT.nusht_type2!(f_out, C_out, plan)
        @info "round-trip ($name, lmax=$lmax, M=$M): $iters iters, rel_res $rel, field $(relerr(f_out, f_in))"
        Test.@test relerr(f_out, f_in) < 1e-10
        NUFSHT.close!(plan)
    end
end

# Where synthesis is a spherical harmonic and where it is only a grid representation.
#
# The coefficient array is square and invertible *on grid samples*: slot `(i, j)` of the order-`m`
# column carries degree `l = i + |m| - 1`, up to `lmax + |m|`. As continuous functions those extra
# degrees are another matter. `plan_sph2fourier` gives `lmax+1` rows per column, so the available
# θ-frequencies are `cos(0θ) … cos(lmax·θ)` for even `|m|` and `sin(1θ) … sin((lmax+1)θ)` for odd,
# while a degree-`l` harmonic needs frequencies up to `l`.
#
# Measured slot by slot against `Y_lm`, the faithful set is exactly `l ≤ lmax` — the band-limited
# subspace, which is also the space `nusht_solve!` fits and the space the API advertises. Beyond it
# the stored coefficients reproduce the CC samples but not the harmonic between them. Checked against
# `Y_lm` rather than `sph_evaluate`, which shares the grid whose aliasing is the point at issue.
Test.@testset "synthesis reproduces Y_lm on exactly the band-limited slots" begin
    lmax = 6
    N, Nf = lmax + 1, 2lmax + 1
    M = 400
    θ, φ = iid_points(M, 31)
    plan = NUFSHT.make_plan(θ, φ, lmax; tol = 1e-13, nthreads = 1)
    f = zeros(M)
    valid = valid_mask_ref(lmax)
    Test.@test sum(valid) == (lmax + 1)^2

    for j in 1:Nf, i in 1:N
        valid[i, j] || continue
        m = iseven(j) ? -(j ÷ 2) : (j ÷ 2)
        l = i + abs(m) - 1
        C = zeros(N, Nf); C[i, j] = 1.0
        NUFSHT.nusht_type2!(f, C, plan)
        g = [ylm_ref(l, m, θ[k], φ[k]) for k in 1:M]
        Test.@test maximum(abs, f .- g) / maximum(abs, g) < 1e-11
    end
    @info "synthesis reproduces Y_lm on all $((lmax + 1)^2) band-limited slots of $(N * Nf) (lmax=$lmax)"
    NUFSHT.close!(plan)
end
