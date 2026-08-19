Test.@testset "Round-trip: type1 then type2 recovers field" begin
    Random.seed!(123)

    lmax = 20

    pts = FastSphericalHarmonics.sph_points(lmax + 1)
    θs_grid = pts[1]
    φs_grid = pts[2]

    θ_nodes = vec([θ for θ in θs_grid, φ in φs_grid])
    φ_nodes = vec([φ for θ in θs_grid, φ in φs_grid])

    M = length(θ_nodes)
    plan = NUFSHT.make_plan(θ_nodes, φ_nodes, lmax; tol=1e-10)

    # A GENERIC coefficient array, not a band-limited one. The array is a square, invertible
    # representation: every slot carries a real degree, those past `ℓ = lmax` included. Seeding only
    # low valid modes would leave the supernumerary half at zero and so would not notice a transform
    # that silently dropped it — the round-trip has to stay exact on the full space.
    C_true = randn(lmax+1, 2lmax+1)
    f_in = vec(FastSphericalHarmonics.sph_evaluate(C_true))

    C_out = similar(plan.C)
    NUFSHT.nusht_type1!(C_out, f_in, plan)

    f_out = similar(f_in)
    NUFSHT.nusht_type2!(f_out, C_out, plan)

    rms_err = sqrt(Statistics.mean(abs2.(f_out .- f_in)))
    @info "Round-trip RMS error (lmax=$lmax): $rms_err"
    Test.@test rms_err < 1e-6 * sqrt(Statistics.mean(abs2.(f_in)) + 1e-30)
end
