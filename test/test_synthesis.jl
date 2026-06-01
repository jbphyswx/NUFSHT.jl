Test.@testset "nusht_type2! accuracy: synthesis vs sph_evaluate at CC grid" begin
    Random.seed!(123)

    lmax = 20
    N = lmax + 1
    Nφ = 2lmax + 1

    pts = FastSphericalHarmonics.sph_points(N)
    θ_nodes = vec([θ for θ in pts[1], φ in pts[2]])
    φ_nodes = vec([φ for θ in pts[1], φ in pts[2]])

    plan = NUFSHT.make_plan(θ_nodes, φ_nodes, lmax; tol=1e-10)

    C_true = zeros(N, Nφ)
    for ℓ in 0:min(5, lmax), m in -ℓ:ℓ
        C_true[FastSphericalHarmonics.sph_mode(ℓ, m)] = randn()
    end

    f_exact = vec(FastSphericalHarmonics.sph_evaluate(C_true))

    f_synth = zeros(length(θ_nodes))
    NUFSHT.nusht_type2!(f_synth, C_true, plan)

    rms_f = sqrt(Statistics.mean(abs2.(f_exact)))
    rms_err = sqrt(Statistics.mean(abs2.(f_synth .- f_exact)))
    @info "nusht_type2! vs sph_evaluate at CC grid (lmax=$lmax): rms_err=$rms_err"
    Test.@test rms_err < 1e-9 * rms_f
end
