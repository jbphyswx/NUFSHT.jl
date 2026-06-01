Test.@testset "Constant field filter" begin
    lmax = 10

    pts  = FastSphericalHarmonics.sph_points(lmax + 1)
    θ = vec([θ for θ in pts[1], φ in pts[2]])
    φ = vec([φ for θ in pts[1], φ in pts[2]])

    plan = NUFSHT.make_plan(θ, φ, lmax)

    Test.@testset "Identity filter (no truncation)" begin
        f     = ones(length(θ))
        f_out = similar(f)

        filter = NUFSHT.TopHatTransfer(lmax)
        NUFSHT.nusht_filter!(f_out, f, filter, plan)

        Test.@test maximum(abs.(f_out .- 1.0)) < 0.05
    end

    Test.@testset "Low-pass filter preserves constant" begin
        f     = ones(length(θ))
        f_out = similar(f)

        filter = NUFSHT.TopHatTransfer(lmax ÷ 2)
        NUFSHT.nusht_filter!(f_out, f, filter, plan)

        Test.@test maximum(abs.(f_out .- 1.0)) < 0.1
    end

    Test.@testset "Gaussian filter preserves constant" begin
        f     = ones(length(θ))
        f_out = similar(f)

        filter = NUFSHT.gaussian_from_scale(500e3)
        NUFSHT.nusht_filter!(f_out, f, filter, plan)

        Test.@test maximum(abs.(f_out .- 1.0)) < 0.1
    end

    Test.@testset "filter_renorm! corrects mask bias on constant field" begin
        f    = ones(length(θ))
        mask = zeros(length(θ))
        mask[1:2:end] .= 1.0   # mask every other point as "ocean"

        f_masked = f .* mask
        f_out    = similar(f)
        filter   = NUFSHT.gaussian_from_scale(500e3)

        NUFSHT.nusht_filter!(f_out, f_masked, filter, plan)
        NUFSHT.nusht_filter_renorm!(f_out, mask, filter, plan)

        ocean_pts = findall(mask .> 0.5)
        Test.@test maximum(abs.(f_out[ocean_pts] .- 1.0)) < 0.15
    end
end
