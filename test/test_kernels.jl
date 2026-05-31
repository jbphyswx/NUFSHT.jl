Test.@testset "Kernel transfer functions" begin
    Test.@testset "TopHat" begin
        f = NUFSHT.TopHatTransfer(10)
        Test.@test NUFSHT.kernel_transfer(f, 0)  == 1.0
        Test.@test NUFSHT.kernel_transfer(f, 10) == 1.0
        Test.@test NUFSHT.kernel_transfer(f, 11) == 0.0
        Test.@test NUFSHT.kernel_transfer(f, 100) == 0.0
    end

    Test.@testset "Gaussian" begin
        f = NUFSHT.gaussian_from_scale(200e3)
        Test.@test NUFSHT.kernel_transfer(f, 0) ≈ 1.0
        h1   = NUFSHT.kernel_transfer(f, 1)
        h10  = NUFSHT.kernel_transfer(f, 10)
        h100 = NUFSHT.kernel_transfer(f, 100)
        Test.@test h1 > h10 > h100 > 0
        Test.@test h100 < 1e-2
    end

    Test.@testset "cutoff_degree" begin
        L = NUFSHT.cutoff_degree(100e3)
        Test.@test L ≈ round(Int, π * 6.371e6 / 100e3)
        Test.@test L > 0
    end
end
