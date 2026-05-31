using NUFSHT: NUFSHT

Test.@testset "DFS doubling and folding" begin
    Nθ, Nφ = 8, 16
    F = randn(Nθ, Nφ)

    F̃ = NUFSHT.dfs_double(F)
    Test.@test size(F̃) == (2Nθ, Nφ)

    Test.@test F̃[1:Nθ, :] ≈ F

    F_back = NUFSHT.dfs_fold(F̃)
    Test.@test size(F_back) == (Nθ, Nφ)
    Test.@test F_back ≈ 2 .* F
end
