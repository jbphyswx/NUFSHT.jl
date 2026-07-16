using Test: Test
using Statistics: Statistics
using Random: Random
using FastSphericalHarmonics: FastSphericalHarmonics
using Aqua: Aqua
using NUFSHT: NUFSHT

Test.@testset "NUFSHT.jl" begin
    Test.@testset "Aqua quality checks" begin
        Aqua.test_all(NUFSHT; ambiguities=false, deps_compat=(check_extras=false,))
    end
    include("test_dfs.jl")
    include("test_constant.jl")
    include("test_kernels.jl")
    include("test_roundtrip.jl")
    include("test_synthesis.jl")
    include("test_solve.jl")
    include("test_spin.jl")
    include("test_precision.jl")
    include("test_adjoint.jl")
    include("test_ka.jl")
    include("test_threaded.jl")
    include("test_distributed.jl")
end
