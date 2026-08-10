# Asserts the CUDA package extension precompiles and loads. CUDA.jl loads without a GPU
# (`CUDA.functional() == false`), so this runs on ordinary CI; it does not exercise the device
# transform (that needs hardware — see test/gpu_cuda.jl).
using CUDA: CUDA

Test.@testset "package extension NUFSHTcuFINUFFTExt loads" begin
    Test.@test Base.get_extension(NUFSHT, :NUFSHTcuFINUFFTExt) !== nothing
    Test.@test hasmethod(NUFSHT._nufft_makeplan,
                         Tuple{NUFSHT.FINUFFTBackend, CUDA.CuArray, Int, Vector{Int64}, Int, Int, Float64})
    @info "NUFSHTcuFINUFFTExt loaded (CUDA.functional() = $(CUDA.functional()))"
end
