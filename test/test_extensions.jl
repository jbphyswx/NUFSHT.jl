# Asserts the CUDA package extension precompiles and loads. CUDA.jl loads without a GPU
# (`CUDA.functional() == false`), so this runs on ordinary CI; it does not exercise the device
# transform (that needs hardware — see test/gpu_cuda.jl).
using CUDA: CUDA

Test.@testset "package extension NUFSHTCUDAExt loads" begin
    Test.@test Base.get_extension(NUFSHT, :NUFSHTCUDAExt) !== nothing
    Test.@test hasmethod(NUFSHT._nufft_makeplan, Tuple{CUDA.CuArray, Int, Vector{Int64}, Int, Int, Float64})
    @info "NUFSHTCUDAExt loaded (CUDA.functional() = $(CUDA.functional()))"
end
