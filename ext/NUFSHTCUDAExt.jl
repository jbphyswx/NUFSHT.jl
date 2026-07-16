"""
    NUFSHTCUDAExt

The single CUDA-specific piece of the device path: the **cuFINUFFT** NUFFT binding. Everything else
that runs on device is vendor-agnostic — the DFS kernels are KernelAbstractions `@kernel`s dispatched
on `AbstractGPUArray` (NUFSHTKernelAbstractionsExt), the FFT (scalar path) goes through `AbstractFFTs`,
and the plan buffers are allocated `similar` to the (device) node arrays. Only the NUFFT has no
device fallback, so it is bound per-vendor here; a ROCm/Metal NUFFT would be an analogous one-method
extension.

A `CuArray` node set selects cuFINUFFT via the creation seam (`_nufft_makeplan`); execution and
teardown dispatch on the returned `cufinufft_plan`. cuFINUFFT and FINUFFT share `setkwopts!`, so the
same keyword options (`dtype`/`modeord`/`nthreads`) forward unchanged (cuFINUFFT ignores `nthreads`).
Loaded by `using CUDA`.
"""
module NUFSHTCUDAExt

using NUFSHT: NUFSHT
using FINUFFT: FINUFFT
using CUDA: CUDA

NUFSHT._nufft_makeplan(::CUDA.CuArray, type, n_modes, iflag, ntrans, tol; kwargs...) =
    FINUFFT.cufinufft_makeplan(type, n_modes, iflag, ntrans, tol; kwargs...)
NUFSHT._nufft_setpts!(p::FINUFFT.cufinufft_plan, x, y) = (FINUFFT.cufinufft_setpts!(p, x, y); p)
NUFSHT._nufft_exec!(p::FINUFFT.cufinufft_plan, input, output) =
    (FINUFFT.cufinufft_exec!(p, input, output); output)
NUFSHT._nufft_destroy!(p::FINUFFT.cufinufft_plan) = FINUFFT.cufinufft_destroy!(p)

end # module NUFSHTCUDAExt
