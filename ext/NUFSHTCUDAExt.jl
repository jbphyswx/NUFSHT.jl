"""
    NUFSHTCUDAExt

The single CUDA-specific piece of the device path: the **cuFINUFFT** NUFFT binding. Everything else
that runs on device is vendor-agnostic — the DFS kernels are KernelAbstractions `@kernel`s dispatched
on `AbstractGPUArray` (NUFSHTKernelAbstractionsExt), the FFT (scalar path) goes through `AbstractFFTs`,
and the plan buffers are allocated `similar` to the (device) node arrays. Only the NUFFT has no
device fallback, so it is bound per-vendor here; a ROCm/Metal NUFFT would be an analogous one-method
extension.

A `CuArray` node set selects cuFINUFFT via the creation seam (`_nufft_makeplan`); execution and
teardown dispatch on the returned plan handle. `dtype`/`modeord` forward to cuFINUFFT unchanged;
`nthreads` is dropped here because `cufinufft_opts` has no such field. Loaded by `using CUDA`.
"""
module NUFSHTCUDAExt

using NUFSHT: NUFSHT
using FINUFFT: FINUFFT
using CUDA: CUDA

# cuFINUFFT's guru plan is exposed differently across FINUFFT versions: a `Requires`-loaded symbol in
# 3.5.0 (defined only at runtime `__init__`, so it cannot be named in a method signature at precompile
# time) vs. a package-extension type in 3.5.2+. Naming it directly (`p::FINUFFT.cufinufft_plan`) makes
# this extension fail to precompile against 3.5.0. Instead wrap the returned plan in a NUFSHT-owned,
# type-parameterized handle and dispatch the exec/setpts/teardown seam on the handle; every
# `FINUFFT.cufinufft_*` reference lives in a function body, resolved at runtime (after `using CUDA` has
# loaded FINUFFT's cuFINUFFT interface, by either mechanism). The handle is immutable — the GC finalizer
# is attached to the wrapped cuFINUFFT plan, which is itself mutable.
struct CuNUFFTPlan{P}
    plan::P
end

# Custom one-line show (like `NUSHTplan`): never let the default field-wise printer recurse into the
# underlying cuFINUFFT plan (a foreign handle whose printer is not safe to call from Julia).
Base.show(io::IO, ::CuNUFFTPlan) = print(io, "CuNUFFTPlan(cuFINUFFT)")

NUFSHT._nufft_makeplan(::CUDA.CuArray, type, n_modes, iflag, ntrans, tol; nthreads = nothing, kwargs...) =
    CuNUFFTPlan(FINUFFT.cufinufft_makeplan(type, n_modes, iflag, ntrans, tol; kwargs...))
NUFSHT._nufft_setpts!(p::CuNUFFTPlan, x, y) = (FINUFFT.cufinufft_setpts!(p.plan, x, y); p)
NUFSHT._nufft_exec!(p::CuNUFFTPlan, input, output) = (FINUFFT.cufinufft_exec!(p.plan, input, output); output)
NUFSHT._nufft_destroy!(p::CuNUFFTPlan) = FINUFFT.cufinufft_destroy!(p.plan)
NUFSHT._nufft_finalize!(p::CuNUFFTPlan) = (finalizer(q -> FINUFFT.cufinufft_destroy!(q), p.plan); p)

end # module NUFSHTCUDAExt
