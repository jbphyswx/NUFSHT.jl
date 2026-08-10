"""
    NUFSHTFINUFFTExt

The FINUFFT implementation of NUFSHT's NUFFT seam, selected by `nufft = FINUFFTBackend()` (or by
`AutoSpectralBackend`, which prefers it when loaded). Loaded by `using FINUFFT`.

Plan creation dispatches on the node array type, so a `CuArray` node set selects cuFINUFFT — that
method lives in `NUFSHTCUDAExt`, since cuFINUFFT's plan type is only available once CUDA is loaded.
Execution, `setpts!` and teardown dispatch on the returned plan handle.

FINUFFT and cuFINUFFT share `setkwopts!`, so identical keyword options (`dtype`/`modeord`/`nthreads`/
`upsampfac`) forward to both — the seam passes them through unchanged.
"""
module NUFSHTFINUFFTExt

using NUFSHT: NUFSHT
using FINUFFT: FINUFFT

#=
    see https://finufft.readthedocs.io/en/stable/tutorial/realinterp1d.html, ensure we are being efficient for real transforms
     (I think this is directly analogous to what https://github.com/jipolanco/NonuniformFFTs.jl exploits for real data, e.g.
     https://jipolanco.github.io/NonuniformFFTs.jl/stable/benchmarks#Real-non-uniform-data)

    STATUS: not yet done. The transform is real on both ends — `Fhat` is the FFT of a real doubled
    field (so Hermitian) and `_copy_real!` discards the imaginary part of the output — so the θ mode
    axis can be folded and the result taken as 2·Re(...), halving the mode count handed to FINUFFT.
    That shrinks FINUFFT's internal upsampled FFT but not its spreading, so the payoff depends on the
    M-vs-modes balance and must be measured per regime before it is turned on.
=#

NUFSHT._nufft_makeplan(::NUFSHT.FINUFFTBackend, ::AbstractVector, type, n_modes, iflag, ntrans, tol;
                       kwargs...) =
    FINUFFT.finufft_makeplan(type, n_modes, iflag, ntrans, tol; kwargs...)

# Host FINUFFT needs host coordinate vectors; `_host` is a no-op for an `Array`.
NUFSHT._nufft_setpts!(p::FINUFFT.finufft_plan, x, y) =
    FINUFFT.finufft_setpts!(p, NUFSHT._host(x), NUFSHT._host(y))

NUFSHT._nufft_exec!(p::FINUFFT.finufft_plan, input, output) =
    FINUFFT.finufft_exec!(p, input, output)

NUFSHT._nufft_destroy!(p::FINUFFT.finufft_plan) = FINUFFT.finufft_destroy!(p)

# Attach the GC finalizer through the seam, so a device plan handle can forward it to its wrapped
# (mutable) cuFINUFFT plan; the host plan is itself mutable.
NUFSHT._nufft_finalize!(p::FINUFFT.finufft_plan) = (finalizer(NUFSHT._nufft_destroy!, p); p)

end # module NUFSHTFINUFFTExt
