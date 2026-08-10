"""
    NUFSHTNonuniformFFTsExt

The NonuniformFFTs.jl implementation of NUFSHT's NUFFT seam, selected by
`nufft = NonuniformFFTsBackend()`. Loaded by `using NonuniformFFTs`.

Conventions are matched to the seam (which follows FINUFFT's, so every backend agrees):
`exec_type1!`/`exec_type2!` carry the `e^{-ikx}` / `e^{+ikx}` signs the plans are built with, and the
mode layout follows `modeord` — `1` is FFTW order (scalar plan), `0` is CMCL-centered (spin plan),
which is NonuniformFFTs' `fftshift = true`.

`tol` maps to a kernel half-support: for the default Kaiser–Bessel kernel at oversampling `σ = 2` the
relative error is roughly `10^-(m+1)`, so `m ≈ -log10(tol) - 1`, clamped to what the kernels support.
Correctness is checked against the dependency-free direct-sum backend, not against FINUFFT, so the
two fast backends are validated independently.
"""
module NUFSHTNonuniformFFTsExt

using NUFSHT: NUFSHT
using NonuniformFFTs: NonuniformFFTs

#=
    This should be nice because it's pure Julia.
    Esp on the GPU side, we can free ourselves from cufinufft being CUDA only, etc, and play nicer with general KA (not sure we if need a shared interface extnsion or how that works)
=#

# NonuniformFFTs is parameterised by kernel half-support, not by a tolerance, so `tol` has to be
# translated. These are measured L² errors for the default kernel at σ = 2 (the package's own accuracy
# methodology: type-1 against an exact direct sum), indexed by m = 2:8. Accuracy floors near 1e-14, so
# a larger half-support past that only costs spreading work.
const _HALFSUPPORT_ERROR = (4.84e-4, 4.45e-6, 4.52e-8, 5.16e-10, 8.76e-12, 2.31e-13, 1.02e-14)

@inline function _halfsupport(tol)
    i = findfirst(<=(tol), _HALFSUPPORT_ERROR)
    return NonuniformFFTs.HalfSupport(isnothing(i) ? length(_HALFSUPPORT_ERROR) + 1 : i + 1)
end

"""
    NonuniformFFTsPlan{P,T}

NUFSHT-owned handle wrapping a `PlanNUFFT`. Owning the handle (rather than dispatching the seam on
NonuniformFFTs' own type) keeps every reference to the library inside function bodies, so this
extension precompiles regardless of which of the library's symbols exist at precompile time.

`type` and the buffers are carried alongside because the seam's `_nufft_exec!` is direction-agnostic:
NonuniformFFTs splits it into `exec_type1!`/`exec_type2!`.
"""
struct NonuniformFFTsPlan{P, V}
    plan::P
    type::Int
    ntrans::Int
    scratch::V          # (M, ntrans) strengths view buffer, as NonuniformFFTs wants per-transform vectors
end

function NUFSHT._nufft_makeplan(::NUFSHT.NonuniformFFTsBackend, nodes::AbstractVector, type, n_modes,
                                iflag, ntrans, tol; dtype = Float64, modeord = 0, kwargs...)
    Ns = (Int(n_modes[1]), Int(n_modes[2]))
    plan = NonuniformFFTs.PlanNUFFT(Complex{dtype}, Ns;
                            ntransforms = Int(ntrans),
                            m = _halfsupport(tol),
                            fftshift = (modeord == 0))
    scratch = similar(nodes, Complex{dtype}, length(nodes) * Int(ntrans))
    return NonuniformFFTsPlan{typeof(plan), typeof(scratch)}(plan, Int(type), Int(ntrans), scratch)
end

NUFSHT._nufft_setpts!(p::NonuniformFFTsPlan, x, y) = (NonuniformFFTs.set_points!(p.plan, (x, y)); p)

# NonuniformFFTs takes one vector per transform; NUFSHT stores strengths as a flat (M, ntrans) block.
@inline function _as_transforms(v, M, ntrans)
    return ntuple(b -> view(vec(v), ((b - 1) * M + 1):(b * M)), ntrans)
end
@inline function _as_modes(u, n1, n2, ntrans)
    return ntuple(b -> reshape(view(vec(u), ((b - 1) * n1 * n2 + 1):(b * n1 * n2)), n1, n2), ntrans)
end

function NUFSHT._nufft_exec!(p::NonuniformFFTsPlan, input, output)
    n1, n2 = size(p.plan)
    if p.type == 2
        M = length(output) ÷ p.ntrans
        NonuniformFFTs.exec_type2!(_as_transforms(output, M, p.ntrans), p.plan,
                           _as_modes(input, n1, n2, p.ntrans))
    else
        M = length(input) ÷ p.ntrans
        NonuniformFFTs.exec_type1!(_as_modes(output, n1, n2, p.ntrans), p.plan,
                           _as_transforms(input, M, p.ntrans))
    end
    return output
end

NUFSHT._nufft_destroy!(::NonuniformFFTsPlan) = nothing
NUFSHT._nufft_finalize!(p::NonuniformFFTsPlan) = p

end # module NUFSHTNonuniformFFTsExt
