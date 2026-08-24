"""
    NUFSHTNonuniformFFTsExt

The NonuniformFFTs.jl implementation of NUFSHT's NUFFT seam, selected by
`nufft = NonuniformFFTsBackend()`. Loaded by `using NonuniformFFTs`.

Conventions are matched to the seam (which follows FINUFFT's, so every backend agrees):
`exec_type1!`/`exec_type2!` carry the `e^{-ikx}` / `e^{+ikx}` signs the plans are built with, and the
mode layout follows `modeord` — `1` is FFTW order (scalar plan), `0` is CMCL-centered (spin plan),
which is NonuniformFFTs' `fftshift = true`.

`tol` is converted to a kernel half-support analytically (see [`_halfsupport`](@ref)) rather than
looked up, so it stays exact at any oversampling factor. `σ` is NUFSHT's `upsampfac` — the same knob
FINUFFT exposes under that name — and the half-support is re-derived from it, so overriding the
oversampling changes the cost of reaching `tol` and never `tol` itself.

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

"""
    _halfsupport(tol, σ) -> HalfSupport

The kernel half-support that reaches relative accuracy `tol` at oversampling factor `σ`. Analytic, not
tabulated: NonuniformFFTs' default backwards-Kaiser–Bessel kernel takes the Potts & Steidl shape
`β = πγM(2 - 1/σ)` (`kaiser_bessel_backwards.jl`; Potts & Steidl 2003 eq. 5.12), for which the
aliasing error is

    ε(σ, M) = exp(-πM·√((2 - 1/σ)² - 1/σ²)) = exp(-2πM·√(1 - 1/σ)),

the second form because `(2 - 1/σ)² - 1/σ² = 4(1 - 1/σ)` identically. Inverting,

    M(σ, tol) = ⌈-ln(tol) / (2π·√(1 - 1/σ))⌉.

So `tol`, `σ` and the half-support are exchangeable by construction, and `σ` selects *how* a tolerance
is paid for — a smaller oversampling shrinks the FFT and grows the spreading half-support — never how
accurate the result is. The `γ` safety factor the kernel applies makes the realised error slightly
better than the estimate, so this rounds up rather than down.

Below double precision's `eps` there is nothing left to resolve, so `tol` is floored there; the library
itself rejects a half-support too large for the grid (`σN ≥ 2M`).
"""
@inline function _halfsupport(tol, σ)
    σ > 1 || throw(ArgumentError("oversampling factor must satisfy σ > 1, got $σ"))
    ε = max(Float64(tol), eps(Float64))
    m = ceil(Int, -log(ε) / (2π * sqrt(1 - 1 / Float64(σ))))
    return NonuniformFFTs.HalfSupport(max(m, 2))
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

# `strengths` is the non-uniform data type, forwarded straight to `PlanNUFFT` — a real one selects the
# real-data transform, which halves the FFT *and* the spreading/interpolation because the strengths are
# real too (the gain this library's benchmarks report, and unavailable in FINUFFT, which has no real
# transform). `fftshift` does not apply to the real axis: `non_oversampled_indices!`'s `r2c` branch
# returns axis 1 as `0 … n₁÷2` ascending while the rest stay centered, which is the folded layout
# `_assemble_modes!` writes.
# `upsampfac` is NUFSHT's name for the oversampling factor; it reaches here only when the caller (or
# tuning) set one, so an unset plan takes the library's own default. Whatever it is, the half-support is
# derived from it, so the requested `tol` is met at every σ.
#
# `sort_points = True()` permutes the point data once, in `set_points!`, to make the spreading and
# interpolation memory accesses local. Its cost is paid per point-set and its benefit per transform,
# which is exactly NUFSHT's shape: a plan sets its points once and is then executed repeatedly — twice
# per LSMR iteration, hundreds of times per solve. Measured here at iso-accuracy and one thread: 1.08x
# at lmax 64 / M 1e4 and 1.26x at lmax 128 / M 2e5, with bit-identical output.
const _DEFAULT_σ = 2.0

function NUFSHT._nufft_makeplan(::NUFSHT.NonuniformFFTsBackend, nodes::AbstractVector, type, n_modes,
                                iflag, ntrans, tol; dtype = Float64, strengths = Complex{dtype},
                                modeord = 0, upsampfac = nothing, kwargs...)
    Ns = (Int(n_modes[1]), Int(n_modes[2]))
    σ = isnothing(upsampfac) ? _DEFAULT_σ : Float64(upsampfac)
    plan = NonuniformFFTs.PlanNUFFT(strengths, Ns;
                            ntransforms = Int(ntrans),
                            m = _halfsupport(tol, σ),
                            σ = σ,
                            fftshift = (modeord == 0),
                            sort_points = NonuniformFFTs.True())
    scratch = similar(nodes, strengths, length(nodes) * Int(ntrans))
    return NonuniformFFTsPlan{typeof(plan), typeof(scratch)}(plan, Int(type), Int(ntrans), scratch)
end

NUFSHT._nufft_setpts!(p::NonuniformFFTsPlan, x, y) = (NonuniformFFTs.set_points!(p.plan, (x, y)); p)

# ── Reduced-width plan slots ────────────────────────────────────────────────────
# `PlanNUFFT` carries the transform count in its type: `to_static(ntransforms)` lifts a runtime value
# into a type parameter, which reappears inside its `Real`/`ComplexNUFFTData` and `BlockDataCPU` fields
# (each holding `NTuple{Nc,…}` members). Plans at different widths are therefore unrelated types, and
# the default on-demand `Vector` cannot hold them: a store grown on demand has to name its element type
# before it has anything to name it from, and here that type follows from a *value*, so inference
# cannot supply it either.
#
# It can be derived, though, and naming a type constructs nothing and specialises nothing. So each
# width gets a slot typed up front and filled on first use — the slots for a whole ladder cost ~4 KiB
# against ~25 MiB for a single plan, so a width nothing narrows to costs neither a plan nor a
# specialisation. Laziness matters more here than for a backend whose count is a runtime field:
# a width in the type means every new one specialises this backend's whole spreading path.

# Rewrite an exact `NTuple` / `Vector{NTuple}` carrying the old count; anything else passes through.
_retype_count(@nospecialize(x), old::Int, k::Int) = x
_retype_count(::Type{NTuple{N,X}}, old::Int, k::Int) where {N,X} = N == old ? NTuple{k,X} : NTuple{N,X}
_retype_count(::Type{Vector{NTuple{N,X}}}, old::Int, k::Int) where {N,X} =
    N == old ? Vector{NTuple{k,X}} : Vector{NTuple{N,X}}

# The count is parameter 3 of the data and block types; their `NTuple` members sit at 5 and beyond.
# Position 4 is skipped deliberately: `RealNUFFTData` keeps a `D`-tuple of identical `Frequencies`
# there, and `Tuple{Frequencies,Frequencies}` *is* an `NTuple{2,…}` — so a by-shape rule would rewrite
# the dimensionality whenever the batch width happened to equal it.
function _inner_at_width(@nospecialize(Q::Type), old::Int, k::Int)
    q = collect(Q.parameters)
    q[3] = k
    for i in 5:length(q)
        q[i] = _retype_count(q[i], old, k)
    end
    return Base.typename(Q).wrapper{q...}
end

"""
    _plan_at_width(P, k) -> Type

The width-`k` counterpart of `PlanNUFFT` type `P`, obtained without constructing anything. Pinned in
the test suite against freshly built plans over both strengths types and a range of widths, so a change
to the upstream parameter layout fails loudly rather than yielding a slot that cannot hold what is put
into it.
"""
function _plan_at_width(@nospecialize(P::Type), k::Int)
    p = collect(P.parameters)
    old = p[3]::Int
    old == k && return P
    p[3] = k
    p[10] = _inner_at_width(p[10], old, k)      # Real/ComplexNUFFTData
    p[11] = _inner_at_width(p[11], old, k)      # BlockDataCPU
    return Base.typename(P).wrapper{p...}
end

# The wrapper's scratch buffer is sized at run time, so only the inner plan type varies with the width.
_wrapper_at_width(::Type{NonuniformFFTsPlan{P,V}}, k::Int) where {P,V} =
    NonuniformFFTsPlan{_plan_at_width(P, k), V}

# One empty, concretely typed slot per width. Nothing is built here — `_pool_lookup!` fills a slot the
# first time a solve narrows to it, and it stays filled for the plan's life.
function NUFSHT._nufft_size_pool(::NUFSHT.NonuniformFFTsBackend, nufft_type2, nufft_type1,
                                 widths::AbstractVector{Int}, _build)
    return ntuple(length(widths)) do i
        k = widths[i]
        E = NamedTuple{(:k, :nufft_type2, :nufft_type1),
                       Tuple{Int, _wrapper_at_width(typeof(nufft_type2), k),
                             _wrapper_at_width(typeof(nufft_type1), k)}}
        (k = k, pair = Ref{Union{Nothing,E}}(nothing))
    end
end

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
