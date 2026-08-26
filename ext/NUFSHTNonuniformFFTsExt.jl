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
    NonuniformFFTsPlan{P,Nc}

NUFSHT-owned handle wrapping a `PlanNUFFT`. Owning the handle (rather than dispatching the seam on
NonuniformFFTs' own type) keeps every reference to the library inside function bodies, so this
extension precompiles regardless of which of the library's symbols exist at precompile time.

`type` is carried alongside because the seam's `_nufft_exec!` is direction-agnostic while NonuniformFFTs
splits it into `exec_type1!`/`exec_type2!`. That is also why `plan` can be **shared** between the two
directions: a `PlanNUFFT` encodes no direction — the sign comes from which exec function is called — so
one of them serves both and only `type` differs here. See `NUFSHT._nufft_share_directions`.

`Nc` is the transform count, a type parameter rather than a field: `exec_type1!`/`exec_type2!` take one
array per transform, so every call builds a tuple of views whose length is that count, and a count read
from a field leaves the length unknown to inference — the views box and the library call becomes a
dynamic dispatch. `PlanNUFFT` already carries the count in its own type (`to_static`), so naming it here
costs no specialisation that did not already exist.
"""
struct NonuniformFFTsPlan{P, Nc}
    plan::P
    type::Int
    derived::Bool     # wraps another handle's plan, so its points are already set
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

"""
    _unblocked(nthreads) -> Bool

Whether to build on the library's `block_size = nothing` path. Two thread counts are reachable and no
others: the blocked path fixes its width at `Threads.nthreads()` when the plan is built (`blocking/cpu.jl`,
one buffer and one block range per thread, no keyword), and the unblocked path spreads from a plain
serial loop over the points. So `nthreads` is honoured at `1` and at whatever Julia is running, and
anything else is refused rather than silently dropped.
"""
function _unblocked(nthreads)
    (isnothing(nthreads) || nthreads == 0 || nthreads == Threads.nthreads()) && return false
    nthreads == 1 || throw(ArgumentError(
        "NonuniformFFTs reaches only two thread counts: 1, or the `Threads.nthreads()` it is running " *
        "under ($(Threads.nthreads())). Got nthreads = $nthreads. `0` takes the default; " *
        "`nufft = FINUFFTBackend()` takes an arbitrary count."))
    return true
end

function NUFSHT._nufft_makeplan(::NUFSHT.NonuniformFFTsBackend, nodes::AbstractVector, type, n_modes,
                                iflag, ntrans, tol; dtype = Float64, strengths = Complex{dtype},
                                modeord = 0, upsampfac = nothing, nthreads = nothing, kwargs...)
    Ns = (Int(n_modes[1]), Int(n_modes[2]))
    σ = isnothing(upsampfac) ? _DEFAULT_σ : Float64(upsampfac)
    kw = (; ntransforms = Int(ntrans),
            m = _halfsupport(tol, σ),
            σ = σ,
            fftshift = (modeord == 0),
            sort_points = NonuniformFFTs.True())
    plan = _unblocked(nthreads) ?
        NonuniformFFTs.PlanNUFFT(strengths, Ns; kw..., block_size = nothing) :
        NonuniformFFTs.PlanNUFFT(strengths, Ns; kw...)
    return NonuniformFFTsPlan{typeof(plan), Int(ntrans)}(plan, Int(type), false)
end

# One `PlanNUFFT` serves both directions: it carries no direction of its own, and the two are never
# executed concurrently (a plan already carries that restriction). Sharing it halves the oversampled
# grid and the sorted point copy a plan owns.
NUFSHT._nufft_share_directions(::NUFSHT.NonuniformFFTsBackend) = true
NUFSHT._nufft_as_type1(p::NonuniformFFTsPlan{P,Nc}) where {P,Nc} =
    NonuniformFFTsPlan{P,Nc}(p.plan, 1, true)
NUFSHT._nufft_derived(p::NonuniformFFTsPlan) = p.derived

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
    length(q) ≥ 3 || return Q          # `NullBlockData` carries no count: an unblocked plan's is here
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

_wrapper_at_width(::Type{NonuniformFFTsPlan{P,Nc}}, k::Int) where {P,Nc} =
    NonuniformFFTsPlan{_plan_at_width(P, k), k}

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

# NonuniformFFTs takes one array per transform; NUFSHT stores both sides as one flat block. The count
# comes in as `Val` so the tuple length is known at compile time and the views stay unboxed.
@inline function _as_transforms(v, M, ::Val{Nc}) where {Nc}
    return ntuple(b -> view(vec(v), ((b - 1) * M + 1):(b * M)), Val(Nc))
end
@inline function _as_modes(u, n1, n2, ::Val{Nc}) where {Nc}
    return ntuple(b -> reshape(view(vec(u), ((b - 1) * n1 * n2 + 1):(b * n1 * n2)), n1, n2), Val(Nc))
end

function NUFSHT._nufft_exec!(p::NonuniformFFTsPlan{P,Nc}, input, output) where {P,Nc}
    n1, n2 = size(p.plan)
    if p.type == 2
        M = length(output) ÷ Nc
        NonuniformFFTs.exec_type2!(_as_transforms(output, M, Val(Nc)), p.plan,
                           _as_modes(input, n1, n2, Val(Nc)))
    else
        M = length(input) ÷ Nc
        NonuniformFFTs.exec_type1!(_as_modes(output, n1, n2, Val(Nc)), p.plan,
                           _as_transforms(input, M, Val(Nc)))
    end
    return output
end

NUFSHT._nufft_destroy!(::NonuniformFFTsPlan) = nothing
NUFSHT._nufft_finalize!(p::NonuniformFFTsPlan) = p

end # module NUFSHTNonuniformFFTsExt
