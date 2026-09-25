"""
    NUFFT.jl — the `N` step of `A = N·F·S`, behind a backend seam.

The transform cores and plan builders never name a NUFFT library: they call `_nufft_makeplan`,
`_nufft_setpts!`, `_nufft_exec!` and `_nufft_destroy!`. Plan creation dispatches on the backend and
takes the nodes, whose array type picks host or device within it; execution, point setting and teardown
dispatch on the returned handle.

Backends are `SpectralBackends` types:

- `DirectSumSpectralBackend` — implemented here. No dependencies, always available, `O(M·K)`: the
  correctness reference the fast backends are validated against. Threaded over the axis each direction
  writes, so `nthreads` means there what it means elsewhere.
- `FlowTransformBindings.NonuniformFFTsBackend()` (`using NonuniformFFTs`) and
  `FlowTransformBindings.FINUFFTBackend()` (`using FINUFFT`, cuFINUFFT on `CuArray` nodes): the plans of
  FlowTransformBindings, which give both a real-data transform.
"""

using SpectralBackends: SpectralBackends

const _FTBLibrary = Union{FTB.FINUFFTBackend, FTB.NonuniformFFTsBackend}

"""
    _width_narrowable(backend) -> Bool
    _width_polymorphic(backend) -> Bool

Whether a batched solve can narrow the backend's NUFFT to its live columns once the rest have retired,
and whether its plans share one concrete type at every transform count. FlowTransformBindings'
FINUFFT plan holds the count at runtime, so one pool element type serves every width. Its NonuniformFFTs
plan carries the count in the library's type, so the pool holds one slot per width, typed by
`FlowTransformBindings.plan_type` and filled on first use.

Correctness never depends on either. The per-column sphere loop and the solver's vector reductions
narrow on every backend.
"""
_width_narrowable(::SpectralBackends.AbstractSpectralBackend) = false
_width_narrowable(::_FTBLibrary) = true
_width_polymorphic(::SpectralBackends.AbstractSpectralBackend) = false
_width_polymorphic(::FTB.FINUFFTBackend) = true

"""
    _nufft_share_directions(backend) -> Bool
    _nufft_as_type1(type2_handle) -> handle

Whether one plan serves both transform directions, and the type-1 handle onto the type-2 one when it
does. A FlowTransformBindings plan evaluates both, so the analysis handle reads the synthesis plan in the
other direction and holds no second plan.
"""
_nufft_share_directions(::SpectralBackends.AbstractSpectralBackend) = false
_nufft_share_directions(::_FTBLibrary) = true
_nufft_as_type1(p) = throw(ArgumentError(
    "this backend does not share one plan between directions; build the type-1 plan directly"))

"""
    _nufft_derived(p) -> Bool

Whether this handle reads a plan another handle owns, so its points are set whenever that one's are.
`false` by default. [`set_nodes!`](@ref) re-points only the handle that owns the plan.
"""
_nufft_derived(p) = false

"""
    _real_capable(backend) -> Bool

Whether the backend implements a **real-data** transform: real non-uniform values, and the uniform side
stored as the half-spectrum a real signal determines. `false` by default.

A real field's mode array is exactly Hermitian, so every plan stores only its `kθ ≥ 0` half. A backend
with a real-data transform reconstructs the conjugate half itself and takes real strengths, halving the
spreading and interpolation as well as the FFT; FlowTransformBindings gives one on both libraries.
Without one the transform is a half-height complex one, whose missing half `_fold_weights!` pays for.

Correctness never depends on this: the folded and full assemblies represent the same series.
"""
_real_capable(::SpectralBackends.AbstractSpectralBackend) = false
_real_capable(::_FTBLibrary) = true

"""
    _resolve_nufft(backend, FE) -> backend

Concrete backends pass through untouched — one named explicitly is honoured or refused, never swapped.
`AutoSpectralBackend` takes NonuniformFFTs when it is loaded, FINUFFT when only it is, and direct
summation when neither is. Override with `nufft=`.
"""
_resolve_nufft(backend::SpectralBackends.AbstractSpectralBackend, ::Type) = backend
function _resolve_nufft(::SpectralBackends.AbstractAutoSpectralBackend, ::Type)
    FTB.is_available(FTB.NonuniformFFTsBackend()) && return FTB.NonuniformFFTsBackend()
    FTB.is_available(FTB.FINUFFTBackend()) && return FTB.FINUFFTBackend()
    return SpectralBackends.DirectSumSpectralBackend()
end

# `maxlog = 1` makes it once per session per logger, so `@test_logs` still sees it.
function _warn_if_directsum(requested, resolved, M::Integer, K::Integer)
    requested isa SpectralBackends.AbstractAutoSpectralBackend || return nothing
    resolved isa SpectralBackends.DirectSumSpectralBackend || return nothing
    @warn "no fast NUFFT backend loaded; using direct summation, $M x $K per transform. " *
          "`using NonuniformFFTs` or `using FINUFFT` to avoid." maxlog = 1
    return nothing
end

_nufft_makeplan(backend::SpectralBackends.AbstractSpectralBackend, nodes, type, n_modes, iflag,
                ntrans, tol; kwargs...) = throw(ArgumentError(
    "NUFSHT's NUFFT takes SpectralBackends.DirectSumSpectralBackend(), " *
    "FlowTransformBindings.NonuniformFFTsBackend() or FlowTransformBindings.FINUFFTBackend(); got " *
    "$(nameof(typeof(backend)))"))

# ── FlowTransformBindings ─────────────────────────────────────────────────────

"""
    _FTBNUFFT{P}

A FlowTransformBindings plan read in one direction: `type = 2` synthesis, `type = 1` its adjoint. The
type-1 handle of a plan is `derived`, reading the plan the type-2 handle owns.
"""
struct _FTBNUFFT{P<:FTB.AbstractNUFFTPlan}
    plan::P
    type::Int
    derived::Bool
end

Base.show(io::IO, h::_FTBNUFFT) = print(io, "_FTBNUFFT(type ", h.type, ", ", h.plan, ")")

# `strengths` is the element type of the non-uniform data, so a real one builds the real-data plan.
# The seam's thread count `0` and oversampling factor `0` take the backend's own choice; for
# FlowTransformBindings that is the session's `Threads.nthreads()` and the library's default.
function _nufft_makeplan(backend::_FTBLibrary, nodes::NTuple{2,AbstractVector}, type, n_modes, iflag,
                         ntrans, tol; dtype = Float64, strengths = Complex{dtype}, modeord = 0,
                         nthreads = nothing, upsampfac = nothing)
    iflag == (type == 2 ? 1 : -1) || throw(ArgumentError(
        "a type-$type NUFFT runs with iflag $(type == 2 ? 1 : -1); got $iflag"))
    modeord == 0 || throw(ArgumentError("NUFSHT's mode arrays are centered (modeord = 0); got $modeord"))
    nt = (nthreads === nothing || nthreads == 0) ? Threads.nthreads() : Int(nthreads)
    uf = (upsampfac === nothing || upsampfac == 0) ? nothing : Float64(upsampfac)
    p = FTB.plan_nufft(backend, strengths, nodes, (Int(n_modes[1]), Int(n_modes[2]));
                       ntrans = Int(ntrans), tol = tol, order = FTB.CenteredModes(), nthreads = nt,
                       upsampfac = uf)
    return _FTBNUFFT(p, Int(type), false)
end

_nufft_as_type1(h::_FTBNUFFT) = _FTBNUFFT(h.plan, 1, true)
_nufft_derived(h::_FTBNUFFT) = h.derived
_nufft_setpts!(h::_FTBNUFFT, x, y) = (FTB.set_nodes!(h.plan, (x, y)); h)

function _nufft_exec!(h::_FTBNUFFT, input, output)
    h.type == 2 ? FTB.nufft_type2!(output, h.plan, input) : FTB.nufft_type1!(output, h.plan, input)
    return output
end

# Idempotent, so closing a plan through both of its handles is safe.
_nufft_destroy!(h::_FTBNUFFT) = FTB.close!(h.plan)

# ── Direct summation ──────────────────────────────────────────────────────────

"""
    DirectSumNUFFTPlan{T,V}

Handle for the direct-summation backend: built once over its nodes, points rewritten by
`_nufft_setpts!`, executed repeatedly. `modeord` is `0` for centered modes and `1` for FFT order;
NUFSHT's plans use `0`. `nthreads` is resolved at construction (`0` meaning all Julia threads) and
bounds the split in [`_nufft_exec!`](@ref).
"""
struct DirectSumNUFFTPlan{T<:AbstractFloat, V<:AbstractVector{T}}
    type::Int
    n1::Int
    n2::Int
    iflag::Int
    ntrans::Int
    modeord::Int
    nthreads::Int
    x::V
    y::V
end

# Frequency carried by index `i` (1-based) of a length-`n` mode axis. FFT order splits at `cld(n,2)`:
# for odd `n`, which `Nφ = 2lmax+1` always is, the non-negative frequencies run to `(n-1)÷2` inclusive.
@inline function _mode_freq(i::Int, n::Int, modeord::Int)
    k = i - 1
    modeord == 0 && return k - n ÷ 2          # centered: -n÷2 … n-1-n÷2
    return k < cld(n, 2) ? k : k - n          # FFTW order
end

function _nufft_makeplan(::SpectralBackends.AbstractDirectSumSpectralBackend, nodes::NTuple{2,AbstractVector},
                         type, n_modes, iflag, ntrans, tol;
                         dtype = Float64, modeord = 0, nthreads = nothing, kwargs...)
    nt = (isnothing(nthreads) || nthreads == 0) ? Threads.nthreads() : Int(nthreads)
    nt ≥ 1 || throw(ArgumentError("nthreads must be positive (or 0 for all threads), got $nthreads"))
    x = similar(first(nodes), dtype, length(first(nodes)))
    p = DirectSumNUFFTPlan{dtype, typeof(x)}(Int(type), Int(n_modes[1]), Int(n_modes[2]),
                                             Int(iflag), Int(ntrans), Int(modeord), nt,
                                             x, similar(x))
    return _nufft_setpts!(p, nodes...)
end

function _nufft_setpts!(p::DirectSumNUFFTPlan, x, y)
    length(x) == length(p.x) || throw(DimensionMismatch(
        "direct-sum NUFFT plan holds $(length(p.x)) points, got $(length(x)); rebuild the plan for a " *
        "different count"))
    copyto!(p.x, x)
    copyto!(p.y, y)
    return p
end

_nufft_destroy!(::DirectSumNUFFTPlan) = nothing

# A synthesis-only plan holds no analysis plan; there is nothing to free.
_nufft_destroy!(::Nothing) = nothing

# Split `1:n` into `nt` contiguous ranges and run `f` on each. One thread spawns nothing, so a serial
# plan stays allocation-free. Callers pass an axis whose ranges write disjoint output, so the split
# needs no synchronisation beyond the closing barrier and the result does not depend on `nt`.
@inline function _ds_split(f::F, n::Int, nt::Int) where {F}
    nt ≤ 1 && return f(1:n)
    chunk = cld(n, nt)
    @sync for t in 1:nt
        lo = (t - 1) * chunk + 1
        hi = min(n, lo + chunk - 1)
        lo > hi && break
        Threads.@spawn f(lo:hi)
    end
    return nothing
end

# Complex exponentials each thread takes before a split: a spawn barrier costs microseconds and an
# exponential nanoseconds, so a thread needs a few thousand of them to repay its share of the barrier.
const _DIRECTSUM_MIN_WORK = 2_000

# Each direction is its own method, so the arrays reach the split as typed arguments and the inner loops
# stay allocation-free.
#
# type 2: values at the M points from the mode array, split over the points. Modes stay outermost, so a
# zero coefficient is skipped once for all points and a single-mode synthesis costs `O(M)`.
function _ds_type2!(vals, modes, p::DirectSumNUFFTPlan{T}, nt::Int) where {T}
    M = length(p.x)
    n1, n2, B = p.n1, p.n2, p.ntrans
    s = T(p.iflag)
    fill!(vals, zero(eltype(vals)))
    _ds_split(M, nt) do js
        @inbounds for b in 1:B, i2 in 1:n2
            k2 = T(_mode_freq(i2, n2, p.modeord))
            for i1 in 1:n1
                c = modes[i1 + (i2 - 1) * n1 + (b - 1) * n1 * n2]
                iszero(c) && continue
                k1 = T(_mode_freq(i1, n1, p.modeord))
                for j in js
                    vals[j + (b - 1) * M] += c * cis(s * (k1 * p.x[j] + k2 * p.y[j]))
                end
            end
        end
    end
    return vals
end

# type 1: the exact adjoint, split over the mode columns. Each mode is a reduction over the points held
# in a register, so no thread accumulates into another's output and the result is the same at any split.
function _ds_type1!(modes, vals, p::DirectSumNUFFTPlan{T}, nt::Int) where {T}
    M = length(p.x)
    n1, n2, B = p.n1, p.n2, p.ntrans
    s = T(p.iflag)
    _ds_split(n2, nt) do i2s
        @inbounds for b in 1:B, i2 in i2s
            k2 = T(_mode_freq(i2, n2, p.modeord))
            for i1 in 1:n1
                k1 = T(_mode_freq(i1, n1, p.modeord))
                acc = zero(eltype(modes))
                for j in 1:M
                    acc += vals[j + (b - 1) * M] * cis(s * (k1 * p.x[j] + k2 * p.y[j]))
                end
                modes[i1 + (i2 - 1) * n1 + (b - 1) * n1 * n2] = acc
            end
        end
    end
    return modes
end

function _nufft_exec!(p::DirectSumNUFFTPlan, input, output)
    nt = min(p.nthreads,
             max(1, (length(p.x) * p.n1 * p.n2 * p.ntrans) ÷ _DIRECTSUM_MIN_WORK))
    p.type == 2 ? _ds_type2!(output, input, p, nt) : _ds_type1!(output, input, p, nt)
    return output
end
