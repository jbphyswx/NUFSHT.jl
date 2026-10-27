# The `N` step of `A = N·F·S`.
#
# The transform cores and plan builders never name a NUFFT library: they call `_nufft_makeplan`,
# `_nufft_setpts!`, `_nufft_exec!` and `_nufft_destroy!`. Plan creation dispatches on the backend and
# takes the nodes, whose array type picks host or device within it; execution, point setting and teardown
# dispatch on the returned handle. Every mode array is in `FlowTransformBindings.CenteredModes()` order.
#
# Backends are `SpectralBackends` types:
#
# - `DirectSumSpectralBackend`: implemented here. No dependencies, always available, `O(M·K)`
#   multiply-adds and `O(M·(n₁ + n₂))` exponentials: the correctness reference the fast backends are
#   validated against. Threaded over the axis each direction writes, so `nthreads` means there what it
#   means elsewhere.
# - `FlowTransformBindings.NonuniformFFTsBackend()` (`using NonuniformFFTs`) and
#   `FlowTransformBindings.FINUFFTBackend()` (`using FINUFFT`, cuFINUFFT on `CuArray` nodes): the plans of
#   FlowTransformBindings, which give both a real-data transform.

using SpectralBackends: SpectralBackends

public SpectralBackends

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

A concrete backend passes through unchanged. `AutoSpectralBackend` takes NonuniformFFTs when it is loaded, FINUFFT when only it is, and direct
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
                ntrans, tol, ::Type, ::Type, nthreads, upsampfac) = throw(ArgumentError(
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

# `_nufft_makeplan(backend, nodes, type, n_modes, iflag, ntrans, tol, T, Z, nthreads, upsampfac)`: `T` is
# the precision and `Z` the element type of the non-uniform data, so a real `Z` builds the real-data
# plan. A thread count `0` takes the session's `Threads.nthreads()`, and an oversampling factor
# `nothing` the library's default.
_nufft_makeplan(backend::_FTBLibrary, nodes::NTuple{2,AbstractVector}, type, n_modes, iflag, ntrans, tol,
                ::Type{T}, ::Type{Z}, nthreads::Integer, upsampfac::Union{Nothing,Float64}) where {T,Z} =
    _FTBNUFFT(_ftb_plan(backend, nodes, type, n_modes, iflag, ntrans, tol, Z, nthreads, upsampfac), Int(type),
              false)

function _ftb_plan(backend::_FTBLibrary, nodes::NTuple{2,AbstractVector}, type, n_modes, iflag, ntrans, tol,
                   ::Type{Z}, nthreads::Integer, upsampfac::Union{Nothing,Float64}) where {Z}
    iflag == (type == 2 ? 1 : -1) || throw(ArgumentError(
        "a type-$type NUFFT runs with iflag $(type == 2 ? 1 : -1); got $iflag"))
    nt = nthreads == 0 ? Threads.nthreads() : Int(nthreads)
    return FTB.plan_nufft(backend, Z, nodes, (Int(n_modes[1]), Int(n_modes[2]));
                          ntrans = Int(ntrans), tol = tol, order = FTB.CenteredModes(), nthreads = nt,
                          upsampfac = upsampfac)
end

# A handle of the type `H` a width's pool slot holds: the library plan is asserted to `H`'s plan type
# before it is wrapped.
_make_nufft_as(::Type{_FTBNUFFT{P}}, backend::_FTBLibrary, nodes, type, n_modes, iflag, ntrans, tol,
               ::Type{T}, nthreads, upsampfac, ::Type{Z}) where {P,T,Z} =
    _FTBNUFFT{P}(_ftb_plan(backend, nodes, type, n_modes, iflag, ntrans, tol, Z, nthreads, upsampfac)::P,
                 Int(type), false)
_make_nufft_as(::Type{H}, backend, nodes, type, n_modes, iflag, ntrans, tol, ::Type{T}, nthreads,
               upsampfac, ::Type{Z}) where {H,T,Z} =
    _make_nufft(backend, nodes, type, n_modes, iflag, ntrans, tol, T, nthreads, upsampfac, Z)::H

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
    DirectSumNUFFTPlan{T,V,S}

Handle for the direct-summation backend: built once over its nodes, points rewritten by
`_nufft_setpts!`, executed repeatedly. `nthreads` is resolved at construction (`0` meaning all Julia
threads) and bounds the split in [`_nufft_exec!`](@ref).

The phase separates, `e^{is(k₁x + k₂y)} = e^{isk₁x} e^{isk₂y}`, so a point's `n₁·n₂` exponentials are
the products of two tables of `n₁` and `n₂`. `ex` and `ey` hold those tables, one column per thread.
"""
struct DirectSumNUFFTPlan{T<:AbstractFloat, V<:AbstractVector{T}, S<:AbstractMatrix{Complex{T}}}
    type::Int
    n1::Int
    n2::Int
    iflag::Int
    ntrans::Int
    nthreads::Int
    x::V
    y::V
    ex::S
    ey::S
end

function _nufft_makeplan(::SpectralBackends.AbstractDirectSumSpectralBackend, nodes::NTuple{2,AbstractVector},
                         type, n_modes, iflag, ntrans, tol, ::Type{T}, ::Type, nthreads::Integer,
                         upsampfac) where {T}
    nt = nthreads == 0 ? Threads.nthreads() : Int(nthreads)
    nt ≥ 1 || throw(ArgumentError("nthreads must be positive (or 0 for all threads), got $nthreads"))
    x = similar(first(nodes), T, length(first(nodes)))
    n1, n2 = Int(n_modes[1]), Int(n_modes[2])
    ex = Matrix{Complex{T}}(undef, n1, nt)
    ey = Matrix{Complex{T}}(undef, n2, nt)
    p = DirectSumNUFFTPlan{T, typeof(x), typeof(ex)}(Int(type), n1, n2, Int(iflag), Int(ntrans), nt,
                                                     x, similar(x), ex, ey)
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

# Split `1:n` into `nt` contiguous ranges and run `f(t, range)` on each, `t` the range's slot in the
# plan's per-thread tables. One thread spawns nothing, so a serial plan stays allocation-free. Callers pass
# an axis whose ranges write disjoint output, so the split needs no synchronisation beyond the closing
# barrier and the result does not depend on `nt`.
@inline function _ds_split(f::F, n::Int, nt::Int) where {F}
    nt ≤ 1 && return f(1, 1:n)
    chunk = cld(n, nt)
    @sync for t in 1:nt
        lo = (t - 1) * chunk + 1
        hi = min(n, lo + chunk - 1)
        lo > hi && break
        Threads.@spawn f(t, lo:hi)
    end
    return nothing
end

# Multiply-adds per thread below which the direct sum runs on one thread, where a spawn barrier
# outweighs the split.
const _DIRECTSUM_MIN_WORK = 32_768

# `e[i] = e^{i s kᵢ x}` over the centred frequencies `kᵢ` of `rng`, out of `n`.
@inline function _ds_phases!(e, s::T, x::T, n::Int, rng) where {T}
    @inbounds for i in rng
        e[i] = cis(s * T(FTB.mode_frequency(FTB.CenteredModes(), i, n)) * x)
    end
    return e
end

# Each direction is its own method, so the arrays reach the split as typed arguments and the inner loops
# stay allocation-free.
#
# type 2: values at the M points from the mode array, split over the points. Each point forms its two
# tables, then sums `Σᵢ₂ ey[i₂] Σᵢ₁ c[i₁,i₂] ex[i₁]`, the inner sum down a contiguous mode column.
function _ds_type2!(vals, modes, p::DirectSumNUFFTPlan{T}, nt::Int) where {T}
    M = length(p.x)
    n1, n2, B = p.n1, p.n2, p.ntrans
    s = T(p.iflag)
    _ds_split(M, nt) do t, js
        ex = view(p.ex, :, t)
        ey = view(p.ey, :, t)
        @inbounds for j in js
            _ds_phases!(ex, s, p.x[j], n1, 1:n1)
            _ds_phases!(ey, s, p.y[j], n2, 1:n2)
            for b in 1:B
                off = (b - 1) * n1 * n2
                acc = zero(eltype(vals))
                for i2 in 1:n2
                    col = zero(eltype(vals))
                    @simd for i1 in 1:n1
                        col += modes[off + i1 + (i2 - 1) * n1] * ex[i1]
                    end
                    acc += col * ey[i2]
                end
                vals[j + (b - 1) * M] = acc
            end
        end
    end
    return vals
end

# type 1: the exact adjoint, split over the mode columns. Each thread adds every point's
# `v·ey[i₂]·ex` into its own columns, the points in order, so the result is the same at any split.
function _ds_type1!(modes, vals, p::DirectSumNUFFTPlan{T}, nt::Int) where {T}
    M = length(p.x)
    n1, n2, B = p.n1, p.n2, p.ntrans
    s = T(p.iflag)
    _ds_split(n2, nt) do t, i2s
        ex = view(p.ex, :, t)
        ey = view(p.ey, :, t)
        @inbounds for b in 1:B, i2 in i2s, i1 in 1:n1
            modes[i1 + (i2 - 1) * n1 + (b - 1) * n1 * n2] = zero(eltype(modes))
        end
        @inbounds for j in 1:M
            _ds_phases!(ex, s, p.x[j], n1, 1:n1)
            _ds_phases!(ey, s, p.y[j], n2, i2s)
            for b in 1:B
                v = vals[j + (b - 1) * M]
                off = (b - 1) * n1 * n2
                for i2 in i2s
                    w = v * ey[i2]
                    @simd for i1 in 1:n1
                        modes[off + i1 + (i2 - 1) * n1] += w * ex[i1]
                    end
                end
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
