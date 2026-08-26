"""
    NUFFT.jl — the `N` step of `A = N·F·D·S`, behind a backend seam.

The transform cores and plan builders never name a NUFFT library: they call `_nufft_makeplan`,
`_nufft_setpts!`, `_nufft_exec!`, `_nufft_destroy!` and `_nufft_finalize!`. Plan *creation* dispatches
on `(backend, node array type)` — the backend picks the library, the node type picks host vs device
within it — while execution, point setting and teardown dispatch on the returned plan handle, so each
library's extension owns its own handle type and needs no cooperation from the others.

Backends are `SpectralBackends` types:

- `DirectSumSpectralBackend` — implemented here. No dependencies, always available, `O(M·K)`: the
  default and the correctness reference the fast backends are validated against, not a fast path.
  Threaded over the axis each direction writes, so `nthreads` means there what it means elsewhere.
- [`FINUFFTBackend`](@ref) — `NUFSHTFINUFFTExt`, `using FINUFFT` (cuFINUFFT on `CuArray` nodes).
- [`NonuniformFFTsBackend`](@ref) — `NUFSHTNonuniformFFTsExt`, `using NonuniformFFTs`. Pure Julia, and
  GPU support through KernelAbstractions rather than a single vendor.

The two library backends need distinct types rather than sharing `NUFFTSpectralBackend`: that type
says "a NUFFT method" without naming an implementation, so both extensions would define methods on it
and collide whenever a user loads both packages.
"""

using SpectralBackends: SpectralBackends

export FINUFFTBackend, NonuniformFFTsBackend

"""
    FINUFFTBackend <: SpectralBackends.AbstractNUFFTSpectralBackend

Evaluate the NUFFT with FINUFFT — cuFINUFFT for `CuArray` nodes. Requires `using FINUFFT`.
"""
struct FINUFFTBackend <: SpectralBackends.AbstractNUFFTSpectralBackend end

"""
    NonuniformFFTsBackend <: SpectralBackends.AbstractNUFFTSpectralBackend

Evaluate the NUFFT with NonuniformFFTs.jl: pure Julia, no binary dependency, and GPU support through
KernelAbstractions rather than one vendor. Requires `using NonuniformFFTs`.
"""
struct NonuniformFFTsBackend <: SpectralBackends.AbstractNUFFTSpectralBackend end

@inline _ext_loaded(name::Symbol) = Base.get_extension(@__MODULE__, name) !== nothing

"""
    _width_narrowable(backend) -> Bool

Whether the backend can build a plan at a reduced transform count at all, so that a batched solve can
transform only its live columns once the rest have retired. `false` by default; a backend that cannot
re-plan has nothing to narrow to.
"""
_width_narrowable(::SpectralBackends.AbstractSpectralBackend) = false
_width_narrowable(::FINUFFTBackend) = true
_width_narrowable(::NonuniformFFTsBackend) = true

"""
    _nufft_share_directions(backend) -> Bool
    _nufft_as_type1(type2_plan) -> plan

Whether one plan object serves both transform directions, and how to obtain the type-1 handle from the
type-2 one when it does. `false` by default: FINUFFT bakes `iflag` into its C plan, so the two
directions are separate objects owning separate FFT grids.

NonuniformFFTs is the other case — a `PlanNUFFT` encodes no direction, the sign coming from whether
`exec_type1!` or `exec_type2!` is called, so one plan serves both and sharing it halves the oversampled
grid *and* the sorted point copy a plan owns. The two directions are never executed concurrently, which
a plan already requires.
"""
_nufft_share_directions(::SpectralBackends.AbstractSpectralBackend) = false
_nufft_as_type1(p) = throw(ArgumentError(
    "this backend does not share one plan between directions; build the type-1 plan directly"))

"""
    _nufft_derived(p) -> Bool

Whether this handle wraps a plan another handle owns, so its points are set whenever that one's are.
`false` by default. Recorded when the handle is built rather than detected afterwards: `===` on a large
foreign immutable struct is not a usable identity test — `PlanNUFFT` compares unequal to *itself* that
way, while every one of its fields compares equal. Used to skip a redundant re-sort in
[`set_nodes!`](@ref).
"""
_nufft_derived(p) = false

"""
    _width_polymorphic(backend) -> Bool

Whether plans for different transform counts share one concrete type. `false` by default, because a
backend is free to carry the count in its plan's type parameters — `NonuniformFFTs.PlanNUFFT` does, in
three of them (the count itself, and again inside its `RealNUFFTData` and `BlockDataCPU` parameters),
while `FINUFFT.finufft_plan{T}` keeps it as a runtime field.

Narrowing needs this as well as [`_width_narrowable`](@ref), and for a stronger reason than the cache
needing one element type: **a width that appears in the plan's type is a separate specialization**, so
each new width recompiles that backend's whole spreading and interpolation path. Where the count is a
runtime field, one compilation serves every width. That difference is large enough to invert the
decision — narrowing buys a modest fraction of a solve, and only when columns retire far apart, which
they do not often do, since every column of a batch shares the point set and therefore `A` and its
singular values. So where this is `false`, staying at full width is the faster choice rather than a
fallback, and the two traits together say which case a backend is in.

Correctness never depends on either. The per-column sphere loop and the solver's vector reductions
narrow on every backend regardless — that part is uniform and free.
"""
_width_polymorphic(::SpectralBackends.AbstractSpectralBackend) = false
_width_polymorphic(::FINUFFTBackend) = true

"""
    _real_capable(backend) -> Bool

Whether the backend implements a genuine **real-data** transform: real non-uniform values, and the
uniform side stored as the half-spectrum a real signal determines. `false` by default.

A real field's mode array is exactly Hermitian, so every plan stores only its `kθ ≥ 0` half. This asks
the narrower question of whether the *backend* knows that — whether it will reconstruct the conjugate
half itself from a real-data transform, which also makes the non-uniform data real and so halves the
spreading/interpolation on top of the FFT. NonuniformFFTs does; FINUFFT has no real transform at all
and is instead given a half-height complex transform, whose missing half `_fold_weights!` pays for.

Correctness never depends on this: the folded and full assemblies represent the same series.
"""
_real_capable(::SpectralBackends.AbstractSpectralBackend) = false
_real_capable(::NonuniformFFTsBackend) = true

"""
    _resolve_nufft(backend, FE) -> backend

Concrete backends pass through untouched — one named explicitly is honoured or refused, never swapped.
`AutoSpectralBackend` chooses, and the field element type is part of that choice: a real `FE` prefers a
[`_real_capable`](@ref) backend, because a real field's mode array is Hermitian in `kθ` and its
non-uniform data is real, so such a backend holds `lmax+2` mode rows instead of `2lmax+3` and a real
strengths buffer instead of a complex one. Override with `nufft=`.

Falls back to direct summation, which is always available and always correct.
"""
_resolve_nufft(backend::SpectralBackends.AbstractSpectralBackend, ::Type) = backend
function _resolve_nufft(::SpectralBackends.AbstractAutoSpectralBackend, ::Type{FE}) where {FE}
    nu = _ext_loaded(:NUFSHTNonuniformFFTsExt)
    # A real field can use a real-data transform; a complex one cannot, so it takes the general order.
    FE <: Real && nu && return NonuniformFFTsBackend()
    _ext_loaded(:NUFSHTFINUFFTExt) && return FINUFFTBackend()
    nu && return NonuniformFFTsBackend()
    return SpectralBackends.DirectSumSpectralBackend()
end

# `maxlog = 1` makes it once per session per logger, so `@test_logs` still sees it.
function _warn_if_directsum(requested, resolved, M::Integer, K::Integer)
    requested isa SpectralBackends.AbstractAutoSpectralBackend || return nothing
    resolved isa SpectralBackends.DirectSumSpectralBackend || return nothing
    @warn "no fast NUFFT backend loaded; using direct summation, $M x $K per transform. " *
          "`using FINUFFT` or `using NonuniformFFTs` to avoid." maxlog = 1
    return nothing
end

# A backend whose extension is not loaded must say so, not fall back to something slower.
_nufft_makeplan(backend::SpectralBackends.AbstractSpectralBackend, nodes, type, n_modes, iflag,
                ntrans, tol; kwargs...) = throw(ArgumentError(
    "$(nameof(typeof(backend))) is not available — load its extension (`using FINUFFT` or " *
    "`using NonuniformFFTs`), or use SpectralBackends.DirectSumSpectralBackend()."))

# ── Direct summation ──────────────────────────────────────────────────────────

"""
    DirectSumNUFFTPlan{T,V}

Handle for the direct-summation backend. Mirrors a guru plan: built once, points written by
`_nufft_setpts!`, executed repeatedly. The node buffers are sized at construction — `_nufft_makeplan`
already receives the node set — so setting points only rewrites their contents.

`modeord` follows FINUFFT's convention so every backend agrees on mode layout: `1` is FFTW order
(what the scalar plan uses), `0` is CMCL-centered (the spin plan). `nthreads` is resolved at
construction (`0` meaning all cores) and bounds the split in [`_nufft_exec!`](@ref).
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

# Frequency carried by index `i` (1-based) of a length-`n` mode axis. The FFTW split is at `cld(n,2)`,
# not `n÷2`: for odd `n` — which `Nφ = 2lmax+1` always is — the non-negative frequencies run to
# `(n-1)÷2` inclusive, so `n÷2` would misplace the middle index.
@inline function _mode_freq(i::Int, n::Int, modeord::Int)
    k = i - 1
    modeord == 0 && return k - n ÷ 2          # CMCL: -n÷2 … n-1-n÷2
    return k < cld(n, 2) ? k : k - n          # FFTW order
end

function _nufft_makeplan(::SpectralBackends.AbstractDirectSumSpectralBackend, nodes::AbstractVector,
                         type, n_modes, iflag, ntrans, tol;
                         dtype = Float64, modeord = 0, nthreads = nothing, kwargs...)
    # `0` is the "all cores" sentinel the seam inherits from FINUFFT.
    nt = (isnothing(nthreads) || nthreads == 0) ? Threads.nthreads() : Int(nthreads)
    nt ≥ 1 || throw(ArgumentError("nthreads must be positive (or 0 for all cores), got $nthreads"))
    x = similar(nodes, dtype, length(nodes))
    return DirectSumNUFFTPlan{dtype, typeof(x)}(Int(type), Int(n_modes[1]), Int(n_modes[2]),
                                                Int(iflag), Int(ntrans), Int(modeord), nt,
                                                x, similar(x))
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
_nufft_finalize!(p::DirectSumNUFFTPlan) = p

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

# Complex exponentials per thread. A task-spawn barrier costs microseconds and an exponential
# nanoseconds, so a barrier buys itself back within order a thousand of them; the gate sits above that,
# where the split is still clearly ahead rather than merely even.
const _DIRECTSUM_MIN_WORK = 2_000

# Each direction is a separate method so the arrays reach the split as arguments. Assigning them from
# `input`/`output` inside a branch and then capturing them would box both, which costs the inner loop
# its types and the whole transform its allocation-freedom.
#
# type 2: values at the M points from the mode array. Split over the points, which is what it writes;
# modes stay outermost so a zero coefficient is skipped once rather than once per point, making a
# single-mode synthesis `O(M)` rather than `O(M·n1·n2)`.
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

# type 1: the exact adjoint. Split over the mode columns it writes; each mode is a reduction over the
# points held in a register, so no thread accumulates into another's output and the result is the same
# at any split.
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
