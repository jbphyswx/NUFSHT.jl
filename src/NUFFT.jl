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

A real field's mode array is exactly Hermitian (`Z[-k] = conj(Z[k])`), so half of it is redundant. A
backend that knows this halves the FFT *and* the spreading/interpolation, since the strengths are real
too. NonuniformFFTs does; FINUFFT has no real transform at all, and the only trick available there is
to hand it a smaller complex mode array — which shrinks the upsampled FFT but not the spreading, and
is measurably a loss once the point count dominates the mode count.

Correctness never depends on this: the folded and full assemblies represent the same series.
"""
_real_capable(::SpectralBackends.AbstractSpectralBackend) = false
_real_capable(::NonuniformFFTsBackend) = true

"""
    _resolve_nufft(backend) -> backend

Concrete backends pass through. `AutoSpectralBackend` prefers a loaded fast backend — FINUFFT first,
then NonuniformFFTs — and falls back to direct summation, which is always available. A backend named
explicitly is honoured or refused, never swapped for another.
"""
_resolve_nufft(backend::SpectralBackends.AbstractSpectralBackend) = backend
function _resolve_nufft(::SpectralBackends.AbstractAutoSpectralBackend)
    _ext_loaded(:NUFSHTFINUFFTExt) && return FINUFFTBackend()
    _ext_loaded(:NUFSHTNonuniformFFTsExt) && return NonuniformFFTsBackend()
    return SpectralBackends.DirectSumSpectralBackend()
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
(what the scalar plan uses), `0` is CMCL-centered (the spin plan).
"""
struct DirectSumNUFFTPlan{T<:AbstractFloat, V<:AbstractVector{T}}
    type::Int
    n1::Int
    n2::Int
    iflag::Int
    ntrans::Int
    modeord::Int
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
                         dtype = Float64, modeord = 0, kwargs...)
    x = similar(nodes, dtype, length(nodes))
    return DirectSumNUFFTPlan{dtype, typeof(x)}(Int(type), Int(n_modes[1]), Int(n_modes[2]),
                                                Int(iflag), Int(ntrans), Int(modeord),
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
_nufft_finalize!(p::DirectSumNUFFTPlan) = p

# type 2: values at the M points from the mode array; type 1 is its exact adjoint.
function _nufft_exec!(p::DirectSumNUFFTPlan{T}, input, output) where {T}
    M = length(p.x)
    n1, n2, B = p.n1, p.n2, p.ntrans
    s = T(p.iflag)
    if p.type == 2
        modes, vals = input, output
        fill!(vals, zero(eltype(vals)))
        @inbounds for b in 1:B, i2 in 1:n2
            k2 = T(_mode_freq(i2, n2, p.modeord))
            for i1 in 1:n1
                c = modes[i1 + (i2 - 1) * n1 + (b - 1) * n1 * n2]
                iszero(c) && continue
                k1 = T(_mode_freq(i1, n1, p.modeord))
                for j in 1:M
                    vals[j + (b - 1) * M] += c * cis(s * (k1 * p.x[j] + k2 * p.y[j]))
                end
            end
        end
    else
        vals, modes = input, output
        fill!(modes, zero(eltype(modes)))
        @inbounds for b in 1:B, i2 in 1:n2
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
    return output
end
