"""
    Plan.jl — Pre-allocated plan struct for NUFSHT transforms.

A `NUSHTplan` pre-allocates every intermediate buffer **and** owns persistent FINUFFT *guru*
plans (built once, points set once), so repeated transforms on the same node set — filtering many
fields, or the hundreds of matvecs inside `nusht_solve!` — allocate nothing and never re-plan
FINUFFT. All array/plan fields are type parameters so the same struct instantiates on host arrays
today and device arrays later.
"""

using AbstractFFTs: AbstractFFTs
using FFTW: FFTW
using FastTransforms: FastTransforms

export AbstractNUSHTplan, NUSHTplan, make_plan, close!, set_nodes!
export AbstractNodeSet, FixedCountNodes, VariableCountNodes
export AbstractPlanTuning, NoTuning, AutoTuning, ThoroughTuning

# The NUFFT seam lives in NUFFT.jl. `_host` is a no-op for an `Array` and copies a device array's
# coords to host; points are set once, so it is off the hot path.
@inline _host(x::Array) = x
@inline _host(x::AbstractArray) = Array(x)

# `upsampfac` is omitted rather than passed as 0, so an untuned plan gets the library's own default.
# These are FINUFFT options; backends without them ignore the keywords.
@inline _make_nufft(backend, nodes, type, n_modes, iflag, B, tol, ::Type{T}, modeord, nthreads,
                    upsampfac) where {T} =
    upsampfac > 0 ?
    _nufft_makeplan(backend, nodes, type, n_modes, iflag, B, tol;
                    dtype = T, modeord = modeord, nthreads = nthreads, upsampfac = upsampfac) :
    _nufft_makeplan(backend, nodes, type, n_modes, iflag, B, tol;
                    dtype = T, modeord = modeord, nthreads = nthreads)

# Backend-generic zeroed buffer shaped like `ref` (host `Array` for CPU nodes, device array for GPU
# nodes) — used so a device node set yields device-resident plan buffers.
@inline _zeros_like(ref::AbstractArray, ::Type{S}, dims::Integer...) where {S} =
    fill!(similar(ref, S, dims...), zero(S))

# A copy of host array `a` moved to `ref`'s backend (host stays host, device→device). Used for the
# small precomputed phase vectors so they match a device node set (a device broadcast against a host
# vector would fail / be wrong).
@inline _to_like(ref::AbstractArray, a::AbstractArray) = copyto!(similar(ref, eltype(a), size(a)...), a)

# A length-`n` array of `ref`'s type and element type, filled from `src` (converting if needed). Used
# when a node set's point count changes, so the new buffers keep the plan's declared types.
@inline _resized_like(ref::AbstractVector, src, n::Integer) =
    copyto!(similar(ref, eltype(ref), n), src)

# ── Planner ownership and plan tuning ─────────────────────────────────────────
# FastTransforms and FFTW.jl share one libfftw3, so the FFTW *planner* thread count is a single
# process global — and `FINUFFT.finufft_setpts!` resets it. It is baked into an FFTW plan when the
# plan is BUILT (execution never re-reads it), so the sphere plans must be built with it pinned or
# they silently inherit whatever the last foreign-library call left behind.
const _PLANNER_LOCK = ReentrantLock()

# Tuning outcomes are memoized: the search is worth paying once per problem shape, not per plan.
const _SPH_TUNING = Dict{Tuple{Int,Int,DataType},Tuple{Int,UInt32}}()
const _NUFFT_TUNING = Dict{Tuple{Int,Int,Int,Int,DataType,Float64,Int,DataType},Tuple{Int,Float64}}()

"""
    AbstractPlanTuning

How hard [`make_plan`](@ref) / [`make_spin_plan`](@ref) searches for the library settings that the
problem does not fix: the FFTW planner thread count and effort for the sphere plans, and FINUFFT's
`nthreads` and `upsampfac`. Subtype this and add `_tune_sph` / `_tune_nufft` methods to define a
custom strategy.
"""
abstract type AbstractPlanTuning end

"""
    NoTuning <: AbstractPlanTuning

Build with fixed, measured-good settings and time nothing — the default. Costs nothing beyond an
ordinary plan build, and captures the large win of pinning the FFTW planner thread count instead of
inheriting whatever the last foreign-library call left in that process global.
"""
struct NoTuning <: AbstractPlanTuning end

"""
    AutoTuning <: AbstractPlanTuning

Time the candidate FFTW planner thread counts and FINUFFT `nthreads`/`upsampfac` pairs under
`FFTW.ESTIMATE` planning, keeping the fastest. Adds roughly 1.1x over [`NoTuning`](@ref) and costs
O(100 ms) of trial builds, so it needs on the order of a thousand transforms on one plan to pay for
itself. Outcomes are memoized per problem shape.
"""
struct AutoTuning <: AbstractPlanTuning end

"""
    ThoroughTuning <: AbstractPlanTuning

[`AutoTuning`](@ref) plus a search over `FFTW.MEASURE` planner effort, which can beat `ESTIMATE`
severalfold on the sphere synthesis/analysis step at large `lmax` but costs FFTW up to seconds to
plan. For a plan that will live for a very long time.
"""
struct ThoroughTuning <: AbstractPlanTuning end

# A candidate must beat the incumbent by this margin, so noise in one early-abandoned sample cannot
# unseat a well-measured configuration.
const _TUNING_MARGIN = 0.95

# FFTW planner efforts each strategy is allowed to consider.
_sph_flagset(::AbstractPlanTuning) = (FFTW.ESTIMATE,)
_sph_flagset(::ThoroughTuning) = (FFTW.ESTIMATE, FFTW.MEASURE)

"""
    _with_fftw_planner_nthreads(f, n)

Run `f()` with the shared FFTW planner thread count pinned to `n`, restoring the previous value.
Serialized on `_PLANNER_LOCK` because that count is process-global.
"""
function _with_fftw_planner_nthreads(f, n::Integer)
    return Base.lock(_PLANNER_LOCK) do
        prev = FFTW.get_num_threads()
        try
            FastTransforms.ft_fftw_plan_with_nthreads(n)
            return f()
        finally
            FastTransforms.ft_fftw_plan_with_nthreads(prev)
        end
    end
end

_thread_candidates() = sort!(unique!(Int[1, 2, 4, cld(Sys.CPU_THREADS, 2), Sys.CPU_THREADS]))

# Every FastTransforms plan the S-step needs, under one pinned planner setting. `plan_sph2fourier` is
# the butterfly step and takes no FFTW flags; the adjoints are built here to inherit the same setting.
function _build_sph_plans(Fslice, nt::Integer, flags::Integer)
    return _with_fftw_planner_nthreads(nt) do
        P  = FastTransforms.plan_sph2fourier(Fslice)
        PS = FastTransforms.plan_sph_synthesis(Fslice; flags = flags)
        PA = FastTransforms.plan_sph_analysis(Fslice; flags = flags)
        return (P, PS, PA, P', PS')
    end
end

# Time a candidate, abandoning it after one sample once it cannot beat `bound` — bad candidates run
# 10-100x slower, so without this the search is dominated by settings it is about to reject.
function _time_candidate(f, bound::Float64, reps::Int = 3)
    t = @elapsed f()
    t < bound || return t
    for _ in 2:reps
        t = min(t, @elapsed f())
    end
    return t
end

# One application of the two FFTW-planned S-step operators. The reset copy is in every candidate's
# time, so it cannot bias the argmin, and it stops repeated application drifting the values.
function _apply_sph!(PS, PA, src, scratch)
    copyto!(scratch, src)
    LinearAlgebra.lmul!(PS, scratch)
    LinearAlgebra.lmul!(PA, scratch)
    return scratch
end

"""
    _tune_sph(Fslice, tuning) -> (nthreads, flags)

Pick the FFTW planner thread count and effort for the sphere synthesis/analysis plans by building
and timing the candidates. Memoized on `(Nθ, Nφ, typeof(tuning))`.
"""
_tune_sph(::AbstractMatrix, ::NoTuning) = (1, UInt32(FFTW.ESTIMATE))

function _tune_sph(Fslice::AbstractMatrix, tuning::AbstractPlanTuning)
    Nθ, Nφ = size(Fslice)
    key = (Nθ, Nφ, typeof(tuning))
    cached = Base.lock(() -> get(_SPH_TUNING, key, nothing), _PLANNER_LOCK)
    cached === nothing || return cached

    # Deterministic, non-degenerate probe data. Transform cost here is data-independent (only
    # subnormals would change it, and this pattern produces none), so no RNG is needed.
    RT = eltype(Fslice)
    src = RT[RT(sinpi((i + 2j) / (Nθ + Nφ))) for i in 1:Nθ, j in 1:Nφ]
    scratch = similar(src)
    best_cfg = (1, UInt32(FFTW.ESTIMATE))
    best_t = Inf
    warmed = false
    for flags in _sph_flagset(tuning), nt in _thread_candidates()
        _, PS, PA, _, _ = _build_sph_plans(Fslice, nt, flags)
        warmed || (_apply_sph!(PS, PA, src, scratch); warmed = true)
        t = _time_candidate(() -> _apply_sph!(PS, PA, src, scratch), best_t)
        if t < best_t * _TUNING_MARGIN
            best_t = t
            best_cfg = (nt, UInt32(flags))
        end
    end
    Base.lock(() -> (_SPH_TUNING[key] = best_cfg), _PLANNER_LOCK)
    return best_cfg
end

"""
    _tune_nufft(θ, φ, n_modes, type, iflag, B, T, tol, modeord, tuning) -> (nthreads, upsampfac)

Pick FINUFFT's thread count and upsampling factor by timing trial guru plans on the real node set.
`nthreads = 0` is FINUFFT's "all cores" sentinel and is one of the candidates. Memoized on the
problem shape with `M` bucketed to a power of two, so a stream of nearby point counts tunes once.
Host `Array` nodes only — a device node set takes the library defaults (its plan is built through
the cuFINUFFT seam, which this host timing loop cannot exercise).
"""
# `nthreads` and `upsampfac` are FINUFFT options, so only that backend has anything to search; every
# other backend (and any device node set, whose plan is built through the cuFINUFFT seam this host
# timing loop cannot exercise) takes the library defaults.
_tune_nufft(backend, ::AbstractArray, ::AbstractArray, n_modes, type, iflag, B, ::Type{T}, tol,
            modeord, ::AbstractPlanTuning) where {T} = (0, 0.0)

# Candidate `(nthreads, upsampfac)` pairs. Both zeros are FINUFFT sentinels: `nthreads = 0` is "all
# cores", `upsampfac = 0.0` is "library chooses" (`finufft_opts.h`: 2.0 std, 1.25 small FFT, 0.0 auto).
# Auto is included because it is usually right — the search exists for the cases where it is not.
_nufft_candidates(::NoTuning) = Tuple{Int,Float64}[]
_nufft_candidates(::AbstractPlanTuning) =
    [(nt, uf) for uf in (0.0, 2.0, 1.25) for nt in Int[0; _thread_candidates()]]

function _tune_nufft(backend::FINUFFTBackend, θ::Array, φ::Array, n_modes, type::Integer,
                     iflag::Integer, B::Integer, ::Type{T}, tol::Float64, modeord::Integer,
                     tuning::AbstractPlanTuning) where {T}
    candidates = _nufft_candidates(tuning)
    isempty(candidates) && return (0, 0.0)
    M = length(θ)
    key = (Int(n_modes[1]), Int(n_modes[2]), Int(B), Int(type), T, tol,
           prevpow(2, max(M, 1)), typeof(tuning))
    cached = Base.lock(() -> get(_NUFFT_TUNING, key, nothing), _PLANNER_LOCK)
    cached === nothing || return cached

    modes = zeros(Complex{T}, n_modes[1], n_modes[2], B)
    strengths = zeros(Complex{T}, M, B)
    inp, outp = type == 2 ? (modes, strengths) : (strengths, modes)
    best_cfg = (0, 2.0)
    best_t = Inf
    for (nt, upsampfac) in candidates
        p = _nufft_makeplan(backend, θ, type, n_modes, iflag, B, tol;
                            dtype = T, modeord = modeord, nthreads = nt, upsampfac = upsampfac)
        try
            _nufft_setpts!(p, θ, φ)
            t = _time_candidate(() -> _nufft_exec!(p, inp, outp), best_t)
            if t < best_t * _TUNING_MARGIN
                best_t = t
                best_cfg = (nt, upsampfac)
            end
        finally
            _nufft_destroy!(p)
        end
    end
    Base.lock(() -> (_NUFFT_TUNING[key] = best_cfg), _PLANNER_LOCK)
    return best_cfg
end

"""
    AbstractNUSHTplan

Supertype of [`NUSHTplan`]() and [`SpinNUSHTplan`](); both carry an [`AbstractNodeSet`]()
in their `nodes` field, so the node-set operations are written once.
"""
abstract type AbstractNUSHTplan end

"""
    AbstractNodeSet

The point-dependent half of a plan: the `M` scattered nodes, the `(M, B)` strengths buffer, and the
two FINUFFT guru plans — which own the loaded point tables, so they belong with the points rather
than with the bandlimit machinery.

Both concrete forms let the nodes **move** freely (that only rewrites array contents). They differ in
whether the *count* may change, which is the only thing that requires rebinding a field:
[`FixedCountNodes`](@ref) is immutable, [`VariableCountNodes`](@ref) is not. `make_plan`'s
`variable_npts` keyword picks one, so mutability is opt-in rather than imposed.
"""
abstract type AbstractNodeSet end

"""
    FixedCountNodes <: AbstractNodeSet

Immutable node set — the default. The nodes may move anywhere via [`set_nodes!`](@ref); only their
count is fixed, because it sizes the point-indexed buffers.
"""
struct FixedCountNodes{RV,SV,CT2,N1,N2} <: AbstractNodeSet
    θ_nodes::RV
    φ_nodes::RV
    θ_nufft::RV
    θ_shift::SV
    fbuf::CT2
    nufft_type2::N2
    nufft_type1::N1
end

"""
    VariableCountNodes <: AbstractNodeSet

Node set whose three point-sized fields are assignable, so [`set_nodes!`](@ref) also accepts a
different number of points. The guru plans stay `const` — `finufft_setpts!` updates their point count
in place. Request one with `make_plan(…; variable_npts = true)`.
"""
mutable struct VariableCountNodes{RV,SV,CT2,N1,N2} <: AbstractNodeSet
    θ_nodes::RV
    φ_nodes::RV
    θ_nufft::RV
    θ_shift::SV
    fbuf::CT2
    const nufft_type2::N2
    const nufft_type1::N1
end

_node_set(::Val{false}, θ, φ, θn, θs, fbuf, p2, p1) = FixedCountNodes(θ, φ, θn, θs, fbuf, p2, p1)
_node_set(::Val{true}, θ, φ, θn, θs, fbuf, p2, p1) = VariableCountNodes(θ, φ, θn, θs, fbuf, p2, p1)

# Internal accessors for the point-dependent fields, so the rest of the package is written against
# one spelling regardless of which node-set form a plan carries.
@inline _θnodes(p) = p.nodes.θ_nodes
@inline _φnodes(p) = p.nodes.φ_nodes
@inline _θnufft(p) = p.nodes.θ_nufft
@inline _θshift(p) = p.nodes.θ_shift
@inline _fbuf(p) = p.nodes.fbuf
@inline _nufft2(p) = p.nodes.nufft_type2
@inline _nufft1(p) = p.nodes.nufft_type1

"""
    NUSHTplan{T}

Pre-computed plan for non-uniform spherical harmonic transforms at `M` scattered points, up to
degree `lmax`, transforming `B` co-located fields per call (`ntrans = B`).

Fields:
- `lmax`, `Nθ = lmax+1`, `Nφ = 2lmax+1`, `B` (batch size / FINUFFT `ntrans`)
- `tol`: FINUFFT accuracy tolerance
- `nodes`: the [`AbstractNodeSet`](@ref) holding `θ_nodes`, `φ_nodes` (colatitudes ∈ [0,π] and
  longitudes ∈ [0,2π) of the `M` points), the `(M, B)` strengths buffer `fbuf`, and the two FINUFFT
  guru plans. [`set_nodes!`](@ref) re-points it.
The field element type `FE` selects the path: a real field's torus spectrum is Hermitian, so it takes
`rfft`/`brfft` and gives the NUFFT only the `Nθ+1` non-negative θ wavenumbers; a complex field takes
`fft`/`bfft` and the full mode array. Everything else is shared.

- `C`, `F`, `F̃`: coefficient scratch / CC map `(Nθ, Nφ, B)` and doubled-torus map `(2Nθ, Nφ, B)`,
  eltype `FE`
- `Fhat`: complex modes handed to the NUFFT — `(Nθ+1, Nφ, B)` if `FE` is real, else `(2Nθ, Nφ, B)`
- `phase_scaled`: `exp(-iπ k_θ / 2Nθ) / (2Nθ·Nφ)` — half-pixel θ correction and FFT normalization,
  times the Hermitian fold weight `{1, 2, …, 2, 1}` in the real case
- `phase_conj`: `conj(exp(-iπ k_θ / 2Nθ)) / (2Nθ·Nφ)`. No fold weight — `brfft` supplies that
  doubling, and applying it twice breaks the adjoint identity.
- `fft_plan`, `ifft_plan`: FFTW plans over dims `(1,2)`; the inverse is unnormalised, so the
  `1/(2Nθ·Nφ)` lives in the phases.
- `Fslice`: `(Nθ, Nφ)` scratch `Matrix` each batch slice is `copyto!`-ed through for the S-step.
  FastTransforms has no `Float32` sphere plan, so it stays double precision and the copy converts.
- `sph_plan`, `sph_plan_synth`, `sph_plan_analysis`: FastTransforms `plan_sph2fourier` (P),
  `plan_sph_synthesis` (PS), `plan_sph_analysis` (PA) on a `(Nθ, Nφ)` slice. Forward S = `PS·P`
  (`sph_evaluate!`); CC-inverse S⁻¹ = `P⁻¹·PA` (`nusht_type1!`, replicating `sph_transform!` with
  persistent plans instead of its per-call rebuild); Euclidean adjoint S† = `P'·PS'`
  (`_nusht_true_adjoint!`)

`nodes.nufft_type2` is the guru type-2 plan (`iflag = +1`, synthesis N) and `nodes.nufft_type1` the
type-1 plan (`iflag = -1`, adjoint N†). `iflag = +1` supplies the reconstruction sign directly, so the
modes need no conjugate-transpose. The axis convention is `x = θ`, `y = φ`.

Mode ordering follows the layout: a full-length θ axis uses FFTW-native order (`modeord = 1`), a
half-length one centered order (`modeord = 0`), since no NUFFT offers non-negative wavenumbers on a
single axis. `nodes.θ_shift` holds the `cis(N0·θ)` removing that offset, `nothing` otherwise.
"""
struct NUSHTplan{T<:AbstractFloat, FE<:Number, AT3<:AbstractArray{FE,3}, DT3<:AbstractArray{FE,3},
                 AT2<:AbstractMatrix, HT, CT3<:AbstractArray{Complex{T},3},
                 CV<:AbstractVector{Complex{T}},
                 ND<:AbstractNodeSet, FP, IP, SP, SPS, SPA, SPADJ, SPSADJ, FT} <: AbstractNUSHTplan
    lmax::Int
    Nθ::Int
    Nφ::Int
    B::Int
    tol::FT
    nodes::ND                     # the point-dependent half; see AbstractNodeSet
    C::AT3
    F::AT3
    F̃::DT3
    Fhalf::HT                     # rfft half-spectrum (Nθ+1, Nφ, B); `nothing` for a complex field
    Fhat::CT3
    Fslice::AT2
    phase_scaled::CV
    phase_conj::CV
    fft_plan::FP
    ifft_plan::IP
    sph_plan::SP
    sph_plan_synth::SPS
    sph_plan_analysis::SPA
    sph_plan_adj::SPADJ           # sph_plan' (P'), stored to keep _nusht_true_adjoint! alloc-free
    sph_plan_synth_adj::SPSADJ    # sph_plan_synth' (PS')
end

"""
    make_plan([FE = Float64,] θ_nodes, φ_nodes, lmax; tol=1e-8, ntrans=1, tuning=NoTuning(), …)

`FE` is the field element type, positional as it is for `zeros(T, …)`, and it selects the transform:
`Float64`/`Float32` build the real specialization (Hermitian torus spectrum → `rfft`/`brfft`, half the
θ wavenumbers reach the NUFFT), `ComplexF64`/`ComplexF32` the complex one (`fft`/`bfft`, full
spectrum). Both are the same spherical harmonic transform; only the symmetry exploited differs. The
`T = FE` keyword form forwards to the positional one.

Construct a `NUSHTplan` for `M` scattered points at colatitudes `θ_nodes ∈ [0,π]` and longitudes
`φ_nodes ∈ [0,2π)`, up to spherical harmonic degree `lmax`, transforming `ntrans` co-located fields
per call. Builds the FINUFFT guru plans and sets the nonuniform points once; they are freed by a
finalizer (or eagerly via [`close!`](@ref)).

Keyword arguments:
- `tol`: FINUFFT accuracy tolerance.
- `T`: floating-point type (`Float64`/`Float32`).
- `ntrans`: batch size `B` — transform `B` co-located fields (same nodes) per call.
- `tuning`: an [`AbstractPlanTuning`](@ref) — [`NoTuning`](@ref) (default), [`AutoTuning`](@ref) or
  [`ThoroughTuning`](@ref). The default already pins the settings that matter most; the searching
  strategies trade O(100 ms)-O(1 s) of trial builds for roughly a further 1.1x per transform, so
  they are worth it only for a plan reused thousands of times. Outcomes are memoized per problem
  shape, so building many plans of one size pays the search once.
- `ft_fftw_nthreads`, `ft_fftw_flags`: override the FFTW planner thread count / flags used for the
  sphere synthesis and analysis plans.
- `nthreads`, `upsampfac`: override FINUFFT's thread count (`0` is its "all cores" sentinel) and
  upsampling factor.

Each override keyword defaults to `nothing`, meaning "whatever `tuning` decides". An explicit value
is honoured exactly and skips the search for that setting.

FINUFFT accepts coordinates in `[-3π, 3π]`, so natural `[0,π]`/`[0,2π)` coordinates are passed
directly.
"""
function make_plan(
    ::Type{FE},
    θ_nodes,
    φ_nodes,
    lmax;
    tol = 1e-8,
    ntrans::Integer = 1,
    tuning::AbstractPlanTuning = NoTuning(),
    nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    variable_npts::Bool = false,
    nthreads::Union{Nothing,Integer} = nothing,
    upsampfac::Union{Nothing,Real} = nothing,
    ft_fftw_nthreads::Union{Nothing,Integer} = nothing,
    ft_fftw_flags::Union{Nothing,Integer} = nothing,
) where {FE<:Number}
    @assert length(θ_nodes) == length(φ_nodes)
    B = Int(ntrans)
    @assert B ≥ 1
    T = real(FE)
    T <: AbstractFloat ||
        throw(ArgumentError("field element type must be a float or complex float, got $FE"))
    realfield = FE <: Real

    Nθ = lmax + 1
    Nφ = 2lmax + 1
    Nθ_dbl = 2Nθ
    M = length(θ_nodes)

    # Nodes keep their input array type (host `Vector` or device array), eltype coerced to `T`; every
    # buffer is allocated `similar` to the nodes, so a device node set yields a device-resident plan.
    θ = T.(θ_nodes)
    φ = T.(φ_nodes)

    # A half-length mode axis shrinks the NUFFT's FFT but not its spreading, so it only pays when modes
    # dominate. No NUFFT offers non-negative wavenumbers on one axis; centered order is a constant
    # offset away, undone per point by `θ_shift`, which puts φ in centered order too.
    Nθ_half = Nθ + 1
    halfmodes = realfield && M < Nθ_dbl * Nφ
    Nk = halfmodes ? Nθ_half : Nθ_dbl
    C     = _zeros_like(θ, FE, Nθ, Nφ, B)
    F     = _zeros_like(θ, FE, Nθ, Nφ, B)
    F̃     = _zeros_like(θ, FE, Nθ_dbl, Nφ, B)
    Fhat  = _zeros_like(θ, Complex{T}, Nk, Nφ, B)
    Fhalf = realfield ? _zeros_like(θ, Complex{T}, Nθ_half, Nφ, B) : nothing
    fbuf  = _zeros_like(θ, Complex{T}, M, B)
    N0 = Nθ_half ÷ 2
    θ_shift = halfmodes ? _to_like(θ, Complex{T}.(cis.(T(N0) .* θ))) : nothing

    # FFTs act on the (θ̃, φ) plane only; the batch axis is left untransformed. Both inverses are the
    # unnormalised ones, so the `1/(2Nθ·Nφ)` lives in `phase_conj` rather than in the plan.
    fft_plan  = realfield ? AbstractFFTs.plan_rfft(F̃, (1, 2)) : AbstractFFTs.plan_fft(F̃, (1, 2))
    ifft_plan = realfield ? AbstractFFTs.plan_brfft(Fhalf, Nθ_dbl, (1, 2)) :
                            AbstractFFTs.plan_bfft(Fhat, (1, 2))

    # Half-pixel θ phase for the CC cell-center offset, moved to the node backend so the
    # `Fhat .*= phase_*` broadcast is device-resident. The real case folds `{1,2,…,2,1}` in here only:
    # `brfft` applies the same doubling on the way back, so repeating it in `phase_conj` would break
    # `⟨Ax,y⟩ = ⟨x,A†y⟩`. The Nyquist slot is `+Nθ` under the rephase and `-Nθ` in the padded array;
    # its phase must be built at whichever sign the NUFFT uses, or that bin comes in negated.
    k_θ = if halfmodes
        collect(0:Nθ)
    elseif realfield
        [k < Nθ ? k : -Nθ for k in 0:Nθ]
    else
        [k < Nθ ? k : k - Nθ_dbl for k in 0:(Nθ_dbl - 1)]
    end
    herm = realfield ? [(k == 0 || k == Nθ) ? one(T) : T(2) for k in 0:Nθ] : ones(T, Nθ_dbl)
    phase        = Complex{T}.(cis.(-π .* T.(k_θ) ./ Nθ_dbl))
    phase_scaled = _to_like(θ, herm .* phase ./ (Nθ_dbl * Nφ))
    phase_conj   = _to_like(θ, conj.(phase) ./ (Nθ_dbl * Nφ))

    # FastTransforms plans operate on a single dense (Nθ, Nφ) HOST `Matrix` (FastTransforms is CPU-only);
    # each batch slice is `copyto!`-ed through `Fslice` (host↔device for a device plan — the S-step is an
    # inherent host bounce). Persistent P/PS/PA plans avoid `sph_transform!`'s per-call rebuild, and are
    # built under a pinned FFTW planner thread count (see `_with_fftw_planner_nthreads`).
    # FastTransforms has Float64 and ComplexF64 sphere plans but no Float32 ones, so this buffer is
    # always double precision; the slice copy that was already happening does the conversion, and
    # every other buffer runs at `FE`.
    Fslice = zeros(realfield ? Float64 : ComplexF64, Nθ, Nφ)
    sph_nt, sph_flags = _tune_sph(Fslice, tuning)
    isnothing(ft_fftw_nthreads) || (sph_nt = Int(ft_fftw_nthreads))
    isnothing(ft_fftw_flags) || (sph_flags = UInt32(ft_fftw_flags))
    sph_plan, sph_plan_synth, sph_plan_analysis, sph_plan_adj, sph_plan_synth_adj =
        _build_sph_plans(Fslice, sph_nt, sph_flags)

    # iflag +1 for synthesis (type 2): reconstruction uses the +i (inverse-DFT) sign, so the raw
    # FFT modes need no conjugation — an axis swap takes the place of a conjugate-transpose
    # (conj(c)·e^{-ikx} = c·e^{+ikx}). type 1 (−1) is the exact adjoint.
    tol64 = Float64(tol)
    n_modes = Int64[Nk, Nφ]
    modeord = halfmodes ? 0 : 1
    nub = _resolve_nufft(nufft)
    nt2, uf2 = _tune_nufft(nub, θ, φ, n_modes, 2, +1, B, T, tol64, modeord, tuning)
    nt1, uf1 = _tune_nufft(nub, θ, φ, n_modes, 1, -1, B, T, tol64, modeord, tuning)
    isnothing(nthreads) || (nt2 = nt1 = Int(nthreads))
    isnothing(upsampfac) || (uf2 = uf1 = Float64(upsampfac))
    nufft_type2 = _make_nufft(nub, θ, 2, n_modes, +1, B, tol64, T, modeord, nt2, uf2)
    nufft_type1 = _make_nufft(nub, θ, 1, n_modes, -1, B, tol64, T, modeord, nt1, uf1)
    _nufft_setpts!(nufft_type2, θ, φ)
    _nufft_setpts!(nufft_type1, θ, φ)
    _nufft_finalize!(nufft_type2)
    _nufft_finalize!(nufft_type1)
    # A scalar plan hands FINUFFT the colatitudes unchanged, so `θ_nufft` aliases `θ_nodes` and costs
    # no extra storage; a spin plan negates them and owns a separate array.
    nodes = _node_set(Val(variable_npts), θ, φ, θ, θ_shift, fbuf, nufft_type2, nufft_type1)

    return NUSHTplan{T, FE, typeof(C), typeof(F̃), typeof(Fslice), typeof(Fhalf), typeof(Fhat),
                     typeof(phase_scaled), typeof(nodes),
                     typeof(fft_plan), typeof(ifft_plan), typeof(sph_plan), typeof(sph_plan_synth),
                     typeof(sph_plan_analysis), typeof(sph_plan_adj), typeof(sph_plan_synth_adj),
                     typeof(tol64)}(
        lmax, Nθ, Nφ, B, tol64, nodes, C, F, F̃, Fhalf, Fhat, Fslice,
        phase_scaled, phase_conj, fft_plan, ifft_plan, sph_plan, sph_plan_synth, sph_plan_analysis,
        sph_plan_adj, sph_plan_synth_adj,
    )
end

# Element type positional, as for `zeros(T, …)`: it carries precision and realness together. Omitted,
# precision comes from the nodes and the field is real.
make_plan(θ_nodes, φ_nodes, lmax; kwargs...) =
    make_plan(float(eltype(θ_nodes)), θ_nodes, φ_nodes, lmax; kwargs...)

"""
    set_nodes!(plan, θ_nodes, φ_nodes) -> plan

Move a plan's nodes to new positions, reusing every structure fixed by `(lmax, B, T)` — the
FastTransforms sphere plans, the FFTW plans and all coefficient buffers. Only FINUFFT's point tables
are rebuilt, so this costs 0.01–0.10 ms against 3–7 ms to build an equivalent plan from scratch
(measured, lmax 45–128).

The points may move anywhere. Changing how *many* there are additionally needs the plan's
point-indexed buffers to be replaced, which requires `make_plan(…; variable_npts = true)`; a plan
built with the default [`FixedCountNodes`](@ref) throws instead of silently reallocating.

With the count unchanged this rewrites array contents only and allocates nothing, on either node-set
form.
"""
function set_nodes!(plan::AbstractNUSHTplan, θ_nodes, φ_nodes)
    @assert length(θ_nodes) == length(φ_nodes)
    _set_nodes!(plan.nodes, θ_nodes, φ_nodes)
    _sync_θshift!(_θshift(plan), _θnufft(plan), _shift_offset(plan))
    _nufft_setpts!(_nufft2(plan), _θnufft(plan), _φnodes(plan))
    _nufft_setpts!(_nufft1(plan), _θnufft(plan), _φnodes(plan))
    return plan
end

# `θ_nufft` either aliases `θ_nodes` (scalar plan, nothing to do) or is its negation (spin plan).
@inline function _sync_θnufft!(nd)
    nd.θ_nufft === nd.θ_nodes || (nd.θ_nufft .= .-nd.θ_nodes)
    return nd
end

# `θ_shift` also moves with the nodes, but its offset is fixed by the bandlimit, which lives on the
# plan rather than the node set — so it is re-derived here rather than inside `_sync_θnufft!`.
@inline _sync_θshift!(::Nothing, _θ, _N0) = nothing
@inline _sync_θshift!(s::AbstractVector, θ, N0) = (s .= cis.(N0 .* θ); s)

# Centered order labels a length-`n` axis from `-(n÷2)`, so that is the offset to undo. Read off the
# mode buffer rather than recomputed from the bandlimit, so the two can never drift apart.
@inline _shift_offset(plan::NUSHTplan{T}) where {T} = T(size(plan.Fhat, 1) ÷ 2)

# Same count: rewrite contents, no allocation, works on either node-set form.
function _set_nodes_inplace!(nd, θ_nodes, φ_nodes)
    copyto!(nd.θ_nodes, θ_nodes)
    copyto!(nd.φ_nodes, φ_nodes)
    return _sync_θnufft!(nd)
end

function _set_nodes!(nd::FixedCountNodes, θ_nodes, φ_nodes)
    M = length(nd.θ_nodes)
    length(θ_nodes) == M || throw(DimensionMismatch(
        "this plan holds $M points and its node set has a fixed count, but got $(length(θ_nodes)). " *
        "Build it with variable_npts = true to allow the count to change."))
    return _set_nodes_inplace!(nd, θ_nodes, φ_nodes)
end

function _set_nodes!(nd::VariableCountNodes, θ_nodes, φ_nodes)
    M = length(θ_nodes)
    M == length(nd.θ_nodes) && return _set_nodes_inplace!(nd, θ_nodes, φ_nodes)
    aliased = nd.θ_nufft === nd.θ_nodes
    B = size(nd.fbuf, 2)
    nd.θ_nodes = _resized_like(nd.θ_nodes, θ_nodes, M)
    nd.φ_nodes = _resized_like(nd.φ_nodes, φ_nodes, M)
    nd.θ_nufft = aliased ? nd.θ_nodes : similar(nd.θ_nodes, eltype(nd.θ_nodes), M)
    nd.fbuf = _zeros_like(nd.θ_nodes, eltype(nd.fbuf), M, B)
    return _sync_θnufft!(nd)
end

"""
    close!(plan::NUSHTplan)

Eagerly free the FINUFFT guru plans owned by `plan` (otherwise freed by their finalizers). Safe to
call more than once (`finufft_destroy!` is idempotent).
"""
function close!(plan::AbstractNUSHTplan)
    _nufft_destroy!(_nufft2(plan))
    _nufft_destroy!(_nufft1(plan))
    return nothing
end

# Custom `show`: the default field-by-field display recurses into the stored FFTW plan, whose
# printer (`fftw_sprint_plan`) can segfault on a plan whose C state has been invalidated (e.g. after
# `close!`, or across a `Distributed` worker). Print a safe one-line summary instead — this is what
# Test/REPL/error-display call when a `NUSHTplan` is in scope.
Base.show(io::IO, plan::NUSHTplan{T}) where {T} =
    print(io, "NUSHTplan{", T, "}(lmax=", plan.lmax, ", M=", length(_θnodes(plan)), ", B=", plan.B, ", tol=", plan.tol, ")")
