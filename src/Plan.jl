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

export AbstractNUSHTplan, NUSHTplan, make_plan, close!, set_nodes!, plan_memory
export AbstractNodeSet, FixedCountNodes, VariableCountNodes
export AbstractPlanTuning, NoTuning, AutoTuning, ThoroughTuning
export AbstractPlanDirections, SynthesisOnly, SynthesisAndAnalysis

# The NUFFT seam lives in NUFFT.jl. `_host` is a no-op for an `Array` and copies a device array's
# coords to host; points are set once, so it is off the hot path.
@inline _host(x::Array) = x
@inline _host(x::AbstractArray) = Array(x)

# `upsampfac` is omitted rather than passed as 0, so an untuned plan gets the library's own default.
# These are FINUFFT options; backends without them ignore the keywords.
#
# `dtype` is the precision; `strengths` is the element type of the non-uniform data, which is what
# selects a real-data transform on a backend that has one (`_real_capable`) — the same way
# `NonuniformFFTs.PlanNUFFT` takes it. A backend without one ignores it.
@inline _make_nufft(backend, nodes, type, n_modes, iflag, B, tol, ::Type{T}, modeord, nthreads,
                    upsampfac, ::Type{Z} = Complex{T}) where {T,Z} =
    upsampfac > 0 ?
    _nufft_makeplan(backend, nodes, type, n_modes, iflag, B, tol;
                    dtype = T, strengths = Z, modeord = modeord, nthreads = nthreads,
                    upsampfac = upsampfac) :
    _nufft_makeplan(backend, nodes, type, n_modes, iflag, B, tol;
                    dtype = T, strengths = Z, modeord = modeord, nthreads = nthreads)

# Stored height of the θ mode axis for a plan whose non-uniform data has element type `Z`. A real `Z`
# selects a real-data transform, which is handed only the `kθ ≥ 0` half — the r2c count `n÷2+1` — and
# supplies the Hermitian remainder itself. One rule, used by `make_plan` and by the tuning search, so a
# trial plan can never be shaped differently from the plan it is tuning.
@inline _stored_modes(n1::Integer, ::Type{Z}) where {Z} = Z <: Real ? Int(n1) ÷ 2 + 1 : Int(n1)

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
const _NUFFT_TUNING =
    Dict{Tuple{Int,Int,Int,Int,DataType,Float64,Int,DataType,DataType,DataType},Tuple{Int,Float64}}()

"""
    AbstractPlanDirections

Which transform directions a plan is built for, so a caller who will only synthesise need not construct
an analysis plan at all.

The saving is modest, not structural: a backend allocates most of its working memory when a transform
*executes*, not when its plan is built, so an analysis plan that is never executed was never costing
much. What this buys is that `nusht_type1!` on such a plan fails immediately and says which keyword to
change, instead of silently working and quietly holding a second plan.
"""
abstract type AbstractPlanDirections end

"""
    SynthesisAndAnalysis <: AbstractPlanDirections

Build both directions — the default. Required by [`nusht_type1!`](@ref), [`nusht_solve!`](@ref) and
[`nusht_filter!`](@ref).
"""
struct SynthesisAndAnalysis <: AbstractPlanDirections end

"""
    SynthesisOnly <: AbstractPlanDirections

Build the synthesis (type-2) plan only. [`nusht_type2!`](@ref) works; anything needing the adjoint
throws and names the keyword to change.
"""
struct SynthesisOnly <: AbstractPlanDirections end

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

# The S-step is `plan_sph2fourier` and its adjoint — the whole of it. Its output is already the DFS
# bivariate Fourier series, so no grid synthesis or analysis plan is needed. `plan_sph2fourier` is the
# butterfly step and takes no FFTW flags; the planner count is pinned anyway, since FastTransforms and
# FFTW.jl share one libfftw3 and the count is baked into a plan when it is built.
function _build_sph_plans(Fslice, nt::Integer, flags::Integer)
    return _with_fftw_planner_nthreads(nt) do
        P = FastTransforms.plan_sph2fourier(Fslice)
        # `P'` leaves `AdjointFTPlan.adjoint` undefined, which FastTransforms then resolves through an
        # `UndefRefError` on every `lmul!`. Name the parent explicitly.
        return (P, FastTransforms.AdjointFTPlan(P, P))
    end
end

"""
    _pool_sizes(B) -> Vector{Int}

Working batch sizes a solve may shrink to: powers of two up to `B`, plus `B` itself. Powers of two
bound the number of plan sets at `log2(B)+1` while never wasting more than a factor of two of transform
width, so a column retiring early costs at most one halving of unused work.
"""
function _pool_sizes(B::Integer)
    B <= 1 && return Int[]
    s = [1 << i for i in 0:floor(Int, log2(B))]
    s[end] == B || push!(s, Int(B))
    return s
end

"""
    _nufft_size_pool(backend, nufft_type2, nufft_type1, widths, build) -> pool

Seam: the store of reduced-width plan sets a batched solve narrows into, filled on demand through
[`_pool_lookup!`](@ref). Its *shape* is the backend's, because what such a store can be depends on how
that backend types its plans, which nothing in `src` can know.

The default is an empty `Vector` typed from the plan's own full-width pair — correct for any backend
that keeps the transform count out of its plan's type, since one element type then covers every width.
A backend that puts the count in the type overrides this (see the NonuniformFFTs extension) with a
store whose slots are typed per width.

Either way nothing is built here: only a solve that actually narrows pays for a width, and `build(k)`
is what it pays with. An empty `widths` disables narrowing and nothing is ever stored.
"""
function _nufft_size_pool(backend, nufft_type2, nufft_type1, widths::AbstractVector{Int}, build)
    # The precondition this default rests on, checked rather than assumed: a backend that types its
    # plans by width must override the seam, and without this it would instead fail later and obscurely,
    # on a `push!` that cannot convert.
    isempty(widths) || _width_polymorphic(backend) || throw(ArgumentError(
        "$(nameof(typeof(backend))) types its plans by transform count, so the default width pool " *
        "cannot hold them; its extension must define `_nufft_size_pool`."))
    E = @NamedTuple{k::Int, nufft_type2::typeof(nufft_type2), nufft_type1::typeof(nufft_type1)}
    pool = E[]
    sizehint!(pool, length(widths))   # final capacity is known, so filling it never regrows
    return pool
end

"""
    _pool_recipe(backend, nt2, uf2, nt1, uf1)

The build inputs a narrower plan set needs that cannot be recovered from a plan: the resolved NUFFT
backend and the tuned thread/upsampling settings. Everything else — mode counts, `modeord`, tolerance,
nodes, realness — is read back off the plan when a width is built.

`narrowable` asks only whether the backend can re-plan at a reduced width at all. *How* those plans are
stored — grown on demand or built together — is [`_nufft_size_pool`](@ref)'s business, since it depends
on the backend's own typing rather than on anything `src` can see.
"""
_pool_recipe(backend, nt2, uf2, nt1, uf1) =
    (backend = backend, nt2 = Int(nt2), uf2 = Float64(uf2), nt1 = Int(nt1), uf1 = Float64(uf1),
     narrowable = _width_narrowable(backend))

"""
    _build_width!(plan, k) -> entry

Build and cache the NUFFT plan pair for working width `k`, returning it.

Mutates `plan.size_pool`, so it is not safe to call concurrently on a shared plan — the same
restriction a plan already carries, since a FINUFFT guru plan cannot be executed concurrently either.
"""
function _build_width!(plan, k::Integer)
    r = plan.pool_recipe
    T = real(eltype(plan.Fhat))
    θn, φn = _θnufft(plan), _φnodes(plan)
    # The strengths type comes off the plan's own buffer, so a narrower set is the same transform as the
    # full-width one rather than always the complex one — and that is also what says which θ count to
    # build: a real transform covers all `2lmax+3` wavenumbers and stores half, a complex half-height one
    # is built at the stored size.
    Z = eltype(_fbuf(plan))
    n_modes = Int64[Z <: Real ? 2plan.lmax + 3 : size(plan.Fhat, 1), plan.Nφ]
    n2 = _make_nufft(r.backend, θn, 2, n_modes, +1, k, plan.tol, T, 0, r.nt2, r.uf2, Z)
    _nufft_setpts!(n2, θn, φn); _nufft_finalize!(n2)
    n1 = if plan.nodes.nufft_type1 === nothing
        nothing                                   # mirror the plan's own directions
    elseif _nufft_share_directions(r.backend)
        _nufft_as_type1(n2)                       # same object, opposite direction — see the seam
    else
        p = _make_nufft(r.backend, θn, 1, n_modes, -1, k, plan.tol, T, 0, r.nt1, r.uf1, Z)
        _nufft_setpts!(p, θn, φn); _nufft_finalize!(p)
        p
    end
    return (k = Int(k), nufft_type2 = n2, nufft_type1 = n1)
end

"""
    _pool_lookup!(pool, k, build) -> entry

The plan pair for working width `k`, built by `build(k)` and stored on a miss. Storing is the pool's
own business because its shape is the backend's — see [`_nufft_size_pool`](@ref) — so this is a seam
too: the default appends to a `Vector`, while a backend whose widths are distinct types overrides it.
"""
function _pool_lookup!(pool::AbstractVector, k::Integer, build)
    @inbounds for e in pool
        e.k == k && return e
    end
    e = build(k)
    push!(pool, e)
    return e
end

# A pool of one slot per width, each slot a `Ref` whose element type was *derived* rather than obtained
# by building anything (see the NonuniformFFTs extension). Naming a type costs nothing, so a width
# nobody uses is never built and never specialised, while the width that is used lands in a concretely
# typed slot and stays there for the plan's life. The walk is over a tuple, so it unrolls.
# The entry is handed to `f` rather than returned: slots of different widths hold *different* concrete
# types, so a returned entry would be a `Union`, and a `Union` of a handle that wraps a large immutable
# is boxed on the heap every call. Passing it in keeps each arm on one type.
#
# Both arms reach `f` through something that strips `Nothing` — the `=== nothing` test on a hit, and
# `something` after a miss has filled the slot — so `f` is specialised on the slot's own entry type.
@inline function _with_pool_entry(f::F, pool::Tuple, k::Integer, build) where {F}
    slot = first(pool)
    slot.k == k || return _with_pool_entry(f, Base.tail(pool), k, build)
    e = slot.pair[]
    e === nothing || return f(e)
    slot.pair[] = build(k)
    return f(something(slot.pair[]))
end
_with_pool_entry(f::F, ::Tuple{}, k::Integer, build) where {F} = f(build(k))

# One element type, so there is no union to split and the plain lookup serves.
_with_pool_entry(f::F, pool::AbstractVector, k::Integer, build) where {F} =
    f(_pool_lookup!(pool, k, build))

"""
    _pool_built(pool) -> Int

How many reduced widths currently hold plans. The two pool shapes express "not built" differently — an
on-demand `Vector` is simply shorter, per-width slots hold `nothing` — so this is what a caller (or a
test) should ask rather than `isempty`, which means opposite things in the two cases.
"""
_pool_built(pool::AbstractVector) = length(pool)
_pool_built(pool::Tuple) = count(s -> s.pair[] !== nothing, pool)

# Free one slot's plans, if it ever held any, and empty it so `close!` stays idempotent.
function _release_width!(slot)
    e = slot.pair[]
    e === nothing && return nothing
    _nufft_destroy!(e.nufft_type2)
    _nufft_destroy!(e.nufft_type1)
    slot.pair[] = nothing
    return nothing
end

"""
    _sph_pool(Fslice, ntasks, nt, flags)

One `Fslice` and one set of sphere plans per task, for threading the per-column sphere loops. A
FastTransforms plan is not safe to apply concurrently, and the loop's slice buffer is shared, so both
have to be replicated. Empty when there is nothing to thread, so a single-threaded plan pays nothing.

Index the result **per task, never by `Threads.threadid()`**: tasks migrate between threads, so a
`threadid()` read at one point need not hold later and two tasks can end up sharing one entry — a data
race, not merely a bad index (it can also exceed `nthreads()` outright when an interactive pool
exists). A spawned task owns its chunk index for its whole lifetime, so that is the safe key.
"""
function _sph_pool(Fslice, ntasks::Integer, nt::Integer, flags::Integer)
    ntasks <= 1 && return typeof((similar(Fslice), _build_sph_plans(Fslice, nt, flags)...))[]
    return [(similar(Fslice), _build_sph_plans(Fslice, nt, flags)...) for _ in 1:ntasks]
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

# One application of the S-step and its adjoint. The reset copy is in every candidate's time, so it
# cannot bias the argmin, and it stops repeated application drifting the values.
function _apply_sph!(P, Padj, src, scratch)
    copyto!(scratch, src)
    LinearAlgebra.lmul!(P, scratch)
    LinearAlgebra.lmul!(Padj, scratch)
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
        P, Padj = _build_sph_plans(Fslice, nt, flags)
        warmed || (_apply_sph!(P, Padj, src, scratch); warmed = true)
        t = _time_candidate(() -> _apply_sph!(P, Padj, src, scratch), best_t)
        if t < best_t * _TUNING_MARGIN
            best_t = t
            best_cfg = (nt, UInt32(flags))
        end
    end
    Base.lock(() -> (_SPH_TUNING[key] = best_cfg), _PLANNER_LOCK)
    return best_cfg
end

"""
    _tune_nufft(backend, θ, φ, n_modes, type, iflag, B, T, tol, modeord, tuning, Z) -> (nthreads, upsampfac)

Pick the backend's thread count and oversampling factor by timing trial plans on the real node set.
`Z` is the non-uniform data element type, so a trial plan is built for the same transform the real
plan will use — a real `Z` selects a real-data transform, whose cost is not the complex one's.
Memoized on the problem shape with `M` bucketed to a power of two, so a stream of nearby point counts
tunes once. Host `Array` nodes only: a device node set takes the library defaults, its plan being
built through the cuFINUFFT seam that this host timing loop cannot exercise.
"""
_tune_nufft(backend, ::AbstractArray, ::AbstractArray, n_modes, type, iflag, B, ::Type{T}, tol,
            modeord, ::AbstractPlanTuning, ::Type{Z} = Complex{T}) where {T,Z} = (0, 0.0)

# Candidate `(nthreads, upsampfac)` pairs for FINUFFT. Both zeros are its sentinels: `nthreads = 0` is
# "all cores", `upsampfac = 0.0` is "library chooses" (`finufft_opts.h`: 2.0 std, 1.25 small FFT, 0.0
# auto). Auto is included because it is usually right — the search exists for the cases where it is not.
_nufft_candidates(::NoTuning) = Tuple{Int,Float64}[]
_nufft_candidates(::AbstractPlanTuning) =
    [(nt, uf) for uf in (0.0, 2.0, 1.25) for nt in Int[0; _thread_candidates()]]

# NonuniformFFTs reaches only two thread counts (1 and `Threads.nthreads()`), so pairing them with σ
# would double the search to compare a plan against a serial one. Its oversampling factor is a real
# choice: the half-support needed for a given `tol` is derived from σ analytically, so a smaller σ
# shrinks the FFT and grows the spreading and the winner depends on how the point count compares with
# the mode count. Accuracy is identical across the candidates by construction, so timing alone decides.
_nufft_candidates(::NonuniformFFTsBackend, ::NoTuning) = Tuple{Int,Float64}[]
_nufft_candidates(::NonuniformFFTsBackend, ::AbstractPlanTuning) =
    [(0, uf) for uf in (2.0, 1.5, 1.25)]

function _tune_nufft(backend::Union{FINUFFTBackend,NonuniformFFTsBackend}, θ::Array, φ::Array,
                     n_modes, type::Integer, iflag::Integer, B::Integer, ::Type{T}, tol::Float64,
                     modeord::Integer, tuning::AbstractPlanTuning,
                     ::Type{Z} = Complex{T}) where {T,Z}
    candidates = backend isa NonuniformFFTsBackend ? _nufft_candidates(backend, tuning) :
                                                     _nufft_candidates(tuning)
    isempty(candidates) && return (0, 0.0)
    M = length(θ)
    key = (Int(n_modes[1]), Int(n_modes[2]), Int(B), Int(type), T, tol,
           prevpow(2, max(M, 1)), typeof(tuning), Z, typeof(backend))
    cached = Base.lock(() -> get(_NUFFT_TUNING, key, nothing), _PLANNER_LOCK)
    cached === nothing || return cached

    modes = zeros(Complex{T}, _stored_modes(n_modes[1], Z), n_modes[2], B)
    strengths = zeros(Z, M, B)
    inp, outp = type == 2 ? (modes, strengths) : (strengths, modes)
    best_cfg = (0, 2.0)
    best_t = Inf
    for (nt, upsampfac) in candidates
        p = _make_nufft(backend, θ, type, n_modes, iflag, B, tol, T, modeord, nt, upsampfac, Z)
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
    nufft_type1::N1                     # `Nothing` for a synthesis-only plan; see `directions`
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
    const nufft_type1::N1               # `Nothing` for a synthesis-only plan; see `directions`
end

_node_set(::Val{false}, θ, φ, θn, θs, fbuf, p2, p1) = FixedCountNodes(θ, φ, θn, θs, fbuf, p2, p1)
_node_set(::Val{true}, θ, φ, θn, θs, fbuf, p2, p1) = VariableCountNodes(θ, φ, θn, θs, fbuf, p2, p1)

# Re-point the analysis plan with the nodes. A derived handle shares the plan just pointed, and a
# synthesis-only plan has none.
@inline _repoint_analysis!(::Nothing, θn, φn) = nothing
@inline _repoint_analysis!(p1, θn, φn) = _nufft_derived(p1) ? nothing : _nufft_setpts!(p1, θn, φn)

# The analysis plan, or `nothing` when the caller declined it — on every backend, so which calls a plan
# answers does not depend on which library is loaded.
_build_analysis(::SynthesisOnly, backend, θ, φ, n_modes, B, tol, ::Type{T}, modeord, nt, uf,
                ::Type{Z}, p2) where {T,Z} = nothing

function _build_analysis(::SynthesisAndAnalysis, backend, θ, φ, n_modes, B, tol, ::Type{T}, modeord,
                         nt, uf, ::Type{Z}, p2) where {T,Z}
    _nufft_share_directions(backend) && return _nufft_as_type1(p2)
    p1 = _make_nufft(backend, θ, 1, n_modes, -1, B, tol, T, modeord, nt, uf, Z)
    _nufft_setpts!(p1, θ, φ)
    _nufft_finalize!(p1)
    return p1
end

# Internal accessors for the point-dependent fields, so the rest of the package is written against
# one spelling regardless of which node-set form a plan carries.
@inline _θnodes(p) = p.nodes.θ_nodes
@inline _φnodes(p) = p.nodes.φ_nodes
@inline _θnufft(p) = p.nodes.θ_nufft
@inline _θshift(p) = p.nodes.θ_shift
@inline _fbuf(p) = p.nodes.fbuf
@inline _nufft2(p) = p.nodes.nufft_type2
@inline _nufft1(p) = _require_analysis(p.nodes.nufft_type1)

# Name the keyword to change, rather than failing on a `nothing` somewhere inside the transform.
@inline _require_analysis(p1) = p1
_require_analysis(::Nothing) = throw(ArgumentError(
    "this plan was built with `directions = SynthesisOnly()` and has no analysis plan. Rebuild with " *
    "`directions = SynthesisAndAnalysis()` to use nusht_type1!, nusht_solve! or nusht_filter!."))

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
- `C`: filter scratch, **empty until the first filter call** — see [`_filter_scratch`](@ref).
  `F`: the bivariate Fourier coefficients `P·C`. Both `(Nθ, Nφ, B)` with
  eltype `FE`
- `Fhat`: the complex mode array the NUFFT evaluates, assembled from `F` by
  [`_assemble_modes!`](@ref). The φ axis carries wavenumbers `-lmax…lmax` (`Nφ = 2lmax+1`), the θ axis
  `-(lmax+1)…lmax+1` (`2lmax+3`) — one further out, since an odd-order column of the coefficient array
  reaches θ-frequency `lmax+1`. Where the field is real *and* the NUFFT backend has a real-data
  transform, only the `kθ ≥ 0` half is stored (`lmax+2` rows) and that backend supplies the rest.
- `Fslice`: `(Nθ, Nφ)` scratch `Matrix` each batch slice is `copyto!`-ed through for the S-step.
  FastTransforms has no `Float32` sphere plan, so it stays double precision and the copy converts.
- `sph_plan`, `sph_plan_adj`: FastTransforms `plan_sph2fourier` (P) and its adjoint on a `(Nθ, Nφ)`
  slice. `P` alone is the whole S-step — its output is already the bivariate Fourier series, so
  synthesis is `P·C` → assemble → one NUFFT, with no equiangular grid in between.

`nodes.nufft_type2` is the guru type-2 plan (`iflag = +1`, synthesis N) and `nodes.nufft_type1` the
type-1 plan (`iflag = -1`, adjoint N†). `iflag = +1` supplies the reconstruction sign directly, so the
modes need no conjugate-transpose. The axis convention is `x = θ`, `y = φ`, and both mode axes are in
centered order (`modeord = 0`) — which the signed wavenumber ranges above already are, so nothing has
to be shifted per point.
"""
struct NUSHTplan{T<:AbstractFloat, FE<:Number, AT3<:AbstractArray{FE,3},
                 AT2<:AbstractMatrix, CT3<:AbstractArray{Complex{T},3},
                 ND<:AbstractNodeSet, SP, SPADJ, SPL, SZP, RCP,
                 FT} <: AbstractNUSHTplan
    lmax::Int
    Nθ::Int
    Nφ::Int
    B::Int
    tol::FT
    nodes::ND                     # the point-dependent half; see AbstractNodeSet
    C::Base.RefValue{Union{Nothing,AT3}}   # filter scratch, empty until used; see _filter_scratch
    F::AT3
    Fhat::CT3
    Fslice::AT2
    sph_plan::SP
    sph_plan_adj::SPADJ           # sph_plan' (P'), stored to keep the adjoint alloc-free
    sph_pool::SPL                 # per-task slice + plans for the threaded column loops; see _sph_pool
    size_pool::SZP                # narrower plan sets, filled on demand; see _nufft_size_pool
    pool_recipe::RCP              # what building one needs that the plan cannot supply
end

"""
    _filter_scratch(plan) -> AbstractArray

The plan's coefficient scratch, allocated on first use. Filtering scales coefficients before
synthesising and must not modify the caller's array, so it needs a buffer of its own — and nothing else
in the plan does. A plan that only synthesises, transforms or solves therefore never allocates one; a
filtering plan allocates it once and reuses it, so repeated filtering allocates nothing after the first
call.
"""
@inline function _filter_scratch(plan::NUSHTplan)
    c = plan.C[]
    c === nothing || return c
    fresh = _zeros_like(plan.F, eltype(plan.F), plan.Nθ, plan.Nφ, plan.B)
    plan.C[] = fresh
    return fresh
end

"""
    plan_memory(plan) -> NamedTuple

Bytes held by each of the plan's own buffers, plus their `total`, so a caller can see where a plan's
memory goes and what `upsampfac`, `ntrans` or the element type do to it.

This counts what the plan allocates in Julia. A backend holding its oversampled grid in C (FINUFFT) is
invisible to any Julia-side accounting, `Base.summarysize` included — measure that as the resident set
of a process that builds one plan. An empty slot counts as zero, which is the point of it.
"""
function plan_memory(plan::NUSHTplan)
    slot(r) = r[] === nothing ? 0 : Base.summarysize(r[])
    C      = slot(plan.C)
    F      = Base.summarysize(plan.F)
    Fhat   = Base.summarysize(plan.Fhat)
    Fslice = Base.summarysize(plan.Fslice)
    fbuf   = Base.summarysize(_fbuf(plan))
    nodes  = Base.summarysize(_θnodes(plan)) + Base.summarysize(_φnodes(plan))
    pool   = Base.summarysize(plan.size_pool)
    sph    = Base.summarysize(plan.sph_pool)
    return (; C, F, Fhat, Fslice, fbuf, nodes, size_pool = pool, sph_pool = sph,
            total = C + F + Fhat + Fslice + fbuf + nodes + pool + sph)
end

"""
    make_plan([FE = Float64,] θ_nodes, φ_nodes, lmax; tol=1e-8, ntrans=1, tuning=NoTuning(), …)

`FE` is the field element type, positional as it is for `zeros(T, …)`. `Float64`/`Float32` assert the
field VALUES are real, which makes the mode array conjugate-symmetric in `kθ`; on a NUFFT backend with
a real-data transform (`NonuniformFFTsBackend`) only the `kθ ≥ 0` half is then built and only real
strengths come back, halving the mode array, the upsampled FFT *and* the spreading.
`ComplexF64`/`ComplexF32` build the full array. Both are the same spherical harmonic transform; only
the symmetry exploited differs.

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
- `nthreads`, `upsampfac`: override the NUFFT backend's thread count (`0` is "let the backend choose")
  and upsampling factor. What a backend can reach differs: FINUFFT takes any count, while
  NonuniformFFTs parallelises over Julia's own threads and so reaches only `1` and `Threads.nthreads()`
  — a count it cannot deliver is an error, never silently something else.

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
    directions::AbstractPlanDirections = SynthesisAndAnalysis(),
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
    M = length(θ_nodes)

    # Nodes keep their input array type (host `Vector` or device array), eltype coerced to `T`; every
    # buffer is allocated `similar` to the nodes, so a device node set yields a device-resident plan.
    θ = T.(θ_nodes)
    φ = T.(φ_nodes)

    # `plan_sph2fourier` already yields the DFS bivariate Fourier series, so the NUFFT's mode array is
    # assembled straight from it (`_assemble_modes!`). Both axes are signed and centered, which is
    # `modeord = 0` exactly, with no offset to undo per point. θ reaches lmax+1 where φ reaches lmax:
    # the supernumerary slots of the square coefficient array carry degrees up to `lmax+|m|`, and an
    # odd-order column's last row is the sine at frequency `lmax+1`.
    nub = _resolve_nufft(nufft, FE)
    # A real field's mode array is exactly Hermitian, so only `kθ ≥ 0` is ever stored — `lmax+2` rows
    # rather than `2lmax+3`. That halves the deconvolution and the FFT and leaves the interpolation
    # untouched, so it is never more work than the full array at any point count.
    #
    # How the other half is supplied splits the two cases. A backend with a real-data transform
    # (`_real_capable`) is handed the half-spectrum of a transform built for the full θ axis and
    # reconstructs the rest itself, with real strengths — so the spreading halves too, and the forward
    # needs no weights. Without one, the transform IS half-height and complex, its centered rows are
    # therefore labelled `kθ - Nkstore÷2`, and the missing conjugate half has to be paid for explicitly:
    # `θ_shift` undoes the labelling per point and `_fold_weights!` doubles every `kθ > 0` row.
    Nk = 2lmax + 3
    r2c  = realfield && _real_capable(nub)
    fold = realfield
    _warn_if_directsum(nufft, nub, M, Nk * Nφ)
    ZS = r2c ? T : Complex{T}                 # element type of the non-uniform data
    Nkstore = fold ? Nk ÷ 2 + 1 : Nk          # `lmax+2` folded
    F    = _zeros_like(θ, FE, Nθ, Nφ, B)
    # Only filtering needs a coefficient scratch, and only so the caller's array is not modified. A
    # plan that synthesises, transforms or solves never allocates one; a filtering plan allocates it
    # once and reuses it, so repeated filtering stays allocation-free. See `_filter_scratch`.
    C    = Base.RefValue{Union{Nothing,typeof(F)}}(nothing)
    Fhat = _zeros_like(θ, Complex{T}, Nkstore, Nφ, B)
    fbuf = _zeros_like(θ, ZS, M, B)
    θ_shift = (fold && !r2c) ?
        _to_like(θ, Complex{T}.(cis.(T(Nkstore ÷ 2) .* θ))) : nothing

    # FastTransforms plans operate on a single dense (Nθ, Nφ) HOST `Matrix` (FastTransforms is CPU-only);
    # each batch slice is `copyto!`-ed through `Fslice` (host↔device for a device plan — the S-step is an
    # inherent host bounce). A persistent P avoids a per-call rebuild.
    # FastTransforms has Float64 and ComplexF64 sphere plans but no Float32 ones, so this buffer is
    # always double precision; the slice copy that was already happening does the conversion, and
    # every other buffer runs at `FE`.
    Fslice = zeros(realfield ? Float64 : ComplexF64, Nθ, Nφ)
    sph_nt, sph_flags = _tune_sph(Fslice, tuning)
    isnothing(ft_fftw_nthreads) || (sph_nt = Int(ft_fftw_nthreads))
    isnothing(ft_fftw_flags) || (sph_flags = UInt32(ft_fftw_flags))
    sph_plan, sph_plan_adj = _build_sph_plans(Fslice, sph_nt, sph_flags)
    # Replicated only when there is something to thread, so a single-threaded plan pays no build cost
    # and no memory. Capped at `B`: more tasks than columns cannot help.
    sph_pool = _sph_pool(Fslice, min(Threads.nthreads(), B), sph_nt, sph_flags)

    # iflag +1 for synthesis (type 2): reconstruction uses the +i (inverse-DFT) sign, so the raw
    # FFT modes need no conjugation — an axis swap takes the place of a conjugate-transpose
    # (conj(c)·e^{-ikx} = c·e^{+ikx}). type 1 (−1) is the exact adjoint.
    tol64 = Float64(tol)
    # A real transform is built for the full θ axis and stores its half; a complex half-height one is
    # built at the stored size.
    n_modes = Int64[r2c ? Nk : Nkstore, Nφ]
    modeord = 0                               # centered: both mode axes are signed and symmetric
    nt2, uf2 = _tune_nufft(nub, θ, φ, n_modes, 2, +1, B, T, tol64, modeord, tuning, ZS)
    nt1, uf1 = _tune_nufft(nub, θ, φ, n_modes, 1, -1, B, T, tol64, modeord, tuning, ZS)
    isnothing(nthreads) || (nt2 = nt1 = Int(nthreads))
    isnothing(upsampfac) || (uf2 = uf1 = Float64(upsampfac))
    # Where a backend's plan carries no direction, one object serves both and the second is a handle
    # onto it — that halves the oversampled grid and the sorted point copy the plan owns, and its
    # points are already set.
    nufft_type2 = _make_nufft(nub, θ, 2, n_modes, +1, B, tol64, T, modeord, nt2, uf2, ZS)
    _nufft_setpts!(nufft_type2, θ, φ)
    _nufft_finalize!(nufft_type2)
    # Deferred unless the backend serves both directions from one plan, in which case it already
    # exists. A synthesis-only caller then holds no type-1 grid at all.
    nufft_type1 = _build_analysis(directions, nub, θ, φ, n_modes, B, tol64, T, modeord, nt1, uf1, ZS,
                                  nufft_type2)
    # A scalar plan hands FINUFFT the colatitudes unchanged, so `θ_nufft` aliases `θ_nodes` and costs
    # no extra storage; a spin plan negates them and owns a separate array.
    # How the narrower plan sets are stored is the backend's call, through `_nufft_size_pool` — whether
    # a width can be added on demand depends on that backend's own typing. `_width_pair` is the builder
    # it uses, so an override never has to reach back into a half-built plan.
    _width_pair(k) = let
        n2 = _make_nufft(nub, θ, 2, n_modes, +1, k, tol64, T, modeord, nt2, uf2, ZS)
        n1 = _make_nufft(nub, θ, 1, n_modes, -1, k, tol64, T, modeord, nt1, uf1, ZS)
        _nufft_setpts!(n2, θ, φ); _nufft_setpts!(n1, θ, φ)
        _nufft_finalize!(n2); _nufft_finalize!(n1)
        (k = Int(k), nufft_type2 = n2, nufft_type1 = n1)
    end
    pool_recipe = _pool_recipe(nub, nt2, uf2, nt1, uf1)
    size_pool = _nufft_size_pool(nub, nufft_type2, nufft_type1,
                                 pool_recipe.narrowable ? _pool_sizes(B) : Int[], _width_pair)

    nodes = _node_set(Val(variable_npts), θ, φ, θ, θ_shift, fbuf, nufft_type2, nufft_type1)

    return NUSHTplan{T, FE, typeof(F), typeof(Fslice), typeof(Fhat), typeof(nodes),
                     typeof(sph_plan), typeof(sph_plan_adj),
                     typeof(sph_pool), typeof(size_pool), typeof(pool_recipe), typeof(tol64)}(
        lmax, Nθ, Nφ, B, tol64, nodes, C, F, Fhat, Fslice,
        sph_plan, sph_plan_adj, sph_pool, size_pool, pool_recipe,
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
    # A derived handle shares the plan just pointed; re-pointing would re-sort the same points.
    _repoint_analysis!(plan.nodes.nufft_type1, _θnufft(plan), _φnodes(plan))
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

"""
    _valid_mask(proto, T, lmax)

`1` at the `(lmax+1)^2` slots holding degrees `l ≤ lmax`, `0` at the `lmax(lmax+1)` supernumerary ones
holding `lmax < l ≤ lmax+|m|`. The index rule is FastTransforms' own (`sphones`): column 1 carries
`m = 0` for every degree, and the column pair `2j, 2j+1` carries `m = ∓j` for `l = j … lmax`, so it
occupies only its first `lmax+1-j` rows.

Shaped `(Nθ, Nφ, 1)` to broadcast over the batch, and built in the plan's array type so applying it is
device-resident and allocation-free.
"""
function _valid_mask(proto, ::Type{T}, lmax::Integer) where {T}
    Nθ, Nφ = lmax + 1, 2lmax + 1
    H = zeros(T, Nθ, Nφ, 1)
    H[:, 1, 1] .= one(T)
    for j in 1:lmax
        H[1:(Nθ - j), 2j, 1] .= one(T)
        H[1:(Nθ - j), 2j + 1, 1] .= one(T)
    end
    return _to_like(proto, H)
end

"""
    _valid_indices(proto, lmax) -> AbstractVector{Int}

Linear positions, within one coefficient column, of the `(lmax+1)^2` slots holding degrees
`l ≤ lmax` — the same set [`_valid_mask`](@ref) marks, as indices rather than a 0/1 array.

The solver carries its vectors packed to just these, so they are `(K, B)` with `K = (lmax+1)^2` rather
than `(Nθ, Nφ, B)` with `lmax(lmax+1)` entries per column pinned to zero. That halves four coefficient
arrays and, since every reduction and axpy in the iteration runs over them, halves that work too.
Built in the plan's array type so packing is device-resident.
"""
function _valid_indices(proto, lmax::Integer)
    Nθ = lmax + 1
    idx = Vector{Int}(undef, (lmax + 1)^2)
    k = 0
    @inbounds for i in 1:Nθ                       # column 1 is m = 0, every degree
        idx[k += 1] = i
    end
    @inbounds for j in 1:lmax, c in (2j, 2j + 1)  # column pair 2j, 2j+1 is m = ∓j, degrees j…lmax
        base = (c - 1) * Nθ
        for i in 1:(Nθ - j)
            idx[k += 1] = base + i
        end
    end
    k == length(idx) || throw(AssertionError("valid-slot count $k ≠ $(length(idx))"))
    return _to_like(proto, idx)
end

# Gather a packed `(K, B)` view out of a full `(Nθ, Nφ, B)` array, and scatter it back with the
# supernumerary slots zeroed. `n` bounds the columns, matching the solver's compaction. These replace
# a `copyto!` plus a mask multiply, so they move no more memory than the code they stand in for.
function _pack_coeffs!(dst, src, idx, srclen::Integer, n::Integer)
    K = length(idx)
    @inbounds for b in 1:n
        so = (b - 1) * srclen
        do_ = (b - 1) * K
        @simd for t in 1:K
            dst[do_ + t] = src[so + idx[t]]
        end
    end
    return dst
end

function _unpack_coeffs!(dst, src, idx, dstlen::Integer, n::Integer)
    K = length(idx)
    @inbounds for b in 1:n
        do_ = (b - 1) * dstlen
        so = (b - 1) * K
        @simd for i in 1:dstlen
            dst[do_ + i] = zero(eltype(dst))
        end
        @simd for t in 1:K
            dst[do_ + idx[t]] = src[so + t]
        end
    end
    return dst
end

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

# The narrower plan sets `_build_width!` caches own NUFFT plans of their own, so `close!` has to reach
# them; a spin plan has no pool. Leaving them to their finalizers is not merely untidy: FINUFFT's
# destructor calls back into Julia to take its FFTW lock, and acquiring a contended lock yields, which
# a GC finalizer may not do ("task switch not allowed from inside gc finalizer"). Destroying eagerly
# means the finalizer later finds an already-destroyed plan and returns without entering C.
_close_pool!(::AbstractNUSHTplan) = nothing
_close_pool!(plan::NUSHTplan) = _close_pool!(plan.size_pool)

# A `Vector` pool is emptied; any other shape is the backend's, so releasing an entry is delegated to
# `_release_width!` and the container itself is left alone (a `Tuple` of slots cannot be emptied — its
# slots are cleared instead, which is what makes `close!` idempotent there too).
function _close_pool!(pool::AbstractVector)
    for e in pool
        _nufft_destroy!(e.nufft_type2)
        _nufft_destroy!(e.nufft_type1)
    end
    empty!(pool)
    return nothing
end

function _close_pool!(pool::Tuple)
    for slot in pool
        _release_width!(slot)
    end
    return nothing
end

"""
    close!(plan::NUSHTplan)

Eagerly free every FINUFFT guru plan the plan owns — its own pair and any narrower pair
[`_build_width!`](@ref) cached for a compacted solve — rather than leaving them to their finalizers.
Safe to call more than once (`finufft_destroy!` is idempotent).

Eager destruction is not just tidiness. FINUFFT's destructor re-enters Julia to take its FFTW lock,
and acquiring that lock when contended yields; a GC finalizer cannot yield, so a pooled plan collected
under contention raises `task switch not allowed from inside gc finalizer`. A solve supplies both
halves of that on its own: it grows the pool as it narrows the batch width, and it allocates.
"""
function close!(plan::AbstractNUSHTplan)
    _nufft_destroy!(_nufft2(plan))
    _nufft_destroy!(plan.nodes.nufft_type1)
    _close_pool!(plan)
    return nothing
end

# Custom `show`: the default field-by-field display recurses into the stored FFTW plan, whose
# printer (`fftw_sprint_plan`) can segfault on a plan whose C state has been invalidated (e.g. after
# `close!`, or across a `Distributed` worker). Print a safe one-line summary instead — this is what
# Test/REPL/error-display call when a `NUSHTplan` is in scope.
Base.show(io::IO, plan::NUSHTplan{T}) where {T} =
    print(io, "NUSHTplan{", T, "}(lmax=", plan.lmax, ", M=", length(_θnodes(plan)), ", B=", plan.B, ", tol=", plan.tol, ")")
