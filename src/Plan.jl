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
using FINUFFT: FINUFFT

export NUSHTplan, make_plan, close!

# ── NUFFT backend seam ────────────────────────────────────────────────────────
# The transform cores + plan builders call these instead of `FINUFFT.*` directly. Execution/teardown
# dispatch on the stored plan type; plan *creation*/`setpts!` dispatch on the node array type (so a
# `CuArray` node set selects cuFINUFFT). The CPU (FINUFFT) methods are here; the device (cuFINUFFT)
# methods are added by the CUDA extension — so the cores + builders stay backend-free. `cufinufft_plan`
# is predeclared in FINUFFT's always-loaded core, so the downstream device methods need no change here.
# FINUFFT and cuFINUFFT share `setkwopts!`, so identical keyword opts (dtype/modeord/nthreads) forward
# to both — the seam passes them through unchanged.
@inline _nufft_exec!(p::FINUFFT.finufft_plan, input, output) = FINUFFT.finufft_exec!(p, input, output)
@inline _nufft_destroy!(p::FINUFFT.finufft_plan) = FINUFFT.finufft_destroy!(p)
@inline _nufft_makeplan(::AbstractVector, type, n_modes, iflag, ntrans, tol; kwargs...) =
    FINUFFT.finufft_makeplan(type, n_modes, iflag, ntrans, tol; kwargs...)
# Host FINUFFT needs host coordinate vectors. `_host` is a no-op for a host `Array` and copies a
# device-array's coords to host (nodes are set once, so the copy is not on any hot path). A real
# `CuArray` node set instead selects the cuFINUFFT `_nufft_setpts!` in the CUDA extension.
@inline _host(x::Array) = x
@inline _host(x::AbstractArray) = Array(x)
@inline _nufft_setpts!(p::FINUFFT.finufft_plan, x, y) = FINUFFT.finufft_setpts!(p, _host(x), _host(y))

# Backend-generic zeroed buffer shaped like `ref` (host `Array` for CPU nodes, device array for GPU
# nodes) — used so a device node set yields device-resident plan buffers.
@inline _zeros_like(ref::AbstractArray, ::Type{S}, dims::Integer...) where {S} =
    fill!(similar(ref, S, dims...), zero(S))

# A copy of host array `a` moved to `ref`'s backend (host stays host, device→device). Used for the
# small precomputed phase vectors so they match a device node set (a device broadcast against a host
# vector would fail / be wrong).
@inline _to_like(ref::AbstractArray, a::AbstractArray) = copyto!(similar(ref, eltype(a), size(a)...), a)

"""
    NUSHTplan{T}

Pre-computed plan for non-uniform spherical harmonic transforms at `M` scattered points, up to
degree `lmax`, transforming `B` co-located fields per call (`ntrans = B`).

Fields:
- `lmax`, `Nθ = lmax+1`, `Nφ = 2lmax+1`, `B` (batch size / FINUFFT `ntrans`)
- `tol`: FINUFFT accuracy tolerance
- `θ_nodes`, `φ_nodes`: colatitudes ∈ [0,π] and longitudes ∈ [0,2π) of the `M` points
- `C`: real coefficient scratch `(Nθ, Nφ, B)` (used by `nusht_filter!`/`nusht_solve!`)
- `F`: real CC map `(Nθ, Nφ, B)`
- `F̃`: complex doubled-torus map `(2Nθ, Nφ, B)` (FFT input)
- `Fhat`: complex Fourier modes `(2Nθ, Nφ, B)` (FFT output / FINUFFT mode array)
- `fbuf`: complex strengths `(M, B)` (FINUFFT nonuniform values)
- `phase_scaled`: `exp(-iπ k_θ / 2Nθ) / (2Nθ·Nφ)`, length `2Nθ` — fused half-pixel θ correction +
  forward FFT normalization
- `phase_conj`: `conj(exp(-iπ k_θ / 2Nθ))`, length `2Nθ` — adjoint half-pixel correction
- `fft_plan`, `ifft_plan`: FFTW plans over dims `(1,2)` of the doubled buffers
- `Fslice`: real `(Nθ, Nφ)` scratch `Matrix` — FastTransforms plans require a concrete `Matrix`, so
  each batch slice is `copyto!`-ed through this buffer for the S-step
- `sph_plan`, `sph_plan_synth`, `sph_plan_analysis`: FastTransforms `plan_sph2fourier` (P),
  `plan_sph_synthesis` (PS), `plan_sph_analysis` (PA) on a `(Nθ, Nφ)` slice. Forward S = `PS·P`
  (`sph_evaluate!`); CC-inverse S⁻¹ = `P⁻¹·PA` (`nusht_type1!`, replicating `sph_transform!` with
  persistent plans instead of its per-call rebuild); Euclidean adjoint S† = `P'·PS'`
  (`_nusht_true_adjoint!`)
- `nufft_type2`: FINUFFT guru type-2 plan (`iflag=+1`, `modeord=1`), synthesis N
- `nufft_type1`: FINUFFT guru type-1 plan (`iflag=-1`, `modeord=1`), adjoint N†

The FINUFFT axis convention is `x = θ` (`ms = 2Nθ`), `y = φ` (`mt = Nφ`); with `modeord = 1` the
FFTW-native mode order is used directly, so no `fftshift`/transpose is needed. `iflag = +1` on the
synthesis plan supplies the reconstruction sign that the old code obtained via a conjugate-transpose.
"""
struct NUSHTplan{T<:AbstractFloat, RV<:AbstractVector{T}, AT3<:AbstractArray{T,3},
                 AT2<:AbstractMatrix{T}, CT3<:AbstractArray{Complex{T},3},
                 CT2<:AbstractArray{Complex{T},2}, CV<:AbstractVector{Complex{T}},
                 FP, IP, SP, SPS, SPA, SPADJ, SPSADJ, N1, N2, FT}
    lmax::Int
    Nθ::Int
    Nφ::Int
    B::Int
    tol::FT
    θ_nodes::RV
    φ_nodes::RV
    C::AT3
    F::AT3
    F̃::CT3
    Fhat::CT3
    fbuf::CT2
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
    nufft_type2::N2
    nufft_type1::N1
end

"""
    make_plan(θ_nodes, φ_nodes, lmax; tol=1e-8, T=Float64, ntrans=1, nthreads=0)

Construct a `NUSHTplan` for `M` scattered points at colatitudes `θ_nodes ∈ [0,π]` and longitudes
`φ_nodes ∈ [0,2π)`, up to spherical harmonic degree `lmax`, transforming `ntrans` co-located fields
per call. Builds the FINUFFT guru plans and sets the nonuniform points once; they are freed by a
finalizer (or eagerly via [`close!`](@ref)).

Keyword arguments:
- `tol`: FINUFFT accuracy tolerance.
- `T`: floating-point type (`Float64`/`Float32`).
- `ntrans`: batch size `B` — transform `B` co-located fields (same nodes) per call.
- `nthreads`: FINUFFT threads for the NUFFT; **`0` (default) uses all available cores** (fastest for
  a single transform). Set a small count (e.g. `1`) when running many plans concurrently via the
  parallel extensions, to avoid oversubscription.

FINUFFT accepts coordinates in `[-3π, 3π]`, so natural `[0,π]`/`[0,2π)` coordinates are passed
directly.
"""
function make_plan(
    θ_nodes,
    φ_nodes,
    lmax;
    tol = 1e-8,
    T::Type{<:AbstractFloat} = Float64,
    ntrans::Integer = 1,
    nthreads::Integer = 0,
)
    @assert length(θ_nodes) == length(φ_nodes)
    B = Int(ntrans)
    @assert B ≥ 1

    Nθ = lmax + 1
    Nφ = 2lmax + 1
    Nθ_dbl = 2Nθ
    M = length(θ_nodes)

    # Nodes keep their input array type (host `Vector` or device array), eltype coerced to `T`; every
    # buffer is allocated `similar` to the nodes, so a device node set yields a device-resident plan.
    θ = T.(θ_nodes)
    φ = T.(φ_nodes)

    C    = _zeros_like(θ, T, Nθ, Nφ, B)
    F    = _zeros_like(θ, T, Nθ, Nφ, B)
    F̃    = _zeros_like(θ, Complex{T}, Nθ_dbl, Nφ, B)
    Fhat = _zeros_like(θ, Complex{T}, Nθ_dbl, Nφ, B)
    fbuf = _zeros_like(θ, Complex{T}, M, B)

    # FFTs act on the (θ̃, φ) plane only; the batch axis is left untransformed. `AbstractFFTs.plan_fft`
    # resolves to FFTW for a host `Array` (byte-identical to `FFTW.plan_fft`) and to CUFFT for a CuArray.
    fft_plan  = AbstractFFTs.plan_fft(F̃, (1, 2))
    ifft_plan = AbstractFFTs.plan_ifft(Fhat, (1, 2))

    # Half-pixel θ phase for the CC cell-center offset, built host-side in FFTW-native (modeord=1) k
    # order, then moved to the node backend so the `Fhat .*= phase_*` broadcast is device-resident.
    k_θ = [k < Nθ_dbl ÷ 2 ? k : k - Nθ_dbl for k in 0:(Nθ_dbl - 1)]
    phase        = Complex{T}.(cis.(-π .* T.(k_θ) ./ Nθ_dbl))
    phase_scaled = _to_like(θ, phase ./ (Nθ_dbl * Nφ))
    phase_conj   = _to_like(θ, conj.(phase))

    # FastTransforms plans operate on a single dense (Nθ, Nφ) HOST `Matrix` (FastTransforms is CPU-only);
    # each batch slice is `copyto!`-ed through `Fslice` (host↔device for a device plan — the S-step is an
    # inherent host bounce). Persistent P/PS/PA plans avoid `sph_transform!`'s per-call rebuild.
    Fslice             = zeros(T, Nθ, Nφ)
    sph_plan           = FastTransforms.plan_sph2fourier(Fslice)
    sph_plan_synth     = FastTransforms.plan_sph_synthesis(Fslice)
    sph_plan_analysis  = FastTransforms.plan_sph_analysis(Fslice)
    sph_plan_adj       = sph_plan'          # P'  (built once; `p'` per call would allocate)
    sph_plan_synth_adj = sph_plan_synth'    # PS'

    # iflag +1 for synthesis (type 2): reconstruction uses the +i (inverse-DFT) sign, so the raw
    # FFT modes need no conjugation. The old code used iflag −1 with a conjugate-transpose (`'`); the
    # conjugation there was load-bearing (conj(c)·e^{-ikx} = c·e^{+ikx}) — dropping the transpose in
    # favor of an axis swap requires flipping the sign instead. type 1 (−1) stays the exact adjoint.
    # `nthreads` is forwarded straight to FINUFFT: 0 (default) = ALL available cores (FINUFFT's
    # sentinel, not "zero threads") — fastest for a single transform; FastTransforms' Legendre step
    # is already multithreaded. Set a small count (e.g. 1) when running many plans concurrently via
    # the parallel extensions, to avoid oversubscription. (A multithreaded Julia adds small external
    # per-call allocation via FINUFFT's thread-safe FFTW-planner lock; NUFSHT's own code is
    # zero-alloc, exactly so single-threaded.)
    nthr = Int(nthreads)
    n_modes = Int64[Nθ_dbl, Nφ]
    nufft_type2 = _nufft_makeplan(θ, 2, n_modes, +1, B, Float64(tol); dtype = T, modeord = 1, nthreads = nthr)
    nufft_type1 = _nufft_makeplan(θ, 1, n_modes, -1, B, Float64(tol); dtype = T, modeord = 1, nthreads = nthr)
    _nufft_setpts!(nufft_type2, θ, φ)
    _nufft_setpts!(nufft_type1, θ, φ)
    finalizer(_nufft_destroy!, nufft_type2)
    finalizer(_nufft_destroy!, nufft_type1)

    tol64 = Float64(tol)
    return NUSHTplan{T, typeof(θ), typeof(C), typeof(Fslice), typeof(F̃), typeof(fbuf),
                     typeof(phase_scaled), typeof(fft_plan), typeof(ifft_plan), typeof(sph_plan),
                     typeof(sph_plan_synth), typeof(sph_plan_analysis), typeof(sph_plan_adj),
                     typeof(sph_plan_synth_adj), typeof(nufft_type1), typeof(nufft_type2), typeof(tol64)}(
        lmax, Nθ, Nφ, B, tol64, θ, φ, C, F, F̃, Fhat, fbuf, Fslice,
        phase_scaled, phase_conj, fft_plan, ifft_plan, sph_plan, sph_plan_synth, sph_plan_analysis,
        sph_plan_adj, sph_plan_synth_adj, nufft_type2, nufft_type1,
    )
end

"""
    close!(plan::NUSHTplan)

Eagerly free the FINUFFT guru plans owned by `plan` (otherwise freed by their finalizers). Safe to
call more than once (`finufft_destroy!` is idempotent).
"""
function close!(plan::NUSHTplan)
    _nufft_destroy!(plan.nufft_type2)
    _nufft_destroy!(plan.nufft_type1)
    return nothing
end

# Custom `show`: the default field-by-field display recurses into the stored FFTW plan, whose
# printer (`fftw_sprint_plan`) can segfault on a plan whose C state has been invalidated (e.g. after
# `close!`, or across a `Distributed` worker). Print a safe one-line summary instead — this is what
# Test/REPL/error-display call when a `NUSHTplan` is in scope.
Base.show(io::IO, plan::NUSHTplan{T}) where {T} =
    print(io, "NUSHTplan{", T, "}(lmax=", plan.lmax, ", M=", length(plan.θ_nodes), ", B=", plan.B, ", tol=", plan.tol, ")")
