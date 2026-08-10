"""
    Spin.jl — Spin-weighted (spin-s) non-uniform spherical harmonic transforms.

Self-contained spin-weighted synthesis/analysis at arbitrary scattered points, built from the
Wigner-d "Fourier factorization" + a 2-D nonuniform FFT — independent of any spin convention in
FastTransforms.

A spin-`s` band-limited field is

    f(θ,φ) = Σ_{ℓ,m} ₛf_{ℓm} ₛY_{ℓm}(θ,φ),   ₛY_{ℓm}(θ,φ) = N_ℓ d^ℓ_{m,−s}(θ) e^{imφ},

with `N_ℓ = √((2ℓ+1)/4π)` and `d^ℓ` the Wigner small-d. Using
`d^ℓ_{mn}(θ) = i^{m−n} Σ_{m'} Δ^ℓ_{m'm} Δ^ℓ_{m'n} e^{−im'θ}` (`Δ^ℓ ≡ d^ℓ(π/2)`), the field is a
bivariate Fourier series

    f(θ,φ) = Σ_{m',m} G_{m'm} e^{−im'θ} e^{imφ},   G_{m'm} = i^{m+s} Σ_ℓ ₛf_{ℓm} N_ℓ Δ^ℓ_{m'm} Δ^ℓ_{m',−s},

evaluated at scattered points by one 2-D NUFFT. The `e^{−im'θ}` factor is absorbed by feeding FINUFFT
the **negated** colatitudes `−θ` with `iflag = +1`, so the mode array `G` (CMCL-centered) maps
directly to the NUFFT modes — no per-call axis-reversal. Analysis is the exact Euclidean adjoint
(FINUFFT type 1 at `−θ`, `iflag = −1`, + the transpose Δ-contraction); `nusht_solve_spin!` inverts by
conjugate gradients on the normal equations. Like the scalar plan, the FINUFFT guru plans are built
once (points set once) so the solver's matvecs never re-plan.

Coefficients use a dense `(lmax+1) × (2lmax+1)` layout: `sf[ℓ+1, m+lmax+1]`. Spin `s = ±1` is the
tangent-vector case (`U = u_θ + i u_φ`), enabling vector/Helmholtz operations at scattered points.
"""

using LinearAlgebra: LinearAlgebra
using SpecialFunctions: loggamma

export SpinNUSHTplan, make_spin_plan, SpinCGWorkspace, WignerTable
export nusht_type2_spin!, nusht_type1_spin!, nusht_solve_spin!
export spin_coeff_index, sYlm

"""
    wigner_d(ℓ, m, n, β)

Wigner small-d matrix element `d^ℓ_{mn}(β)` (Wikipedia/Varshalovich convention), via the explicit
alternating sum with log-gamma factorials for stability.
"""
function wigner_d(ℓ::Integer, m::Integer, n::Integer, β::Real)
    (abs(m) > ℓ || abs(n) > ℓ) && return 0.0
    c = cos(β / 2); s = sin(β / 2)
    kmin = max(0, n - m); kmax = min(ℓ + n, ℓ - m)
    pref = 0.5 * (loggamma(ℓ + m + 1) + loggamma(ℓ - m + 1) + loggamma(ℓ + n + 1) + loggamma(ℓ - n + 1))
    tot = 0.0
    for k in kmin:kmax
        den = loggamma(ℓ + n - k + 1) + loggamma(k + 1) + loggamma(ℓ - m - k + 1) + loggamma(k + m - n + 1)
        tot += ((-1)^k) * exp(pref - den) * (c^(2ℓ + n - m - 2k)) * (s^(2k + m - n))
    end
    return tot
end

@inline _Nℓ(ℓ) = sqrt((2ℓ + 1) / (4π))

# i^k for integer k ∈ ℤ, by its period-4 cycle 1, i, −1, −i. `Complex^Integer` goes through
# exp(k·log z) and costs ~56 ns; this appears once per (degree, order) in the spin contraction, and is
# also the form the GPU kernels need (no generic complex power on device).
@inline function _im_pow(::Type{CT}, k::Integer) where {CT}
    r = mod(k, 4)
    return r == 0 ? CT(1, 0) : r == 1 ? CT(0, 1) : r == 2 ? CT(-1, 0) : CT(0, -1)
end

"""
    sYlm(s, ℓ, m, θ, φ) -> Complex

Spin-weighted spherical harmonic `ₛY_{ℓm}(θ,φ) = N_ℓ d^ℓ_{m,−s}(θ) e^{imφ}` (this package's
convention). Provided for validation / direct evaluation.
"""
sYlm(s::Integer, ℓ::Integer, m::Integer, θ::Real, φ::Real) = _Nℓ(ℓ) * wigner_d(ℓ, m, -s, θ) * cis(m * φ)

"""
    spin_coeff_index(ℓ, m, lmax) -> CartesianIndex

Index into the dense `(lmax+1, 2lmax+1)` spin coefficient array for degree `ℓ`, order `m`.
"""
@inline spin_coeff_index(ℓ::Integer, m::Integer, lmax::Integer) = CartesianIndex(ℓ + 1, m + lmax + 1)

"""
    WignerTable(lmax, s; T = Float64)

Precomputed table of `Q^ℓ_{m'm} = Δ^ℓ_{m'm} · Δ^ℓ_{m',−s}` for every degree `ℓ ≤ lmax` — the only
combination of the Wigner-d(π/2) planes that the spin contraction reads. It depends solely on
`(lmax, s, T)`, so one table is **read-only and shareable**: hand the same one to any number of plans,
on any number of threads.

Without a table the whole `O(lmax³)` Trapani–Navaza sweep is regenerated on every transform — twice
per CG iteration in [`nusht_solve_spin!`](@ref), where the planes are identical every time. With one,
that sweep is paid once and the contraction additionally runs `ℓ`-innermost, keeping each output
column of `G` hot across the degree sum instead of re-streaming `G` once per degree.

The cost is `O(lmax³)` memory — roughly 2.8 MiB at `lmax = 64`, 22 MiB at 128, 175 MiB at 256 and
1.4 GiB at 512 in `Float64`. These are the same two modes s2fft exposes as "precompute" and "on the
fly"; choose per problem rather than globally.

    tbl = WignerTable(64, 1)
    plans = [make_spin_plan(θs[i], φs[i], 64, 1; wigner_table = tbl) for i in eachindex(θs)]
"""
struct WignerTable{T, V<:AbstractVector{T}, VI<:AbstractVector{Int}}
    lmax::Int
    s::Int
    Q::V            # plane ℓ is (2ℓ+1)² column-major at offsets[ℓ+1]; (mp,m) → +(m+ℓ)(2ℓ+1)+(mp+ℓ)
    offsets::VI
end

function WignerTable(lmax::Integer, s::Integer; T::Type{<:AbstractFloat} = Float64)
    L = 2lmax + 1; off = lmax + 1
    offsets = Vector{Int}(undef, lmax + 2)
    offsets[1] = 1
    for ℓ in 0:lmax
        offsets[ℓ + 2] = offsets[ℓ + 1] + (2ℓ + 1)^2
    end
    Q = zeros(T, offsets[lmax + 2] - 1)
    dl = zeros(T, L, L); dlp = zeros(T, L, L)
    _wigner_d_halfpi_step!(dl, dlp, 0, off)
    for ℓ in 0:lmax
        if ℓ > 0
            dl, dlp = dlp, dl
            _wigner_d_halfpi_step!(dl, dlp, ℓ, off)
        end
        ℓ < abs(s) && continue
        nl = 2ℓ + 1; base = offsets[ℓ + 1]
        @inbounds for m in -ℓ:ℓ
            col = base + (m + ℓ) * nl + ℓ
            @simd for mp in -ℓ:ℓ
                Q[col + mp] = dl[mp + off, m + off] * dl[mp + off, (-s) + off]
            end
        end
    end
    return WignerTable{T, typeof(Q), typeof(offsets)}(Int(lmax), Int(s), Q, offsets)
end

Base.show(io::IO, t::WignerTable{T}) where {T} = print(io, "WignerTable{", T, "}(lmax=", t.lmax,
    ", s=", t.s, ", ", round(sizeof(t.Q) / 2^20; digits = 2), " MiB)")

@inline function _check_table(t::WignerTable, plan)
    (t.lmax == plan.lmax && t.s == plan.s) || throw(ArgumentError(
        "WignerTable is for (lmax=$(t.lmax), s=$(t.s)) but this plan is (lmax=$(plan.lmax), s=$(plan.s))"))
    return t
end

"""
    SpinNUSHTplan{T}

Plan for spin-`s` non-uniform spherical harmonic transforms at `M` scattered points up to degree
`lmax`, transforming `B` co-located fields per call. Owns persistent FINUFFT guru plans (type 2
`iflag=+1`, type 1 `iflag=−1`) whose points are set to `−θ` once. The Wigner `Δ^ℓ = d^ℓ(π/2)` planes
are generated **on the fly** by the Trapani–Navaza recurrence into two reused `(2lmax+1)²` buffers
(`dl_curr`/`dl_prev`), so the plan is **O(lmax²) memory**, and the recurrence is numerically stable to
`ℓ ≈ 1024` (the explicit-factorial `wigner_d` sum it replaced loses all accuracy above `ℓ ≈ 40`).

Pass a [`WignerTable`](@ref) as `wigner_table` to precompute those planes instead: faster per
transform, at O(lmax³) memory.
"""
struct SpinNUSHTplan{T<:AbstractFloat, MT<:AbstractMatrix{T}, CT3<:AbstractArray{Complex{T},3},
                     ND<:AbstractNodeSet, W<:Union{Nothing,WignerTable}, FT} <: AbstractNUSHTplan
    lmax::Int
    s::Int
    B::Int
    tol::FT
    nodes::ND              # points, (M,B) strengths and the FINUFFT guru plans; see AbstractNodeSet
    dl_curr::MT            # (2lmax+1)² reused Wigner-d(π/2) plane for the current degree ℓ
    dl_prev::MT            # (2lmax+1)² reused plane for degree ℓ-1 (Trapani–Navaza recurrence)
    G::CT3                 # (L, L, B) bivariate-Fourier mode buffer, L = 2lmax+1
    wigner::W              # precomputed Δ-product table, or `nothing` for the on-the-fly recurrence
end

"""
    make_spin_plan([FE = ComplexF64,] θ_nodes, φ_nodes, lmax, s; tol=1e-10, ntrans=1, …)

Build a `SpinNUSHTplan`. Colatitudes `θ ∈ [0,π]`, longitudes `φ ∈ [0,2π)`.

`FE` is the field element type, positional as in [`make_plan`](@ref); a spin field is complex in
general, hence the default. A real `FE` asserts the field VALUES are real, which makes the mode array
conjugate-symmetric and halves both the Δ-contraction and the NUFFT's θ axis — correct only if the
coefficients satisfy the reality condition.

`tuning` ([`AbstractPlanTuning`](@ref)) and the `nthreads` / `upsampfac` overrides behave as in
[`make_plan`](@ref); there are no FastTransforms plans here, so only the FINUFFT settings are
searched.
"""
# Same convention as `make_plan`, except the field is complex when omitted: a spin field is complex in
# general, and the folded real layout asserts something about the data that only the caller can state.
make_spin_plan(θ_nodes, φ_nodes, lmax::Integer, s::Integer; kwargs...) =
    make_spin_plan(Complex{float(eltype(θ_nodes))}, θ_nodes, φ_nodes, lmax, s; kwargs...)

function make_spin_plan(::Type{FE}, θ_nodes, φ_nodes, lmax::Integer, s::Integer;
                        tol = 1e-10, ntrans::Integer = 1,
                        tuning::AbstractPlanTuning = NoTuning(),
                        nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
                        variable_npts::Bool = false,
                        wigner_table::Union{Nothing,WignerTable} = nothing,
                        nthreads::Union{Nothing,Integer} = nothing,
                        upsampfac::Union{Nothing,Real} = nothing) where {FE<:Number}
    @assert length(θ_nodes) == length(φ_nodes)
    B = Int(ntrans)
    @assert B ≥ 1
    T = real(FE)
    T <: AbstractFloat ||
        throw(ArgumentError("field element type must be a float or complex float, got $FE"))
    L = 2lmax + 1
    M = length(θ_nodes)

    # A real spin field has `G[-m', -m] = conj(G[m', m])`, so only `m' ≥ 0` is built or transformed;
    # `L` odd means no self-paired Nyquist row. The `m' = 0` row pairs with itself across `m`.
    realfield = FE <: Real
    Lθ = realfield ? lmax + 1 : L

    # Node vectors keep their input array type (host `Vector` or device array), with eltype `T`.
    θ = T.(θ_nodes)
    φ = T.(φ_nodes)
    negθ = -θ                                  # absorbs the e^{-im'θ} factor into iflag=+1

    # Buffers are shaped like the nodes, so device nodes ⇒ device-resident plan. Reused Wigner-d(π/2)
    # recurrence buffers are O(lmax²), not the O(lmax³) of a dense per-ℓ store.
    dl_curr = _zeros_like(θ, T, L, L)
    dl_prev = _zeros_like(θ, T, L, L)
    G    = _zeros_like(θ, Complex{T}, Lθ, L, B)
    fbuf = _zeros_like(θ, Complex{T}, M, B)
    # Centered order labels a length-`Lθ` axis `-(Lθ÷2) … `, so a `m' ≥ 0` axis is that ordering shifted
    # by a constant, undone per point. `θ_nufft` is `-θ`, which is the coordinate this pairs with.
    θ_shift = realfield ? _to_like(θ, Complex{T}.(cis.(T(Lθ ÷ 2) .* negθ))) : nothing

    # NUFFT built through the backend seam: host nodes → FINUFFT, device nodes → cuFINUFFT (CUDA ext).
    # `modeord = 0` (CMCL-centered) here, unlike the scalar plan — the G mode array is already centered.
    tol64 = Float64(tol)
    n_modes = Int64[Lθ, L]
    nub = _resolve_nufft(nufft)
    nt2, uf2 = _tune_nufft(nub, negθ, φ, n_modes, 2, +1, B, T, tol64, 0, tuning)
    nt1, uf1 = _tune_nufft(nub, negθ, φ, n_modes, 1, -1, B, T, tol64, 0, tuning)
    isnothing(nthreads) || (nt2 = nt1 = Int(nthreads))
    isnothing(upsampfac) || (uf2 = uf1 = Float64(upsampfac))
    nufft_type2 = _make_nufft(nub, negθ, 2, n_modes, +1, B, tol64, T, 0, nt2, uf2)
    nufft_type1 = _make_nufft(nub, negθ, 1, n_modes, -1, B, tol64, T, 0, nt1, uf1)
    _nufft_setpts!(nufft_type2, negθ, φ)
    _nufft_setpts!(nufft_type1, negθ, φ)
    _nufft_finalize!(nufft_type2)
    _nufft_finalize!(nufft_type1)
    # `negθ` is this plan's `θ_nufft`: a separate array, unlike the scalar plan's alias, because the
    # e^{-im'θ} factor is absorbed by handing FINUFFT the negated colatitudes.
    nodes = _node_set(Val(variable_npts), θ, φ, negθ, θ_shift, fbuf, nufft_type2, nufft_type1)

    isnothing(wigner_table) || _check_table(wigner_table, (lmax = Int(lmax), s = Int(s)))
    return SpinNUSHTplan{T, typeof(dl_curr), typeof(G), typeof(nodes), typeof(wigner_table),
                         typeof(tol64)}(
        lmax, Int(s), B, tol64, nodes, dl_curr, dl_prev, G, wigner_table)
end

# Safe one-line `show` (see the NUSHTplan note): avoid recursing into stored FFTW/FINUFFT plan
# pointers, whose printers can segfault on an invalidated C state.
@inline _shift_offset(plan::SpinNUSHTplan{T}) where {T} = T(size(plan.G, 1) ÷ 2)
_copy_out!(f, fbuf, plan::SpinNUSHTplan) = _copy_out!(f, fbuf, _θshift(plan))
_copy_out!(f, fbuf, ::Nothing) = _copy_field!(f, fbuf)
_copy_out!(f, fbuf, ::AbstractVector) = _copy_real!(f, fbuf)

# Row index for wavenumber `m'` in the mode buffer: the full layout centers it, the folded one keeps
# only `m' ≥ 0`. `_mprange` is the matching set of `m'` the contraction has to visit.
# The layout is carried by whether the plan has a `θ_shift` — which only the folded one needs — so it
# lives in the node-set type and needs no separate parameter on the plan.
@inline _grow(plan::SpinNUSHTplan, mp) = _grow(plan, mp, _θshift(plan))
@inline _grow(plan::SpinNUSHTplan, mp, ::Nothing) = mp + plan.lmax + 1
@inline _grow(::SpinNUSHTplan, mp, ::AbstractVector) = mp + 1
@inline _mprange(plan::SpinNUSHTplan, ℓ) = _mprange(ℓ, _θshift(plan))
@inline _mprange(ℓ, ::Nothing) = -ℓ:ℓ
@inline _mprange(ℓ, ::AbstractVector) = 0:ℓ

# Each retained mode stands for itself and its partner, so it counts twice — except `(0,0)`, its own
# partner. The `m' = 0` row's `m < 0` half is the dropped one.
_fold_weights!(G, plan::SpinNUSHTplan) = _fold_weights!(G, plan, _θshift(plan))
_fold_weights!(G, ::SpinNUSHTplan, ::Nothing) = G
function _fold_weights!(G, plan::SpinNUSHTplan{T}, ::AbstractVector) where {T}
    lmax = plan.lmax
    G .*= T(2)
    # Views, not element writes: a GPU array rejects scalar indexing.
    @views fill!(G[1, 1:lmax, :], zero(eltype(G)))
    @views G[1, lmax + 1, :] ./= T(2)
    return G
end

Base.show(io::IO, plan::SpinNUSHTplan{T}) where {T} =
    print(io, "SpinNUSHTplan{", T, "}(lmax=", plan.lmax, ", s=", plan.s, ", M=", length(_θnodes(plan)), ", B=", plan.B, ")")

"""
    _wigner_d_halfpi_step!(dl, dlp, ℓ, off) -> dl

Overwrite `dl` with the degree-`ℓ` Wigner-d plane at β = π/2, from the degree-`(ℓ-1)` plane `dlp`, via
the **Trapani–Navaza** recurrence (Trapani & Navaza 2006; ssht `ssht_dl.c`; s2fft `trapani.py`). Both
`dl`/`dl_prev` are `(2lmax+1)²`, `off = lmax+1`. O(ℓ²) work, one previous plane — so a full sweep
ℓ = 0…lmax is O(lmax³) work in O(lmax²) memory, numerically stable to ℓ ≈ 1024 (double).

**Layout is `n`-major: `dl[n+off, m+off] = d^ℓ_{m,n}(π/2)`** (McEwen–Wiaux/ssht convention,
transposed). The recurrence runs downward in `m`, so storing `n` first makes every inner loop
contiguous and independent — Stage A, Stage B and the `n → −n` fill all vectorize — and it is also
the order the contraction reads, since `Δ^ℓ_{m',m} = dl[m'+off, m+off]`. Relation to this package's
`wigner_d`: `wigner_d(ℓ, a, b, π/2) == dl[a+off, b+off]`.

Only the eighth `0 ≤ n ≤ m ≤ ℓ` is recurred; the rest is filled by the d(π/2) symmetries (transpose,
m→−m, n→−n).
"""
function _wigner_d_halfpi_step!(dl::AbstractMatrix{T}, dlp::AbstractMatrix{T}, ℓ::Integer, off::Integer) where {T}
    if ℓ == 0
        dl[off, off] = one(T)
        return dl
    end
    @inbounds begin
        # Stage A — boundary m = ℓ (T&N Eqns 9 & 10), read from the previous plane's m = ℓ-1 column.
        dl[off, ℓ + off] = -sqrt(T(2ℓ - 1) / T(2ℓ)) * dlp[off, (ℓ - 1) + off]
        @simd for n in 1:ℓ
            dl[n + off, ℓ + off] =
                sqrt(T(ℓ) * T(2ℓ - 1) / (T(2) * T(ℓ + n) * T(ℓ + n - 1))) * dlp[(n - 1) + off, (ℓ - 1) + off]
        end
        # Stage B — three-term downward recurrence in m (T&N Eqn 11), eighth 0 ≤ n ≤ m ≤ ℓ. The two
        # square roots depend only on (ℓ, m), so hoisting them out of the n loop makes the sqrt count
        # O(ℓ) per degree instead of O(ℓ²), and leaves the n loop a dependency-free axpy.
        for m in (ℓ - 1):-1:0
            c1 = T(2) / sqrt(T(ℓ - m) * T(ℓ + m + 1))
            if m == ℓ - 1                                  # second term vanishes
                @simd for n in 0:m
                    dl[n + off, m + off] = c1 * T(n) * dl[n + off, (m + 1) + off]
                end
            else
                s2 = sqrt(T(ℓ - m - 1) * T(ℓ + m + 2) / (T(ℓ - m) * T(ℓ + m + 1)))
                @simd for n in 0:m
                    dl[n + off, m + off] =
                        c1 * T(n) * dl[n + off, (m + 1) + off] - s2 * dl[n + off, (m + 2) + off]
                end
            end
        end
        # Symmetry fill to the full (2ℓ+1)² plane.
        for m in 0:ℓ, n in (m + 1):ℓ                       # S1 transpose: eighth → quarter
            dl[n + off, m + off] = ifelse(iseven(m + n), one(T), -one(T)) * dl[m + off, n + off]
        end
        for m in -ℓ:-1                                     # S2 m → −m: quarter → half
            @simd for n in 0:ℓ
                dl[n + off, m + off] = ifelse(iseven(ℓ + n), one(T), -one(T)) * dl[n + off, (-m) + off]
            end
        end
        for m in -ℓ:ℓ                                      # S3 n → −n: half → full
            sgn = ifelse(iseven(ℓ + abs(m)), one(T), -one(T))
            @simd for n in -ℓ:-1
                dl[n + off, m + off] = sgn * dl[(-n) + off, m + off]
            end
        end
    end
    return dl
end

@inline _spin_npts(plan::SpinNUSHTplan) = length(_θnodes(plan))

@inline function _assert_spin_coeffs(sf, plan::SpinNUSHTplan)
    @assert length(sf) == (plan.lmax + 1) * (2plan.lmax + 1) * plan.B "spin coefficient array has $(length(sf)) entries, expected (lmax+1)·(2lmax+1)·B"
end
@inline function _assert_spin_field(f, plan::SpinNUSHTplan)
    @assert length(f) == _spin_npts(plan) * plan.B "spin field array has $(length(f)) entries, expected M·B"
end

# Assemble bivariate Fourier coefficients G_{m'm} (CMCL-centered) from coefficients `sf`, in place.
# `sf` is indexed `[ℓ+1, m+lmax+1, b]`, which works for a 2-D `(lmax+1,2lmax+1)` array (B=1, trailing
# singleton) or the batched 3-D array — no reshape, so the hot path stays zero-alloc.
_assemble_G!(G, sf, plan::SpinNUSHTplan) =
    _fold_weights!(_assemble_G_impl!(G, sf, plan, plan.wigner), plan)

# Precomputed path: Δ is already contracted into Q, so ℓ can run innermost and each output column of
# G stays hot across the whole degree sum.
function _assemble_G_impl!(G, sf, plan::SpinNUSHTplan{T}, tbl::WignerTable) where {T}
    _check_table(tbl, plan)
    lmax = plan.lmax; s = plan.s; off = lmax + 1
    Q = tbl.Q; qoff = tbl.offsets
    fill!(G, zero(Complex{T}))
    @inbounds for m in -lmax:lmax
        ph0 = _im_pow(Complex{T}, m + s)
        for b in 1:plan.B
            for ℓ in max(abs(m), abs(s)):lmax
                val = sf[ℓ + 1, m + lmax + 1, b]
                iszero(val) && continue
                c = ph0 * val * T(_Nℓ(ℓ))
                nl = 2ℓ + 1; col = qoff[ℓ + 1] + (m + ℓ) * nl + ℓ
                @simd for mp in _mprange(plan, ℓ)
                    G[_grow(plan, mp), m + off, b] += c * Q[col + mp]
                end
            end
        end
    end
    return G
end

# On-the-fly path: regenerate each Δ^ℓ plane by the Trapani–Navaza recurrence into the reused
# `dl_curr`/`dl_prev` buffers, ℓ ascending, ping-ponged — O(lmax²) memory instead of O(lmax³).
function _assemble_G_impl!(G, sf, plan::SpinNUSHTplan{T}, ::Nothing) where {T}
    lmax = plan.lmax; s = plan.s; off = lmax + 1
    dl = plan.dl_curr; dlp = plan.dl_prev
    fill!(G, zero(Complex{T}))
    _wigner_d_halfpi_step!(dl, dlp, 0, off)                 # degree 0
    @inbounds for ℓ in 0:lmax
        if ℓ > 0
            dl, dlp = dlp, dl
            _wigner_d_halfpi_step!(dl, dlp, ℓ, off)
        end
        ℓ < abs(s) && continue
        Nℓ = T(_Nℓ(ℓ))
        # `m` outermost so the i^{m+s}·N_ℓ phase is formed once per order rather than once per
        # (batch, order); `mp` innermost so both Δ reads and the G write run down contiguous columns.
        for m in -ℓ:ℓ
            ph0 = _im_pow(Complex{T}, m + s) * Nℓ
            for b in 1:plan.B
                val = sf[ℓ + 1, m + lmax + 1, b]
                iszero(val) && continue
                ph = ph0 * val
                @simd for mp in _mprange(plan, ℓ)
                    G[_grow(plan, mp), m + off, b] +=
                        ph * (dl[mp + off, m + off] * dl[mp + off, (-s) + off])
                end
            end
        end
    end
    return G
end

# Adjoint of _assemble_G!: contract NUFFT-type1 modes Ĝ back to coefficients, in place.
# The fold weights are a real diagonal, hence self-adjoint, so the adjoint applies the same ones —
# before the contraction here, after it in `_assemble_G!`.
_assemble_G_adjoint!(sf, Ĝ, plan::SpinNUSHTplan) =
    _assemble_G_adjoint_impl!(sf, _fold_weights!(Ĝ, plan), plan, plan.wigner)

function _assemble_G_adjoint_impl!(sf, Ĝ, plan::SpinNUSHTplan{T}, tbl::WignerTable) where {T}
    _check_table(tbl, plan)
    lmax = plan.lmax; s = plan.s; off = lmax + 1
    Q = tbl.Q; qoff = tbl.offsets
    fill!(sf, zero(Complex{T}))
    @inbounds for m in -lmax:lmax
        phc = conj(_im_pow(Complex{T}, m + s))
        for b in 1:plan.B
            for ℓ in max(abs(m), abs(s)):lmax
                nl = 2ℓ + 1; col = qoff[ℓ + 1] + (m + ℓ) * nl + ℓ
                acc = zero(Complex{T})
                @simd for mp in _mprange(plan, ℓ)
                    acc += Q[col + mp] * Ĝ[_grow(plan, mp), m + off, b]
                end
                sf[ℓ + 1, m + lmax + 1, b] = phc * T(_Nℓ(ℓ)) * acc
            end
        end
    end
    return sf
end

function _assemble_G_adjoint_impl!(sf, Ĝ, plan::SpinNUSHTplan{T}, ::Nothing) where {T}
    lmax = plan.lmax; s = plan.s; off = lmax + 1
    dl = plan.dl_curr; dlp = plan.dl_prev
    fill!(sf, zero(Complex{T}))
    _wigner_d_halfpi_step!(dl, dlp, 0, off)
    @inbounds for ℓ in 0:lmax
        if ℓ > 0
            dl, dlp = dlp, dl
            _wigner_d_halfpi_step!(dl, dlp, ℓ, off)
        end
        ℓ < abs(s) && continue
        Nℓ = T(_Nℓ(ℓ))
        for m in -ℓ:ℓ
            phc = conj(_im_pow(Complex{T}, m + s)) * Nℓ
            for b in 1:plan.B
                acc = zero(Complex{T})
                @simd for mp in _mprange(plan, ℓ)
                    acc += dl[mp + off, m + off] * dl[mp + off, (-s) + off] *
                           Ĝ[_grow(plan, mp), m + off, b]
                end
                sf[ℓ + 1, m + lmax + 1, b] = phc * acc
            end
        end
    end
    return sf
end

"""
    nusht_type2_spin!(f, sf, plan) -> f

Spin-weighted synthesis: evaluate the spin-`s` field with coefficients `sf` at the `M` scattered
points, writing complex values into `f`. Batched: `sf` is `(lmax+1, 2lmax+1[, B])`, `f` is
length-`M` / `(M, B)`.
"""
function nusht_type2_spin!(f, sf, plan::SpinNUSHTplan{T}) where {T}
    _assert_spin_coeffs(sf, plan)
    _assert_spin_field(f, plan)
    _assemble_G!(plan.G, sf, plan)
    _nufft_exec!(_nufft2(plan), plan.G, _fbuf(plan))
    _rephase!(_fbuf(plan), _θshift(plan))
    _copy_out!(f, _fbuf(plan), plan)
    return f
end

"""
    nusht_type1_spin!(sf, f, plan) -> sf

Exact Euclidean adjoint of `nusht_type2_spin!`: scattered values `f` → spin-`s` coefficients `sf`.
"""
function nusht_type1_spin!(sf, f, plan::SpinNUSHTplan{T}) where {T}
    _assert_spin_field(f, plan)
    _assert_spin_coeffs(sf, plan)
    copyto!(_fbuf(plan), f)
    _rephase_conj!(_fbuf(plan), _θshift(plan))
    _nufft_exec!(_nufft1(plan), _fbuf(plan), plan.G)
    _assemble_G_adjoint!(sf, plan.G, plan)
    return sf
end

# ─────────────────────────────────────────────────────────────────────────────
# Exact inversion: batched complex CG on the normal equations A†A sf = A† f
# ─────────────────────────────────────────────────────────────────────────────

"""
    SpinCGWorkspace(plan)

Reusable scratch for [`nusht_solve_spin!`](@ref) (allocation-free repeated solves; per-column scalars
so a batched solve equals `B` independent single-column solves).
"""
struct SpinCGWorkspace{CT3<:AbstractArray, CT2<:AbstractMatrix, VT<:AbstractVector}
    x::CT3
    r::CT3
    p::CT3
    Ap::CT3
    rhs::CT3
    f::CT2
    rsold::VT
    rsnew::VT
    pAp::VT
    α::VT
    β::VT
    rel::VT
    rhsnorm::VT
end

function SpinCGWorkspace(plan::SpinNUSHTplan{T}) where {T}
    lmax = plan.lmax; B = plan.B; M = _spin_npts(plan)
    # Buffers shaped like the plan's (so a device plan gets a device workspace).
    z3() = _zeros_like(plan.G, Complex{T}, lmax + 1, 2lmax + 1, B)
    vB() = _zeros_like(_θnodes(plan), T, B)
    return SpinCGWorkspace(z3(), z3(), z3(), z3(), z3(), _zeros_like(_fbuf(plan), Complex{T}, M, B),
                           vB(), vB(), vB(), vB(), vB(), vB(), vB())
end

# Per-column Hermitian reductions / updates over a (…, B) complex array — zero-allocation.
function _col_hdot!(dst, a, b)
    @inbounds for k in axes(a, 3)
        s = zero(real(eltype(a)))
        for j in axes(a, 2), i in axes(a, 1)
            s += real(conj(a[i, j, k]) * b[i, j, k])
        end
        dst[k] = s
    end
    return dst
end

function _col_axpy_c!(y, α, x, σ)   # y[:,:,k] += σ·α[k]·x[:,:,k], α real
    @inbounds for k in axes(y, 3)
        c = σ * α[k]
        for j in axes(y, 2), i in axes(y, 1)
            y[i, j, k] += c * x[i, j, k]
        end
    end
    return y
end

function _col_pbp_c!(p, r, β)       # p[:,:,k] = r[:,:,k] + β[k]·p[:,:,k], β real
    @inbounds for k in axes(p, 3)
        c = β[k]
        for j in axes(p, 2), i in axes(p, 1)
            p[i, j, k] = r[i, j, k] + c * p[i, j, k]
        end
    end
    return p
end

function _AtA_spin!(Ap, p, ws::SpinCGWorkspace, plan::SpinNUSHTplan)
    nusht_type2_spin!(ws.f, p, plan)
    nusht_type1_spin!(Ap, ws.f, plan)
    return Ap
end

"""
    nusht_solve_spin!(sf, f, plan; ws=SpinCGWorkspace(plan), maxiter=500, rtol=1e-8, verbose=false)

Exact inversion of the spin-weighted synthesis at arbitrary scattered points: solve `A sf = f` by
conjugate gradients on the normal equations `(A†A) sf = A† f`. Batched (`B > 1`) runs the columns as
independent single-column CGs. Returns `(sf, iters, rel_res)` with `rel_res = max_k ‖r_k‖/‖A†f_k‖`.
"""
function nusht_solve_spin!(
    sf, f, plan::SpinNUSHTplan{T};
    ws::SpinCGWorkspace = SpinCGWorkspace(plan),
    maxiter::Int = 500,
    rtol::Real = 1e-8,
    verbose::Bool = false,
) where {T}
    nusht_type1_spin!(ws.rhs, f, plan)
    _col_hdot!(ws.rhsnorm, ws.rhs, ws.rhs)
    ws.rhsnorm .= sqrt.(ws.rhsnorm)

    fill!(ws.x, zero(Complex{T}))
    copyto!(ws.r, ws.rhs)
    copyto!(ws.p, ws.r)
    _col_hdot!(ws.rsold, ws.r, ws.r)
    fill!(ws.rel, one(T))

    iters = 0
    for i in 1:maxiter
        iters = i
        _AtA_spin!(ws.Ap, ws.p, ws, plan)
        _col_hdot!(ws.pAp, ws.p, ws.Ap)
        @. ws.α = ifelse(ws.pAp == 0, zero(T), ws.rsold / ws.pAp)
        _col_axpy_c!(ws.x, ws.α, ws.p, one(T))
        _col_axpy_c!(ws.r, ws.α, ws.Ap, -one(T))
        _col_hdot!(ws.rsnew, ws.r, ws.r)
        @. ws.rel = ifelse(ws.rhsnorm == 0, zero(T), sqrt(ws.rsnew) / ws.rhsnorm)
        verbose && @info "nusht_solve_spin! iter $i rel_res=$(maximum(ws.rel))"
        maximum(ws.rel) < rtol && break
        @. ws.β = ifelse(ws.rsold == 0, zero(T), ws.rsnew / ws.rsold)
        _col_pbp_c!(ws.p, ws.r, ws.β)
        copyto!(ws.rsold, ws.rsnew)
    end

    copyto!(sf, ws.x)
    return sf, iters, maximum(ws.rel)
end
