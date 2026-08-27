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
LSMR on the bidiagonalization of `A`. Like the scalar plan, the FINUFFT guru plans are built
once (points set once) so the solver's matvecs never re-plan.

Coefficients use a dense `(lmax+1) × (2lmax+1)` layout: `sf[ℓ+1, m+lmax+1]`. Spin `s = ±1` is the
tangent-vector case (`U = u_θ + i u_φ`), enabling vector/Helmholtz operations at scattered points.
"""

using LinearAlgebra: LinearAlgebra
using SpecialFunctions: loggamma

export SpinNUSHTplan, make_spin_plan, WignerTable
export nusht_type2_spin!, nusht_type1_spin!, nusht_solve_spin!
export spin_coeff_index, sYlm

"""
    wigner_d(ℓ, m, n, β)

Wigner small-d matrix element `d^ℓ_{mn}(β) = ⟨ℓm|exp(-iβ Jy)|ℓn⟩` in the Condon–Shortley basis, via
the explicit alternating sum with log-gamma factorials for stability.

Accuracy degrades above `ℓ ≈ 40`; use [`_wigner_d_halfpi_step!`](@ref) for `β = π/2` at large `ℓ`.
"""
function wigner_d(ℓ::Integer, m::Integer, n::Integer, β::Real)
    (abs(m) > ℓ || abs(n) > ℓ) && return 0.0
    c = cos(β / 2); s = sin(β / 2)
    kmin = max(0, n - m); kmax = min(ℓ + n, ℓ - m)
    pref = 0.5 * (loggamma(ℓ + m + 1) + loggamma(ℓ - m + 1) + loggamma(ℓ + n + 1) + loggamma(ℓ - n + 1))
    tot = 0.0
    for k in kmin:kmax
        den = loggamma(ℓ + n - k + 1) + loggamma(k + 1) + loggamma(ℓ - m - k + 1) + loggamma(k + m - n + 1)
        tot += ((-1)^(k + m - n)) * exp(pref - den) * (c^(2ℓ + n - m - 2k)) * (s^(2k + m - n))
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

Spin-weighted spherical harmonic `ₛY_{ℓm}(θ,φ) = N_ℓ d^ℓ_{m,−s}(θ) e^{imφ}` in the
Goldberg / Newman–Penrose convention, with `d` the rotation matrix of [`wigner_d`](@ref). Provided for
direct evaluation.
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
per solver iteration in [`nusht_solve_spin!`](@ref), where the planes are identical every time. With one,
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

# Same convention as `make_plan`, except the field is complex when omitted: a spin field is complex in
# general, and the folded real layout asserts something about the data that only the caller can state.
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
make_spin_plan(θ_nodes, φ_nodes, lmax::Integer, s::Integer; kwargs...) =
    make_spin_plan(Complex{float(eltype(θ_nodes))}, θ_nodes, φ_nodes, lmax, s; kwargs...)

function make_spin_plan(::Type{FE}, θ_nodes, φ_nodes, lmax::Integer, s::Integer;
                        tol = 1e-10, ntrans::Integer = 1,
                        tuning::AbstractPlanTuning = NoTuning(),
                        nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
                        variable_npts::Bool = false,
                        directions::AbstractPlanDirections = SynthesisAndAnalysis(),
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
    # `L` odd means no self-paired Nyquist row.
    #
    # Who supplies the absent `m' < 0` half decides everything downstream. A backend with a real-data
    # transform does it itself, from a transform built for the full `L` θ wavenumbers, and returns real
    # strengths — halving the spreading as well as the FFT. Without one the transform is itself
    # half-height and complex: `θ_shift` undoes its centered row labelling per point, and the missing
    # half is paid for in the weights.
    realfield = FE <: Real
    nub = _resolve_nufft(nufft, FE)
    r2c = realfield && _real_capable(nub)
    Lθ = realfield ? lmax + 1 : L
    ZS = r2c ? T : Complex{T}                  # element type of the non-uniform data

    # Node vectors keep their input array type (host `Vector` or device array), with eltype `T`.
    θ = T.(θ_nodes)
    φ = T.(φ_nodes)
    negθ = -θ                                  # absorbs the e^{-im'θ} factor into iflag=+1

    # Buffers are shaped like the nodes, so device nodes ⇒ device-resident plan. Reused Wigner-d(π/2)
    # recurrence buffers are O(lmax²), not the O(lmax³) of a dense per-ℓ store.
    dl_curr = _zeros_like(θ, T, L, L)
    dl_prev = _zeros_like(θ, T, L, L)
    G    = _zeros_like(θ, Complex{T}, Lθ, L, B)
    fbuf = _zeros_like(θ, ZS, M, B)
    # Centered order labels a length-`Lθ` axis `-(Lθ÷2) … `, so a `m' ≥ 0` axis is that ordering shifted
    # by a constant, undone per point. `θ_nufft` is `-θ`, which is the coordinate this pairs with. A
    # real-data transform's half-spectrum is labelled `0 … n₁÷2` already and needs no shift.
    θ_shift = (realfield && !r2c) ?
        _to_like(θ, Complex{T}.(cis.(T(Lθ ÷ 2) .* negθ))) : nothing

    # NUFFT built through the backend seam: host nodes → FINUFFT, device nodes → cuFINUFFT (CUDA ext).
    # `modeord = 0` (CMCL-centered) here, unlike the scalar plan — the G mode array is already centered.
    tol64 = Float64(tol)
    # A real transform covers the full θ axis and stores its half; a complex half-height one is built
    # at the stored size.
    n_modes = Int64[r2c ? L : Lθ, L]
    _warn_if_directsum(nufft, nub, M, Lθ * L)
    nt2, uf2 = _tune_nufft(nub, negθ, φ, n_modes, 2, +1, B, T, tol64, 0, tuning, ZS)
    nt1, uf1 = _tune_nufft(nub, negθ, φ, n_modes, 1, -1, B, T, tol64, 0, tuning, ZS)
    isnothing(nthreads) || (nt2 = nt1 = Int(nthreads))
    isnothing(upsampfac) || (uf2 = uf1 = Float64(upsampfac))
    # See `_nufft_share_directions`: where the backend's plan carries no direction, one object serves
    # both and the second is a handle onto it, already pointed at the same nodes.
    nufft_type2 = _make_nufft(nub, negθ, 2, n_modes, +1, B, tol64, T, 0, nt2, uf2, ZS)
    _nufft_setpts!(nufft_type2, negθ, φ)
    _nufft_finalize!(nufft_type2)
    nufft_type1 = _build_analysis(directions, nub, negθ, φ, n_modes, B, tol64, T, 0, nt1, uf1,
                                  ZS, nufft_type2)
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

"""
    FullModes, FoldedComplex, FoldedReal

Which Hermitian treatment a spin plan's mode array needs. A complex field has no symmetry to exploit
([`FullModes`](@ref)); a real one stores only `m' ≥ 0` either way, and the two folded kinds differ in
**who supplies the absent `m' < 0` half** — which is what fixes the row indexing and the weights.

The kind is read off the plan rather than stored: only a half-height *complex* transform needs a
`θ_shift`, and only a real-data transform has real strengths, so the pair `(θ_shift, fbuf)` names all
three without a further plan parameter.
"""
struct FullModes end
struct FoldedComplex end
struct FoldedReal end

@inline _fold_kind(plan::SpinNUSHTplan) = _fold_kind(_θshift(plan), _fbuf(plan))
@inline _fold_kind(::AbstractVector, _fbuf) = FoldedComplex()
@inline _fold_kind(::Nothing, ::AbstractArray{<:Real}) = FoldedReal()
@inline _fold_kind(::Nothing, ::AbstractArray) = FullModes()

# Row index for wavenumber `m'` in the mode buffer, and the set of `m'` the contraction visits: the
# full layout centers `m'`, both folded ones keep only `m' ≥ 0` starting at row 1.
@inline _grow(plan::SpinNUSHTplan, mp) = _grow(plan, mp, _fold_kind(plan))
@inline _grow(plan::SpinNUSHTplan, mp, ::FullModes) = mp + plan.lmax + 1
@inline _grow(::SpinNUSHTplan, mp, ::Union{FoldedComplex,FoldedReal}) = mp + 1
@inline _mprange(plan::SpinNUSHTplan, ℓ) = _mprange(ℓ, _fold_kind(plan))
@inline _mprange(ℓ, ::FullModes) = -ℓ:ℓ
@inline _mprange(ℓ, ::Union{FoldedComplex,FoldedReal}) = 0:ℓ

# A half-height complex transform returns only what it was given, so each retained mode has to stand
# for its partner too: everything counts twice except `(0,0)`, which is its own, and the `m' = 0` row's
# `m < 0` half, which is the one dropped. A real-data transform reconstructs the partner itself, so the
# forward hands it the half unweighted.
_fold_weights!(G, plan::SpinNUSHTplan) = _fold_weights!(G, plan, _fold_kind(plan))
_fold_weights!(G, ::SpinNUSHTplan, ::Union{FullModes,FoldedReal}) = G
function _fold_weights!(G, plan::SpinNUSHTplan{T}, ::FoldedComplex) where {T}
    lmax = plan.lmax
    G .*= T(2)
    # Views, not element writes: a GPU array rejects scalar indexing.
    @views fill!(G[1, 1:lmax, :], zero(eltype(G)))
    @views G[1, lmax + 1, :] ./= T(2)
    return G
end

# The transpose of the forward. `FoldedComplex` is self-adjoint in its weights, so it repeats them.
# `FoldedReal` has no forward weight, but its embedding `G[-m',-m] = conj(G[m',m])` is R-linear and not
# C-linear, so the transpose picks up a factor 2 on every `m' > 0` row; the `m' = 0` row is stored whole
# and is its own conjugate, so it keeps weight 1.
_fold_weights_adjoint!(G, plan::SpinNUSHTplan) = _fold_weights_adjoint!(G, plan, _fold_kind(plan))
_fold_weights_adjoint!(G, ::SpinNUSHTplan, ::FullModes) = G
_fold_weights_adjoint!(G, plan::SpinNUSHTplan, k::FoldedComplex) = _fold_weights!(G, plan, k)
function _fold_weights_adjoint!(G, ::SpinNUSHTplan{T}, ::FoldedReal) where {T}
    @views G[2:end, :, :] .*= T(2)
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
the order the contraction reads, since `Δ^ℓ_{m',m} = dl[m'+off, m+off]`. So the relation to
[`wigner_d`](@ref) is transposed: `wigner_d(ℓ, m, n, π/2) == dl[n+off, m+off]`.

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
        # Symmetry fill to the full (2ℓ+1)² plane. Serial on purpose: one multiply per element makes
        # these streaming rather than compute, and threading them measurably loses. The contraction has
        # arithmetic per element and is the part worth threading.
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
# Work, in fused multiply-adds, an assembly must carry before threading it is worth a `@sync` barrier —
# roughly an order of magnitude more than a barrier costs. Counted in FMAs rather than orders so it
# scales with the batch.
const _CONTRACT_MIN_WORK = 30_000

# With a table there is no recurrence to sequence, so orders run outermost over the whole range and
# split with no per-degree barrier — one `@sync` for the entire assembly, which is why this pays where
# the on-the-fly path (barrier per degree) does not. Each order owns its own column of `G` / entry of
# `sf`, so the ranges need no synchronisation. One thread spawns nothing, keeping the transform
# allocation-free.
@inline function _thread_all_orders(f::F, lmax::Int, B::Int) where {F}
    nord = 2lmax + 1
    nt = min(Threads.nthreads(), max(1, (nord * nord * lmax * B) ÷ _CONTRACT_MIN_WORK))
    nt <= 1 && return f(-lmax:lmax)
    chunk = cld(nord, nt)
    @sync for t in 1:nt
        lo = -lmax + (t - 1) * chunk
        hi = min(lmax, lo + chunk - 1)
        lo > hi && continue
        Threads.@spawn f(lo:hi)
    end
    return nothing
end

@inline function _table_orders!(G, sf, plan::SpinNUSHTplan{T}, tbl::WignerTable, ms,
                                s::Int, off::Int, lmax::Int) where {T}
    Q = tbl.Q; qoff = tbl.offsets
    @inbounds for m in ms
        ph0 = _im_pow(Complex{T}, -(m + s))
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

function _assemble_G_impl!(G, sf, plan::SpinNUSHTplan{T}, tbl::WignerTable) where {T}
    _check_table(tbl, plan)
    lmax = plan.lmax; s = plan.s; off = lmax + 1
    fill!(G, zero(Complex{T}))
    _thread_all_orders(lmax, plan.B) do ms
        _table_orders!(G, sf, plan, tbl, ms, s, off, lmax)
    end
    return G
end

# One degree's contraction over a subrange of orders. Split out so one body serves the serial and the
# threaded path, and so a spawned task captures a concrete range rather than a loop variable.
# `m` outermost so the i^{m+s}·N_ℓ phase is formed once per order rather than once per (batch, order);
# `mp` innermost so both Δ reads and the G write run down contiguous columns.
@inline function _contract_orders!(G, sf, plan::SpinNUSHTplan{T}, dl, ℓ::Int, ms,
                                   Nℓ::T, s::Int, off::Int, lmax::Int) where {T}
    @inbounds for m in ms
        ph0 = _im_pow(Complex{T}, -(m + s)) * Nℓ
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
        # Serial over orders. They are independent, but the on-the-fly path can only split *within* a
        # degree — the recurrence sequences the degrees — and a barrier per degree costs more than the
        # split returns. Measured neutral on the forward and a clear loss on the adjoint. The table
        # path has no such barrier and is threaded.
        _contract_orders!(G, sf, plan, dl, ℓ, -ℓ:ℓ, T(_Nℓ(ℓ)), s, off, lmax)
    end
    return G
end

# Adjoint of _assemble_G!: contract NUFFT-type1 modes Ĝ back to coefficients, in place.
# The fold weights are a real diagonal, hence self-adjoint, so the adjoint applies the same ones —
# before the contraction here, after it in `_assemble_G!`.
_assemble_G_adjoint!(sf, Ĝ, plan::SpinNUSHTplan) =
    _assemble_G_adjoint_impl!(sf, _fold_weights_adjoint!(Ĝ, plan), plan, plan.wigner)

@inline function _table_orders_adj!(sf, Ĝ, plan::SpinNUSHTplan{T}, tbl::WignerTable, ms,
                                    s::Int, off::Int, lmax::Int) where {T}
    Q = tbl.Q; qoff = tbl.offsets
    @inbounds for m in ms
        phc = conj(_im_pow(Complex{T}, -(m + s)))
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

function _assemble_G_adjoint_impl!(sf, Ĝ, plan::SpinNUSHTplan{T}, tbl::WignerTable) where {T}
    _check_table(tbl, plan)
    lmax = plan.lmax; s = plan.s; off = lmax + 1
    fill!(sf, zero(Complex{T}))
    _thread_all_orders(lmax, plan.B) do ms
        _table_orders_adj!(sf, Ĝ, plan, tbl, ms, s, off, lmax)
    end
    return sf
end

# Adjoint counterpart of `_contract_orders!`: each order reduces over `m'` into its own `sf` entry.
@inline function _contract_orders_adj!(sf, Ĝ, plan::SpinNUSHTplan{T}, dl, ℓ::Int, ms,
                                       Nℓ::T, s::Int, off::Int, lmax::Int) where {T}
    @inbounds for m in ms
        phc = conj(_im_pow(Complex{T}, -(m + s))) * Nℓ
        for b in 1:plan.B
            acc = zero(Complex{T})
            @simd for mp in _mprange(plan, ℓ)
                acc += dl[mp + off, m + off] * dl[mp + off, (-s) + off] *
                       Ĝ[_grow(plan, mp), m + off, b]
            end
            sf[ℓ + 1, m + lmax + 1, b] = phc * acc
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
        _contract_orders_adj!(sf, Ĝ, plan, dl, ℓ, -ℓ:ℓ, T(_Nℓ(ℓ)), s, off, lmax)  # serial: see forward
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
# Exact inversion: batched LSMR, sharing the core in NUFSHT.jl
# ─────────────────────────────────────────────────────────────────────────────

"""
    LSMRWorkspace(plan::SpinNUSHTplan)

Reusable scratch for [`nusht_solve_spin!`](@ref) — the same workspace the scalar path uses, with no
`valid` projection: the spin coefficient array is dense, with no supernumerary slots to exclude. The
coefficient vectors are complex because spin coefficients are; `u` is a point-space residual and so
takes the plan's strengths type, which is real wherever the backend has a real-data transform.
"""
function LSMRWorkspace(plan::SpinNUSHTplan{T}) where {T}
    lmax, B, M = plan.lmax, plan.B, _spin_npts(plan)
    CT = Complex{T}
    # A real field's iterate is packed to the `(lmax+1)²` real degrees its coefficients actually have
    # (see `_hermitian_domain`); a complex one carries the full array. `w` holds `A†u`, which lands in
    # the full complex layout either way, so it is the one buffer whose shape does not follow.
    # A complex field's iterate keeps the coefficient array's own `(lmax+1, 2lmax+1, B)` shape, which
    # `_assemble_G!` indexes directly; a packed real one is `(K, B)` and is expanded before assembly.
    z3() = _hermitian_domain(plan) ? _zeros_like(plan.G, T, _herm_len(lmax), B) :
                                     _zeros_like(plan.G, CT, lmax + 1, 2lmax + 1, B)
    vB() = _zeros_like(_θnodes(plan), T, B)
    hB() = zeros(T, B)
    nrm, cf = vB(), vB()
    return LSMRWorkspace(z3(), z3(), z3(), z3(),
                         _zeros_like(plan.G, CT, lmax + 1, 2lmax + 1, B),
                         _zeros_like(_fbuf(plan), eltype(_fbuf(plan)), M, B),
                         nothing, collect(1:B),
                         nrm, cf, _host_mirror(nrm), _host_mirror(cf),
                         hB(), hB(), hB(), hB(), hB(), hB(), hB(), hB(),
                         hB(), hB(), hB(), hB(), hB(), hB(), hB())
end

@inline _coefflen(plan::SpinNUSHTplan) =
    _hermitian_domain(plan) ? _herm_len(plan.lmax) : (plan.lmax + 1) * (2plan.lmax + 1)

# `v` starts as `A†f` and is folded with `A†u` each iteration. Both arrive in the full complex layout,
# so on a real field both go through the packing, which is where the restriction to the Hermitian
# subspace is applied — once, rather than as a projection bolted onto every step.
_lsmr_init_v!(ws, plan::SpinNUSHTplan) = _lsmr_init_v!(ws, plan, _fold_kind(plan))
_lsmr_init_v!(ws, ::SpinNUSHTplan, ::FullModes) = copyto!(ws.v, ws.w)
_lsmr_init_v!(ws, plan::SpinNUSHTplan, ::Union{FoldedComplex,FoldedReal}) =
    _pack_herm!(ws.v, ws.w, plan.lmax, plan.B)

_lsmr_fold_v!(ws, plan::SpinNUSHTplan, n::Integer) = _lsmr_fold_v!(ws, plan, n, _fold_kind(plan))
_lsmr_fold_v!(ws, ::SpinNUSHTplan, n::Integer, ::FullModes) = _col_pbp!(ws.v, ws.w, ws.cf, n)
_lsmr_fold_v!(ws, plan::SpinNUSHTplan, n::Integer, ::Union{FoldedComplex,FoldedReal}) =
    _pack_herm!(ws.v, ws.w, plan.lmax, n, ws.cf)

# The caller's `sf` is the full complex array whatever the iterate is, so a packed solution expands on
# the way out.
_write_solution!(C, ws::LSMRWorkspace, plan::SpinNUSHTplan, slot::Integer, dstcol::Integer) =
    _write_solution!(C, ws, plan, slot, dstcol, _fold_kind(plan))
_write_solution!(C, ws::LSMRWorkspace, plan::SpinNUSHTplan, slot::Integer, dstcol::Integer,
                 ::FullModes) = _copy_col!(C, dstcol, ws.x, slot, _coefflen(plan))
function _write_solution!(C, ws::LSMRWorkspace, plan::SpinNUSHTplan, slot::Integer, dstcol::Integer,
                          ::Union{FoldedComplex,FoldedReal})
    lmax = plan.lmax
    K = _herm_len(lmax)
    full = (lmax + 1) * (2lmax + 1)
    T = real(eltype(C))
    s2 = one(T) / sqrt(T(2))
    so = (slot - 1) * K
    do_ = (dstcol - 1) * full
    @inbounds begin
        for i in 1:full
            C[do_ + i] = zero(eltype(C))
        end
        # `spin_coeff_index` is Cartesian, so the destination column is a trailing index rather than a
        # linear offset — which also serves a 2-D `C` at `B = 1`, its trailing axis being singleton.
        for ℓ in 0:lmax
            o = _herm_offset(ℓ)
            C[spin_coeff_index(ℓ, 0, lmax), dstcol] = ws.x[so + o + 1]
            for m in 1:ℓ
                a = complex(ws.x[so + o + 2m], ws.x[so + o + 2m + 1]) * s2
                C[spin_coeff_index(ℓ,  m, lmax), dstcol] = a
                C[spin_coeff_index(ℓ, -m, lmax), dstcol] =
                    ifelse(iseven(m), one(T), -one(T)) * conj(a)
            end
        end
    end
    return C
end
# The spin transform has no width-narrowing machinery — `_assemble_G!` and the NUFFT are built for `B`
# columns — so compaction here reduces the per-column vector work only.
@inline _lsmr_widths(plan::SpinNUSHTplan, ::Integer) = (plan.B, plan.B)
@inline _assert_solve(sf, f, plan::SpinNUSHTplan) =
    (_assert_spin_coeffs(sf, plan); _assert_spin_field(f, plan))

_add_out!(f, fbuf, plan::SpinNUSHTplan, n) = _add_out!(f, fbuf, _θshift(plan), n)
_add_out!(f, fbuf, ::Nothing, n) = _add_field!(f, fbuf, n)
_add_out!(f, fbuf, ::AbstractVector, n) = _add_real!(f, fbuf, n)

"""
    _hermitian_domain(plan) -> Bool

Whether the plan's field is real, and so whether its coefficients are constrained.

A real field's coefficients are Hermitian — `sf[ℓ,-m] = (-1)^m conj(sf[ℓ,m])`, from
`conj(Y_ℓm) = (-1)^m Y_ℓ,-m` — which is `(lmax+1)²` real numbers rather than the `(lmax+1)(2lmax+1)`
complex ones the array holds. Synthesis needs no such restriction: it is handed coefficients and
evaluates them. A *solve* does. Searching the unrestricted space asks a question with no unique
answer, and the operator's continuation off the Hermitian subspace is arbitrary — it is fixed by
whichever fold the backend uses, not by the mathematics — so the search lands wherever that
continuation happens to point.
"""
@inline _hermitian_domain(plan::SpinNUSHTplan) = _hermitian_domain(_fold_kind(plan))
@inline _hermitian_domain(::FullModes) = false
@inline _hermitian_domain(::Union{FoldedComplex,FoldedReal}) = true

# Packed offset of degree `ℓ`: the degrees below it occupy `Σ (2ℓ'+1) = ℓ²` slots. `m = 0` is one real
# number, each `m > 0` a real and an imaginary part.
@inline _herm_offset(ℓ::Integer) = ℓ * ℓ
@inline _herm_len(lmax::Integer) = (lmax + 1)^2

"""
    _unpack_herm!(sf, p, lmax, B) -> sf

Expand packed real coefficients into the full Hermitian array. Scaled by `1/√2` off `m = 0` so the map
is an **isometry**: `‖U p‖ = ‖p‖`, which is what keeps "minimum norm" meaning the same thing on both
sides of it. [`_pack_herm!`](@ref) is its exact adjoint, not its inverse-by-projection.
"""
function _unpack_herm!(sf, p, lmax::Integer, B::Integer)
    T = real(eltype(sf))
    s2 = one(T) / sqrt(T(2))
    K = _herm_len(lmax)
    fill!(sf, zero(eltype(sf)))
    @inbounds for b in 1:B
        po = (b - 1) * K
        for ℓ in 0:lmax
            o = _herm_offset(ℓ)
            sf[spin_coeff_index(ℓ, 0, lmax), b] = p[po + o + 1]
            for m in 1:ℓ
                a = complex(p[po + o + 2m], p[po + o + 2m + 1]) * s2
                sf[spin_coeff_index(ℓ,  m, lmax), b] = a
                sf[spin_coeff_index(ℓ, -m, lmax), b] = ifelse(iseven(m), one(T), -one(T)) * conj(a)
            end
        end
    end
    return sf
end

"""
    _pack_herm!(p, g, lmax, B, β = nothing) -> p

The exact adjoint of [`_unpack_herm!`](@ref) under the real inner product `Re⟨a,b⟩`. With `β` given it
also folds `p ← U†g + β·p`, which is the packed counterpart of `_col_pbp!` and saves a pass.
"""
function _pack_herm!(p, g, lmax::Integer, B::Integer, β = nothing)
    T = real(eltype(g))
    s2 = one(T) / sqrt(T(2))
    K = _herm_len(lmax)
    @inbounds for b in 1:B
        po = (b - 1) * K
        c = β === nothing ? zero(T) : T(β[b])
        for ℓ in 0:lmax
            o = _herm_offset(ℓ)
            i0 = po + o + 1
            p[i0] = real(g[spin_coeff_index(ℓ, 0, lmax), b]) + c * p[i0]
            for m in 1:ℓ
                gp = g[spin_coeff_index(ℓ,  m, lmax), b]
                gm = g[spin_coeff_index(ℓ, -m, lmax), b]
                sg = ifelse(iseven(m), one(T), -one(T))
                ir, ii = po + o + 2m, po + o + 2m + 1
                p[ir] = (real(gp) + sg * real(gm)) * s2 + c * p[ir]
                p[ii] = (imag(gp) - sg * imag(gm)) * s2 + c * p[ii]
            end
        end
    end
    return p
end

# `u ← A v − α u`, with `A v` landing in the plan's own strengths buffer so no second point-space
# array is needed: scale `u` first, then accumulate the synthesis onto it. On a real field the iterate
# is packed, and expanding it into `ws.w` is free: that buffer is live only between `_lsmr_Atu!`
# writing it and the fold that reads it, which is exactly the window this fills.
function _lsmr_Av_axpy!(ws::LSMRWorkspace, plan::SpinNUSHTplan, ::Integer, ::Integer, n::Integer)
    _col_scale!(ws.u, ws.cf, n)
    _assemble_G!(plan.G, _av_coeffs(ws, plan), plan)
    _nufft_exec!(_nufft2(plan), plan.G, _fbuf(plan))
    _rephase!(_fbuf(plan), _θshift(plan))
    return _add_out!(ws.u, _fbuf(plan), plan, n)
end

@inline _av_coeffs(ws, plan::SpinNUSHTplan) = _av_coeffs(ws, plan, _fold_kind(plan))
@inline _av_coeffs(ws, ::SpinNUSHTplan, ::FullModes) = ws.v
@inline _av_coeffs(ws, plan::SpinNUSHTplan, ::Union{FoldedComplex,FoldedReal}) =
    _unpack_herm!(ws.w, ws.v, plan.lmax, plan.B)

function _lsmr_Atu!(ws::LSMRWorkspace, plan::SpinNUSHTplan, ::Integer, ::Integer)
    nusht_type1_spin!(ws.w, ws.u, plan)
    return ws.w
end

"""
    nusht_solve_spin!(sf, f, plan; ws=LSMRWorkspace(plan), maxiter=500, rtol=1e-8, conlim=0, verbose=false)

Exact inversion of the spin-weighted synthesis at arbitrary scattered points: solve
`min ‖A sf − f‖` by LSMR on the Golub–Kahan bidiagonalization of `A`. Batched (`B > 1`) runs the
columns as independent single-column solves. Same contract and return as [`nusht_solve!`](@ref).

On a real-field plan the fit runs over the `(lmax+1)²` real degrees a real field's coefficients have
rather than the full complex array — see [`_hermitian_domain`](@ref) — which is both the well-posed
problem and a quarter of the unknowns.

That restriction exists only at `s = 0`. Conjugating a spin-`s` field flips its spin weight
(`conj(ₛY_ℓm) = (-1)^{m+s} ₋ₛY_ℓ,-m`), so for `s ≠ 0` no reality condition closes, no subspace makes
the fit well-posed, and this raises rather than returning one of the arbitrarily many coefficient sets
that reproduce the samples. Synthesis and the adjoint are unaffected and stay exact there.
"""
function nusht_solve_spin!(sf, f, plan::SpinNUSHTplan;
                           ws::LSMRWorkspace = LSMRWorkspace(plan), kwargs...)
    if _hermitian_domain(plan) && plan.s != 0
        throw(ArgumentError(
            "a spin-$(plan.s) field cannot be real — conjugation maps spin s to spin -s — so a " *
            "real-field plan at s ≠ 0 has no coefficient subspace on which this fit is determined. " *
            "Build the plan with a complex element type to invert, or keep the real one for " *
            "synthesis and the adjoint, which are exact."))
    end
    return _lsmr!(sf, f, plan, ws; kwargs...)
end
