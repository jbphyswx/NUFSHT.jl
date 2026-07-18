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

using FINUFFT: FINUFFT
using LinearAlgebra: LinearAlgebra
using SpecialFunctions: loggamma

export SpinNUSHTplan, make_spin_plan, SpinCGWorkspace
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
    SpinNUSHTplan{T}

Plan for spin-`s` non-uniform spherical harmonic transforms at `M` scattered points up to degree
`lmax`, transforming `B` co-located fields per call. Owns persistent FINUFFT guru plans (type 2
`iflag=+1`, type 1 `iflag=−1`) whose points are set to `−θ` once. The Wigner `Δ^ℓ = d^ℓ(π/2)` planes
are generated **on the fly** by the Trapani–Navaza recurrence into two reused `(2lmax+1)²` buffers
(`dl_curr`/`dl_prev`), so the plan is **O(lmax²) memory** (the earlier dense `Δ[ℓ]` store was
O(lmax³)). The recurrence is also numerically stable to `ℓ ≈ 1024`, whereas the explicit-factorial
`wigner_d` sum it replaced loses all accuracy above `ℓ ≈ 40`.
"""
struct SpinNUSHTplan{T<:AbstractFloat, RV<:AbstractVector{T}, MT<:AbstractMatrix{T},
                     CT3<:AbstractArray{Complex{T},3}, CT2<:AbstractArray{Complex{T},2}, N1, N2, FT}
    lmax::Int
    s::Int
    B::Int
    tol::FT
    θ_nodes::RV
    φ_nodes::RV
    dl_curr::MT            # (2lmax+1)² reused Wigner-d(π/2) plane for the current degree ℓ
    dl_prev::MT            # (2lmax+1)² reused plane for degree ℓ-1 (Trapani–Navaza recurrence)
    G::CT3                 # (L, L, B) bivariate-Fourier mode buffer, L = 2lmax+1
    fbuf::CT2              # (M, B) scattered strengths
    nufft_type2::N2
    nufft_type1::N1
end

"""
    make_spin_plan(θ_nodes, φ_nodes, lmax, s; tol=1e-10, T=Float64, ntrans=1)

Build a `SpinNUSHTplan`. Colatitudes `θ ∈ [0,π]`, longitudes `φ ∈ [0,2π)`.
"""
function make_spin_plan(θ_nodes, φ_nodes, lmax::Integer, s::Integer;
                        tol = 1e-10, T::Type{<:AbstractFloat} = Float64, ntrans::Integer = 1,
                        nthreads::Integer = 0)
    @assert length(θ_nodes) == length(φ_nodes)
    B = Int(ntrans)
    @assert B ≥ 1
    L = 2lmax + 1
    M = length(θ_nodes)

    # Node vectors keep their input array type (host `Vector` or device array), with eltype `T`.
    θ = T.(θ_nodes)
    φ = T.(φ_nodes)
    negθ = -θ                                  # absorbs the e^{-im'θ} factor into iflag=+1

    # Buffers are shaped like the nodes, so device nodes ⇒ device-resident plan. Reused Wigner-d(π/2)
    # recurrence buffers are O(lmax²) (vs the old O(lmax³) dense per-ℓ store).
    dl_curr = _zeros_like(θ, T, L, L)
    dl_prev = _zeros_like(θ, T, L, L)
    G    = _zeros_like(θ, Complex{T}, L, L, B)
    fbuf = _zeros_like(θ, Complex{T}, M, B)

    # NUFFT built through the backend seam: host nodes → FINUFFT, device nodes → cuFINUFFT (CUDA ext).
    # nthreads forwarded (FINUFFT: 0 = all cores; cuFINUFFT ignores it).
    nthr = Int(nthreads)
    n_modes = Int64[L, L]
    nufft_type2 = _nufft_makeplan(negθ, 2, n_modes, +1, B, Float64(tol); dtype = T, nthreads = nthr)
    nufft_type1 = _nufft_makeplan(negθ, 1, n_modes, -1, B, Float64(tol); dtype = T, nthreads = nthr)
    _nufft_setpts!(nufft_type2, negθ, φ)
    _nufft_setpts!(nufft_type1, negθ, φ)
    finalizer(_nufft_destroy!, nufft_type2)
    finalizer(_nufft_destroy!, nufft_type1)

    tol64 = Float64(tol)
    return SpinNUSHTplan{T, typeof(θ), typeof(dl_curr), typeof(G), typeof(fbuf),
                         typeof(nufft_type1), typeof(nufft_type2), typeof(tol64)}(
        lmax, Int(s), B, tol64, θ, φ, dl_curr, dl_prev, G, fbuf, nufft_type2, nufft_type1)
end

"""
    close!(plan::SpinNUSHTplan)

Eagerly free the FINUFFT guru plans owned by `plan` (otherwise freed by their finalizers). Idempotent.
"""
function close!(plan::SpinNUSHTplan)
    _nufft_destroy!(plan.nufft_type2)
    _nufft_destroy!(plan.nufft_type1)
    return nothing
end

# Safe one-line `show` (see the NUSHTplan note): avoid recursing into stored FFTW/FINUFFT plan
# pointers, whose printers can segfault on an invalidated C state.
Base.show(io::IO, plan::SpinNUSHTplan{T}) where {T} =
    print(io, "SpinNUSHTplan{", T, "}(lmax=", plan.lmax, ", s=", plan.s, ", M=", length(plan.θ_nodes), ", B=", plan.B, ")")

"""
    _wigner_d_halfpi_step!(dl, dlp, ℓ, off) -> dl

Overwrite `dl` with the degree-`ℓ` Wigner-d plane at β = π/2, `dl[m+off, n+off] = d^ℓ_{m,n}(π/2)`
(McEwen–Wiaux/ssht convention), from the degree-`(ℓ-1)` plane `dlp`, via the **Trapani–Navaza**
recurrence (Trapani & Navaza 2006; ssht `ssht_dl.c`; s2fft `trapani.py`). Both `dl`/`dl_prev` are
`(2lmax+1)²`, `off = lmax+1`. O(ℓ²) work, one previous plane — so a full sweep ℓ = 0…lmax is O(lmax³)
work in O(lmax²) memory, numerically stable to ℓ ≈ 1024 (double).

Relation to this package's `wigner_d` convention (which is the transpose): `wigner_d(ℓ, a, b, π/2) ==
dl[b+off, a+off]`. Only the eighth `0 ≤ n ≤ m ≤ ℓ` is recurred; the rest is filled by the d(π/2)
symmetries (transpose, m→−m, n→−n).
"""
function _wigner_d_halfpi_step!(dl::AbstractMatrix{T}, dlp::AbstractMatrix{T}, ℓ::Integer, off::Integer) where {T}
    if ℓ == 0
        dl[off, off] = one(T)
        return dl
    end
    @inbounds begin
        # Stage A — boundary row m = ℓ  (T&N Eqns 9 & 10), read from the previous plane's row ℓ-1.
        dl[ℓ + off, off] = -sqrt(T(2ℓ - 1) / T(2ℓ)) * dlp[(ℓ - 1) + off, off]
        for n in 1:ℓ
            dl[ℓ + off, n + off] =
                sqrt(T(ℓ) * T(2ℓ - 1) / (T(2) * T(ℓ + n) * T(ℓ + n - 1))) * dlp[(ℓ - 1) + off, (n - 1) + off]
        end
        # Stage B — three-term downward recurrence in m at fixed n (T&N Eqn 11), eighth 0 ≤ n ≤ m ≤ ℓ.
        for n in 0:ℓ
            if ℓ - 1 ≥ n                                   # m = ℓ-1 (second term vanishes)
                m = ℓ - 1
                dl[m + off, n + off] = T(2n) / sqrt(T(ℓ - m) * T(ℓ + m + 1)) * dl[(m + 1) + off, n + off]
            end
            for m in (ℓ - 2):-1:n
                s1 = sqrt(T(ℓ - m) * T(ℓ + m + 1))
                s2 = sqrt(T(ℓ - m - 1) * T(ℓ + m + 2) / (T(ℓ - m) * T(ℓ + m + 1)))
                dl[m + off, n + off] =
                    T(2n) / s1 * dl[(m + 1) + off, n + off] - s2 * dl[(m + 2) + off, n + off]
            end
        end
        # Symmetry fill to the full (2ℓ+1)² plane.
        for m in 0:ℓ, n in (m + 1):ℓ                       # S1 transpose: eighth → quarter
            dl[m + off, n + off] = ifelse(iseven(m + n), one(T), -one(T)) * dl[n + off, m + off]
        end
        for n in 0:ℓ, m in -ℓ:-1                           # S2 m → −m: quarter → half
            dl[m + off, n + off] = ifelse(iseven(ℓ + n), one(T), -one(T)) * dl[(-m) + off, n + off]
        end
        for m in -ℓ:ℓ, n in -ℓ:-1                          # S3 n → −n: half → full
            dl[m + off, n + off] = ifelse(iseven(ℓ + abs(m)), one(T), -one(T)) * dl[m + off, (-n) + off]
        end
    end
    return dl
end

@inline _spin_npts(plan::SpinNUSHTplan) = length(plan.θ_nodes)

@inline function _assert_spin_coeffs(sf, plan::SpinNUSHTplan)
    @assert length(sf) == (plan.lmax + 1) * (2plan.lmax + 1) * plan.B "spin coefficient array has $(length(sf)) entries, expected (lmax+1)·(2lmax+1)·B"
end
@inline function _assert_spin_field(f, plan::SpinNUSHTplan)
    @assert length(f) == _spin_npts(plan) * plan.B "spin field array has $(length(f)) entries, expected M·B"
end

# Assemble bivariate Fourier coefficients G_{m'm} (CMCL-centered) from coefficients `sf`, in place.
# `sf` is indexed `[ℓ+1, m+lmax+1, b]`, which works for a 2-D `(lmax+1,2lmax+1)` array (B=1, trailing
# singleton) or the batched 3-D array — no reshape, so the hot path stays zero-alloc. The Wigner-d(π/2)
# plane for each degree ℓ is generated on the fly (Trapani–Navaza) into the reused `dl_curr`/`dl_prev`
# buffers, ℓ ascending, ping-ponged. Δ^ℓ_{mp,m} = wigner_d(ℓ,mp,m,π/2) = dl[m+off, mp+off].
function _assemble_G!(G, sf, plan::SpinNUSHTplan{T}) where {T}
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
        for b in 1:plan.B
            for m in -ℓ:ℓ
                val = sf[ℓ + 1, m + lmax + 1, b]
                val == 0 && continue
                ph = (Complex{T}(0, 1))^(m + s) * val * Nℓ
                for mp in -ℓ:ℓ
                    G[mp + off, m + off, b] += ph * dl[m + off, mp + off] * dl[-s + off, mp + off]
                end
            end
        end
    end
    return G
end

# Adjoint of _assemble_G!: contract NUFFT-type1 modes Ĝ back to coefficients, in place. Same on-the-fly
# recurrence for Δ^ℓ.
function _assemble_G_adjoint!(sf, Ĝ, plan::SpinNUSHTplan{T}) where {T}
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
        for b in 1:plan.B
            for m in -ℓ:ℓ
                phc = conj((Complex{T}(0, 1))^(m + s)) * Nℓ
                acc = zero(Complex{T})
                for mp in -ℓ:ℓ
                    acc += dl[m + off, mp + off] * dl[-s + off, mp + off] * Ĝ[mp + off, m + off, b]
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
    _nufft_exec!(plan.nufft_type2, plan.G, plan.fbuf)
    copyto!(f, plan.fbuf)
    return f
end

"""
    nusht_type1_spin!(sf, f, plan) -> sf

Exact Euclidean adjoint of `nusht_type2_spin!`: scattered values `f` → spin-`s` coefficients `sf`.
"""
function nusht_type1_spin!(sf, f, plan::SpinNUSHTplan{T}) where {T}
    _assert_spin_field(f, plan)
    _assert_spin_coeffs(sf, plan)
    copyto!(plan.fbuf, f)
    _nufft_exec!(plan.nufft_type1, plan.fbuf, plan.G)
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
    vB() = _zeros_like(plan.θ_nodes, T, B)
    return SpinCGWorkspace(z3(), z3(), z3(), z3(), z3(), _zeros_like(plan.fbuf, Complex{T}, M, B),
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
