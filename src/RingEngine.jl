# The ring S-step of a spin plan. For each order `m` the degree sum
#
#     F_m(θ) = Σ_ℓ ₛf_{ℓm} N_ℓ d^ℓ_{m,−s}(θ)
#
# is evaluated at the rings `θ_j = 2πj/N`, `j = 0 … N/2`, by the three-term recurrence of `d^ℓ_{m,n}(θ)`
# in `ℓ` (`n = −s`, `x = cos θ`),
#
#     ℓ √(((ℓ+1)²−m²)((ℓ+1)²−n²)) d^{ℓ+1} = (2ℓ+1)(ℓ(ℓ+1)x − mn) d^ℓ − (ℓ+1) √((ℓ²−m²)(ℓ²−n²)) d^{ℓ−1},
#
# started from the closed form of `d^{ℓ0}`, `ℓ0 = max(|m|,|n|)` (`d^1_{00} = x` where `ℓ0 = 0`). With
# `F_m(θ) = Σ_{m'} G_{m'm} e^{−im'θ}` and `F_m(2π − θ) = (−1)^{m+s} F_m(θ)`, one length-`N` FFT of the
# ring values gives column `m` of `G`, exactly for `N ≥ 2lmax + 2`.
#
# Written as `y_{ℓ+1} = (αx + β) y_ℓ − γ y_{ℓ−1}`, a rounding error `ε` in `αx + β` moves the angle the
# recurrence oscillates at by `ε / sin θ`. The recurrence instead carries `e_ℓ = y_ℓ − σ y_{ℓ−1}` with
# `u = 1 − σx`, `σ = 1` near `θ = 0` and `σ = −1` near `θ = π` (Reinsch's modification):
#
#     e_{ℓ+1} = (c_σ − σαu) y_ℓ + σγ e_ℓ,    y_{ℓ+1} = σ y_ℓ + e_{ℓ+1},
#
# `c₊ = α + β − 1 − γ`, `c₋ = β − α + 1 + γ`. `u` is `2sin²(θ/2)` or `2cos²(θ/2)`, accurate where it is
# small, and the angle moves by `ε tan(θ/2)` per step, so the values carry an error of order `ℓ eps(R)`
# in the plan's precision `R`.
#
# Float32 plans take that form. Wider precisions take the plain one, `cos θ` carried as `x + xl`: its
# error at the polar rings is of order `√ℓ lmax eps / π`, and its step is one dependent operation where
# Reinsch's is two.
#
# Below its turning point `d^ℓ` is exponentially small, so a ring value is held as `y·2^{Kk}` with an
# integer scale `k ≤ 0`, rescaled as it grows, and enters the sum once `k ≥ −1`; a term left out is below
# `2^{−3K/2}`.

# Rings a host task carries through the recurrence together: two vector registers of `R` on AVX2, so two
# independent chains cover the step's two dependent operations.
@inline _ring_lanes(::Type{Float32}) = 16
@inline _ring_lanes(::Type{<:AbstractFloat}) = 8

# Whether plans of precision `R` step in Reinsch's form.
@inline _ring_reinsch(::Type{Float32}) = true
@inline _ring_reinsch(::Type{<:AbstractFloat}) = false
const _RING_MIN_WORK = 50_000            # ring steps of a transform below which it runs on one task

# The circle's sample count for degree `lmax`: even, at least `2lmax + 2`, with factors 2, 3 and 5 only.
@inline _ring_samples(lmax::Integer) = 2nextprod((2, 3, 5), lmax + 1)

"""
    _RingEngine

The ring S-step of a spin plan in the plan's precision `R`: the circle's sample count `N`; per ring `u`
and `pos`, `u = 1 − x` where `pos` (`σ = 1`) and `1 + x` elsewhere, `pos` shared by the rings of a host
chunk, and `x + xl`, the ring cosine to twice `R`'s precision (all padded to a whole number of chunks);
`log2 cos(θ_j/2)` and `log2 sin(θ_j/2)` in `S`, the
precision of the starting values; and per order `m` the constants of
`d^{ℓ0}_{m,−s}(θ) = sg · 2^{lC + p1 log2 cos(θ/2) + p2 log2 sin(θ/2)}`. `work` holds one
[`_RingWork`](@ref) per task on the host, and the device buffers and FFT plans of a device plan.
"""
struct _RingEngine{R<:AbstractFloat, S<:AbstractFloat, V<:AbstractVector{R}, VB<:AbstractVector{Bool},
                   VS<:AbstractVector{S}, VI<:AbstractVector{Int32}, W}
    N::Int
    nring::Int
    every::Int             # steps between scale checks of a host chunk; see _ring_check_interval
    u::V
    pos::VB
    x::V
    xl::V
    lc::VS
    ls::VS
    lC::VS
    p1::VI
    p2::VI
    sg::VS
    work::W
end

"""
    _ring_tables(R, S, lmax, s, padded) -> NamedTuple

The host tables of a ring engine: `u` in `R` from `2sin²(θ_j/2)` or `2cos²(θ_j/2)` in `BigFloat`, `pos`
true on the chunks whose middle lies at `θ ≤ π/2`, `x`, `xl` in `R` from `cos θ_j` in `BigFloat`, and the
logarithms and order constants in `S`,
formed in `Float64` at least. `padded` rounds the ring count up to whole chunks, the extra rings
starting at zero.
"""
function _ring_tables(::Type{R}, ::Type{S}, lmax::Int, s::Int, padded::Int) where {R,S}
    F = promote_type(S, Float64)
    N = _ring_samples(lmax)
    nring = N ÷ 2 + 1
    u = zeros(R, padded); x = zeros(R, padded); xl = zeros(R, padded)
    c = _ring_lanes(R)
    pos = [4(j - j % c + c ÷ 2) <= N for j in 0:(padded - 1)]
    lc = fill(S(-Inf), padded)
    ls = fill(S(-Inf), padded)
    for j in 0:(nring - 1)
        h = pos[j + 1] ? sinpi(BigFloat(j) / N) : cospi(BigFloat(j) / N)
        u[j + 1] = R(2h^2)
        cj = cospi(BigFloat(2j) / N)
        x[j + 1] = R(cj); xl[j + 1] = R(cj - x[j + 1])
        lc[j + 1] = S(log2(cospi(F(j) / N)))
        ls[j + 1] = S(log2(sinpi(F(j) / N)))
    end
    n = -s
    no = 2lmax + 1
    lC = zeros(S, no); sg = zeros(S, no)
    p1 = zeros(Int32, no); p2 = zeros(Int32, no)
    for m in -lmax:lmax
        st = FTB.wigner_d_start(S, m, n)
        i = m + lmax + 1
        lC[i] = st.lC; p1[i] = st.p1; p2[i] = st.p2; sg[i] = st.sg
    end
    return (; N, nring, u, pos, x, xl, lc, ls, lC, p1, p2, sg)
end

# `d^{ℓ0}` at one ring as `(y, k)` from the order constants, held per order in `lC`, `p1`, `p2`, `sg` so a
# device work-item reads them from arrays, and the ring's logarithms.
@inline _ring_start(::Type{R}, lC::S, p1::Int32, p2::Int32, sg::S, lc::S, ls::S) where {R,S} =
    FTB.wigner_d_start_value(R, FTB.WignerDStart{S}(lC, p1, p2, sg), lc, ls)

# The weight of a value at scale `k`: whole at `0`, `dn = 2^{−K}` at `−1`, left out below.
@inline _ring_weight(k::Int32, dn::R) where {R} =
    ifelse(k == Int32(0), one(R), ifelse(k == Int32(-1), dn, zero(R)))

"""
    _ring_check_interval(R, lmax) -> Int

Steps a host chunk takes between scale checks. One step multiplies a value by at most
`√(2ℓ+1) + 2` at degree `ℓ ≤ lmax`, so over the interval a value below `2^{K/2}` stays below
`2^{K−8}`, and a value held at weight zero stays below `2^{−K−8}` of the largest.
"""
_ring_check_interval(::Type{R}, lmax::Integer) where {R} =
    max(1, floor(Int, (FTB.wigner_d_span(R) ÷ 2 - 8) / log2(sqrt(2lmax + 1) + 2)))

const _RING_DEGREES = 16                 # degrees the adjoint carries every ring through per sweep

"""
    _RingWork

One host task's buffers for the ring S-step: the step coefficients `A`, `Cp`, `Cm`, `Γ` (`α`, `c₊`, `c₋`,
`γ`) of an order and the weighted coefficients `cr`, `ci` of up to two batch columns, `(lmax+1)` long;
the ring values `Fr`, `Fi` of those columns; the adjoint's ring state `Ys`, `Es`, `Ks` between sweeps and
its per-lane sums `Q` over a sweep's degrees, `(lane, degree, re/im, column)`; and the circle `circ`
with its in-place FFTs.
"""
struct _RingWork{R, VR<:AbstractVector{R}, MR<:AbstractMatrix{R}, VK<:AbstractVector{Int32},
                 AR<:AbstractArray{R,4}, VC<:AbstractVector{Complex{R}}, PF, PB}
    A::VR
    Cp::VR
    Cm::VR
    Γ::VR
    cr::MR
    ci::MR
    Fr::MR
    Fi::MR
    Ys::VR
    Es::VR
    Ks::VK
    Q::AR
    circ::VC
    fwd::PF
    bwd::PB
end

function _ring_work(::Type{R}, lmax::Int, N::Int, padded::Int) where {R}
    L = lmax + 1
    circ = zeros(Complex{R}, N)
    fwd = FFTW.plan_fft!(circ; flags = FFTW.MEASURE, num_threads = 1)
    bwd = FFTW.plan_bfft!(circ; flags = FFTW.MEASURE, num_threads = 1)
    return _RingWork(zeros(R, L), zeros(R, L), zeros(R, L), zeros(R, L), zeros(R, L, 2), zeros(R, L, 2),
                     zeros(R, padded, 2), zeros(R, padded, 2), zeros(R, padded), zeros(R, padded),
                     zeros(Int32, padded), zeros(R, _ring_lanes(R), _RING_DEGREES, 2, 2), circ, fwd, bwd)
end

# Rings padded to whole chunks.
@inline _ring_padded(::Type{R}, lmax::Integer) where {R} =
    cld(_ring_samples(lmax) ÷ 2 + 1, _ring_lanes(R)) * _ring_lanes(R)

"""
    _ring_engine(θ, lmax, s, ntasks) -> _RingEngine

The ring S-step for a host plan over nodes `θ`, in their element type, its starting values in
`Float64` at least, with one workspace per task.
"""
function _ring_engine(::AbstractArray{R}, lmax::Int, s::Int, ntasks::Int) where {R}
    S = promote_type(R, Float64)
    padded = _ring_padded(R, lmax)
    t = _ring_tables(R, S, lmax, s, padded)
    work = [_ring_work(R, lmax, t.N, padded) for _ in 1:ntasks]
    return _RingEngine(t.N, t.nring, _ring_check_interval(R, lmax), t.u, t.pos, t.x, t.xl, t.lc, t.ls, t.lC,
                       t.p1, t.p2, t.sg, work)
end

Base.show(io::IO, e::_RingEngine{R}) where {R} =
    print(io, "_RingEngine{", R, "}(N=", e.N, ", rings=", e.nring, ")")

# Tasks one transform runs on: the plan's, fewer when the ring steps are too few to divide.
@inline _ring_tasks(plan::SpinNUSHTplan, eng::_RingEngine) =
    min(length(eng.work), max(1, ((plan.lmax + 1)^2 * eng.nring * plan.B) ÷ _RING_MIN_WORK))

@inline _lanes(f::F, ::Type{R}) where {F,R} = ntuple(f, Val(_ring_lanes(R)))
@inline _axpy(a, v, acc) = map((p, q) -> muladd(a, q, p), acc, v)

# `acc[g] + c[ℓ + 1, g] v` for each of the `G` batch columns.
@inline _axpy_cols(c, ℓ::Int, v, acc::NTuple{G}) where {G} =
    ntuple(g -> _axpy(@inbounds(c[ℓ + 1, g]), v, acc[g]), Val(G))

# One degree up across a chunk's lanes, on side `P`. The chunk's ring constants `g` select the form:
# `u` lanes step in Reinsch's form from `(y, e)`, `(x, xl)` lanes in the plain one from `(y, y_{ℓ−1})`,
# where the step coefficients read `(α, β, ·, γ)`.
@inline function _ring_step(::Val{P}, α, c₊, c₋, γ, u::Tuple{AbstractFloat,Vararg{AbstractFloat}}, y, e) where {P}
    σα, c, σγ, σ = FTB.wigner_d_signed(P, α, c₊, c₋, γ)
    ye = map((uj, yj, ej) -> FTB.wigner_d_step(σα, c, σγ, σ, uj, yj, ej), u, y, e)
    return map(first, ye), map(last, ye)
end
@inline _ring_step(::Val, α, β, _, γ, (x, xl)::Tuple{Tuple,Tuple}, y, yo) =
    map((xj, lj, yj, oj) -> first(FTB.wigner_d_step_plain(α, β, γ, xj, lj, yj, oj)), x, xl, y, yo), y

# The ring constants of the chunk at `j0` its step reads, and the second state value at the start degree:
# `e = y` in Reinsch's form, `y_{ℓ0−1} = 0` in the plain one.
@inline _ring_geometry(eng::_RingEngine{R}, j0::Int) where {R} =
    _ring_reinsch(R) ? _load_lanes(eng.u, j0, R) : (_load_lanes(eng.x, j0, R), _load_lanes(eng.xl, j0, R))
@inline _ring_e0(::Tuple{AbstractFloat,Vararg{AbstractFloat}}, y) = y
@inline _ring_e0(::Tuple{Tuple,Tuple}, y) = map(zero, y)

# A scaled value above `up` and its `e` scaled by `dn` together, across the chunk's lanes.
@inline function _ring_rescale(y, e, k, up::R, dn::R) where {R}
    r = map((yj, ej, kj) -> FTB.wigner_d_rescale(yj, ej, kj, up, dn), y, e, k)
    return map(t -> t[1], r), map(t -> t[2], r), map(t -> t[3], r)
end

# The chunk of rings `j0 + 1 … j0 + _ring_lanes(R)` at its starting degree: its ring constants, `y`, `k`.
@inline function _ring_chunk_start(eng::_RingEngine{R}, i::Int, j0::Int) where {R}
    lC = eng.lC[i]; p1 = eng.p1[i]; p2 = eng.p2[i]; sg = eng.sg[i]
    yk = _lanes(l -> @inbounds(_ring_start(R, lC, p1, p2, sg, eng.lc[j0 + l], eng.ls[j0 + l])), R)
    return _ring_geometry(eng, j0), map(first, yk), map(last, yk)
end

# The chunk's side, as a type, for the kernels below.
@inline _ring_side(f::F, eng::_RingEngine, j0::Int) where {F} =
    @inbounds(eng.pos[j0 + 1]) ? f(Val(true)) : f(Val(false))

# Forward: `Fr + iFi` of the chunk's rings for `G` batch columns, `Σ_ℓ (cr + i ci)[ℓ] d^ℓ`.
@inline _ring_chunk!(w::_RingWork, eng::_RingEngine, i::Int, ℓ0::Int, lmax::Int, j0::Int, g::Val) =
    _ring_side(side -> _ring_chunk!(w, eng, i, ℓ0, lmax, j0, g, side), eng, j0)

@inline function _ring_chunk!(w::_RingWork, eng::_RingEngine{R}, i::Int, ℓ0::Int, lmax::Int, j0::Int,
                              ::Val{G}, side::Val) where {R,G}
    u, y, k = _ring_chunk_start(eng, i, j0)
    up, dn = FTB.wigner_d_bounds(R)
    e = _ring_e0(u, y)
    ar = ntuple(_ -> _lanes(_ -> zero(R), R), Val(G)); ai = ar
    A = w.A; Cp = w.Cp; Cm = w.Cm; Γ = w.Γ; cr = w.cr; ci = w.ci
    ℓ = ℓ0
    @inbounds while ℓ <= lmax && any(<(Int32(0)), k)
        ℓe = min(lmax, ℓ + eng.every - 1)
        f = map(kj -> _ring_weight(kj, dn), k)
        if any(!iszero, f)
            while ℓ <= ℓe
                v = map(*, y, f)
                ar = _axpy_cols(cr, ℓ, v, ar)
                ai = _axpy_cols(ci, ℓ, v, ai)
                y, e = _ring_step(side, A[ℓ + 1], Cp[ℓ + 1], Cm[ℓ + 1], Γ[ℓ + 1], u, y, e)
                ℓ += 1
            end
        else
            while ℓ <= ℓe
                y, e = _ring_step(side, A[ℓ + 1], Cp[ℓ + 1], Cm[ℓ + 1], Γ[ℓ + 1], u, y, e)
                ℓ += 1
            end
        end
        y, e, k = _ring_rescale(y, e, k, up, dn)
    end
    @inbounds while ℓ <= lmax
        ar = _axpy_cols(cr, ℓ, y, ar)
        ai = _axpy_cols(ci, ℓ, y, ai)
        y, e = _ring_step(side, A[ℓ + 1], Cp[ℓ + 1], Cm[ℓ + 1], Γ[ℓ + 1], u, y, e)
        ℓ += 1
    end
    @inbounds for g in 1:G, l in 1:_ring_lanes(R)
        w.Fr[j0 + l, g] = ar[g][l]
        w.Fi[j0 + l, g] = ai[g][l]
    end
    return nothing
end

@inline _load_lanes(a, j0::Int, ::Type{R}) where {R} = _lanes(l -> @inbounds(a[j0 + l]), R)
@inline function _store_lanes!(a, j0::Int, v::NTuple{L}) where {L}
    ntuple(l -> (@inbounds a[j0 + l] = v[l]; nothing), Val(L))
    return nothing
end

# Adjoint: degrees `ℓb … ℓb + nb − 1` of the chunk's rings, from the state left by the previous sweep,
# their sums `Σ_rings d^ℓ (Fr + iFi)` added per lane into `Q[:, ℓ − ℓb + 1, :, g]`.
@inline _ring_chunk_adjoint!(w::_RingWork, eng::_RingEngine, ℓb::Int, nb::Int, j0::Int, g::Val) =
    _ring_side(side -> _ring_chunk_adjoint!(w, eng, ℓb, nb, j0, g, side), eng, j0)

@inline function _ring_chunk_adjoint!(w::_RingWork, eng::_RingEngine{R}, ℓb::Int, nb::Int, j0::Int,
                                      ::Val{G}, side::Val) where {R,G}
    up, dn = FTB.wigner_d_bounds(R)
    u = _ring_geometry(eng, j0)
    y = _load_lanes(w.Ys, j0, R); e = _load_lanes(w.Es, j0, R); k = _load_lanes(w.Ks, j0, R)
    hr = ntuple(g -> _lanes(l -> @inbounds(w.Fr[j0 + l, g]), R), Val(G))
    hi = ntuple(g -> _lanes(l -> @inbounds(w.Fi[j0 + l, g]), R), Val(G))
    A = w.A; Cp = w.Cp; Cm = w.Cm; Γ = w.Γ
    q = 1
    @inbounds while q <= nb && any(<(Int32(0)), k)
        qe = min(nb, q + eng.every - 1)
        f = map(kj -> _ring_weight(kj, dn), k)
        if any(!iszero, f)
            while q <= qe
                ℓ = ℓb + q - 1
                _ring_project!(w.Q, map(*, y, f), hr, hi, q)
                y, e = _ring_step(side, A[ℓ + 1], Cp[ℓ + 1], Cm[ℓ + 1], Γ[ℓ + 1], u, y, e)
                q += 1
            end
        else
            while q <= qe
                ℓ = ℓb + q - 1
                y, e = _ring_step(side, A[ℓ + 1], Cp[ℓ + 1], Cm[ℓ + 1], Γ[ℓ + 1], u, y, e)
                q += 1
            end
        end
        y, e, k = _ring_rescale(y, e, k, up, dn)
    end
    @inbounds while q <= nb
        ℓ = ℓb + q - 1
        _ring_project!(w.Q, y, hr, hi, q)
        y, e = _ring_step(side, A[ℓ + 1], Cp[ℓ + 1], Cm[ℓ + 1], Γ[ℓ + 1], u, y, e)
        q += 1
    end
    _store_lanes!(w.Ys, j0, y); _store_lanes!(w.Es, j0, e); _store_lanes!(w.Ks, j0, k)
    return nothing
end

@inline function _ring_project!(Q, v, hr::NTuple{1}, hi::NTuple{1}, q::Int)
    _ring_lane_fma!(Q, v, hr[1], q, 1, 1); _ring_lane_fma!(Q, v, hi[1], q, 2, 1)
    return nothing
end

@inline function _ring_project!(Q, v, hr::NTuple{2}, hi::NTuple{2}, q::Int)
    _ring_lane_fma!(Q, v, hr[1], q, 1, 1); _ring_lane_fma!(Q, v, hi[1], q, 2, 1)
    _ring_lane_fma!(Q, v, hr[2], q, 1, 2); _ring_lane_fma!(Q, v, hi[2], q, 2, 2)
    return nothing
end

# `Q[:, q, c, g] += v .* h` across the chunk's lanes.
@inline function _ring_lane_fma!(Q, v::NTuple{L}, h, q::Int, c::Int, g::Int) where {L}
    @inbounds for l in 1:L
        Q[l, q, c, g] = muladd(v[l], h[l], Q[l, q, c, g])
    end
    return nothing
end

# `sf[:, m, b … b + G − 1]` for order index `i`: every ring's state at `ℓ0`, then the degrees in sweeps
# of `_RING_DEGREES`, each sweep's per-lane sums reduced and scaled by `N_ℓ`.
function _ring_adjoint_sums!(sf, w::_RingWork{R}, eng::_RingEngine, i::Int, b::Int, ℓ0::Int, lmax::Int,
                             ::Val{G}) where {R,G}
    CT = eltype(sf)
    @inbounds for g in 1:G, ℓ in 0:(ℓ0 - 1)
        sf[ℓ + 1, i, b + g - 1] = zero(CT)
    end
    np = length(eng.u); c = _ring_lanes(R)
    for j0 in 0:c:(np - 1)
        g, y, k = _ring_chunk_start(eng, i, j0)
        _store_lanes!(w.Ys, j0, y); _store_lanes!(w.Ks, j0, k); _store_lanes!(w.Es, j0, _ring_e0(g, y))
    end
    ℓb = ℓ0
    while ℓb <= lmax
        nb = min(_RING_DEGREES, lmax - ℓb + 1)
        fill!(w.Q, zero(R))
        for j0 in 0:c:(np - 1)
            _ring_chunk_adjoint!(w, eng, ℓb, nb, j0, Val(G))
        end
        @inbounds for g in 1:G, q in 1:nb
            sr = zero(R); si = zero(R)
            for l in 1:c
                sr += w.Q[l, q, 1, g]; si += w.Q[l, q, 2, g]
            end
            ℓ = ℓb + q - 1
            sf[ℓ + 1, i, b + g - 1] = CT(complex(sr, si) * _ring_norm(R, ℓ))
        end
        ℓb += nb
    end
    return sf
end

# The step coefficients of order `m` from `ℓ0` to `lmax`.
# The plain form reads `β` from `Cp`.
function _ring_order_coeffs!(w::_RingWork{R}, m::Int, n::Int, ℓ0::Int, lmax::Int) where {R}
    F = promote_type(R, Float64)
    @inbounds for ℓ in ℓ0:lmax
        α, β, γ, c₊, c₋ = FTB.wigner_d_coeffs(F, ℓ, m, n)
        w.A[ℓ + 1] = α; w.Cm[ℓ + 1] = c₋; w.Γ[ℓ + 1] = γ
        w.Cp[ℓ + 1] = _ring_reinsch(R) ? c₊ : β
    end
    return w
end

@inline _ring_norm(::Type{R}, ℓ::Int) where {R} = sqrt(R(2ℓ + 1) / (4 * R(π)))

# `N_ℓ sf[ℓ, m, b]` into slot `g` of the weighted coefficients.
function _ring_weights!(w::_RingWork{R}, sf, i::Int, b::Int, g::Int, ℓ0::Int, lmax::Int) where {R}
    @inbounds for ℓ in ℓ0:lmax
        c = Complex{R}(sf[ℓ + 1, i, b]) * _ring_norm(R, ℓ)
        w.cr[ℓ + 1, g] = real(c); w.ci[ℓ + 1, g] = imag(c)
    end
    return w
end

# Column `b` of `G`, order `m`, from slot `g` of the ring values: the circle by the reflection, one
# backward FFT, and the rows of the plan's layout scaled by `1/N`.
function _ring_modes!(G, w::_RingWork{R}, eng::_RingEngine, plan::SpinNUSHTplan, kind, m::Int, b::Int,
                      g::Int) where {R}
    N = eng.N; nring = eng.nring; circ = w.circ
    @inbounds for j in 1:nring
        circ[j] = complex(w.Fr[j, g], w.Fi[j, g])
    end
    σ = _parity(R, m + plan.s)
    @inbounds for j in (nring + 1):N
        circ[j] = σ * circ[N - j + 2]
    end
    w.bwd * circ
    inv = one(R) / N
    col = m + plan.lmax + 1
    CT = eltype(G)
    @inbounds for mp in _mprange(plan.lmax, kind)
        G[_grow(plan, mp, kind), col, b] = CT(circ[mod(mp, N) + 1] * inv)
    end
    return G
end

# The transpose of `_ring_modes!`: slot `g` of the ring values from column `b`, order `m`, of `Ĝ`, the
# rows absent from the layout counting as zero.
function _ring_rings!(w::_RingWork{R}, eng::_RingEngine, Ĝ, plan::SpinNUSHTplan, kind, m::Int, b::Int,
                      g::Int) where {R}
    N = eng.N; nring = eng.nring; circ = w.circ
    col = m + plan.lmax + 1
    fill!(circ, zero(Complex{R}))
    @inbounds for mp in _mprange(plan.lmax, kind)
        circ[mod(mp, N) + 1] = Ĝ[_grow(plan, mp, kind), col, b]
    end
    w.fwd * circ
    σ = _parity(R, m + plan.s)
    inv = one(R) / N
    @inbounds begin
        w.Fr[1, g] = real(circ[1]) * inv; w.Fi[1, g] = imag(circ[1]) * inv
        for j in 2:(nring - 1)
            h = (circ[j] + σ * circ[N - j + 2]) * inv
            w.Fr[j, g] = real(h); w.Fi[j, g] = imag(h)
        end
        w.Fr[nring, g] = real(circ[nring]) * inv; w.Fi[nring, g] = imag(circ[nring]) * inv
        for j in (nring + 1):size(w.Fr, 1)
            w.Fr[j, g] = zero(R); w.Fi[j, g] = zero(R)
        end
    end
    return w
end

function _ring_forward_order!(G, sf, plan::SpinNUSHTplan, eng::_RingEngine{R}, w::_RingWork, kind,
                              m::Int) where {R}
    lmax = plan.lmax; n = -plan.s
    ℓ0 = max(abs(m), abs(n))
    i = m + lmax + 1
    _ring_order_coeffs!(w, m, n, ℓ0, lmax)
    b = 1
    while b <= plan.B
        if b < plan.B
            _ring_weights!(w, sf, i, b, 1, ℓ0, lmax); _ring_weights!(w, sf, i, b + 1, 2, ℓ0, lmax)
            for j0 in 0:_ring_lanes(R):(length(eng.u) - 1)
                _ring_chunk!(w, eng, i, ℓ0, lmax, j0, Val(2))
            end
            _ring_modes!(G, w, eng, plan, kind, m, b, 1); _ring_modes!(G, w, eng, plan, kind, m, b + 1, 2)
            b += 2
        else
            _ring_weights!(w, sf, i, b, 1, ℓ0, lmax)
            for j0 in 0:_ring_lanes(R):(length(eng.u) - 1)
                _ring_chunk!(w, eng, i, ℓ0, lmax, j0, Val(1))
            end
            _ring_modes!(G, w, eng, plan, kind, m, b, 1)
            b += 1
        end
    end
    return G
end

function _ring_adjoint_order!(sf, Ĝ, plan::SpinNUSHTplan, eng::_RingEngine, w::_RingWork, kind, m::Int)
    lmax = plan.lmax; n = -plan.s
    ℓ0 = max(abs(m), abs(n))
    i = m + lmax + 1
    _ring_order_coeffs!(w, m, n, ℓ0, lmax)
    b = 1
    while b <= plan.B
        if b < plan.B
            _ring_rings!(w, eng, Ĝ, plan, kind, m, b, 1); _ring_rings!(w, eng, Ĝ, plan, kind, m, b + 1, 2)
            _ring_adjoint_sums!(sf, w, eng, i, b, ℓ0, lmax, Val(2))
            b += 2
        else
            _ring_rings!(w, eng, Ĝ, plan, kind, m, b, 1)
            _ring_adjoint_sums!(sf, w, eng, i, b, ℓ0, lmax, Val(1))
            b += 1
        end
    end
    return sf
end

# The orders run on the transform's tasks, task `t` taking `±c` for `c ≡ t − 1 (mod nt)`; each order
# writes its own column of the output.
function _assemble_G_impl!(G, sf, plan::SpinNUSHTplan, eng::_RingEngine)
    kind = _fold_kind(plan)
    nt = _ring_tasks(plan, eng)
    _on_tasks(nt) do t
        w = eng.work[t]
        for c in (t - 1):nt:plan.lmax, m in -c:max(2c, 1):c
            _ring_forward_order!(G, sf, plan, eng, w, kind, m)
        end
    end
    return G
end

function _assemble_G_adjoint_impl!(sf, Ĝ, plan::SpinNUSHTplan, eng::_RingEngine)
    kind = _fold_kind(plan)
    nt = _ring_tasks(plan, eng)
    _on_tasks(nt) do t
        w = eng.work[t]
        for c in (t - 1):nt:plan.lmax, m in -c:max(2c, 1):c
            _ring_adjoint_order!(sf, Ĝ, plan, eng, w, kind, m)
        end
    end
    return sf
end
