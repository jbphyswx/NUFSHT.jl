"""
    NUFSHTKernelAbstractionsExt

Device-generic methods for the array-indexed steps of the transform, as KernelAbstractions `@kernel`s
dispatched on `GPUArraysCore.AbstractGPUArray`. They run on any KernelAbstractions backend (CUDA, ROCm,
Metal, oneAPI, and the `JLArray` / KA-CPU reference backends) and take precedence over the host methods
for a device array.

The kernels cover the scalar mode assembly and its adjoint, the spin ring S-step and its adjoint, the
solver's per-column steps, the weighted strengths and Hermitian projection of an analysis, the
scalar ↔ spin-0 conversions and the spectral filter. The NUFFT goes through the `_nufft_*` functions to
FlowTransformBindings, which builds the plan on the nodes' device.

Loaded by `using KernelAbstractions` (with `GPUArraysCore`).
"""
module NUFSHTKernelAbstractionsExt

using NUFSHT: NUFSHT
# `@kernel`, `@index` and `@Const` are imported by bare name: the `@kernel` macro recognizes the DSL
# keywords only unqualified, and a qualified `KA.@index` is not rewritten and miscompiles.
using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const, @localmem, @private, @synchronize
using GPUArraysCore: GPUArraysCore
using ComputationalBackends: ComputationalBackends

# A plan under `execution = GPUBackend(b)`: its nodes, and so every buffer, in `b`'s memory.
_on_device(b::ComputationalBackends.GPUBackend, x) =
    copyto!(KernelAbstractions.allocate(b.backend, eltype(x), (length(x),)), x)

NUFSHT._make_plan(::Type{FE}, θ, φ, lmax, b::ComputationalBackends.GPUBackend; kwargs...) where {FE} =
    NUFSHT._build_plan(FE, _on_device(b, θ), _on_device(b, φ), lmax; kwargs...)
NUFSHT._make_spin_plan(::Type{FE}, θ, φ, lmax, s, b::ComputationalBackends.GPUBackend; kwargs...) where {FE} =
    NUFSHT._build_spin_plan(FE, _on_device(b, θ), _on_device(b, φ), lmax, s; kwargs...)

# Kernel launches are asynchronous on device backends (CUDA/ROCm/…), so results are synchronized before
# the host reads them; JLArrays' reference backend is synchronous and defines no `synchronize`.
@inline function _sync(backend)
    applicable(KernelAbstractions.synchronize, backend) && KernelAbstractions.synchronize(backend)
    return nothing
end

# ── Device scalar mode assembly (F) ─────────────────────────────────────────────
# Both directions of `src/Modes.jl` are per-element gathers, so each is one workitem per output
# element with no atomics and no fill pass, calling the `_mode_entry`/`_mode_adjoint_entry` the host
# loops call. The plan buffers are `(…, …, B)`, so `ndrange` is the whole output including the batch
# axis.

@kernel function _modes_kern!(Z, @Const(G), lmax, offθ, offφ, n0, nm, h)
    r, c, b = @index(Global, NTuple)
    @inbounds Z[r, c, b] = NUFSHT._mode_entry(eltype(Z), G, lmax, r, c, b, offθ, offφ, n0, nm, h)
end

@kernel function _modes_adj_kern!(G, @Const(Z), lmax, offθ, offφ, fold, n0, nm, h)
    i, j, b = @index(Global, NTuple)
    @inbounds G[i, j, b] = NUFSHT._modes_out(eltype(G),
        NUFSHT._mode_adjoint_entry(eltype(Z), Z, lmax, i, j, b, offθ, offφ, fold, n0, nm, h))
end

function NUFSHT._assemble_modes!(Z::GPUArraysCore.AbstractGPUArray, G, lmax::Integer)
    n0, nm, h = NUFSHT._mode_norms(real(eltype(Z)))
    offθ = NUFSHT._folded(Z, lmax) ? 1 : size(Z, 1) ÷ 2 + 1
    backend = KernelAbstractions.get_backend(Z)
    _modes_kern!(backend)(Z, G, Int(lmax), offθ, size(Z, 2) ÷ 2 + 1, n0, nm, h;
                          ndrange = size(Z))
    _sync(backend)
    return Z
end

function NUFSHT._assemble_modes_adjoint!(G::GPUArraysCore.AbstractGPUArray, Z, lmax::Integer)
    n0, nm, h = NUFSHT._mode_norms(real(eltype(Z)))
    fold = NUFSHT._folded(Z, lmax)
    backend = KernelAbstractions.get_backend(G)
    _modes_adj_kern!(backend)(G, Z, Int(lmax), fold ? 1 : size(Z, 1) ÷ 2 + 1, size(Z, 2) ÷ 2 + 1,
                              fold, n0, nm, h; ndrange = size(G))
    _sync(backend)
    return G
end

# ── Device spin S-step: the ring recurrence of `src/RingEngine.jl` ──────────────────
# Forward: one workitem per (ring, order, column) carries `d^ℓ_{m,−s}` up the degrees and writes its
# ring value and its reflection into the circle `C`; one batched FFT along the circle; a gather into the
# rows of `G`. Adjoint: the transpose, the sum over rings of each degree a workgroup reduction, one
# workgroup per (ring block, order, column), the blocks added by a last kernel. The step coefficients
# are tables over `(ℓ, |m|)`, `c₊` and `c₋` stored for `m ≥ 0`: `c₊(−m) = −c₋(m)`, `c₋(−m) = −c₊(m)`.
# Launches on one backend run in order, so only a host read needs a sync.

"""
    _RingDevice

A device ring engine's buffers: the circle `C`, `(N, 2lmax+1, B)`; the adjoint's per-block degree sums
`P`, `(lmax+1, nblk, 2lmax+1, B)`; the step coefficient tables `A`, `Cp`, `Cm`, `Γ` (`α`, `c₊`, `c₋`,
`γ`), `(lmax+1, lmax+1)`; the in-place FFTs along the circle; and the adjoint's workgroup width `wg`.
"""
struct _RingDevice{CA, PA, MA, PF, PB}
    C::CA
    P::PA
    A::MA
    Cp::MA
    Cm::MA
    Γ::MA
    fwd::PF
    bwd::PB
    wg::Int
end

# The coefficient tables of `n = −s` in `R`, formed in `Float64`.
function _ring_coeff_tables(::Type{R}, lmax::Int, n::Int) where {R}
    A, Cp, Cm, Γ = (zeros(R, lmax + 1, lmax + 1) for _ in 1:4)
    for μ in 0:lmax, ℓ in max(μ, abs(n)):lmax
        α, c₊, c₋, γ = NUFSHT._ring_coeffs(promote_type(R, Float64), ℓ, μ, n)
        A[ℓ + 1, μ + 1] = α; Cp[ℓ + 1, μ + 1] = c₊; Cm[ℓ + 1, μ + 1] = c₋; Γ[ℓ + 1, μ + 1] = γ
    end
    return A, Cp, Cm, Γ
end

# The ring S-step for a device plan over nodes `θ`, in their element type; its starting values in
# `Float64` where the device has it.
function NUFSHT._spin_engine(θ::GPUArraysCore.AbstractGPUArray{R}, lmax::Int, s::Int, ::Int,
                             B::Int) where {R}
    backend = KernelAbstractions.get_backend(θ)
    S = KernelAbstractions.supports_float64(backend) ? promote_type(R, Float64) : R
    nring = NUFSHT._ring_samples(lmax) ÷ 2 + 1
    t = NUFSHT._ring_tables(R, S, lmax, s, nring)
    dev(a) = NUFSHT._to_like(θ, a)
    A, Cp, Cm, Γ = _ring_coeff_tables(R, lmax, -s)
    C = NUFSHT._zeros_like(θ, Complex{R}, t.N, 2lmax + 1, B)
    wg = min(1024, nextpow(2, nring))
    P = NUFSHT._zeros_like(θ, Complex{R}, lmax + 1, cld(nring, wg), 2lmax + 1, B)
    work = _RingDevice(C, P, dev(A), dev(Cp), dev(Cm), dev(Γ), NUFSHT.FFTW.plan_fft!(C, 1),
                       NUFSHT.FFTW.plan_bfft!(C, 1), wg)
    return NUFSHT._RingEngine(t.N, t.nring, NUFSHT._ring_check_interval(R, lmax), dev(t.u), dev(t.pos),
                              dev(t.lc), dev(t.ls), dev(t.lC), dev(t.p1), dev(t.p2), dev(t.sg), work)
end

# `(α, c₊, c₋, γ)` of the step from `ℓ` at order `m` from the tables.
@inline function _ring_table_coeffs(A, Cp, Cm, Γ, ℓ::Int, m::Int)
    μ = abs(m) + 1
    α, p, q, γ = @inbounds (A[ℓ + 1, μ], Cp[ℓ + 1, μ], Cm[ℓ + 1, μ], Γ[ℓ + 1, μ])
    neg = m < 0
    return α, ifelse(neg, -q, p), ifelse(neg, -p, q), γ
end

# One degree up from `(y, e)` at a ring on side `pos`, by the host's step.
@inline function _ring_device_step(A, Cp, Cm, Γ, ℓ::Int, m::Int, pos::Bool, u, y, e)
    α, c₊, c₋, γ = _ring_table_coeffs(A, Cp, Cm, Γ, ℓ, m)
    σα, c, σγ, σ = NUFSHT._ring_signed(pos, α, c₊, c₋, γ)
    e′ = NUFSHT._ring_e(σα, c, σγ, u, y, e)
    return NUFSHT._ring_y(σ, y, e′), e′
end

@kernel function _ring_forward_kern!(C, @Const(sf), @Const(u), @Const(pos), @Const(lc), @Const(ls),
                                     @Const(lC), @Const(p1), @Const(p2), @Const(sg), @Const(A), @Const(Cp),
                                     @Const(Cm), @Const(Γ), lmax, s, N, nring)
    j, i, b = @index(Global, NTuple)
    R = eltype(u)
    m = i - lmax - 1
    up, dn = NUFSHT._ring_bounds(R)
    @inbounds begin
        y, k = NUFSHT._ring_start(R, lC[i], p1[i], p2[i], sg[i], lc[j], ls[j])
        uj = u[j]; pj = pos[j]
        e = y
        acc = zero(Complex{R})
        for ℓ in max(abs(m), abs(s)):lmax
            c = Complex{R}(sf[ℓ + 1, i, b]) * NUFSHT._ring_norm(R, ℓ)
            acc += c * (y * NUFSHT._ring_weight(k, dn))
            yn, en = _ring_device_step(A, Cp, Cm, Γ, ℓ, m, pj, uj, y, e)
            big = (k < Int32(0)) & (abs(yn) > up)
            sc = ifelse(big, dn, one(R))
            e = en * sc; y = yn * sc; k += Int32(big)
        end
        C[j, i, b] = acc
        if 1 < j < nring
            C[N - j + 2, i, b] = NUFSHT._parity(R, m + s) * acc
        end
    end
end

# Row `r` of the layout holds `m' = r − 1 + mp0`: `mp0 = −lmax` on the full layout, `0` on a folded one.
@kernel function _ring_modes_kern!(G, @Const(C), N, mp0)
    r, i, b = @index(Global, NTuple)
    R = real(eltype(C))
    @inbounds G[r, i, b] = eltype(G)(C[mod(r - 1 + mp0, N) + 1, i, b] / R(N))
end

@kernel function _ring_scatter_kern!(C, @Const(Ĝ), N, mp0)
    r, i, b = @index(Global, NTuple)
    @inbounds C[mod(r - 1 + mp0, N) + 1, i, b] = Ĝ[r, i, b]
end

# Rows `1 … nring` of `C` become the transpose of the reflection and of `1/N`; a row reads only rows
# past `nring`, which no workitem writes.
@kernel function _ring_half_kern!(C, lmax, s, N, nring)
    j, i, b = @index(Global, NTuple)
    R = real(eltype(C))
    @inbounds if j == 1 || j == nring
        C[j, i, b] = C[j, i, b] / R(N)
    else
        C[j, i, b] = (C[j, i, b] + NUFSHT._parity(R, i - lmax - 1 + s) * C[N - j + 2, i, b]) / R(N)
    end
end

# Each degree's `Σ_rings d^ℓ h` over a block of `WG` rings, into `P[ℓ+1, block, i, b]`. State a
# workitem carries across a barrier is in `@private` arrays, and everything else is formed again after
# each barrier.
@kernel function _ring_adjoint_kern!(P, @Const(C), @Const(u), @Const(pos), @Const(lc), @Const(ls),
                                     @Const(lC), @Const(p1), @Const(p2), @Const(sg), @Const(A), @Const(Cp),
                                     @Const(Cm), @Const(Γ), lmax, s, nring, ::Val{WG}) where {WG}
    t = @index(Local, Linear)
    blk, i, b = @index(Group, NTuple)
    st = @private eltype(u) (4,)            # y, e, Re h, Im h
    kk = @private Int32 (1,)
    sr = @localmem eltype(u) (WG,)
    si = @localmem eltype(u) (WG,)
    @inbounds begin
        j = (blk - 1) * WG + t
        if j <= nring
            y0, k0 = NUFSHT._ring_start(eltype(u), lC[i], p1[i], p2[i], sg[i], lc[j], ls[j])
            st[1] = y0; st[2] = y0; kk[1] = k0
            st[3] = real(C[j, i, b]); st[4] = imag(C[j, i, b])
        else
            st[1] = zero(eltype(u)); st[2] = zero(eltype(u)); kk[1] = Int32(0)
            st[3] = zero(eltype(u)); st[4] = zero(eltype(u))
        end
    end
    for ℓ in 0:lmax
        @inbounds begin
            R = eltype(u)
            m = i - lmax - 1
            up, dn = NUFSHT._ring_bounds(R)
            on = ℓ >= max(abs(m), abs(s))
            v = ifelse(on, st[1] * NUFSHT._ring_weight(kk[1], dn), zero(R))
            sr[t] = v * st[3]; si[t] = v * st[4]
            if on
                jj = min((blk - 1) * WG + t, nring)
                k = kk[1]
                yn, en = _ring_device_step(A, Cp, Cm, Γ, ℓ, m, pos[jj], u[jj], st[1], st[2])
                big = (k < Int32(0)) & (abs(yn) > up)
                sc = ifelse(big, dn, one(R))
                st[2] = en * sc; st[1] = yn * sc; kk[1] = k + Int32(big)
            end
        end
        @synchronize
        for lev in 1:trailing_zeros(WG)
            @inbounds begin
                h = WG >> lev
                if t <= h
                    sr[t] += sr[t + h]; si[t] += si[t + h]
                end
            end
            @synchronize
        end
        @inbounds if t == 1
            P[ℓ + 1, blk, i, b] = complex(sr[1], si[1])
        end
        @synchronize
    end
end

@kernel function _ring_blocks_kern!(sf, @Const(P), nblk)
    r, i, b = @index(Global, NTuple)
    R = real(eltype(P))
    acc = zero(eltype(P))
    @inbounds for q in 1:nblk
        acc += P[r, q, i, b]
    end
    @inbounds sf[r, i, b] = eltype(sf)(acc * NUFSHT._ring_norm(R, r - 1))
end

@inline _row_origin(G, lmax) = size(G, 1) == 2lmax + 1 ? -lmax : 0

function NUFSHT._assemble_G_impl!(G::GPUArraysCore.AbstractGPUArray, sf, plan::NUFSHT.SpinNUSHTplan,
                                  eng::NUFSHT._RingEngine)
    lmax = plan.lmax; w = eng.work
    backend = KernelAbstractions.get_backend(G)
    _ring_forward_kern!(backend)(w.C, _as3(sf, lmax), eng.u, eng.pos, eng.lc, eng.ls, eng.lC, eng.p1, eng.p2,
                                 eng.sg, w.A, w.Cp, w.Cm, w.Γ, lmax, plan.s, eng.N, eng.nring;
                                 ndrange = (eng.nring, 2lmax + 1, plan.B))
    w.bwd * w.C
    _ring_modes_kern!(backend)(G, w.C, eng.N, _row_origin(G, lmax); ndrange = size(G))
    _sync(backend)
    return G
end

function NUFSHT._assemble_G_adjoint_impl!(sf::GPUArraysCore.AbstractGPUArray, Ĝ, plan::NUFSHT.SpinNUSHTplan,
                                          eng::NUFSHT._RingEngine)
    lmax = plan.lmax; w = eng.work
    backend = KernelAbstractions.get_backend(sf)
    fill!(w.C, zero(eltype(w.C)))
    _ring_scatter_kern!(backend)(w.C, Ĝ, eng.N, _row_origin(Ĝ, lmax); ndrange = size(Ĝ))
    w.fwd * w.C
    _ring_half_kern!(backend)(w.C, lmax, plan.s, eng.N, eng.nring; ndrange = (eng.nring, 2lmax + 1, plan.B))
    nblk = size(w.P, 2)
    _ring_adjoint_kern!(backend, (w.wg, 1, 1))(w.P, w.C, eng.u, eng.pos, eng.lc, eng.ls, eng.lC, eng.p1,
                                               eng.p2, eng.sg, w.A, w.Cp, w.Cm, w.Γ, lmax, plan.s, eng.nring,
                                               Val(w.wg); ndrange = (nblk * w.wg, 2lmax + 1, plan.B))
    _ring_blocks_kern!(backend)(_as3(sf, lmax), w.P, nblk; ndrange = (lmax + 1, 2lmax + 1, plan.B))
    _sync(backend)
    return sf
end

# ── Device solver steps ─────────────────────────────────────────────────────────
# The column-wise steps of `nusht_solve!` and `nusht_solve_spin!` that are NUFSHT's own, as broadcasts
# over the first `n` columns; FlowTransformBindings supplies the generic column updates and norms.
# `_cols` flattens a batch buffer to (stride, columns), so one method serves point space and
# coefficient space.
@inline _cols(A, n) = view(reshape(A, :, size(A, ndims(A))), :, 1:n)

# ── Device pack/unpack between the packed solver vectors and the plan's full layout ──────────────
# The scalar solver carries its vectors packed to the `l ≤ lmax` slots, so crossing to and from the
# plan's `(Nθ, Nφ, B)` layout is a gather/scatter by index vector, here a broadcast over reshaped views.
function NUFSHT._pack_coeffs!(dst::GPUArraysCore.AbstractGPUArray, src, idx, srclen::Integer,
                              n::Integer)
    _cols(dst, n) .= view(reshape(src, srclen, :), idx, 1:n)
    return dst
end

function NUFSHT._unpack_coeffs!(dst::GPUArraysCore.AbstractGPUArray, src, idx, dstlen::Integer,
                                n::Integer)
    fill!(dst, zero(eltype(dst)))
    view(reshape(dst, dstlen, :), idx, 1:n) .= _cols(src, n)
    return dst
end

function NUFSHT._col_pbp_pack!(v::GPUArraysCore.AbstractGPUArray, wfull, β, idx,
                               fulllen::Integer, n::Integer)
    _cols(v, n) .= view(reshape(wfull, fulllen, :), idx, 1:n) .+
                   reshape(view(β, 1:n), 1, :) .* _cols(v, n)
    return v
end

function NUFSHT._write_solution!(C::GPUArraysCore.AbstractGPUArray, x, idx, full::Integer,
                                 slot::Integer, dstcol::Integer)
    col = view(reshape(C, full, :), :, dstcol)
    fill!(col, zero(eltype(C)))
    view(col, idx) .= view(reshape(x, length(idx), :), :, slot)
    return C
end

# ── Device Hermitian pack/unpack (the real-field spin solve) ─────────────────────
# A real spin-0 field's solver vectors are packed to `(lmax+1)²` real values per column. Each output
# entry is a gather from index arithmetic alone, so each is one workitem calling the same
# `_herm_entry`/`_pack_entry` the host loops do. The arrays are reshaped to three axes first, so a
# `B = 1` coefficient matrix takes the same kernel.
@kernel function _unpack_herm_kern!(sf, @Const(p), lmax, K, pcol0, scol0)
    i, j, b = @index(Global, NTuple)
    @inbounds sf[i, j, b + scol0] =
        NUFSHT._herm_entry(eltype(sf), p, i - 1, j - 1 - lmax, (b + pcol0 - 1) * K)
end

@kernel function _pack_herm_kern!(p, @Const(g), lmax, K, β)
    t, b = @index(Global, NTuple)
    i = (b - 1) * K + t
    @inbounds p[i] = NUFSHT._fold_packed(NUFSHT._pack_entry(g, t, lmax, b), β, p, i, b)
end

@inline _as3(A, lmax) = reshape(A, lmax + 1, 2lmax + 1, :)

function NUFSHT._unpack_herm!(sf::GPUArraysCore.AbstractGPUArray, p, lmax::Integer, B::Integer)
    backend = KernelAbstractions.get_backend(sf)
    _unpack_herm_kern!(backend)(_as3(sf, lmax), p, Int(lmax), NUFSHT._herm_len(lmax), 0, 0;
                                ndrange = (lmax + 1, 2lmax + 1, B))
    _sync(backend)
    return sf
end

function NUFSHT._pack_herm!(p::GPUArraysCore.AbstractGPUArray, g, lmax::Integer, B::Integer,
                            β = nothing)
    backend = KernelAbstractions.get_backend(p)
    K = NUFSHT._herm_len(lmax)
    _pack_herm_kern!(backend)(p, _as3(g, lmax), Int(lmax), K, β; ndrange = (K, B))
    _sync(backend)
    return p
end

function NUFSHT._write_solution!(C::GPUArraysCore.AbstractGPUArray, x, plan::NUFSHT.SpinNUSHTplan,
                                 slot::Integer, dstcol::Integer,
                                 ::Union{NUFSHT.FoldedComplex,NUFSHT.FoldedReal})
    lmax = plan.lmax
    backend = KernelAbstractions.get_backend(C)
    _unpack_herm_kern!(backend)(_as3(C, lmax), x, lmax, NUFSHT._herm_len(lmax), Int(slot) - 1,
                                Int(dstcol) - 1; ndrange = (lmax + 1, 2lmax + 1, 1))
    _sync(backend)
    return C
end

# ── Device real↔complex field copy (scalar type-2/type-1 bracket) ───────────────
# `f` may be `(M,)` or `(M,B)`; `fbuf` is `(M,B)` and equal length — a `reshape`d broadcast handles
# either shape (the `src` versions are CPU scalar loops). Dispatched on the plan buffer `fbuf`.
function NUFSHT._copy_real!(f, fbuf::GPUArraysCore.AbstractGPUArray)       # f = real(fbuf)
    f .= real.(reshape(fbuf, size(f)))
    return f
end
function NUFSHT._copy_field!(fbuf::GPUArraysCore.AbstractGPUArray, f)      # fbuf = f (real→complex)
    fbuf .= reshape(f, size(fbuf))
    return fbuf
end

# The accumulating pair, over the first `n` columns: the solver's `u ← A v − α u` adds the synthesis
# onto the scaled `u`, dispatched on the plan buffer `fbuf` as the copies above are.
function NUFSHT._add_real!(f, fbuf::GPUArraysCore.AbstractGPUArray,        # f[:,k] += Re fbuf[:,k]
                           n::Integer = size(f, ndims(f)))
    _cols(f, n) .+= real.(_cols(fbuf, n))
    return f
end
function NUFSHT._add_field!(f, fbuf::GPUArraysCore.AbstractGPUArray,       # f[:,k] += fbuf[:,k]
                            n::Integer = size(f, ndims(f)))
    _cols(f, n) .+= _cols(fbuf, n)
    return f
end

# ── Device analysis: weighted strengths, Hermitian projection, scalar ↔ spin-0 ──────────────────
# `fbuf[j, b] = w[j] f[j, b]` by `NUFSHT._weighted_value`, the weights on the plan's device.
function NUFSHT._copy_weighted!(fbuf::GPUArraysCore.AbstractGPUArray, f, w)
    M = length(w)
    reshape(fbuf, M, :) .= NUFSHT._weighted_value.(reshape(w, M, 1), reshape(f, M, :))
    return fbuf
end

# One workitem per pair `(ℓ, ±m)`, `m ≥ 0`, of a column, each calling the per-pair function the host
# loop calls.
@kernel function _hermitian_part_kern!(sf, lmax)
    i, j, b = @index(Global, NTuple)
    if j <= i
        NUFSHT._hermitian_pair!(sf, i - 1, j - 1, lmax, b)
    end
end

@kernel function _scalar_to_spin0_kern!(sf, @Const(C), lmax)
    i, j, b = @index(Global, NTuple)
    if j <= i
        NUFSHT._scalar_pair_to_spin0!(sf, C, i - 1, j - 1, lmax, b)
    end
end

@kernel function _spin0_to_scalar_kern!(C, @Const(sf), lmax)
    i, j, b = @index(Global, NTuple)
    if j <= i
        NUFSHT._spin0_pair_to_scalar!(C, sf, i - 1, j - 1, lmax, b)
    end
end

function NUFSHT._hermitian_part!(sf::GPUArraysCore.AbstractGPUArray, plan::NUFSHT.SpinNUSHTplan,
                                 ::Union{NUFSHT.FoldedComplex,NUFSHT.FoldedReal})
    lmax = plan.lmax
    backend = KernelAbstractions.get_backend(sf)
    _hermitian_part_kern!(backend)(_as3(sf, lmax), Int(lmax); ndrange = (lmax + 1, lmax + 1, plan.B))
    _sync(backend)
    return sf
end

function NUFSHT._scalar_to_spin0_pairs!(sf::GPUArraysCore.AbstractGPUArray, C, lmax::Int, B::Int)
    backend = KernelAbstractions.get_backend(sf)
    _scalar_to_spin0_kern!(backend)(_as3(sf, lmax), _as3(C, lmax), lmax; ndrange = (lmax + 1, lmax + 1, B))
    _sync(backend)
    return sf
end

function NUFSHT._spin0_to_scalar_pairs!(C::GPUArraysCore.AbstractGPUArray, sf, lmax::Int, B::Int)
    backend = KernelAbstractions.get_backend(C)
    _spin0_to_scalar_kern!(backend)(_as3(C, lmax), _as3(sf, lmax), lmax; ndrange = (lmax + 1, lmax + 1, B))
    _sync(backend)
    return C
end

# ── Device spectral filter (× H(ℓ)) ─────────────────────────────────────────────
# `C` is `(lmax+1, 2lmax+1, B)`, multiplied by the transfer matrix broadcast across the batch. The
# matrix depends only on the filter, `lmax`, the element type and the array type, so each is built and
# moved to the device once and kept here.
const _TRANSFER_CACHE = Dict{Tuple{Any,Int,DataType,DataType},Any}()
const _TRANSFER_LOCK = ReentrantLock()

function _device_transfer(C, filter, lmax)
    RT = real(eltype(C))
    key = (filter, lmax, RT, typeof(C))
    return Base.lock(_TRANSFER_LOCK) do
        get!(_TRANSFER_CACHE, key) do
            H = NUFSHT._transfer_matrix(filter, lmax, RT)
            reshape(NUFSHT._to_like(C, H), size(H, 1), size(H, 2), 1)
        end
    end
end

function NUFSHT.apply_transfer!(C::GPUArraysCore.AbstractGPUArray, filter, lmax)
    C .*= _device_transfer(C, filter, lmax)
    return C
end

end # module NUFSHTKernelAbstractionsExt
