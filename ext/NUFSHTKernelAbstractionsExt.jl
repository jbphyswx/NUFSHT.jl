"""
    NUFSHTKernelAbstractionsExt

Device-generic methods for the array-indexed steps of the transform, as KernelAbstractions `@kernel`s
dispatched on `GPUArraysCore.AbstractGPUArray`. They run on any KernelAbstractions backend (CUDA, ROCm,
Metal, oneAPI, the CPU backend) and take precedence over the host methods for a device array.

Device nodes build a [`NUFSHT.RingNUSHTplan`](@ref) for a scalar field, so every plan on a device runs
the ring S-step. The kernels cover that step and its adjoint, the real-field fit's packing, the weighted
strengths and Hermitian projection of an analysis, the scalar ↔ spin-0 conversions, the spectral filter
and the direct-sum NUFFT. A fast NUFFT goes through the `_nufft_*` functions to FlowTransformBindings,
which builds the plan on the nodes' device.

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

# A scalar plan over device nodes runs on the ring S-step, whose every step has a kernel here.
NUFSHT._build_plan(::Type{FE}, θ::GPUArraysCore.AbstractGPUArray, φ, lmax; kwargs...) where {FE<:Number} =
    NUFSHT._build_ring_plan(FE, θ, φ, lmax; kwargs...)

# Kernel launches are asynchronous on device backends (CUDA/ROCm/…) and run in order on the task's queue,
# and a copy to the host waits for the queue, so no launch here synchronizes. cuFINUFFT issues on the
# CUDA default stream, which CUDA orders against CUDA.jl's task streams (created blocking).

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
        α, _, γ, c₊, c₋ = NUFSHT.FTB.wigner_d_coeffs(promote_type(R, Float64), ℓ, μ, n)
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
                              dev(t.x), dev(t.xl), dev(t.lc), dev(t.ls), dev(t.lC), dev(t.p1), dev(t.p2),
                              dev(t.sg), work)
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
    σα, c, σγ, σ = NUFSHT.FTB.wigner_d_signed(pos, α, c₊, c₋, γ)
    return NUFSHT.FTB.wigner_d_step(σα, c, σγ, σ, u, y, e)
end

@kernel function _ring_forward_kern!(C, @Const(sf), @Const(u), @Const(pos), @Const(lc), @Const(ls),
                                     @Const(lC), @Const(p1), @Const(p2), @Const(sg), @Const(A), @Const(Cp),
                                     @Const(Cm), @Const(Γ), lmax, s, N, nring)
    j, i, b = @index(Global, NTuple)
    R = eltype(u)
    m = i - lmax - 1
    up, dn = NUFSHT.FTB.wigner_d_bounds(R)
    @inbounds begin
        y, k = NUFSHT._ring_start(R, lC[i], p1[i], p2[i], sg[i], lc[j], ls[j])
        uj = u[j]; pj = pos[j]
        e = y
        acc = zero(Complex{R})
        for ℓ in max(abs(m), abs(s)):lmax
            c = Complex{R}(sf[ℓ + 1, i, b]) * NUFSHT._ring_norm(R, ℓ)
            acc += c * (y * NUFSHT._ring_weight(k, dn))
            yn, en = _ring_device_step(A, Cp, Cm, Γ, ℓ, m, pj, uj, y, e)
            y, e, k = NUFSHT.FTB.wigner_d_rescale(yn, en, k, up, dn)
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
            up, dn = NUFSHT.FTB.wigner_d_bounds(R)
            on = ℓ >= max(abs(m), abs(s))
            v = ifelse(on, st[1] * NUFSHT._ring_weight(kk[1], dn), zero(R))
            sr[t] = v * st[3]; si[t] = v * st[4]
            if on
                jj = min((blk - 1) * WG + t, nring)
                yn, en = _ring_device_step(A, Cp, Cm, Γ, ℓ, m, pos[jj], u[jj], st[1], st[2])
                y, e, k = NUFSHT.FTB.wigner_d_rescale(yn, en, kk[1], up, dn)
                st[1] = y; st[2] = e; kk[1] = k
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
    return sf
end

# The first `n` columns of a batch buffer flattened to (stride, columns), so one method serves point
# space and coefficient space.
@inline _cols(A, n) = view(reshape(A, :, size(A, ndims(A))), :, 1:n)

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
    return sf
end

function NUFSHT._pack_herm!(p::GPUArraysCore.AbstractGPUArray, g, lmax::Integer, B::Integer,
                            β = nothing)
    backend = KernelAbstractions.get_backend(p)
    K = NUFSHT._herm_len(lmax)
    _pack_herm_kern!(backend)(p, _as3(g, lmax), Int(lmax), K, β; ndrange = (K, B))
    return p
end

function NUFSHT._write_solution!(C::GPUArraysCore.AbstractGPUArray, x, plan::NUFSHT.SpinNUSHTplan,
                                 slot::Integer, dstcol::Integer,
                                 ::Union{NUFSHT.FoldedComplex,NUFSHT.FoldedReal})
    lmax = plan.lmax
    backend = KernelAbstractions.get_backend(C)
    _unpack_herm_kern!(backend)(_as3(C, lmax), x, lmax, NUFSHT._herm_len(lmax), Int(slot) - 1,
                                Int(dstcol) - 1; ndrange = (lmax + 1, 2lmax + 1, 1))
    return C
end

# ── Device real↔complex field copy ──────────────────────────────────────────────
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
    return sf
end

function NUFSHT._scalar_to_spin0_pairs!(sf::GPUArraysCore.AbstractGPUArray, C, lmax::Int, B::Int)
    backend = KernelAbstractions.get_backend(sf)
    _scalar_to_spin0_kern!(backend)(_as3(sf, lmax), _as3(C, lmax), lmax; ndrange = (lmax + 1, lmax + 1, B))
    return sf
end

function NUFSHT._spin0_to_scalar_pairs!(C::GPUArraysCore.AbstractGPUArray, sf, lmax::Int, B::Int)
    backend = KernelAbstractions.get_backend(C)
    _spin0_to_scalar_kern!(backend)(_as3(C, lmax), _as3(sf, lmax), lmax; ndrange = (lmax + 1, lmax + 1, B))
    return C
end

# ── Device direct-sum NUFFT ─────────────────────────────────────────────────────
# Both directions are gathers: one workitem per output value, summing over the other side, each phase
# `e^{is(k₁x + k₂y)}` formed exactly, as the host direct sum's tables hold it.
@inline _ds_freq(::Type{R}, i::Int, n::Int) where {R} =
    R(NUFSHT.FTB.mode_frequency(NUFSHT.FTB.CenteredModes(), i, n))

@kernel function _ds_type2_kern!(vals, @Const(modes), @Const(x), @Const(y), n1, n2, s)
    j, b = @index(Global, NTuple)
    R = eltype(x)
    acc = zero(eltype(vals))
    @inbounds begin
        xj = x[j]; yj = y[j]
        for i2 in 1:n2
            col = zero(eltype(vals))
            for i1 in 1:n1
                col += modes[i1, i2, b] * cis(s * _ds_freq(R, i1, n1) * xj)
            end
            acc += col * cis(s * _ds_freq(R, i2, n2) * yj)
        end
        vals[j, b] = acc
    end
end

@kernel function _ds_type1_kern!(modes, @Const(vals), @Const(x), @Const(y), n1, n2, s, M)
    i1, i2, b = @index(Global, NTuple)
    R = eltype(x)
    k1 = s * _ds_freq(R, i1, n1); k2 = s * _ds_freq(R, i2, n2)
    acc = zero(eltype(modes))
    @inbounds for j in 1:M
        acc += vals[j, b] * cis(k1 * x[j] + k2 * y[j])
    end
    @inbounds modes[i1, i2, b] = acc
end

# The struct's bounds are restated, so the signature is a subtype of the host method's and dispatch
# prefers it.
function NUFSHT._nufft_exec!(p::NUFSHT.DirectSumNUFFTPlan{T,<:GPUArraysCore.AbstractGPUArray{T,1}}, input,
                             output) where {T<:AbstractFloat}
    M = length(p.x)
    backend = KernelAbstractions.get_backend(p.x)
    if p.type == 2
        modes = reshape(input, p.n1, p.n2, :)
        vals = reshape(output, M, :)
        _ds_type2_kern!(backend)(vals, modes, p.x, p.y, p.n1, p.n2, T(p.iflag); ndrange = size(vals))
    else
        vals = reshape(input, M, :)
        modes = reshape(output, p.n1, p.n2, :)
        _ds_type1_kern!(backend)(modes, vals, p.x, p.y, p.n1, p.n2, T(p.iflag), M; ndrange = size(modes))
    end
    return output
end

# ── Device spectral filter (× H(ℓ)) ─────────────────────────────────────────────
# `H(ℓ)` for the `2lmax + 1` degrees the array holds, evaluated on the host in `C`'s real type, and one
# workitem per slot scaling it by its degree's value.
@kernel function _transfer_kern!(C, @Const(h))
    i, j, b = @index(Global, NTuple)
    @inbounds C[i, j, b] *= h[NUFSHT._slot_degree(i, j) + 1]
end

function NUFSHT.apply_transfer!(C::GPUArraysCore.AbstractGPUArray, filter, lmax)
    RT = real(eltype(C))
    h = NUFSHT._to_like(C, [RT(NUFSHT.kernel_transfer(filter, ℓ)) for ℓ in 0:(2lmax)])
    C3 = _as3(C, lmax)
    _transfer_kern!(KernelAbstractions.get_backend(C))(C3, h; ndrange = size(C3))
    return C
end

end # module NUFSHTKernelAbstractionsExt
