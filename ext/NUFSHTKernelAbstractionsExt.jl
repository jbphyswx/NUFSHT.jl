"""
    NUFSHTKernelAbstractionsExt

Device-generic methods for the array-indexed steps of the transform, as KernelAbstractions `@kernel`s
dispatched on `GPUArraysCore.AbstractGPUArray`. They run on any KernelAbstractions backend (CUDA, ROCm,
Metal, oneAPI, and the `JLArray` / KA-CPU reference backends) and take precedence over the host methods
for a device array.

The kernels cover the scalar mode assembly and its adjoint, the spin Wigner recurrence and contraction,
the solver's per-column steps, the weighted strengths and Hermitian projection of an analysis, the
scalar ↔ spin-0 conversions and the spectral filter. The NUFFT goes through the `_nufft_*` functions to
FlowTransformBindings, which builds the plan on the nodes' device.

Loaded by `using KernelAbstractions` (with `GPUArraysCore`).
"""
module NUFSHTKernelAbstractionsExt

using NUFSHT: NUFSHT
# `@kernel`, `@index` and `@Const` are imported by bare name: the `@kernel` macro recognizes the DSL
# keywords only unqualified, and a qualified `KA.@index` is not rewritten and miscompiles.
using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const
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

# ── Device spin S-engine: the blocked plane build and contraction of `src/Spin.jl` ──
# A block of `K` degrees is three launches: the quarter planes from the stored `m = ℓ` columns, one
# workitem per (row, degree); their lower triangles by transpose; and the contraction, one workitem per
# output entry, summing the block's degrees in ascending order as the host does. Launches on one
# backend run in order, so only a host read needs a sync.

# Quarter planes per block on a device.
const _DEVICE_PLANES = 32
NUFSHT._plane_count(::GPUArraysCore.AbstractGPUArray, ::Integer) = _DEVICE_PLANES

# Row `n` of plane `k` (degree `ℓ0 + k − 1`): its column `m = ℓ`, then the downward recurrence in `m` on
# `n ≤ m`. A row depends on itself alone, and consecutive workitems touch consecutive addresses.
@kernel function _quarter_kern!(P, @Const(bnd), ℓ0)
    i, k = @index(Global, NTuple)
    n = i - 1; ℓ = ℓ0 + k - 1
    RT = eltype(P)
    if n <= ℓ
        @inbounds begin
            P[n + 1, ℓ + 1, k] = bnd[n + 1, ℓ + 1]
            for m in (ℓ - 1):-1:n
                c1 = RT(2) / sqrt(RT(ℓ - m) * RT(ℓ + m + 1))
                if m == ℓ - 1
                    P[n + 1, m + 1, k] = c1 * RT(n) * P[n + 1, m + 2, k]
                else
                    s2 = sqrt(RT(ℓ - m - 1) * RT(ℓ + m + 2) / (RT(ℓ - m) * RT(ℓ + m + 1)))
                    P[n + 1, m + 1, k] = c1 * RT(n) * P[n + 1, m + 2, k] - s2 * P[n + 1, m + 3, k]
                end
            end
        end
    end
end

# `Δ_{n,m} = (−1)^{m+n} Δ_{m,n}` below the diagonal; it reads only above it.
@kernel function _quarter_transpose_kern!(P, ℓ0)
    i, j, k = @index(Global, NTuple)
    n = i - 1; m = j - 1
    if m < n <= ℓ0 + k - 1
        @inbounds P[n + 1, m + 1, k] = NUFSHT._parity(eltype(P), m + n) * P[m + 1, n + 1, k]
    end
end

# `G_{m'm} += Σ_ℓ i^{m+s} N_ℓ sf_{ℓm} Δ^ℓ_{m'm} Δ^ℓ_{m',−s}` over the block, one workitem per entry.
# Row `i` holds `m' = i − 1 + mp0`: `mp0 = −lmax` on the full layout, `0` on a folded one.
@kernel function _contract_block_kern!(G, @Const(P), @Const(sf), ℓ0, ℓ1, s, lmax, mp0)
    i, j, b = @index(Global, NTuple)
    mp = i - 1 + mp0; m = j - 1 - lmax
    CT = eltype(G); RT = real(CT)
    @inbounds begin
        acc = G[i, j, b]
        ph0 = NUFSHT._im_pow(CT, -(m + s))
        for ℓ in max(ℓ0, abs(mp), abs(m)):ℓ1
            ph = ph0 * sqrt(RT(2ℓ + 1) / (4 * RT(π))) * sf[ℓ + 1, j, b]
            acc += ph * NUFSHT._qprod(P, ℓ - ℓ0 + 1, RT, ℓ, mp, m, s)
        end
        G[i, j, b] = acc
    end
end

# `sf_{ℓm} = conj(i^{m+s}) N_ℓ Σ_{m'} Δ^ℓ_{m'm} Δ^ℓ_{m',−s} Ĝ_{m'm}`, one workitem per (order, column,
# degree of the block).
@kernel function _contract_block_adj_kern!(sf, @Const(P), @Const(Ĝ), ℓ0, s, lmax, mp0)
    j, b, k = @index(Global, NTuple)
    m = j - 1 - lmax; ℓ = ℓ0 + k - 1
    CT = eltype(sf); RT = real(CT)
    if abs(m) <= ℓ
        @inbounds begin
            acc = zero(CT)
            for mp in max(mp0, -ℓ):ℓ
                acc += NUFSHT._qprod(P, k, RT, ℓ, mp, m, s) * Ĝ[mp - mp0 + 1, j, b]
            end
            sf[ℓ + 1, j, b] = conj(NUFSHT._im_pow(CT, -(m + s))) * sqrt(RT(2ℓ + 1) / (4 * RT(π))) * acc
        end
    end
end

@inline _row_origin(G, lmax) = size(G, 1) == 2lmax + 1 ? -lmax : 0

function _device_planes!(backend, eng::NUFSHT._WignerPlanes, ℓ0, ℓ1)
    nb = ℓ1 - ℓ0 + 1
    _quarter_kern!(backend)(eng.planes, eng.bnd, ℓ0; ndrange = (ℓ1 + 1, nb))
    _quarter_transpose_kern!(backend)(eng.planes, ℓ0; ndrange = (ℓ1 + 1, ℓ1 + 1, nb))
    return nothing
end

function NUFSHT._assemble_G_impl!(G::GPUArraysCore.AbstractGPUArray, sf,
                                  plan::NUFSHT.SpinNUSHTplan{T}, eng::NUFSHT._WignerPlanes) where {T}
    lmax = plan.lmax; K = size(eng.planes, 3)
    backend = KernelAbstractions.get_backend(G)
    fill!(G, zero(Complex{T}))
    mp0 = _row_origin(G, lmax)
    ℓ0 = abs(plan.s)
    while ℓ0 <= lmax
        ℓ1 = min(lmax, ℓ0 + K - 1)
        _device_planes!(backend, eng, ℓ0, ℓ1)
        _contract_block_kern!(backend)(G, eng.planes, sf, ℓ0, ℓ1, plan.s, lmax, mp0; ndrange = size(G))
        ℓ0 = ℓ1 + 1
    end
    _sync(backend)
    return G
end

function NUFSHT._assemble_G_adjoint_impl!(sf::GPUArraysCore.AbstractGPUArray, Ĝ,
                                          plan::NUFSHT.SpinNUSHTplan{T}, eng::NUFSHT._WignerPlanes) where {T}
    lmax = plan.lmax; K = size(eng.planes, 3)
    backend = KernelAbstractions.get_backend(sf)
    fill!(sf, zero(Complex{T}))
    mp0 = _row_origin(Ĝ, lmax)
    ℓ0 = abs(plan.s)
    while ℓ0 <= lmax
        ℓ1 = min(lmax, ℓ0 + K - 1)
        _device_planes!(backend, eng, ℓ0, ℓ1)
        _contract_block_adj_kern!(backend)(sf, eng.planes, Ĝ, ℓ0, plan.s, lmax, mp0;
                                           ndrange = (2lmax + 1, plan.B, ℓ1 - ℓ0 + 1))
        ℓ0 = ℓ1 + 1
    end
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

# A precomputed WignerTable is a host array read elementwise, which a device contraction cannot read,
# so a device plan refuses one.
_no_device_table() = throw(ArgumentError(
    "a device spin plan cannot use a precomputed WignerTable; build it with wigner_table = nothing " *
    "so the Δ planes are generated on device by the recurrence kernels."))
NUFSHT._assemble_G_impl!(::GPUArraysCore.AbstractGPUArray, sf, plan::NUFSHT.SpinNUSHTplan,
                         ::NUFSHT.WignerTable) = _no_device_table()
NUFSHT._assemble_G_adjoint_impl!(::GPUArraysCore.AbstractGPUArray, Ĝ, plan::NUFSHT.SpinNUSHTplan,
                                 ::NUFSHT.WignerTable) = _no_device_table()

end # module NUFSHTKernelAbstractionsExt
