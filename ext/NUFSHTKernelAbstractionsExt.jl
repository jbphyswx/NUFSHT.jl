"""
    NUFSHTKernelAbstractionsExt

Device-generic GPU methods for the array-indexed steps of the transform, written **once** as
KernelAbstractions `@kernel`s and dispatched on `GPUArraysCore.AbstractGPUArray` — so they run on any
KA backend (CUDA/ROCm/Metal/oneAPI, and the reference `JLArray`/KA-CPU backend used for testing)
without per-vendor code. The plain CPU `Array` methods in `src` are untouched; a GPU-backed array is
`<:AbstractGPUArray` so these more-specific methods win for it.

Every array-indexed step needs a kernel — the scalar mode assembly and its adjoint, the spin Wigner
recurrence and contraction, the solver's per-column primitives and the spectral filter. The NUFFT is
not among them: it is dispatched through the `_nufft_*` seam (cuFINUFFT via the CUDA extension).

Loaded by `using KernelAbstractions` (with `GPUArraysCore`).
"""
module NUFSHTKernelAbstractionsExt

using NUFSHT: NUFSHT
# The KernelAbstractions kernel DSL macros (`@kernel`, `@index`, `@Const`) are recognized by the
# `@kernel` macro *syntactically* and only work unqualified — a module-qualified `KA.@index` is not
# rewritten and miscompiles. So they are the one necessary exception to the qualified-import
# convention; all non-DSL KA entry points stay qualified (`KernelAbstractions.get_backend`, etc.).
using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const
using GPUArraysCore: GPUArraysCore

# GPU kernel launches are asynchronous on real device backends (CUDA/ROCm/…), so results must be
# synchronized before the host reads them; JLArrays' reference backend is synchronous and defines no
# `synchronize`, so guard the call. (For chained on-device ops this is conservative; it can be relaxed
# once validated on real hardware.)
@inline function _sync(backend)
    applicable(KernelAbstractions.synchronize, backend) && KernelAbstractions.synchronize(backend)
    return nothing
end

# ── Device scalar mode assembly (F) ─────────────────────────────────────────────
# Both directions of `src/Modes.jl` are per-element gathers, so each is one workitem per output
# element with no atomics and no fill pass, calling the very same `_mode_entry`/`_mode_adjoint_entry`
# the host loops do — the arithmetic exists once. The plan buffers are `(…, …, B)`, so `ndrange` is
# the whole output including the batch axis.

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

# ── Device spin S-engine: on-the-fly Trapani–Navaza recurrence + G-contraction ──
# The spin transform's `_assemble_G!` (Legendre step + bivariate-Fourier assembly) as KA kernels, so
# the whole spin path runs on device (recurrence buffers `similar` to the device nodes; NUFFT via the
# cuFINUFFT seam). Same math as the CPU `src/Spin.jl` methods — validated bit-for-bit on JLArray. The
# degree loop is sequential (ℓ depends on ℓ-1); per degree, Stage A+B is one thread per column, the
# symmetry fills and the contraction are elementwise.

# Trapani–Navaza degree step, one workitem per column n∈0:ℓ: Stage A (row m=ℓ) then the downward
# Stage B m-sweep for that column (self-contained — no cross-column dependency).
# `dl` is n-major (`dl[n+off, m+off] = d^ℓ_{m,n}`), matching `src/Spin.jl`: consecutive workitems then
# read and write consecutive addresses, so these accesses coalesce.
@kernel function _recur_AB_kern!(dl, @Const(dlp), ℓ, off)
    c = @index(Global)
    n = c - 1
    RT = eltype(dl)
    @inbounds begin
        if n == 0
            dl[off, ℓ + off] = -sqrt(RT(2ℓ - 1) / RT(2ℓ)) * dlp[off, (ℓ - 1) + off]
        else
            dl[n + off, ℓ + off] =
                sqrt(RT(ℓ) * RT(2ℓ - 1) / (RT(2) * RT(ℓ + n) * RT(ℓ + n - 1))) * dlp[(n - 1) + off, (ℓ - 1) + off]
        end
        if ℓ - 1 >= n
            m = ℓ - 1
            dl[n + off, m + off] = RT(2n) / sqrt(RT(ℓ - m) * RT(ℓ + m + 1)) * dl[n + off, (m + 1) + off]
        end
        for m in (ℓ - 2):-1:n
            s1 = sqrt(RT(ℓ - m) * RT(ℓ + m + 1))
            s2 = sqrt(RT(ℓ - m - 1) * RT(ℓ + m + 2) / (RT(ℓ - m) * RT(ℓ + m + 1)))
            dl[n + off, m + off] = RT(2n) / s1 * dl[n + off, (m + 1) + off] - s2 * dl[n + off, (m + 2) + off]
        end
    end
end

@kernel function _recur_seed_kern!(dl, off)          # ℓ=0 seed d⁰₀₀ = 1
    @inbounds dl[off, off] = one(eltype(dl))
end

@kernel function _recur_S1_kern!(dl, ℓ, off)         # transpose: eighth → quarter (n>m)
    i, j = @index(Global, NTuple)
    m = i - 1; n = j - 1
    if n > m
        RT = eltype(dl)
        @inbounds dl[n + off, m + off] = ifelse(iseven(m + n), one(RT), -one(RT)) * dl[m + off, n + off]
    end
end

@kernel function _recur_S2_kern!(dl, ℓ, off)         # m → −m: quarter → half
    i, j = @index(Global, NTuple)
    mabs = i; n = j - 1
    RT = eltype(dl)
    @inbounds dl[n + off, (-mabs) + off] = ifelse(iseven(ℓ + n), one(RT), -one(RT)) * dl[n + off, mabs + off]
end

@kernel function _recur_S3_kern!(dl, ℓ, off)         # n → −n: half → full
    i, j = @index(Global, NTuple)
    m = i - 1 - ℓ; nabs = j
    RT = eltype(dl)
    @inbounds dl[(-nabs) + off, m + off] = ifelse(iseven(ℓ + abs(m)), one(RT), -one(RT)) * dl[nabs + off, m + off]
end

# Forward contraction for degree ℓ: G_{m'm} += i^{m+s} N_ℓ sf_{ℓm} Δ^ℓ_{m'm} Δ^ℓ_{m',−s}.
# `mp0` is the first m' the mode buffer carries (`-ℓ` complex, `0` real) and `goff` maps m' to its row.
@kernel function _contract_kern!(G, @Const(dl), @Const(sf), ℓ, s, lmax, off, Nℓ, mp0, goff)
    i, j, b = @index(Global, NTuple)
    mp = i - 1 + mp0; m = j - 1 - ℓ
    CT = eltype(G)
    @inbounds begin
        ph = NUFSHT._im_pow(CT, -(m + s)) * sf[ℓ + 1, m + lmax + 1, b] * Nℓ
        G[mp + goff, m + off, b] += ph * dl[mp + off, m + off] * dl[mp + off, (-s) + off]
    end
end

# Adjoint contraction for degree ℓ: sf_{ℓm} = conj(i^{m+s}) N_ℓ Σ_{m'} Δ^ℓ_{m'm} Δ^ℓ_{m',−s} Ĝ_{m'm}.
@kernel function _contract_adj_kern!(sf, @Const(dl), @Const(Ĝ), ℓ, s, lmax, off, Nℓ, mp0, goff)
    j, b = @index(Global, NTuple)
    m = j - 1 - ℓ
    CT = eltype(sf)
    @inbounds begin
        phc = conj(NUFSHT._im_pow(CT, -(m + s))) * Nℓ
        acc = zero(CT)
        for mp in mp0:ℓ
            acc += dl[mp + off, m + off] * dl[mp + off, (-s) + off] * Ĝ[mp + goff, m + off, b]
        end
        sf[ℓ + 1, m + lmax + 1, b] = phc * acc
    end
end

# Which of the two layouts the plan's mode buffer is in, read off its θ extent.
@inline _fold_index(G, lmax, off, ℓ) =
    size(G, 1) == 2lmax + 1 ? (-ℓ, off) : (0, 1)

# The four stages are dependent (S1 reads Stage B, S2 reads S1, S3 reads S2) but they run on one
# backend, which executes launches in order — so no sync is needed between them, only before a host
# read. Syncing per launch cost ~4·lmax full device barriers per spin transform.
function _recur_step_device!(backend, dl, dlp, ℓ, off)
    _recur_AB_kern!(backend)(dl, dlp, ℓ, off; ndrange = (ℓ + 1,))
    _recur_S1_kern!(backend)(dl, ℓ, off; ndrange = (ℓ + 1, ℓ + 1))
    _recur_S2_kern!(backend)(dl, ℓ, off; ndrange = (ℓ, ℓ + 1))
    _recur_S3_kern!(backend)(dl, ℓ, off; ndrange = (2ℓ + 1, ℓ))
    return nothing
end

function NUFSHT._assemble_G_impl!(G::GPUArraysCore.AbstractGPUArray, sf,
                                  plan::NUFSHT.SpinNUSHTplan{T}, ::Nothing) where {T}
    lmax = plan.lmax; s = plan.s; off = lmax + 1; B = plan.B
    dl = plan.dl_curr; dlp = plan.dl_prev
    backend = KernelAbstractions.get_backend(G)
    fill!(G, zero(Complex{T}))
    _recur_seed_kern!(backend)(dl, off; ndrange = (1,))
    for ℓ in 0:lmax
        if ℓ > 0
            dl, dlp = dlp, dl
            _recur_step_device!(backend, dl, dlp, ℓ, off)
        end
        ℓ < abs(s) && continue
        Nℓ = T(sqrt((2ℓ + 1) / (4π)))
        mp0, goff = _fold_index(G, lmax, off, ℓ)
        _contract_kern!(backend)(G, dl, sf, ℓ, s, lmax, off, Nℓ, mp0, goff;
                                 ndrange = (ℓ - mp0 + 1, 2ℓ + 1, B))
    end
    _sync(backend)                     # one barrier for the whole sweep, before any host read
    return G
end

function NUFSHT._assemble_G_adjoint_impl!(sf::GPUArraysCore.AbstractGPUArray, Ĝ,
                                          plan::NUFSHT.SpinNUSHTplan{T}, ::Nothing) where {T}
    lmax = plan.lmax; s = plan.s; off = lmax + 1; B = plan.B
    dl = plan.dl_curr; dlp = plan.dl_prev
    backend = KernelAbstractions.get_backend(sf)
    fill!(sf, zero(Complex{T}))
    _recur_seed_kern!(backend)(dl, off; ndrange = (1,))
    for ℓ in 0:lmax
        if ℓ > 0
            dl, dlp = dlp, dl
            _recur_step_device!(backend, dl, dlp, ℓ, off)
        end
        ℓ < abs(s) && continue
        Nℓ = T(sqrt((2ℓ + 1) / (4π)))
        mp0, goff = _fold_index(Ĝ, lmax, off, ℓ)
        _contract_adj_kern!(backend)(sf, dl, Ĝ, ℓ, s, lmax, off, Nℓ, mp0, goff;
                                     ndrange = (2ℓ + 1, B))
    end
    _sync(backend)                     # one barrier for the whole sweep, before any host read
    return sf
end

# ── Device solver column primitives ─────────────────────────────────────────────
# The per-column workspace ops as device-generic broadcasts/reductions (the `src` versions are CPU
# scalar loops), dispatched on the data array being a device array, so `nusht_solve!` and
# `nusht_solve_spin!` run on the GPU. One set serves both paths: `_col_hdot!` degenerates to the plain
# dot product on a real array, since `conj` and `real` are identities there.
#
# `_cols` flattens a batch buffer to (stride, columns) so one method serves point space and
# coefficient space, matching the linear-column addressing the `src` versions use.
@inline _cols(A, n) = view(reshape(A, :, size(A, ndims(A))), :, 1:n)
#
# The trailing `n` is the live column count a compacted solve narrows to; the solver passes it on
# every call, so these must accept it or they are not the methods it selects. `src` takes a count
# rather than a view to stay allocation-free; here a view costs nothing against the kernel launch.
function NUFSHT._col_hdot!(dst, a::GPUArraysCore.AbstractGPUArray, b,      # dst[k] = Re Σ conj(a)·b
                           n::Integer = size(a, ndims(a)))
    av = _cols(a, n); bv = _cols(b, n)
    view(dst, 1:n) .= dropdims(sum(real.(conj.(av) .* bv); dims = 1); dims = 1)
    return dst
end
function NUFSHT._col_axpy!(y::GPUArraysCore.AbstractGPUArray, α, x, σ,     # y[:,k] += σ·α[k]·x[:,k]
                           n::Integer = size(y, ndims(y)))
    _cols(y, n) .+= σ .* reshape(view(α, 1:n), 1, :) .* _cols(x, n)
    return y
end
function NUFSHT._col_pbp!(p::GPUArraysCore.AbstractGPUArray, r, β,         # p[:,k] = r[:,k] + β[k]·p[:,k]
                          n::Integer = size(p, ndims(p)))
    _cols(p, n) .= _cols(r, n) .+ reshape(view(β, 1:n), 1, :) .* _cols(p, n)
    return p
end
function NUFSHT._col_scale!(y::GPUArraysCore.AbstractGPUArray, s,          # y[:,k] *= s[k]
                            n::Integer = size(y, ndims(y)))
    _cols(y, n) .*= reshape(view(s, 1:n), 1, :)
    return y
end

# ── Device pack/unpack between the packed solver vectors and the plan's full layout ──────────────
# The scalar solver carries its vectors packed to the `l ≤ lmax` slots, so crossing to and from the
# plan's `(Nθ, Nφ, B)` layout is a gather/scatter by index vector. The `src` versions are scalar loops
# (zero-allocation on a host array); here they are broadcasts over reshaped views, which a device array
# can execute — a `SubArray` header costs nothing against a kernel launch.
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

function NUFSHT._write_solution!(C::GPUArraysCore.AbstractGPUArray, ws::NUFSHT.LSMRWorkspace,
                                 plan::NUFSHT.NUSHTplan, slot::Integer, dstcol::Integer)
    full = NUFSHT._fulllen(plan)
    col = view(reshape(C, full, :), :, dstcol)
    fill!(col, zero(eltype(C)))
    view(col, ws.valid) .= view(reshape(ws.x, length(ws.valid), :), :, slot)
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

# ── Device spectral filter (× H(ℓ)) ─────────────────────────────────────────────
# `C` is `(lmax+1, 2lmax+1, B)`; build the host transfer matrix once, move it to `C`'s backend, and
# broadcast-multiply across the batch (the `src` version is a scalar mode loop).
# `nusht_filter!` applies this once per field, and the transfer matrix depends only on
# (filter, lmax, element type, array type) — so build it, upload it and reshape it once rather than
# allocating a host matrix plus a device copy on every call.
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

# A precomputed WignerTable is a host array read elementwise; contracting against it from a device
# array would fall back to scalar indexing. Refuse explicitly rather than silently crawling.
_no_device_table() = throw(ArgumentError(
    "a device spin plan cannot use a precomputed WignerTable; build it with wigner_table = nothing " *
    "so the Δ planes are generated on device by the recurrence kernels."))
NUFSHT._assemble_G_impl!(::GPUArraysCore.AbstractGPUArray, sf, plan, ::NUFSHT.WignerTable) = _no_device_table()
NUFSHT._assemble_G_adjoint_impl!(::GPUArraysCore.AbstractGPUArray, Ĝ, plan, ::NUFSHT.WignerTable) = _no_device_table()

end # module NUFSHTKernelAbstractionsExt
