"""
    NUFSHTKernelAbstractionsExt

Device-generic GPU methods for the array-indexed steps of the transform, written **once** as
KernelAbstractions `@kernel`s and dispatched on `GPUArraysCore.AbstractGPUArray` — so they run on any
KA backend (CUDA/ROCm/Metal/oneAPI, and the reference `JLArray`/KA-CPU backend used for testing)
without per-vendor code. The plain CPU `Array` methods in `src` are untouched; a GPU-backed array is
`<:AbstractGPUArray` so these more-specific methods win for it.

Only the genuinely scatter/gather steps need kernels — the Double-Fourier-Sphere doubling/folding.
Everything else in the pipeline is already array-generic: the half-pixel phase is a broadcast, the
2-D FFT goes through `AbstractFFTs` plans (→ CUFFT/rocFFT/… by array type), and the NUFFT is
dispatched through the `_nufft_*` seam (cuFINUFFT via the CUDA extension).

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

# ── Double-Fourier-Sphere doubling: F̃(2Nθ,Nφ,B) ← F(Nθ,Nφ,B) ──────────────────
# Top half copies F; bottom half is the pole-reflected, φ+π-shifted mirror. The `mod1(j+half,Nφ)`
# column shift is the same proper cyclic permutation the CPU method uses (correct for odd Nφ).
@kernel function _dfs_double_kern!(F̃, @Const(F), Nθ, Nφ, half)
    i, j, b = @index(Global, NTuple)
    if i <= Nθ
        @inbounds F̃[i, j, b] = F[i, j, b]
    else
        ii = i - Nθ
        js = mod1(j + half, Nφ)
        @inbounds F̃[i, j, b] = F[Nθ + 1 - ii, js, b]
    end
end

function NUFSHT.dfs_double!(F̃::GPUArraysCore.AbstractGPUArray, F::GPUArraysCore.AbstractGPUArray)
    Nθ, Nφ, B = size(F, 1), size(F, 2), size(F, 3)
    @assert ndims(F̃) == 3 && ndims(F) == 3 "GPU dfs_double! requires 3-D (Nθ,Nφ,B) buffers"
    @assert size(F̃, 1) == 2Nθ && size(F̃, 2) == Nφ && size(F̃, 3) == B
    backend = KernelAbstractions.get_backend(F̃)
    _dfs_double_kern!(backend)(F̃, F, Nθ, Nφ, Nφ ÷ 2; ndrange = (2Nθ, Nφ, B))
    _sync(backend)
    return F̃
end

# ── DFS folding (exact adjoint of doubling): F(Nθ,Nφ,B) ← real(F̃ + mirror(F̃)) ──
# The `-half` column shift is the exact adjoint of `dfs_double!`'s `+half`; `real` is the adjoint of
# the real→complex embedding. This keeps the composite `nusht_solve!` adjoint exact on device too.
@kernel function _dfs_fold_kern!(F, @Const(F̃), Nθ, Nφ, half)
    i, j, b = @index(Global, NTuple)
    im = 2Nθ + 1 - i
    js = mod1(j - half, Nφ)
    @inbounds F[i, j, b] = real(F̃[i, j, b] + F̃[im, js, b])
end

function NUFSHT.dfs_fold!(F::GPUArraysCore.AbstractGPUArray, F̃::GPUArraysCore.AbstractGPUArray)
    Nθ, Nφ, B = size(F, 1), size(F, 2), size(F, 3)
    @assert ndims(F̃) == 3 && ndims(F) == 3 "GPU dfs_fold! requires 3-D (Nθ,Nφ,B) buffers"
    @assert size(F̃, 1) == 2Nθ && size(F̃, 2) == Nφ && size(F̃, 3) == B
    backend = KernelAbstractions.get_backend(F)
    _dfs_fold_kern!(backend)(F, F̃, Nθ, Nφ, Nφ ÷ 2; ndrange = (Nθ, Nφ, B))
    _sync(backend)
    return F
end

# ── Device spin S-engine: on-the-fly Trapani–Navaza recurrence + G-contraction ──
# The spin transform's `_assemble_G!` (Legendre step + bivariate-Fourier assembly) as KA kernels, so
# the whole spin path runs on device (recurrence buffers `similar` to the device nodes; NUFFT via the
# cuFINUFFT seam). Same math as the CPU `src/Spin.jl` methods — validated bit-for-bit on JLArray. The
# degree loop is sequential (ℓ depends on ℓ-1); per degree, Stage A+B is one thread per column, the
# symmetry fills and the contraction are elementwise.

# Trapani–Navaza degree step, one workitem per column n∈0:ℓ: Stage A (row m=ℓ) then the downward
# Stage B m-sweep for that column (self-contained — no cross-column dependency).
@kernel function _recur_AB_kern!(dl, @Const(dlp), ℓ, off)
    c = @index(Global)
    n = c - 1
    RT = eltype(dl)
    @inbounds begin
        if n == 0
            dl[ℓ + off, off] = -sqrt(RT(2ℓ - 1) / RT(2ℓ)) * dlp[(ℓ - 1) + off, off]
        else
            dl[ℓ + off, n + off] =
                sqrt(RT(ℓ) * RT(2ℓ - 1) / (RT(2) * RT(ℓ + n) * RT(ℓ + n - 1))) * dlp[(ℓ - 1) + off, (n - 1) + off]
        end
        if ℓ - 1 >= n
            m = ℓ - 1
            dl[m + off, n + off] = RT(2n) / sqrt(RT(ℓ - m) * RT(ℓ + m + 1)) * dl[(m + 1) + off, n + off]
        end
        for m in (ℓ - 2):-1:n
            s1 = sqrt(RT(ℓ - m) * RT(ℓ + m + 1))
            s2 = sqrt(RT(ℓ - m - 1) * RT(ℓ + m + 2) / (RT(ℓ - m) * RT(ℓ + m + 1)))
            dl[m + off, n + off] = RT(2n) / s1 * dl[(m + 1) + off, n + off] - s2 * dl[(m + 2) + off, n + off]
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
        @inbounds dl[m + off, n + off] = ifelse(iseven(m + n), one(RT), -one(RT)) * dl[n + off, m + off]
    end
end

@kernel function _recur_S2_kern!(dl, ℓ, off)         # m → −m: quarter → half
    i, j = @index(Global, NTuple)
    mabs = i; n = j - 1
    RT = eltype(dl)
    @inbounds dl[(-mabs) + off, n + off] = ifelse(iseven(ℓ + n), one(RT), -one(RT)) * dl[mabs + off, n + off]
end

@kernel function _recur_S3_kern!(dl, ℓ, off)         # n → −n: half → full
    i, j = @index(Global, NTuple)
    m = i - 1 - ℓ; nabs = j
    RT = eltype(dl)
    @inbounds dl[m + off, (-nabs) + off] = ifelse(iseven(ℓ + abs(m)), one(RT), -one(RT)) * dl[m + off, nabs + off]
end

# i^k for integer k ∈ ℤ (GPU-safe, no generic complex power): cycles 1, i, −1, −i.
@inline function _im_pow(::Type{CT}, k) where {CT}
    r = mod(k, 4)
    return r == 0 ? CT(1, 0) : r == 1 ? CT(0, 1) : r == 2 ? CT(-1, 0) : CT(0, -1)
end

# Forward contraction for degree ℓ: G_{m'm} += i^{m+s} N_ℓ sf_{ℓm} Δ^ℓ_{m'm} Δ^ℓ_{m',−s}.
@kernel function _contract_kern!(G, @Const(dl), @Const(sf), ℓ, s, lmax, off, Nℓ)
    i, j, b = @index(Global, NTuple)
    mp = i - 1 - ℓ; m = j - 1 - ℓ
    CT = eltype(G)
    @inbounds begin
        ph = _im_pow(CT, m + s) * sf[ℓ + 1, m + lmax + 1, b] * Nℓ
        G[mp + off, m + off, b] += ph * dl[m + off, mp + off] * dl[(-s) + off, mp + off]
    end
end

# Adjoint contraction for degree ℓ: sf_{ℓm} = conj(i^{m+s}) N_ℓ Σ_{m'} Δ^ℓ_{m'm} Δ^ℓ_{m',−s} Ĝ_{m'm}.
@kernel function _contract_adj_kern!(sf, @Const(dl), @Const(Ĝ), ℓ, s, lmax, off, Nℓ)
    j, b = @index(Global, NTuple)
    m = j - 1 - ℓ
    CT = eltype(sf)
    @inbounds begin
        phc = conj(_im_pow(CT, m + s)) * Nℓ
        acc = zero(CT)
        for mp in -ℓ:ℓ
            acc += dl[m + off, mp + off] * dl[(-s) + off, mp + off] * Ĝ[mp + off, m + off, b]
        end
        sf[ℓ + 1, m + lmax + 1, b] = phc * acc
    end
end

function _recur_step_device!(backend, dl, dlp, ℓ, off)
    _recur_AB_kern!(backend)(dl, dlp, ℓ, off; ndrange = (ℓ + 1,)); _sync(backend)
    _recur_S1_kern!(backend)(dl, ℓ, off; ndrange = (ℓ + 1, ℓ + 1)); _sync(backend)
    _recur_S2_kern!(backend)(dl, ℓ, off; ndrange = (ℓ, ℓ + 1)); _sync(backend)
    _recur_S3_kern!(backend)(dl, ℓ, off; ndrange = (2ℓ + 1, ℓ)); _sync(backend)
    return nothing
end

function NUFSHT._assemble_G!(G::GPUArraysCore.AbstractGPUArray, sf, plan::NUFSHT.SpinNUSHTplan{T}) where {T}
    lmax = plan.lmax; s = plan.s; off = lmax + 1; B = plan.B
    dl = plan.dl_curr; dlp = plan.dl_prev
    backend = KernelAbstractions.get_backend(G)
    fill!(G, zero(Complex{T}))
    _recur_seed_kern!(backend)(dl, off; ndrange = (1,)); _sync(backend)
    for ℓ in 0:lmax
        if ℓ > 0
            dl, dlp = dlp, dl
            _recur_step_device!(backend, dl, dlp, ℓ, off)
        end
        ℓ < abs(s) && continue
        Nℓ = T(sqrt((2ℓ + 1) / (4π)))
        _contract_kern!(backend)(G, dl, sf, ℓ, s, lmax, off, Nℓ; ndrange = (2ℓ + 1, 2ℓ + 1, B))
        _sync(backend)
    end
    return G
end

function NUFSHT._assemble_G_adjoint!(sf::GPUArraysCore.AbstractGPUArray, Ĝ, plan::NUFSHT.SpinNUSHTplan{T}) where {T}
    lmax = plan.lmax; s = plan.s; off = lmax + 1; B = plan.B
    dl = plan.dl_curr; dlp = plan.dl_prev
    backend = KernelAbstractions.get_backend(sf)
    fill!(sf, zero(Complex{T}))
    _recur_seed_kern!(backend)(dl, off; ndrange = (1,)); _sync(backend)
    for ℓ in 0:lmax
        if ℓ > 0
            dl, dlp = dlp, dl
            _recur_step_device!(backend, dl, dlp, ℓ, off)
        end
        ℓ < abs(s) && continue
        Nℓ = T(sqrt((2ℓ + 1) / (4π)))
        _contract_adj_kern!(backend)(sf, dl, Ĝ, ℓ, s, lmax, off, Nℓ; ndrange = (2ℓ + 1, B))
        _sync(backend)
    end
    return sf
end

# ── Device spin CG reductions ───────────────────────────────────────────────────
# The per-column CG workspace ops as device-generic broadcasts/reductions (the `src` versions are CPU
# scalar loops). Dispatched on the data array being a device array. This lets `nusht_solve_spin!` run
# on the GPU (the scalars stay length-B device vectors; `maximum` for the convergence check syncs to
# host). CPU stays on the zero-alloc loop methods.
function NUFSHT._col_hdot!(dst, a::GPUArraysCore.AbstractGPUArray, b)   # dst[k] = Re Σ_ij conj(a)·b
    dst .= real.(dropdims(sum(conj.(a) .* b; dims = (1, 2)); dims = (1, 2)))
    return dst
end
function NUFSHT._col_axpy_c!(y::GPUArraysCore.AbstractGPUArray, α, x, σ)   # y[:,:,k] += σ·α[k]·x[:,:,k]
    y .+= σ .* reshape(α, 1, 1, :) .* x
    return y
end
function NUFSHT._col_pbp_c!(p::GPUArraysCore.AbstractGPUArray, r, β)       # p[:,:,k] = r + β[k]·p[:,:,k]
    p .= r .+ reshape(β, 1, 1, :) .* p
    return p
end

# ── Device scalar CG reductions ─────────────────────────────────────────────────
# The real-valued scalar-path analogues of the spin `_col_*_c!` above (the `src` versions are CPU
# scalar loops over a real `(Nθ,Nφ,B)` array — no `conj`/`real`). Dispatched on the data array being a
# device array so `nusht_solve!` runs on the GPU; CPU stays on the zero-alloc loop methods.
function NUFSHT._col_dot!(dst, a::GPUArraysCore.AbstractGPUArray, b)       # dst[k] = Σ_ij a·b
    dst .= dropdims(sum(a .* b; dims = (1, 2)); dims = (1, 2))
    return dst
end
function NUFSHT._col_axpy!(y::GPUArraysCore.AbstractGPUArray, α, x, σ)     # y[:,:,k] += σ·α[k]·x[:,:,k]
    y .+= σ .* reshape(α, 1, 1, :) .* x
    return y
end
function NUFSHT._col_pbp!(p::GPUArraysCore.AbstractGPUArray, r, β)         # p[:,:,k] = r + β[k]·p[:,:,k]
    p .= r .+ reshape(β, 1, 1, :) .* p
    return p
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
function NUFSHT.apply_transfer!(C::GPUArraysCore.AbstractGPUArray, filter, lmax)
    H = NUFSHT._transfer_matrix(filter, lmax, real(eltype(C)))
    Hd = NUFSHT._to_like(C, H)
    C .*= reshape(Hd, size(Hd, 1), size(Hd, 2), 1)
    return C
end

end # module NUFSHTKernelAbstractionsExt
