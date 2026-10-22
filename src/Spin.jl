# Spin-weighted (spin-s) non-uniform spherical harmonic transforms at arbitrary scattered points, from
# the Wigner-d Fourier factorization and a 2-D nonuniform FFT. A spin-`s` band-limited field is
#
#     f(θ,φ) = Σ_{ℓ,m} ₛf_{ℓm} ₛY_{ℓm}(θ,φ),   ₛY_{ℓm}(θ,φ) = N_ℓ d^ℓ_{m,−s}(θ) e^{imφ},
#
# with `N_ℓ = √((2ℓ+1)/4π)` and `d^ℓ` the Wigner small-d. With
# `d^ℓ_{mn}(θ) = i^{m−n} Σ_{m'} Δ^ℓ_{m'm} Δ^ℓ_{m'n} e^{−im'θ}` (`Δ^ℓ ≡ d^ℓ(π/2)`) the field is the
# bivariate Fourier series
#
#     f(θ,φ) = Σ_{m',m} G_{m'm} e^{−im'θ} e^{imφ},   G_{m'm} = i^{m+s} Σ_ℓ ₛf_{ℓm} N_ℓ Δ^ℓ_{m'm} Δ^ℓ_{m',−s},
#
# which one 2-D NUFFT evaluates at the points. The NUFFT is handed the negated colatitudes `−θ` with
# `iflag = +1`, which absorbs `e^{−im'θ}`, so the centered mode array `G` is the NUFFT's modes. Analysis
# is the exact Euclidean adjoint (type 1 at `−θ`, `iflag = −1`, and the transpose Δ-contraction), and
# `nusht_solve_spin!` inverts by LSMR.
#
# Coefficients are a dense `(lmax+1) × (2lmax+1)` array, `sf[ℓ+1, m+lmax+1]`. Spin `s = ±1` is the
# tangent-vector case, `U = u_θ + i u_φ`.
#
# A plan over host `Float64` nodes forms `G` with FastTransforms' `plan_spinsph2fourier` (Givens
# rotations, then triangular Jacobi-to-Chebyshev products); any other element type, and device nodes,
# with the Δ contraction. FastTransforms holds `ₛf_{ℓm}` at row `ℓ − max(|m|,|s|) + 1` of column `1`
# (`m = 0`), `2m+1` (`m > 0`) or `2|m|` (`m < 0`), as the coefficient of `σ_{ms} ₛY_ℓm` with
# `σ_{ms} = (−1)^{m+s}` for `m + s > 0` and `1` otherwise, and returns column `m` as the series
#
#     (2π)^{−1/2} e^{imφ} Σ_{j=0}^{lmax} B_j cos(jθ)          (m + s even),
#     (2π)^{−1/2} e^{imφ} Σ_{j=0}^{lmax} B_j sin((j+1)θ)      (m + s odd),
#
# whose sine row `B_lmax` is zero on coefficients of degree `ℓ ≤ lmax`. Sources: FastTransforms' C
# library, `ft_plan_rotspinsphere` and `kernel_spinsph_hi2lo_default` (`src/rotations*`),
# `ft_execute_spinsph2fourier` (`src/drivers.c`), `ft_execute_spinsph_synthesis` (`src/fftw.c`).

using LinearAlgebra: LinearAlgebra
using SpecialFunctions: loggamma

export SpinNUSHTplan, make_spin_plan, WignerTable
export nusht_type2_spin!, nusht_type1_spin!, nusht_solve_spin!
export spin_coeff_index, sYlm
export scalar_to_spin0!, spin0_to_scalar!

"""
    wigner_d(ℓ, m, n, β)

Wigner small-d matrix element `d^ℓ_{mn}(β) = ⟨ℓm|exp(-iβ Jy)|ℓn⟩` in the Condon–Shortley basis, via
the explicit alternating sum with log-gamma factorials for stability.

Accuracy degrades above `ℓ ≈ 40`; at `β = π/2` [`_wigner_quarter!`](@ref) holds to large `ℓ`.
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

# i^k for integer k, by its period-4 cycle 1, i, −1, −i: exact, and a form a device kernel evaluates
# (`Complex^Integer` goes through `exp(k log z)`).
@inline function _im_pow(::Type{CT}, k::Integer) where {CT}
    r = mod(k, 4)
    return r == 0 ? CT(1, 0) : r == 1 ? CT(0, 1) : r == 2 ? CT(-1, 0) : CT(0, -1)
end

"""
    sYlm(s, ℓ, m, θ, φ) -> Complex

Spin-weighted spherical harmonic `ₛY_{ℓm}(θ,φ) = N_ℓ d^ℓ_{m,−s}(θ) e^{imφ}` in the
Goldberg / Newman–Penrose convention, with `d` the rotation matrix of [`wigner_d`](@ref).
"""
sYlm(s::Integer, ℓ::Integer, m::Integer, θ::Real, φ::Real) = _Nℓ(ℓ) * wigner_d(ℓ, m, -s, θ) * cis(m * φ)

"""
    spin_coeff_index(ℓ, m, lmax) -> CartesianIndex

Index into the dense `(lmax+1, 2lmax+1)` spin coefficient array for degree `ℓ`, order `m`.
"""
@inline spin_coeff_index(ℓ::Integer, m::Integer, lmax::Integer) = CartesianIndex(ℓ + 1, m + lmax + 1)

"""
    WignerTable(lmax, s; T = Float64)

Precomputed `Q^ℓ_{m'm} = Δ^ℓ_{m'm} · Δ^ℓ_{m',−s}` for every degree `ℓ ≤ lmax`, the combination of the
Wigner-d(π/2) planes the spin contraction reads. It depends only on `(lmax, s, T)` and is read-only, so
one table serves any number of plans on any number of threads.

A host plan given a table contracts with it on every transform, and holds `Σ_ℓ (2ℓ+1)² ≈ (4/3) lmax³`
values. The Wigner recurrence builds every degree's plane on each transform, twice per
[`nusht_solve_spin!`](@ref) iteration. A device plan refuses a table.

    tbl = WignerTable(64, 1; T = Float32)
    plans = [make_spin_plan(ComplexF32, θs[i], φs[i], 64, 1; wigner_table = tbl) for i in eachindex(θs)]
"""
struct WignerTable{T, V<:AbstractVector{T}, VI<:AbstractVector{Int}}
    lmax::Int
    s::Int
    Q::V            # plane ℓ is (2ℓ+1)² column-major at offsets[ℓ+1]; (mp,m) → +(m+ℓ)(2ℓ+1)+(mp+ℓ)
    offsets::VI
end

function WignerTable(lmax::Integer, s::Integer; T::Type{<:AbstractFloat} = Float64)
    offsets = Vector{Int}(undef, lmax + 2)
    offsets[1] = 1
    for ℓ in 0:lmax
        offsets[ℓ + 2] = offsets[ℓ + 1] + (2ℓ + 1)^2
    end
    Q = zeros(T, offsets[lmax + 2] - 1)
    bnd = _wigner_boundary(T, lmax)
    P = zeros(T, lmax + 1, lmax + 1, 1)
    for ℓ in abs(s):lmax
        _wigner_quarter!(P, 1, bnd, ℓ)
        nl = 2ℓ + 1; base = offsets[ℓ + 1]
        @inbounds for m in -ℓ:ℓ
            col = base + (m + ℓ) * nl + ℓ
            @simd for mp in -ℓ:ℓ
                Q[col + mp] = _qprod(P, 1, T, ℓ, mp, m, Int(s))
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
`lmax`, transforming `B` co-located fields per call. Owns persistent NUFFT plans (type 2 `iflag=+1`,
type 1 `iflag=−1`) over the nodes `(−θ, φ)`, and the engine that takes coefficients to the bivariate
Fourier series `G` and back, chosen by the nodes' array and element type:

- host `Float64` nodes: FastTransforms' `plan_spinsph2fourier` and its adjoint, one pair per task of a
  batch's column loop.
- any other element type, or device nodes: the Wigner `Δ^ℓ = d^ℓ(π/2)` planes, built from their stored
  `m = ℓ` columns ([`_wigner_boundary`](@ref)) a block of `K` degrees at a time, one `(lmax+1)²`
  quarter plane per task ([`_wigner_quarter!`](@ref)), and contracted over the orders on the same
  tasks; `K` is `ntasks` on the host. The recurrence is numerically stable to `ℓ ≈ 1024`, where the
  explicit sum of [`wigner_d`](@ref) loses its accuracy above `ℓ ≈ 40`.
- a [`WignerTable`](@ref) passed as `wigner_table`: the precomputed products the contraction reads, at
  O(lmax³) memory.

Each engine computes the same transform.
"""
struct SpinNUSHTplan{T<:AbstractFloat, CT3<:AbstractArray{Complex{T},3}, ND<:AbstractNodeSet, E,
                     FT} <: AbstractNUSHTplan
    lmax::Int
    s::Int
    B::Int
    tol::FT
    ntasks::Int            # the plan's thread count: the S-step's tasks, FastTransforms' OpenMP count
    nodes::ND              # points, (M,B) strengths and the NUFFT handles; see AbstractNodeSet
    G::CT3                 # (L, L, B) bivariate-Fourier mode buffer, L = 2lmax+1
    engine::E              # _FTSpinEngine, _WignerPlanes or WignerTable; see _spin_engine
    solve_w::Base.RefValue{Union{Nothing,CT3}}   # the fit's full-layout coefficients, built by the first solve
end

"""
    make_spin_plan([FE = Complex{float(eltype(θ_nodes))},] θ_nodes, φ_nodes, lmax, s; execution = AutoBackend(), tol=1e-10, ntrans=1, …)

Build a `SpinNUSHTplan`. Colatitudes `θ ∈ [0,π]`, longitudes `φ ∈ [0,2π)`.

`FE` is the field element type, positional as in [`make_plan`](@ref); a spin field is complex in
general, hence the default. A real `FE` asserts the field's values are real, which makes the mode array
conjugate-symmetric, so the plan forms and transforms only its `m' ≥ 0` half — correct only if the
coefficients satisfy the reality condition.

On host `Float64` nodes the step from coefficients to the bivariate Fourier series is FastTransforms'
spin-weighted transform; for any other element type, and on device nodes, it is the Wigner recurrence
([`SpinNUSHTplan`](@ref)). `wigner_table`, a [`WignerTable`](@ref) for `(lmax, s)`, makes a host plan
contract with the table.

`nthreads` and `upsampfac` are as in [`make_plan`](@ref); `nthreads` is also the task count of that
step, the columns of a batch under FastTransforms and the degree blocks of the recurrence, and the
thread count FastTransforms runs at outside those tasks. `execution` is
`SerialBackend()`, `ThreadedBackend()`, `AutoBackend()` (the default) or `GPUBackend(b)` as in
[`make_plan`](@ref), with the same precedence of an explicit `nthreads`. A spin plan holds one process's
points; an MPI fit passes each rank's plan to `nusht_solve_spin!(sf, f_local, plan, MPIBackend(; comm))`,
and an MPI analysis to `nusht_analyze!(sf, f_local, plan, MPIBackend(; comm))`.
"""
make_spin_plan(θ_nodes, φ_nodes, lmax::Integer, s::Integer; kwargs...) =
    make_spin_plan(Complex{float(eltype(θ_nodes))}, θ_nodes, φ_nodes, lmax, s; kwargs...)

make_spin_plan(::Type{FE}, θ_nodes, φ_nodes, lmax::Integer, s::Integer;
               execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
               kwargs...) where {FE<:Number} =
    _make_spin_plan(FE, θ_nodes, φ_nodes, lmax, s, execution; kwargs...)

# `make_spin_plan` under `execution`; NUFSHTKernelAbstractionsExt adds `GPUBackend`.
_make_spin_plan(::Type{FE}, θ, φ, lmax, s, execution::_HostExecution; nthreads = nothing,
                kwargs...) where {FE} =
    _build_spin_plan(FE, θ, φ, lmax, s; nthreads = _plan_threads(execution, nthreads), kwargs...)
_make_spin_plan(::Type{FE}, θ, φ, lmax, s, e::ComputationalBackends.AbstractExecutionBackend;
                kwargs...) where {FE} =
    _execution_unavailable(e, "a spin plan")

function _build_spin_plan(::Type{FE}, θ_nodes, φ_nodes, lmax::Integer, s::Integer;
                          tol = 1e-10, ntrans::Integer = 1,
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
    # A backend with a real-data transform supplies the absent `m' < 0` half itself, from a transform
    # over the full `L` θ wavenumbers, and returns real strengths. Without one the transform is
    # half-height and complex: `θ_shift` undoes its centered row labelling per point, and the fold
    # weights account for the missing half.
    realfield = FE <: Real
    nub = _resolve_nufft(nufft, FE)
    r2c = realfield && _real_capable(nub)
    Lθ = realfield ? lmax + 1 : L
    ZS = r2c ? T : Complex{T}                  # element type of the non-uniform data

    # Node vectors keep their input array type (host `Vector` or device array), with eltype `T`.
    θ = T.(θ_nodes)
    φ = T.(φ_nodes)
    negθ = -θ                                  # absorbs the e^{-im'θ} factor into iflag=+1

    # Buffers are allocated like the nodes, so device nodes give a device-resident plan.
    ntasks = _task_count(nthreads)
    isnothing(wigner_table) || _check_table(wigner_table, (lmax = Int(lmax), s = Int(s)))
    engine = _spin_engine(θ, Int(lmax), Int(s), ntasks, B, wigner_table)
    G    = _zeros_like(θ, Complex{T}, Lθ, L, B)
    fbuf = _zeros_like(θ, ZS, M, B)
    # Centered order labels a length-`Lθ` axis `-(Lθ÷2) … `, so a `m' ≥ 0` axis is that ordering shifted
    # by a constant, undone per point. `θ_nufft` is `-θ`, which is the coordinate this pairs with. A
    # real-data transform's half-spectrum is labelled `0 … n₁÷2` already and needs no shift.
    θ_shift = (realfield && !r2c) ?
        _to_like(θ, Complex{T}.(cis.(T(Lθ ÷ 2) .* negθ))) : nothing

    # The NUFFT is built on the nodes' device, in centered mode order. A real transform covers the full
    # θ axis and stores its half; a complex half-height one is built at the stored size.
    tol64 = Float64(tol)
    n_modes = Int64[r2c ? L : Lθ, L]
    _warn_if_directsum(nufft, nub, M, Lθ * L)
    nt = nthreads === nothing ? 0 : Int(nthreads)
    uf = upsampfac === nothing ? 0.0 : Float64(upsampfac)
    # See `_nufft_share_directions`: where the backend's plan carries no direction, one object serves
    # both and the second is a handle onto it, already pointed at the same nodes.
    nufft_type2 = _make_nufft(nub, (negθ, φ), 2, n_modes, +1, B, tol64, T, nt, uf, ZS)
    nufft_type1 = _build_analysis(directions, nub, negθ, φ, n_modes, B, tol64, T, nt, uf, ZS, nufft_type2)
    nodes =_node_set(Val(variable_npts), θ, φ, negθ, θ_shift, fbuf, nufft_type2, nufft_type1)

    return SpinNUSHTplan{T, typeof(G), typeof(nodes), typeof(engine), typeof(tol64)}(
        lmax, Int(s), B, tol64, ntasks, nodes, G, engine, Base.RefValue{Union{Nothing,typeof(G)}}(nothing))
end

# Quarter planes a plan holds: one per task of a degree block.
_plane_count(::AbstractArray, ntasks::Integer) = Int(ntasks)

"""
    _WignerPlanes(bnd, planes)

The Wigner recurrence's state: `bnd[n+1, ℓ+1] = Δ^ℓ_{n,ℓ}` ([`_wigner_boundary`](@ref)) and the quarter
planes of one degree block, `(lmax+1, lmax+1, K)`, both in the nodes' array type.
"""
struct _WignerPlanes{MT<:AbstractMatrix, PT<:AbstractArray{<:Any,3}}
    bnd::MT
    planes::PT
end

# The recurrence's state for nodes `θ`, with the planes `ntasks` tasks fill.
_wigner_planes(θ::AbstractVector{T}, lmax::Integer, ntasks::Integer) where {T} =
    _WignerPlanes(_to_like(θ, _wigner_boundary(T, lmax)),
                  _zeros_like(θ, T, lmax + 1, lmax + 1, _plane_count(θ, ntasks)))

"""
    _FTSpinEngine

FastTransforms' spin-weighted harmonic to bivariate Fourier plan for `(lmax, s)`, its adjoint, and the
`(lmax+1, 2lmax+1)` complex slice one coefficient column passes through. When a batch's columns run on
`n > 1` tasks, `pool` holds `n` such triples, the first of them the engine's own. A FastTransforms plan
holds its own work array, so it serves one task at a time.
"""
struct _FTSpinEngine{P, PA, PL<:AbstractVector}
    plan::P
    adj::PA
    slice::Matrix{ComplexF64}
    pool::PL
end

# Built at one FFTW planner thread, under FFTW.jl's planner lock, as the scalar sphere plans are, and
# FastTransforms at the plan's thread count `n`. The two-argument `AdjointFTPlan` defines the `adjoint`
# field, which `P'` leaves undefined.
function _ft_spin_plans(slice::Matrix{ComplexF64}, s::Integer, n::Integer)
    return FFTW.set_num_threads(1) do
        P = _ft_call(() -> FastTransforms.plan_spinsph2fourier(slice, Int(s)), n)
        return (P, FastTransforms.AdjointFTPlan(P, P))
    end
end

function _ft_spin_engine(lmax::Integer, s::Integer, nslots::Integer, n::Integer)
    slice = zeros(ComplexF64, lmax + 1, 2lmax + 1)
    P, Padj = _ft_spin_plans(slice, s, n)
    pool = Tuple{Matrix{ComplexF64}, typeof(P), typeof(Padj)}[]
    nslots > 1 || return _FTSpinEngine(P, Padj, slice, pool)
    push!(pool, (slice, P, Padj))
    for _ in 2:nslots
        sl = similar(slice)
        push!(pool, (sl, _ft_spin_plans(sl, s, n)...))
    end
    return _FTSpinEngine(P, Padj, slice, pool)
end

# The engine of a plan over nodes `θ`: the caller's table; FastTransforms on host `Float64` nodes, with
# one slot per task the batch's columns can use; the Wigner recurrence on any other nodes.
_spin_engine(θ, lmax::Int, s::Int, ntasks::Int, B::Int, tbl::WignerTable) = tbl
_spin_engine(θ::Array{Float64}, lmax::Int, s::Int, ntasks::Int, B::Int, ::Nothing) =
    _ft_spin_engine(lmax, s, min(ntasks, B), ntasks)
_spin_engine(θ::AbstractArray, lmax::Int, s::Int, ntasks::Int, B::Int, ::Nothing) =
    _wigner_planes(θ, lmax, ntasks)

@inline _shift_offset(plan::SpinNUSHTplan{T}) where {T} = T(size(plan.G, 1) ÷ 2)
_copy_out!(f, fbuf, plan::SpinNUSHTplan) = _copy_out!(f, fbuf, _θshift(plan))
_copy_out!(f, fbuf, ::Nothing) = _copy_field!(f, fbuf)
_copy_out!(f, fbuf, ::AbstractVector) = _copy_real!(f, fbuf)

"""
    FullModes, FoldedComplex, FoldedReal

Which Hermitian treatment a spin plan's mode array needs. A complex field has no symmetry to exploit
([`FullModes`](@ref)); a real one stores only `m' ≥ 0` either way, and the two folded kinds differ in
who supplies the absent `m' < 0` half, which fixes the row indexing and the weights.

The kind is read off the plan: only a half-height complex transform has a `θ_shift`, and only a
real-data transform has real strengths, so the pair `(θ_shift, fbuf)` names all three.
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
@inline _mprange(ℓ::Integer, ::FullModes) = -ℓ:ℓ
@inline _mprange(ℓ::Integer, ::Union{FoldedComplex,FoldedReal}) = 0:ℓ

# A half-height complex transform returns only what it was given, so each retained mode has to stand
# for its partner too: everything counts twice except `(0,0)`, which is its own, and the `m' = 0` row's
# `m < 0` half, which is the one dropped. A real-data transform reconstructs the partner itself, so the
# forward hands it the half unweighted.
_fold_weights!(G, plan::SpinNUSHTplan) = _fold_weights!(G, plan, _fold_kind(plan))
_fold_weights!(G, ::SpinNUSHTplan, ::Union{FullModes,FoldedReal}) = G
function _fold_weights!(G, plan::SpinNUSHTplan{T}, ::FoldedComplex) where {T}
    lmax = plan.lmax
    G .*= T(2)
    # Views, which a GPU array takes; it rejects scalar indexing.
    @views fill!(G[1, 1:lmax, :], zero(eltype(G)))
    @views G[1, lmax + 1, :] ./= T(2)
    return G
end

# The transpose of the forward. `FoldedComplex` is self-adjoint in its weights, so it repeats them.
# `FoldedReal` has no forward weight, but its embedding `G[-m',-m] = conj(G[m',m])` is linear over the
# reals only, so its transpose takes a factor 2 on every `m' > 0` row; the `m' = 0` row is stored whole
# and is its own conjugate, so it keeps weight 1.
_fold_weights_adjoint!(G, plan::SpinNUSHTplan) = _fold_weights_adjoint!(G, plan, _fold_kind(plan))
_fold_weights_adjoint!(G, ::SpinNUSHTplan, ::FullModes) = G
_fold_weights_adjoint!(G, plan::SpinNUSHTplan, k::FoldedComplex) = _fold_weights!(G, plan, k)
function _fold_weights_adjoint!(G, ::SpinNUSHTplan{T}, ::FoldedReal) where {T}
    @views G[2:end, :, :] .*= T(2)
    return G
end

coefficient_size(plan::SpinNUSHTplan) = (plan.lmax + 1, 2plan.lmax + 1, plan.B)
allocate_coefficients(plan::SpinNUSHTplan{T}) where {T} =
    _zeros_like(plan.G, Complex{T}, coefficient_size(plan)...)

Base.show(io::IO, plan::SpinNUSHTplan{T}) where {T} =
    print(io, "SpinNUSHTplan{", T, "}(lmax=", plan.lmax, ", s=", plan.s, ", M=", length(_θnodes(plan)), ", B=", plan.B, ")")

"""
    _wigner_boundary(T, lmax) -> Matrix{T}

The `m = ℓ` column of every degree's Wigner-d(π/2) plane, `bnd[n+1, ℓ+1] = Δ^ℓ_{n,ℓ}` for
`0 ≤ n ≤ ℓ ≤ lmax`, from the degree below by Trapani & Navaza (2006) Eqs. 9 and 10. A plane follows from
its own column ([`_wigner_quarter!`](@ref)), so the degrees of a transform are independent.

The storage is `n`-major, `Δ^ℓ_{n,m} = d^ℓ_{m,n}(π/2)` (McEwen & Wiaux 2011, transposed), the order the
contraction reads `Δ^ℓ_{m',m}`.
"""
function _wigner_boundary(::Type{T}, lmax::Integer) where {T}
    bnd = zeros(T, lmax + 1, lmax + 1)
    bnd[1, 1] = one(T)
    @inbounds for ℓ in 1:lmax
        bnd[1, ℓ + 1] = -sqrt(T(2ℓ - 1) / T(2ℓ)) * bnd[1, ℓ]
        @simd for n in 1:ℓ
            bnd[n + 1, ℓ + 1] = sqrt(T(ℓ) * T(2ℓ - 1) / (T(2) * T(ℓ + n) * T(ℓ + n - 1))) * bnd[n, ℓ]
        end
    end
    return bnd
end

"""
    _wigner_quarter!(P, k, bnd, ℓ) -> P

Plane `k` of `P` becomes the `0 ≤ n, m ≤ ℓ` quarter of the degree-`ℓ` plane, `P[n+1, m+1, k] = Δ^ℓ_{n,m}`:
column `ℓ` from [`_wigner_boundary`](@ref), the downward three-term recurrence in `m` (Trapani & Navaza
2006, Eq. 11) on the eighth `n ≤ m`, each `n` independent, then `Δ_{n,m} = (−1)^{m+n} Δ_{m,n}` below the
diagonal. [`_quarter_sign`](@ref) gives the other three quarters. O(ℓ²) work, numerically stable to
`ℓ ≈ 1024` in double precision.
"""
function _wigner_quarter!(P::AbstractArray{T,3}, k::Integer, bnd::AbstractMatrix{T}, ℓ::Integer) where {T}
    @inbounds begin
        @simd for n in 0:ℓ
            P[n + 1, ℓ + 1, k] = bnd[n + 1, ℓ + 1]
        end
        # The two square roots depend on (ℓ, m) alone, so the `n` loop is an axpy.
        for m in (ℓ - 1):-1:0
            c1 = T(2) / sqrt(T(ℓ - m) * T(ℓ + m + 1))
            if m == ℓ - 1                                  # second term vanishes
                @simd for n in 0:m
                    P[n + 1, m + 1, k] = c1 * T(n) * P[n + 1, m + 2, k]
                end
            else
                s2 = sqrt(T(ℓ - m - 1) * T(ℓ + m + 2) / (T(ℓ - m) * T(ℓ + m + 1)))
                @simd for n in 0:m
                    P[n + 1, m + 1, k] = c1 * T(n) * P[n + 1, m + 2, k] - s2 * P[n + 1, m + 3, k]
                end
            end
        end
        for m in 0:ℓ, n in (m + 1):ℓ
            P[n + 1, m + 1, k] = _parity(T, m + n) * P[m + 1, n + 1, k]
        end
    end
    return P
end

@inline _parity(::Type{T}, k::Integer) where {T} = ifelse(iseven(k), one(T), -one(T))

"""
    _quarter_sign(T, ℓ, n, m) -> ±1

`Δ^ℓ_{n,m} = _quarter_sign(T, ℓ, n, m) · Δ^ℓ_{|n|,|m|}`, from the symmetries of `d(π/2)`: a factor
`(−1)^{ℓ+|n|}` for `m < 0` and `(−1)^{ℓ+|m|}` for `n < 0`.
"""
@inline _quarter_sign(::Type{T}, ℓ::Int, n::Int, m::Int) where {T} =
    ifelse(m < 0, _parity(T, ℓ + abs(n)), one(T)) * ifelse(n < 0, _parity(T, ℓ + abs(m)), one(T))

# `Δ^ℓ_{mp,m} Δ^ℓ_{mp,−s}` from quarter plane `k` of `P`. The signs are exact, so this is the product of
# the two full-plane entries.
@inline function _qprod(P, k::Integer, ::Type{T}, ℓ::Int, mp::Int, m::Int, s::Int) where {T}
    p = abs(mp) + 1
    σ = _quarter_sign(T, ℓ, mp, m) * _quarter_sign(T, ℓ, mp, -s)
    return @inbounds σ * (P[p, abs(m) + 1, k] * P[p, abs(s) + 1, k])
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
    _fold_weights!(_assemble_G_impl!(G, sf, plan, plan.engine), plan)

# Work, in fused multiply-adds over orders × degrees × batch, below which an assembly runs on one
# task, where a `@sync` barrier outweighs the split.
const _CONTRACT_MIN_WORK = 30_000

# Tasks one assembly runs on: the plan's, fewer when the work is too small to split.
@inline _spin_tasks(plan::SpinNUSHTplan) =
    min(plan.ntasks, max(1, ((2plan.lmax + 1)^2 * plan.lmax * plan.B) ÷ _CONTRACT_MIN_WORK))

# `f(k)` for `k in 1:n`, one task each; `n = 1` runs on the caller and spawns nothing.
@inline function _on_tasks(f::F, n::Int) where {F}
    n <= 1 && return (f(1); nothing)
    @sync for k in 1:n
        Threads.@spawn f(k)
    end
    return nothing
end

# With a table there is no recurrence to sequence, so the orders split over the whole range under one
# `@sync` for the entire assembly. Each order owns its column of `G` and its entries of `sf`, so the
# ranges need no synchronisation; one task spawns nothing, so the transform allocates nothing.
@inline function _thread_all_orders(f::F, nt::Int, lmax::Int) where {F}
    nord = 2lmax + 1
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

# Precomputed path: Δ is already contracted into Q, so ℓ runs innermost and each output column of G
# stays in cache across the degree sum.
function _assemble_G_impl!(G, sf, plan::SpinNUSHTplan{T}, tbl::WignerTable) where {T}
    _check_table(tbl, plan)
    lmax = plan.lmax; s = plan.s; off = lmax + 1
    fill!(G, zero(Complex{T}))
    _thread_all_orders(_spin_tasks(plan), lmax) do ms
        _table_orders!(G, sf, plan, tbl, ms, s, off, lmax)
    end
    return G
end

# Recurrence path. The degrees run in blocks of `nt`: the block's quarter planes are built one per
# task, then contracted on the same tasks, task `t` taking the orders `±c` for `c ≡ t−1 (mod nt)`. Each
# entry of `G` sums its degrees in ascending order whatever `nt` is. A block holds no more degrees than
# `eng` has planes.
@inline _block_size(plan::SpinNUSHTplan, eng::_WignerPlanes) = min(_spin_tasks(plan), size(eng.planes, 3))

function _assemble_G_impl!(G, sf, plan::SpinNUSHTplan{T}, eng::_WignerPlanes) where {T}
    fill!(G, zero(Complex{T}))
    nt = _block_size(plan, eng)
    ℓ0 = abs(plan.s)
    while ℓ0 <= plan.lmax
        ℓ1 = min(plan.lmax, ℓ0 + nt - 1)
        _forward_block!(G, sf, plan, eng, ℓ0, ℓ1, nt)
        ℓ0 = ℓ1 + 1
    end
    return G
end

# Contraction work of a block, in multiply-adds, below which it runs on the calling task.
const _BLOCK_MIN_WORK = 50_000

@inline _block_tasks(plan::SpinNUSHTplan, nt::Int, ℓ0::Int, ℓ1::Int) =
    (ℓ1 - ℓ0 + 1) * (2ℓ1 + 1)^2 * plan.B >= _BLOCK_MIN_WORK ? nt : 1

function _plane_block!(eng::_WignerPlanes, ℓ0::Int, ℓ1::Int, nt::Int)
    if nt == 1
        for k in 1:(ℓ1 - ℓ0 + 1)
            _wigner_quarter!(eng.planes, k, eng.bnd, ℓ0 + k - 1)
        end
    else
        _on_tasks(ℓ1 - ℓ0 + 1) do k
            _wigner_quarter!(eng.planes, k, eng.bnd, ℓ0 + k - 1)
        end
    end
    return eng.planes
end

function _forward_block!(G, sf, plan::SpinNUSHTplan, eng::_WignerPlanes, ℓ0::Int, ℓ1::Int, nt::Int)
    ntb = _block_tasks(plan, nt, ℓ0, ℓ1)
    P = _plane_block!(eng, ℓ0, ℓ1, ntb)
    _on_tasks(ntb) do t
        _contract_block!(G, sf, plan, P, (t - 1):ntb:ℓ1, ℓ0, ℓ1)
    end
    return G
end

# Orders `±c` for `c ∈ cs` over the block's degrees, from the block's quarter planes `P`: `m` outermost,
# so `i^{m+s}` is formed once per order; `p` innermost, down contiguous columns of `G` and of each plane.
function _contract_block!(G, sf, plan::SpinNUSHTplan{T}, P, cs, ℓ0::Int, ℓ1::Int) where {T}
    lmax = plan.lmax; s = plan.s; off = lmax + 1
    @inbounds for c in cs, m in -c:max(2c, 1):c
        ph0 = _im_pow(Complex{T}, -(m + s))
        for b in 1:plan.B, ℓ in max(ℓ0, c):ℓ1
            val = sf[ℓ + 1, m + lmax + 1, b]
            iszero(val) && continue
            _add_column!(G, plan, P, ℓ - ℓ0 + 1, ph0 * T(_Nℓ(ℓ)) * val, ℓ, m, m + off, b)
        end
    end
    return G
end

# For order `m` at degree `ℓ`, `Δ^ℓ_{±p,m} Δ^ℓ_{±p,−s} = σ(±p) w(p)` with `w(p) = Δ^ℓ_{p,|m|} Δ^ℓ_{p,|s|}`
# from quarter plane `k`: `σ(p) = (−1)^{ℓ+p}` when exactly one of `m < 0`, `s > 0` holds and `1`
# otherwise, and `σ(−p) = (−1)^{|m|+|s|} σ(p)` ([`_quarter_sign`](@ref)). So `p ≥ 0` reads each plane
# entry once for both `±p`.
@inline _row_sign(::Type{T}, flip::Bool, ℓ::Int, p::Int) where {T} =
    ifelse(flip, _parity(T, ℓ + p), one(T))

# `G[m', m] += ph Δ^ℓ_{m',m} Δ^ℓ_{m',−s}` down column `mcol` of `G`, from quarter plane `k` of `P`.
@inline function _add_column!(G, plan::SpinNUSHTplan{T}, P, k::Int, ph, ℓ::Int, m::Int, mcol::Int,
                              b::Int) where {T}
    s = plan.s
    am = abs(m) + 1; as = abs(s) + 1
    flip = (m < 0) != (s > 0)
    @inbounds begin
        @simd for p in 0:ℓ
            G[_grow(plan, p), mcol, b] += ph * (_row_sign(T, flip, ℓ, p) * (P[p + 1, am, k] * P[p + 1, as, k]))
        end
        if _fold_kind(plan) isa FullModes
            c = _parity(T, abs(m) + abs(s))
            @simd for p in 1:ℓ
                G[_grow(plan, -p), mcol, b] +=
                    ph * ((c * _row_sign(T, flip, ℓ, p)) * (P[p + 1, am, k] * P[p + 1, as, k]))
            end
        end
    end
    return G
end

# `Σ_{m'} Δ^ℓ_{m',m} Δ^ℓ_{m',−s} Ĝ[m', m]` down column `mcol` of `Ĝ`, from quarter plane `k` of `P`,
# each plane entry read once for both rows `±p`.
@inline function _reduce_column(Ĝ, plan::SpinNUSHTplan{T}, P, k::Int, ℓ::Int, m::Int, mcol::Int,
                                b::Int) where {T}
    s = plan.s
    am = abs(m) + 1; as = abs(s) + 1
    flip = (m < 0) != (s > 0)
    @inbounds begin
        acc = (_row_sign(T, flip, ℓ, 0) * (P[1, am, k] * P[1, as, k])) * Ĝ[_grow(plan, 0), mcol, b]
        if _fold_kind(plan) isa FullModes
            c = _parity(T, abs(m) + abs(s))
            @simd for p in 1:ℓ
                acc += (_row_sign(T, flip, ℓ, p) * (P[p + 1, am, k] * P[p + 1, as, k])) *
                       (Ĝ[_grow(plan, p), mcol, b] + c * Ĝ[_grow(plan, -p), mcol, b])
            end
        else
            @simd for p in 1:ℓ
                acc += (_row_sign(T, flip, ℓ, p) * (P[p + 1, am, k] * P[p + 1, as, k])) *
                       Ĝ[_grow(plan, p), mcol, b]
            end
        end
    end
    return acc
end

# Adjoint of _assemble_G!: contract NUFFT-type1 modes Ĝ back to coefficients, in place.
# The fold weights are a real diagonal, hence self-adjoint, so the adjoint applies the same ones —
# before the contraction here, after it in `_assemble_G!`.
_assemble_G_adjoint!(sf, Ĝ, plan::SpinNUSHTplan) =
    _assemble_G_adjoint_impl!(sf, _fold_weights_adjoint!(Ĝ, plan), plan, plan.engine)

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
    _thread_all_orders(_spin_tasks(plan), lmax) do ms
        _table_orders_adj!(sf, Ĝ, plan, tbl, ms, s, off, lmax)
    end
    return sf
end

# Adjoint counterpart of `_assemble_G_impl!`: each order reduces over `m'` into its own `sf` entries.
function _assemble_G_adjoint_impl!(sf, Ĝ, plan::SpinNUSHTplan{T}, eng::_WignerPlanes) where {T}
    fill!(sf, zero(Complex{T}))
    nt = _block_size(plan, eng)
    ℓ0 = abs(plan.s)
    while ℓ0 <= plan.lmax
        ℓ1 = min(plan.lmax, ℓ0 + nt - 1)
        _adjoint_block!(sf, Ĝ, plan, eng, ℓ0, ℓ1, nt)
        ℓ0 = ℓ1 + 1
    end
    return sf
end

function _adjoint_block!(sf, Ĝ, plan::SpinNUSHTplan, eng::_WignerPlanes, ℓ0::Int, ℓ1::Int, nt::Int)
    ntb = _block_tasks(plan, nt, ℓ0, ℓ1)
    P = _plane_block!(eng, ℓ0, ℓ1, ntb)
    _on_tasks(ntb) do t
        _contract_block_adj!(sf, Ĝ, plan, P, (t - 1):ntb:ℓ1, ℓ0, ℓ1)
    end
    return sf
end

function _contract_block_adj!(sf, Ĝ, plan::SpinNUSHTplan{T}, P, cs, ℓ0::Int, ℓ1::Int) where {T}
    lmax = plan.lmax; s = plan.s; off = lmax + 1
    @inbounds for c in cs, m in -c:max(2c, 1):c
        phc = conj(_im_pow(Complex{T}, -(m + s)))
        for b in 1:plan.B, ℓ in max(ℓ0, c):ℓ1
            sf[ℓ + 1, m + lmax + 1, b] =
                phc * T(_Nℓ(ℓ)) * _reduce_column(Ĝ, plan, P, ℓ - ℓ0 + 1, ℓ, m, m + off, b)
        end
    end
    return sf
end

# ── FastTransforms path ───────────────────────────────────────────────────────
# One coefficient column at a time through `plan_spinsph2fourier`, in the layout and sign of the file
# header, then the cosine and sine series as the coefficients of `e^{−im'θ}`.

# FastTransforms' column for order `m`, and the sign `σ_{ms}` of its harmonic `σ_{ms} ₛY_ℓm`.
@inline _ft_column(m::Int) = m >= 0 ? 2m + 1 : -2m
@inline _ft_sign(::Type{T}, m::Int, s::Int) where {T} = m + s > 0 ? _parity(T, m + s) : one(T)
@inline _times_i(z::Complex) = complex(-imag(z), real(z))

# Column `b` of `sf` into FastTransforms' layout: degree `ℓ` at row `ℓ − max(|m|,|s|) + 1`, the rows past
# `lmax` zero.
function _ft_pack!(sl, sf, lmax::Int, s::Int, b::Int)
    n = lmax + 1
    @inbounds for m in -lmax:lmax
        c = _ft_column(m); j = m + lmax + 1
        ℓ0 = max(abs(m), abs(s)); σ = _ft_sign(Float64, m, s)
        nv = max(n - ℓ0, 0)
        for r in 1:nv
            sl[r, c] = σ * sf[ℓ0 + r, j, b]
        end
        for r in (nv + 1):n
            sl[r, c] = zero(eltype(sl))
        end
    end
    return sl
end

# The transpose of `_ft_pack!`, with the slots of degree `ℓ < max(|m|,|s|)` zero.
function _ft_unpack!(sf, sl, lmax::Int, s::Int, b::Int)
    @inbounds for m in -lmax:lmax
        c = _ft_column(m); j = m + lmax + 1
        ℓ0 = max(abs(m), abs(s)); σ = _ft_sign(Float64, m, s)
        for ℓ in 0:(min(ℓ0, lmax + 1) - 1)
            sf[ℓ + 1, j, b] = zero(eltype(sf))
        end
        for ℓ in ℓ0:lmax
            sf[ℓ + 1, j, b] = σ * sl[ℓ - ℓ0 + 1, c]
        end
    end
    return sf
end

# Column `b` of `G` from FastTransforms' series: `cos(kθ) = (e^{ikθ} + e^{−ikθ})/2` and
# `sin(kθ) = i(e^{−ikθ} − e^{ikθ})/2`, each scaled by `(2π)^{−1/2}`. Every row of the layout is written.
function _ft_modes!(G, sl, plan::SpinNUSHTplan{T}, kind, b::Int) where {T}
    lmax = plan.lmax; s = plan.s
    κ = one(T) / sqrt(2 * T(π)); h = κ / 2
    full = kind isa FullModes
    @inbounds for m in -lmax:lmax
        c = _ft_column(m); j = m + lmax + 1
        if iseven(m + s)
            G[_grow(plan, 0, kind), j, b] = κ * sl[1, c]
            for k in 1:lmax
                v = h * sl[k + 1, c]
                G[_grow(plan, k, kind), j, b] = v
                full && (G[_grow(plan, -k, kind), j, b] = v)
            end
        else
            G[_grow(plan, 0, kind), j, b] = zero(Complex{T})
            for k in 1:lmax
                v = _times_i(h * sl[k, c])
                G[_grow(plan, k, kind), j, b] = v
                full && (G[_grow(plan, -k, kind), j, b] = -v)
            end
        end
    end
    return G
end

# The transpose of `_ft_modes!`: a cosine row takes `h (Ĝ[k] + Ĝ[−k])`, a sine row `−i h (Ĝ[k] − Ĝ[−k])`,
# a folded layout's absent rows `−k` count as zero, and the sine row of frequency `lmax + 1` is zero.
function _ft_fourier!(sl, Ĝ, plan::SpinNUSHTplan{T}, kind, b::Int) where {T}
    lmax = plan.lmax; s = plan.s
    κ = one(T) / sqrt(2 * T(π)); h = κ / 2
    full = kind isa FullModes
    @inbounds for m in -lmax:lmax
        c = _ft_column(m); j = m + lmax + 1
        if iseven(m + s)
            sl[1, c] = κ * Ĝ[_grow(plan, 0, kind), j, b]
            for k in 1:lmax
                g = Ĝ[_grow(plan, k, kind), j, b]
                full && (g += Ĝ[_grow(plan, -k, kind), j, b])
                sl[k + 1, c] = h * g
            end
        else
            for k in 1:lmax
                g = Ĝ[_grow(plan, k, kind), j, b]
                full && (g -= Ĝ[_grow(plan, -k, kind), j, b])
                sl[k, c] = -_times_i(h * g)
            end
            sl[lmax + 1, c] = zero(eltype(sl))
        end
    end
    return sl
end

function _ft_synthesis_column!(G, sf, plan::SpinNUSHTplan, kind, sl, P, b::Int)
    _ft_pack!(sl, sf, plan.lmax, plan.s, b)
    _ft_lmul!(P, sl, plan)
    return _ft_modes!(G, sl, plan, kind, b)
end

function _ft_adjoint_column!(sf, Ĝ, plan::SpinNUSHTplan, kind, sl, Padj, b::Int)
    _ft_fourier!(sl, Ĝ, plan, kind, b)
    _ft_lmul!(Padj, sl, plan)
    return _ft_unpack!(sf, sl, plan.lmax, plan.s, b)
end

# The batch's columns on the engine's tasks (`_pool_columns!`), each writing its own column of the
# output.
function _assemble_G_impl!(G, sf, plan::SpinNUSHTplan, eng::_FTSpinEngine)
    kind = _fold_kind(plan)
    _pool_columns!((sl, P, _, b) -> _ft_synthesis_column!(G, sf, plan, kind, sl, P, b),
                   (eng.slice, eng.plan, eng.adj), eng.pool, plan.B)
    return G
end

function _assemble_G_adjoint_impl!(sf, Ĝ, plan::SpinNUSHTplan, eng::_FTSpinEngine)
    kind = _fold_kind(plan)
    _pool_columns!((sl, _, Padj, b) -> _ft_adjoint_column!(sf, Ĝ, plan, kind, sl, Padj, b),
                   (eng.slice, eng.plan, eng.adj), eng.pool, plan.B)
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

Exact Euclidean adjoint of `nusht_type2_spin!`: scattered values `f` → spin-`s` coefficients `sf`, with
no quadrature weights; [`nusht_analyze!`](@ref) gives the coefficients of `f`.
"""
function nusht_type1_spin!(sf, f, plan::SpinNUSHTplan{T}) where {T}
    _assert_spin_field(f, plan)
    _assert_spin_coeffs(sf, plan)
    copyto!(_fbuf(plan), f)
    return _strengths_adjoint!(sf, plan)
end

# `A†` applied to the plan's strengths buffer, into `sf`.
function _strengths_adjoint!(sf, plan::SpinNUSHTplan)
    _rephase_conj!(_fbuf(plan), _θshift(plan))
    _nufft_exec!(_nufft1(plan), _fbuf(plan), plan.G)
    _assemble_G_adjoint!(sf, plan.G, plan)
    return sf
end

# `Σⱼ wⱼ fⱼ conj(ₛY_ℓm(xⱼ))`. On a real-field plan the adjoint is taken over the reals, and its Hermitian
# part is the complex adjoint of the real data: for every Hermitian `x`, `Re⟨x, A†y⟩ = ⟨Ax, y⟩`.
function _quadrature!(sf, f, w, plan::SpinNUSHTplan)
    _check_spin_solvable(plan)
    _assert_spin_field(f, plan)
    _assert_spin_coeffs(sf, plan)
    _assert_weights(w, plan)
    _copy_weighted!(_fbuf(plan), f, w)
    _strengths_adjoint!(sf, plan)
    return _hermitian_part!(sf, plan)
end

# The orthogonal projection onto the coefficients of a real spin-0 field,
# `sf[ℓ,-m] = (-1)^m conj(sf[ℓ,m])`, in place; the identity on a complex-field plan.
_hermitian_part!(sf, plan::SpinNUSHTplan) = _hermitian_part!(sf, plan, _fold_kind(plan))
_hermitian_part!(sf, ::SpinNUSHTplan, ::FullModes) = sf
function _hermitian_part!(sf, plan::SpinNUSHTplan, ::Union{FoldedComplex,FoldedReal})
    lmax = plan.lmax
    for b in 1:plan.B, ℓ in 0:lmax, m in 0:ℓ
        _hermitian_pair!(sf, ℓ, m, lmax, b)
    end
    return sf
end

# The pair `(ℓ, ±m)`, `m ≥ 0`, of column `b` replaced by its Hermitian part; `m = 0` keeps the real part.
# One call per pair, so the host loop and the device kernel evaluate the same expression.
@inline function _hermitian_pair!(sf, ℓ::Int, m::Int, lmax::Int, b::Int)
    T = real(eltype(sf))
    i = ℓ + 1
    @inbounds if m == 0
        sf[i, lmax + 1, b] = real(sf[i, lmax + 1, b])
    else
        σ = _parity(T, m)
        h = (sf[i, lmax + 1 + m, b] + σ * conj(sf[i, lmax + 1 - m, b])) / T(2)
        sf[i, lmax + 1 + m, b] = h
        sf[i, lmax + 1 - m, b] = σ * conj(h)
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Exact inversion: the plan as the operator of FlowTransformBindings' LSMR, as in NUFSHT.jl
# ─────────────────────────────────────────────────────────────────────────────
# A complex field's fit runs over the dense `(lmax+1, 2lmax+1, B)` coefficient array. A real field's
# runs over the `(lmax+1)²` real degrees its coefficients have (see `_hermitian_domain`), packed, and
# its iterate is expanded into `solve_w` before assembly. The adjoint lands in `solve_w` in the full
# layout either way, and a real field's packing of it is the restriction to the Hermitian subspace.
# The transform runs at the full width `B` whatever the live column count.

# The full-layout coefficient buffer of the fit, built by the first solve and kept on the plan.
@inline function _solve_w(plan::SpinNUSHTplan)
    w = plan.solve_w[]
    w === nothing || return w
    return _build_solve_w!(plan)
end
@noinline _build_solve_w!(plan::SpinNUSHTplan{T}) where {T} =
    (plan.solve_w[] = _zeros_like(plan.G, Complex{T}, coefficient_size(plan)...))

FTB.lsmr_allocate_domain(plan::SpinNUSHTplan{T}) where {T} =
    (_solve_w(plan); _hermitian_domain(plan) ? _zeros_like(plan.G, T, _herm_len(plan.lmax), plan.B) :
                                               _zeros_like(plan.G, Complex{T}, coefficient_size(plan)...))
FTB.lsmr_allocate_range(plan::SpinNUSHTplan) =
    _zeros_like(_fbuf(plan), eltype(_fbuf(plan)), _spin_npts(plan), plan.B)
FTB.lsmr_check_solution(sf, plan::SpinNUSHTplan, ws) = _assert_spin_coeffs(sf, plan)

# The caller's `sf` is the full complex array whatever the iterate is, so a packed solution expands on
# the way out.
FTB.lsmr_write!(sf, plan::SpinNUSHTplan, x, k, j) = _write_solution!(sf, x, plan, k, j, _fold_kind(plan))
_write_solution!(C, x, plan::SpinNUSHTplan, slot::Integer, dstcol::Integer, ::FullModes) =
    FTB.colcopy!(C, dstcol, x, slot, plan.B)
function _write_solution!(C, x, plan::SpinNUSHTplan, slot::Integer, dstcol::Integer,
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
        # `spin_coeff_index` is Cartesian, so the destination column is a trailing index, which also
        # serves a 2-D `C` at `B = 1`, its trailing axis being singleton.
        for ℓ in 0:lmax
            o = _herm_offset(ℓ)
            C[spin_coeff_index(ℓ, 0, lmax), dstcol] = x[so + o + 1]
            for m in 1:ℓ
                a = complex(x[so + o + 2m], x[so + o + 2m + 1]) * s2
                C[spin_coeff_index(ℓ,  m, lmax), dstcol] = a
                C[spin_coeff_index(ℓ, -m, lmax), dstcol] =
                    ifelse(iseven(m), one(T), -one(T)) * conj(a)
            end
        end
    end
    return C
end

_add_out!(f, fbuf, plan::SpinNUSHTplan, n) = _add_out!(f, fbuf, _θshift(plan), n)
_add_out!(f, fbuf, ::Nothing, n) = _add_field!(f, fbuf, n)
_add_out!(f, fbuf, ::AbstractVector, n) = _add_real!(f, fbuf, n)

"""
    _hermitian_domain(plan) -> Bool

Whether the plan's field is real, and so whether its coefficients are constrained.

A real field's coefficients are Hermitian, `sf[ℓ,-m] = (-1)^m conj(sf[ℓ,m])` from
`conj(Y_ℓm) = (-1)^m Y_ℓ,-m`: `(lmax+1)²` real numbers, of the `(lmax+1)(2lmax+1)` complex ones the
array holds. A solve runs on that subspace. Off it the operator's values are set by the backend's fold,
so a fit over the whole array has no unique answer.
"""
@inline _hermitian_domain(plan::SpinNUSHTplan) = _hermitian_domain(_fold_kind(plan))
@inline _hermitian_domain(::FullModes) = false
@inline _hermitian_domain(::Union{FoldedComplex,FoldedReal}) = true

# Packed offset of degree `ℓ`: the degrees below it occupy `Σ (2ℓ'+1) = ℓ²` slots. `m = 0` is one real
# number, each `m > 0` a real and an imaginary part.
@inline _herm_offset(ℓ::Integer) = ℓ * ℓ
@inline _herm_len(lmax::Integer) = (lmax + 1)^2

"""
    _herm_entry(CT, p, ℓ, m, po) -> CT

The entry `_unpack_herm!` writes at degree `ℓ`, order `m`, from the packed column starting after linear
offset `po` of `p`: zero where `|m| > ℓ`, the real value at `m = 0`, `(p[2m] + i p[2m+1])/√2` for
`m > 0` and `(-1)^m` times its conjugate for `m < 0`. One per output entry, so the host loop and the
device kernel evaluate the same expression.
"""
@inline function _herm_entry(::Type{CT}, p, ℓ::Int, m::Int, po::Int) where {CT}
    abs(m) > ℓ && return zero(CT)
    T = real(CT)
    o = po + _herm_offset(ℓ)
    @inbounds begin
        m == 0 && return CT(p[o + 1])
        a = complex(p[o + 2abs(m)], p[o + 2abs(m) + 1]) * (one(T) / sqrt(T(2)))
        return CT(m > 0 ? a : ifelse(iseven(m), one(T), -one(T)) * conj(a))
    end
end

"""
    _pack_entry(g, t, lmax, b) -> real

Packed value `t` of column `b` under [`_pack_herm!`](@ref), read from the full array `g`. With
`ℓ = isqrt(t-1)` and `r = t - 1 - ℓ²`: the real part of `g[ℓ,0]` at `r = 0`, and for
`m = ⌈r/2⌉` the real (`r` odd) or imaginary (`r` even) part of `(g[ℓ,m] ± (-1)^m g[ℓ,-m])/√2`.
"""
@inline function _pack_entry(g, t::Int, lmax::Int, b::Int)
    T = real(eltype(g))
    ℓ = isqrt(t - 1)
    r = t - 1 - ℓ * ℓ
    @inbounds begin
        r == 0 && return real(g[ℓ + 1, lmax + 1, b])
        m = (r + 1) >> 1
        gp = g[ℓ + 1, m + lmax + 1, b]
        gm = g[ℓ + 1, lmax + 1 - m, b]
        sg = ifelse(iseven(m), one(T), -one(T))
        s2 = one(T) / sqrt(T(2))
        return isodd(r) ? (real(gp) + sg * real(gm)) * s2 : (imag(gp) - sg * imag(gm)) * s2
    end
end

# `p ← v` without `β`, `p ← v + β[b]·p` with it.
@inline _fold_packed(v, ::Nothing, p, i, b) = v
@inline _fold_packed(v, β, p, i, b) = @inbounds v + β[b] * p[i]

"""
    _unpack_herm!(sf, p, lmax, B) -> sf

Expand packed real coefficients into the full Hermitian array. Scaled by `1/√2` off `m = 0` so the map
is an **isometry**: `‖U p‖ = ‖p‖`, so a minimum-norm solution is minimum-norm on both sides of it.
[`_pack_herm!`](@ref) is its exact adjoint.
"""
function _unpack_herm!(sf, p, lmax::Integer, B::Integer)
    K = _herm_len(lmax)
    @inbounds for b in 1:B, j in 1:(2lmax + 1), i in 1:(lmax + 1)
        sf[i, j, b] = _herm_entry(eltype(sf), p, i - 1, j - 1 - lmax, (b - 1) * K)
    end
    return sf
end

"""
    _pack_herm!(p, g, lmax, B, β = nothing) -> p

The exact adjoint of [`_unpack_herm!`](@ref) under the real inner product `Re⟨a,b⟩`. With `β` given it
also folds `p ← U†g + β·p`, the packed counterpart of `FTB.colxpby!`, in the same pass.
"""
function _pack_herm!(p, g, lmax::Integer, B::Integer, β = nothing)
    K = _herm_len(lmax)
    @inbounds for b in 1:B, t in 1:K
        i = (b - 1) * K + t
        p[i] = _fold_packed(_pack_entry(g, t, lmax, b), β, p, i, b)
    end
    return p
end

# `u ← A v + c u`, the synthesis landing in the plan's strengths buffer and added to the scaled `u`.
# `solve_w` is free from the adjoint's fold to here, so a packed iterate expands into it.
function FTB.lsmr_forward!(u, plan::SpinNUSHTplan, v, c, n)
    FTB.colscale!(u, c, n, plan.B)
    _assemble_G!(plan.G, _av_coeffs(v, plan), plan)
    _nufft_exec!(_nufft2(plan), plan.G, _fbuf(plan))
    _rephase!(_fbuf(plan), _θshift(plan))
    return _add_out!(u, _fbuf(plan), plan, n)
end

@inline _av_coeffs(v, plan::SpinNUSHTplan) = _av_coeffs(v, plan, _fold_kind(plan))
@inline _av_coeffs(v, ::SpinNUSHTplan, ::FullModes) = v
@inline _av_coeffs(v, plan::SpinNUSHTplan, ::Union{FoldedComplex,FoldedReal}) =
    _unpack_herm!(_solve_w(plan), v, plan.lmax, plan.B)

_adjoint_full!(plan::SpinNUSHTplan, u, n::Integer) = nusht_type1_spin!(_solve_w(plan), u, plan)

# `v ← w + c v`, a real field's `w` packed onto the Hermitian subspace.
_fold!(v, plan::SpinNUSHTplan, w, c, n::Integer) = _fold!(v, plan, w, c, n, _fold_kind(plan))
_fold!(v, plan::SpinNUSHTplan, w, c, n::Integer, ::FullModes) = FTB.colxpby!(v, w, c, n, plan.B)
_fold!(v, plan::SpinNUSHTplan, w, c, n::Integer, ::Union{FoldedComplex,FoldedReal}) =
    _pack_herm!(v, w, plan.lmax, n, c)

"""
    nusht_solve_spin!(sf, f, plan; ws = FlowTransformBindings.LSMRWorkspace(plan), weights = nothing,
                      maxiter = 500, rtol = 1e-6, conlim = 0)
    nusht_solve_spin!(sf, f, FlowTransformBindings.RangeWeighted(plan, weights); ws, …)
    nusht_solve_spin!(sf, f_local, plan, execution; ws, weights = nothing, …)

Exact inversion of the spin-weighted synthesis at arbitrary scattered points: solve
`min ‖A sf − f‖` by LSMR on the Golub–Kahan bidiagonalization of `A`. Batched (`B > 1`) runs the
columns as independent single-column solves. Same contract, return and `weights` as
[`nusht_solve!`](@ref): `weights`, nonnegative and one per point, makes the fit
`min Σⱼ wⱼ |(A sf − f)ⱼ|²`, a zero weight leaving its point out, and the second form takes that weighted operator built once. The
four-argument form is the fit over the points of every process `execution` names, `plan`, `f_local` and
`weights` being this process's.

On a real-field plan the fit runs over the `(lmax+1)²` real degrees of a real field's coefficients
(`_hermitian_domain`), a quarter of the unknowns.

That subspace exists only at `s = 0`: conjugation maps spin `s` to `-s`
(`conj(ₛY_ℓm) = (-1)^{m+s} ₋ₛY_ℓ,-m`), so a real-field plan at `s ≠ 0` has no subspace on which the fit
is determined, and the solve throws. Its synthesis and adjoint are exact. [`nusht_analyze!`](@ref)
takes the same plans, with quadrature weights or by this fit.
"""
function nusht_solve_spin!(sf, f, plan::SpinNUSHTplan; ws::FTB.LSMRWorkspace = FTB.LSMRWorkspace(plan),
                           weights = nothing, kwargs...)
    _check_spin_solvable(plan)
    return _solve!(sf, f, _weighted(plan, weights), ws; kwargs...)
end

function nusht_solve_spin!(sf, f, op::FTB.RangeWeighted{<:SpinNUSHTplan};
                           ws::FTB.LSMRWorkspace = FTB.LSMRWorkspace(op), kwargs...)
    _check_spin_solvable(op.op)
    return _solve!(sf, f, op, ws; kwargs...)
end

function nusht_solve_spin!(sf, f_local, plan::SpinNUSHTplan, execution::ComputationalBackends.AbstractExecutionBackend;
                           ws::FTB.LSMRWorkspace = FTB.LSMRWorkspace(plan), weights = nothing, kwargs...)
    _check_spin_solvable(plan)
    return _solve!(sf, f_local, _weighted(_point_share(plan, execution), weights), ws; kwargs...)
end

# `nusht_analyze!` without weights on a spin plan: the fit.
_analyze!(sf, f, plan::SpinNUSHTplan, ::Nothing; kwargs...) = nusht_solve_spin!(sf, f, plan; kwargs...)
_analyze!(sf, f, plan::SpinNUSHTplan, ::Nothing, execution; kwargs...) =
    nusht_solve_spin!(sf, f, plan, execution; kwargs...)

function _check_spin_solvable(plan::SpinNUSHTplan)
    (_hermitian_domain(plan) && plan.s != 0) && throw(ArgumentError(
        "a spin-$(plan.s) field cannot be real — conjugation maps spin s to spin -s — so a " *
        "real-field plan at s ≠ 0 has no coefficient subspace on which a fit or an analysis is " *
        "determined. Build the plan with a complex element type to fit or analyse, or keep the real " *
        "one for synthesis and the adjoint, which are exact."))
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Scalar ↔ spin-0 coefficients
# ─────────────────────────────────────────────────────────────────────────────
# With `Λ_ℓm(θ) = √((2ℓ+1)/4π · (ℓ-m)!/(ℓ+m)!) P_ℓ^m(cos θ)`, `m ≥ 0`, the Condon–Shortley phase in
# `P_ℓ^m`: a scalar plan's harmonics are FastTransforms' real ones, `Y_ℓ0 = Λ_ℓ0`,
# `Y_ℓm = √2 (-1)^m Λ_ℓm cos(mφ)` and `Y_ℓ,-m = √2 (-1)^m Λ_ℓm sin(mφ)` at `sph_mode(ℓ, ±m)`; a spin
# plan's at `s = 0` are `₀Y_ℓm = N_ℓ d^ℓ_{m0}(θ) e^{imφ}`, which is `Λ_ℓm e^{imφ}`, and
# `₀Y_ℓ,-m = (-1)^m Λ_ℓm e^{-imφ}`. Writing the cosine and sine as exponentials, for `m > 0`:
#
#     s_ℓ0 = C_ℓ0,   s_ℓm = (-1)^m (C_ℓm - i C_ℓ,-m)/√2,   s_ℓ,-m = (C_ℓm + i C_ℓ,-m)/√2,
#     C_ℓm = ((-1)^m s_ℓm + s_ℓ,-m)/√2,   C_ℓ,-m = i ((-1)^m s_ℓm - s_ℓ,-m)/√2.

"""
    scalar_to_spin0!(sf, C) -> sf

The spin-0 coefficients of the field with scalar coefficients `C`, written into `sf`: `C` in a scalar
plan's `sph_mode` layout, `sf` complex in a spin plan's [`spin_coeff_index`](@ref) layout, both
`(lmax+1, 2lmax+1[, B])`. A spin-0 plan synthesizes from `sf` the field a scalar plan synthesizes from
`C`; the slots of `C` above degree `lmax` are not read.
"""
function scalar_to_spin0!(sf, C)
    lmax, B = _check_spin0_pair(sf, C)
    fill!(sf, zero(eltype(sf)))
    return _scalar_to_spin0_pairs!(sf, C, lmax, B)
end

"""
    spin0_to_scalar!(C, sf) -> C

The inverse of [`scalar_to_spin0!`](@ref): the scalar coefficients of the field with spin-0 coefficients
`sf`, written into `C`, its slots above degree `lmax` zeroed. A real `C` receives the coefficients of the
field's real part, the whole field when `sf` is Hermitian, `sf[ℓ,-m] = (-1)^m conj(sf[ℓ,m])`.
"""
function spin0_to_scalar!(C, sf)
    lmax, B = _check_spin0_pair(sf, C)
    fill!(C, zero(eltype(C)))
    return _spin0_to_scalar_pairs!(C, sf, lmax, B)
end

function _check_spin0_pair(sf, C)
    eltype(sf) <: Complex || throw(ArgumentError(
        "spin-0 coefficients are complex; got an array of $(eltype(sf)). Pass a complex array, such as " *
        "a spin plan's `allocate_coefficients`."))
    lmax = size(C, 1) - 1
    (lmax >= 0 && size(C, 2) == 2lmax + 1) || throw(DimensionMismatch(
        "a coefficient array is (lmax+1, 2lmax+1[, B]); got size $(size(C))"))
    (size(sf, 1) == size(C, 1) && size(sf, 2) == size(C, 2) && length(sf) == length(C)) ||
        throw(DimensionMismatch("the spin-0 array has size $(size(sf)) and the scalar one $(size(C))"))
    return lmax, length(C) ÷ (size(C, 1) * size(C, 2))
end

# Host loops over the pairs `(ℓ, ±m)`, `m ≥ 0`; the device methods are kernels in the KernelAbstractions
# extension over the same per-pair functions.
function _scalar_to_spin0_pairs!(sf, C, lmax::Int, B::Int)
    for b in 1:B, ℓ in 0:lmax, m in 0:ℓ
        _scalar_pair_to_spin0!(sf, C, ℓ, m, lmax, b)
    end
    return sf
end

function _spin0_to_scalar_pairs!(C, sf, lmax::Int, B::Int)
    for b in 1:B, ℓ in 0:lmax, m in 0:ℓ
        _spin0_pair_to_scalar!(C, sf, ℓ, m, lmax, b)
    end
    return C
end

@inline function _scalar_pair_to_spin0!(sf, C, ℓ::Int, m::Int, lmax::Int, b::Int)
    CT = eltype(sf)
    T = real(CT)
    @inbounds if m == 0
        sf[ℓ + 1, lmax + 1, b] = CT(C[ℓ + 1, 1, b])
    else
        r = one(T) / sqrt(T(2))
        cp = CT(C[ℓ - m + 1, 2m + 1, b])              # sph_mode(ℓ, m)
        cm = CT(C[ℓ - m + 1, 2m, b])                  # sph_mode(ℓ, -m)
        sf[ℓ + 1, lmax + 1 + m, b] = (_parity(T, m) * r) * (cp - CT(0, 1) * cm)
        sf[ℓ + 1, lmax + 1 - m, b] = r * (cp + CT(0, 1) * cm)
    end
    return nothing
end

@inline function _spin0_pair_to_scalar!(C, sf, ℓ::Int, m::Int, lmax::Int, b::Int)
    E = eltype(C)
    CT = eltype(sf)
    T = real(CT)
    @inbounds if m == 0
        C[ℓ + 1, 1, b] = _modes_out(E, sf[ℓ + 1, lmax + 1, b])
    else
        r = one(T) / sqrt(T(2))
        sp = _parity(T, m) * sf[ℓ + 1, lmax + 1 + m, b]
        sm = sf[ℓ + 1, lmax + 1 - m, b]
        C[ℓ - m + 1, 2m + 1, b] = _modes_out(E, r * (sp + sm))
        C[ℓ - m + 1, 2m, b] = _modes_out(E, CT(0, 1) * r * (sp - sm))
    end
    return nothing
end
