"""
    TensorProduct()

Declare separate latitude and longitude axes for a spherical transform.

Longitudes sample one full circle at equal spacing, with the endpoint excluded.
Fields follow `(length(θ), length(φ), ntrans)` in column-major order.
"""
struct TensorProduct end

public TensorProduct

struct _LongitudeTransform{F,B}
    forward::F
    backward::B
end

struct _LongitudeSynthesis{B}
    backward::B
end

"""
    TensorProductSpinPlan{T}

A spin transform on separate latitude and longitude axes, built by [`make_spin_plan`](@ref) with
[`TensorProduct`](@ref), using latitude degree sums and longitude FFTs in precision `T`.
"""
struct TensorProductSpinPlan{T<:AbstractFloat,V,W,F,C,E,P,L} <: AbstractSpinNUSHTplan{T}
    lmax::Int
    s::Int
    B::Int
    ntasks::Int
    θ::V
    φ::W
    fbuf::F
    G::C
    engine::E
    phase::P
    longitude::L
    solve_w::C
end

public TensorProductSpinPlan

"""
    make_spin_plan(FE, θ, φ, lmax, s, TensorProduct(); ntrans = 1, execution = AutoBackend())

Build a spin transform using latitude degree sums and longitude FFTs.

`θ` contains colatitudes in `[0, π]`; `φ` is a range of at least `2lmax + 1` longitudes.
"""
function make_spin_plan(::Type{FE}, θ::AbstractVector, φ::AbstractRange, lmax::Integer, s::Integer,
                        ::TensorProduct;
                        ntrans::Integer = 1,
                        execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
                        nthreads::Union{Nothing,Integer} = nothing,
                        directions::AbstractPlanDirections = SynthesisAndAnalysis()) where {FE<:Number}
    return _tensor_product_plan(FE, θ, φ, Int(lmax), Int(s), execution, directions;
                                ntrans, nthreads)
end

make_spin_plan(θ::AbstractVector, φ::AbstractRange, lmax::Integer, s::Integer, grid::TensorProduct;
               kwargs...) = make_spin_plan(Complex{float(eltype(θ))}, θ, φ, lmax, s, grid; kwargs...)

"""
    make_plan(FE, θ, φ, lmax, TensorProduct(); kwargs...)

Build a scalar tensor-product transform in the existing scalar coefficient layout.
"""
function make_plan(::Type{FE}, θ::AbstractVector, φ::AbstractRange, lmax::Integer, grid::TensorProduct;
                   kwargs...) where {FE<:Number}
    spin = make_spin_plan(FE, θ, φ, lmax, 0, grid; kwargs...)
    sf = allocate_coefficients(spin)
    scratch = _zeros_like(sf, FE, coefficient_size(spin)...)
    tol = eps(real(FE))
    return RingNUSHTplan{real(FE),FE,typeof(spin),typeof(sf),typeof(scratch),typeof(tol)}(
        Int(lmax), spin.B, tol, spin, sf, Ref{Union{Nothing,typeof(scratch)}}(scratch))
end

make_plan(θ::AbstractVector, φ::AbstractRange, lmax::Integer, grid::TensorProduct; kwargs...) =
    make_plan(float(eltype(θ)), θ, φ, lmax, grid; kwargs...)

function _tensor_product_plan(::Type{FE}, θ, φ, lmax::Int, s::Int, execution::_HostExecution,
                              directions; ntrans, nthreads) where {FE}
    nt = _task_count(_plan_threads(execution, nthreads))
    return _build_tensor_product_plan(FE, θ, φ, lmax, s, nt, directions, Int(ntrans))
end

_tensor_product_plan(::Type, θ, φ, lmax, s, execution, directions; kwargs...) =
    _execution_unavailable(execution, "a tensor-product plan")

function _build_tensor_product_plan(::Type{FE}, θ, φ, lmax::Int, s::Int, nt::Int, directions,
                                    B::Int) where {FE}
    T = real(FE)
    T <: AbstractFloat || throw(ArgumentError("field element type must be floating point; got $FE"))
    lmax >= 0 || throw(ArgumentError("lmax must be nonnegative; got $lmax"))
    abs(s) <= lmax || throw(ArgumentError("spin $s exceeds lmax $lmax"))
    _check_ntrans(B)
    isempty(θ) && throw(ArgumentError("a tensor-product plan needs at least one latitude"))
    length(φ) >= 2lmax + 1 || throw(ArgumentError("longitude count must be at least 2lmax + 1"))
    nt >= 1 || throw(ArgumentError("nthreads must be positive; got $nt"))
    th = T.(θ)
    engine = _tensor_latitude(th, lmax, s, nt, B)
    phase = _to_like(th, Complex{T}[cis(T(m) * T(first(φ))) for m in -lmax:lmax])
    G = _zeros_like(th, Complex{T}, length(th), length(φ), B)
    longitude = _longitude_transform(directions, G, nt)
    fields = _zeros_like(th, FE, length(th) * length(φ), B)
    scratch = _zeros_like(th, Complex{T}, lmax + 1, 2lmax + 1, B)
    return TensorProductSpinPlan{T,typeof(th),typeof(φ),typeof(fields),typeof(G),typeof(engine),
                                 typeof(phase),typeof(longitude)}(
        lmax, s, B, nt, th, φ, fields, G, engine, phase, longitude, scratch)
end

function _longitude_transform(::SynthesisAndAnalysis, G, nt::Int)
    return _LongitudeTransform(_longitude_forward(G, nt), _longitude_backward(G, nt))
end

_longitude_transform(::SynthesisOnly, G, nt::Int) = _LongitudeSynthesis(_longitude_backward(G, nt))
_longitude_forward(G::Array, nt::Int) = FFTW.plan_fft!(G, 2; flags = FFTW.MEASURE, num_threads = nt)
_longitude_backward(G::Array, nt::Int) = FFTW.plan_bfft!(G, 2; flags = FFTW.MEASURE, num_threads = nt)
_longitude_forward(G::AbstractArray, ::Int) = FFTW.plan_fft!(G, 2)
_longitude_backward(G::AbstractArray, ::Int) = FFTW.plan_bfft!(G, 2)

_longitude_adjoint(longitude::_LongitudeTransform, G) = longitude.forward * G
_longitude_adjoint(::_LongitudeSynthesis, G) = throw(ArgumentError(
    "this plan was built with directions = SynthesisOnly(); analysis requires SynthesisAndAnalysis()"))

function _tensor_tables(::Type{R}, ::Type{S}, θ, lmax::Int, s::Int, padded::Int) where {R,S}
    u = zeros(R, padded); x = zeros(R, padded); xl = zeros(R, padded)
    pos = fill(false, padded)
    lc = fill(S(-Inf), padded); ls = fill(S(-Inf), padded)
    lanes = _ring_lanes(R)
    for j0 in 0:lanes:(padded - 1)
        side = θ[min(j0 + lanes ÷ 2 + 1, length(θ))] <= R(π) / 2
        pos[(j0 + 1):(j0 + lanes)] .= side
        for j in (j0 + 1):min(j0 + lanes, length(θ))
            zero(R) <= θ[j] <= R(π) || throw(ArgumentError("colatitudes must lie in [0, π]"))
            angle = θ[j] == R(π) ? BigFloat(π) : BigFloat(θ[j])
            sh, ch = sincos(angle / 2)
            θ[j] == R(π) && (ch = zero(ch))
            u[j] = R(2 * (side ? sh : ch)^2)
            cosine = cos(angle)
            x[j] = R(cosine); xl[j] = R(cosine - x[j])
            lc[j] = S(log2(ch)); ls[j] = S(log2(sh))
        end
    end
    starts = [FTB.wigner_d_start(S, m, -s) for m in -lmax:lmax]
    return (; u, pos, x, xl, lc, ls, lC = getfield.(starts, :lC), p1 = getfield.(starts, :p1),
            p2 = getfield.(starts, :p2), sg = getfield.(starts, :sg))
end

function _tensor_latitude(θ::AbstractVector{R}, lmax::Int, s::Int, nt::Int, ::Int) where {R}
    padded = cld(length(θ), _ring_lanes(R)) * _ring_lanes(R)
    t = _tensor_tables(R, promote_type(R, Float64), θ, lmax, s, padded)
    work = [_latitude_work(R, lmax, padded) for _ in 1:nt]
    return _RingEngine(length(θ), _ring_check_interval(R, lmax), t.u, t.pos, t.x, t.xl,
                       t.lc, t.ls, t.lC, t.p1, t.p2, t.sg, work)
end

@inline _npts(plan::TensorProductSpinPlan) = length(plan.θ) * length(plan.φ)
@inline _θnodes(plan::TensorProductSpinPlan) = plan.θ
@inline _fbuf(plan::TensorProductSpinPlan) = plan.fbuf
@inline _solve_w(plan::TensorProductSpinPlan) = plan.solve_w
@inline _fold_kind(plan::TensorProductSpinPlan) = _tensor_fold_kind(eltype(plan.fbuf))
@inline _tensor_fold_kind(::Type{<:Real}) = FoldedReal()
@inline _tensor_fold_kind(::Type{<:Complex}) = FullModes()
_spin_node_memory(plan::TensorProductSpinPlan) = Base.summarysize(plan.θ) + Base.summarysize(plan.φ)
_spin_solve_memory(plan::TensorProductSpinPlan) = Base.summarysize(plan.solve_w)

function Base.show(io::IO, plan::TensorProductSpinPlan{T}) where {T}
    print(io, "TensorProductSpinPlan{", T, "}(lmax=", plan.lmax,
          ", s=", plan.s, ", grid=", (length(plan.θ), length(plan.φ)), ", B=", plan.B, ")")
end

function _assemble_G!(G, sf, plan::TensorProductSpinPlan)
    return _tensor_synthesis!(G, sf, plan, plan.engine)
end

function _tensor_synthesis!(G, sf, plan, eng::_RingEngine)
    fill!(G, zero(eltype(G)))
    nt = _ring_tasks(plan, eng)
    _on_tasks(nt) do t
        w = eng.work[t]
        for c in (t - 1):nt:plan.lmax, m in -c:max(2c, 1):c
            i = m + plan.lmax + 1
            lo = max(abs(m), abs(plan.s))
            _ring_order_coeffs!(w, m, -plan.s, lo, plan.lmax)
            col = mod(m, size(G, 2)) + 1
            phase = plan.phase[i]
            for b in 1:plan.B
                _ring_weights!(w, sf, i, b, 1, lo, plan.lmax)
                for j0 in 0:_ring_lanes(eltype(eng.u)):(length(eng.u) - 1)
                    _ring_chunk!(w, eng, i, lo, plan.lmax, j0, Val(1))
                end
                @inbounds for j in 1:eng.nring
                    G[j, col, b] = phase * complex(w.Fr[j, 1], w.Fi[j, 1])
                end
            end
        end
    end
    return G
end

function nusht_type2_spin!(f, sf, plan::TensorProductSpinPlan)
    _check_spin_coeffs(sf, plan)
    _check_spin_field(f, plan)
    _assemble_G!(plan.G, sf, plan)
    plan.longitude.backward * plan.G
    _tensor_values!(f, plan.G, eltype(_fbuf(plan)))
    return f
end

function _tensor_values!(f, G, ::Type{<:Real})
    f .= real.(reshape(G, size(f)))
    return f
end

function _tensor_values!(f, G, ::Type{<:Complex})
    copyto!(f, G)
    return f
end

function nusht_type1_spin!(sf, f, plan::TensorProductSpinPlan)
    _check_spin_coeffs(sf, plan)
    _check_spin_field(f, plan)
    copyto!(plan.G, f)
    return _strengths_adjoint!(sf, plan)
end

function _strengths_adjoint!(sf, plan::TensorProductSpinPlan)
    _longitude_adjoint(plan.longitude, plan.G)
    return _tensor_adjoint!(sf, plan.G, plan, plan.engine)
end

function _tensor_adjoint!(sf, G, plan, eng::_RingEngine)
    nt = _ring_tasks(plan, eng)
    _on_tasks(nt) do t
        w = eng.work[t]
        for c in (t - 1):nt:plan.lmax, m in -c:max(2c, 1):c
            i = m + plan.lmax + 1
            lo = max(abs(m), abs(plan.s))
            _ring_order_coeffs!(w, m, -plan.s, lo, plan.lmax)
            col = mod(m, size(G, 2)) + 1
            phase = conj(plan.phase[i])
            for b in 1:plan.B
                fill!(w.Fr, zero(eltype(w.Fr)))
                fill!(w.Fi, zero(eltype(w.Fi)))
                @inbounds for j in 1:eng.nring
                    z = phase * G[j, col, b]
                    w.Fr[j, 1] = real(z); w.Fi[j, 1] = imag(z)
                end
                _ring_adjoint_sums!(sf, w, eng, i, b, lo, plan.lmax, Val(1))
            end
        end
    end
    return sf
end

function _quadrature!(sf, f, w, plan::TensorProductSpinPlan)
    _check_spin_solvable(plan)
    _check_spin_coeffs(sf, plan)
    _check_spin_field(f, plan)
    _check_weights(w, plan)
    _copy_weighted!(_fbuf(plan), f, w)
    copyto!(plan.G, _fbuf(plan))
    _strengths_adjoint!(sf, plan)
    return _hermitian_part!(sf, plan)
end

function FTB.lsmr_forward!(u, plan::TensorProductSpinPlan, v, c, n)
    FTB.colscale!(u, c, n, plan.B)
    nusht_type2_spin!(_fbuf(plan), _av_coeffs(v, plan), plan)
    return _add_field!(u, _fbuf(plan), n)
end

close!(::TensorProductSpinPlan) = nothing

set_nodes!(plan::TensorProductSpinPlan, θ, φ) = throw(ArgumentError(
    "a tensor-product plan fixes its axes; build a plan for the new axes"))
