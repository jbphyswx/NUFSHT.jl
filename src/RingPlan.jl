# A scalar plan on the ring S-step: the transforms of a spin-0 plan, its coefficients in a scalar plan's
# `sph_mode` layout converted on the way in and out (`scalar_to_spin0!`, `spin0_to_scalar!`).

export RingNUSHTplan

"""
    RingNUSHTplan{T,FE}

A scalar plan whose S-step is the ring engine of a spin-0 [`SpinNUSHTplan`](@ref), every step in the
plan's precision `T`, on the host or on the nodes' device; [`make_plan`](@ref) builds one for device
nodes. Its coefficients are a scalar plan's, `(lmax+1, 2lmax+1, B)` in the `sph_mode` layout, and pass
through the spin-0 layout by [`scalar_to_spin0!`](@ref) and [`spin0_to_scalar!`](@ref). It takes every
scalar transform, fit, analysis and filter a [`NUSHTplan`](@ref) takes. Its synthesis reads only the
slots of degree `l ≤ lmax`, and its adjoint writes zero in the others.

Fields: `lmax`, `B` and `tol` as a `NUSHTplan`'s; `spin`, the spin-0 plan; `sf`, the spin-0 coefficients
the conversions pass through; `C`, the filter scratch, empty until the first filter call.
"""
struct RingNUSHTplan{T<:AbstractFloat, FE<:Number, P<:SpinNUSHTplan{T}, CT3<:AbstractArray{Complex{T},3},
                     AT3<:AbstractArray{FE,3}, FT} <: AbstractScalarNUSHTplan{T,FE}
    lmax::Int
    B::Int
    tol::FT
    spin::P
    sf::CT3
    C::Base.RefValue{Union{Nothing,AT3}}
end

# The ring plan over the nodes as given, under `make_plan`'s keywords.
function _build_ring_plan(::Type{FE}, θ_nodes, φ_nodes, lmax; tol = 1e-8, kwargs...) where {FE<:Number}
    spin = _build_spin_plan(FE, θ_nodes, φ_nodes, lmax, 0; tol, kwargs...)
    sf = allocate_coefficients(spin)
    AT3 = typeof(similar(sf, FE, 0, 0, 0))
    return RingNUSHTplan{real(FE), FE, typeof(spin), typeof(sf), AT3, typeof(spin.tol)}(
        Int(lmax), spin.B, spin.tol, spin, sf, Base.RefValue{Union{Nothing,AT3}}(nothing))
end

coefficient_size(plan::RingNUSHTplan) = (plan.lmax + 1, 2plan.lmax + 1, plan.B)
allocate_coefficients(plan::RingNUSHTplan{T,FE}) where {T,FE} =
    _zeros_like(plan.sf, FE, coefficient_size(plan)...)

@inline _θnodes(plan::RingNUSHTplan) = _θnodes(plan.spin)
@inline _fbuf(plan::RingNUSHTplan) = _fbuf(plan.spin)

set_nodes!(plan::RingNUSHTplan, θ_nodes, φ_nodes) = (set_nodes!(plan.spin, θ_nodes, φ_nodes); plan)
close!(plan::RingNUSHTplan) = close!(plan.spin)

Base.show(io::IO, plan::RingNUSHTplan{T}) where {T} =
    print(io, "RingNUSHTplan{", T, "}(lmax=", plan.lmax, ", M=", _npts(plan), ", B=", plan.B, ", tol=",
          plan.tol, ")")

function plan_memory(plan::RingNUSHTplan)
    slot(r) = r[] === nothing ? 0 : Base.summarysize(r[])
    sp = plan.spin
    C       = slot(plan.C)
    sf      = Base.summarysize(plan.sf)
    G       = Base.summarysize(sp.G)
    fbuf    = Base.summarysize(_fbuf(sp))
    nodes   = Base.summarysize(_θnodes(sp)) + Base.summarysize(_φnodes(sp)) + Base.summarysize(_θnufft(sp))
    engine  = Base.summarysize(sp.engine)
    solve_w = slot(sp.solve_w)
    return (; C, sf, G, fbuf, nodes, engine, solve_w,
            total = C + sf + G + fbuf + nodes + engine + solve_w)
end

function nusht_type2!(f, C, plan::RingNUSHTplan)
    _check_coeffs(C, plan)
    _check_field(f, plan)
    scalar_to_spin0!(plan.sf, C)
    return nusht_type2_spin!(f, plan.sf, plan.spin)
end

# On a real field, `spin0_to_scalar!` into a real `C` is the transpose of `scalar_to_spin0!` over the
# reals, so this is the exact adjoint of the synthesis in both cases.
function _nusht_true_adjoint!(C, f, plan::RingNUSHTplan)
    _check_field(f, plan)
    _check_coeffs(C, plan)
    nusht_type1_spin!(plan.sf, f, plan.spin)
    return spin0_to_scalar!(C, plan.sf)
end

function _quadrature!(C, f, w, plan::RingNUSHTplan)
    _check_coeffs(C, plan)
    _quadrature!(plan.sf, f, w, plan.spin)
    return spin0_to_scalar!(C, plan.sf)
end

# The fit runs on the spin-0 plan's coefficients. The conversion is unitary (for a real field, an
# isometry onto the Hermitian coefficients), so the converted solution is the scalar fit's. A finished
# column lands in `sf`, and `_finish_solution!` converts the solution into the caller's array.
FTB.lsmr_allocate_domain(plan::RingNUSHTplan) = FTB.lsmr_allocate_domain(plan.spin)
FTB.lsmr_allocate_range(plan::RingNUSHTplan) = FTB.lsmr_allocate_range(plan.spin)
FTB.lsmr_check_solution(C, plan::RingNUSHTplan, ws) = _check_coeffs(C, plan)
FTB.lsmr_forward!(u, plan::RingNUSHTplan, v, c, n) = FTB.lsmr_forward!(u, plan.spin, v, c, n)
FTB.lsmr_write!(_, plan::RingNUSHTplan, x, k, j) = FTB.lsmr_write!(plan.sf, plan.spin, x, k, j)
_adjoint_full!(plan::RingNUSHTplan, u, n::Integer) = _adjoint_full!(plan.spin, u, n)
_fold!(v, plan::RingNUSHTplan, w, c, n::Integer) = _fold!(v, plan.spin, w, c, n)
_finish_solution!(C, plan::RingNUSHTplan) = spin0_to_scalar!(C, plan.sf)
