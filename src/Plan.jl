"""
    Plan.jl — Pre-allocated plan struct for NUFSHT transforms.

A NUSHTplan pre-allocates all intermediate arrays so that repeated transforms
(e.g. filtering multiple fields at the same grid) minimise allocation.
"""

export NUSHTplan, make_plan

"""
    NUSHTplan{T}

Pre-computed plan for non-uniform spherical harmonic transforms.

Fields:
- `lmax`: Maximum spherical harmonic degree
- `Nθ`, `Nφ`: Size of the equiangular CC grid (Nθ = lmax+1, Nφ = 2lmax+1)
- `C`: Coefficient array (lmax+1) × (2lmax+1) — reused across calls
- `F`: Real map on equiangular CC grid (Nθ × Nφ)
- `F̃`: Doubled real map on torus (2Nθ × Nφ)
- `Fhat`: Complex Fourier coefficients of doubled map (2Nθ × Nφ)
- `θ_nodes`: Colatitudes (in [-π,π] for FINUFFT) of scattered points
- `φ_nodes`: Longitudes (in [-π,π] for FINUFFT) of scattered points
- `tol`: FINUFFT accuracy tolerance
"""
struct NUSHTplan{T<:AbstractFloat}
    lmax::Int
    Nθ::Int
    Nφ::Int
    C::Matrix{T}
    F::Matrix{T}
    F̃::Matrix{T}
    Fhat::Matrix{Complex{T}}
    θ_nodes::Vector{T}
    φ_nodes::Vector{T}
    tol::Float64
end

"""
    make_plan(θ_nodes, φ_nodes, lmax; tol=1e-8, T=Float64)

Construct a NUSHTplan for M scattered points at colatitudes θ_nodes ∈ [0,π]
and longitudes φ_nodes ∈ [0,2π), up to spherical harmonic degree lmax.

FINUFFT expects coordinates in [-π, π], so φ and θ̃ are remapped internally.
"""
function make_plan(
    θ_nodes,
    φ_nodes,
    lmax;
    tol = 1e-8,
    T::Type{<:AbstractFloat} = Float64,
)
    @assert length(θ_nodes) == length(φ_nodes)

    Nθ = lmax + 1
    Nφ = 2lmax + 1

    C    = zeros(T, Nθ, Nφ)
    F    = zeros(T, Nθ, Nφ)
    F̃    = zeros(T, 2Nθ, Nφ)
    Fhat = zeros(Complex{T}, 2Nθ, Nφ)

    θ = Vector{T}(θ_nodes)
    φ = Vector{T}(φ_nodes)

    return NUSHTplan{T}(lmax, Nθ, Nφ, C, F, F̃, Fhat, θ, φ, Float64(tol))
end
