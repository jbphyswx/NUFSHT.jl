"""
spin_synthesis.jl — Spin-weighted (spin-1) scattered transforms in NUFSHT.

Demonstrates:
1. Spin-1 synthesis at scattered points and agreement with direct ₛY_ℓm evaluation.
2. Exact inversion (`nusht_solve_spin!`) at arbitrary scattered points.
3. The spin-1 Hodge/Helmholtz split of a tangent vector field (U = u_θ + i u_φ): the
   symmetric/antisymmetric parts of the spin(±1) coefficients give the rotational/divergent
   components — the building block for scattered-spherical Helmholtz decomposition.

Run from the NUFSHT.jl directory:
    julia --project=examples examples/spin_synthesis.jl
"""

using NUFSHT: NUFSHT
using Random: Random

relnorm(x) = sqrt(sum(abs2, x))
Random.seed!(1)

lmax, s = 16, 1
shp = (lmax + 1, 2lmax + 1)
M = 3000
θ = π .* rand(M)
φ = 2π .* rand(M)
plan = NUFSHT.make_spin_plan(θ, φ, lmax, s; tol = 1e-11)

# 1. Synthesis vs direct ₛY_ℓm
sf = zeros(ComplexF64, shp)
for ℓ in abs(s):lmax, m in -ℓ:ℓ
    sf[NUFSHT.spin_coeff_index(ℓ, m, lmax)] = randn(ComplexF64)
end
f = zeros(ComplexF64, M)
NUFSHT.nusht_type2_spin!(f, sf, plan)
ref = [sum(sf[NUFSHT.spin_coeff_index(ℓ, m, lmax)] * NUFSHT.sYlm(s, ℓ, m, θ[j], φ[j])
           for ℓ in abs(s):lmax for m in -ℓ:ℓ) for j in 1:M]
println("1. spin-$s synthesis vs direct ₛY: rel_err = ", relnorm(f .- ref) / relnorm(ref))

# 2. Exact scattered inversion
sol = zeros(ComplexF64, shp)
_, iters, relres = NUFSHT.nusht_solve_spin!(sol, f, plan; rtol = 1e-10)
println("2. nusht_solve_spin!: $iters iters, rel_res=$relres, coeff_err=", relnorm(sol .- sf) / relnorm(sf))

# 3. Spin-1 Hodge split of a pure-rotational tangent field (Rossby-like)
n, m = 3, 2
lat = (π / 2) .- θ
ue = [n * cos(m * φ[j]) * cos(lat[j])^(n - 1) * sin(lat[j]) for j in 1:M]   # u_east
un = [-m / cos(lat[j]) * sin(m * φ[j]) * cos(lat[j])^n for j in 1:M]        # u_north
uθ = -un; uφ = ue                                                          # θ̂ south, φ̂ east
planp = NUFSHT.make_spin_plan(θ, φ, lmax, +1; tol = 1e-11)
planm = NUFSHT.make_spin_plan(θ, φ, lmax, -1; tol = 1e-11)
ap = zeros(ComplexF64, shp); NUFSHT.nusht_solve_spin!(ap, uθ .+ im .* uφ, planp; rtol = 1e-9)
am = zeros(ComplexF64, shp); NUFSHT.nusht_solve_spin!(am, uθ .- im .* uφ, planm; rtol = 1e-9)
sym = (ap .+ am) ./ 2; anti = (ap .- am) ./ 2
function speed(a1, a2)
    V1 = zeros(ComplexF64, M); NUFSHT.nusht_type2_spin!(V1, a1, planp)
    V2 = zeros(ComplexF64, M); NUFSHT.nusht_type2_spin!(V2, a2, planm)
    relnorm(real.((V1 .- V2) ./ (2im))) + relnorm(real.((V1 .+ V2) ./ 2))
end
U = relnorm(ue) + relnorm(un)
println("3. Rossby (pure rotational): rotational/|U| = ", round(speed(sym, sym) / U, sigdigits = 4),
        "   divergent/|U| = ", round(speed(anti, .-anti) / U, sigdigits = 4))
