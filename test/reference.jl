# Independent references for the test suite.
#
# The transform is only ever as trustworthy as what it is scored against. Everything here is computed
# WITHOUT the package: `ylm_ref` from the standard normalised associated-Legendre recurrence,
# `design_matrix` from repeated unit-mode synthesis, `cond_A` from its SVD. `ylm_ref` is itself pinned
# against `FastSphericalHarmonics.sph_evaluate` on the Clenshaw-Curtis grid (see `test_synthesis.jl`)
# before anything relies on it, so its convention cannot silently drift.
#
# Point-set fixtures carry their measured `cond(A)` so a test asserts the regime it thinks it is in.
# That is not decoration: the suite previously used an index-linked spiral, whose design matrix is
# near-degenerate (cond ≈ 1e8 at lmax = 10), for its "well-conditioned" accuracy test.

# Fully-normalised P̄_l^m(cosθ) for m = 0…l by the standard stable recurrence, with the
# Condon-Shortley phase in the sectorial seed. Normalised so ∫|Y|² dΩ = 1.
function plm_column_ref(lmax::Int, m::Int, θ::Float64)
    c = cos(θ); s = sin(θ)
    p = zeros(lmax + 1)
    pmm = 1 / sqrt(4π)
    for k in 1:m
        pmm *= -sqrt((2k + 1) / (2k)) * s
    end
    m > lmax && return p
    p[m + 1] = pmm
    m + 1 <= lmax && (p[m + 2] = sqrt(2m + 3) * c * pmm)
    for l in (m + 2):lmax
        a = sqrt((4l^2 - 1) / (l^2 - m^2))
        b = sqrt(((l - 1)^2 - m^2) / (4 * (l - 1)^2 - 1))
        p[l + 1] = a * (c * p[l] - b * p[l - 1])
    end
    return p
end

"""
    ylm_ref(l, m, θ, φ)

Real spherical harmonic in FastSphericalHarmonics' convention: orthonormal, `cos(mφ)` for `m > 0` and
`sin(|m|φ)` for `m < 0`, and **no** Condon-Shortley phase (the `(-1)^|m|` below strips the one the
recurrence seed carries). Pinned against `sph_evaluate` in `test_synthesis.jl`.
"""
function ylm_ref(l::Integer, m::Integer, θ::Real, φ::Real)
    am = abs(m)
    p = plm_column_ref(Int(l), am, Float64(θ))[l + 1] * (-1)^am
    m == 0 && return p
    return sqrt(2) * p * (m > 0 ? cos(am * φ) : sin(am * φ))
end

"""
    synth_ref(C, lmax, θ, φ) -> f

Direct `O(M·lmax³)` evaluation of `Σ_{l,m} C[sph_mode(l,m)] · Y_lm` at scattered points — the thing
`nusht_type2!` is supposed to compute, computed a completely different way.
"""
function synth_ref(C, lmax::Integer, θ::AbstractVector, φ::AbstractVector)
    f = zeros(length(θ))
    for i in eachindex(θ), l in 0:lmax, m in -l:l
        c = C[FastSphericalHarmonics.sph_mode(l, m)]
        iszero(c) && continue
        f[i] += c * ylm_ref(l, m, θ[i], φ[i])
    end
    return f
end

"""
    valid_mask_ref(lmax) -> BitMatrix

`true` on the `(lmax+1)²` slots that are genuine degrees `l ≤ lmax`. The remaining `lmax(lmax+1)`
slots of the square array carry degrees `lmax < l ≤ lmax+|m|`; the transform needs them, a
least-squares fit must not use them.
"""
function valid_mask_ref(lmax::Integer)
    v = falses(lmax + 1, 2lmax + 1)
    for l in 0:lmax, m in -l:l
        v[FastSphericalHarmonics.sph_mode(l, m)] = true
    end
    return v
end

"""
    design_matrix(plan, lmax, M) -> (A, idx)

The `M × (lmax+1)²` design matrix over the `l ≤ lmax` modes, built by synthesising each unit mode
through the plan. `idx` are the corresponding coefficient-array indices.
"""
function design_matrix(plan, lmax::Integer, M::Integer)
    idx = findall(valid_mask_ref(lmax))
    A = zeros(M, length(idx))
    C1 = zeros(lmax + 1, 2lmax + 1)
    f1 = zeros(M)
    for (k, I) in enumerate(idx)
        fill!(C1, 0.0); C1[I] = 1.0
        NUFSHT.nusht_type2!(f1, C1, plan)
        A[:, k] .= f1
    end
    return A, idx
end

cond_A(A) = (sv = LinearAlgebra.svdvals(A); sv[1] / sv[end])

# ── Point-set fixtures ─────────────────────────────────────────────────────────
# `cond` is the measured design-matrix condition number at 4x overdetermination, lmax 6-10.

"Golden-angle lattice: equidistributed, deterministic, cond(A) ≈ 1.04 — as well conditioned as a
scattered set gets, so a solve there converges in a handful of iterations."
function fib_points(M::Integer)
    ga = π * (3 - sqrt(5))
    θ = acos.(clamp.(1 .- 2 .* ((0:M-1) .+ 0.5) ./ M, -1.0, 1.0))
    return clamp.(θ, 1e-10, π - 1e-10), collect(mod.(ga .* (0:M-1), 2π))
end

"Area-uniform i.i.d. points: cond(A) ≈ 3-7 at 4x. The generic scattered case."
function iid_points(M::Integer, seed::Integer)
    Random.seed!(seed)
    return clamp.(acos.(2 .* rand(M) .- 1), 1e-10, π - 1e-10), 2π .* rand(M)
end

"Half the points crammed into a 0.02-cosine polar cap: cond(A) ≈ 40-80 at 4x. Badly sampled but
still solvable — the regime a real swath or station network lands in."
function clustered_points(M::Integer, seed::Integer)
    Random.seed!(seed)
    m1 = M ÷ 2
    θ1 = acos.(1 .- 0.02 .* rand(m1)); φ1 = 2π .* rand(m1)
    θ2 = acos.(2 .* rand(M - m1) .- 1); φ2 = 2π .* rand(M - m1)
    return clamp.(vcat(θ1, θ2), 1e-10, π - 1e-10), vcat(φ1, φ2)
end

"Index-linked spiral, φ advancing 2π/M per point while θ sweeps pole to pole. It winds only once, so
the design matrix is near-degenerate: cond(A) ≈ 2e4 at lmax 6 rising to ≈ 1e8 at lmax 10. Present so
the suite records that fact and nothing reuses it as a well-conditioned fixture."
function spiral_points(M::Integer, seed::Integer)
    Random.seed!(seed)
    φb = (2π / M) .* (0:M-1)
    θb = acos.(clamp.(2 .* ((0:M-1) .+ 0.5) ./ M .- 1, -1.0, 1.0))
    θ = clamp.(θb .+ (rand(M) .- 0.5) .* (0.4π / sqrt(M)), 1e-10, π - 1e-10)
    return θ, mod.(φb .+ (rand(M) .- 0.5) .* (0.8π / sqrt(M)), 2π)
end

"Random coefficients on the `l ≤ lmax` modes only, decaying like 1/(1+l)."
function rand_coeffs(lmax::Integer, seed::Integer, B::Integer = 1)
    Random.seed!(seed)
    C = zeros(lmax + 1, 2lmax + 1, B)
    for b in 1:B, l in 0:lmax, m in -l:l
        C[FastSphericalHarmonics.sph_mode(l, m), b] = randn() / (1 + l)
    end
    return B == 1 ? C[:, :, 1] : C
end

relerr(a, b) = sqrt(sum(abs2, a .- b) / sum(abs2, b))
