"""
    bench_spin_recurrence.jl — spin S-engine: O(lmax²) memory + large-lmax stability.

The spin transform's Legendre step is an on-the-fly Trapani–Navaza Wigner-`d` recurrence into two
reused `(2lmax+1)²` buffers (O(lmax²) memory), replacing the old dense per-degree `Δ^ℓ` store
(O(lmax³)) which was also silently inaccurate above lmax ≈ 40. This shows the plan memory scaling,
that the recurrence assembly is zero-allocation, and that a single unit mode stays bounded (|f| ≲ N_ℓ)
at large lmax — the regime where the old dense evaluation blew up to |f| ~ 1e9.

Run:  julia --project=. benchmark/bench_spin_recurrence.jl
"""

using NUFSHT: NUFSHT
using Random: Random
using Printf: Printf

Random.seed!(0)
s = 1
println("lmax │  plan buffers (MiB)   │ O(lmax²) vs O(lmax³) │ assemble alloc │ unit-mode max|f|")
println("─────┼───────────────────────┼──────────────────────┼────────────────┼─────────────────")
for lmax in (16, 64, 128, 256, 512)
    L = 2lmax + 1
    M = 200
    θ = clamp.(π .* rand(M), 1e-9, π - 1e-9)
    φ = 2π .* rand(M)
    plan = NUFSHT.make_spin_plan(θ, φ, lmax, s; tol = 1e-8)

    # recurrence buffers: 2 real (L×L) + G/fbuf complex — O(lmax²).
    buf_bytes = 2 * sizeof(plan.dl_curr) + sizeof(plan.G) + sizeof(plan.fbuf)
    dense_bytes = sum((2ℓ + 1)^2 for ℓ in 0:lmax) * sizeof(Float64)   # old O(lmax³) dense-Δ store

    # zero-alloc assembly on the hot path
    sf = zeros(ComplexF64, lmax + 1, L)
    for ℓ in s:min(lmax, 8), m in -ℓ:ℓ
        sf[NUFSHT.spin_coeff_index(ℓ, m, lmax)] = randn(ComplexF64)
    end
    NUFSHT._assemble_G!(plan.G, sf, plan)
    a = @allocated NUFSHT._assemble_G!(plan.G, sf, plan)

    # single unit mode at ℓ=lmax stays O(1) (dense-Δ gave ~1e9 here for lmax≳64)
    sf1 = zeros(ComplexF64, lmax + 1, L); sf1[NUFSHT.spin_coeff_index(lmax, 0, lmax)] = 1
    f = zeros(ComplexF64, M); NUFSHT.nusht_type2_spin!(f, sf1, plan)

    Printf.@printf("%4d │  %7.3f (recurrence)   │ %6.1f×  smaller       │ %6d bytes   │ %.4g\n",
                   lmax, buf_bytes / 2^20, dense_bytes / buf_bytes, a, maximum(abs, f))
    NUFSHT.close!(plan)
end
