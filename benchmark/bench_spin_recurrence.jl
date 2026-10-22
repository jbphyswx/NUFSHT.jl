"""
    bench_spin_recurrence.jl — Wigner recurrence memory and large-lmax stability.

The Wigner recurrence builds each degree's `d(π/2)` plane from its stored `m = ℓ` column, one
`(lmax+1)²` quarter plane per task of a degree block. A host Float64 plan runs FastTransforms, so the
recurrence is passed to the assembly as its engine here. This prints the recurrence's buffer memory
against a precomputed `WignerTable`, the allocation of a one-task assembly, and the largest value of a
single unit mode at `ℓ = lmax`, which stays of order `N_ℓ`.

Run:  julia --project=. benchmark/bench_spin_recurrence.jl
"""

using NUFSHT: NUFSHT
using Random: Random
using Printf: Printf

Random.seed!(0)
s = 1
println("julia ", VERSION, ", threads ", Threads.nthreads(), ", one-task recurrence")
println("lmax │ plan buffers (MiB) │ WignerTable (MiB) │ assemble alloc │ unit-mode max|f|")
println("─────┼────────────────────┼───────────────────┼────────────────┼─────────────────")
for lmax in (16, 64, 128, 256, 512)
    L = 2lmax + 1
    M = 200
    θ = clamp.(π .* rand(M), 1e-9, π - 1e-9)
    φ = 2π .* rand(M)
    plan = NUFSHT.make_spin_plan(θ, φ, lmax, s; tol = 1e-8, nthreads = 1)
    eng = NUFSHT._wigner_planes(NUFSHT._θnodes(plan), lmax, 1)

    buf_bytes = sizeof(eng.bnd) + sizeof(eng.planes) + sizeof(plan.G) + sizeof(plan.nodes.fbuf)
    table_bytes = sum((2ℓ + 1)^2 for ℓ in 0:lmax) * sizeof(Float64)

    sf = zeros(ComplexF64, lmax + 1, L)
    for ℓ in s:min(lmax, 8), m in -ℓ:ℓ
        sf[NUFSHT.spin_coeff_index(ℓ, m, lmax)] = randn(ComplexF64)
    end
    NUFSHT._assemble_G_impl!(plan.G, sf, plan, eng)
    a = @allocated NUFSHT._assemble_G_impl!(plan.G, sf, plan, eng)

    # The unit mode through the recurrence and the plan's type-2 NUFFT; a complex plan's mode array is
    # the NUFFT's input as assembled.
    sf1 = zeros(ComplexF64, lmax + 1, L); sf1[NUFSHT.spin_coeff_index(lmax, 0, lmax)] = 1
    NUFSHT._assemble_G_impl!(plan.G, sf1, plan, eng)
    NUFSHT._nufft_exec!(NUFSHT._nufft2(plan), plan.G, NUFSHT._fbuf(plan))

    Printf.@printf("%4d │ %18.3f │ %17.1f │ %8d bytes │ %.4g\n",
                   lmax, buf_bytes / 2^20, table_bytes / 2^20, a, maximum(abs, NUFSHT._fbuf(plan)))
    NUFSHT.close!(plan)
end
