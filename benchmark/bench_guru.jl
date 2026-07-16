"""
    bench_guru.jl — M1 performance demonstration.

Shows the effect of the guru-plan / zero-allocation rewrite: warmed-up transforms allocate nothing
and never re-plan FINUFFT, so `nusht_solve!` (hundreds of CG matvecs) and `nusht_filter!` are fast,
and a batched `ntrans = B` transform beats `B` separate single-field calls.

Run from the package directory:
    julia --project=. benchmark/bench_guru.jl
"""

using NUFSHT: NUFSHT
using Random: Random

function best_time(f!, n)
    f!()                       # warmup / compile
    best = Inf
    for _ in 1:n
        best = min(best, @elapsed f!())
    end
    return best
end

us(t) = string(round(t * 1e6; digits = 1), " µs")

Random.seed!(0)
lmax = 40
Nθ, Nφ = lmax + 1, 2lmax + 1
M = 8 * (lmax + 1)^2
θ = π .* rand(M)
φ = 2π .* rand(M)

plan = NUFSHT.make_plan(θ, φ, lmax; tol = 1e-8)
C = randn(Nθ, Nφ)
f = zeros(M); NUFSHT.nusht_type2!(f, C, plan)
out = zeros(M)
Cout = zeros(Nθ, Nφ)
filt = NUFSHT.gaussian_from_scale(1000e3)

println("lmax=$lmax  M=$M  (K=(lmax+1)(2lmax+1)=$(Nθ * Nφ))\n")
for (name, thunk, alloc) in (
    ("nusht_type2!", () -> NUFSHT.nusht_type2!(f, C, plan), () -> @allocated NUFSHT.nusht_type2!(f, C, plan)),
    ("nusht_type1!", () -> NUFSHT.nusht_type1!(Cout, f, plan), () -> @allocated NUFSHT.nusht_type1!(Cout, f, plan)),
    ("nusht_filter!", () -> NUFSHT.nusht_filter!(out, f, filt, plan), () -> @allocated NUFSHT.nusht_filter!(out, f, filt, plan)),
)
    t = best_time(thunk, 100)
    println(rpad(name, 16), rpad(us(t), 14), "alloc = ", alloc(), " B")
end

# CG solve (reuses the two guru plans across every matvec — no re-planning).
Csolve = zeros(Nθ, Nφ)
ws = NUFSHT.CGWorkspace(plan)
tsolve = best_time(() -> NUFSHT.nusht_solve!(Csolve, f, plan; ws = ws, rtol = 1e-6, maxiter = 400), 3)
_, iters, rel = NUFSHT.nusht_solve!(Csolve, f, plan; ws = ws, rtol = 1e-6, maxiter = 400)
println("\nnusht_solve!  $iters CG iters, rel_res=$(round(rel; sigdigits = 2))  ->  ",
        round(tsolve * 1e3; digits = 2), " ms  (", round(tsolve / (2iters) * 1e6; digits = 1), " µs / matvec)")

# Batched throughput: one ntrans=B call vs B separate single-field calls.
B = 8
planB = NUFSHT.make_plan(θ, φ, lmax; tol = 1e-8, ntrans = B)
CB = randn(Nθ, Nφ, B); fB = zeros(M, B)
tbatch = best_time(() -> NUFSHT.nusht_type2!(fB, CB, planB), 50)
cols = [CB[:, :, b] for b in 1:B]
fb = zeros(M)
tloop = best_time(() -> (for b in 1:B; NUFSHT.nusht_type2!(fb, cols[b], plan); end), 50)
println("\nsynthesis of B=$B fields:  batched ", us(tbatch), "   vs   looped ", us(tloop),
        "   (", round(tloop / tbatch; digits = 2), "x)")
