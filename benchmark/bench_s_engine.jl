# Phase 0a — which S-engine produces the DFS Fourier coefficients faster?
#
# `S` must turn spherical harmonic coefficients into the bivariate Fourier coefficients the NUFFT
# evaluates. Two engines do that:
#
#   scalar : FastTransforms `plan_sph2fourier` (butterfly), O(L² log² L)
#   spin   : the Wigner-d factorization at s = 0 (`_assemble_G!`), O(L³)
#
#     d^ℓ_{mn}(θ) = i^{m−n} Σ_{m'} Δ^ℓ_{m'm} Δ^ℓ_{m'n} e^{−im'θ},   Y_ℓm = N_ℓ d^ℓ_{m,0}(θ) e^{imφ}
#
# so at s = 0 the spin contraction IS a scalar S-engine — one already threaded, Float32-generic and
# device-capable, and one that would let FastTransforms (with its OpenMP-in-a-task hazard, Float64
# lock and host bounce) be dropped entirely.
#
# The butterfly is asymptotically better, so this measures where the crossover actually is rather than
# assuming either answer. Only the S step is timed: no NUFFT, no real/complex basis conversion.
using NUFSHT: NUFSHT
using FINUFFT: FINUFFT
using FastSphericalHarmonics: FastSphericalHarmonics
using LinearAlgebra: LinearAlgebra
using Printf: @printf
using Random: Random

# Cold (first call at this shape) and warm (steady state) are reported separately: a min-over-repeats
# hides first-call specialization, and for a shape used once that is the number that matters.
function timed(f, n)
    cold = @elapsed f()
    warm = Inf
    for _ in 1:n
        warm = min(warm, @elapsed f())
    end
    return cold, warm
end

# Work the spin contraction does: Σ_ℓ (2ℓ+1)² triples, plus the same again for the Δ recurrence.
# Reported next to the time so the O(L³) claim is checkable without a stopwatch.
spin_triples(L) = sum((2ℓ + 1)^2 for ℓ in 0:L)

function main()
    @printf("julia -t%d | BLAS=%d | FFTW planner=%d | FastTransforms default nthreads=%d\n",
            Threads.nthreads(), LinearAlgebra.BLAS.get_num_threads(),
            NUFSHT.FFTW.get_num_threads(), NUFSHT._fasttransforms_default_nthreads())
    @printf("loadavg=%s | %s physical / %s logical cores\n",
            strip(read(`sysctl -n vm.loadavg`, String)),
            strip(read(`sysctl -n hw.physicalcpu`, String)),
            strip(read(`sysctl -n hw.ncpu`, String)))
    println("="^104)
    @printf("%-6s %-12s | %-11s %-11s | %-11s %-11s | %-9s %s\n",
            "lmax", "spin triples", "S butterfly", "(cold)", "S spin s=0", "(cold)",
            "spin/bfly", "verdict")

    for lmax in (32, 64, 128, 256, 512)
        # A handful of points: the NUFFT is not timed here, but a plan needs nodes.
        M = 64
        Random.seed!(1)
        θ = clamp.(acos.(2 .* rand(M) .- 1), 1e-10, π - 1e-10)
        φ = 2π .* rand(M)
        N, Nf = lmax + 1, 2lmax + 1

        C = zeros(N, Nf)
        for l in 0:lmax, m in -l:l
            C[FastSphericalHarmonics.sph_mode(l, m)] = randn() / (1 + l)
        end
        sf = zeros(ComplexF64, N, Nf)
        for l in 0:lmax, m in -l:l
            sf[NUFSHT.spin_coeff_index(l, m, lmax)] = (randn() + im * randn()) / (1 + l)
        end

        be = NUFSHT.FINUFFTBackend()
        ps = NUFSHT.make_plan(Float64, θ, φ, lmax; tol = 1e-8, nufft = be)
        pg = NUFSHT.make_spin_plan(ComplexF64, θ, φ, lmax, 0; tol = 1e-8, nufft = be)

        # Butterfly arm: exactly what `nusht_type2!` does for S — load the slice buffer, apply P.
        # `_sph_evaluate!` is in-place on plan.F, so the reload is part of the step, as in production.
        bc, bw = timed(() -> (copyto!(ps.F, C); NUFSHT._sph_evaluate!(ps)), 5)
        # Recurrence arm: coefficients -> G, the same DFS Fourier coefficients, not in place.
        gc, gw = timed(() -> NUFSHT._assemble_G!(pg.G, sf, pg), 5)

        r = gw / bw
        @printf("%-6d %-12.3g | %-11.3f %-11.3f | %-11.3f %-11.3f | %-9.2f %s\n",
                lmax, float(spin_triples(lmax)), 1e3bw, 1e3bc, 1e3gw, 1e3gc, r,
                r < 1 ? "recurrence wins" : "butterfly wins")
        flush(stdout)
        NUFSHT.close!(ps); NUFSHT.close!(pg)
    end
    return nothing
end

main()
