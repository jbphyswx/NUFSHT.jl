# Phase 0 — where does a transform's time and memory actually go?
#
# `A = N·F·S`: the Legendre step (FastTransforms butterfly), the mode assembly, and the NUFFT. Nothing
# should be optimised before this says which of them dominates, and at which sizes.
#
# Three rules, each answering a way an earlier measurement in this package went wrong:
#   * self-describing — every run prints its resolved backend and every thread count, because a flat or
#     surprising number is a configuration smell before it is a finding;
#   * counts beside times — grid cells and spread points, so a complexity claim is checkable without a
#     stopwatch;
#   * cold and warm separately — a min-over-repeats hides first-call specialization, and for a shape
#     used once that is the number the user actually experiences.
using NUFSHT: NUFSHT
using FINUFFT: FINUFFT
using NonuniformFFTs: NonuniformFFTs
using FastSphericalHarmonics: FastSphericalHarmonics
using LinearAlgebra: LinearAlgebra
using Printf: Printf
using Random: Random

function timed(f, n)
    cold = @elapsed f()
    warm = Inf
    for _ in 1:n
        warm = min(warm, @elapsed f())
    end
    return cold, warm
end

# Oversampled grid the backend actually plans, so the FFT's size is reported rather than assumed.
function smooth(n::Integer)
    m = Int(n)
    while true
        x = m
        for p in (2, 3, 5)
            while x % p == 0
                x ÷= p
            end
        end
        x == 1 && return m
        m += 1
    end
end

function header()
    Printf.@printf("julia -t%d | BLAS=%d | FFTW planner=%d | FastTransforms default=%d | loadavg=%s\n",
            Threads.nthreads(), LinearAlgebra.BLAS.get_num_threads(),
            NUFSHT.FFTW.get_num_threads(), NUFSHT._fasttransforms_default_nthreads(),
            strip(read(`sysctl -n vm.loadavg`, String)))
    println("="^118)
    Printf.@printf("%-5s %-8s %-9s %-15s | %-8s %-8s %-8s | %-7s | %-8s %-9s | %s\n",
            "lmax", "M", "backend", "grid (σ-rounded)", "S", "F", "N", "S share",
            "type2", "solve/it", "plan MiB")
end

function run(lmax, M, FE, be, tol = 1e-8)
    N, Nf = lmax + 1, 2lmax + 1
    Random.seed!(7)
    θ = clamp.(acos.(2 .* rand(M) .- 1), 1e-10, π - 1e-10)
    φ = 2π .* rand(M)
    C = zeros(FE, N, Nf)
    for l in 0:lmax, m in -l:l
        C[FastSphericalHarmonics.sph_mode(l, m)] = FE <: Real ? randn() / (1 + l) :
                                                   (randn() + im * randn()) / (1 + l)
    end
    p = NUFSHT.make_plan(FE, θ, φ, lmax; tol = tol, nufft = be)
    f = zeros(FE, M)

    # S: load the coefficient slice and apply the butterfly, exactly as nusht_type2! does.
    _, tS = timed(() -> (copyto!(p.F, C); NUFSHT._sph_evaluate!(p)), 5)
    # F: the mode assembly alone.
    _, tF = timed(() -> NUFSHT._assemble_modes!(p.Fhat, p.F, lmax), 5)
    # N: the NUFFT alone.
    _, tN = timed(() -> NUFSHT._nufft_exec!(NUFSHT._nufft2(p), p.Fhat, NUFSHT._fbuf(p)), 5)
    _, t2 = timed(() -> NUFSHT.nusht_type2!(f, C, p), 5)

    NUFSHT.nusht_type2!(f, C, p)
    S = zeros(FE, N, Nf); ws = NUFSHT.LSMRWorkspace(p)
    _, tsolve = timed(() -> NUFSHT.nusht_solve!(S, f, p; ws = ws, rtol = 1e-6, maxiter = 20), 2)
    _, iters, = NUFSHT.nusht_solve!(S, f, p; ws = ws, rtol = 1e-6, maxiter = 20)

    g1, g2 = smooth(round(Int, 1.25 * (2lmax + 3))), smooth(round(Int, 1.25 * (2lmax + 1)))
    mem = NUFSHT.plan_memory(p)
    Printf.@printf("%-5d %-8d %-9s %-15s | %-8.3f %-8.3f %-8.3f | %-6.0f%% | %-8.3f %-9.3f | %.1f\n",
            lmax, M, be isa NUFSHT.FINUFFTBackend ? "FINUFFT" : "NUFFTs",
            string(g1, "x", g2), 1e3tS, 1e3tF, 1e3tN, 100 * tS / t2,
            1e3t2, 1e3tsolve / max(iters, 1), mem.total / 2^20)
    NUFSHT.close!(p)
    return nothing
end

header()
for (lmax, M) in ((32, 4_000), (64, 20_000), (128, 80_000), (256, 300_000))
    for be in (NUFSHT.FINUFFTBackend(), NUFSHT.NonuniformFFTsBackend())
        run(lmax, M, Float64, be)
    end
    flush(stdout)
end