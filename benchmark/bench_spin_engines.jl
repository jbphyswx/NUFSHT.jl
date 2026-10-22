"""
    bench_spin_engines.jl — the spin S-step on FastTransforms against the Wigner recurrence.

Times the step from spin-weighted coefficients to the bivariate Fourier mode array and its transpose
on one host Float64 plan, through the plan's FastTransforms engine and through the Wigner recurrence
passed as the engine, at lmax 256, 512 and 1024 with one field (`B = 1`). No NUFFT is timed. The
configuration prints first; then, per lmax, each step's best-of-`repeats` time with its first call in
parentheses, the recurrence-to-FastTransforms ratio, and the relative difference of the two outputs.

Run:  julia --project=. -t auto benchmark/bench_spin_engines.jl
"""

using NUFSHT: NUFSHT
using LinearAlgebra: LinearAlgebra
using Printf: Printf
using Random: Random

# First call and best of `n` more, in seconds.
function timed(f, n)
    cold = @elapsed f()
    warm = Inf
    for _ in 1:n
        warm = min(warm, @elapsed f())
    end
    return cold, warm
end

relerr(a, b) = sqrt(sum(abs2, a .- b) / sum(abs2, b))

function main(; s::Int = 1, lmaxs = (256, 512, 1024), repeats::Int = 3)
    omp = ccall((:omp_get_max_threads, NUFSHT.FastTransforms.libfasttransforms), Cint, ())
    Printf.@printf("julia %s -t%d | BLAS=%d | FFTW planner=%d | FastTransforms OpenMP=%d\n", VERSION,
            Threads.nthreads(), LinearAlgebra.BLAS.get_num_threads(), NUFSHT.FFTW.get_num_threads(), omp)
    Printf.@printf("%s | %d logical cores | loadavg %s\n", Sys.cpu_info()[1].model, Sys.CPU_THREADS,
            join(round.(Sys.loadavg(); digits = 2), " "))
    Printf.@printf("spin s=%d | B=1 | times in ms, best of %d (first call) | recurrence tasks per block shown\n",
            s, repeats)
    println("="^126)
    Printf.@printf("%-5s %-5s %-9s | %-21s %-21s %-7s | %-21s %-21s %-7s | %-9s %-9s\n",
            "lmax", "tasks", "build ms", "FT forward", "Wigner forward", "ratio",
            "FT adjoint", "Wigner adjoint", "ratio", "Δ fwd", "Δ adj")

    for lmax in lmaxs
        Random.seed!(lmax)
        M = 64
        θ = clamp.(acos.(2 .* rand(M) .- 1), 1e-10, π - 1e-10)
        φ = 2π .* rand(M)
        build = @elapsed (plan = NUFSHT.make_spin_plan(ComplexF64, θ, φ, lmax, s; tol = 1e-8,
                                         nufft = NUFSHT.SpectralBackends.DirectSumSpectralBackend()))
        plan.engine isa NUFSHT._FTSpinEngine ||
            error("expected the FastTransforms engine on a host Float64 plan, got $(typeof(plan.engine))")
        eng = NUFSHT._wigner_planes(NUFSHT._θnodes(plan), lmax, plan.ntasks)

        sf = zeros(ComplexF64, lmax + 1, 2lmax + 1)
        for ℓ in abs(s):lmax, m in -ℓ:ℓ
            sf[NUFSHT.spin_coeff_index(ℓ, m, lmax)] = randn(ComplexF64) / (1 + ℓ)
        end
        Ĝ = randn(ComplexF64, size(plan.G))
        Gf = similar(plan.G); Gw = similar(plan.G)
        af = similar(sf); aw = similar(sf)

        ff = timed(() -> NUFSHT._assemble_G_impl!(Gf, sf, plan, plan.engine), repeats)
        fw = timed(() -> NUFSHT._assemble_G_impl!(Gw, sf, plan, eng), repeats)
        jf = timed(() -> NUFSHT._assemble_G_adjoint_impl!(af, Ĝ, plan, plan.engine), repeats)
        jw = timed(() -> NUFSHT._assemble_G_adjoint_impl!(aw, Ĝ, plan, eng), repeats)

        cell((cold, warm)) = Printf.@sprintf("%9.2f (%9.2f)", 1e3warm, 1e3cold)
        Printf.@printf("%-5d %-5d %-9.1f | %-21s %-21s %-7.2f | %-21s %-21s %-7.2f | %-9.1e %-9.1e\n",
                lmax, NUFSHT._block_size(plan, eng), 1e3build, cell(ff), cell(fw), fw[2] / ff[2],
                cell(jf), cell(jw), jw[2] / jf[2], relerr(Gf, Gw), relerr(af, aw))
        flush(stdout)
        NUFSHT.close!(plan)
    end
    return nothing
end

main()
