"""
    bench_spin_engines.jl — the spin S-step in Float64 and Float32.

Times the step from spin-weighted coefficients to the bivariate Fourier mode array, and its transpose,
for one field (`B = 1`) at lmax 256, 512 and 1024: the ring S-step of a Float64 and of a Float32 plan,
and FastTransforms' `plan_spinsph2fourier` applied to a coefficient matrix of the same size as a
yardstick. No NUFFT is timed. The configuration prints first, then per lmax each step's best of
`repeats` with its first call in parentheses. For the ring S-step it prints the fraction of the
machine's dense matrix product rate in the same precision that its arithmetic reaches, at 11
floating-point operations per ring and degree, and the Float32 result's relative difference from the
Float64 one.

Run:  julia --project=. -t auto benchmark/bench_spin_engines.jl
"""

using NUFSHT: NUFSHT
using FlowTransformBindings: FlowTransformBindings as FTB
using LinearAlgebra: LinearAlgebra
using Printf: Printf
using Random: Random

const FLOPS_PER_STEP = 11

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

# Floating-point operations per second of a dense `n × n` matrix product in `T`, best of three.
function gemm_rate(::Type{T}, n = 2048) where {T}
    A = rand(T, n, n); B = rand(T, n, n); C = similar(A)
    LinearAlgebra.mul!(C, A, B)
    t = minimum(@elapsed(LinearAlgebra.mul!(C, A, B)) for _ in 1:3)
    return 2n^3 / t
end

# Ring steps of one transform: every order's degrees on every ring of the engine.
ring_steps(eng, lmax, s) = sum(m -> (lmax - max(abs(m), abs(s)) + 1) * length(eng.u), -lmax:lmax)

function main(; s::Int = 1, lmaxs = (256, 512, 1024), repeats::Int = 3)
    nt = Threads.nthreads()
    FT = NUFSHT.FastTransforms
    LinearAlgebra.BLAS.set_num_threads(nt)
    Printf.@printf("julia %s -t%d | BLAS=%d | FFTW planner=%d\n", VERSION, nt,
                   LinearAlgebra.BLAS.get_num_threads(), NUFSHT.FFTW.get_num_threads())
    Printf.@printf("%s | %d logical cores | loadavg %s\n", Sys.cpu_info()[1].model, Sys.CPU_THREADS,
                   join(round.(Sys.loadavg(); digits = 2), " "))
    peak = Dict(T => gemm_rate(T) for T in (Float64, Float32))
    Printf.@printf("dense product rate: Float64 %.1f GFLOP/s, Float32 %.1f GFLOP/s\n",
                   peak[Float64] / 1e9, peak[Float32] / 1e9)
    Printf.@printf("spin s=%d | B=1 | %d tasks, FastTransforms at %d threads | ms, best of %d (first call)\n",
                   s, nt, nt, repeats)
    println("="^108)
    Printf.@printf("%-5s %-9s | %-21s | %-21s %-8s | %-21s %-8s %-9s\n", "lmax", "step",
                   "FastTransforms F64", "ring F64", "of peak", "ring F32", "of peak", "Δ F32")

    for lmax in lmaxs
        Random.seed!(lmax)
        M = 64
        θ = acos.(2 .* rand(M) .- 1); φ = 2π .* rand(M)
        DS = NUFSHT.SpectralBackends.DirectSumSpectralBackend()
        p64 = NUFSHT.make_spin_plan(ComplexF64, θ, φ, lmax, s; nthreads = nt, nufft = DS)
        p32 = NUFSHT.make_spin_plan(ComplexF32, Float32.(θ), Float32.(φ), lmax, s; nthreads = nt, nufft = DS)
        work = ring_steps(p64.engine, lmax, s) * FLOPS_PER_STEP

        sf = zeros(ComplexF64, lmax + 1, 2lmax + 1)
        for ℓ in abs(s):lmax, m in -ℓ:ℓ
            sf[NUFSHT.spin_coeff_index(ℓ, m, lmax)] = randn(ComplexF64) / (1 + ℓ)
        end
        sf32 = ComplexF32.(sf)
        G64 = similar(p64.G); G32 = similar(p32.G)
        a64 = similar(sf); a32 = similar(sf32)
        Ĝ = randn(ComplexF64, size(p64.G)); Ĝ32 = ComplexF32.(Ĝ)

        A = copy(sf)
        P = FTB.with_fasttransforms_threads(() -> FT.plan_spinsph2fourier(A, s), nt)
        ft(Q) = FTB.with_fasttransforms_threads(() -> (copyto!(A, sf); LinearAlgebra.lmul!(Q, A)), nt)

        rows = (("forward", timed(() -> ft(P), repeats),
                 timed(() -> NUFSHT._assemble_G_impl!(G64, sf, p64, p64.engine), repeats),
                 timed(() -> NUFSHT._assemble_G_impl!(G32, sf32, p32, p32.engine), repeats),
                 () -> relerr(ComplexF64.(G32), G64)),
                ("adjoint", timed(() -> ft(P'), repeats),
                 timed(() -> NUFSHT._assemble_G_adjoint_impl!(a64, Ĝ, p64, p64.engine), repeats),
                 timed(() -> NUFSHT._assemble_G_adjoint_impl!(a32, Ĝ32, p32, p32.engine), repeats),
                 () -> relerr(ComplexF64.(a32), a64)))
        cell((cold, warm)) = Printf.@sprintf("%9.2f (%9.2f)", 1e3warm, 1e3cold)
        for (name, tf, t64, t32, Δ) in rows
            Printf.@printf("%-5d %-9s | %-21s | %-21s %-8.3f | %-21s %-8.3f %-9.1e\n", lmax, name,
                           cell(tf), cell(t64), work / t64[2] / peak[Float64],
                           cell(t32), work / t32[2] / peak[Float32], Δ())
        end
        flush(stdout)
        NUFSHT.close!(p64); NUFSHT.close!(p32)
    end
    return nothing
end

main()
