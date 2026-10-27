# Which scalar plan transforms a host field faster: the FastTransforms `NUSHTplan` (its S-step in Float64
# whatever the field's precision) or the `RingNUSHTplan` (the ring S-step of a spin-0 plan in the field's
# precision)? Whole transforms are timed, synthesis and adjoint, on one NUFFT library, at `M = 2(lmax+1)²`
# random points.
using NUFSHT: NUFSHT
using FlowTransformBindings: FlowTransformBindings as FTB
using NonuniformFFTs: NonuniformFFTs
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

function main()
    Printf.@printf("julia -t%d | BLAS=%d | FFTW planner=%d | FastTransforms nthreads=%d\n",
            Threads.nthreads(), LinearAlgebra.BLAS.get_num_threads(), NUFSHT.FFTW.get_num_threads(),
            ccall((:omp_get_max_threads, NUFSHT.FastTransforms.libfasttransforms), Cint, ()))
    Printf.@printf("loadavg=%s | %s physical / %s logical cores | NUFFT NonuniformFFTs\n",
            strip(read(`sysctl -n vm.loadavg`, String)),
            strip(read(`sysctl -n hw.physicalcpu`, String)),
            strip(read(`sysctl -n hw.ncpu`, String)))
    println("="^100)
    Printf.@printf("%-8s %-6s %-9s | %-10s %-10s | %-10s %-10s | %s\n",
            "FE", "lmax", "M", "synth FT", "synth ring", "adj FT", "adj ring", "ring/FT (synth, adj)")
    nu = FTB.NonuniformFFTsBackend()
    for lmax in (256, 512, 1024), FE in (Float32, Float64)
        T = real(FE)
        M = 2 * (lmax + 1)^2
        rng = Random.Xoshiro(1)
        θ = T.(acos.(2 .* rand(rng, M) .- 1)); φ = T.(2π .* rand(rng, M))
        tol = T === Float32 ? 1e-6 : 1e-10
        pf = NUFSHT.make_plan(FE, θ, φ, lmax; nufft = nu, tol)
        pr = NUFSHT._build_ring_plan(FE, θ, φ, lmax; nufft = nu, tol)
        C = randn(rng, FE, lmax + 1, 2lmax + 1)
        NUFSHT.apply_transfer!(C, NUFSHT.TopHatTransfer(lmax), lmax)
        f = zeros(FE, M); y = randn(rng, FE, M); A = similar(C)
        _, sf = timed(() -> NUFSHT.nusht_type2!(f, C, pf), 3)
        _, sr = timed(() -> NUFSHT.nusht_type2!(f, C, pr), 3)
        _, af = timed(() -> NUFSHT.nusht_type1!(A, y, pf), 3)
        _, ar = timed(() -> NUFSHT.nusht_type1!(A, y, pr), 3)
        Printf.@printf("%-8s %-6d %-9d | %-10.4f %-10.4f | %-10.4f %-10.4f | %.2f, %.2f\n",
                string(FE), lmax, M, sf, sr, af, ar, sr / sf, ar / af)
        flush(stdout)
        NUFSHT.close!(pf); NUFSHT.close!(pr)
    end
    return nothing
end

main()
