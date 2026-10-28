using NUFSHT: NUFSHT
using FlowTransformBindings: FlowTransformBindings as FTB
using FastSphericalHarmonics: FastSphericalHarmonics
using LinearAlgebra: LinearAlgebra
using Random: Random
using Printf: Printf

function elapsed(f)
    f()
    return minimum(@elapsed(f()) for _ in 1:3)
end

function fsh_adjoint!(out, input, synthesis_adjoint, conversion_adjoint, nt)
    copyto!(out, input)
    FTB.with_fasttransforms_threads(nt) do
        LinearAlgebra.lmul!(synthesis_adjoint, out)
        LinearAlgebra.lmul!(conversion_adjoint, out)
    end
    return out
end

function main()
    nt = Threads.nthreads()
    println("project=", Base.active_project(), " Julia=", VERSION, " threads=", nt,
            " BLAS=", LinearAlgebra.BLAS.get_num_threads())
    println("execution=AutoBackend() transform=TensorProduct() longitude=FFTW nthreads=", nt)
    println("CPU=", Sys.cpu_info()[1].model, " load=", Sys.loadavg())
    println("lmax plan_s fsh_plan_s synthesis_s fsh_synthesis_s ratio adjoint_s fsh_adjoint_s ratio error")
    rng = Random.Xoshiro(1)
    for lmax in (256, 512, 1024)
        θ, φ = FastSphericalHarmonics.sph_points(lmax + 1)
        C = randn(rng, lmax + 1, 2lmax + 1)
        NUFSHT.apply_transfer!(C, NUFSHT.TopHatTransfer(lmax), lmax)
        grid = NUFSHT.TensorProduct()
        plan_time = @elapsed p = NUFSHT.make_plan(θ, φ, lmax, grid; nthreads = nt)
        println("plan=", p, " longitude=", typeof(p.spin.longitude))
        cache = FastSphericalHarmonics.SphPlanCache{Float64}()
        work = copy(C)
        fsh_plan_time = @elapsed FTB.with_fasttransforms_threads(nt) do
            FastSphericalHarmonics.sph_evaluate!(work; cache)
        end
        reference = copy(work)
        f = zeros(size(C))
        A = similar(C)
        input = randn(rng, size(C))
        FT = NUFSHT.FastTransforms
        ps = cache.PS[lmax + 1]
        pc = cache.P[lmax + 1]
        psadj = FT.AdjointFTPlan(ps, ps)
        pcadj = FT.AdjointFTPlan(pc, pc)
        synth = elapsed(() -> NUFSHT.nusht_type2!(f, C, p))
        fsynth = elapsed(() -> FTB.with_fasttransforms_threads(nt) do
            copyto!(work, C)
            FastSphericalHarmonics.sph_evaluate!(work; cache)
        end)
        adj = elapsed(() -> NUFSHT.nusht_type1!(A, input, p))
        fadj = elapsed(() -> fsh_adjoint!(work, input, psadj, pcadj, nt))
        error = sqrt(sum(abs2, f .- reference) / sum(abs2, reference))
        Printf.@printf("%d %.6f %.6f %.6f %.6f %.3f %.6f %.6f %.3f %.3e\n",
                       lmax, plan_time, fsh_plan_time, synth, fsynth, synth / fsynth,
                       adj, fadj, adj / fadj, error)
        NUFSHT.close!(p)
        flush(stdout)
    end
    println("TENSOR_PRODUCT_BENCHMARK_COMPLETE")
end

main()
