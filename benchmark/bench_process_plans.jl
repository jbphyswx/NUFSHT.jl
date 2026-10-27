# What a scalar plan divided among processes costs against one process holding it all: synthesis and a
# 20-iteration fit at `M = 2(lmax+1)²` random points, one thread per process, for one field (divided by
# point) and a batch of 8 (divided by field).
#
#   julia benchmark/bench_process_plans.jl distributed      # one process, then 2 Distributed workers
#   mpiexec -n 2 julia benchmark/bench_process_plans.jl mpi  # 2 MPI ranks
using NUFSHT: NUFSHT
using FlowTransformBindings: FlowTransformBindings as FTB
using ComputationalBackends: ComputationalBackends as CB
using NonuniformFFTs: NonuniformFFTs
using Distributed: Distributed
using MPI: MPI
using Printf: Printf
using Random: Random

const LMAX = 256

const BATCHES = (1, 8)

function points(lmax, B)
    M = 2 * (lmax + 1)^2
    rng = Random.Xoshiro(1)
    θ = acos.(2 .* rand(rng, M) .- 1); φ = 2π .* rand(rng, M)
    C = randn(rng, lmax + 1, 2lmax + 1, B)
    NUFSHT.apply_transfer!(C, NUFSHT.TopHatTransfer(lmax), lmax)
    return θ, φ, C
end

function timed(f, n)
    f()
    best = Inf
    for _ in 1:n
        best = min(best, @elapsed f())
    end
    return best
end

function measure(label, plan, θ, C)
    f = zeros(length(θ), size(C, 3))
    t2 = timed(() -> NUFSHT.nusht_type2!(f, C, plan), 3)
    S = NUFSHT.allocate_coefficients(plan)
    ts = timed(() -> NUFSHT.nusht_solve!(S, f, plan; rtol = 1e-14, maxiter = 20), 2)
    return (label, t2, ts)
end

report((label, t2, ts)) = Printf.@printf("%-28s | synth %8.4f s | 20-it fit %8.3f s\n", label, t2, ts)

function header(mode, nproc)
    Printf.@printf("mode=%s | processes=%d | julia -t%d | lmax=%d | M=%d | NonuniformFFTs | loadavg=%s\n",
            mode, nproc, Threads.nthreads(), LMAX, 2 * (LMAX + 1)^2,
            strip(read(`sysctl -n vm.loadavg`, String)))
end

const NU = FTB.NonuniformFFTsBackend()

if get(ARGS, 1, "") == "mpi"
    MPI.Init()
    comm = MPI.COMM_WORLD
    MPI.Comm_rank(comm) == 0 && header("mpi", MPI.Comm_size(comm))
    for B in BATCHES
        θ, φ, C = points(LMAX, B)
        p = NUFSHT.make_plan(Float64, θ, φ, LMAX; execution = CB.MPIBackend(CB.SerialBackend(); comm),
                             nufft = NU, ntrans = B)
        r = measure("MPI, $(MPI.Comm_size(comm)) ranks, B=$B", p, θ, C)
        MPI.Comm_rank(comm) == 0 && report(r)
        NUFSHT.close!(p)
    end
    MPI.Finalize()
else
    header("distributed", 1)
    for B in BATCHES
        θ, φ, C = points(LMAX, B)
        p = NUFSHT.make_plan(Float64, θ, φ, LMAX; execution = CB.SerialBackend(), nufft = NU, ntrans = B)
        report(measure("one process, 1 thread, B=$B", p, θ, C))
        NUFSHT.close!(p)
    end
    Distributed.addprocs(2; exeflags = ["--project=$(Base.active_project())", "-t1"])
    Distributed.@everywhere begin
        using NUFSHT: NUFSHT
        using NonuniformFFTs: NonuniformFFTs
    end
    for B in BATCHES
        θ, φ, C = points(LMAX, B)
        pd = NUFSHT.make_plan(Float64, θ, φ, LMAX; execution = CB.DistributedBackend(), nufft = NU, ntrans = B)
        report(measure("Distributed, 2 workers, B=$B", pd, θ, C))
        NUFSHT.close!(pd)
    end
end
