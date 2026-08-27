"""
    NUFSHTDistributedExt

Coarse-grained farming of *independent* NUFSHT problems across `Distributed` **worker processes**,
selected by passing a `ComputationalBackends.DistributedBackend`. FINUFFT plans hold C pointers and
cannot be serialized, so this extension implements the **node-set** entry points (`nusht_type2`,
`nusht_solve`): each worker builds its own plan from the node set it is given, runs the transform,
frees the plan, and returns the result. Loaded by `using Distributed` (with `@everywhere using
NUFSHT` so workers have the package).

## Why processes, not local tasks

FastTransforms (the Legendre step) runs in OpenMP regions that corrupt process-global state when
driven from a non-root Julia task — and empirically `Distributed.pmap` on the *local* process
(`nworkers() == 1`, no `addprocs`) corrupts even a single-threaded plan, because it still runs the
work in a message-handler task on the main process. Separate worker processes are safe: each runs on
its own root task with its own FastTransforms state. This extension therefore:

  * when there are **no** worker processes, runs the problems **serially on the main task** (nothing
    to distribute; FastTransforms is safe on the root task — no thread manipulation needed); and
  * when there **are** workers, `pmap`s to them, each worker forcing FastTransforms single-threaded
    in its own process via `NUFSHT._fasttransforms_single!()` for task-safe transforms there.

Plans are built with `nthreads = 1` by default: a farm's parallelism is across problems/processes, so
each transform is single-threaded — this is deterministic (multi-threaded FINUFFT type-1 spreading is
not bit-reproducible) and avoids oversubscribing cores across concurrently farmed problems.
"""
module NUFSHTDistributedExt

using NUFSHT: NUFSHT
using ComputationalBackends: ComputationalBackends
using Distributed: Distributed

# Run `work(i)` for i in 1:n on worker processes when available, else serially on the main task.
# `on_worker` is invoked inside each worker's remotecall task (to set that process's FastTransforms
# state); it is NOT called on the serial main-task path, which needs no thread manipulation.
function _farm(work, on_worker, n::Integer)
    if Distributed.nworkers() == 1
        return [work(i) for i in 1:n]
    else
        return Distributed.pmap(1:n) do i
            on_worker()
            return work(i)
        end
    end
end

function NUFSHT.nusht_type2(θs, φs, Cs, lmax, ::ComputationalBackends.AbstractDistributedBackend;
                            tol = 1e-8, nthreads = 1, kwargs...)
    @assert length(θs) == length(φs) == length(Cs) "θs, φs and Cs must have equal length"
    return _farm(NUFSHT._fasttransforms_single!, length(θs)) do i
        plan = NUFSHT.make_plan(θs[i], φs[i], lmax; tol = tol, nthreads = nthreads, kwargs...)
        try
            f = zeros(eltype(plan.F), length(θs[i]))
            NUFSHT.nusht_type2!(f, Cs[i], plan)
            return f
        finally
            NUFSHT.close!(plan)
        end
    end
end

function NUFSHT.nusht_solve(θs, φs, fs, lmax, ::ComputationalBackends.AbstractDistributedBackend;
                            tol = 1e-8, nthreads = 1,
                            rtol = 1e-6, maxiter = 500, kwargs...)
    @assert length(θs) == length(φs) == length(fs) "θs, φs and fs must have equal length"
    return _farm(NUFSHT._fasttransforms_single!, length(fs)) do i
        plan = NUFSHT.make_plan(θs[i], φs[i], lmax; tol = tol, nthreads = nthreads, kwargs...)
        try
            C = zeros(eltype(plan.F), lmax + 1, 2lmax + 1)
            NUFSHT.nusht_solve!(C, fs[i], plan; rtol = rtol, maxiter = maxiter)
            return C
        finally
            NUFSHT.close!(plan)
        end
    end
end

end # module NUFSHTDistributedExt