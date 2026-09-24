"""
    NUFSHTDistributedExt

Coarse-grained farming of *independent* NUFSHT problems across `Distributed` **worker processes**,
selected by passing a `ComputationalBackends.DistributedBackend`. FINUFFT plans hold C pointers and
cannot be serialized, so this extension implements the **node-set** entry points (`nusht_type2`,
`nusht_solve`): each worker builds its own plan from the node set it is given, runs the transform,
frees the plan, and returns the result. Loaded by `using Distributed` (with `@everywhere using
NUFSHT` so workers have the package).

With no worker processes the problems run serially on the calling task. On a worker the processes
carry the parallelism, so FastTransforms runs on one OpenMP thread there
(`FlowTransformBindings.FASTTRANSFORMS_THREADS`), and plans are built with `nthreads = 1` by default:
multi-threaded FINUFFT type-1 spreading is not bit-reproducible, and one thread per transform does not
oversubscribe cores across concurrently farmed problems.
"""
module NUFSHTDistributedExt

using NUFSHT: NUFSHT
using ComputationalBackends: ComputationalBackends
using Distributed: Distributed
using FlowTransformBindings: FlowTransformBindings as FTB

# `work(i)` for i in 1:n on the worker processes when there are any, else serially here.
function _farm(work, n::Integer)
    Distributed.nworkers() == 1 && return [work(i) for i in 1:n]
    return Distributed.pmap(1:n) do i
        Base.ScopedValues.with(() -> work(i), FTB.FASTTRANSFORMS_THREADS => 1)
    end
end

function NUFSHT.nusht_type2(θs, φs, Cs, lmax, ::ComputationalBackends.AbstractDistributedBackend;
                            tol = 1e-8, nthreads = 1, kwargs...)
    @assert length(θs) == length(φs) == length(Cs) "θs, φs and Cs must have equal length"
    return _farm(length(θs)) do i
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
    return _farm(length(fs)) do i
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
