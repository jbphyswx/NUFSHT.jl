"""
    NUFSHTDistributedExt

Coarse-grained farming of *independent* NUFSHT problems across `Distributed` **worker processes**,
selected by passing a `ComputationalBackends.DistributedBackend`. NUFFT plans hold library state that
cannot be serialized, so this extension implements the **node-set** entry points (`nusht_type2`,
`nusht_solve`): each worker builds its own plan from the node set it is given, runs the transform,
frees the plan, and returns the result. Loaded by `using Distributed`; workers need NUFSHT and the NUFFT
library their plans use (`@everywhere using NUFSHT, NonuniformFFTs`), or `Auto` gives them direct
summation.

A worker runs one problem at a time at the thread count it was started with: the NUFFT's `nthreads`
defaults to the worker's `Threads.nthreads()`, and FastTransforms runs at the same count
(`FlowTransformBindings.FASTTRANSFORMS_THREADS`), so `addprocs(n; exeflags = "-t k")` gives each
problem `k` threads. An `nthreads` keyword is passed to every plan. With no worker processes the
problems run serially on the calling process.
"""
module NUFSHTDistributedExt

using NUFSHT: NUFSHT
using ComputationalBackends: ComputationalBackends
using Distributed: Distributed
using FlowTransformBindings: FlowTransformBindings as FTB

# `work(i)` for i in 1:n on the worker processes when there are any, else serially here.
function _farm(work, n::Integer)
    Distributed.nprocs() == 1 && return [work(i) for i in 1:n]
    return Distributed.pmap(1:n) do i
        Base.ScopedValues.with(() -> work(i), FTB.FASTTRANSFORMS_THREADS => Threads.nthreads())
    end
end

function NUFSHT.nusht_type2(θs, φs, Cs, lmax, ::ComputationalBackends.AbstractDistributedBackend;
                            tol = 1e-8, kwargs...)
    @assert length(θs) == length(φs) == length(Cs) "θs, φs and Cs must have equal length"
    return _farm(length(θs)) do i
        plan = NUFSHT.make_plan(θs[i], φs[i], lmax; tol = tol, kwargs...)
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
                            tol = 1e-8, rtol = 1e-6, maxiter = 500, kwargs...)
    @assert length(θs) == length(φs) == length(fs) "θs, φs and fs must have equal length"
    return _farm(length(fs)) do i
        plan = NUFSHT.make_plan(θs[i], φs[i], lmax; tol = tol, kwargs...)
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
