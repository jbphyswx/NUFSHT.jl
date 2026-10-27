"""
    NUFSHTOhMyThreadsExt

Node-local multithreaded execution over a collection of **independent** problems (distinct point sets
→ distinct plans), one plan per task: a NUFFT plan's buffers are mutated in place and are not safe to
share across threads. Selected by `execution = ComputationalBackends.ThreadedBackend()`. Loaded by
`using OhMyThreads`.

The tasks carry the parallelism, so every farm runs FastTransforms on one OpenMP thread inside them
(`FlowTransformBindings.FASTTRANSFORMS_THREADS`). Build the plans with `nthreads = 1` so the NUFFT
does not oversubscribe the cores across concurrent tasks either.
"""
module NUFSHTOhMyThreadsExt

using NUFSHT: NUFSHT
using ComputationalBackends: ComputationalBackends
using FlowTransformBindings: FlowTransformBindings as FTB
using OhMyThreads: OhMyThreads

NUFSHT._threaded_collections_available(::NUFSHT._ThreadedCollections) = true

@inline _farm(f::F) where {F} = Base.ScopedValues.with(f, FTB.FASTTRANSFORMS_THREADS => 1)

# ── Scalar collections ────────────────────────────────────────────────────────

function NUFSHT._farm_type2!(fs, Cs, plans::AbstractVector, ::ComputationalBackends.AbstractThreadedBackend)
    NUFSHT._check_farm(fs, Cs, plans)
    _farm() do
        OhMyThreads.@tasks for i in eachindex(plans)
            NUFSHT.nusht_type2!(fs[i], Cs[i], plans[i])
        end
    end
    return fs
end

function NUFSHT._farm_type1!(Cs, fs, plans::AbstractVector, ::ComputationalBackends.AbstractThreadedBackend)
    NUFSHT._check_farm(Cs, fs, plans)
    _farm() do
        OhMyThreads.@tasks for i in eachindex(plans)
            NUFSHT.nusht_type1!(Cs[i], fs[i], plans[i])
        end
    end
    return Cs
end

function NUFSHT._farm_solve!(Cs, fs, plans::AbstractVector, ::ComputationalBackends.AbstractThreadedBackend; kwargs...)
    NUFSHT._check_farm(Cs, fs, plans)
    _farm() do
        OhMyThreads.@tasks for i in eachindex(plans)
            NUFSHT.nusht_solve!(Cs[i], fs[i], plans[i]; kwargs...)
        end
    end
    return Cs
end

# Scalar or spin plans.
function NUFSHT._farm_analyze!(Cs, fs, plans::AbstractVector, weights,
                               ::ComputationalBackends.AbstractThreadedBackend; kwargs...)
    NUFSHT._check_farm(Cs, fs, plans)
    NUFSHT._check_farm_weights(weights, plans)
    _farm() do
        OhMyThreads.@tasks for i in eachindex(plans)
            NUFSHT.nusht_analyze!(Cs[i], fs[i], plans[i]; weights = NUFSHT._problem_weights(weights, i),
                                  kwargs...)
        end
    end
    return Cs
end

function NUFSHT._farm_filter!(outs, ins, filter, plans::AbstractVector, ::ComputationalBackends.AbstractThreadedBackend)
    NUFSHT._check_farm(outs, ins, plans)
    _farm() do
        OhMyThreads.@tasks for i in eachindex(plans)
            NUFSHT.nusht_filter!(outs[i], ins[i], filter, plans[i])
        end
    end
    return outs
end

# ── Spin collections ──────────────────────────────────────────────────────────

function NUFSHT._farm_type2_spin!(fs, sfs, plans::AbstractVector, ::ComputationalBackends.AbstractThreadedBackend)
    NUFSHT._check_farm(fs, sfs, plans)
    _farm() do
        OhMyThreads.@tasks for i in eachindex(plans)
            NUFSHT.nusht_type2_spin!(fs[i], sfs[i], plans[i])
        end
    end
    return fs
end

function NUFSHT._farm_type1_spin!(sfs, fs, plans::AbstractVector, ::ComputationalBackends.AbstractThreadedBackend)
    NUFSHT._check_farm(sfs, fs, plans)
    _farm() do
        OhMyThreads.@tasks for i in eachindex(plans)
            NUFSHT.nusht_type1_spin!(sfs[i], fs[i], plans[i])
        end
    end
    return sfs
end

function NUFSHT._farm_solve_spin!(sfs, fs, plans::AbstractVector, ::ComputationalBackends.AbstractThreadedBackend; kwargs...)
    NUFSHT._check_farm(sfs, fs, plans)
    _farm() do
        OhMyThreads.@tasks for i in eachindex(plans)
            NUFSHT.nusht_solve_spin!(sfs[i], fs[i], plans[i]; kwargs...)
        end
    end
    return sfs
end

end # module NUFSHTOhMyThreadsExt
