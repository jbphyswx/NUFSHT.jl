"""
    NUFSHTOhMyThreadsExt

Node-local multithreaded execution over a collection of **independent** problems (distinct point sets
→ distinct plans), one plan per task: a single FINUFFT plan's buffers are mutated in place and are not
safe to share across threads. Selected by passing a `ComputationalBackends.ThreadedBackend`. Loaded by
`using OhMyThreads`.

The tasks carry the parallelism, so the scalar farms run FastTransforms on one OpenMP thread inside
them (`FlowTransformBindings.FASTTRANSFORMS_THREADS`). Build the plans with `nthreads = 1` to keep
FINUFFT from oversubscribing across concurrent tasks too. The spin path makes no FastTransforms calls.
"""
module NUFSHTOhMyThreadsExt

using NUFSHT: NUFSHT
using ComputationalBackends: ComputationalBackends
using FlowTransformBindings: FlowTransformBindings as FTB
using OhMyThreads: OhMyThreads

@inline _farm(f::F) where {F} = Base.ScopedValues.with(f, FTB.FASTTRANSFORMS_THREADS => 1)

# ── Scalar collections ────────────────────────────────────────────────────────

function NUFSHT.nusht_type2!(fs, Cs, plans::AbstractVector, ::ComputationalBackends.AbstractThreadedBackend)
    NUFSHT._check_farm(fs, Cs, plans)
    _farm() do
        OhMyThreads.@tasks for i in eachindex(plans)
            NUFSHT.nusht_type2!(fs[i], Cs[i], plans[i])
        end
    end
    return fs
end

function NUFSHT.nusht_type1!(Cs, fs, plans::AbstractVector, ::ComputationalBackends.AbstractThreadedBackend)
    NUFSHT._check_farm(Cs, fs, plans)
    _farm() do
        OhMyThreads.@tasks for i in eachindex(plans)
            NUFSHT.nusht_type1!(Cs[i], fs[i], plans[i])
        end
    end
    return Cs
end

function NUFSHT.nusht_solve!(Cs, fs, plans::AbstractVector, ::ComputationalBackends.AbstractThreadedBackend; kwargs...)
    NUFSHT._check_farm(Cs, fs, plans)
    _farm() do
        OhMyThreads.@tasks for i in eachindex(plans)
            NUFSHT.nusht_solve!(Cs[i], fs[i], plans[i]; kwargs...)
        end
    end
    return Cs
end

function NUFSHT.nusht_filter!(outs, ins, filter, plans::AbstractVector, ::ComputationalBackends.AbstractThreadedBackend)
    NUFSHT._check_farm(outs, ins, plans)
    _farm() do
        OhMyThreads.@tasks for i in eachindex(plans)
            NUFSHT.nusht_filter!(outs[i], ins[i], filter, plans[i])
        end
    end
    return outs
end

# ── Spin collections ──────────────────────────────────────────────────────────

function NUFSHT.nusht_type2_spin!(fs, sfs, plans::AbstractVector, ::ComputationalBackends.AbstractThreadedBackend)
    NUFSHT._check_farm(fs, sfs, plans)
    OhMyThreads.@tasks for i in eachindex(plans)
        NUFSHT.nusht_type2_spin!(fs[i], sfs[i], plans[i])
    end
    return fs
end

function NUFSHT.nusht_type1_spin!(sfs, fs, plans::AbstractVector, ::ComputationalBackends.AbstractThreadedBackend)
    NUFSHT._check_farm(sfs, fs, plans)
    OhMyThreads.@tasks for i in eachindex(plans)
        NUFSHT.nusht_type1_spin!(sfs[i], fs[i], plans[i])
    end
    return sfs
end

function NUFSHT.nusht_solve_spin!(sfs, fs, plans::AbstractVector, ::ComputationalBackends.AbstractThreadedBackend; kwargs...)
    NUFSHT._check_farm(sfs, fs, plans)
    OhMyThreads.@tasks for i in eachindex(plans)
        NUFSHT.nusht_solve_spin!(sfs[i], fs[i], plans[i]; kwargs...)
    end
    return sfs
end

end # module NUFSHTOhMyThreadsExt
