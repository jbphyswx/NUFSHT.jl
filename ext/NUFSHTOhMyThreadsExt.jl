"""
    NUFSHTOhMyThreadsExt

Node-local multithreaded execution over a collection of **independent** problems (distinct point sets
→ distinct plans), one plan per task: a single FINUFFT plan's buffers are mutated in place and are not
safe to share across threads. Selected by passing a `ComputationalBackends.ThreadedBackend`. Loaded by
`using OhMyThreads`.

## FastTransforms threading inside Julia tasks

FastTransforms runs its transforms in OpenMP parallel regions (`ceil(CPU_THREADS/2)` threads by
default), which silently **corrupt results** when invoked from a non-root Julia task — see the
`NUFSHT._with_fasttransforms_single` comment for the mechanism and evidence. The scalar entry points
therefore bracket their whole `@tasks` section with FastTransforms single-threaded (the problem-level
parallelism comes from OhMyThreads, not FastTransforms' internal threads); wrapping the whole section
— not each task body — means the set/restore happen in the root task around the spawn/join barrier, so
concurrent worker tasks never race on the global thread count. Build the plans with `nthreads = 1` to
also keep FINUFFT from oversubscribing across concurrent tasks.

The spin entry points touch no FastTransforms state, so they need no such bracket.
"""
module NUFSHTOhMyThreadsExt

using NUFSHT: NUFSHT
using ComputationalBackends: ComputationalBackends
using OhMyThreads: OhMyThreads

# ── Scalar collections (bracketed: the S-step drives FastTransforms) ──────────

function NUFSHT.nusht_type2!(fs, Cs, plans::AbstractVector, ::ComputationalBackends.AbstractThreadedBackend)
    NUFSHT._check_farm(fs, Cs, plans)
    NUFSHT._with_fasttransforms_single() do
        OhMyThreads.@tasks for i in eachindex(plans)
            NUFSHT.nusht_type2!(fs[i], Cs[i], plans[i])
        end
    end
    return fs
end

function NUFSHT.nusht_type1!(Cs, fs, plans::AbstractVector, ::ComputationalBackends.AbstractThreadedBackend)
    NUFSHT._check_farm(Cs, fs, plans)
    NUFSHT._with_fasttransforms_single() do
        OhMyThreads.@tasks for i in eachindex(plans)
            NUFSHT.nusht_type1!(Cs[i], fs[i], plans[i])
        end
    end
    return Cs
end

function NUFSHT.nusht_solve!(Cs, fs, plans::AbstractVector, ::ComputationalBackends.AbstractThreadedBackend; kwargs...)
    NUFSHT._check_farm(Cs, fs, plans)
    NUFSHT._with_fasttransforms_single() do
        OhMyThreads.@tasks for i in eachindex(plans)
            NUFSHT.nusht_solve!(Cs[i], fs[i], plans[i]; kwargs...)
        end
    end
    return Cs
end

function NUFSHT.nusht_filter!(outs, ins, filter, plans::AbstractVector, ::ComputationalBackends.AbstractThreadedBackend)
    NUFSHT._check_farm(outs, ins, plans)
    NUFSHT._with_fasttransforms_single() do
        OhMyThreads.@tasks for i in eachindex(plans)
            NUFSHT.nusht_filter!(outs[i], ins[i], filter, plans[i])
        end
    end
    return outs
end

# ── Spin collections (no FastTransforms state, so no bracket) ─────────────────

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