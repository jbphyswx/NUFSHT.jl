"""
    NUFSHTOhMyThreadsExt

Node-local multithreaded execution of NUFSHT transforms over a collection of **independent**
problems (distinct point sets → distinct plans), one plan per task (a single FINUFFT plan's buffers
are mutated in place and are not safe to share across threads). Loaded by `using OhMyThreads`.

## FastTransforms threading inside Julia tasks

FastTransforms runs its transforms in OpenMP parallel regions (`ceil(CPU_THREADS/2)` threads by
default), which silently **corrupt results** when invoked from a non-root Julia task — see the
`NUFSHT._with_fasttransforms_single` comment for the mechanism and evidence. This extension therefore
brackets its whole `@tasks` section with FastTransforms single-threaded (the problem-level parallelism
comes from OhMyThreads, not FastTransforms' internal threads); wrapping the whole section — not each
task body — means the set/restore happen in the root task around the spawn/join barrier, so the
concurrent worker tasks never race on the global thread count. Build the plans with `nthreads = 1` to
also keep FINUFFT from oversubscribing across concurrent tasks.
"""
module NUFSHTOhMyThreadsExt

using NUFSHT: NUFSHT
using OhMyThreads: OhMyThreads

function NUFSHT.nusht_type2_threaded!(fs, Cs, plans)
    @assert length(fs) == length(Cs) == length(plans) "fs, Cs, plans must have equal length"
    NUFSHT._with_fasttransforms_single() do
        OhMyThreads.@tasks for i in eachindex(plans)
            NUFSHT.nusht_type2!(fs[i], Cs[i], plans[i])
        end
    end
    return fs
end

function NUFSHT.nusht_type1_threaded!(Cs, fs, plans)
    @assert length(Cs) == length(fs) == length(plans) "Cs, fs, plans must have equal length"
    NUFSHT._with_fasttransforms_single() do
        OhMyThreads.@tasks for i in eachindex(plans)
            NUFSHT.nusht_type1!(Cs[i], fs[i], plans[i])
        end
    end
    return Cs
end

function NUFSHT.nusht_solve_threaded!(Cs, fs, plans; kwargs...)
    @assert length(Cs) == length(fs) == length(plans) "Cs, fs, plans must have equal length"
    NUFSHT._with_fasttransforms_single() do
        OhMyThreads.@tasks for i in eachindex(plans)
            NUFSHT.nusht_solve!(Cs[i], fs[i], plans[i]; kwargs...)
        end
    end
    return Cs
end

end # module NUFSHTOhMyThreadsExt
