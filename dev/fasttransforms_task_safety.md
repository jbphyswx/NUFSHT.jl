# FastTransforms wrong-results-inside-a-Julia-Task — findings & reproducer

Local notes for study. **Not filed upstream.** The workaround is already applied in NUFSHT's
task-based extensions (`_with_fasttransforms_single` / `_fasttransforms_single!`). Decide separately
whether to report upstream.

## Claim (verified airtight, this machine)

A `plan_sph2fourier` transform silently returns a **wrong result** when the `ccall` executes inside a
non-root Julia `Task` (`@async`, `Threads.@spawn`, a `Distributed` worker's message-handler task).
Not a crash; not a shared-plan data race — the plan and input are built on the main task and only the
multiply runs in the task. Forcing FastTransforms single-threaded makes it exact. **Reproduces at
`julia -t1`** (one OS thread) → it's the library's internal OpenMP region entered from a non-root
task, not Julia thread migration.

## MWE

```julia
using FastTransforms, LinearAlgebra
A = FastTransforms.sphrandn(Float64, 64, 127)   # valid coeff array, MAIN task
P = plan_sph2fourier(A)                          # plan built on MAIN task
Bmain = P * A                                    # transform on MAIN task

Btask = fetch(@async (P * A))                    # identical P*A, inside a Task
println("rel err (task vs main): ", norm(Btask - Bmain) / norm(Bmain))

FastTransforms.ft_set_num_threads(1); FastTransforms.ft_fftw_plan_with_nthreads(1)
Bft1 = fetch(@async (P * A))
println("rel err (task, ft=1):   ", norm(Bft1 - Bmain) / norm(Bmain))
```

Observed:
```
rel err (task vs main): 1.6778552593796554     # WRONG (also ~0.5 for a sph2fourier roundtrip)
rel err (task, ft=1):   0.0                     # exact once single-threaded
```
Same at `-t1` and `-t4`; same with `Threads.@spawn`.

## Environment
```
Julia Version 1.12.6 (2026-04-09), x86_64-apple-darwin24.0.0, 8 × i7-7700K (4 physical)
FastTransforms  v0.17.2
FINUFFT         v3.5.2   (FFTW-backed finufft_jll; has the FFTW planner lock)
FFTW            v1.10.0
```
FastTransforms `__init__` sets both OpenMP and FFTW to `ceil(CPU_THREADS/2)` threads regardless of
Julia's `-t`.

## Context from the maintainers (already-documented, related)
- **FT.jl #250** (Slevinsky): "FastTransforms uses multithreading via OpenMP … the threading occurs
  inside the C library, not through Julia." Control via `ft_set_num_threads`.
- **ClassicalOrthogonalPolynomials.jl #204** (Olver): "Rule number 1 for multithreading: Don't do
  it! … plans … store temporary working vectors which are changed. … Instead: do a matrix transform.
  The multithreading is taken care of by LibFastTransforms." — documents the *shared-plan* `@threads`
  race (wrong results), which is related but distinct from this fresh-plan/non-root-task case.
- **FT.jl #267**: same `__init__` OpenMP threading manifests as a ~100× FFTW *slowdown*; single-
  threading is the posted workaround. This reproducer is the *correctness* face of the same root.
- **FFTW manual (Thread safety)**: only `fftw_execute` is thread-safe; the planner must be called
  from one thread at a time.

## What upstream does NOT document
The specific "fresh plan, single non-root task, `-t1` → silently wrong numbers" symptom is not in the
FastTransforms.jl / FFTW.jl trackers (they have the slowdown #267, a crash #238, and the shared-plan
race COP#204). This MWE is the missing correctness case — hence worth reporting if desired.

## Open question for maintainers (if reported)
Is calling FastTransforms from a non-root Julia Task simply unsupported (documented model = "call from
the main thread; parallelize via internal OpenMP / matrix transforms")? Or is a thread-safe path
intended (a lock / build flag, analogous to FINUFFT's `fftw_lock_fun` and FFTW.jl's planner lock)?
