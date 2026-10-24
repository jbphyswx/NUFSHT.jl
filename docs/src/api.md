# API Reference

## Plan construction

```@docs
make_plan
NUSHTplan
close!
set_nodes!
plan_memory
```

### Which directions a plan holds

`directions` says which transform directions a plan holds; asking a synthesis-only plan to analyse
raises. Where a backend's plan carries no direction of its own, both directions share one object and
the choice costs nothing either way.

```@docs
NUFSHT.AbstractPlanDirections
NUFSHT.SynthesisOnly
NUFSHT.SynthesisAndAnalysis
```

### Node sets

A plan's points can be moved without rebuilding it. The node set fixes whether the point *count* may
change too.

```@docs
NUFSHT.AbstractNodeSet
NUFSHT.FixedCountNodes
NUFSHT.VariableCountNodes
```

## Core transforms

```@docs
nusht_type2!
nusht_type1!
nusht_analyze!
nusht_solve!
```

## Spin-weighted transforms

Synthesis/analysis/inversion of spin-weighted (`spin = s`) fields at arbitrary scattered
points: the ring S-step turns the coefficients into a bivariate Fourier series and a 2-D NUFFT
evaluates it. Spin-1 is the
tangent-vector case (velocity `u_θ + i u_φ`), enabling vector / Helmholtz decomposition on
scattered spherical data.

```@docs
make_spin_plan
SpinNUSHTplan
nusht_type2_spin!
nusht_type1_spin!
nusht_solve_spin!
sYlm
spin_coeff_index
scalar_to_spin0!
spin0_to_scalar!
NUFSHT.wigner_d
NUFSHT._RingEngine
NUFSHT._ring_tables
NUFSHT._ring_check_interval
```

## Filtering

```@docs
nusht_filter!
nusht_filter_renorm!
```

## Transfer functions (spectral filters)

```@docs
GaussianTransfer
gaussian_from_scale
TopHatTransfer
SharpSpectralTransfer
kernel_transfer
cutoff_degree
```

## Parallel execution

Parallelism is selected by a `ComputationalBackends.AbstractExecutionBackend`, with the backends
themselves provided by the accelerator extensions (`using OhMyThreads` / `KernelAbstractions` /
`Distributed` / `MPI`). See the performance section of the README for the threads-vs-processes
trade-offs.

`make_plan` and `make_spin_plan` take the `execution` keyword for where the plan runs, and the
collection entry points farm independent problems across it. `nusht_type1!`, `nusht_analyze!`,
`nusht_solve!` and `nusht_solve_spin!` take a backend as a fourth argument, `MPIBackend(; comm)` for
instance, and then treat the plan as one rank's share of a single transform's points.

```@docs
nusht_type2
nusht_solve
```

## Plotting

```@docs
plot_field
```

## Internal helpers

```@docs
NUFSHT.apply_transfer!
NUFSHT._nusht_true_adjoint!
NUFSHT._assemble_modes!
NUFSHT._assemble_modes_adjoint!
```
