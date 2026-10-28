# API Reference

```@meta
CurrentModule = NUFSHT
```

## Plan construction

```@docs
make_plan
AbstractNUSHTplan
AbstractScalarNUSHTplan
NUSHTplan
RingNUSHTplan
TensorProduct
TensorProductSpinPlan
close!
set_nodes!
plan_memory
coefficient_size
allocate_coefficients
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

A plan's points can be moved without rebuilding it. The node set, chosen by `point_count`, fixes
whether the point *count* may change too.

```@docs
NUFSHT.AbstractNodeSet
NUFSHT.FixedCountNodes
NUFSHT.VariableCountNodes
NUFSHT.AbstractPointCount
NUFSHT.FixedCount
NUFSHT.VariableCount
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
NUFSHT._RingWork
NUFSHT._LatitudeWork
NUFSHT._ring_tables
NUFSHT._ring_check_interval
```

## Filtering

```@docs
nusht_synthesize!
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
NUFSHT._mode_entry
NUFSHT._mode_adjoint_entry
NUFSHT._filter_scratch
NUFSHT._valid_indices
NUFSHT._sph_pool
NUFSHT._plan_threads
NUFSHT._resolve_execution
NUFSHT._PointShare
```

### NUFFT backends and batched widths

```@docs
NUFSHT.DirectSumNUFFTPlan
NUFSHT._FTBNUFFT
NUFSHT._resolve_nufft
NUFSHT._real_capable
NUFSHT._nufft_share_directions
NUFSHT._nufft_derived
NUFSHT._width_narrowable
NUFSHT._pool_sizes
NUFSHT._nufft_size_pool
NUFSHT._pool_recipe
NUFSHT._build_width!
NUFSHT._pool_lookup!
NUFSHT._pool_built
```

### Spin plans: folds, the real-field fit and the ring engine

```@docs
NUFSHT.FullModes
NUFSHT._hermitian_domain
NUFSHT._herm_entry
NUFSHT._pack_entry
NUFSHT._pack_herm!
NUFSHT._unpack_herm!
NUFSHT._ring_engine
```
