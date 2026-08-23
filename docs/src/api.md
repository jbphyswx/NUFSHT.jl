# API Reference

## Plan construction

```@docs
make_plan
NUSHTplan
close!
set_nodes!
```

### Node sets

A plan's points can be moved without rebuilding it. The node set fixes whether the point *count* may
change too.

```@docs
NUFSHT.AbstractNodeSet
NUFSHT.FixedCountNodes
NUFSHT.VariableCountNodes
```

### Plan tuning

How hard plan construction searches for the library settings the problem does not fix.

```@docs
NUFSHT.AbstractPlanTuning
NUFSHT.NoTuning
NUFSHT.AutoTuning
NUFSHT.ThoroughTuning
```

## Core transforms

```@docs
nusht_type2!
nusht_type1!
nusht_solve!
```

## Spin-weighted transforms

Synthesis/analysis/inversion of spin-weighted (`spin = s`) fields at arbitrary scattered
points, built from the Wigner-`d` Fourier factorization + a 2-D NUFFT. Spin-1 is the
tangent-vector case (velocity `u_θ + i u_φ`), enabling vector / Helmholtz decomposition on
scattered spherical data.

```@docs
make_spin_plan
SpinNUSHTplan
nusht_type2_spin!
nusht_type1_spin!
nusht_solve_spin!
WignerTable
sYlm
spin_coeff_index
NUFSHT.wigner_d
NUFSHT._wigner_d_halfpi_step!
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

Parallelism is selected by a `ComputationalBackends.AbstractExecutionBackend` argument, with the
backends themselves provided by the accelerator extensions (`using OhMyThreads` / `Distributed` /
`MPI`). See the performance section of the README for the threads-vs-processes trade-offs.

The collection entry points farm independent problems across a backend; `MPIBackend` instead
decomposes a single transform's points across ranks.

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
