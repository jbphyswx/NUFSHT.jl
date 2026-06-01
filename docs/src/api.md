# API Reference

## Plan construction

```@docs
make_plan
NUSHTplan
```

## Core transforms

```@docs
nusht_type2!
nusht_type1!
nusht_solve!
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

## DFS utilities (internal / advanced)

```@docs
NUFSHT.dfs_double
NUFSHT.dfs_fold
NUFSHT.dfs_grid_coords
NUFSHT.dfs_doubled_grid_coords
```

## Internal helpers

```@docs
NUFSHT.fft2_to_coeffs
NUFSHT.ifft2_from_coeffs
NUFSHT.apply_transfer!
NUFSHT._nusht_true_adjoint!
```
