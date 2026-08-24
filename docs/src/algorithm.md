# Algorithm

## Overview

NUFSHT.jl evaluates a spherical harmonic expansion at arbitrary scattered points by
writing it as a **Double Fourier Sphere (DFS) bivariate Fourier series** and handing
that series to a NUFFT. The synthesis operator
``A : \mathbb{R}^K \to \mathbb{R}^M`` (SH coefficients → scattered field values)
is decomposed as:

```math
A = N \cdot F \cdot S
```

| Step | Name | Size in → out | Cost |
|------|------|---------------|------|
| **S** | `plan_sph2fourier` | ``K \to N_\theta \times N_\phi`` | ``O(K \log K)`` via FastTransforms |
| **F** | `_assemble_modes!` | ``N_\theta \times N_\phi \to (2l_\text{max}+3) \times (2l_\text{max}+1)`` | ``O(K)`` |
| **N** | NUFFT type 2 | Fourier modes ``\to \mathbb{R}^M`` | ``O(K \log K + M)``, via the chosen backend |

where ``K = (l_\text{max}+1)(2l_\text{max}+1)`` is the number of SH coefficients and
``N_\theta = l_\text{max}+1``, ``N_\phi = 2l_\text{max}+1``.

## The bivariate Fourier series

`plan_sph2fourier` already produces the DFS series — that is what the transform is
for. There is no equiangular grid in the pipeline and nothing is doubled: forming
grid samples and Fourier-transforming them back would be a round trip from Fourier
coefficients to Fourier coefficients.

For order ``m`` in the `sph_mode` column packing, the series `S` returns is

```math
f(\theta,\varphi) = \sum_m \Phi_m(\varphi)\, N_m \sum_i G[i, c(m)]\, \Theta_{i,m}(\theta)
```

with ``\Phi_m = \cos(|m|\varphi)`` on the ``+m`` column and ``\sin(|m|\varphi)`` on the
``-m`` column, ``N_0 = 1/\sqrt{2\pi}``, ``N_{m \neq 0} = 1/\sqrt{\pi}``, and

```math
\Theta_{i,m}(\theta) = \begin{cases}
\cos((i-1)\theta) & |m| \text{ even} \\
\sin(i\theta) & |m| \text{ odd}
\end{cases}
```

That parity is not a storage convention. The DFS extension of ``Y_{\ell m}`` in
``\theta`` is ``\sin^{|m|}\theta \cdot Q(\cos\theta)``, which is even in ``\theta``
for even ``m`` and odd for odd ``m`` — the same fact the glide reflection
``f(\theta,\varphi) \mapsto f(2\pi-\theta, \varphi+\pi)`` encodes, expressed in a basis
where it costs nothing.

`_assemble_modes!` rewrites those cosines and sines as complex exponentials, giving the
mode array the NUFFT evaluates. The ``\theta`` axis runs ``-(l_\text{max}+1) \ldots
l_\text{max}+1``: the coefficient array is a *square, invertible* representation whose
supernumerary slots hold degrees ``l_\text{max} < l \leq l_\text{max}+|m|``, and an
odd-order column's last row is the sine at frequency ``l_\text{max}+1``.

## The adjoint

``A^\dagger : \mathbb{R}^M \to \mathbb{R}^K`` reverses the chain:

```math
A^\dagger = S^\dagger \cdot F^\dagger \cdot N^\dagger
```

| Step | Forward | Adjoint |
|------|---------|---------|
| ``N`` | NUFFT type 2 | NUFFT type 1 |
| ``F`` | `_assemble_modes!` | `_assemble_modes_adjoint!` (gather, real part for a real array) |
| ``S`` | `plan_sph2fourier` (``P``) | ``P'`` (FastTransforms conjugate plan) |

`nusht_type1!` **is** ``A^\dagger`` — the transpose, not an inverse. At scattered
points ``A^\dagger A \neq I``, and only quadrature on a grid makes the two coincide;
for exact analysis of a field already sampled on the Clenshaw-Curtis grid use
`FastSphericalHarmonics.sph_transform`. Inversion at scattered points is
`nusht_solve!`.

## LSMR inversion (`nusht_solve!`)

`nusht_solve!` solves

```math
\min_c \; \lVert A c - f \rVert
```

by **LSMR** (Fong & Saunders 2011) on the Golub–Kahan bidiagonalization of ``A``.
One ``A`` and one ``A^\dagger`` per iteration, the same cost as conjugate gradients on
the normal equations, but it works at ``\kappa(A)`` rather than ``\kappa(A)^2`` and
decreases ``\lVert A^\dagger r \rVert`` — the quantity the solve reports and stops on —
monotonically.

**Convergence** depends on ``\kappa(A)``, which is a property of the *point
distribution*, not of `lmax`. Measured on the design matrix over the ``l \leq
l_\text{max}`` modes at fourfold overdetermination:

| Point set | ``\kappa(A)`` | Iterations to ``10^{-10}`` |
|-----------|---------------|---------------------------|
| Golden-angle (Fibonacci) lattice | ``\approx 1.04`` | ``\approx 6`` |
| i.i.d. area-uniform | ``3`` – ``7`` | ``\approx 40`` |
| Half the points in a small polar cap | ``40`` – ``80`` | ``O(100)`` |

**Note:** the fit is over the ``(l_\text{max}+1)^2`` modes with ``l \leq l_\text{max}``,
which is the only ``SO(3)``-invariant choice; the supernumerary slots of the coefficient
array are excluded from the fit even though synthesis uses them. If the point set does
not determine those coefficients (``M`` below ``(l_\text{max}+1)^2``), the solve returns
the minimum-norm least-squares solution and reports `converged = false` when it cannot
reach `rtol`.

## The real fast path

The field element type passed to `make_plan` is a statement about the data, not a storage
preference: `Float64`/`Float32` assert the field **values** are real, and a real field's
mode array is Hermitian, ``Z[-k] = \overline{Z[k]}``. On a NUFFT backend with a genuine
real-data transform (`NonuniformFFTsBackend`; FINUFFT has none) only the ``k_\theta \geq 0``
half is then built — ``l_\text{max}+2`` rows instead of ``2 l_\text{max}+3`` — and only real
strengths come back, so the mode array, the upsampled FFT **and** the spreading/interpolation
all halve. Measured against the same backend's unfolded path at the same points and modes,
identical iteration counts, results agreeing to ``2 \times 10^{-16}``:

| ``l_\text{max}``, ``M`` | synthesis | `nusht_solve!` |
|---|---|---|
| 64, ``10^4`` | 1.52× | 1.51× |
| 64, ``10^5`` | 1.15× | 1.31× |
| 128, ``2\times10^4`` | 1.96× | 1.89× |
| 128, ``2\times10^5`` | 1.23× | 1.21× |

Near 2× where modes dominate, settling to ~1.2× where the point count does.

Forward, the fold is free: writing the half and letting the complex-to-real transform imply
the conjugate is exact, with no weights. Its **transpose** is not. The embedding
``E : Z_{1/2} \mapsto Z``, ``Z[-k] = \overline{Z[k]}``, is ``\mathbb{R}``-linear but not
``\mathbb{C}``-linear, so

```math
E^\dagger \hat{Z} = \hat{Z}[k] + \overline{\hat{Z}[-k]} = 2\,\hat{Z}[k] \quad (k_\theta > 0),
```

using ``\overline{\hat{Z}[-k]} = \hat{Z}[k]`` for real strengths. Hence the diagonal
``\{1, 2, 2, \ldots\}``: one at ``k_\theta = 0``, which is its own partner, two above it. There
is no third case at the top, because the ``\theta`` axis has odd length ``2 l_\text{max}+3``
and therefore no self-paired Nyquist row — an even axis would need one. A complex field, or a
backend without a real transform, uses the full array and no weights in either direction.

## References

- Merilees, P.E. (1973): The pseudospectral approximation applied to the shallow water equations on a sphere. *Atmosphere*, 11(1), 13–20.
- Townsend, A. & Olver, S. (2015): The automatic solution of partial differential equations using a global spectral method. *J. Comput. Phys.*, 299, 106–123.
- Reinecke, M. & Seljebotn, D.S. (2013): Libsharp – spherical harmonic transforms revisited. *A&A*, 554, A112. [doi:10.1051/0004-6361/201220728](https://doi.org/10.1051/0004-6361/201220728)
- Belkner, S. et al. (2024): cunuSHT – GPU Accelerated Spherical Harmonic Transforms on Arbitrary Pixelizations. *arXiv:2406.14542*.
