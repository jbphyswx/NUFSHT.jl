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
| **S** | `plan_sph2fourier` | ``K \to N_\theta \times N_\phi`` | ``O(l_\text{max}^3)``: FastTransforms' Givens rotations and triangular products |
| **F** | `_assemble_modes!` | ``N_\theta \times N_\phi \to (2l_\text{max}+3) \times (2l_\text{max}+1)`` | ``O(K)`` |
| **N** | NUFFT type 2 | Fourier modes ``\to \mathbb{R}^M`` | ``O(K \log K + M)``, via the chosen backend |

where ``K = (l_\text{max}+1)(2l_\text{max}+1)`` is the number of SH coefficients and
``N_\theta = l_\text{max}+1``, ``N_\phi = 2l_\text{max}+1``.

## The bivariate Fourier series

`plan_sph2fourier` produces the DFS series directly, so the pipeline has no equiangular
grid and no doubling.

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

The parity comes from the harmonics: the DFS extension of ``Y_{\ell m}`` in ``\theta`` is
``\sin^{|m|}\theta \cdot Q(\cos\theta)``, which is even in ``\theta`` for even ``m`` and odd
for odd ``m``, the fact the glide reflection
``f(\theta,\varphi) \mapsto f(2\pi-\theta, \varphi+\pi)`` encodes.

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

`nusht_type1!` **is** ``A^\dagger``, the transpose, with no quadrature weights. At scattered
points ``A^\dagger A \neq I``. Quadrature weights ``w_j`` make the weighted adjoint
``A^\dagger W`` the analysis,

```math
c_k = \sum_j w_j f_j \overline{Y_k(x_j)},
```

exact for a degree-``l_\text{max}`` field when the rule integrates the product of two
degree-``l_\text{max}`` harmonics exactly; that is `nusht_analyze!(C, f, plan; weights)`. At
scattered points without such weights the analysis is the fit `nusht_solve!` (also
`nusht_analyze!` without weights).

## LSMR inversion (`nusht_solve!`)

`nusht_solve!` solves

```math
\min_c \; \lVert A c - f \rVert
```

by **LSMR** (Fong & Saunders 2011) on the Golub–Kahan bidiagonalization of ``A``.
One ``A`` and one ``A^\dagger`` per iteration, as for conjugate gradients on the normal
equations. LSMR's convergence is governed by ``\kappa(A)``, the normal equations' by
``\kappa(A)^2``, and it decreases ``\lVert A^\dagger r \rVert`` — the quantity the solve
reports and stops on — monotonically.

**Convergence** depends on ``\kappa(A)``, a property of the *point distribution*
independent of `lmax`. For the design matrix over the ``l \leq l_\text{max}`` modes at
fourfold overdetermination:

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

The field element type passed to `make_plan` is a statement about the data:
`Float64`/`Float32` assert the field **values** are real, and a real field's mode array is
Hermitian, ``Z[-k] = \overline{Z[k]}``. Only the ``k_\theta \geq 0`` half is therefore
stored, ``l_\text{max}+2`` of the ``2 l_\text{max}+3`` rows, on **every** backend: halving the
``\theta`` axis halves the deconvolution and the upsampled FFT and leaves the interpolation
untouched.

How the missing half is supplied splits into two cases. Both NUFFT libraries, through
FlowTransformBindings, take the half-spectrum of a transform built for the full ``\theta`` axis and
supply the rest themselves — with **real strengths**, so the spreading/interpolation halves too and
the forward needs no weights. Direct summation's transform is half-height and complex: its centered
rows are labelled ``k_\theta - \lceil\cdot\rceil``, which a per-point phase undoes, and the absent
conjugate half is paid for by doubling every ``k_\theta > 0`` row on the way in. The saving is
largest where the mode count dominates the cost.

Forward, the fold is free: writing the half and letting the complex-to-real transform imply
the conjugate is exact, with no weights. Its **transpose** is not. The embedding
``E : Z_{1/2} \mapsto Z``, ``Z[-k] = \overline{Z[k]}``, is ``\mathbb{R}``-linear but not
``\mathbb{C}``-linear, so

```math
E^\dagger \hat{Z} = \hat{Z}[k] + \overline{\hat{Z}[-k]} = 2\,\hat{Z}[k] \quad (k_\theta > 0),
```

using ``\overline{\hat{Z}[-k]} = \hat{Z}[k]`` for real strengths. Hence the diagonal
``\{1, 2, 2, \ldots\}``: one at ``k_\theta = 0``, which is its own partner, two above it. There
is no third case at the top: the ``\theta`` axis has odd length ``2 l_\text{max}+3`` and so no
self-paired Nyquist row. A complex field is not Hermitian and uses the full array with no
weights in either direction.

## Spin-weighted plans

A spin-``s`` plan synthesizes ``f = \sum_{\ell m} {}_s f_{\ell m}\, {}_sY_{\ell m}`` with
``{}_sY_{\ell m}(\theta,\varphi) = N_\ell\, d^\ell_{m,-s}(\theta)\, e^{im\varphi}``, written as the
bivariate Fourier series ``f = \sum_{m',m} G_{m'm}\, e^{-im'\theta} e^{im\varphi}`` that one NUFFT
evaluates. ``G`` comes from the ring S-step below.

### The ring S-step

Column ``m`` of ``G`` holds the Fourier coefficients of
``F_m(\theta) = \sum_\ell {}_s f_{\ell m} N_\ell\, d^\ell_{m,-s}(\theta)``. The ring S-step evaluates
``F_m`` at ``\theta_j = 2\pi j/N``, ``j = 0, \ldots, N/2``, with ``N \geq 2 l_\text{max} + 2`` even, by the
three-term recurrence in ``\ell`` (``n = -s``, ``x = \cos\theta``)

```math
\ell \sqrt{((\ell+1)^2-m^2)((\ell+1)^2-n^2)}\; d^{\ell+1}_{mn}
= (2\ell+1)\left(\ell(\ell+1)x - mn\right) d^\ell_{mn}
- (\ell+1)\sqrt{(\ell^2-m^2)(\ell^2-n^2)}\; d^{\ell-1}_{mn},
```

started from the closed form of ``d^{\ell_0}_{mn}`` at ``\ell_0 = \max(|m|,|n|)``. Since
``F_m(2\pi-\theta) = (-1)^{m+s} F_m(\theta)``, the half circle fixes ``F_m`` on all ``N`` samples, and one
length-``N`` FFT returns ``G_{m'm}`` for ``|m'| \leq l_\text{max}`` exactly.

Written as ``y_{\ell+1} = (\alpha x + \beta)\, y_\ell - \gamma\, y_{\ell-1}``, a rounding error ``\epsilon`` in
``\alpha x + \beta`` moves the angle the recurrence oscillates at by ``\epsilon / \sin\theta``, large at the
rings next to the poles. The S-step carries ``e_\ell = y_\ell - \sigma y_{\ell-1}`` with ``u = 1 - \sigma x``,
``\sigma = 1`` for ``\theta \leq \pi/2`` and ``-1`` beyond (Reinsch's modification),

```math
e_{\ell+1} = (c_\sigma - \sigma \alpha u)\, y_\ell + \sigma \gamma\, e_\ell, \qquad
y_{\ell+1} = \sigma y_\ell + e_{\ell+1},
```

with ``c_+ = \alpha + \beta - 1 - \gamma`` and ``c_- = \beta - \alpha + 1 + \gamma``, each formed as a sum of
terms of one sign. ``u`` is ``2\sin^2(\theta/2)`` or ``2\cos^2(\theta/2)``, accurate where it is small, and
the angle moves by ``\epsilon \tan(\theta/2)`` per step. Below its turning point
``d^\ell_{mn}`` is exponentially small, so the recurrence holds it as ``y\, 2^{Kk}`` with an integer
scale ``k \leq 0`` and adds it to the sum once ``k \geq -1``. The cost is
``O(l_\text{max}^3)``, with the work of each ring and degree a few fused multiply–adds on values in
registers, and the relative error is below ``l_\text{max}\, \epsilon`` of the plan's precision. The
adjoint runs the same recurrence and sums each degree over the rings.

## References

- Merilees, P.E. (1973): The pseudospectral approximation applied to the shallow water equations on a sphere. *Atmosphere*, 11(1), 13–20.
- Townsend, A. & Olver, S. (2015): The automatic solution of partial differential equations using a global spectral method. *J. Comput. Phys.*, 299, 106–123.
- Reinecke, M. & Seljebotn, D.S. (2013): Libsharp – spherical harmonic transforms revisited. *A&A*, 554, A112. [doi:10.1051/0004-6361/201220728](https://doi.org/10.1051/0004-6361/201220728)
- Belkner, S. et al. (2024): cunuSHT – GPU Accelerated Spherical Harmonic Transforms on Arbitrary Pixelizations. *arXiv:2406.14542*.
