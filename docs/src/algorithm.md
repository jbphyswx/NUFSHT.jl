# Algorithm

## Overview

NUFSHT.jl implements the **Double Fourier Sphere (DFS) + NUFFT** algorithm
(Reinecke & Seljebotn 2013, Belkner et al. 2024). The synthesis operator
``A : \mathbb{R}^K \to \mathbb{R}^M`` (SH coefficients → scattered field values)
is decomposed as:

```math
A = N \cdot F \cdot D \cdot S
```

and evaluated efficiently by chaining four sub-operations:

| Step | Name | Size in → out | Cost |
|------|------|---------------|------|
| **S** | Iso-latitude rSHT | ``K \to N_\theta \times N_\phi`` | ``O(K \log K)`` via FastTransforms |
| **D** | DFS doubling | ``N_\theta \times N_\phi \to 2N_\theta \times N_\phi`` | ``O(K)`` |
| **F** | 2D FFT + phase | ``2N_\theta \times N_\phi \to`` Fourier modes | ``O(K \log K)`` via FFTW |
| **N** | NUFFT type 2 | Fourier modes ``\to \mathbb{R}^M`` | ``O(K \log K + M)`` via FINUFFT |

where ``K = (l_\text{max}+1)(2l_\text{max}+1)`` is the number of SH coefficients and
``N_\theta = l_\text{max}+1``, ``N_\phi = 2l_\text{max}+1``.

## The DFS doubling step

The Clenshaw-Curtis (CC) grid covers colatitudes ``\theta \in (0, \pi)`` on an
open interval (no poles). The DFS method extends this to a doubly-periodic function
on the torus ``[0, 2\pi) \times [0, 2\pi)`` by reflecting across the south pole:

```math
\tilde{F}[N_\theta + i,\, j] = F\bigl[N_\theta + 1 - i,\; \text{mod}_1(j + \lfloor N_\phi/2 \rfloor,\, N_\phi)\bigr]
\quad i = 1,\ldots,N_\theta
```

The ``\phi + \pi`` shift ensures the reflected field is smooth across the south
pole. The shift is implemented as `mod1(j + Nφ÷2, Nφ)` — a proper cyclic
permutation valid for **any** ``N_\phi``, including odd values (``N_\phi = 2l_\text{max}+1``
is always odd).

> **Subtle bug fixed:** The naive conditional shift ``j \leq N_\phi/2 \;?\; j + N_\phi/2 : j - N_\phi/2``
> is **not a bijection** for odd ``N_\phi``: two columns map to the same target, silently
> overwriting data. `mod1` fixes this.

## The adjoint

The adjoint ``A^\dagger : \mathbb{R}^M \to \mathbb{R}^K`` reverses the chain:

```math
A^\dagger = S^\dagger \cdot D^\dagger \cdot F^\dagger \cdot N^\dagger
```

Each step has a corresponding adjoint:

| Step | Forward | Adjoint |
|------|---------|---------|
| ``N`` | `nufft2d2` (type 2) | `nufft2d1` (type 1) |
| ``F`` | FFT + phase ``e^{-i\pi k_\theta / N_\theta}`` | Conjugate phase + IFFT |
| ``D`` | `dfs_double!` (shift ``+N_\phi/2``) | `dfs_fold!` (shift ``-N_\phi/2``, accumulate) |
| ``S`` | `sph_evaluate!` (``PS \cdot P``) | `PS' \cdot P'`` (FastTransforms conjugate plans) |

### `nusht_type1!` vs `_nusht_true_adjoint!`

`nusht_type1!` uses `sph_transform!` for the ``S^\dagger`` step, which is the
**exact inverse** of `sph_evaluate!` on the CC grid (not the Euclidean matrix transpose).
On the CC grid, CC quadrature makes ``A^\dagger A = I`` so it gives machine-precision
round-trips. Off the CC grid, it is only an approximation.

`nusht_solve!` uses `_nusht_true_adjoint!` internally, which applies
``PS' \cdot P'`` (the exact matrix-transpose adjoint via FastTransforms conjugate plans).
This makes ``A^\dagger A`` symmetric positive definite for **any** scattered point
distribution, enabling guaranteed convergence of Conjugate Gradients.

## The `dfs_fold!` adjoint identity

Given the doubling operator ``D``, its matrix-transpose adjoint ``D^\dagger``
satisfies:

```math
\langle D^\dagger \tilde{u},\, v \rangle = \langle \tilde{u},\, D v \rangle
\quad \forall\, v \in \mathbb{R}^{N_\theta \times N_\phi},\; \tilde{u} \in \mathbb{R}^{2N_\theta \times N_\phi}
```

In `dfs_fold!` this is:

```math
F[i,j] = \tilde{F}[i,j] + \tilde{F}[2N_\theta + 1 - i,\; \text{mod}_1(j - \lfloor N_\phi/2 \rfloor,\, N_\phi)]
```

The **inverse** shift ``-N_\phi/2 \pmod{N_\phi}`` is essential: for odd ``N_\phi``,
``+N_\phi/2`` and ``-N_\phi/2`` are different column permutations.

## Conjugate Gradient inversion (`nusht_solve!`)

`nusht_solve!` solves the normal equations:

```math
(A^\dagger A)\, c = A^\dagger f
```

using the standard Conjugate Gradient algorithm. Since ``A^\dagger A`` is symmetric
positive definite (with the true Euclidean adjoint), CG converges monotonically.

**Convergence** depends on the condition number ``\kappa(A^\dagger A)``, which
scales with the point distribution. For jittered-uniform points with ``M \approx 4K``,
typical iteration counts are ``O(100)``–``O(500)`` for ``l_\text{max} \leq 20``.

**Note:** `nusht_solve!` recovers the field ``f`` to the NUFFT tolerance but does
not necessarily recover the exact input coefficients ``c_\text{true}`` if the
true field was generated from a narrower bandwidth than `lmax`. The CG solution
is the minimum-norm least-squares solution.

## Phase convention

The CC grid places data at cell centres
``\theta_i = \pi(i - 1/2)/N_\theta`` for ``i = 1,\ldots,N_\theta``, which are
offset by half a pixel from FFTW's assumed integer-index positions.

The FFT step applies a per-mode phase correction
``e^{-i\pi k_\theta / N_\theta}`` to compensate, so that FINUFFT evaluated at
natural ``\theta \in [0,\pi]`` coordinates gives the correct values.

## References

- Merilees, P.E. (1973): The pseudospectral approximation applied to the shallow water equations on a sphere. *Atmosphere*, 11(1), 13–20.
- Townsend, A. & Olver, S. (2015): The automatic solution of partial differential equations using a global spectral method. *J. Comput. Phys.*, 299, 106–123.
- Reinecke, M. & Seljebotn, D.S. (2013): Libsharp – spherical harmonic transforms revisited. *A&A*, 554, A112. [doi:10.1051/0004-6361/201220728](https://doi.org/10.1051/0004-6361/201220728)
- Belkner, S. et al. (2024): cunuSHT – GPU Accelerated Spherical Harmonic Transforms on Arbitrary Pixelizations. *arXiv:2406.14542*.
