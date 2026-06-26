# Spin-1 scattered synthesis (issue #1) — progress notes

Branch: `feature/spin1-scattered-synthesis`. Goal: spin-weighted (spin-1) synthesis +
adjoint + solve at scattered points, to unblock scattered-spherical Helmholtz decomposition
in HelmholtzDecomposition.jl (`U = u_east + i·u_north = −ð(χ+iψ)` ⇒ one complex spin-1
inverse + synthesis).

## Solved / de-risked
- **Feasibility:** FastTransforms `plan_spinsph2fourier(C,s)` + `plan_spinsph_synthesis` +
  `plan_spinsph_analysis` exist and operate on `(Nθ,Nφ)=(lmax+1, 2lmax+1)` COMPLEX coefficient
  arrays; synthesis↔analysis round-trips to ~1e-15.
- **Coefficient layout (reverse-engineered):** same column convention as scalar `sph_mode`,
  row offset by the spin floor:
    col(m) = 2|m| + (m≥0 ? 1 : 0)
    row(ℓ,m,s) = ℓ − max(|m|,|s|) + 1     (valid for ℓ ≥ max(|m|,|s|))
  Verified by probing (column→m via φ-FFT; row→ℓ via θ sign-changes).

## The hard core (NOT yet solved)
The scalar NUFSHT pipeline is `N·F·D·S` where only `S` is scalar-specific. Swapping `S` for
the spin synthesis and reusing the scalar DFS doubling `D` (reflect rows across the pole with
φ+π shift, factor (−1)^s) **fails**: rel_err ≈ 1.8 vs an exact finer-grid reference (coeff
embedding into lmax2, FastTransforms synthesis on the finer CC grid). Crucially the error is
O(1) and identical for sign = ±1 even when evaluating at the ORIGINAL CC grid points — so the
spin double-Fourier-sphere extension is structurally different from the scalar even-extension,
not merely a sign/phase. (`dev/spin_validation_harness.jl` reproduces this.)

## Path forward
Implement the genuine spin-weighted DFS construction (ssht/s2fft "doubling" for spin, or the
spin-weighted associated-Legendre / Wigner-d recurrence). The reflected part of a spin-s
function carries m,s-dependent factors and the spin sign may flip (s→−s) under the pole
reflection; the θ-Fourier basis for spin-s is shifted relative to scalar. References in the
issue: pyspherical, ssht/s2fft, Huffenberger & Wandelt, Price & McEwen (2024).

Validation harness (`dev/spin_validation_harness.jl`) gives a convention-free exact reference
(finer-grid via coeff embedding) to iterate the doubling rule against.
