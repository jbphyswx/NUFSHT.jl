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

## CORRECTION (important)
The standalone harness's hand-rolled fft2_to_coeffs+FINUFFT replication is itself buggy: it
fails the pure DFT->IDFT interpolation identity (arbitrary doubled F̃ -> FFT+phase -> FINUFFT
type2 at the doubled grid -> should return F̃, but rel_err≈1.6, NO spin involved). So the
"naive (−1)^s doubling fails" result is CONFOUNDED by this machinery bug, not conclusive.
Correct plan: do NOT re-derive the FFT/phase/FINUFFT layout in a prototype — reuse NUFSHT's
existing tested scalar machinery and swap ONLY (a) the S step (scalar->spin via FastTransforms
spin plans) and (b) the doubling rule (from the spin-DFS derivation). Re-test vs finer-grid ref.

## Update 2 — blocker precisely isolated (Spin.jl is WIP/unvalidated)
Built `src/Spin.jl` (SpinNUSHTplan, make_spin_plan, nusht_type2_spin!/type1/solve, spin_sph_mode,
spin DFS doubling). Validation status:
- Grid CONFIRMED: FastSphericalHarmonics.sph_points = θ=π(i-0.5)/Nθ, φ=2π(j-1)/Nφ exactly.
- SCALAR NUFSHT reproduces synthesis at that grid to 5.8e-13 (machinery + grid correct).
- nusht_solve_spin! is self-consistent (synth∘solve round-trips ~1e-10) but that only proves
  internal consistency, not correctness.
- SPIN type2 vs exact reference: rel_err ≈ 1.5–1.8. Adjoint test fails (~1.6).
- Doubling sweep {sgn=±1}×{conj} ALL give identical ≈1.77; m-space (−1)^(m+s) also fails;
  and crucially **s=0 through the SPIN plans also fails (≈1.38)** while scalar `sph2fourier`
  path passes. ⇒ the failure is NOT the DFS doubling rule.

### Root cause
`spinsph2fourier`/`spinsph_synthesis` (FastTransforms) use a DIFFERENT bivariate-Fourier
representation / grid mapping than the scalar `sph2fourier`/`sph_synthesis`. NUFSHT's scalar
DFS+FFT+FINUFFT machinery (phase_θ half-pixel correction, even-DFS doubling, FINUFFT mode
layout) was tuned to the SCALAR Fourier output and does not transfer to the spin output.

### Clean paths forward (need the authoritative convention, not guessing)
1. **Direct bivariate-Fourier → FINUFFT:** `spinsph2fourier` (P) already yields the bivariate
   Fourier coefficients; map P*C straight into the FINUFFT 2D mode array (skip grid+DFS+FFT).
   Requires Slevinsky's spin coefficient/Fourier layout (cos/sin vs complex, (m',m) ordering).
2. **Spin DFS:** keep the grid pipeline but use the spin synthesis grid/Fourier convention
   that FastTransforms actually produces (read `ft_execute_spinsph_synthesis` C source or
   Slevinsky 2019 + spin generalization).

Action: obtain the FastTransforms/Slevinsky spin bivariate-Fourier convention properly
(re-run focused research or read FastTransforms C source), then finish + validate against the
exact finer-grid reference and the adjoint test.
