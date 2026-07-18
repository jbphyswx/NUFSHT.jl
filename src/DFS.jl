"""
    DFS.jl — Double Fourier Sphere (DFS) method utilities.

The DFS method (Merilees 1973, Townsend & Olver 2015, Reinecke & Seljebotn 2013) periodizes a
function on the sphere [0,π] × [0,2π) to a doubly-periodic function on the torus [0,2π) × [0,2π) by
reflecting across the south pole, enabling the standard 2D FFT / nuFFT on spherical data.

`dfs_double!`/`dfs_fold!` operate on `(Nθ, Nφ)` or batched `(Nθ, Nφ, B)` arrays; the doubled buffer
may be complex (so an in-place c2c FFT can act on it) while the CC map is real. `dfs_fold!` is the
exact matrix transpose of `dfs_double!` (the adjoint of the real→complex embedding is `real`), which
makes the composite `nusht_solve!` adjoint exact.

References:
- Merilees, P.E. (1973): The pseudospectral approximation applied to the shallow water equations on
  a sphere. Atmosphere, 11(1), 13–20.
- Townsend, A. & Olver, S. (2015): The automatic solution of PDEs using a global spectral method.
  J. Comput. Phys., 299, 106–123.
- Reinecke, M. & Seljebotn, D.S. (2013): Libsharp – spherical harmonic transforms revisited.
  A&A, 554, A112.
- Belkner et al. (2024): cunuSHT – GPU Accelerated Spherical Harmonic Transforms on Arbitrary
  Pixelizations. arXiv:2406.14542.
"""

export dfs_double, dfs_fold, dfs_double!, dfs_fold!

"""
    dfs_double(F) -> F̃

Given `F` of size `(Nθ, Nφ)` (a scalar field on the open CC grid, no poles), produce `F̃` of size
`(2Nθ, Nφ)` on the doubly-periodic torus:

    F̃[1:Nθ, :]  = F
    F̃[Nθ+i, j]  = F[Nθ+1-i, mod1(j + Nφ÷2, Nφ)]   for i = 1..Nθ

The φ+π column shift `mod1(j + Nφ÷2, Nφ)` is a proper cyclic permutation for any `Nφ` (including odd,
which `Nφ = 2lmax+1` always is); the naive conditional shift is not a bijection for odd `Nφ`.
"""
function dfs_double(F::AbstractMatrix)
    Nθ, Nφ = size(F)
    F̃ = similar(F, 2Nθ, Nφ)   # every entry is written by dfs_double! — no pre-zeroing
    dfs_double!(F̃, F)
    return F̃
end

"""
    dfs_double!(F̃, F) -> F̃

In-place doubling. `F` is `(Nθ, Nφ[, B])` (typically real); `F̃` is `(2Nθ, Nφ[, B])` (may be complex
— real values are embedded). All of `F̃` is written (no pre-zeroing needed).
"""
function dfs_double!(F̃::AbstractArray, F::AbstractArray)
    Nθ, Nφ = size(F, 1), size(F, 2)
    B = size(F, 3)
    @assert size(F̃, 1) == 2Nθ && size(F̃, 2) == Nφ && size(F̃, 3) == B
    half = Nφ ÷ 2
    @inbounds for b in 1:B
        for j in 1:Nφ, i in 1:Nθ
            F̃[i, j, b] = F[i, j, b]
        end
        for j in 1:Nφ
            j_shifted = mod1(j + half, Nφ)
            for i in 1:Nθ
                F̃[Nθ + i, j, b] = F[Nθ + 1 - i, j_shifted, b]
            end
        end
    end
    return F̃
end

"""
    dfs_fold(F̃) -> F

Adjoint (matrix transpose) of `dfs_double!`. Given `F̃` of size `(2Nθ, Nφ)`, fold back to `F` of size
`(Nθ, Nφ)` (real):

    F[i,j] = real( F̃[i,j] + F̃[2Nθ+1-i, mod1(j - Nφ÷2, Nφ)] )   for i = 1..Nθ

The **inverse** shift `-Nφ÷2 (mod Nφ)` is the exact adjoint of the `+Nφ÷2` shift in `dfs_double!`
(they differ for odd `Nφ`), and `real` is the adjoint of the real→complex embedding.
"""
function dfs_fold(F̃::AbstractMatrix)
    Nθ_double, Nφ = size(F̃)
    @assert iseven(Nθ_double)
    Nθ = Nθ_double ÷ 2
    F = similar(F̃, real(eltype(F̃)), Nθ, Nφ)   # every entry is written by dfs_fold! — no pre-zeroing
    dfs_fold!(F, F̃)
    return F
end

"""
    dfs_fold!(F, F̃) -> F

In-place fold (adjoint of `dfs_double!`). `F` is `(Nθ, Nφ[, B])` real; `F̃` is `(2Nθ, Nφ[, B])`
(real or complex — the real part of the accumulated sum is stored).
"""
function dfs_fold!(F::AbstractArray, F̃::AbstractArray)
    Nθ, Nφ = size(F, 1), size(F, 2)
    B = size(F, 3)
    @assert size(F̃, 1) == 2Nθ && size(F̃, 2) == Nφ && size(F̃, 3) == B
    half = Nφ ÷ 2
    @inbounds for b in 1:B
        for j in 1:Nφ
            j_shifted = mod1(j - half, Nφ)
            for i in 1:Nθ
                i_mirror = 2Nθ + 1 - i
                F[i, j, b] = real(F̃[i, j, b] + F̃[i_mirror, j_shifted, b])
            end
        end
    end
    return F
end

"""
    dfs_grid_coords(Nθ, Nφ) -> (θs, φs)

Colatitude/longitude of the CC equiangular grid used by FastSphericalHarmonics for an
`(Nθ × Nφ) = (lmax+1 × 2lmax+1)` map: `θ_i = π(i-0.5)/Nθ ∈ (0,π)`, `φ_j = 2π(j-1)/Nφ ∈ [0,2π)`.
"""
function dfs_grid_coords(Nθ, Nφ)
    θs = [π * (i - 0.5) / Nθ for i in 1:Nθ]
    φs = [2π * (j - 1) / Nφ for j in 1:Nφ]
    return θs, φs
end

"""
    dfs_doubled_grid_coords(Nθ, Nφ) -> (θ̃s, φs)

Colatitudes of the doubled (torus) grid `θ̃ ∈ [0, 2π)` and the longitudes `φ ∈ [0, 2π)`.
"""
function dfs_doubled_grid_coords(Nθ, Nφ)
    θ̃s = [π * (i - 0.5) / Nθ for i in 1:2Nθ]
    φs  = [2π * (j - 1) / Nφ  for j in 1:Nφ]
    return θ̃s, φs
end
