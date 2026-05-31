"""
    DFS.jl — Double Fourier Sphere (DFS) method utilities.

The DFS method (Merilees 1973, Townsend & Olver 2015, Reinecke & Seljebotn 2013)
periodizes a function on the sphere [0,π] × [0,2π) to a doubly-periodic function
on the torus [0,2π) × [0,2π) by reflecting across the south pole.

This enables application of the standard 2D FFT / nuFFT to spherical data, which
is the core trick behind the DFS-based nuSHT algorithm.

The key steps are:
1. `dfs_double`: Given an (Nθ × Nφ) map on colatitude θ ∈ [0,π] on the open CC
   grid (no poles), produce a (2Nθ × Nφ) map on θ̃ ∈ [0,2π) by reflecting
   all Nθ rows across the south pole with a φ+π shift.
2. `dfs_fold`: The adjoint — given a (2Nθ × Nφ) map, accumulate mirror rows
   back to (Nθ × Nφ).

References:
- Merilees, P.E. (1973): The pseudospectral approximation applied to the shallow
  water equations on a sphere. Atmosphere, 11(1), 13–20.
- Townsend, A. & Olver, S. (2015): The automatic solution of partial differential
  equations using a global spectral method. J. Comput. Phys., 299, 106–123.
- Reinecke, M. & Seljebotn, D.S. (2013): Libsharp – spherical harmonic transforms
  revisited. A&A, 554, A112.
- Belkner et al. (2024): cunuSHT – GPU Accelerated Spherical Harmonic Transforms
  on Arbitrary Pixelizations. arXiv:2406.14542.
"""

export dfs_double, dfs_fold, dfs_double!, dfs_fold!

"""
    dfs_double(F) -> F̃

Given an array F of size (Nθ, Nφ) representing a scalar field on the sphere
sampled on the open CC grid (no poles, θ_i = π/Nθ * (i-0.5)), produce F̃ of
size (2Nθ, Nφ) on the doubly-periodic torus covering [0, 2π) in θ.

All Nθ rows are reflected (the open CC grid has no pole duplicates):
  F̃[1:Nθ, :]        = F[1:Nθ, :]            (original northern half)
  F̃[Nθ+i, j]        = F[Nθ+1-i, j+Nφ/2]   for i=1..Nθ (reflected, φ-shifted)

The doubled grid has θ cell-centres at 2π/(2Nθ) * (i-0.5) for i=1..2Nθ, which
matches the FFTW grid after the half-pixel phase correction applied in
`ifft2_from_coeffs`.
"""
function dfs_double(F::AbstractMatrix)
    Nθ, Nφ = size(F)
    F̃ = zeros(eltype(F), 2Nθ, Nφ)
    dfs_double!(F̃, F)
    return F̃
end

function dfs_double!(F̃::AbstractMatrix, F::AbstractMatrix)
    Nθ, Nφ = size(F)
    @assert size(F̃) == (2Nθ, Nφ)

    F̃[1:Nθ, :] .= F

    half = Nφ ÷ 2
    for i in 1:Nθ
        i_mirror = Nθ + i
        for j in 1:Nφ
            j_shifted = j <= half ? j + half : j - half
            F̃[i_mirror, j] = F[Nθ + 1 - i, j_shifted]
        end
    end
    return F̃
end

"""
    dfs_fold(F̃) -> F

Adjoint of `dfs_double`. Given F̃ of size (2Nθ, Nφ) on the doubled torus,
fold back to F of size (Nθ, Nφ) on [0,π].

All rows accumulate their mirror (there are no poles in the open CC grid):
  F[i,j] = F̃[i,j] + F̃[Nθ+Nθ+1-i, j_shifted]  for i=1..Nθ
"""
function dfs_fold(F̃::AbstractMatrix)
    Nθ_double, Nφ = size(F̃)
    @assert iseven(Nθ_double)
    Nθ = Nθ_double ÷ 2
    F = zeros(eltype(F̃), Nθ, Nφ)
    dfs_fold!(F, F̃)
    return F
end

function dfs_fold!(F::AbstractMatrix, F̃::AbstractMatrix)
    Nθ, Nφ = size(F)
    @assert size(F̃) == (2Nθ, Nφ)

    half = Nφ ÷ 2
    for i in 1:Nθ
        i_mirror = Nθ + (Nθ + 1 - i)
        for j in 1:Nφ
            j_shifted = j <= half ? j + half : j - half
            F[i, j] = F̃[i, j] + F̃[i_mirror, j_shifted]
        end
    end
    return F
end

"""
    dfs_grid_coords(Nθ, Nφ) -> (θs, φs)

Return the colatitude and longitude coordinates of the CC equiangular grid
used by FastSphericalHarmonics for an (Nθ × Nφ) = (lmax+1 × 2lmax+1) map.
θ ∈ (0, π), φ ∈ [0, 2π).

The CC grid has colatitudes θ_i = π*(i-0.5)/Nθ for i = 1:Nθ.
"""
function dfs_grid_coords(Nθ, Nφ)
    θs = [π * (i - 0.5) / Nθ for i in 1:Nθ]
    φs = [2π * (j - 1) / Nφ for j in 1:Nφ]
    return θs, φs
end

"""
    dfs_doubled_grid_coords(Nθ, Nφ) -> (θ̃s, φs)

Return colatitudes of the doubled (torus) grid θ̃ ∈ [0, 2π).
"""
function dfs_doubled_grid_coords(Nθ, Nφ)
    θ̃s = [π * (i - 0.5) / Nθ for i in 1:2Nθ]
    φs  = [2π * (j - 1) / Nφ  for j in 1:Nφ]
    return θ̃s, φs
end
