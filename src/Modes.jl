"""
    Modes.jl — bivariate Fourier coefficients → the NUFFT's complex exponential mode array.

`FastTransforms.plan_sph2fourier` already converts spherical harmonic coefficients into the
Double-Fourier-Sphere bivariate Fourier series (its README: "`sph2fourier` converts the representation
into a bivariate Fourier series"). This turns that real cos/sin basis into the complex exponentials a
NUFFT evaluates, so synthesis is `P·C` → assemble → one NUFFT with no grid in between.

For order `m` in the `sph_mode` column packing, the series `P·C` represents is

    f(θ,φ) = Σ_m Φ_m(φ) · N_m · Σ_i G[i, col(m)] · Θ_{i,m}(θ)

with `Φ_m = cos(|m|φ)` on the `+m` column and `sin(|m|φ)` on the `−m` column, `N_0 = 1/√(2π)` and
`N_{m≠0} = 1/√π`, and `Θ = cos((i−1)θ)` for even `|m|` but `sin(iθ)` for odd `|m|`. That parity is not
a storage convention: the DFS extension of `Y_ℓm` in θ is `sin^{|m|}θ · Q(cosθ)`, even in θ for even
`m` and odd for odd `m`, which is the same fact the glide reflection encodes.

Odd `|m|` therefore reaches θ-frequency `i`, so row `lmax+1` of such a column carries frequency
`lmax+1`. A degree-`lmax` expansion has nothing there, but the coefficient array is a square,
invertible representation whose supernumerary slots hold degrees `lmax < l ≤ lmax+|m|`, and those do
reach it — so the mode array runs `-(lmax+1) … lmax+1` on θ and the transform stays faithful on the
whole array rather than only on the band-limited subspace.
"""

# Column `j` of the (lmax+1, 2lmax+1) coefficient array → its signed order, matching `sph_mode`:
# column 1 is m = 0, then the pairs (2k, 2k+1) are m = −k, +k.
@inline _col_order(j::Integer) = iseven(j) ? -(j >> 1) : (j >> 1)

# The θ-frequency row `i` of an order-`m` column carries, and whether that column is a sine series.
@inline _theta_freq(i::Integer, am::Integer) = iseven(am) ? i - 1 : i

"""
    _assemble_modes!(Z, G, lmax)

`G` (bivariate Fourier coefficients, `sph2fourier` layout, `(lmax+1, 2lmax+1, B)`) → `Z`, the
`(2lmax+1, 2lmax+1, B)` complex mode array with `Z[kθ+lmax+1, kφ+lmax+1]` the coefficient of
`exp(i(kθ·θ + kφ·φ))`. Exact: no grid, no doubling, no FFT.
"""
function _assemble_modes!(Z, G, lmax::Integer)
    CT = eltype(Z)
    RT = real(CT)
    offθ = size(Z, 1) ÷ 2 + 1          # rows run -(size÷2) … ; both axes are centered
    offφ = size(Z, 2) ÷ 2 + 1
    n0 = RT(1) / sqrt(RT(2) * RT(π))
    nm = RT(1) / sqrt(RT(π))
    h = RT(0.5)
    fill!(Z, zero(CT))
    @inbounds for b in axes(G, 3), j in axes(G, 2)
        m = _col_order(j)
        am = abs(m)
        am > lmax && continue
        nrm = m == 0 ? n0 : nm
        # φ factor: 1 for m = 0, else cos/sin split over e^{±i|m|φ}.
        pp, pm = m == 0 ? (CT(1), CT(0)) :
                 m > 0  ? (CT(h), CT(h)) : (CT(0, -h), CT(0, h))
        c1 = am + offφ
        c2 = -am + offφ
        sine = isodd(am)
        for i in axes(G, 1)
            g = G[i, j, b]
            iszero(g) && continue
            kθ = _theta_freq(i, am)
            tp, tm = sine ? (CT(0, -h), CT(0, h)) :
                     kθ == 0 ? (CT(1), CT(0)) : (CT(h), CT(h))
            c = nrm * g
            r1 = kθ + offθ
            r2 = -kθ + offθ
            Z[r1, c1, b] += c * tp * pp
            iszero(pm) || (Z[r1, c2, b] += c * tp * pm)
            if !iszero(tm)
                Z[r2, c1, b] += c * tm * pp
                iszero(pm) || (Z[r2, c2, b] += c * tm * pm)
            end
        end
    end
    return Z
end

"""
    _assemble_modes_adjoint!(G, Z, lmax)

Exact Euclidean adjoint of [`_assemble_modes!`](@ref): each `G[i,j]` gathers `conj(w)·Z` over the same
(up to four) mode entries the forward map scattered into. A real `G` takes the real part, which is the
adjoint of the real→complex embedding; a complex one does not.
"""
function _assemble_modes_adjoint!(G, Z, lmax::Integer)
    CT = eltype(Z)
    RT = real(CT)
    offθ = size(Z, 1) ÷ 2 + 1
    offφ = size(Z, 2) ÷ 2 + 1
    n0 = RT(1) / sqrt(RT(2) * RT(π))
    nm = RT(1) / sqrt(RT(π))
    h = RT(0.5)
    fill!(G, zero(eltype(G)))
    @inbounds for b in axes(G, 3), j in axes(G, 2)
        m = _col_order(j)
        am = abs(m)
        am > lmax && continue
        nrm = m == 0 ? n0 : nm
        pp, pm = m == 0 ? (CT(1), CT(0)) :
                 m > 0  ? (CT(h), CT(h)) : (CT(0, -h), CT(0, h))
        c1 = am + offφ
        c2 = -am + offφ
        sine = isodd(am)
        for i in axes(G, 1)
            kθ = _theta_freq(i, am)
            tp, tm = sine ? (CT(0, -h), CT(0, h)) :
                     kθ == 0 ? (CT(1), CT(0)) : (CT(h), CT(h))
            r1 = kθ + offθ
            r2 = -kθ + offθ
            acc = conj(tp * pp) * Z[r1, c1, b]
            iszero(pm) || (acc += conj(tp * pm) * Z[r1, c2, b])
            if !iszero(tm)
                acc += conj(tm * pp) * Z[r2, c1, b]
                iszero(pm) || (acc += conj(tm * pm) * Z[r2, c2, b])
            end
            G[i, j, b] = _modes_out(eltype(G), nrm * acc)
        end
    end
    return G
end

# A real coefficient array takes the real part (adjoint of the real→complex embedding); a complex one
# keeps the value.
@inline _modes_out(::Type{R}, z) where {R<:Real} = R(real(z))
@inline _modes_out(::Type{C}, z) where {C<:Complex} = C(z)
