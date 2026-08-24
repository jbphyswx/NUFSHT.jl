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

Both directions are written as **gathers** — one output element reading its sources — so neither needs
atomics and both are the same arithmetic on a host array and in a `@kernel`. The forward direction is
a gather because at most two coefficient entries reach any one mode slot: the slot's column fixes
`|m|`, its row fixes the θ-frequency and hence `i`, leaving only the `±m` pair of columns.
"""

# Column `j` of the (lmax+1, 2lmax+1) coefficient array → its signed order, matching `sph_mode`:
# column 1 is m = 0, then the pairs (2k, 2k+1) are m = −k, +k.
@inline _col_order(j::Integer) = iseven(j) ? -(j >> 1) : (j >> 1)

# The θ-frequency row `i` of an order-`m` column carries, and whether that column is a sine series.
@inline _theta_freq(i::Integer, am::Integer) = iseven(am) ? i - 1 : i

# Whether the mode array holds only `kθ ≥ 0` (`lmax+2` rows) rather than the centered `-(lmax+1) …
# lmax+1` (`2lmax+3` rows). A real field's array is exactly Hermitian, so a backend with a real-data
# transform takes the half and supplies the rest itself — see `_real_capable`.
@inline _folded(Z, lmax::Integer) = size(Z, 1) == lmax + 2

# The three normalizations every element needs, formed once per call instead of per element.
@inline _mode_norms(::Type{RT}) where {RT} =
    (RT(1) / sqrt(RT(2) * RT(π)), RT(1) / sqrt(RT(π)), RT(0.5))

"""
    _mode_entry(CT, G, lmax, r, c, b, offθ, offφ, n0, nm, h) -> CT

The coefficient of `exp(i(kθ·θ + kφ·φ))` at mode-array position `(r, c)` of batch slice `b`, gathered
from the bivariate Fourier coefficients `G`. `offθ`/`offφ` map a wavenumber to its index, so the
folded layout is just `offθ = 1` — every row it holds has `kθ ≥ 0` and the gather is unchanged.
"""
@inline function _mode_entry(::Type{CT}, G, lmax::Integer, r::Integer, c::Integer, b::Integer,
                             offθ::Integer, offφ::Integer, n0::RT, nm::RT, h::RT) where {CT, RT}
    kφ = c - offφ
    am = abs(kφ)
    am > lmax && return zero(CT)
    kθs = r - offθ
    sine = isodd(am)
    # θ factor: the cos/sin series in θ split over `e^{±ikθθ}`. An odd-order column is a sine series
    # and so never reaches `kθ = 0`; that row therefore takes only the even columns' unsplit cosine.
    tfac = kθs == 0 ? (sine ? zero(CT) : CT(1)) :
           kθs > 0  ? (sine ? CT(0, -h) : CT(h)) :
                      (sine ? CT(0,  h) : CT(h))
    iszero(tfac) && return zero(CT)
    # An even-order column tops out at `kθ = lmax`, one below the array's reach, so this also drops
    # the rows only the odd orders' supernumerary slots populate.
    i = sine ? abs(kθs) : abs(kθs) + 1
    i > size(G, 1) && return zero(CT)
    am == 0 && return (n0 * G[i, 1, b]) * tfac      # m = 0: one column, no φ split
    # The `+m` column is a cosine in φ (both halves `+1/2`), the `−m` column a sine (`∓i/2`), and the
    # sign of `kφ` says which of `e^{±i|m|φ}` this slot is.
    return (nm * tfac) * (CT(h) * G[i, 2am + 1, b] +
                          (kφ > 0 ? CT(0, -h) : CT(0, h)) * G[i, 2am, b])
end

"""
    _mode_adjoint_entry(CT, Z, lmax, i, j, b, offθ, offφ, fold, n0, nm, h) -> CT

Entry `(i, j)` of batch slice `b` of the exact Euclidean adjoint: `conj(w)·Z` summed over the (up to
four) mode slots [`_mode_entry`](@ref) draws that entry into.

On a folded array the `kθ < 0` slots are absent, and each retained `kθ > 0` row carries **twice** the
weight. The forward needs no such factor — writing the half and letting the backend's complex-to-real
transform imply the conjugate is exact — but its transpose does: the embedding `E` maps the half to the
full array by `Z[-k] = conj(Z[k])`, so `E†` gives `Ẑ[k] + conj(Ẑ[-k])`, and with real strengths
`conj(Ẑ[-k]) = Ẑ[k]`. Hence `{1, 2, 2, …}`: one at `kθ = 0`, which is its own partner, two above it.
There is no third case at the top because the θ axis has odd length `2lmax+3` and so has no
self-paired Nyquist row — an even axis would need one.
"""
@inline function _mode_adjoint_entry(::Type{CT}, Z, lmax::Integer, i::Integer, j::Integer, b::Integer,
                                     offθ::Integer, offφ::Integer, fold::Bool,
                                     n0::RT, nm::RT, h::RT) where {CT, RT}
    m = _col_order(j)
    am = abs(m)
    am > lmax && return zero(CT)
    nrm = m == 0 ? n0 : nm
    pp, pm = m == 0 ? (CT(1), CT(0)) :
             m > 0  ? (CT(h), CT(h)) : (CT(0, -h), CT(0, h))
    c1 = am + offφ
    c2 = -am + offφ
    sine = isodd(am)
    kθ = _theta_freq(i, am)
    tp, tm = sine ? (CT(0, -h), CT(0, h)) :
             kθ == 0 ? (CT(1), CT(0)) : (CT(h), CT(h))
    r1 = kθ + offθ
    w = fold && !iszero(kθ) ? RT(2) : RT(1)
    acc = conj(tp * pp) * Z[r1, c1, b]
    iszero(pm) || (acc += conj(tp * pm) * Z[r1, c2, b])
    if !iszero(tm) && !fold
        r2 = -kθ + offθ
        acc += conj(tm * pp) * Z[r2, c1, b]
        iszero(pm) || (acc += conj(tm * pm) * Z[r2, c2, b])
    end
    return (nrm * w) * acc
end

"""
    _assemble_modes!(Z, G, lmax)

`G` (bivariate Fourier coefficients, `sph2fourier` layout, `(lmax+1, 2lmax+1, B)`) → `Z`, the complex
mode array with `Z[kθ+offθ, kφ+offφ]` the coefficient of `exp(i(kθ·θ + kφ·φ))`. Exact: no grid, no
doubling, no FFT. The device method is a `@kernel` over the same [`_mode_entry`](@ref).

`Z` may be the full `(2lmax+3, 2lmax+1, B)` array or the folded `(lmax+2, …)` half — see
[`_folded`](@ref). Row `r` is written unconditionally, so no `fill!` is needed.
"""
function _assemble_modes!(Z, G, lmax::Integer)
    CT = eltype(Z)
    n0, nm, h = _mode_norms(real(CT))
    offθ = _folded(Z, lmax) ? 1 : size(Z, 1) ÷ 2 + 1
    offφ = size(Z, 2) ÷ 2 + 1
    # `r` innermost: `Z` is written straight down its columns, and a fixed `c` fixes `|m|`, so the two
    # `G` columns read are fixed too and `i` marches with `r`.
    @inbounds for b in axes(Z, 3), c in axes(Z, 2), r in axes(Z, 1)
        Z[r, c, b] = _mode_entry(CT, G, lmax, r, c, b, offθ, offφ, n0, nm, h)
    end
    return Z
end

"""
    _assemble_modes_adjoint!(G, Z, lmax)

Exact Euclidean adjoint of [`_assemble_modes!`](@ref), one entry of `G` per
[`_mode_adjoint_entry`](@ref). A real `G` takes the real part, which is the adjoint of the
real→complex embedding; a complex one does not. The device method is a `@kernel` over the same
per-element map.
"""
function _assemble_modes_adjoint!(G, Z, lmax::Integer)
    CT = eltype(Z)
    n0, nm, h = _mode_norms(real(CT))
    fold = _folded(Z, lmax)
    offθ = fold ? 1 : size(Z, 1) ÷ 2 + 1
    offφ = size(Z, 2) ÷ 2 + 1
    GE = eltype(G)
    @inbounds for b in axes(G, 3), j in axes(G, 2), i in axes(G, 1)
        G[i, j, b] = _modes_out(GE, _mode_adjoint_entry(CT, Z, lmax, i, j, b,
                                                        offθ, offφ, fold, n0, nm, h))
    end
    return G
end

# A real coefficient array takes the real part (adjoint of the real→complex embedding); a complex one
# keeps the value.
@inline _modes_out(::Type{R}, z) where {R<:Real} = R(real(z))
@inline _modes_out(::Type{C}, z) where {C<:Complex} = C(z)
