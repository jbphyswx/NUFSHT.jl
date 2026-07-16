"""
    NUFSHTCairoMakieExt

CairoMakie plotting recipe for NUFSHT fields. Loaded by `using CairoMakie`; provides `plot_field`.
"""
module NUFSHTCairoMakieExt

using NUFSHT: NUFSHT
using CairoMakie: CairoMakie

function NUFSHT.plot_field(θ, φ, f; colormap = :RdBu, markersize = 8, title = "",
                           colorbarlabel = "Field value")
    vals = real.(f)
    m = maximum(abs, vals)
    m = m == 0 ? one(m) : m
    fig = CairoMakie.Figure()
    ax = CairoMakie.Axis(fig[1, 1]; title = title, xlabel = "Longitude φ (rad)",
                         ylabel = "Colatitude θ (rad)", yreversed = true)
    sc = CairoMakie.scatter!(ax, φ, θ; color = vals, colormap = colormap,
                             markersize = markersize, colorrange = (-m, m))
    CairoMakie.Colorbar(fig[1, 2], sc; label = colorbarlabel, width = 12)
    return fig
end

end # module NUFSHTCairoMakieExt
