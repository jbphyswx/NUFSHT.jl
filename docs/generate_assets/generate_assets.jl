"""
Generate static figure assets for NUFSHT.jl docs.

Run from this directory:
    julia --project=. generate_assets.jl

Outputs PNG files to ../assets/ which are checked into the repo
and referenced from README.md and docs/ markdown.
"""

using NUFSHT: NUFSHT
using FastSphericalHarmonics: FastSphericalHarmonics
using CairoMakie: CairoMakie
using LinearAlgebra: LinearAlgebra
using Random: Random
using Statistics: Statistics

Random.seed!(42)

const ASSETS_DIR = joinpath(@__DIR__, "..", "assets")
mkpath(ASSETS_DIR)

# ─── Figure 1: Synthesis + Round-Trip Accuracy ────────────────────────────

function figure_synthesis_and_accuracy()
    lmax = 20

    # Panel 1a: Synthesis at scattered points
    M = 2000
    θ_sc = rand(M) .* π
    φ_sc = rand(M) .* 2π
    plan_sc = NUFSHT.make_plan(θ_sc, φ_sc, lmax; tol=1e-8)

    C1 = zeros(lmax+1, 2lmax+1)
    C1[FastSphericalHarmonics.sph_mode(2, 0)] = 1.0
    C1[FastSphericalHarmonics.sph_mode(3, 1)] = 0.6
    C1[FastSphericalHarmonics.sph_mode(5, -2)] = -0.4
    C1[FastSphericalHarmonics.sph_mode(4, 3)] = 0.3

    f_sc = zeros(M)
    NUFSHT.nusht_type2!(f_sc, C1, plan_sc)

    # Panel 1b: CC-grid round-trip error per degree
    pts = FastSphericalHarmonics.sph_points(lmax + 1)
    θ_cc = vec([θ for θ in pts[1], φ in pts[2]])
    φ_cc = vec([φ for θ in pts[1], φ in pts[2]])
    plan_cc = NUFSHT.make_plan(θ_cc, φ_cc, lmax; tol=1e-10)

    C_rand = randn(lmax+1, 2lmax+1)
    f_cc = zeros(length(θ_cc))
    NUFSHT.nusht_type2!(f_cc, C_rand, plan_cc)

    C_rec = similar(plan_cc.C)
    NUFSHT.nusht_type1!(C_rec, f_cc, plan_cc)

    ell_rms_rel = Float64[]
    for ℓ in 0:lmax
        idx = [FastSphericalHarmonics.sph_mode(ℓ, m) for m in -ℓ:ℓ]
        err = sqrt(Statistics.mean(abs2.(C_rec[idx] .- C_rand[idx])))
        ref = sqrt(Statistics.mean(abs2.(C_rand[idx])))
        push!(ell_rms_rel, err / (ref + 1e-30))
    end
    rms_total = sqrt(Statistics.mean(abs2.(C_rec .- C_rand))) / sqrt(Statistics.mean(abs2.(C_rand)))

    # Build figure
    fig = CairoMakie.Figure(; size=(1200, 450), fontsize=13)
    CairoMakie.Label(fig[0, 1:4], "NUFSHT.jl — Synthesis and Round-Trip Accuracy";
                      fontsize=16, font=:bold)

    # Panel 1: scattered synthesis
    ax1 = CairoMakie.Axis(fig[1, 1]; title="Synthesis: Y₂⁰ + Y₃¹ + Y₅⁻² + Y₄³\nat 2000 scattered points",
                           xlabel="Longitude φ (rad)", ylabel="Colatitude θ (rad)", yreversed=true)
    clim = maximum(abs.(f_sc))
    sc = CairoMakie.scatter!(ax1, φ_sc, θ_sc; color=f_sc, colormap=:RdBu,
                              markersize=4, colorrange=(-clim, clim))
    CairoMakie.Colorbar(fig[1, 2], sc; label="Field value", width=12)

    # Panel 2: per-degree error
    ax2 = CairoMakie.Axis(fig[1, 3]; title="CC-Grid Round-Trip Error\n(lmax=$lmax)",
                           xlabel="Degree ℓ", ylabel="Relative RMS error",
                           yscale=CairoMakie.log10)
    CairoMakie.barplot!(ax2, 0:lmax, max.(ell_rms_rel, 1e-16); color=:steelblue)
    CairoMakie.hlines!(ax2, [1e-10]; color=:red, linestyle=:dash)
    CairoMakie.text!(ax2, lmax÷2, 3e-9; text="Total RMS: $(round(rms_total, sigdigits=2))", fontsize=11)

    outpath = joinpath(ASSETS_DIR, "synthesis_and_accuracy.png")
    CairoMakie.save(outpath, fig; px_per_unit=2)
    println("Saved: $outpath")
end

# ─── Figure 2: CG Inversion at Scattered Points ──────────────────────────

function figure_cg_inversion()
    lmax_cg = 10
    K_cg = (lmax_cg+1) * (2lmax_cg+1)
    M_cg = 4 * K_cg

    # Jittered uniform points
    φ_jit = collect((2π / M_cg) .* (0:M_cg-1))
    θ_jit = acos.(clamp.(2 .* ((0:M_cg-1) .+ 0.5) ./ M_cg .- 1, -1.0, 1.0))
    θ_jit .+= (rand(M_cg) .- 0.5) .* (0.4π / sqrt(M_cg))
    φ_jit .+= (rand(M_cg) .- 0.5) .* (0.4 * 2π / sqrt(M_cg))
    θ_jit = clamp.(θ_jit, 1e-10, π - 1e-10)
    φ_jit = mod.(φ_jit, 2π)

    plan_cg = NUFSHT.make_plan(θ_jit, φ_jit, lmax_cg; tol=1e-10)

    C_cg_true = zeros(lmax_cg+1, 2lmax_cg+1)
    for ℓ in 1:4, m in -ℓ:ℓ
        C_cg_true[FastSphericalHarmonics.sph_mode(ℓ, m)] = randn()
    end
    f_obs = zeros(M_cg)
    NUFSHT.nusht_type2!(f_obs, C_cg_true, plan_cg)

    C_cg_sol = similar(plan_cg.C)
    C_cg_sol, cg_iters, cg_res = NUFSHT.nusht_solve!(C_cg_sol, f_obs, plan_cg; rtol=1e-6, maxiter=1000)

    f_rec = zeros(M_cg)
    NUFSHT.nusht_type2!(f_rec, C_cg_sol, plan_cg)
    field_err = LinearAlgebra.norm(f_rec .- f_obs) / LinearAlgebra.norm(f_obs)

    fig = CairoMakie.Figure(; size=(1000, 400), fontsize=13)
    CairoMakie.Label(fig[0, 1:2], "NUFSHT.jl — CG Inversion at Arbitrary Scattered Points";
                      fontsize=16, font=:bold)

    n_show = min(200, M_cg)
    ax = CairoMakie.Axis(fig[1, 1]; title="nusht_solve!: lmax=$lmax_cg, M=$M_cg, $(cg_iters) CG iters",
                          xlabel="Point index", ylabel="Field value")
    CairoMakie.lines!(ax, 1:n_show, f_obs[1:n_show]; label="Observed f", color=:black, linewidth=1.5)
    CairoMakie.lines!(ax, 1:n_show, f_rec[1:n_show]; label="Recovered Ac", color=:crimson,
                       linestyle=:dash, linewidth=1.5)
    CairoMakie.axislegend(ax; position=:rt, framevisible=false)
    CairoMakie.text!(ax, n_show÷2, minimum(f_obs[1:n_show]) * 0.9;
                      text="field error = $(round(field_err, sigdigits=2))", fontsize=11)

    # Residual panel
    ax2 = CairoMakie.Axis(fig[1, 2]; title="Pointwise error", xlabel="Point index", ylabel="f_rec − f_obs")
    CairoMakie.lines!(ax2, 1:n_show, (f_rec .- f_obs)[1:n_show]; color=:steelblue, linewidth=1)
    CairoMakie.hlines!(ax2, [0.0]; color=:black, linestyle=:dash)

    outpath = joinpath(ASSETS_DIR, "cg_inversion.png")
    CairoMakie.save(outpath, fig; px_per_unit=2)
    println("Saved: $outpath")
end

# ─── Figure 3: Spectral Filtering ────────────────────────────────────────

function figure_spectral_filtering()
    lmax_f = 30
    pts_f = FastSphericalHarmonics.sph_points(lmax_f + 1)
    θ_f = vec([θ for θ in pts_f[1], φ in pts_f[2]])
    φ_f = vec([φ for θ in pts_f[1], φ in pts_f[2]])
    plan_f = NUFSHT.make_plan(θ_f, φ_f, lmax_f; tol=1e-8)

    C_f = randn(lmax_f+1, 2lmax_f+1)
    f_full = zeros(length(θ_f))
    NUFSHT.nusht_type2!(f_full, C_f, plan_f)

    filt_gauss = NUFSHT.gaussian_from_scale(2000e3)
    filt_tophat = NUFSHT.TopHatTransfer(10)
    f_gauss = similar(f_full)
    NUFSHT.nusht_filter!(f_gauss, f_full, filt_gauss, plan_f)
    f_tophat = similar(f_full)
    NUFSHT.nusht_filter!(f_tophat, f_full, filt_tophat, plan_f)

    # Power per degree
    function per_degree_power(C, lmax)
        [sqrt(Statistics.mean(abs2.(C[[FastSphericalHarmonics.sph_mode(ℓ, m) for m in -ℓ:ℓ]]))) for ℓ in 0:lmax]
    end

    C_full = copy(plan_f.C);   NUFSHT.nusht_type1!(C_full, f_full, plan_f)
    C_gauss = copy(plan_f.C);  NUFSHT.nusht_type1!(C_gauss, f_gauss, plan_f)
    C_tophat = copy(plan_f.C); NUFSHT.nusht_type1!(C_tophat, f_tophat, plan_f)

    pow_full = per_degree_power(C_full, lmax_f)
    pow_gauss = per_degree_power(C_gauss, lmax_f)
    pow_tophat = per_degree_power(C_tophat, lmax_f)

    fig = CairoMakie.Figure(; size=(900, 450), fontsize=13)
    CairoMakie.Label(fig[0, 1:2], "NUFSHT.jl — Spectral Filtering";
                      fontsize=16, font=:bold)

    ax = CairoMakie.Axis(fig[1, 1]; title="Power Per Degree ℓ (lmax=$lmax_f)",
                          xlabel="Degree ℓ", ylabel="RMS amplitude",
                          yscale=CairoMakie.log10)
    CairoMakie.lines!(ax, 0:lmax_f, pow_full; label="Full field", color=:black, linewidth=2)
    CairoMakie.lines!(ax, 0:lmax_f, pow_gauss; label="Gaussian (2000 km)", color=:royalblue, linewidth=2)
    CairoMakie.lines!(ax, 0:lmax_f, pow_tophat; label="Top-hat (L≤10)", color=:darkorange,
                       linewidth=2, linestyle=:dash)
    CairoMakie.vlines!(ax, [10]; color=:darkorange, linestyle=:dot, linewidth=1)
    CairoMakie.axislegend(ax; position=:lb, framevisible=false)

    # Transfer functions
    ax2 = CairoMakie.Axis(fig[1, 2]; title="Transfer Functions H(ℓ)",
                           xlabel="Degree ℓ", ylabel="H(ℓ)")
    ells = 0:lmax_f
    h_gauss = [NUFSHT.kernel_transfer(filt_gauss, ℓ) for ℓ in ells]
    h_tophat = [NUFSHT.kernel_transfer(filt_tophat, ℓ) for ℓ in ells]
    CairoMakie.lines!(ax2, collect(ells), h_gauss; label="Gaussian (2000 km)", color=:royalblue, linewidth=2)
    CairoMakie.lines!(ax2, collect(ells), h_tophat; label="Top-hat (L≤10)", color=:darkorange,
                       linewidth=2, linestyle=:dash)
    CairoMakie.axislegend(ax2; position=:rt, framevisible=false)

    outpath = joinpath(ASSETS_DIR, "spectral_filtering.png")
    CairoMakie.save(outpath, fig; px_per_unit=2)
    println("Saved: $outpath")
end

# ─── Figure 4: Ocean Mask + Renormalization ───────────────────────────────
#
# The problem: when filtering a field on the sphere with a land mask (ocean=1, land=0),
# naive filtering (set land to 0, then filter) gives BIASED values near coastlines
# because the filter kernel partially overlaps land where data is artificially zero.
# The fix: divide the filtered field by the filtered mask (renormalization).
#
# This figure shows a spatially-structured field (Y₃¹ + Y₅²) on a latitude-band
# "ocean" mask, comparing:
#   (a) True filtered field (no mask, gold standard)
#   (b) Naive filtered (biased near mask edges)
#   (c) Renormalized (bias corrected — matches truth)

function figure_mask_renorm()
    lmax_f = 30
    pts_f = FastSphericalHarmonics.sph_points(lmax_f + 1)
    θ_f = vec([θ for θ in pts_f[1], φ in pts_f[2]])
    φ_f = vec([φ for θ in pts_f[1], φ in pts_f[2]])
    plan_f = NUFSHT.make_plan(θ_f, φ_f, lmax_f; tol=1e-8)

    filt_gauss = NUFSHT.gaussian_from_scale(800e3)  # 800 km Gaussian — realistic mesoscale filter

    # Structured field: multiple harmonics for rich spatial pattern
    C_true = zeros(lmax_f+1, 2lmax_f+1)
    C_true[FastSphericalHarmonics.sph_mode(3, 1)] = 1.0
    C_true[FastSphericalHarmonics.sph_mode(5, 2)] = 0.7
    C_true[FastSphericalHarmonics.sph_mode(8, -3)] = 0.5
    C_true[FastSphericalHarmonics.sph_mode(2, 0)] = 0.4
    f_true = zeros(length(θ_f))
    NUFSHT.nusht_type2!(f_true, C_true, plan_f)

    # "Ocean" mask: latitude band — ocean between 30°S and 60°N (colatitude 30° to 120°)
    # This creates a clear coastline boundary where the bias is visible
    mask = Float64.((θ_f .> deg2rad(30)) .& (θ_f .< deg2rad(150)))

    # (a) Truth: filter the unmasked field directly
    f_truth_filtered = similar(f_true)
    NUFSHT.nusht_filter!(f_truth_filtered, f_true, filt_gauss, plan_f)

    # (b) Naive: zero out land, filter (biased near boundaries)
    f_masked = f_true .* mask
    f_naive = similar(f_true)
    NUFSHT.nusht_filter!(f_naive, f_masked, filt_gauss, plan_f)

    # (c) Renormalized: divide by filtered mask to correct bias
    f_renorm = copy(f_naive)
    NUFSHT.nusht_filter_renorm!(f_renorm, mask, filt_gauss, plan_f)

    # Only look at ocean points for comparison
    ocean_idx = findall(mask .> 0.5)

    # Reshape for 2D plotting (CC grid is Nθ × Nφ)
    Nθ = length(pts_f[1])
    Nφ = length(pts_f[2])
    θ_vals = rad2deg.(pts_f[1])  # colatitude in degrees
    φ_vals = rad2deg.(pts_f[2])  # longitude in degrees

    reshape2d(v) = reshape(v, Nθ, Nφ)

    f_true_2d = reshape2d(f_true)
    f_naive_2d = reshape2d(f_naive)
    f_renorm_2d = reshape2d(f_renorm)
    f_truth_filt_2d = reshape2d(f_truth_filtered)
    mask_2d = reshape2d(mask)

    # Compute error maps (only meaningful at ocean points)
    err_naive_2d = (f_naive_2d .- f_truth_filt_2d) .* mask_2d
    err_renorm_2d = (f_renorm_2d .- f_truth_filt_2d) .* mask_2d

    fig = CairoMakie.Figure(; size=(1400, 700), fontsize=12)
    CairoMakie.Label(fig[0, 1:8],
        "NUFSHT.jl — Ocean Mask Filtering: Why Renormalization Matters";
        fontsize=16, font=:bold)

    clim = maximum(abs.(f_true))

    # Mask boundary colatitudes for overlay on all panels
    θ_lo = 30.0  # colatitude degrees
    θ_hi = 150.0

    # Helper: draw dashed mask boundary lines on an axis
    function add_mask_boundary!(ax)
        CairoMakie.hlines!(ax, [θ_lo, θ_hi]; color=(:black, 0.6), linestyle=:dash, linewidth=1.2)
    end

    # Row 1: Original field, Mask, Truth-filtered
    ax1 = CairoMakie.Axis(fig[1, 1]; title="Original field\n(Y₃¹ + Y₅² + Y₈⁻³ + Y₂⁰)",
                           xlabel="φ (°)", ylabel="θ (°)", yreversed=true)
    hm1 = CairoMakie.heatmap!(ax1, φ_vals, θ_vals, f_true_2d'; colormap=:RdBu, colorrange=(-clim, clim))
    add_mask_boundary!(ax1)
    CairoMakie.Colorbar(fig[1, 2], hm1; width=10)

    ax2 = CairoMakie.Axis(fig[1, 3]; title="Ocean mask\n(latitude band 30°–150° colatitude)",
                           xlabel="φ (°)", ylabel="θ (°)", yreversed=true)
    CairoMakie.heatmap!(ax2, φ_vals, θ_vals, mask_2d'; colormap=[:gray80, :steelblue])
    add_mask_boundary!(ax2)

    ax3 = CairoMakie.Axis(fig[1, 5]; title="Truth: filtered (no mask)\n(gold standard)",
                           xlabel="φ (°)", ylabel="θ (°)", yreversed=true)
    hm3 = CairoMakie.heatmap!(ax3, φ_vals, θ_vals, f_truth_filt_2d'; colormap=:RdBu, colorrange=(-clim, clim))
    add_mask_boundary!(ax3)
    CairoMakie.Colorbar(fig[1, 6], hm3; width=10)

    # Row 2: Naive vs Renormalized, plus error comparison
    ax4 = CairoMakie.Axis(fig[2, 1]; title="Naive: filter(f·mask)\n(BIASED near coastlines)",
                           xlabel="φ (°)", ylabel="θ (°)", yreversed=true)
    hm4 = CairoMakie.heatmap!(ax4, φ_vals, θ_vals, f_naive_2d'; colormap=:RdBu, colorrange=(-clim, clim))
    add_mask_boundary!(ax4)
    CairoMakie.Colorbar(fig[2, 2], hm4; width=10)

    ax5 = CairoMakie.Axis(fig[2, 3]; title="Renormalized: filter(f·mask)/filter(mask)\n(CORRECTED)",
                           xlabel="φ (°)", ylabel="θ (°)", yreversed=true)
    hm5 = CairoMakie.heatmap!(ax5, φ_vals, θ_vals, f_renorm_2d'; colormap=:RdBu, colorrange=(-clim, clim))
    add_mask_boundary!(ax5)
    CairoMakie.Colorbar(fig[2, 4], hm5; width=10)

    # Error maps
    err_clim = maximum(abs.(err_naive_2d)) * 0.8
    ax6 = CairoMakie.Axis(fig[2, 5]; title="Error: naive − truth\n(large bias at mask edges)",
                           xlabel="φ (°)", ylabel="θ (°)", yreversed=true)
    hm6 = CairoMakie.heatmap!(ax6, φ_vals, θ_vals, err_naive_2d'; colormap=:RdBu,
                               colorrange=(-err_clim, err_clim))
    add_mask_boundary!(ax6)
    CairoMakie.Colorbar(fig[2, 6], hm6; width=10, label="Error")

    ax7 = CairoMakie.Axis(fig[2, 7]; title="Error: renorm − truth\n(bias removed)",
                           xlabel="φ (°)", ylabel="θ (°)", yreversed=true)
    hm7 = CairoMakie.heatmap!(ax7, φ_vals, θ_vals, err_renorm_2d'; colormap=:RdBu,
                               colorrange=(-err_clim, err_clim))
    add_mask_boundary!(ax7)
    CairoMakie.Colorbar(fig[2, 8], hm7; width=10, label="Error")

    outpath = joinpath(ASSETS_DIR, "mask_renorm.png")
    CairoMakie.save(outpath, fig; px_per_unit=2)
    println("Saved: $outpath")
end

# ─── Generate all ─────────────────────────────────────────────────────────

println("Generating documentation assets for NUFSHT.jl...")
println()
figure_synthesis_and_accuracy()
figure_cg_inversion()
figure_spectral_filtering()
figure_mask_renorm()
println()
println("Done! Assets saved to: $ASSETS_DIR")
