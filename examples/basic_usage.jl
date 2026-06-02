"""
basic_usage.jl — Runnable examples for NUFSHT.jl with CairoMakie figures

Demonstrates all major features, producing a multi-panel figure saved to
  examples/basic_usage_output.png

Features shown:
1. Synthesis (type2): evaluate Y_2^0 + Y_3^1 at 2000 scattered points
2. CC-grid round-trip accuracy: coefficient recovery error vs ℓ
3. CG inversion (nusht_solve!): observed vs recovered field at scattered points
4. Spectral filtering: full / Gaussian / top-hat on CC grid
5. Ocean-mask filter + renorm: constant-field recovery with 50% random mask

Run from the NUFSHT.jl directory:
    julia --project=examples examples/basic_usage.jl
"""

using NUFSHT: NUFSHT
using FastSphericalHarmonics: FastSphericalHarmonics
using CairoMakie: CairoMakie
using LinearAlgebra: LinearAlgebra
using Random: Random
using Statistics: Statistics

Random.seed!(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthesis at scattered points
# ─────────────────────────────────────────────────────────────────────────────
println("Running example 1: synthesis...")

lmax = 20
M    = 2000
θ_sc = rand(M) .* π
φ_sc = rand(M) .* 2π
plan_sc = NUFSHT.make_plan(θ_sc, φ_sc, lmax; tol=1e-8)

C1 = zeros(lmax+1, 2lmax+1)
C1[FastSphericalHarmonics.sph_mode(2,  0)] =  1.0
C1[FastSphericalHarmonics.sph_mode(3,  1)] =  0.6
C1[FastSphericalHarmonics.sph_mode(5, -2)] = -0.4
C1[FastSphericalHarmonics.sph_mode(4,  3)] =  0.3

f_sc = zeros(M)
NUFSHT.nusht_type2!(f_sc, C1, plan_sc)

# ─────────────────────────────────────────────────────────────────────────────
# 2. CC-grid round-trip: error per degree ℓ
# ─────────────────────────────────────────────────────────────────────────────
println("Running example 2: round-trip accuracy...")

pts = FastSphericalHarmonics.sph_points(lmax + 1)
θ_cc = vec([θ for θ in pts[1], φ in pts[2]])
φ_cc = vec([φ for θ in pts[1], φ in pts[2]])
plan_cc = NUFSHT.make_plan(θ_cc, φ_cc, lmax; tol=1e-10)

C_rand = randn(lmax+1, 2lmax+1)
f_cc   = zeros(length(θ_cc))
NUFSHT.nusht_type2!(f_cc, C_rand, plan_cc)

C_rec = similar(plan_cc.C)
NUFSHT.nusht_type1!(C_rec, f_cc, plan_cc)

# Per-degree RMS error
ell_rms_err = Float64[]
ell_rms_ref = Float64[]
for ℓ in 0:lmax
    idx = [FastSphericalHarmonics.sph_mode(ℓ, m) for m in -ℓ:ℓ]
    push!(ell_rms_err, sqrt(Statistics.mean(abs2.(C_rec[idx] .- C_rand[idx]))))
    push!(ell_rms_ref, sqrt(Statistics.mean(abs2.(C_rand[idx]))))
end
ell_rms_rel = ell_rms_err ./ (ell_rms_ref .+ 1e-30)

rms_total = sqrt(Statistics.mean(abs2.(C_rec .- C_rand))) / sqrt(Statistics.mean(abs2.(C_rand)))
println("  Total round-trip relative RMS: $(round(rms_total, sigdigits=3))")
@assert rms_total < 1e-8

# ─────────────────────────────────────────────────────────────────────────────
# 3. CG inversion at scattered points
# ─────────────────────────────────────────────────────────────────────────────
println("Running example 3: CG inversion...")

lmax_cg = 10
K_cg    = (lmax_cg+1) * (2lmax_cg+1)
M_cg    = 4 * K_cg

φ_jit = collect((2π / M_cg) .* (0:M_cg-1))
θ_jit = acos.(clamp.(2 .* ((0:M_cg-1) .+ 0.5) ./ M_cg .- 1, -1.0, 1.0))
θ_jit .+= (rand(M_cg) .- 0.5) .* (0.4π / sqrt(M_cg))
φ_jit .+= (rand(M_cg) .- 0.5) .* (0.4 * 2π / sqrt(M_cg))
θ_jit  = clamp.(θ_jit, 1e-10, π - 1e-10)
φ_jit  = mod.(φ_jit, 2π)

plan_cg = NUFSHT.make_plan(θ_jit, φ_jit, lmax_cg; tol=1e-10)

C_cg_true = zeros(lmax_cg+1, 2lmax_cg+1)
for ℓ in 1:4, m in -ℓ:ℓ
    C_cg_true[FastSphericalHarmonics.sph_mode(ℓ, m)] = randn()
end
f_obs = zeros(M_cg);  NUFSHT.nusht_type2!(f_obs, C_cg_true, plan_cg)

C_cg_sol = similar(plan_cg.C)
C_cg_sol, cg_iters, cg_res = NUFSHT.nusht_solve!(C_cg_sol, f_obs, plan_cg; rtol=1e-6, maxiter=1000)

f_rec = zeros(M_cg);  NUFSHT.nusht_type2!(f_rec, C_cg_sol, plan_cg)
field_err = LinearAlgebra.norm(f_rec .- f_obs) / LinearAlgebra.norm(f_obs)
println("  CG: $(cg_iters) iters, field_err=$(round(field_err, sigdigits=3))")
@assert field_err < 1e-3

# ─────────────────────────────────────────────────────────────────────────────
# 4. Spectral filtering
# ─────────────────────────────────────────────────────────────────────────────
println("Running example 4: spectral filtering...")

lmax_f = 30
pts_f  = FastSphericalHarmonics.sph_points(lmax_f + 1)
θ_f    = vec([θ for θ in pts_f[1], φ in pts_f[2]])
φ_f    = vec([φ for θ in pts_f[1], φ in pts_f[2]])
plan_f = NUFSHT.make_plan(θ_f, φ_f, lmax_f; tol=1e-8)

C_f    = randn(lmax_f+1, 2lmax_f+1)
f_full = zeros(length(θ_f));  NUFSHT.nusht_type2!(f_full, C_f, plan_f)

filt_gauss  = NUFSHT.gaussian_from_scale(2000e3)
filt_tophat = NUFSHT.TopHatTransfer(10)
f_gauss     = similar(f_full);  NUFSHT.nusht_filter!(f_gauss,  f_full, filt_gauss,  plan_f)
f_tophat    = similar(f_full);  NUFSHT.nusht_filter!(f_tophat, f_full, filt_tophat, plan_f)

# Spectral power per degree for plotting
function per_degree_power(C, lmax)
    [sqrt(Statistics.mean(abs2.(C[[FastSphericalHarmonics.sph_mode(ℓ, m) for m in -ℓ:ℓ]]))) for ℓ in 0:lmax]
end

# Get filtered coefficients for power spectra
C_full   = copy(plan_f.C);   NUFSHT.nusht_type1!(C_full,   f_full,   plan_f)
C_gauss  = copy(plan_f.C);   NUFSHT.nusht_type1!(C_gauss,  f_gauss,  plan_f)
C_tophat = copy(plan_f.C);   NUFSHT.nusht_type1!(C_tophat, f_tophat, plan_f)

pow_full   = per_degree_power(C_full,   lmax_f)
pow_gauss  = per_degree_power(C_gauss,  lmax_f)
pow_tophat = per_degree_power(C_tophat, lmax_f)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Mask + renorm
# ─────────────────────────────────────────────────────────────────────────────
println("Running example 5: mask + renorm...")

# Use a structured field and spatial mask to show WHY renorm matters.
# The filter is 800 km Gaussian; the mask is a latitude band (colatitude 30°–150°).
# Naive filtering (zero out land, then filter) gives biased values near coastlines
# because the kernel overlaps the zeroed-out land. Renormalization corrects this.
filt_renorm = NUFSHT.gaussian_from_scale(800e3)
C_mask_test = zeros(lmax_f+1, 2lmax_f+1)
C_mask_test[FastSphericalHarmonics.sph_mode(3, 1)] = 1.0
C_mask_test[FastSphericalHarmonics.sph_mode(5, 2)] = 0.7
C_mask_test[FastSphericalHarmonics.sph_mode(8, -3)] = 0.5
C_mask_test[FastSphericalHarmonics.sph_mode(2, 0)] = 0.4
f_mask_true = zeros(length(θ_f))
NUFSHT.nusht_type2!(f_mask_true, C_mask_test, plan_f)

mask = Float64.((θ_f .> deg2rad(30)) .& (θ_f .< deg2rad(150)))

# Truth: filter without mask
f_mask_truth_filt = similar(f_mask_true)
NUFSHT.nusht_filter!(f_mask_truth_filt, f_mask_true, filt_renorm, plan_f)

# Naive: zero land, filter
f_masked = f_mask_true .* mask
f_naive = similar(f_mask_true)
NUFSHT.nusht_filter!(f_naive, f_masked, filt_renorm, plan_f)

# Renormalized: correct for mask bias
f_filt_masked = copy(f_naive)
NUFSHT.nusht_filter_renorm!(f_filt_masked, mask, filt_renorm, plan_f)

ocean_idx = findall(mask .> 0.5)
naive_err = maximum(abs.((f_naive .- f_mask_truth_filt)[ocean_idx]))
renorm_err = maximum(abs.((f_filt_masked .- f_mask_truth_filt)[ocean_idx]))
println("  Naive max error at ocean pts:  $(round(naive_err, sigdigits=3))")
println("  Renorm max error at ocean pts: $(round(renorm_err, sigdigits=3))")
@assert renorm_err < naive_err

# ─────────────────────────────────────────────────────────────────────────────
# Build the figure
# ─────────────────────────────────────────────────────────────────────────────
println("Building figure...")

fig = CairoMakie.Figure(size=(1600, 1100), fontsize=13)

## Panel 1: Synthesis — scatter plot coloured by field value
ax1 = CairoMakie.Axis(fig[1, 1];
    title  = "1. Synthesis at 2000 scattered points\n(Y₂⁰ + Y₃¹ + Y₅⁻² + Y₄³)",
    xlabel = "Longitude φ (rad)",
    ylabel = "Colatitude θ (rad)",
    yreversed = true,
)
sc = CairoMakie.scatter!(ax1, φ_sc, θ_sc; color=f_sc, colormap=:RdBu, markersize=5,
              colorrange=(-maximum(abs.(f_sc)), maximum(abs.(f_sc))))
CairoMakie.Colorbar(fig[1, 2], sc; label="Field value")

## Panel 2: Round-trip per-degree error
ax2 = CairoMakie.Axis(fig[1, 3];
    title   = "2. CC-grid round-trip error per ℓ\n(lmax=$(lmax))",
    xlabel  = "Degree ℓ",
    ylabel  = "Relative RMS error",
    yscale  = log10,
    yticks  = CairoMakie.LogTicks(CairoMakie.LinearTicks(5)),
)
CairoMakie.barplot!(ax2, 0:lmax, max.(ell_rms_rel, 1e-16); color=:steelblue)
CairoMakie.hlines!(ax2, [1e-10]; color=:red, linestyle=:dash, label="machine eps guide")
CairoMakie.text!(ax2, lmax÷2, 1e-9; text="Total: $(round(rms_total, sigdigits=2))", fontsize=11)

## Panel 3: CG — observed vs recovered field (first 200 points for clarity)
n_show = min(200, M_cg)
ax3 = CairoMakie.Axis(fig[2, 1];
    title  = "3. CG inversion (nusht_solve!)\nlmax=$(lmax_cg), M=$(M_cg), $(cg_iters) iters",
    xlabel = "Point index",
    ylabel = "Field value",
)
CairoMakie.lines!(ax3, 1:n_show, f_obs[1:n_show];  label="Observed f",   color=:black, linewidth=1.5)
CairoMakie.lines!(ax3, 1:n_show, f_rec[1:n_show];  label="Recovered Ac", color=:crimson, linestyle=:dash, linewidth=1.5)
CairoMakie.axislegend(ax3; position=:rt, framevisible=false)
CairoMakie.text!(ax3, n_show÷2, minimum(f_obs[1:n_show]);
      text="field err = $(round(field_err, sigdigits=2))", fontsize=10)

## Panel 4: Power spectrum — full vs filtered
ax4 = CairoMakie.Axis(fig[2, 2:3];
    title   = "4. Spectral filtering — power per degree\n(lmax=$(lmax_f))",
    xlabel  = "Degree ℓ",
    ylabel  = "RMS amplitude",
    yscale  = log10,
    yticks  = CairoMakie.LogTicks(CairoMakie.LinearTicks(5)),
)
CairoMakie.lines!(ax4, 0:lmax_f, pow_full;   label="Full field",           color=:black,      linewidth=2)
CairoMakie.lines!(ax4, 0:lmax_f, pow_gauss;  label="Gaussian (2000 km)",   color=:royalblue,  linewidth=2)
CairoMakie.lines!(ax4, 0:lmax_f, pow_tophat; label="Top-hat (L≤10)",       color=:darkorange, linewidth=2, linestyle=:dash)
CairoMakie.vlines!(ax4, [10]; color=:darkorange, linestyle=:dot, linewidth=1)
CairoMakie.axislegend(ax4; position=:lb, framevisible=false)

## Panel 5: Mask renorm — naive vs renorm error at ocean points
ax5 = CairoMakie.Axis(fig[3, 1:2];
    title  = "5. Mask renorm: error vs truth at ocean points\n(lat-band mask, 800 km Gaussian)",
    xlabel = "Point index (ocean only, first 300)",
    ylabel = "Error (filtered − truth)",
)
n_ocean = min(300, length(ocean_idx))
naive_errors = (f_naive .- f_mask_truth_filt)[ocean_idx[1:n_ocean]]
renorm_errors = (f_filt_masked .- f_mask_truth_filt)[ocean_idx[1:n_ocean]]
CairoMakie.lines!(ax5, 1:n_ocean, naive_errors; color=:crimson, linewidth=1.2, label="Naive (biased)")
CairoMakie.lines!(ax5, 1:n_ocean, renorm_errors; color=:seagreen, linewidth=1.5, label="Renormalized")
CairoMakie.hlines!(ax5, [0.0]; color=:black, linestyle=:dash)
CairoMakie.axislegend(ax5; position=:rt, framevisible=false)
CairoMakie.text!(ax5, n_ocean÷2, maximum(abs.(naive_errors))*0.8;
      text="Naive max err = $(round(naive_err, sigdigits=2))\nRenorm max err = $(round(renorm_err, sigdigits=2))", fontsize=10)

## Panel 6: Mask renorm error histogram comparison
ax6 = CairoMakie.Axis(fig[3, 3];
    title  = "5b. Error distribution at ocean points",
    xlabel = "Error (filtered − truth)",
    ylabel = "Count",
)
CairoMakie.hist!(ax6, (f_naive .- f_mask_truth_filt)[ocean_idx]; bins=40, color=(:crimson, 0.5), label="Naive")
CairoMakie.hist!(ax6, (f_filt_masked .- f_mask_truth_filt)[ocean_idx]; bins=40, color=(:seagreen, 0.5), label="Renorm")
CairoMakie.vlines!(ax6, [0.0]; color=:black, linestyle=:dash)
CairoMakie.axislegend(ax6; position=:rt, framevisible=false)

# Final layout tweaks
CairoMakie.Label(fig[0, :], "NUFSHT.jl — Non-Uniform Fast Spherical Harmonic Transforms";
      fontsize=16, font=:bold)

outpath = joinpath(@__DIR__, "basic_usage_output.png")
CairoMakie.save(outpath, fig; px_per_unit=2)
println("Figure saved to: $outpath")

println()
println("All examples completed successfully.")
println("  Round-trip relative RMS : $(round(rms_total, sigdigits=3))")
println("  CG field error           : $(round(field_err, sigdigits=3))")
println("  Naive mask error         : $(round(naive_err, sigdigits=3))")
println("  Renorm mask error        : $(round(renorm_err, sigdigits=3))")
