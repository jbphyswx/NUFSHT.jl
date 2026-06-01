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

using NUFSHT
using FastSphericalHarmonics: FastSphericalHarmonics, sph_mode, sph_points, sph_evaluate
using CairoMakie
using LinearAlgebra: norm
using Random: Random, seed!
using Statistics: mean, std

seed!(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthesis at scattered points
# ─────────────────────────────────────────────────────────────────────────────
println("Running example 1: synthesis...")

lmax = 20
M    = 2000
θ_sc = rand(M) .* π
φ_sc = rand(M) .* 2π
plan_sc = make_plan(θ_sc, φ_sc, lmax; tol=1e-8)

C1 = zeros(lmax+1, 2lmax+1)
C1[sph_mode(2,  0)] =  1.0
C1[sph_mode(3,  1)] =  0.6
C1[sph_mode(5, -2)] = -0.4
C1[sph_mode(4,  3)] =  0.3

f_sc = zeros(M)
nusht_type2!(f_sc, C1, plan_sc)

# ─────────────────────────────────────────────────────────────────────────────
# 2. CC-grid round-trip: error per degree ℓ
# ─────────────────────────────────────────────────────────────────────────────
println("Running example 2: round-trip accuracy...")

pts = sph_points(lmax + 1)
θ_cc = vec([θ for θ in pts[1], φ in pts[2]])
φ_cc = vec([φ for θ in pts[1], φ in pts[2]])
plan_cc = make_plan(θ_cc, φ_cc, lmax; tol=1e-10)

C_rand = randn(lmax+1, 2lmax+1)
f_cc   = zeros(length(θ_cc))
nusht_type2!(f_cc, C_rand, plan_cc)

C_rec = similar(plan_cc.C)
nusht_type1!(C_rec, f_cc, plan_cc)

# Per-degree RMS error
ell_rms_err = Float64[]
ell_rms_ref = Float64[]
for ℓ in 0:lmax
    idx = [sph_mode(ℓ, m) for m in -ℓ:ℓ]
    push!(ell_rms_err, sqrt(mean(abs2.(C_rec[idx] .- C_rand[idx]))))
    push!(ell_rms_ref, sqrt(mean(abs2.(C_rand[idx]))))
end
ell_rms_rel = ell_rms_err ./ (ell_rms_ref .+ 1e-30)

rms_total = sqrt(mean(abs2.(C_rec .- C_rand))) / sqrt(mean(abs2.(C_rand)))
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

plan_cg = make_plan(θ_jit, φ_jit, lmax_cg; tol=1e-10)

C_cg_true = zeros(lmax_cg+1, 2lmax_cg+1)
for ℓ in 1:4, m in -ℓ:ℓ
    C_cg_true[sph_mode(ℓ, m)] = randn()
end
f_obs = zeros(M_cg);  nusht_type2!(f_obs, C_cg_true, plan_cg)

C_cg_sol = similar(plan_cg.C)
C_cg_sol, cg_iters, cg_res = nusht_solve!(C_cg_sol, f_obs, plan_cg; rtol=1e-6, maxiter=1000)

f_rec = zeros(M_cg);  nusht_type2!(f_rec, C_cg_sol, plan_cg)
field_err = norm(f_rec .- f_obs) / norm(f_obs)
println("  CG: $(cg_iters) iters, field_err=$(round(field_err, sigdigits=3))")
@assert field_err < 1e-3

# ─────────────────────────────────────────────────────────────────────────────
# 4. Spectral filtering
# ─────────────────────────────────────────────────────────────────────────────
println("Running example 4: spectral filtering...")

lmax_f = 30
pts_f  = sph_points(lmax_f + 1)
θ_f    = vec([θ for θ in pts_f[1], φ in pts_f[2]])
φ_f    = vec([φ for θ in pts_f[1], φ in pts_f[2]])
plan_f = make_plan(θ_f, φ_f, lmax_f; tol=1e-8)

C_f    = randn(lmax_f+1, 2lmax_f+1)
f_full = zeros(length(θ_f));  nusht_type2!(f_full, C_f, plan_f)

filt_gauss  = gaussian_from_scale(2000e3)
filt_tophat = TopHatTransfer(10)
f_gauss     = similar(f_full);  nusht_filter!(f_gauss,  f_full, filt_gauss,  plan_f)
f_tophat    = similar(f_full);  nusht_filter!(f_tophat, f_full, filt_tophat, plan_f)

# Spectral power per degree for plotting
function per_degree_power(C, lmax)
    [sqrt(mean(abs2.(C[[sph_mode(ℓ, m) for m in -ℓ:ℓ]]))) for ℓ in 0:lmax]
end

# Get filtered coefficients for power spectra
C_full   = copy(plan_f.C);   nusht_type1!(C_full,   f_full,   plan_f)
C_gauss  = copy(plan_f.C);   nusht_type1!(C_gauss,  f_gauss,  plan_f)
C_tophat = copy(plan_f.C);   nusht_type1!(C_tophat, f_tophat, plan_f)

pow_full   = per_degree_power(C_full,   lmax_f)
pow_gauss  = per_degree_power(C_gauss,  lmax_f)
pow_tophat = per_degree_power(C_tophat, lmax_f)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Mask + renorm
# ─────────────────────────────────────────────────────────────────────────────
println("Running example 5: mask + renorm...")

mask     = Float64.(rand(length(θ_f)) .> 0.5)
f_const  = ones(length(θ_f))
f_masked = f_const .* mask

f_filt_masked = similar(f_const)
nusht_filter!(f_filt_masked, f_masked, filt_gauss, plan_f)
nusht_filter_renorm!(f_filt_masked, mask, filt_gauss, plan_f)

ocean_idx = findall(mask .> 0.5)
max_err   = maximum(abs.(f_filt_masked[ocean_idx] .- 1.0))
println("  Mask renorm max error: $(round(max_err, sigdigits=3))")
@assert max_err < 0.05

# ─────────────────────────────────────────────────────────────────────────────
# Build the figure
# ─────────────────────────────────────────────────────────────────────────────
println("Building figure...")

fig = Figure(size=(1600, 1100), fontsize=13)

## Panel 1: Synthesis — scatter plot coloured by field value
ax1 = Axis(fig[1, 1];
    title  = "1. Synthesis at 2000 scattered points\n(Y₂⁰ + Y₃¹ + Y₅⁻² + Y₄³)",
    xlabel = "Longitude φ (rad)",
    ylabel = "Colatitude θ (rad)",
    yreversed = true,
)
sc = scatter!(ax1, φ_sc, θ_sc; color=f_sc, colormap=:RdBu, markersize=5,
              colorrange=(-maximum(abs.(f_sc)), maximum(abs.(f_sc))))
Colorbar(fig[1, 2], sc; label="Field value")

## Panel 2: Round-trip per-degree error
ax2 = Axis(fig[1, 3];
    title   = "2. CC-grid round-trip error per ℓ\n(lmax=$(lmax))",
    xlabel  = "Degree ℓ",
    ylabel  = "Relative RMS error",
    yscale  = log10,
    yticks  = LogTicks(LinearTicks(5)),
)
barplot!(ax2, 0:lmax, max.(ell_rms_rel, 1e-16); color=:steelblue)
hlines!(ax2, [1e-10]; color=:red, linestyle=:dash, label="machine eps guide")
text!(ax2, lmax÷2, 1e-9; text="Total: $(round(rms_total, sigdigits=2))", fontsize=11)

## Panel 3: CG — observed vs recovered field (first 200 points for clarity)
n_show = min(200, M_cg)
ax3 = Axis(fig[2, 1];
    title  = "3. CG inversion (nusht_solve!)\nlmax=$(lmax_cg), M=$(M_cg), $(cg_iters) iters",
    xlabel = "Point index",
    ylabel = "Field value",
)
lines!(ax3, 1:n_show, f_obs[1:n_show];  label="Observed f",   color=:black, linewidth=1.5)
lines!(ax3, 1:n_show, f_rec[1:n_show];  label="Recovered Ac", color=:crimson, linestyle=:dash, linewidth=1.5)
axislegend(ax3; position=:rt, framevisible=false)
text!(ax3, n_show÷2, minimum(f_obs[1:n_show]);
      text="field err = $(round(field_err, sigdigits=2))", fontsize=10)

## Panel 4: Power spectrum — full vs filtered
ax4 = Axis(fig[2, 2:3];
    title   = "4. Spectral filtering — power per degree\n(lmax=$(lmax_f))",
    xlabel  = "Degree ℓ",
    ylabel  = "RMS amplitude",
    yscale  = log10,
    yticks  = LogTicks(LinearTicks(5)),
)
lines!(ax4, 0:lmax_f, pow_full;   label="Full field",           color=:black,      linewidth=2)
lines!(ax4, 0:lmax_f, pow_gauss;  label="Gaussian (2000 km)",   color=:royalblue,  linewidth=2)
lines!(ax4, 0:lmax_f, pow_tophat; label="Top-hat (L≤10)",       color=:darkorange, linewidth=2, linestyle=:dash)
vlines!(ax4, [10]; color=:darkorange, linestyle=:dot, linewidth=1)
axislegend(ax4; position=:lb, framevisible=false)

## Panel 5: Mask renorm — recovered field at ocean points
ax5 = Axis(fig[3, 1:2];
    title  = "5. Mask + renorm: constant-field recovery\n(50% random ocean mask, Gaussian 2000 km)",
    xlabel = "Point index (ocean only, first 300)",
    ylabel = "Recovered value",
)
n_ocean = min(300, length(ocean_idx))
hlines!(ax5, [1.0]; color=:black, linestyle=:dash, label="True value = 1")
scatter!(ax5, 1:n_ocean, f_filt_masked[ocean_idx[1:n_ocean]];
         color=:seagreen, markersize=4, label="Renorm output")
ylims!(ax5, 0.85, 1.15)
axislegend(ax5; position=:rb, framevisible=false)
text!(ax5, n_ocean÷2, 0.87;
      text="Max error = $(round(max_err, sigdigits=2))", fontsize=10)

## Panel 6: Mask renorm error histogram
ax6 = Axis(fig[3, 3];
    title  = "5b. Error distribution at ocean points",
    xlabel = "Recovered − 1.0",
    ylabel = "Count",
)
hist!(ax6, f_filt_masked[ocean_idx] .- 1.0; bins=40, color=(:seagreen, 0.7))
vlines!(ax6, [0.0]; color=:black, linestyle=:dash)

# Final layout tweaks
Label(fig[0, :], "NUFSHT.jl — Non-Uniform Fast Spherical Harmonic Transforms";
      fontsize=16, font=:bold)

outpath = joinpath(@__DIR__, "basic_usage_output.png")
save(outpath, fig; px_per_unit=2)
println("Figure saved to: $outpath")

println()
println("All examples completed successfully.")
println("  Round-trip relative RMS : $(round(rms_total, sigdigits=3))")
println("  CG field error           : $(round(field_err, sigdigits=3))")
println("  Mask renorm max error    : $(round(max_err, sigdigits=3))")
