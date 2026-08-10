using FINUFFT: FINUFFT                     # triggers NUFSHTFINUFFTExt
using NonuniformFFTs: NonuniformFFTs        # triggers NUFSHTNonuniformFFTsExt
using SpectralBackends: SpectralBackends

# The `N` step is pluggable: direct summation (no dependencies, always available) plus one extension
# per library. Direct summation is the independent reference — it shares no code with either library —
# so both fast backends are checked against it rather than against each other, and the adjoint is
# checked too, since `nusht_solve!`'s convergence depends on the pair being an exact transpose.
Test.@testset "NUFFT backends agree with direct summation" begin
    Random.seed!(77)
    for (lmax, M) in ((8, 200), (12, 300))
        N = lmax + 1; Nφ = 2lmax + 1
        θ = clamp.(π .* rand(M), 1e-9, π - 1e-9)
        φ = 2π .* rand(M)
        C = randn(N, Nφ)

        vals = Dict{String,Vector{Float64}}()
        coef = Dict{String,Matrix{Float64}}()
        for (name, backend) in ("directsum" => SpectralBackends.DirectSumSpectralBackend(),
                                "finufft" => NUFSHT.FINUFFTBackend(),
                                "nonuniformffts" => NUFSHT.NonuniformFFTsBackend())
            plan = NUFSHT.make_plan(θ, φ, lmax; tol = 1e-12, nufft = backend)
            f = zeros(M); NUFSHT.nusht_type2!(f, C, plan); vals[name] = f
            Cb = zeros(N, Nφ); NUFSHT.nusht_type1!(Cb, f, plan); coef[name] = Cb
            NUFSHT.close!(plan)
        end

        relerr(a, b) = sqrt(sum(abs2, a .- b) / sum(abs2, b))
        for name in ("finufft", "nonuniformffts")
            Test.@test relerr(vals[name], vals["directsum"]) < 1e-10
            Test.@test relerr(coef[name], coef["directsum"]) < 1e-10
        end
    end
    @info "NUFFT backends: FINUFFT and NonuniformFFTs match direct summation (synthesis and adjoint)"
end

# `AutoSpectralBackend` must choose a loaded fast backend, and a backend whose extension is missing
# must be refused rather than quietly swapped for a slower one.
Test.@testset "NUFFT backend selection" begin
    Test.@test NUFSHT._resolve_nufft(SpectralBackends.AutoSpectralBackend()) isa NUFSHT.FINUFFTBackend
    Test.@test NUFSHT._resolve_nufft(NUFSHT.NonuniformFFTsBackend()) isa NUFSHT.NonuniformFFTsBackend
    Test.@test NUFSHT._resolve_nufft(SpectralBackends.DirectSumSpectralBackend()) isa
               SpectralBackends.DirectSumSpectralBackend
end
