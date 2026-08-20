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

# A batched solve retires columns as they converge and then transforms only the live ones. Narrowing
# the FFT/NUFFT needs plans of several widths to share one type, which is a per-backend property —
# FINUFFT keeps the transform count as a runtime field, NonuniformFFTs as a type parameter. Every
# backend must reach the same answer regardless, so this runs the retiring path on each of them; the
# solve tests alone cannot, since they resolve to whichever backend is preferred.
Test.@testset "batched solve retires correctly on every backend" begin
    Random.seed!(20)
    lmax, M, B = 8, 150, 4                       # barely overdetermined ⇒ columns retire apart
    N, Nf = lmax + 1, 2lmax + 1
    θ = acos.(2 .* rand(M) .- 1); φ = 2π .* rand(M)
    F = randn(M, B)

    for backend in (NUFSHT.FINUFFTBackend(), NUFSHT.NonuniformFFTsBackend())
        pB = NUFSHT.make_plan(Float64, θ, φ, lmax; ntrans = B, nufft = backend)
        p1 = NUFSHT.make_plan(Float64, θ, φ, lmax; nufft = backend)

        # The trait must match what the backend's plans actually do, not what we assume.
        narrow = pB.pool_recipe.narrowable
        Test.@test narrow == (backend isa NUFSHT.FINUFFTBackend)

        CB = zeros(N, Nf, B)
        _, itB, relB = NUFSHT.nusht_solve!(CB, F, pB; rtol = 1e-8, maxiter = 500)
        Test.@test itB < 500                     # did not diverge or stall
        Test.@test relB < 1e-8

        for b in 1:B
            c = zeros(N, Nf)
            NUFSHT.nusht_solve!(c, F[:, b], p1; rtol = 1e-8, maxiter = 500)
            Test.@test maximum(abs, CB[:, :, b] .- c) / maximum(abs, c) < 1e-4
        end

        # Only a narrowable backend may ever cache a reduced-width plan set.
        narrow || Test.@test isempty(pB.size_pool)
        @info "$(nameof(typeof(backend))): narrowable=$narrow iters=$itB widths=$([e.k for e in pB.size_pool])"
        NUFSHT.close!(pB); NUFSHT.close!(p1)
    end
end
