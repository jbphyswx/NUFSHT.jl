# Comprehensive allocation audit: EVERY mutating / hot-path method that is meant to run allocation-free is
# asserted `@allocated == 0` here — no cherry-picking. Covers both batch shapes (B=1 and B=2), the scalar
# and spin transforms, their DFS/S-engine/filter steps, all CG reductions, and both CG solvers.
#
# Measurement methodology (important — this is what makes the numbers trustworthy): each method is measured
# INSIDE a small barrier function whose parameters carry the concrete buffer types. A bare top-level
# `@allocated f(x)` over non-const globals reports phantom allocations from boxing the global reads (e.g.
# `maximum(ws.rel)` measures ~13 kB at global scope but 0 in typed code) — so the barrier is not a
# convenience, it is the only way to measure the method's true internal allocation. Each barrier also runs
# the call twice before measuring, to clear one-time per-plan initialization (FastTransforms adjoint-plan /
# FFTW planner setup allocates once on the first real call, then never again).
#
# Zero allocation is a single-threaded guarantee: under a multithreaded Julia, FINUFFT's
# thread-safe FFTW-planner lock adds small external per-call allocations outside NUFSHT's control, so these
# assertions run only at `-t1` (matching `test_adjoint.jl`).

# ── barrier measurers ────────────────────────────────────────────────────────────────────────────────────
# Each measures its method's TRUE steady-state allocation. The method is run three times: a plain warm call
# (clears one-time per-plan FastTransforms/FFTW init), then an `@allocated` warm call (compiles the measured
# specialization — the `@allocated`-wrapped form is a distinct compilation from the plain call), then the
# returned `@allocated` measurement. Measuring inside a typed-parameter function also excludes the phantom
# allocations a bare top-level `@allocated` reports from boxing non-const global reads.
_a_type2(f, C, p)        = (NUFSHT.nusht_type2!(f, C, p); @allocated NUFSHT.nusht_type2!(f, C, p); @allocated NUFSHT.nusht_type2!(f, C, p))
_a_type1(C, f, p)        = (NUFSHT.nusht_type1!(C, f, p); @allocated NUFSHT.nusht_type1!(C, f, p); @allocated NUFSHT.nusht_type1!(C, f, p))
_a_adj(C, f, p)          = (NUFSHT._nusht_true_adjoint!(C, f, p); @allocated NUFSHT._nusht_true_adjoint!(C, f, p); @allocated NUFSHT._nusht_true_adjoint!(C, f, p))
_a_synth(p)              = (NUFSHT._dfn_synthesis!(p); @allocated NUFSHT._dfn_synthesis!(p); @allocated NUFSHT._dfn_synthesis!(p))
_a_analy(p)              = (NUFSHT._dfn_analysis!(p); @allocated NUFSHT._dfn_analysis!(p); @allocated NUFSHT._dfn_analysis!(p))
_a_transfer(C, ft, l)    = (NUFSHT.apply_transfer!(C, ft, l); @allocated NUFSHT.apply_transfer!(C, ft, l); @allocated NUFSHT.apply_transfer!(C, ft, l))
_a_filter(o, i, ft, p, w) = (NUFSHT.nusht_filter!(o, i, ft, p; ws = w); @allocated NUFSHT.nusht_filter!(o, i, ft, p; ws = w); @allocated NUFSHT.nusht_filter!(o, i, ft, p; ws = w))
_a_renorm(o, m, ft, p, s, w) = (NUFSHT.nusht_filter_renorm!(o, m, ft, p; mask_filt = s, ws = w); @allocated NUFSHT.nusht_filter_renorm!(o, m, ft, p; mask_filt = s, ws = w); @allocated NUFSHT.nusht_filter_renorm!(o, m, ft, p; mask_filt = s, ws = w))
_a_asmM(Z, G, l)         = (NUFSHT._assemble_modes!(Z, G, l); @allocated NUFSHT._assemble_modes!(Z, G, l); @allocated NUFSHT._assemble_modes!(Z, G, l))
_a_asmMa(G, Z, l)        = (NUFSHT._assemble_modes_adjoint!(G, Z, l); @allocated NUFSHT._assemble_modes_adjoint!(G, Z, l); @allocated NUFSHT._assemble_modes_adjoint!(G, Z, l))
_a_caxpy(y, al, x, s)    = (NUFSHT._col_axpy!(y, al, x, s); @allocated NUFSHT._col_axpy!(y, al, x, s); @allocated NUFSHT._col_axpy!(y, al, x, s))
_a_cpbp(p, r, be)        = (NUFSHT._col_pbp!(p, r, be); @allocated NUFSHT._col_pbp!(p, r, be); @allocated NUFSHT._col_pbp!(p, r, be))
_a_cscale(y, s)          = (NUFSHT._col_scale!(y, s); @allocated NUFSHT._col_scale!(y, s); @allocated NUFSHT._col_scale!(y, s))
_a_solve(C, f, p, ws)    = (NUFSHT.nusht_solve!(C, f, p; ws = ws, maxiter = 60, rtol = 1e-8); @allocated NUFSHT.nusht_solve!(C, f, p; ws = ws, maxiter = 60, rtol = 1e-8); @allocated NUFSHT.nusht_solve!(C, f, p; ws = ws, maxiter = 60, rtol = 1e-8))
_a_type2s(f, sf, p)      = (NUFSHT.nusht_type2_spin!(f, sf, p); @allocated NUFSHT.nusht_type2_spin!(f, sf, p); @allocated NUFSHT.nusht_type2_spin!(f, sf, p))
_a_type1s(sf, f, p)      = (NUFSHT.nusht_type1_spin!(sf, f, p); @allocated NUFSHT.nusht_type1_spin!(sf, f, p); @allocated NUFSHT.nusht_type1_spin!(sf, f, p))
_a_asmG(G, sf, p)        = (NUFSHT._assemble_G!(G, sf, p); @allocated NUFSHT._assemble_G!(G, sf, p); @allocated NUFSHT._assemble_G!(G, sf, p))
_a_asmGa(sf, G, p)       = (NUFSHT._assemble_G_adjoint!(sf, G, p); @allocated NUFSHT._assemble_G_adjoint!(sf, G, p); @allocated NUFSHT._assemble_G_adjoint!(sf, G, p))
_a_chdot(d, a, b)        = (NUFSHT._col_hdot!(d, a, b); @allocated NUFSHT._col_hdot!(d, a, b); @allocated NUFSHT._col_hdot!(d, a, b))
_a_solves(sf, f, p, ws)  = (NUFSHT.nusht_solve_spin!(sf, f, p; ws = ws, maxiter = 60, rtol = 1e-8); @allocated NUFSHT.nusht_solve_spin!(sf, f, p; ws = ws, maxiter = 60, rtol = 1e-8); @allocated NUFSHT.nusht_solve_spin!(sf, f, p; ws = ws, maxiter = 60, rtol = 1e-8))

Test.@testset "allocation: full hot-path surface is allocation-free" begin
    single = Threads.nthreads() == 1
    if !single
        @info "zero-alloc is a single-threaded guarantee; skipping @allocated==0 asserts at -t$(Threads.nthreads())"
    end
    Random.seed!(9)
    lmax = 8
    Nθ, Nφ = lmax + 1, 2lmax + 1
    M = 4 * (lmax + 1)^2
    θ, φ = fib_points(M)          # cond(A) ≈ 1.04, so the solve converges in a few iterations
    filt = NUFSHT.gaussian_from_scale(2000e3)

    for B in (1, 2)
        Test.@testset "scalar transform + CG (B=$B)" begin
            plan = NUFSHT.make_plan(θ, φ, lmax; tol = 1e-10, ntrans = B)
            C = randn(Nθ, Nφ, B); f = zeros(M, B); Cout = zeros(Nθ, Nφ, B); out = zeros(M, B)
            NUFSHT.nusht_type2!(f, C, plan)
            mask = abs.(randn(M, B)) .+ 0.5; scratch = similar(out)
            C_true = zeros(Nθ, Nφ, B)
            for b in 1:B, ℓ in 1:min(4, lmax), m in -ℓ:ℓ
                C_true[FastSphericalHarmonics.sph_mode(ℓ, m), b] = randn()
            end
            ftrue = zeros(M, B); NUFSHT.nusht_type2!(ftrue, C_true, plan)
            Csol = similar(plan.C); ws = NUFSHT.LSMRWorkspace(plan)
            # Warm the whole pipeline on this plan: the first real use of the adjoint / solve path builds
            # FastTransforms' lazy adjoint plan (a one-time ~few-hundred-byte per-plan setup cost, not a
            # per-call allocation). After warmup, steady-state is zero.
            for _ in 1:3
                NUFSHT.nusht_type1!(Cout, f, plan)
                NUFSHT._nusht_true_adjoint!(Cout, f, plan)
                NUFSHT.nusht_filter!(out, f, filt, plan; ws = ws)
                NUFSHT.nusht_solve!(Csol, ftrue, plan; ws = ws, maxiter = 60, rtol = 1e-8)
            end
            if single
                Test.@test _a_type2(f, C, plan)           == 0
                Test.@test _a_type1(Cout, f, plan)        == 0
                Test.@test _a_adj(Cout, f, plan)          == 0
                Test.@test _a_synth(plan)                 == 0
                Test.@test _a_analy(plan)                 == 0
                Test.@test _a_transfer(plan.C, filt, lmax) == 0
                Test.@test _a_filter(out, f, filt, plan, ws) == 0
                Test.@test _a_renorm(out, mask, filt, plan, scratch, ws) == 0
                Test.@test _a_asmM(plan.Fhat, plan.F, lmax) == 0
                Test.@test _a_asmMa(plan.F, plan.Fhat, lmax) == 0
                Test.@test _a_chdot(ws.nrm, ws.v, ws.v)   == 0
                Test.@test _a_caxpy(ws.x, ws.cf, ws.hbar, 1.0) == 0
                Test.@test _a_cpbp(ws.h, ws.v, ws.cf)     == 0
                Test.@test _a_cscale(ws.u, ws.cf)         == 0
                Test.@test _a_solve(Csol, ftrue, plan, ws) == 0
            end
            NUFSHT.close!(plan)
        end

        Test.@testset "spin transform + CG (B=$B)" begin
            s = 1
            plan = NUFSHT.make_spin_plan(θ, φ, lmax, s; tol = 1e-10, ntrans = B)
            sf = zeros(ComplexF64, Nθ, Nφ, B); fs = zeros(ComplexF64, M, B); sfo = zeros(ComplexF64, Nθ, Nφ, B)
            for b in 1:B, ℓ in abs(s):min(4, lmax), m in -ℓ:ℓ
                sf[NUFSHT.spin_coeff_index(ℓ, m, lmax), b] = randn(ComplexF64)
            end
            NUFSHT.nusht_type2_spin!(fs, sf, plan)
            Ghat = randn(ComplexF64, Nφ, Nφ, B)
            sws = NUFSHT.LSMRWorkspace(plan)
            for _ in 1:3   # warm the plan's full pipeline (one-time per-plan init; see scalar block)
                NUFSHT.nusht_type1_spin!(sfo, fs, plan)
                NUFSHT.nusht_solve_spin!(sfo, fs, plan; ws = sws, maxiter = 60, rtol = 1e-8)
            end
            if single
                Test.@test _a_type2s(fs, sf, plan)          == 0
                Test.@test _a_type1s(sfo, fs, plan)         == 0
                Test.@test _a_asmG(plan.G, sf, plan)        == 0
                Test.@test _a_asmGa(sf, Ghat, plan)         == 0
                Test.@test _a_chdot(sws.nrm, sws.v, sws.v)  == 0
                Test.@test _a_caxpy(sws.x, sws.cf, sws.hbar, 1.0) == 0
                Test.@test _a_cpbp(sws.h, sws.v, sws.cf)    == 0
                Test.@test _a_cscale(sws.u, sws.cf)         == 0
                Test.@test _a_solves(sfo, fs, plan, sws)    == 0
            end
            NUFSHT.close!(plan)
        end
    end
end
