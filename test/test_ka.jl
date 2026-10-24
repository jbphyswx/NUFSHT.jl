using KernelAbstractions: KernelAbstractions   # loads NUFSHTKernelAbstractionsExt (with GPUArraysCore)
using GPUArraysCore: GPUArraysCore
using JLArrays: JLArrays                        # reference GPU-array backend for GPU-less testing
using ComputationalBackends: ComputationalBackends

# A NUFFT for device plans where no device NUFFT library exists: the plan's buffers stay on the device
# and each execution moves its input and output across in bulk, which the scalar-indexing guard allows.
# Everything else a solve runs under that guard is NUFSHT's own device path.
struct HostBounceNUFFT <: NUFSHT.SpectralBackends.AbstractNUFFTSpectralBackend end
struct HostBouncePlan{P}
    inner::P
end
NUFSHT._nufft_makeplan(::HostBounceNUFFT, nodes, type, n_modes, iflag, ntrans, tol; kwargs...) =
    HostBouncePlan(NUFSHT._nufft_makeplan(NUFSHT.SpectralBackends.DirectSumSpectralBackend(),
                                          map(Array, nodes), type, n_modes, iflag, ntrans, tol; kwargs...))
NUFSHT._nufft_setpts!(p::HostBouncePlan, x, y) = (NUFSHT._nufft_setpts!(p.inner, Array(x), Array(y)); p)
NUFSHT._nufft_destroy!(::HostBouncePlan) = nothing
function NUFSHT._nufft_exec!(p::HostBouncePlan, input, output)
    hout = Array(output)
    NUFSHT._nufft_exec!(p.inner, Array(input), hout)
    return copyto!(output, hout)
end

# An FFT for JLArrays, which has none, in the same way: each execution moves its array across in bulk.
const _AbstractFFTs = NUFSHT.FFTW.AbstractFFTs
struct HostBounceFFT{P}
    inner::P
end
_AbstractFFTs.plan_fft!(x::JLArrays.JLArray, region; kwargs...) =
    HostBounceFFT(_AbstractFFTs.plan_fft!(Array(x), region))
_AbstractFFTs.plan_bfft!(x::JLArrays.JLArray, region; kwargs...) =
    HostBounceFFT(_AbstractFFTs.plan_bfft!(Array(x), region))
function Base.:*(p::HostBounceFFT, x::JLArrays.JLArray)
    h = Array(x)
    p.inner * h
    return copyto!(x, h)
end

# The solver steps that are NUFSHT's own, on device arrays with scalar indexing an error: the scatter of
# the packed iterate into the plan's `(Nθ, Nφ, B)` layout, the gather of the adjoint back into it (with
# and without the fold of the previous vector, over the first `n` columns), and the write-out of a
# finished column to another column. Each is compared with the host loop.
Test.@testset "KernelAbstractions extension: device packing never scalar-indexes (JLArray)" begin
    Random.seed!(707)
    lmax, B, n = 6, 4, 3
    N, Nf = lmax + 1, 2lmax + 1
    full = N * Nf
    J(v) = JLArrays.JLArray(v)
    no_scalar(f) = task_local_storage(f, :ScalarIndexing, GPUArraysCore.ScalarDisallowed)
    Test.@test_throws ErrorException no_scalar(() -> J(zeros(3))[1])   # the guard bites

    idx = NUFSHT._valid_indices(zeros(0), lmax)
    K = length(idx)
    xp = randn(K, B); w = randn(N, Nf, B); v = randn(K, B); c = randn(B)

    Fc = zeros(N, Nf, B); NUFSHT._unpack_coeffs!(Fc, xp, idx, full, B)
    vc = copy(v); NUFSHT._col_pbp_pack!(vc, w, c, idx, full, n)
    pc = zeros(K, B); NUFSHT._pack_coeffs!(pc, w, idx, full, B)
    Cc = zeros(N, Nf, B); NUFSHT._write_solution!(Cc, xp, idx, full, 2, 3)

    Fj = J(zeros(N, Nf, B)); vj = J(copy(v)); pj = J(zeros(K, B)); Cj = J(zeros(N, Nf, B))
    idxj = J(idx)
    no_scalar() do
        NUFSHT._unpack_coeffs!(Fj, J(xp), idxj, full, B)
        NUFSHT._col_pbp_pack!(vj, J(w), J(c), idxj, full, n)
        NUFSHT._pack_coeffs!(pj, J(w), idxj, full, B)
        NUFSHT._write_solution!(Cj, J(xp), idxj, full, 2, 3)
    end
    Test.@test Array(Fj) == Fc
    Test.@test Array(vj) ≈ vc
    Test.@test Array(vj)[:, (n + 1):end] == v[:, (n + 1):end]   # the count bounded the fold
    Test.@test Array(pj) == pc
    Test.@test Array(Cj) == Cc
    # Column 2 of the iterate lands in column 3, at the valid slots, with zero everywhere else.
    Test.@test Cc[:, :, 3][idx] == xp[:, 2]
    Test.@test all(iszero, vec(Cc[:, :, 3])[setdiff(1:full, idx)])
end

# `nusht_solve!` and `nusht_solve_spin!` end to end on device arrays with scalar indexing an error,
# scored against the coefficients the fields were synthesised from by an independent evaluation. The
# plans are built with `execution = GPUBackend(JLBackend())`.
Test.@testset "KernelAbstractions extension: device solves never scalar-index (JLArray)" begin
    J(v) = JLArrays.JLArray(v)
    no_scalar(f) = task_local_storage(f, :ScalarIndexing, GPUArraysCore.ScalarDisallowed)
    Test.@test_throws ErrorException no_scalar(() -> J(zeros(3))[1])   # the guard bites
    lmax, B = 6, 2
    M = 4 * (lmax + 1)^2
    θ, φ = fib_points(M)
    nb = HostBounceNUFFT()
    gpu = ComputationalBackends.GPUBackend(JLArrays.JLBackend())

    # Scalar plan, real and complex field.
    for FE in (Float64, ComplexF64)
        Random.seed!(808)
        Cr = rand_coeffs(lmax, 808, B)
        Ct = FE <: Real ? Cr : Cr .+ im .* rand_coeffs(lmax, 809, B)
        f = zeros(FE, M, B)
        for b in 1:B
            f[:, b] .= synth_ref(real.(Ct[:, :, b]), lmax, θ, φ)
            FE <: Complex && (f[:, b] .+= im .* synth_ref(imag.(Ct[:, :, b]), lmax, θ, φ))
        end
        p = NUFSHT.make_plan(FE, θ, φ, lmax; execution = gpu, ntrans = B, nufft = nb)
        ws = FTB.LSMRWorkspace(p)
        C = NUFSHT.allocate_coefficients(p)
        Test.@test C isa GPUArraysCore.AbstractGPUArray
        fj = J(f)
        res = no_scalar(() -> NUFSHT.nusht_solve!(C, fj, p; ws = ws, rtol = 1e-11, maxiter = 100))
        Test.@test res[4]
        Test.@test relerr(Array(C), Ct) < 1e-9
        g = J(zeros(FE, M, B))
        no_scalar(() -> NUFSHT.nusht_type2!(g, C, p))
        Test.@test relerr(Array(g), f) < 1e-9
        NUFSHT.close!(p)
    end

    # Spin plans: a real spin-0 field, whose fit runs over the packed Hermitian degrees, and a
    # complex spin-1 one over the full array.
    for (FE, s) in ((Float64, 0), (ComplexF64, 1))
        Random.seed!(810 + s)
        sf = zeros(ComplexF64, lmax + 1, 2lmax + 1, B)
        for b in 1:B, ℓ in abs(s):lmax
            sf[NUFSHT.spin_coeff_index(ℓ, 0, lmax), b] = FE <: Real ? randn() : randn(ComplexF64)
            for m in 1:ℓ
                a = randn(ComplexF64)
                sf[NUFSHT.spin_coeff_index(ℓ, m, lmax), b] = a
                sf[NUFSHT.spin_coeff_index(ℓ, -m, lmax), b] =
                    FE <: Real ? (-1)^m * conj(a) : randn(ComplexF64)
            end
        end
        fc = [sum(sf[NUFSHT.spin_coeff_index(ℓ, m, lmax), b] * NUFSHT.sYlm(s, ℓ, m, θ[i], φ[i])
                  for ℓ in abs(s):lmax for m in -ℓ:ℓ) for i in 1:M, b in 1:B]
        f = FE <: Real ? real.(fc) : fc
        p = NUFSHT.make_spin_plan(FE, θ, φ, lmax, s; execution = gpu, ntrans = B, nufft = nb)
        ws = FTB.LSMRWorkspace(p)
        S = NUFSHT.allocate_coefficients(p)
        Test.@test S isa GPUArraysCore.AbstractGPUArray
        fj = J(f)
        res = no_scalar(() -> NUFSHT.nusht_solve_spin!(S, fj, p; ws = ws, rtol = 1e-11, maxiter = 100))
        Test.@test res[4]
        Test.@test relerr(Array(S), sf) < 1e-9
        NUFSHT.close!(p)
    end
end

# A device spin plan runs the ring S-step in its kernels; its synthesis and adjoint equal a host plan's
# of the same element type, batched, on both layouts, with scalar indexing an error.
Test.@testset "KernelAbstractions extension: device ring S-step equals the host plan (JLArray)" begin
    J(v) = JLArrays.JLArray(v)
    no_scalar(f) = task_local_storage(f, :ScalarIndexing, GPUArraysCore.ScalarDisallowed)
    lmax, B, M = 24, 2, 300
    θ, φ = iid_points(M, 840)
    gpu = ComputationalBackends.GPUBackend(JLArrays.JLBackend())
    DS = NUFSHT.SpectralBackends.DirectSumSpectralBackend()
    for (FE, s) in ((ComplexF64, 1), (ComplexF64, -2), (Float64, 0), (ComplexF32, 1), (Float32, 0))
        T = real(FE)
        pd = NUFSHT.make_spin_plan(FE, T.(θ), T.(φ), lmax, s; execution = gpu, ntrans = B,
                                   nufft = HostBounceNUFFT(), tol = 1e-12)
        ph = NUFSHT.make_spin_plan(FE, T.(θ), T.(φ), lmax, s; nthreads = 1, ntrans = B, nufft = DS, tol = 1e-12)
        Test.@test pd.engine isa NUFSHT._RingEngine
        sf = Complex{T}.(cat((_spin_coeffs(lmax, s, 841 + b; hermitian = FE <: Real) for b in 1:B)...; dims = 3))
        fh = zeros(FE, M, B); NUFSHT.nusht_type2_spin!(fh, sf, ph)
        fd = J(zeros(FE, M, B)); no_scalar(() -> NUFSHT.nusht_type2_spin!(fd, J(sf), pd))
        Random.seed!(843 + s)
        y = FE <: Real ? randn(T, M, B) : randn(Complex{T}, M, B)
        ah = zeros(Complex{T}, size(sf)); NUFSHT.nusht_type1_spin!(ah, y, ph)
        ad = J(zeros(Complex{T}, size(sf))); no_scalar(() -> NUFSHT.nusht_type1_spin!(ad, J(y), pd))
        tol = T === Float64 ? 1e-12 : lmax * eps(Float32)
        Test.@test relerr(Array(fd), fh) < tol
        Test.@test relerr(Array(ad), ah) < tol
        NUFSHT.close!(pd); NUFSHT.close!(ph)
    end
end

# Single-precision device scalar plans: the sphere step runs in double precision on the host, so each
# batch slice crosses between the plan's element type on the device and Float64 on the host, in both
# directions. Synthesis is scored against the direct sum, the adjoint against `Σⱼ yⱼ Y_lm(xⱼ)` evaluated
# directly, at single-precision accuracy.
Test.@testset "KernelAbstractions extension: single-precision device scalar plans (JLArray)" begin
    J(v) = JLArrays.JLArray(v)
    no_scalar(f) = task_local_storage(f, :ScalarIndexing, GPUArraysCore.ScalarDisallowed)
    lmax, B = 6, 2
    M = 4 * (lmax + 1)^2
    θ, φ = fib_points(M)
    valid = valid_mask_ref(lmax)
    gpu = ComputationalBackends.GPUBackend(JLArrays.JLBackend())
    for FE in (Float32, ComplexF32)
        Ct = rand_coeffs(lmax, 820, B)
        FE <: Complex && (Ct = Ct .+ im .* rand_coeffs(lmax, 821, B))
        ref = zeros(FE <: Real ? Float64 : ComplexF64, M, B)
        for b in 1:B
            ref[:, b] .= synth_ref(real.(Ct[:, :, b]), lmax, θ, φ)
            FE <: Complex && (ref[:, b] .+= im .* synth_ref(imag.(Ct[:, :, b]), lmax, θ, φ))
        end
        p = NUFSHT.make_plan(FE, Float32.(θ), Float32.(φ), lmax; execution = gpu, ntrans = B,
                             nufft = HostBounceNUFFT(), tol = 1e-6)
        g = J(zeros(FE, M, B))
        no_scalar(() -> NUFSHT.nusht_type2!(g, J(FE.(Ct)), p))
        Test.@test relerr(Array(g), ref) < 2e-5

        Random.seed!(822)
        y = FE <: Real ? randn(M, B) : randn(ComplexF64, M, B)
        A = NUFSHT.allocate_coefficients(p)
        no_scalar(() -> NUFSHT.nusht_type1!(A, J(FE.(y)), p))
        for b in 1:B
            Ab = Array(A)[:, :, b]
            Test.@test relerr(Ab[valid], quadrature_ref(y[:, b], ones(M), lmax, θ, φ)[valid]) < 2e-5
        end
        NUFSHT.close!(p)
    end
end

# `nusht_analyze!` with weights and the scalar ↔ spin-0 conversions on device arrays with scalar indexing
# an error. On a rule exact for the product of two degree-`lmax` harmonics the quadrature returns the
# coefficients a field was synthesized from by the direct sum, for a scalar plan and a real-field spin-0
# plan (whose result passes through the Hermitian projection). The converted coefficients synthesize the
# direct sum's field.
Test.@testset "KernelAbstractions extension: device analysis and spin-0 conversions (JLArray)" begin
    J(v) = JLArrays.JLArray(v)
    no_scalar(f) = task_local_storage(f, :ScalarIndexing, GPUArraysCore.ScalarDisallowed)
    lmax = 6
    θ, φ, w = gauss_legendre_grid(lmax)
    M = length(θ)
    nb = HostBounceNUFFT()
    gpu = ComputationalBackends.GPUBackend(JLArrays.JLBackend())

    C = rand_coeffs(lmax, 830)
    f = synth_ref(C, lmax, θ, φ)
    p = NUFSHT.make_plan(Float64, θ, φ, lmax; execution = gpu, nufft = nb)
    Cd = NUFSHT.allocate_coefficients(p)
    res = no_scalar(() -> NUFSHT.nusht_analyze!(Cd, J(f), p; weights = J(w)))
    Test.@test res.iterations == 0 && res.converged
    Test.@test relerr(Array(Cd)[:, :, 1], C) < 1e-11
    # A zero weight drops its point whatever the field holds there.
    wm = copy(w); wm[1:5:M] .= 0
    fz = copy(f); fz[1:5:M] .= 0
    fn = copy(f); fn[1:5:M] .= NaN
    Cz = NUFSHT.allocate_coefficients(p); Cn = NUFSHT.allocate_coefficients(p)
    no_scalar(() -> NUFSHT.nusht_analyze!(Cz, J(fz), p; weights = J(wm)))
    no_scalar(() -> NUFSHT.nusht_analyze!(Cn, J(fn), p; weights = J(wm)))
    Test.@test Array(Cn) == Array(Cz)
    Test.@test relerr(Array(Cn)[:, :, 1], quadrature_ref(fz, wm, lmax, θ, φ)) < 1e-11
    NUFSHT.close!(p)

    Random.seed!(831)
    sh = zeros(ComplexF64, lmax + 1, 2lmax + 1)
    for ℓ in 0:lmax
        sh[NUFSHT.spin_coeff_index(ℓ, 0, lmax)] = randn()
        for m in 1:ℓ
            a = randn(ComplexF64)
            sh[NUFSHT.spin_coeff_index(ℓ, m, lmax)] = a
            sh[NUFSHT.spin_coeff_index(ℓ, -m, lmax)] = (-1)^m * conj(a)
        end
    end
    f0 = [real(sum(sh[NUFSHT.spin_coeff_index(ℓ, m, lmax)] * NUFSHT.sYlm(0, ℓ, m, θ[j], φ[j])
                   for ℓ in 0:lmax for m in -ℓ:ℓ)) for j in 1:M]
    p0 = NUFSHT.make_spin_plan(Float64, θ, φ, lmax, 0; execution = gpu, nufft = nb)
    S0 = NUFSHT.allocate_coefficients(p0)
    no_scalar(() -> NUFSHT.nusht_analyze!(S0, J(f0), p0; weights = J(w)))
    Test.@test relerr(Array(S0)[:, :, 1], sh) < 1e-10
    NUFSHT.close!(p0)

    pc = NUFSHT.make_spin_plan(ComplexF64, θ, φ, lmax, 0; execution = gpu, nufft = nb)
    sfd = NUFSHT.allocate_coefficients(pc)
    no_scalar(() -> NUFSHT.scalar_to_spin0!(sfd, J(C)))
    g = J(zeros(ComplexF64, M))
    no_scalar(() -> NUFSHT.nusht_type2_spin!(g, sfd, pc))
    Test.@test relerr(Array(g), complex.(f)) < 1e-11
    NUFSHT.close!(pc)

    Cd2 = J(zeros(lmax + 1, 2lmax + 1))
    no_scalar(() -> NUFSHT.spin0_to_scalar!(Cd2, J(sh)))
    Test.@test relerr(synth_ref(Array(Cd2), lmax, θ, φ), f0) < 1e-12
end

# The device spectral filter against the host slot loop, for a smooth and a sharp transfer.
Test.@testset "KernelAbstractions extension: device apply_transfer! (JLArray)" begin
    Random.seed!(606)
    lmax = 8; N = lmax + 1; Nφ = 2lmax + 1; B = 3
    for filt in (NUFSHT.gaussian_from_scale(2000e3), NUFSHT.TopHatTransfer(4))
        C = randn(N, Nφ, B)
        Cc = copy(C); NUFSHT.apply_transfer!(Cc, filt, lmax)
        Cj = JLArrays.JLArray(copy(C)); NUFSHT.apply_transfer!(Cj, filt, lmax)
        Test.@test Array(Cj) == Cc
    end
end
