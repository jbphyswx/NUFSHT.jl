using FastTransforms: FastTransforms as FT
using FINUFFT: FINUFFT
using FFTW
using Random; Random.seed!(3)

spin_col(m) = 2*abs(m) + (m >= 0 ? 1 : 0)
spin_row(ℓ,m,s) = ℓ - max(abs(m),abs(s)) + 1
spin_idx(ℓ,m,s) = CartesianIndex(spin_row(ℓ,m,s), spin_col(m))

# scattered spin synthesis: coefficients C (Nθ,Nφ) at lmax -> complex values at (θpts,φpts)
function spin_synth_scattered(C, s, lmax, θpts, φpts; tol=1e-12, sign_reflect=(-1)^s)
    Nθ = lmax+1; Nφ = 2lmax+1
    P  = FT.plan_spinsph2fourier(C, s)
    PS = FT.plan_spinsph_synthesis(C, s)
    F = PS*(P*copy(C))                       # (Nθ,Nφ) complex grid values
    # spin-aware DFS doubling
    F̃ = zeros(ComplexF64, 2Nθ, Nφ)
    F̃[1:Nθ,:] .= F
    half = Nφ ÷ 2
    for i in 1:Nθ, j in 1:Nφ
        F̃[Nθ+i, j] = sign_reflect * F[Nθ+1-i, mod1(j+half, Nφ)]
    end
    # fft2_to_coeffs (complex)
    Nθd = 2Nθ
    kθ = [k < Nθd÷2 ? k : k - Nθd for k in 0:(Nθd-1)]
    phase = exp.(-im .* π .* kθ ./ Nθd)
    Fhat = (fft(F̃) .* phase) ./ (Nθd*Nφ)
    Fhat2d = collect(fftshift(Fhat)')
    f = FINUFFT.nufft2d2(collect(float(φpts)), collect(float(θpts)), -1, tol, Fhat2d)
    return f
end

function embed(C, s, lmax, lmax2)
    Nθ2=lmax2+1; Nφ2=2lmax2+1
    C2 = zeros(ComplexF64, Nθ2, Nφ2)
    for m in -lmax:lmax, ℓ in max(abs(m),abs(s)):lmax
        C2[spin_idx(ℓ,m,s)] = C[spin_idx(ℓ,m,s)]
    end
    return C2
end

lmax=6; s=1; Nθ=lmax+1; Nφ=2lmax+1
C = FT.spinsphrandn(ComplexF64, Nθ, Nφ, s)

lmax2 = 2lmax; Nθ2=lmax2+1; Nφ2=2lmax2+1
C2 = embed(C, s, lmax, lmax2)
P2 = FT.plan_spinsph2fourier(C2, s); PS2 = FT.plan_spinsph_synthesis(C2, s)
F2 = PS2*(P2*copy(C2))                       # exact band-limited field on finer CC grid
θ2 = [π*(i-0.5)/Nθ2 for i in 1:Nθ2]; φ2 = [2π*(j-1)/Nφ2 for j in 1:Nφ2]
θpts = repeat(θ2, outer=Nφ2); φpts = repeat(φ2, inner=Nθ2)
ref = vec(F2)

for sr in (+1, -1)
    f = spin_synth_scattered(C, s, lmax, θpts, φpts; sign_reflect=sr)
    err = maximum(abs.(f .- ref)) / maximum(abs.(ref))
    println("sign_reflect=$sr : rel_err=", err)
end

# Diagnostic: evaluate at ORIGINAL CC grid points (northern only) — isolates grid/phase
# convention from the pole-reflection rule.
θ1 = [π*(i-0.5)/Nθ for i in 1:Nθ]; φ1 = [2π*(j-1)/Nφ for j in 1:Nφ]
θg = repeat(θ1, outer=Nφ); φg = repeat(φ1, inner=Nφ==0 ? 1 : Nθ)
F = (FT.plan_spinsph_synthesis(C,s))*((FT.plan_spinsph2fourier(C,s))*copy(C))
for sr in (+1,-1)
    f = spin_synth_scattered(C, s, lmax, θg, φg; sign_reflect=sr)
    err = maximum(abs.(f .- vec(F)))/maximum(abs.(vec(F)))
    println("origin-grid sign=$sr rel_err=", err)
end
