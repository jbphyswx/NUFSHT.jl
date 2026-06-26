using NUFSHT
using FastTransforms: FastTransforms as FT
using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using FINUFFT: FINUFFT
using FFTW
using Random; Random.seed!(2)
relerr(a,b)=maximum(abs.(a.-b))/maximum(abs.(b))

lmax=6; Nθ=lmax+1; Nφ=2lmax+1
θg, φg = FSH.sph_points(Nθ)
θpts = repeat(θg, outer=Nφ); φpts = repeat(φg, inner=Nθ)

# generic spin synthesis with configurable doubling: sgn and conj on reflected rows
function synth(C, s, lmax, θpts, φpts; sgn=1, conj_reflect=false, tol=1e-12)
    Nθ=lmax+1; Nφ=2lmax+1
    P = FT.plan_spinsph2fourier(C,s); PS = FT.plan_spinsph_synthesis(C,s)
    F = PS*(P*copy(C))
    F̃ = zeros(ComplexF64, 2Nθ, Nφ); half = Nφ÷2
    F̃[1:Nθ,:] .= F
    for i in 1:Nθ, j in 1:Nφ
        v = F[Nθ+1-i, mod1(j+half,Nφ)]
        F̃[Nθ+i,j] = sgn*(conj_reflect ? conj(v) : v)
    end
    Nθd=2Nθ; kθ=[k<Nθd÷2 ? k : k-Nθd for k in 0:Nθd-1]
    phase=exp.(-im .* π .* kθ ./ Nθd)
    Fhat=(fft(F̃).*phase)./(Nθd*Nφ); Fhat2d=collect(fftshift(Fhat)')
    return FINUFFT.nufft2d2(collect(φpts),collect(θpts),-1,tol,Fhat2d)
end

# Test A: spin-0 through this machinery vs scalar synthesis
Cs0 = complex.(randn(Nθ,Nφ))
F0 = (FT.plan_sph_synthesis(real(Cs0)))*((FT.plan_sph2fourier(real(Cs0)))*real(copy(Cs0)))
# use spin synth with s=0
a = synth(complex(real(Cs0)), 0, lmax, θpts, φpts; sgn=1, conj_reflect=false)
println("Test A (s=0 via spin machinery) rel_err=", relerr(a, vec(F0)))

# Test B: sweep for s=1
s=1; C = FT.spinsphrandn(ComplexF64,Nθ,Nφ,s)
Fsp = (FT.plan_spinsph_synthesis(C,s))*((FT.plan_spinsph2fourier(C,s))*copy(C))
for sgn in (1,-1), cj in (false,true)
    f = synth(C, s, lmax, θpts, φpts; sgn=sgn, conj_reflect=cj)
    println("s=1 sgn=$sgn conj=$cj : rel_err=", relerr(f, vec(Fsp)))
end
