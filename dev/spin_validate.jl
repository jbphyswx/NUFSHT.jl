using NUFSHT
using FastTransforms: FastTransforms as FT
using Random; Random.seed!(11)
relerr(a,b)=maximum(abs.(a.-b))/maximum(abs.(b))

s = 1
lmax = 8; Nθ=lmax+1; Nφ=2lmax+1
C = FT.spinsphrandn(ComplexF64, Nθ, Nφ, s)

# Exact reference: embed C into a finer band-limit, synthesize on the finer CC grid.
embed(C,lmax,lmax2) = begin
    Nθ2=lmax2+1; Nφ2=2lmax2+1; C2=zeros(ComplexF64,Nθ2,Nφ2)
    for m in -lmax:lmax, ℓ in max(abs(m),abs(s)):lmax
        C2[NUFSHT.spin_sph_mode(ℓ,m,s)] = C[NUFSHT.spin_sph_mode(ℓ,m,s)]
    end
    C2
end
lmax2=2lmax+2; Nθ2=lmax2+1; Nφ2=2lmax2+1
C2 = embed(C,lmax,lmax2)
F2 = (FT.plan_spinsph_synthesis(C2,s))*((FT.plan_spinsph2fourier(C2,s))*copy(C2))
θ2=[π*(i-0.5)/Nθ2 for i in 1:Nθ2]; φ2=[2π*(j-1)/Nφ2 for j in 1:Nφ2]
θpts=repeat(θ2,outer=Nφ2); φpts=repeat(φ2,inner=Nθ2)
ref = vec(F2)

# Production synthesis at those scattered points
plan = NUFSHT.make_spin_plan(θpts, φpts, lmax, s; tol=1e-12)
f = similar(ref); NUFSHT.nusht_type2_spin!(f, C, plan)
println("spin-1 synthesis vs exact finer-grid ref: rel_err = ", relerr(f, ref))

# Adjoint test: <A x, y> == <x, A† y>
M = length(ref)
xv = FT.spinsphrandn(ComplexF64, Nθ, Nφ, s)
yv = randn(ComplexF64, M)
Ax = similar(yv); NUFSHT.nusht_type2_spin!(Ax, xv, plan)
Aty = similar(xv); NUFSHT.nusht_type1_spin!(Aty, yv, plan)
using LinearAlgebra
lhs = dot(Ax, yv); rhs = dot(vec(xv), vec(Aty))
println("adjoint test |<Ax,y>-<x,A'y>|/|.| = ", abs(lhs-rhs)/abs(lhs))

# Round-trip solve: random scattered points, synth a field, solve back, re-synth, compare
M2 = 4*(lmax+1)^2
θr = π .* rand(M2); φr = 2π .* rand(M2)
planr = NUFSHT.make_spin_plan(θr, φr, lmax, s; tol=1e-12)
fr = similar(θr, ComplexF64); NUFSHT.nusht_type2_spin!(fr, C, planr)
Csol = similar(C); _,iters,rr = NUFSHT.nusht_solve_spin!(Csol, fr, planr; rtol=1e-10, maxiter=400)
fr2 = similar(fr); NUFSHT.nusht_type2_spin!(fr2, Csol, planr)
println("solve round-trip: iters=$iters rel_res=$rr  field rel_err=", relerr(fr2, fr))
