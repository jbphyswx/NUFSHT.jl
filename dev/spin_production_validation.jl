using NUFSHT, LinearAlgebra, Random
Random.seed!(9); relerr(a,b)=maximum(abs.(a.-b))/maximum(abs.(b))
lmax=10; s=1; shp=(lmax+1,2lmax+1)
θ=π .* rand(800); φ=2π .* rand(800)
plan = NUFSHT.make_spin_plan(θ,φ,lmax,s; tol=1e-12)

# synthesis vs direct sYlm
sf = zeros(ComplexF64, shp)
for ℓ in 1:lmax, m in -ℓ:ℓ; sf[NUFSHT.spin_coeff_index(ℓ,m,lmax)] = randn(ComplexF64); end
f = similar(θ,ComplexF64); NUFSHT.nusht_type2_spin!(f, sf, plan)
ref = [sum(sf[NUFSHT.spin_coeff_index(ℓ,m,lmax)]*NUFSHT.sYlm(s,ℓ,m,θ[j],φ[j]) for ℓ in 1:lmax for m in -ℓ:ℓ) for j in eachindex(θ)]
println("PROD synthesis vs direct sYlm rel_err=", relerr(f, ref))

# adjoint test
x = randn(ComplexF64, shp); y = randn(ComplexF64, length(θ))
Ax=similar(y); NUFSHT.nusht_type2_spin!(Ax, x, plan)
Aty=similar(x); NUFSHT.nusht_type1_spin!(Aty, y, plan)
println("PROD adjoint test = ", abs(dot(Ax,y)-dot(vec(x),vec(Aty)))/abs(dot(Ax,y)))

# round-trip solve (overdetermined)
M=4*(lmax+1)^2; θr=π .* rand(M); φr=2π .* rand(M)
planr=NUFSHT.make_spin_plan(θr,φr,lmax,s; tol=1e-12)
fr=similar(θr,ComplexF64); NUFSHT.nusht_type2_spin!(fr, sf, planr)
sol=zeros(ComplexF64,shp); _,it,rr=NUFSHT.nusht_solve_spin!(sol, fr, planr; rtol=1e-10, maxiter=400)
println("PROD solve: iters=$it rel_res=$rr coeff_err=", relerr(sol, sf))
