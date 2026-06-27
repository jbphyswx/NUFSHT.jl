using NUFSHT, LinearAlgebra, Random
Random.seed!(1); rn(x)=sqrt(sum(abs2,x))

# Pure-rotational Rossby field on a lat-lon point set (treated as scattered)
Nlon,Nlat=48,40
lons=collect(range(0,2π-2π/Nlon,length=Nlon)); lats=collect(range(-1.3,1.3,length=Nlat))
λ=Float64[]; φlat=Float64[]; ue=Float64[]; un=Float64[]
n,m=3,2
for la in lats, lo in lons
    cphi=cos(la); sphi=sin(la)
    push!(λ,lo); push!(φlat,la)
    push!(ue, n*cos(m*lo)*cphi^(n-1)*sphi)             # u_east (rotational rossby)
    push!(un, -m/cphi*sin(m*lo)*cphi^n)                # u_north
end
θ = π/2 .- φlat                                        # colatitude
# spin combos: u_θ = -u_north (θ̂ points south), u_φ = u_east
uθ = -un; uφ = ue
Vp = uθ .+ im.*uφ        # spin +1
Vm = uθ .- im.*uφ        # spin -1

lmax=24; shp=(lmax+1,2lmax+1)
planp = NUFSHT.make_spin_plan(θ, λ, lmax, +1; tol=1e-11)
planm = NUFSHT.make_spin_plan(θ, λ, lmax, -1; tol=1e-11)
ap=zeros(ComplexF64,shp); NUFSHT.nusht_solve_spin!(ap, Vp, planp; rtol=1e-9, maxiter=600)
am=zeros(ComplexF64,shp); NUFSHT.nusht_solve_spin!(am, Vm, planm; rtol=1e-9, maxiter=600)

sym=(ap.+am)./2; anti=(ap.-am)./2
# reconstruct field from a given (a+1,a-1) pair
function recon(ap1, am1)
    Vp1=similar(Vp); NUFSHT.nusht_type2_spin!(Vp1, ap1, planp)
    Vm1=similar(Vm); NUFSHT.nusht_type2_spin!(Vm1, am1, planm)
    uθ1=(Vp1.+Vm1)./2; uφ1=(Vp1.-Vm1)./(2im)
    ue1=real(uφ1); un1=-real(uθ1)
    return ue1, un1
end
# candidate A: sym as E-type (a+1=a-1=sym)
ueA,unA = recon(sym, sym)
# candidate B: anti as B-type (a+1=anti, a-1=-anti)
ueB,unB = recon(anti, -anti)
U=vcat(ue,un)
println("‖fieldA(sym)‖/‖U‖ = ", rn(vcat(ueA,unA))/rn(U))
println("‖fieldB(anti)‖/‖U‖ = ", rn(vcat(ueB,unB))/rn(U))
println("A+B reconstruction err = ", maximum(abs.(vcat(ueA.+ueB, unA.+unB) .- U))/maximum(abs.(U)))
