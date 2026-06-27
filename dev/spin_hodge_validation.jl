using NUFSHT, LinearAlgebra, Random
rn(x)=sqrt(sum(abs2,x))
Nlon,Nlat=48,40
lons=collect(range(0,2π-2π/Nlon,length=Nlon)); lats=collect(range(-1.2,1.2,length=Nlat))
λ=Float64[]; φlat=Float64[]
for la in lats, lo in lons; push!(λ,lo); push!(φlat,la); end
θ=π/2 .- φlat
lmax=26; shp=(lmax+1,2lmax+1)
planp=NUFSHT.make_spin_plan(θ,λ,lmax,+1; tol=1e-11)
planm=NUFSHT.make_spin_plan(θ,λ,lmax,-1; tol=1e-11)

function hodge(ue,un)
    uθ=-un; uφ=ue; Vp=uθ.+im.*uφ; Vm=uθ.-im.*uφ
    ap=zeros(ComplexF64,shp); NUFSHT.nusht_solve_spin!(ap,Vp,planp;rtol=1e-9,maxiter=700)
    am=zeros(ComplexF64,shp); NUFSHT.nusht_solve_spin!(am,Vm,planm;rtol=1e-9,maxiter=700)
    sym=(ap.+am)./2; anti=(ap.-am)./2
    rec(a1,a2)=begin
        V1=similar(Vp); NUFSHT.nusht_type2_spin!(V1,a1,planp)
        V2=similar(Vm); NUFSHT.nusht_type2_spin!(V2,a2,planm)
        uθ1=(V1.+V2)./2; uφ1=(V1.-V2)./(2im); (real(uφ1), -real(uθ1))   # (u_east,u_north)
    end
    ue_rot,un_rot = rec(sym,sym)       # rotational
    ue_div,un_div = rec(anti,-anti)    # divergent
    return ue_rot,un_rot,ue_div,un_div
end

# pure divergent: χ=sin(qλ)cos^p φ
q,p=2,3; ue=Float64[]; un=Float64[]
for (lo,la) in zip(λ,φlat)
    c=cos(la)
    push!(ue, q/c*cos(q*lo)*c^p)               # (1/cosφ)∂_λχ
    push!(un, -p*sin(q*lo)*c^(p-1)*sin(la))    # ∂_φχ
end
U=vcat(ue,un); uer,unr,ued,und=hodge(ue,un)
println("DIV field: rot/U=",rn(vcat(uer,unr))/rn(U)," div/U=",rn(vcat(ued,und))/rn(U),
        " recon=",maximum(abs.(vcat(uer.+ued,unr.+und).-U))/maximum(abs.(U)))

# mixed: rotational rossby + 0.5*divergent
ue2=Float64[]; un2=Float64[]; n,m=3,2
for (lo,la) in zip(λ,φlat)
    c=cos(la); s=sin(la)
    push!(ue2, n*cos(m*lo)*c^(n-1)*s + 0.5*(q/c*cos(q*lo)*c^p))
    push!(un2, -m/c*sin(m*lo)*c^n + 0.5*(-p*sin(q*lo)*c^(p-1)*s))
end
U2=vcat(ue2,un2); uer2,unr2,ued2,und2=hodge(ue2,un2)
println("MIXED: rot/U=",round(rn(vcat(uer2,unr2))/rn(U2),digits=3),
        " div/U=",round(rn(vcat(ued2,und2))/rn(U2),digits=3),
        " recon=",maximum(abs.(vcat(uer2.+ued2,unr2.+und2).-U2))/maximum(abs.(U2)))
