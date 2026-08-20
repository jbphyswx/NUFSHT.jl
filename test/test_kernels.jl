Test.@testset "Kernel transfer functions" begin
    Test.@testset "TopHat" begin
        f = NUFSHT.TopHatTransfer(10)
        Test.@test NUFSHT.kernel_transfer(f, 0)  == 1.0
        Test.@test NUFSHT.kernel_transfer(f, 10) == 1.0
        Test.@test NUFSHT.kernel_transfer(f, 11) == 0.0
        Test.@test NUFSHT.kernel_transfer(f, 100) == 0.0
    end

    Test.@testset "Gaussian" begin
        f = NUFSHT.gaussian_from_scale(200e3)
        Test.@test NUFSHT.kernel_transfer(f, 0) ≈ 1.0
        h1   = NUFSHT.kernel_transfer(f, 1)
        h10  = NUFSHT.kernel_transfer(f, 10)
        h100 = NUFSHT.kernel_transfer(f, 100)
        Test.@test h1 > h10 > h100 > 0
        Test.@test h100 < 1e-2
    end

    Test.@testset "cutoff_degree" begin
        L = NUFSHT.cutoff_degree(100e3)
        Test.@test L ≈ round(Int, π * 6.371e6 / 100e3)
        Test.@test L > 0
    end

    # Every slot of the coefficient array holds a real degree `ℓ = i + j÷2 - 1`, including those past
    # `ℓ = lmax` that make the transform square and invertible. A spectral filter must attenuate by
    # each slot's own degree; leaving them at unity gain lets a low-pass pass `ℓ > lmax` through at
    # full amplitude.
    Test.@testset "transfer uses each slot's own degree" begin
        lmax = 6
        filt = NUFSHT.TopHatTransfer(3)                  # H(ℓ) = 1 for ℓ ≤ 3, else 0
        C = ones(lmax + 1, 2lmax + 1)
        NUFSHT.apply_transfer!(C, filt, lmax)
        for j in 1:(2lmax + 1), i in 1:(lmax + 1)
            Test.@test C[i, j] == (i + (j ÷ 2) - 1 <= 3 ? 1.0 : 0.0)
        end
        # Supernumerary slots exist at this bandlimit and must be attenuated, not passed.
        Test.@test any(i -> i + lmax - 1 > lmax, 1:(lmax + 1))
        # The device broadcast matrix and the CPU loop must agree on every slot, not just ℓ ≤ lmax.
        Test.@test NUFSHT._transfer_matrix(filt, lmax, Float64) == C
    end
end

# `kernel_transfer` must be evaluated once per DEGREE, not once per slot — the array holds
# `(lmax+1)(2lmax+1)` slots spanning only `2lmax+1` degrees. Counted, not timed: deterministic, and it
# pins the property rather than a wall-clock threshold.
struct CountingTransfer
    n::Base.RefValue{Int}
end
NUFSHT.kernel_transfer(f::CountingTransfer, ℓ::Integer) = (f.n[] += 1; 1.0)

Test.@testset "transfer is evaluated once per degree" begin
    for lmax in (8, 32, 64)
        f = CountingTransfer(Ref(0))
        NUFSHT.apply_transfer!(ones(lmax + 1, 2lmax + 1), f, lmax)
        Test.@test f.n[] == 2lmax + 1                       # not (lmax+1)(2lmax+1)

        g = CountingTransfer(Ref(0))
        NUFSHT._transfer_matrix(g, lmax, Float64)
        Test.@test g.n[] == 2lmax + 1
    end
    # Allocation-free: the degree sweep must not build a table.
    C = ones(9, 17); filt = NUFSHT.gaussian_from_scale(200e3)
    NUFSHT.apply_transfer!(C, filt, 8)
    Test.@test (@allocated NUFSHT.apply_transfer!(C, filt, 8)) == 0
end
