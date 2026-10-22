Test.@testset "Kernel transfer functions" begin
    Test.@testset "Gaussian" begin
        f = NUFSHT.gaussian_from_scale(200e3)
        σ = 200e3 / 6.371e6
        for ℓ in (0, 1, 10, 100)
            Test.@test NUFSHT.kernel_transfer(f, ℓ) ≈ exp(-ℓ * (ℓ + 1) * σ^2 / 2)
        end
    end

    Test.@testset "cutoff_degree" begin
        Test.@test NUFSHT.cutoff_degree(100e3) == 200            # π · 6.371e6 / 100e3 = 200.15
    end

    # Slot `(i, j)` holds degree `ℓ = i + j÷2 - 1`, the supernumerary slots past `ℓ = lmax` included, and
    # a filter scales each slot by its own `H(ℓ)`.
    Test.@testset "transfer uses each slot's own degree" begin
        lmax = 6
        filt = NUFSHT.TopHatTransfer(3)                  # H(ℓ) = 1 for ℓ ≤ 3, else 0
        C = ones(lmax + 1, 2lmax + 1)
        NUFSHT.apply_transfer!(C, filt, lmax)
        Test.@test C == [(i + (j ÷ 2) - 1 <= 3 ? 1.0 : 0.0) for i in 1:(lmax + 1), j in 1:(2lmax + 1)]
    end
end

# `kernel_transfer` runs once per degree: the array's `(lmax+1)(2lmax+1)` slots span `2lmax+1` degrees.
struct CountingTransfer
    n::Base.RefValue{Int}
end
NUFSHT.kernel_transfer(f::CountingTransfer, ℓ::Integer) = (f.n[] += 1; 1.0)

Test.@testset "transfer is evaluated once per degree" begin
    for lmax in (8, 32, 64)
        f = CountingTransfer(Ref(0))
        NUFSHT.apply_transfer!(ones(lmax + 1, 2lmax + 1), f, lmax)
        Test.@test f.n[] == 2lmax + 1
    end
end
