# Agent notes (NUFSHT.jl)

## Julia Import Style — `using X: X` then `X.y()`

**Rule:** Always use `using X: X` to bring the module into scope, then call all methods as `X.y()`. Never use bare `using X`, bare `import X`, or `using X: y` (listing individual symbols).

**Always write:**
```julia
using FFTW: FFTW
using FINUFFT: FINUFFT
using FastSphericalHarmonics: FastSphericalHarmonics
# then call as: FFTW.fft(...), FINUFFT.nufft2d1(...), FastSphericalHarmonics.sph_mode(...)
```

**Never write:**
```julia
using FFTW                   # ❌ bare using, pollutes namespace
import FFTW                  # ❌ bare import
using FFTW: fft, ifft        # ❌ listing individual symbols
import FFTW: fft             # ❌ listing individual symbols
```

This is enforced by `Aqua.jl` (`test_all`) in `test/runtests.jl`.

## Other Rules

- **Never invent Julia package UUIDs.** Use `Pkg.add` for strong deps; use registry lookup for extras/weakdeps/extensions. See memory note.
- **Destructive git:** Do not use `git checkout` / `git restore` / `git reset --hard` unless explicitly requested.
- **No Legacy Cruft:** Strictly adhere to spec discipline — no shims, no fallback paths for renamed things.
