# OpenBLAS Performance Troubleshooting & Resolution Guide

## 1. Symptom

During CPU inference execution, the CLI or runtime server logs:

```text
[OpenTail.Stingray] OpenBLAS: not found (fallback to sequential)
```

**Performance Impact:**
* CPU decode throughput drops from **~26.4 tokens/sec** down to **~4.0 tokens/sec** (a ~6.5x slowdown).
* CPU prefill throughput drops from **~145.6 tokens/sec** down to **~18.6 tokens/sec**.
* Execution degrades to single-threaded sequential C# SIMD fallback loops instead of multi-threaded native BLAS GEMM (`cblas_sgemm`).

---

## 2. Root Cause

OpenTail uses `BlasInterop.cs` to bind native `cblas_sgemm` matrix multiplication. 

At startup, `BlasInterop.ProbeLibrary()` probes:
1. Standard system `PATH` for `libopenblas.dll`.
2. Relative directory `tools/openblas/libopenblas.dll` (anchored to `AppContext.BaseDirectory`).

If `libopenblas.dll` is missing from both locations, `BlasInterop.IsAvailable` evaluates to `false`, and the engine safely degrades to CPU single-threaded sequential fallback.

---

## 3. Resolution (Developer / Developer Workstation)

Run the included setup script from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup-openblas.ps1
```

**What this script does:**
1. Downloads OpenBLAS v0.3.28 (`OpenBLAS-0.3.28-x64.zip`) from OpenMathLib releases.
2. Extracts `libopenblas.dll` (48.6 MB).
3. Places it in `tools/openblas/libopenblas.dll`.

Once installed, restarting any `dotnet run` command automatically detects OpenBLAS, eliminates the warning, and restores full multi-threaded CPU throughput (**26.4+ t/s**).

---

## 4. Future Packaging & Distribution Note

> [!NOTE]
> **Packaging Directive (Future Action Item)**:
> In future production releases (NuGet packages, standalone NativeAOT binaries, or installer bundles), `libopenblas.dll` should be packaged directly into the runtime output directory (e.g. via `.csproj` `<Content Include="tools/openblas/libopenblas.dll" CopyToOutputDirectory="PreserveNewest" />` or native runtime asset dependency) so end users achieve maximum CPU performance out-of-the-box without manual setup.
