# OpenBLAS Performance Troubleshooting & Resolution Guide

> **Status 2026-10-31: the numbers below are historical.** `CpuBackend.MatMul` now uses the in-tree `CpuSgemm` (pure C#, no native library; commit `c9b47c69`), and `libopenblas.dll` no longer ships. Measured on this machine with no OpenBLAS on `PATH` or in `tools/openblas` (Release CLI, Qwen3-4B Q4_K_M, 799-token prompt, `-g 0 --temp 0`, 3 runs): prefill 102.8, 89.4 and 98.3 t/s, decode 12.7, 12.0 and 12.3 t/s. That is nowhere near the ~4 t/s decode quoted in section 1, which was measured on a different model before `CpuSgemm`. `BlasInterop` still probes for the library and uses it only for F32 batched products; the `OpenBLAS: not found (fallback to sequential)` line is only printed from `SimdKernels.MatMulBatched` and no longer signals a large slowdown. Section 3 (the setup script) is optional.

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
