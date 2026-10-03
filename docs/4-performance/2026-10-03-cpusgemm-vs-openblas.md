# CpuSgemm vs OpenBLAS (2026-10-03)

**Decision:** `CpuBackend.Sgemm` (C[M,N] = A[M,K]·B[N,K]ᵀ, F32) runs on `CpuSgemm`, a pure-C# BLIS-style
packed GEMM ported from TensorSharp (`src/OpenTail.Stingray.Cpu/CpuSgemm.cs`, BSD-3). It replaces
OpenBLAS `cblas_sgemm`, whose fallback was a naive scalar triple loop. This was OpenBLAS's last live
call; the text path stopped using it on 2026-08-20 ([openblas-elimination-findings](../done/openblas-elimination-findings-2026-08-20.md)).

**Who uses this path on CPU:** every `VaeDecoder` built with a `CpuBackend` (FLUX, Z-Image, SD3,
SDXL, SD1.5, FLUX.2 image decode): each conv is im2col + `Sgemm`. The DiTs' CPU paths use the
quantized `MatMulBatched` kernels instead, and `CpuBackend` isn't an image/vision-ops backend, so
those `imageOps.Sgemm` sites never reach it.

## Method

- Machine: Ryzen 7 5700G (Zen 3, 8 cores / 16 threads, AVX2, no AVX-512), idle (load 2%).
- Kernel benchmark: `tools/kernel-bench-cs`, `OpenTail.Stingray.KernelBench sgemm 7`
  (`DOTNET_TC_QuickJitForLoops=0`, default tiering). 16 shapes taken from the CPU callers. The arms
  alternate inside each repetition; median of 7.
- Real weights: the FLUX.1 VAE (`models/flux1-schnell/ae.safetensors`) decoding one fixed random latent
  (16×64×64 → 512×512) on `CpuBackend`. The old arm was a worktree of 8a096e36 with `libopenblas.dll`
  beside it (`BlasAvailable=True` printed). Runs were ordered old, new, old, new, three decodes each.

## Result 1: OpenBLAS's lead was SMT

At first CpuSgemm (8 threads, the kernels' physical-core default) lost 3-22% on the large
compute-bound shapes. A KC/MC/NC cache-blocking sweep (`sgemm-sweep`, 13 configurations) found
nothing beyond noise (best geo-mean 0.987). The JIT's 6×16 inner loop has 2 loads, 6 broadcasts and
12 FMAs, with no spills.

The difference was the thread count: OpenBLAS runs 16 threads. With equal counts:

| Shape (M×K×N) | OpenBLAS 8T ms | CpuSgemm 8T ms | ratio | OpenBLAS default (16T) ms | CpuSgemm 16T ms | ratio |
|---|---:|---:|---:|---:|---:|---:|
| flux qkv 1024×3072×9216 | 122.98 | 110.96 | 1.11 | 84.95 | 81.63 | 1.04 |
| flux mlp-up 1024×3072×12288 | 139.88 | 128.92 | 1.09 | 125.57 | 108.83 | 1.15 |
| flux mlp-down 1024×12288×3072 | 138.98 | 137.67 | 1.01 | 111.26 | 113.67 | 0.98 |
| sd3 attn 1101×1536×1536 | 7.80 | 7.81 | 1.00 | 8.56 | 7.68 | 1.11 |
| sd3 mlp-up 1101×1536×6144 | 34.62 | 29.46 | 1.18 | 36.90 | 30.30 | 1.22 |
| zimage qkv 1024×3840×3840 | 71.78 | 60.61 | 1.18 | 44.85 | 44.19 | 1.01 |
| f5 attn 800×1024×1024 | 2.93 | 2.48 | 1.18 | 3.42 | 2.78 | 1.23 |
| f5 ff-up 800×1024×2048 | 8.95 | 8.51 | 1.05 | 5.99 | 5.28 | 1.14 |
| t5 ff-up 128×4096×10240 | 24.37 | 20.02 | 1.22 | 24.10 | 19.76 | 1.22 |
| clip mlp 77×768×3072 | 1.27 | 0.71 | 1.79 | 1.03 | 0.75 | 1.38 |
| vae conv 4096×4608×512 | 31.67 | 41.61 | 0.76 | 28.25 | 27.39 | 1.03 |
| vae conv 16384×1152×128 | 14.89 | 8.14 | 1.83 | 10.94 | 7.94 | 1.38 |
| narrow conv_out 16384×1152×3 | 8.27 | 2.04 | 4.05 | 8.21 | 2.08 | 3.94 |
| narrow 4096×512×16 | 0.27 | 0.18 | 1.48 | 0.33 | 0.21 | 1.54 |
| skinny 4×3072×3072 | 1.64 | 0.96 | 1.71 | 1.70 | 0.99 | 1.71 |
| small 16×1024×4096 | 0.90 | 0.57 | 1.56 | 1.12 | 0.62 | 1.81 |

ratio = OpenBLAS ms / CpuSgemm ms (>1: CpuSgemm faster). Max relative difference between the two:
1e-5..7e-5 (FP32 summation order).

So CpuSgemm defaults to the logical cores (`Environment.ProcessorCount`). `STINGRAY_CPU_THREADS`
still caps it. GEMM is compute-bound and SMT helps it; the memory-bound matvecs stay on physical cores.

## Result 2: real FLUX VAE decode

| Arm | decode times (s): round 1 / round 2 | warm mean |
|---|---|---:|
| OpenBLAS (8a096e36) | 13.99, 12.75, 11.81 / 13.85, 12.08, 11.88 | 12.13 s |
| CpuSgemm | 12.75, 11.28, 11.19 / 12.93, 11.27, 11.04 | 11.20 s |

7.7% less wall time end to end (the decode also does norms, im2col and upsampling). Output: max
abs difference 1.7e-6, PSNR 141.6 dB; 10 of 786,432 8-bit values differ, each by one level.

## Tests

`tests/OpenTail.Stingray.Tests.ForwardPass.Fast/CpuSgemmTests.cs` (176 cases on this CPU): every
supported kernel against a double-precision reference, all four transpose layouts, 20 shapes (skinny,
narrow dot, serial and parallel packed tiles, edge tiles, K tails, split-K narrow path). Also
alpha/beta semantics, NaN-in-C with beta = 0, Inf through the narrow-dot tail mask, arbitrary strides,
batched, reuse of the per-thread buffers, and run-to-run bitwise determinism.

## Not measured

- AVX-512 kernels (8×32, 8×24): no AVX-512 here; they are covered by the tests only where supported.
- Other CPUs. TensorSharp tuned the blocking on an i7-11800H; the sweep above found the defaults fine on Zen 3.
