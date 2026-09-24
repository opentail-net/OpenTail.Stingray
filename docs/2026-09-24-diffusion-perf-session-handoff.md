# Diffusion correctness + performance session: handoff (2026-09-24)

This session covered Z-Image-Turbo, FLUX.1, SDXL-Turbo and FLUX.2. It ran on the usual dev box
(Ryzen 5700G, Radeon Vega 8 iGPU, shared RAM). Commits `21969b6` through `0958d6f` are on `main`
and not yet pushed. The measured numbers are also in `README.md` and `PerformanceLeague.md`
(the 2026-09-24 rows).

## 1. Status of the models touched

| Model | CPU | GPU (Vulkan iGPU) | Notes |
|---|---|---|---|
| Z-Image-Turbo 256² / 4 steps | ✅ 87.8–97.8s | ✅ 52.0–57.6s (GPU-resident chain re-enabled) | Mosaic was the timestep-embedding bug (below) |
| FLUX.1-schnell 512² / 4 steps | ✅ 178.8s (C++ ≈191.7s) | ✅ 238.9s | Both paths had been crashing |
| SDXL-Turbo 512² / 4 steps, CFG 0 | ✅ 42.4s | ✅ 35.8s, then ~4.1s/step after the new SGEMM | CLI now defaults Turbo to CFG 0 / 4 steps |
| FLUX.2-dev 128² / 4 steps | ✅ 73.6s | ✅ hybrid 134s, full GPU 161–268s | Not broken on GPU (checked visually) |
| FLUX.2-dev 512² / 2 steps | ✅ 258.8s | full 307.9–340s, hybrid 291–295s | GPU numbers from before `0958d6f` are noisy (see §4) |
| Wan2.1-T2V-1.3B 256² / 1f / 20 steps, CFG 6 | ✅ 74–77s generate (was ~190s; C++ CPU 256.6s total) | not re-measured | F32 packed SGEMM, see §3 |

## 2. Bugs found (all fixed and committed)

- **Z-Image mosaic**: `DiffusionOps.SinusoidalTimestepEmbedding` gained `flipSinToCos` with default
  `false`, which gives `[sin,cos]`. Every other caller passed `true`; Z-Image didn't. The restoration
  plan's other theories (WanAttention "conflation", int8 activations) were **measured and disproven**.
  Both were reverted to the faster options.
- **Z-Image GPU residency** had been disabled as a "FP16 precision" artifact. It was the same
  timestep bug. Re-enabled (`UseGpuResidency`).
- **FLUX.1 CPU crash** (since c1cc771, 2026-09-19): `FluxDiT` passed an unresolved tensor name to
  `QuantizedWeightCache`, whose loader only adds prefixes. Fixed by passing `ti.Name`.
- **FLUX.1 GPU OOM**: T5-XXL kept ~9.5GB of GPU weights plus a ~19GB FP32 host cache resident
  next to the 24GB FP16 DiT. The fix is `T5Encoder.ReleaseMemory()`, called after encoding. It
  also helps on CPU (memory pressure).
- **SDXL-Turbo image quality**: the CLI defaulted every SDXL model to CFG 7.5 and 20 steps. Turbo
  was running with the wrong settings (blotchy image, 2× UNet work).

## 3. Shared-kernel performance work (benefits every model using them)

| Change | Measured |
|---|---|
| `DiffusionOps.Conv2D`: im2col + GEMM, computed as `[outC, pixels]` | ~30× per 3×3 conv vs the old scalar loop, then another ~1.8× from the orientation swap |
| `DiffusionOps.Linear` (n>1) → `SimdKernels.MatMulBatchedF32` | 1.9–2.3× on T5/CLIP shapes (4096² unchanged) |
| `DiffusionOps.SiluInPlace` parallel/chunked | 1.23s → 0.11s across a 512² VAE decode |
| CPU VAE decode 512² | ~18s → ~10s (standalone) |
| Vulkan `SgemmF16` 128×256 tiles, vec4 LDS, 512 threads | 1.5–1.6× (598 → 916 GFLOP/s on FLUX.2 linear1) |
| Vulkan `SgemmSiluGateF16` on the same kernel | 525 → 325ms (1.6×) |
| FLUX.2 Mistral truncated to 30 layers (last tap is layer 29) | text encode 34.9s → 25.9s |
| `PackedSgemmF32` + F32 pack-once in `QuantizedWeightCache` (safetensors had no raw access, so every Linear re-read its weight) | Wan DiT 9.0 → 3.2s/step at 256². Applies to every safetensors DiT going through the cache (LTX-Video, HunyuanVideo, ...) |

Tried and rejected (not measurably better, reverted): a branch-free im2col interior path, register
prefetch in the SGEMM (VGPR pressure), 128×128 tiles (128×256 was better), full k-loop unroll
(12s driver compile per process), no unroll (2× slower). Also rejected: resident raw-Q4_K FLUX.2
single blocks, because the per-step cost was really a first-time disk read and gained nothing on
this machine.

New permanent tests: `DiffusionOpsConv2DTests`, `DiffusionOpsLinearTests`. Both are exact-match
checks against scalar or double-precision references.

## 4. Open items / where to pick up

1. **FLUX.2 GPU E2E after `0958d6f`: still unmeasured, come back to it.** The end-of-session
   runs are **invalid, so discard them**. The timing loop was accidentally launched twice (about
   13s apart), so two test processes ran concurrently and fought over CPU, iGPU and RAM (each held
   4–10GB). The results were full GPU 529–545s (vs the clean 308–340s above) and hybrid failing
   with `OutOfMemoryException` in every run. Both are artifacts of the doubled load, not a
   regression. The stray process was killed and not re-run. Next time, run **one** loop at a time,
   and before starting, check that no other `OpenTail.Stingray.Tests.Diffusion.exe` is running.
   Harness: the untracked `tests/OpenTail.Stingray.Tests.Diffusion/ZzFlux2ProfTmp.cs`. Env vars:
   `ZZ_RES=512`, `ZZ_STEPS=2`, `ZZ_SB=1` for full GPU (unset for hybrid), `ZZ_CPU=1` for CPU.
   Output goes to `docs/diffusion-samples/zz_flux2_prof.png`. Take ≥2 runs per mode; single runs
   varied by ±10% (page-cache/disk state after heavy runs). The earlier "full GPU got slower"
   reading was probably the ~12s-per-kernel driver compile, which `0958d6f` removed.
2. **FLUX.2 GPU remaining costs at 512²**:
   - Single-block streaming (~28s per 2 steps). It's really a first-time disk read of ~13GB.
   - The SGEMM is now ~900 GFLOP/s against ~2 TFLOP/s FP32 peak.
   - Next levers: an int8 dot-product (`integer_dot_product` is available on this GPU) quantized
     GEMM that keeps weights as Q4_K/Q8. That would cut DRAM traffic ~2–4× and lift the FP32 ALU
     ceiling. Big job. Also: `VulkanMatMulPathConfig` "Path 2" (tiled quantized GEMM) is an empty seam.
3. **Next models in the table**: (Wan CPU done 2026-09-24; `WanTests` has 4 pre-existing pack/unpack failures, identical on clean `main`, likely stale expectations) LTX-Video, Qwen Image (GPU broken), SD3/3.5
   (composition issue), HunyuanVideo (noise). The new shared kernels will change their CPU (and
   FP16-GEMM GPU) numbers, so re-measure before trusting old table figures.
4. **Safety net (strongly recommended)**: add a cheap per-pipeline E2E smoke test. Use a small
   resolution and step count with real weights, and check the latent stats and/or a perceptual
   hash against a stored reference. It must fail loudly when checkpoints are missing, and it
   should run before any commit to shared `DiffusionOps`, scheduler, loader or Vulkan GEMM code.
   Every regression found in this session (a timestep default flip, a name-resolution change,
   memory residency, a CLI default) would have been caught in minutes.
   `Flux2GpuWiredEndToEndTests` only asserts finite output, which is the silent-pass pattern.

## 5. Lessons worth keeping

- Measure before "fixing". Three plausible-sounding diagnoses this session (attention conflation,
  int8 activations, FP16 precision) were wrong. A latent-stats dump and a stage profile located
  each real bug in minutes.
- Shared helpers with changed defaults silently break the one caller that didn't opt in. Prefer
  required parameters for behavior flags like `flipSinToCos`.
- On this iGPU the FP16 SGEMM was DRAM-traffic bound, not ALU or LDS bound. With K in the
  thousands a tile's K-strip doesn't fit in L2, so traffic scales with (4/tileN + 2/tileM) and
  bigger tiles were the lever.
- Watch driver first-use compile time for heavily unrolled shaders. It's paid once per process
  and dominates short benchmark runs.

## 6. Four-model plan (started 2026-09-24, worked in this order)

Rules: get a measured C++ `sd-cli` reference at the same config first. Run one heavy process at a
time. Correctness before perf. Check images by eye. CPU first, then GPU. Scratch harnesses are
`Zz*ProfTmp.cs` (untracked).

- [x] **LTX-Video** (done): 311.6s → 133.3s at 512²/20 steps (DiT 2.2×, VAE 13×). No C++ reference
      possible (vendored sd-cli is LTX-2 only). The CPU-vs-Vulkan "quality gap" was a seed mix-up in the
      Vulkan test. Remaining: T5 ~22s (first-read + bf16 conversion), DiT ~5.3s per step.
- [x] **SD 3.5** (done, 🟢 CPU and GPU): composition bug = T5 wrongly masked for SD3 + OpenCLIP-bigG
      had 16×80 heads instead of 20×64. The image matches C++ with the same noise. CPU 284.8s → 82.5s.
- [ ] **Qwen Image** (🟢 CPU, slow; GPU broken): dequant-on-the-fly packed GEMM for K-quants
      (F32 activations), then GPU bisection vs CPU.
- [ ] **HunyuanVideo** (🟡 noise): confirm `sd-cli` runs our checkpoints coherently, then bisect
      block by block. Timeboxed; write down the blocker if it stalls.
