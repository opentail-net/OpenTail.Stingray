# Diffusion samples to finish, and the Qwen Image performance plan (2026-10-10)

Two goals: finish the apple gallery in `docs/diffusion-samples/`, and find out why Qwen Image is so slow. This
machine has an integrated Radeon only, so every timing here is an iGPU timing (CLAUDE.md rule 13). Rules 7, 11
and 12 apply: measure before and after, flag any performance-pass cost, and a timing is only evidence if the log
shows what really ran. One model run at a time, always.

## A. Remaining images (serial, lightest first, single frame, GPU)

Prompt `a red apple on a wooden table`, seed 42, `--device 0`. Each result records the backend actually used (read
from the log), the time, and what the image shows. Already done: SD1.5, SDXL-Turbo, SDXL base (1024), FLUX.1-schnell,
Z-Image-Turbo, SD3.5 Medium, Wan 2.1 (one frame, CPU), LTX-Video (one frame, abstract output).

| Order | Item | Notes |
|---|---|---|
| 1 | Qwen Image Q3_K_S, 512x512 | Section B. First attempt hit a 20-minute cap on the CPU path (no `--device 0`). |
| 2 | Wan 2.2 14B, one frame, 256x256 | Heavy. |
| 3 | HunyuanVideo 720 fp8, one frame, 256x256 | Heavy. |
| 4 | FLUX.2-dev + Mistral-Small-3.2-24B, 512x512 | Heaviest. Never run before, and its CLI route is new. |

Dropped, with the reason stated in the gallery README: video clips (a 33-frame Wan 2.1 attempt ran 2h37m with no
output). SDXL base Q4_0 GGUF is on hold: it uses sd.cpp-style tensor names (`input_blocks.…`) that our loader does
not read.

## B. Qwen Image performance

### B0. Make the workload real (done in this change)
- The CLI never dispatched Qwen Image (`IsQwenImage` was unused) and never passed a text encoder to
  `QwenImagePipeline.Load`. Fixed: dispatch added, `--qwen-encoder` now selects the overload that builds the real
  Qwen2.5-VL encoder. Without it the pipeline silently falls back to non-text conditioning, which is not a valid
  baseline. Open item: make that fallback loud (a warning) instead of silent.
- Confirm from the log that the run says "on Vulkan GPU", not "on CPU".

### B1. Baseline (before any optimisation)
1. 512x512, **2 steps**, CFG 2.5, seed 42, `--device 0`: split load / text encode / first forward / later forward /
   VAE decode.
2. Same at 256x256 for resolution scaling.
3. Repeat the 512x512 2-step case three or more times without extra instrumentation for a stable end-to-end figure.
4. The full 512x512, 20-step run, once, with a long cap. If it times out, record a timeout, not an invented time.
   Known context: `PerformanceLeague.md` has Qwen Image at 256x256, 8 steps, CFG 4: Vulkan about 160s, CPU about
   277s. That is not a baseline for 512x512/20 steps.

### B2. Instrument (only what the baseline cannot answer)
Stage timers in the Qwen Image CLI route: model load, text encode, per-step conditional and unconditional forward,
VAE, PNG write; Vulkan submit/wait counts and time; peak working set. Treat `submitWait` as submission plus GPU
plus fence wait, not pure GPU time. Vulkan timestamp queries are the right tool for a per-kernel GPU split if the
stage timers say the GPU is the problem.

### B3. Candidate optimisations (one at a time, each A/B'd against B1)
Ranked by what to investigate first. Each is a hypothesis to verify in the code before it is attempted.
1. **Workspace and RoPE cache churn.** `QwenImageModel` keeps a single resident GPU workspace and RoPE cache keyed
   by image and text token counts; conditional and unconditional contexts can differ in length, so they may evict
   each other every forward. Verify by logging workspace creations and cache hits.
2. **Submission count.** If the GPU forward begins and ends a batch per transformer block, that is about 61
   submit-and-wait operations per forward and about 2,400 per 20-step CFG run. Test a single batch across the
   resident block chain, keeping the barriers.
3. **Work shared between the CFG branches.** Timestep modulation outputs and the image input projection are
   identical for both branches at a given step; cache them.
4. **True batched CFG.** Only after the above, prototyped on one block against two separate forwards. High risk
   (joint attention, differing text lengths).
5. **Kernel tuning (Q3_K GEMM, fused attention).** Only if profiling names them as the bottleneck. Quantised GPU
   weights and a tiled attention path already exist; do not regress them.

Out of scope for engine work, but possible operating-point changes: fewer steps or a distilled LoRA (needs
compatible LoRA support), text-embedding caching across runs, lower-resolution presets.

### B4. A/B protocol (every change)
Fixed model files, prompt, seed, steps, CFG and resolution; verify the device and that all blocks are resident.
At least five alternating A/B/A/B runs on the short workload; compare medians and spread, not the best run; then
the full target for finalists. Re-run the existing Qwen Image numerical check against `sd-cli` (same noise and
settings): a speedup that changes the result beyond tolerance is not a win. Keep a change only if it is measurably
faster; revert otherwise. Write measured numbers into `docs/PerformanceLeague.md`-style records, dated.

### B5. What iGPU timings cannot prove
A GPU path losing to or beating the CPU here describes this APU and driver, not a discrete GPU. A roughly 16 GB
placement budget is shared system memory, not VRAM.

## Status
- 2026-10-10: plan written. B0 code change built and under test.

## Measured results (2026-10-10, Ryzen 5700G iGPU, Vulkan, Qwen Image Q3_K_S, 60/60 blocks resident)

Workload: 512x512, real Qwen2.5-VL encoder, seed 42. Reference: stable-diffusion.cpp `sd-cli --backend vulkan0`,
same files and settings. `STINGRAY_QWEN_PROFILE=1` (stage log) and `=ops` (per-group split) are new, opt-in.

| Stage (512x512, 2 steps, CFG 2.5 = 4 forwards) | Stingray | sd-cli (Vulkan) |
|---|---|---|
| Text encode (cond + uncond) | 2.0 s | 2.9 s |
| One DiT forward | about 24.4 s (407 ms/block) | about 20.6 s |
| First-forward weight upload | +5.4 s | lazy, 2.1 s |
| VAE decode | 9.0 s | 9.5 s |
| Total | 115.4 s (an earlier uninstrumented run: 138.2 s) | 94.7 s |

Also: 256x256, 8 steps, CFG 4: 151.3 s (recorded 159.4 / 161.1 s on 2026-09-24: no regression); 512x512 20 steps would
be 40 forwards, about 16 minutes of DiT time at the current rate.

Where the forward time goes (sum over 60 blocks, 1 forward, `=ops` split, perturbed by extra syncs): MLP up/down +
norm 13.7 s (57%), norm + QKV GEMMs 4.5 s (19%), out-proj + gate 2.0 s (8%), QK-norm + concat + RoPE 1.9 s (8%),
attention 1.5 s (6%), modulation 0.2 s (1%). The forward is GEMM-bound; effective GEMM throughput is about 700-1000
GFLOP/s, roughly a third of the iGPU's FP32 peak, and the C++ reference sits at about the same level.

Hypotheses checked against the data:
- **Workspace / RoPE cache churn: rejected.** The workspace is rebuilt every forward (cond and uncond text lengths
  differ: 12 vs 5 tokens) but costs about 40 ms of a 24,000 ms forward.
- **Per-block submit batching: not the cost.** Everything outside the 60 blocks is under 100 ms per forward.
- **Raster order of the quantized GEMM (so concurrent workgroups share the A slab): no effect.** Reverted.
- **Push-constant split-K inside the quantized GEMM: rejected.** It made MLP-down slower (125 ms vs 96-116 ms) and
  perturbed the other shapes. Reverted.
- **Real, repeatable finding.** Microbenchmark (M=1024, Q3_K weights): throughput falls with the activation's row
  stride, not with FLOPs: 971 GFLOP/s at K=3072 (N=3072), 927 at K=6144, 668 at K=12288. The same K=12288 product run as
  4 chunks of K=3072 in separate contiguous buffers took 77.6 ms instead of 112 ms (-31%). MLP-down (K=12288) is the
  affected shape; 58 of 60 blocks are Q3_K. Estimated win if the MLP hidden activation were kept as four contiguous
  3072-wide chunks: about 27-35 ms/block, 1.6-2.0 s per forward (6-8%), before the cost of summing the four partial
  outputs. Not implemented: it needs the MLP weights re-laid out into chunks at upload and a partial-sum step, for a
  gain that still leaves the forward above the reference.
- **Remaining gap to the reference (about 18% per forward)** is probably the GEMM kernel itself: ggml-vulkan's matmul
  uses FP16 shared-memory tiles, ours uses FP32. A redesign is the only lever with a large upside, and it is a kernel
  project with its own numerics decision (rule: approved numerics changes are documented everywhere).
