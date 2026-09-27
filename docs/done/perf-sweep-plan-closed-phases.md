# PerformanceLeague sweep: closed phases

> **ARCHIVED 2026-09-27.** Moved verbatim out of [../perf-sweep-plan.md](../4-performance/perf-sweep-plan.md).
> - **Phase 12** (FLUX.1 Vulkan speedup): answered. Per-stage profiling exists (2026-09-25 row in
>   `PerformanceLeague.md`: CLIP 0.6s, T5 15.8s on CPU, DiT 23.5s per step, VAE 7.3s), the GPU gap was
>   measured kernel by kernel against ggml's `test-backend-ops` (017765c), and FLUX.1 Vulkan went
>   173.6s -> 132.0s with register-tiled flash attention (a5daf08). The "near-zero speedup" no longer holds.
> - **Phase 14** (post-norm GPU layer split): `HybridForwardPass` gained optional pre-norm and
>   post-norms for EXAONE 4.5 (93332cf, 69141da), matching llama-server exactly at `-g 16`, with hybrid
>   decode 1.8-1.9 t/s and prefill 1.8 t/s measured on the iGPU (log in
>   [../101-work-queue-after-coverage-plan.md](101-work-queue-after-coverage-plan.md)).

## Phase 12 — FLUX.1-schnell's near-zero Vulkan speedup (~3%, unexplained, flagged in the doc itself)

`PerformanceLeague.md`'s own FLUX.1-schnell row explicitly flags this as unresolved: "not yet
known whether the bottleneck is the T5-XXL encoder... or the DiT body itself" — a real, named
open question on a flagship diffusion model, not something this plan is inventing.

- [x] 12.1 Added real per-stage `Stopwatch` timing to `ImagePipeline.Generate`
      (`src/OpenTail.Stingray.Diffusion/ImagePipeline.cs`) — CLIP-L encode / T5-XXL encode /
      noise+pos-id setup / DiT denoise loop / VAE decode, gated behind the same
      `STINGRAY_PROFILE_DECODE=1` env var used for ACE-Step's identical Phase 9 instrumentation
      (reused, not duplicated). Builds clean.
- [ ] 12.2 Only `models/flux1-schnell/tokenizer_t5/tokenizer.json` was present locally (no
      DiT/CLIP-L/T5-XXL/VAE weights) — downloading the full ~13GB checkpoint now (68GB free,
      user-confirmed OK to use) to actually run the profiling.
- [ ] 12.3 Run the profiled CPU and Vulkan generations, answer whether T5-XXL or the DiT
      dominates (and specifically whether T5-XXL ignores the backend flag the way several
      ONNX-based TTS rows elsewhere in this doc do), implement + verify (real on-prompt image
      content check, not just non-degeneracy — this pipeline has a known separate
      background-tiling artifact already, re-verify any fix doesn't touch or worsen that) +
      re-benchmark + record.

## Phase 14 — Post-norm architecture GPU-layer-split gap (`HybridForwardPass`, blocks EXAONE-4.5/OLMo2-style models from GPU offload entirely)

Real, infra-level perf gap, not a one-model bug: `HybridForwardPass.cs` (the CPU+GPU layer-split
path) hardcodes pre-norm tensor names (`attn_norm.weight`/`ffn_norm.weight`) with no fallback to
post-norm-only architectures' real tensor names (`post_attention_norm.weight`/`post_ffw_norm.weight`),
unlike plain `ForwardPass.cs` which already has this fallback. This forces EXAONE-4.5-33B (and any
other post-norm-only architecture) onto CPU-only (`-g 0`), leaving real GPU-offload throughput
entirely unmeasured and unavailable for this whole architecture family.

- [ ] 14.1 Mirror `ForwardPass.cs`'s existing `FindTensor(...) is not null` fallback + post-norm
      forward math (norm applied after attn/ffn, before the residual add) into
      `HybridForwardPass.cs` — real architectural work per the doc's own assessment, not a
      one-line tensor-name swap.
- [ ] 14.2 Verify correctness first on a model this codebase already handles correctly via plain
      `ForwardPass` (confirm `HybridForwardPass`'s new post-norm path produces identical logits to
      the working CPU-only path) before trusting any new GPU-offload throughput number.
- [ ] 14.3 Benchmark EXAONE-4.5-33B with real GPU layer-split enabled (currently impossible) vs
      the existing CPU-only 1.6/1.7 t/s baseline — record the real speedup this unlocks.

