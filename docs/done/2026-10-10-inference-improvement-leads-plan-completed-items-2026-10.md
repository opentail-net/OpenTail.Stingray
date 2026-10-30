# Completed items split from 2026-10-10-inference-improvement-leads-plan.md (2026-10-30)

Finished items moved verbatim out of [../4-performance/2026-10-10-inference-improvement-leads-plan.md](../4-performance/2026-10-10-inference-improvement-leads-plan.md). Open items stay there. Each block notes the section it came from.

> from: ### A. Decode on Vulkan (new, top of the list: the attention win generalises)
- [x] **A0** Diagnose decode: elimination experiment plus the ggml per-op reference showed about 68 ms of a 159 ms token was single-token attention at about 500 context tokens (Qwen3-4B). Evidence log: "Lead 2b".

> from: ### A. Decode on Vulkan (new, top of the list: the attention win generalises)
- [x] **A1** GQA-group split-KV decode attention kernel (fp32 KV, head dim 64/128, 1-4 query heads per KV head), default on, `STINGRAY_VULKAN_DECODE_ATTN_GQA=0` rolls back. Kernel 11.8x faster (132 vs 1,566 us/layer, ggml about 140); Qwen3-4B decode after a 493-token prompt 6.3 to 12.1 t/s (+88%), text identical. Commit `2624089d`.

> from: ### A. Decode on Vulkan (new, top of the list: the attention win generalises)
- [x] **A2** GQA kernel at every context length (`GpuForwardPass.UseSplitKvDecodeFp32`): short-prompt decode (Qwen3-4B, 96 tokens) **10.5-10.6 -> 12.6-12.7 t/s (+20%)**, text identical in both A/B rounds (estimated +10%; attention was a bigger share than extrapolated). Commit in the same change as A3.

> from: ### A. Decode on Vulkan (new, top of the list: the attention win generalises)
- [x] **A3** Head dimension 256 (Gemma family) accepted by `SupportsSplitKvGqa`; parity cases added (16/4/256 group 4, 8/8/256 group 1), 8 of 8 tests pass. Not yet run end to end on a Gemma model.

> from: ### A. Decode on Vulkan (new, top of the list: the attention win generalises)
- [x] **A4** Remaining decode attention call sites: the fp32 path inside `GpuForwardPass` (both sites) and the hybrid Qwen3.5/3.6 engine (`VulkanHybridGdnForwardPass.AttentionAuto`, trunk and MTP head, partial buffers allocated when the shape is supported). Hybrid Qwen3.5-4B (16 heads / 4 KV heads, head dim 256, attention every 4th layer), 493-token prompt, alternating A/B: **decode 9.5 -> 10.9 t/s (+15%)**, text identical. Left: the Gemma-4 per-layer path (`ForwardGemma4`) and the layer-split engine, see A4b.

> from: ### A. Decode on Vulkan (new, top of the list: the attention win generalises)
- [x] **A4b** Gemma-4 per-layer attention (fp32 KV) now uses the GQA split-KV kernel (head dim 256 SWA layers and 512 global layers; `SupportsSplitKvGqa` extended to 512; partial buffers sized by `_maxHeadDim`; `STINGRAY_VULKAN_DECODE_ATTN_GQA=0` rolls back). Gemma-4 E4B Q4_K_M after a 1,953-token prompt, 2 alternating rounds: **decode 5.4/5.4 -> 9.5/9.6 t/s (1.77x)**, its per-token prefill 6.6 -> 9.7 t/s, text identical. Kernel parity cases added for 256 and 512. Caveat: the repo Gemma-4 Vulkan test classes all skipped here (fixtures absent), so the A/B text plus the kernel parity test are the evidence. Gemma-4 prefill is still the per-token path (6-10 t/s at 2k): see new item B8.

> from: ### A. Decode on Vulkan (new, top of the list: the attention win generalises)
- [x] **A5** GQA groups of 5-8: groups above 4 use ceil(G/4) workgroups per KV head (`SupportsSplitKvGqa` now accepts 1-8). 13 of 13 parity cases pass (groups 5, 6, 7, 8; head dim 256 with group 8). End to end, Qwen2.5-VL-7B (28 heads / 4 KV heads, group 7), 493-token prompt, alternating A/B: **decode 6.5 -> 7.5 t/s (+15%)**, text identical; its per-token prefill 7.0 -> 7.6 t/s.

> from: ### A. Decode on Vulkan (new, top of the list: the attention win generalises)
- [x] **A7** gpt-oss attention sinks on the GQA split-KV path (sink-aware combine kernel `AttentionSplitKvCombineSinks`: the sink logit joins the max and the denominator; group 8 served by two workgroups per KV head). gpt-oss-20b-MXFP4 after a 1,876-token prompt, 2 alternating rounds: **decode 4.8/4.9 -> 7.8/8.1 t/s (1.65x)**, per-token prefill 6.0 -> 7.8-7.9 t/s, text identical. Parity test vs the sinks kernel (incl. 128 SWA window), GptOssGpuParityTests and shader-drift pass. `STINGRAY_VULKAN_DECODE_ATTN_GQA=0` rolls back.

> from: ### A. Decode on Vulkan (new, top of the list: the attention win generalises)
- [x] **A8** Verify on more models and contexts. Done: Qwen3-4B at about 1,900 tokens of context, alternating A/B: **decode 3.3 -> 10.9-11.0 t/s (3.3x)**, text identical (the gain grows with context, as predicted). Done 2026-10-11: SmolLM2-1.7B (head dim 64, MHA) after a 2,126-token prompt, 2 alternating rounds: **decode 8.3/8.2 -> 17.2/16.7 t/s (2.0x)**, text identical, prefill unchanged (85 t/s). Done 2026-10-11: Qwen3-8B-Q4_K_M after a 2,896-token prompt (SnapKV auto-on, budget 2048), 2 alternating rounds: **decode 2.2/2.2 -> 6.3/5.5 t/s (2.5-2.9x)**, text identical, prefill unchanged (40 t/s). Done 2026-10-11: Gemma-3-4B-Q4_K_M after a 2,885-token prompt: **decode 5.5 -> 10.9/11.2 t/s (2.0x)**, prefill 11.7 -> 37 t/s (the old path also slows prefill); the old-kernel round 2 hit `VK_ERROR_DEVICE_LOST` (baseline path only, one of two runs; the GQA path ran clean twice); text compared only by the first 140 chars. Done 2026-10-11: Qwen3-4B after a 4,324-token prompt, 2 alternating rounds: **decode 1.8/1.8 -> 9.0/9.3 t/s (5x)**, prefill unchanged (54.6 t/s); text not compared this run. A8 complete.

> from: ### B. Prefill on Vulkan (Lead 2)
- [x] **B0** Diagnose: the batched prefill runs at about 0.3 TFLOP/s (multi-token mat-vec, 16 tokens per dispatch); a larger chunk silently falls back to a slow attention kernel. Evidence log: "Lead 2".

> from: ### B. Prefill on Vulkan (Lead 2)
- [x] **B1** Opt-in Path 3 (`STINGRAY_VULKAN_MM_PATH=3`): tiled quantized GEMM incl. a new Q6_K decoder, `SgemmQuant`, and flash attention run in 16-query slices. Qwen3-4B prefill 38.4 to 71.3 t/s (+86%), text identical. Commit `eef12b83`.

> from: ### B. Prefill on Vulkan (Lead 2)
- [x] **B2** Path 3 vs Path 1 as an alternating A/B, two rounds (Qwen3-4B-Q4_K_M, Vulkan): 493-token prompt **38.3 / 38.4 -> 72.4 / 71.8 t/s (+88%)**; 1,915-token prompt **27.4 / 27.4 -> 41.2 / 41.4 t/s (+51%)**; decode unchanged (12.0-12.4 and 10.9 t/s in both paths).

> from: ### B. Prefill on Vulkan (Lead 2)
- [x] **B3** Path 3 chunk 128 to 256 (default; `STINGRAY_VK_PATH3_CHUNK` overrides, 16-1024). Qwen3-4B 493-token prompt, 2 alternating rounds: 128 = 71.3/71.5, 256 = 77.4/78.0 (+8.6%), 512 = 81.7/81.5 (+14%, single chunk; not made default: scratch grows, unmeasured on large models). Text identical; chunk-split 9/9, batched-prefill 3/3, spec-verify 18/18 at 256.

> from: ### B. Prefill on Vulkan (Lead 2)
- [x] **B3b** Causal GQA register-tiled prefill attention (`FlashPrefillGqa128`, head dim 128, fp32 KV; `STINGRAY_VULKAN_FLASH_PREFILL=0` rolls back): a whole chunk runs in one dispatch (64x64 tile, 4x4 per thread) instead of 16-query slices that each re-read K/V. Qwen3-4B 1,915-token prompt, 2 alternating rounds: prefill **43.0/43.1 -> 97.9/99.5 t/s (2.3x)**, text identical, decode unchanged. Parity test `VulkanFlashPrefillGqaTests` (8 cases vs the sliced kernel), shader-drift, batched-prefill, chunk-split, spec-verify, arch-parity (18) and Qwen3-VL suites pass. Head dim 64 added (`FlashPrefillGqa64`): SmolLM2-1.7B 2,126-token prompt prefill 84.5/84.2 -> 153.5/153.3 t/s (1.8x), text identical, arch-parity 18/18. Open: other head dims, bf16/q8 KV and SWA still use the old slices.

> from: ### B. Prefill on Vulkan (Lead 2)
- [x] **B4** Second dense model and decode check with Path 3 on (2026-10-11): Mistral-7B-Q4_K_M prefill 22.7 to 46.6 t/s (tokens per dispatch 16 to 115), SmolLM2-1.7B 81.7 to 180.2 t/s, Qwen3-4B 38 to 72; decode unchanged (7.8 and 24.5 to 24.8 t/s), greedy text identical.

> from: ### B. Prefill on Vulkan (Lead 2)
- [x] **B5** Path 3 is now the default (2026-10-11; `STINGRAY_VULKAN_MM_PATH=1` rolls back). Evidence: arch-parity 18/18, chunk-split 9/9, spec batch-verify 18/18, layer-split 13/13, hybrid-OLMoE 7/7, Qwen3-VL M-RoPE 2/2, KV dtype 4/4, flash 14/14, SnapKV, batched-prefill 3/3 (Path 3 tolerance widened to 2e-2 like Path 2: logit delta 9.2e-3, argmax identical). DeepSeek2 CPU-parity fails identically on Path 1 and 3 (pre-existing, separate). Still open: Q3_K/Q5_K weights in the batched trunk.

> from: ### B. Prefill on Vulkan (Lead 2)
- [x] **B5b** (found by the 2026-10-11 all-checkpoint recheck) K-quants other than Q4_K/Q6_K stay off the batched trunk: SmolLM2-1.7B Q5_K_M prefills at 22.3 tok/s on Vulkan at 555 tokens against 235 tok/s for the Q4_K_M file, because `ComputeCanBatchedTrunk` accepts only Q4_K/Q6_K weights while Path 3 already serves Q3_K/Q5_K. Fix: allow Q3_K/Q5_K in `IsBatchable` when Path 3 is active (MatMulBatched already routes them); then re-run the arch-parity suite. FIXED 2026-10-11: `IsBatchable` accepts Q3_K/Q5_K when Path 3 is active. SmolLM2-1.7B Q5_K_M, 554-token prompt: **prefill 21.9 -> 219.7 tok/s (10x)**, greedy text identical to the per-token path; arch-parity suite 19/19 in 192 s incl. a new Q5_K_M case (worst cosine 0.9987, 0/9 argmax flips).

> from: ### B. Prefill on Vulkan (Lead 2)
- [x] **B8** Gemma-4 prefill via the CPU-prefill KV handoff (2026-10-11; not a GPU batched trunk). Gemma-4 E4B Q4_K_M, 495-token prompt: **prefill 9.8 -> 56.0 t/s (5.7x)** (CPU batched pass 59 t/s + 62 ms KV upload), decode 9.3-9.5 unchanged, greedy text identical. The handoff now uploads per-layer head dims (256/512) and skips the aliased shared-KV tail layers; `Gemma4CpuPrefillHandoffTests` (default and F32 KV, 24 teacher-forced steps vs the sequential GPU prefill: cosine 0.9995-0.9999, prefill logits 0.9988) is the receipt, family `gemma4` registered for the full-GPU path. Regression check done 2026-10-11 after the `HandoffKv` refactor: `GpuCpuPrefillHandoffTests` 6/6 pass in 345 s on real weights (OLMoE, Qwen1.5-MoE, Phi-3.5-MoE; prefill cosine 0.9995-0.9999, startPos>0 continuation 0.9998). A batched GPU Gemma-4 trunk (for discrete GPUs) is still open.

> from: ### B. Prefill on Vulkan (Lead 2)
- [x] **B7** Batched trunk for M-RoPE models while no image is registered (`BatchedTrunkNow`; also requires sections to cover all pairs). Qwen2.5-VL-7B-Q4_K_M, 493-token text prompt: prefill 7.6 to 53.5 t/s (7x), greedy text identical to the per-token path (`STINGRAY_VULKAN_NO_BATCHED_PREFILL=1`), decode unchanged 7.4-7.5; Qwen3-VL M-RoPE parity tests 2/2 (images still take the per-token path).

> from: ### C. Mat-vec kernels (Lead 1)
- [x] **C1** MXFP4: one block per lane, packed-LUT decode. 14 to 25 GB/s; gpt-oss decode 6.7 to 7.5 t/s (+12%). Commit `c30e455a`.

> from: ### C. Mat-vec kernels (Lead 1)
- [x] **C2** Q8_0 (33 to 43 GB/s, ggml 45) and Q4_0 (+7%). Qwen3-0.6B-Q8_0 decode +7-10%. Commit `c7bd0f80`.

> from: ### D. Qwen Image (earlier thread; the plan for it lives in the other plan file)
- [x] **D1** Small-M quantized GEMM, **D2** chunked MLP: forward 24.5 s to 21.9 s (reference about 20.7 s). Rejected and logged: raster order, split-K, 128x128 tile, unrolled loop.

> from: ### E. Other leads and deferred work
- [x] **E1** Hybrid-GDN (Qwen3.5/3.6) batched prefill, re-measured and fixed 2026-10-11. Cause: the batched trunk was capped at 8 tokens per chunk and ran the dense FFN per row (weights re-streamed per token). Now: chunk = min(128, active matmul path width) (`STINGRAY_VULKAN_HYBRID_PREFILL_CHUNK`), dense-GPU FFN batched across the chunk, and `MatMulBatched` falls back instead of throwing for dtypes Path 3 does not serve. 480-token prompt: **Qwen3.5-4B 11.0 -> 46.1 t/s (4.2x)** (chunk 8: 17.6, 64: 33.6, 128: 46.1, 256: 48.2, 512: 50.8), **Qwen3.5-9B 6.4 -> 35.5 t/s (5.5x)**, decode unchanged, greedy text identical to the sequential path on both. Follow-up: head-dim-256 causal prefill attention kernel (`FlashPrefillGqa256`) used by the hybrid attention layers: 45.4 -> 53.2 t/s on Qwen3.5-4B, text identical; elimination experiments (skip one stage, 480 tokens, 10.4 s baseline): scan costs 2.2 s, per-query attention cost 2.9 s, conv 0; the opt-in chunked scan gives no gain (46.0 vs 46.1). Open: the recurrence scan (about 2 s); MoE hybrid FFN is still per row; the CUDA-oracle hybrid tests cannot run here.

