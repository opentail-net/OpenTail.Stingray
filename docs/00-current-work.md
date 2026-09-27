# Current work

The open engineering backlog, in priority order. **Open items only.** This file is not a log:
when an item closes, delete its line here, record the evidence in the model's
[STATUS.md](STATUS.md) row (or the detail doc), and move any finished detail doc to [done](done).
Working notes, measurements and histories go in the item's detail doc or in `done/`, never here.

The previous version of this file (a working log up to 2026-09-27, including every fixed bug and
its history) is kept verbatim at
[done/00-current-work-log-to-2026-09-27.md](done/00-current-work-log-to-2026-09-27.md). Older
closed sections: [done/00-current-work-closed-2026-09.md](done/00-current-work-closed-2026-09.md),
[done/00-current-work-standing-state-2026-08.md](done/00-current-work-standing-state-2026-08.md).

## How the list is ordered

The goal is still **run any GGUF from Hugging Face**, but the order below puts one thing ahead of
new coverage: a model we already list as supported that gives a wrong answer is worse than a model
we do not list, because users trust the STATUS matrix. So:

1. **Correctness** of what is already claimed ([1-correctness/](1-correctness)).
2. **Model coverage**: new families and half-built ones ([2-coverage/](2-coverage)).
3. **Product and runtime**: the first-run experience, serving, sessions ([3-product-and-runtime/](3-product-and-runtime)).
4. **Performance**, last locally; measure first ([4-performance/](4-performance)).
9. **Needs hardware this machine does not have** ([9-external-hardware/](9-external-hardware)).

Within each section the first items have the biggest effect for the least work. Every fix follows
CLAUDE.md: real weights, an independent reference, timed test runs (rule 12), and the STATUS row
updated with dated evidence in the same pass.

---

## 1. Correctness of what is already claimed

1. **HunyuanVideo numeric verification** against `sd-cli` latents: the last diffusion row still at
   ⚪ confidence. Same method as SD3.5 / FLUX.2 / Qwen Image (`DiffusionParityHooks`, injected
   noise, patched `sd-cli` dumps); includes the RoPE pairing-convention check from
   [done/092](done/092-diffusion-rope-pairing-convention-audit.md). Method and prior results: #13 in
   [done/102](done/102-status-open-items-plan.md).
2. **Vision features that exist only on the CPU `ForwardPass`.** 2D M-RoPE image positions
   (PaddleOCR-VL, Qwen2.5-VL, …) and Granite 4.0 Vision deepstack are not applied by the CUDA and
   Vulkan passes, nor by the server's image path (`InferenceEngine` handles Gemma-style placeholders
   only). Today those models give correct answers from the CLI on CPU only. From #6 and #9 in
   [done/102](done/102-status-open-items-plan.md).
3. **Per-pipeline diffusion end-to-end smoke tests**: real weights, small resolution, a stored
   reference, and a loud failure when the checkpoint is missing. Item 4 of §4 in
   [done/2026-09-24-diffusion-perf-session-handoff.md](done/2026-09-24-diffusion-perf-session-handoff.md).
4. **Silent-no-op test sweep** (CLAUDE.md rule 12). Several real-weight tests search only
   `models/`, not `models/_models/`, and "pass" in 0.1 s without loading anything (Parakeet,
   Orpheus/SNAC and Z-Image were found by accident). Sweep every model-search helper against the
   real `models/_models/` contents; better, make a missing checkpoint skip visibly.
5. **`HybridGdnChunkedPrefill_MatchesSequentialPrefill` fails** (found 2026-09-25, not re-checked).
   See Phase 8 of [done/2026-09-25-hf-top-downloads-coverage-plan.md](done/2026-09-25-hf-top-downloads-coverage-plan.md).
6. **NaN in `ForwardPass`'s f16 `qwen3` path** (last layer, one position; Q8_0 is fine). Not root-caused.
7. **CPU greedy-decode non-determinism**: two sightings under CPU contention, neither reproduced.
8. **Jinja chat-template gaps** (string concatenation inside a conditional) on Gemma-3-4B-it and
   Qwen3.8-27B, logged as warnings. Parenthesised ternaries (65e0ff1) and dict literals
   (2026-09-12) were fixed since; re-run both templates and check whether the warnings remain.
9. **Stable Audio 3 padding masks**: the reference masks padded tokens inside the APG norm at
   CFG > 1 (our CFG-7 final latent is 0.9978, not 0.999+), and neither DiT accepts an attention
   mask over the 6 s padding region. #11 in [done/102](done/102-status-open-items-plan.md).
10. **Youtu-VL text**: one 1024-token wikitext window is +5% PPL vs llama.cpp (others −2.1% to
    +0.8%). Diff per-token log-probs over wiki.test.raw [1024,2048) against `llama-server`.
    #5 in [done/102](done/102-status-open-items-plan.md).
11. **FunASR-Nano** has only ever been tested on a synthetic tone (repetitive output, expected);
    run it on real speech and give it a STATUS row. Separately, the local `paraformer-q8.gguf` lacks
    `pf.vocab` (bad conversion); the ONNX Paraformer path works.
12. **Classic LLaVA-1.5** (plain 32000-token vocab, no image token): the CLI assumes a tokenizable
    placeholder, so image splicing fails. Needs a direct-splice path like `llava-cli`. No local
    LLaVA-1.5/1.6 checkpoint at the moment, so the `llava_uhd` changes are unverified for LLaVA too.
13. **CosyVoice 2 garbled endings — BLOCKED** on an independent reference: one recorded upstream
    `inference_zero_shot` run (input token ids, generated speech tokens, ideally per-step top-k),
    checked in as data. What is already ruled out: #10 in [done/102](done/102-status-open-items-plan.md).
14. **Small, known leftovers**: [1-correctness/bugstofix.md](1-correctness/bugstofix.md)
    (`SpeculativeDecoder` StepSampled/PLD latent defect, dtype/op drift notes); Apertus greedy
    re-check vs `llama-server --no-jinja`; the stale Gemma 4 prefill comment at `ForwardPass.cs`
    ~line 1350 (batched prefill has supported per-layer head dims since 2026-09-16).

Not fixable on this machine (kept 🔴 in STATUS): MiMo-VL (upstream mmproj projects to 3584, the
text model wants 4096), Llama 4 vision (93 GB), MobileNetV5 (no checkpoint declares the projector).

## 2. Model coverage

1. **Mamba-2 hybrid layer + state cache**, admitting IBM Granite 4.0 (`granitehybrid`, Apache-2.0,
   1B-32B) first; Falcon-H1 comes almost free. One layer type unlocks item 2 too. #14 in
   [done/102](done/102-status-open-items-plan.md).
2. **`nemotron_h`**: Nemotron Nano v2 / Nemotron 3 Nano, and the text backbone of
   Nemotron-Nano-12B-v2-VL (vision encoder already builds). Reuses item 1. #3/#15 in
   [done/102](done/102-status-open-items-plan.md).
3. **GLM-4.5 / 4.6 / 4.7 incl. Air (`glm4moe`)**: a top open family; GLM-4 dense already runs.
4. **Liquid LFM2 / LFM2-MoE (`lfm2`)**: decide the licence question first (free commercial use is
   capped at $10M revenue).
5. **Qwen3-VL**: needs IMROPE plus the `qwen3vl` text architecture (not admitted).
6. **Gemma 4 E4B vision (`gemma4v`)**: encoder implemented, no STATUS row, never parity-checked.
   The oracle now exists (`llama-mtmd-debug` stage fingerprints, as used for Kimi/Youtu).
   [2-coverage/03-gemma4-e4b-vision-plan.md](2-coverage/03-gemma4-e4b-vision-plan.md).
7. **ACE-Step 1.5 Turbo**: V1 works end to end; needs numeric parity and a STATUS row. The
   `audio.cpp` head-to-head is blocked on the `acestep-5Hz-lm-1.7B` package.
   [2-coverage/064-acestep-implementation-plan.md](2-coverage/064-acestep-implementation-plan.md).
8. **Qwen3.5 MoE / Gated DeltaNet**: GDN state-lifecycle conformance tests (incl. retained
   sessions), then a benchmark. [2-coverage/02-qwen35moe-plan.md](2-coverage/02-qwen35moe-plan.md).
9. **Parakeet TDT decode head** (only CTC exists).
10. **ONNX pipelines**: SenseVoice and ONNX Paraformer work but are not wired into `stingray stt`;
    now that several ONNX pipelines exist, see whether a shared shape is worth extracting.
11. **Gemma 1 / Gemma 2**: no `ModelGraph` branch at all (no local checkpoint to verify with).
12. **Newer LTX families** (LTX-2.3 / 2.5), a later campaign.
13. **Missing GGML op kernels**: [2-coverage/050-ggml-op-coverage-gap-plan.md](2-coverage/050-ggml-op-coverage-gap-plan.md)
    (none blocks an admitted architecture today).
14. **Lower priority, not planned**: AI21 Jamba, Kimi Linear, RWKV-7, Arcee AFM, ServiceNow
    Apriel, Ant Ling (`bailingmoe2`), MiniMax-M2 (too large); DeepSeek-OCR v1 (no checkpoint).
15. **Out of scope for this PC**: DeepSeek-V3.2 / V4, alpha code never run on real weights.
    [2-coverage/058-deepseek-full-lineage-implementation-plan.md](2-coverage/058-deepseek-full-lineage-implementation-plan.md).

## 3. Product and runtime

1. **Front door, steps 2-4**: `stingray setup`, a starter manifest, a model home, README recipes
   around task commands. [3-product-and-runtime/103-front-door-design.md](3-product-and-runtime/103-front-door-design.md).
2. **Configuration ownership**: source-tracked effective configuration beyond the static planning
   knobs, and an owner decision per obsolete-looking switch.
   [3-product-and-runtime/04-quality-of-life-improvements-plan.md](3-product-and-runtime/04-quality-of-life-improvements-plan.md).
3. **Multi-model runtime**: execution-level service quantum / starvation protection (Phase 6) and
   durable session restore across a restart in multi-model mode.
   [3-product-and-runtime/032-multi-model-inference-runtime-plan.md](3-product-and-runtime/032-multi-model-inference-runtime-plan.md).
4. **Sessions**: `Fork()` skill/instruction propagation (design question, wait for a real caller),
   per-session LoRA in the batched engine, and forward-pass context isolation for forks
   (`IForwardPass.CreateContext` still returns `this`).
   [051](3-product-and-runtime/051-hotsession-capability-wiring-plan.md),
   [010](3-product-and-runtime/010-forward-pass-context-isolation-for-session-forking-plan.md).
5. **Releases**: follow [3-product-and-runtime/nuget-release-checklist.md](3-product-and-runtime/nuget-release-checklist.md).
6. **Parked** (useful as is, does not move the goal): DSpark speculative decoding, SafeTensors
   Phases 4-6, the ONNX-runtime path for engines that already have native ports.

## 4. Performance

Rules: dispatch proof, interleaved control/candidate samples, named-model end-to-end measurement,
numerical validation; no single-run result counts. An iGPU loss is not evidence against a GPU path
(CLAUDE.md rule 13). Do not reopen the closed Q4_K repacked-GEMM investigation. The cross-model
sweep is [4-performance/perf-sweep-plan.md](4-performance/perf-sweep-plan.md).

**CPU, LLM** ([4-performance/cpu/](4-performance/cpu))
1. SmolLM2 prefill at ~0.89x of llama.cpp: the Q4_K Path-2 GEMM is 65% of trunk time; then RoPE
   (scalar, ~3%) and attention (~6%). History: "SmolLM2 prefill" in
   [done/101](done/101-work-queue-after-coverage-plan.md).
2. Qwen3.6-35B-A3B prefill at 0.63x of llama.cpp (Phase 8 of
   [done/2026-09-25-hf-top-downloads-coverage-plan.md](done/2026-09-25-hf-top-downloads-coverage-plan.md)).
3. Image-token prefill in VLMs runs per token (~9 t/s here); a batched embedding prefill.
4. The CPU kernel programme: [4-performance/cpu/05-cpu-architecture-kernel-opportunities.md](4-performance/cpu/05-cpu-architecture-kernel-opportunities.md).

**GPU, LLM** ([4-performance/gpu/](4-performance/gpu))
5. Qwen3.8-27B GPU-only: the embedding-lookup fix is implemented and waits for a real-weight run.
   [4-performance/gpu/084-vulkan-large-tensor-sharding-plan.md](4-performance/gpu/084-vulkan-large-tensor-sharding-plan.md).
6. Batched prefill for the Vulkan layer split (`VulkanLayerSplitForwardPass` prefills per token).
7. Raw-quant matvec bandwidth on Vulkan (gpt-oss decode is MXFP4 matvec-bound, ~18 GB/s here).

**Diffusion** ([4-performance/diffusion/](4-performance/diffusion))
8. FLUX.2 GPU: a clean end-to-end timing after `0958d6f`; Experiment 3 (production-shape GEMM
   ladder) and the `DoubleBlockGpu` row-offset audit; int8 dot-product quantized GEMM; the empty
   `VulkanMatMulPathConfig` "Path 2" seam. [093](4-performance/diffusion/093-flux2-gpu-performance-optimization-plan.md).
9. FLUX.1 DiT gap to sd.cpp (target 99.8 s, achieved 374.7 s then 132 s GPU with flash attention):
   [069](4-performance/diffusion/069-flux-vulkan-gemm-perf-handoff.md), Phase 6 of
   [094](4-performance/diffusion/094-diffusion-performance-plan.md).
10. Wan 2.1 DiT per block (target < 450 ms, achieved 770 ms):
    [073](4-performance/diffusion/073-wan21-kernel-fusion-and-qkv-plan.md).
11. HunyuanVideo: blocks are compute-bound at ~42% of the iGPU's fp32 peak (the GEMM kernel is the
    lever); the VAE mid-block causal attention is still scalar (measure its share first).
12. Cross-model DRY and perf-doc consistency: Phase 9 of [094](4-performance/diffusion/094-diffusion-performance-plan.md).
13. LTX-Video has no C++ comparison yet (sd.cpp's path for it is blocked).

**Audio** ([4-performance/audio/](4-performance/audio))
14. MiniMax-Music3 vocoder decode 29-33 s vs the reference's 13.1 s
    ([066](4-performance/audio/066-minimax-music3-future-plan.md)); flow-transformer GPU residency
    ([079](4-performance/audio/079-minimax-music3-gpu-residency-plan.md)).
15. MusicGen / AudioGen performance and DRY passes (CFG as a batch-2 GEMM, a T5 kernel shared with
    Parler) plus top-p sampling. "Known gaps" in [done/062](done/062-musicgen-implementation-plan.md).
16. CosyVoice3 ODE step count: the fewer-steps A/B (by ear plus a Whisper round trip) was never run;
    the default is still 10. Item 2 of [done/tts-performance-baseline-and-plan.md](done/tts-performance-baseline-and-plan.md).
17. Stable Audio 3: `IStableAudio3Engine` consolidation now that three variants exist (Sprint 3 of
    [done/065](done/065-stable-audio-3-medium-smallsfx-future-plan.md)).
18. TTS/ASR GPU residency and Vulkan `--backend` options, one engine at a time:
    [080](4-performance/audio/080-tts-asr-gpu-residency-checklist.md),
    [052](4-performance/audio/052-vulkan-backend-for-tts-engines-plan.md),
    [053](4-performance/audio/053-vulkan-backend-audit-remaining-audio-engines.md).

**Measurement gaps in `PerformanceLeague.md`**
19. Vision timings: Pixtral 12B and GLM-4.6V (gated; need an `HF_TOKEN` with the licence accepted),
    LLaVA-NeXT/OneVision, GLM-4V/OCR, Hunyuan-VL, Llama 4 vision.

## 9. Needs hardware this machine does not have

[9-external-hardware/90-external-hardware-work.md](9-external-hardware/90-external-hardware-work.md):
CUDA release evidence, Gemma 4 12B on CUDA, CUDA gate/up fusion, the CUDA graph default, ARM64.
Also waiting on hardware:
- CUDA passes apply `rope_freqs` for Gemma 4 only, so Llama-3.1-style models on CUDA are wrong at
  long context (fixed on Vulkan 2026-09-26).
- CUDA partial offload for the newer architectures (Vulkan `-g N` is done).
- Discrete-GPU measurements before changing any default: speculative decoding with a draft model
  on Vulkan, the UMT5 GPU path, gpt-oss on Vulkan.
