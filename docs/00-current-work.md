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

1. **Vision features that exist only on the CPU `ForwardPass`.** 2D M-RoPE image positions
   (PaddleOCR-VL, Qwen2.5-VL, …) and Granite 4.0 Vision deepstack are not applied by the CUDA and
   Vulkan passes, nor by the server's image path (`InferenceEngine` handles Gemma-style placeholders
   only). Today those models give correct answers from the CLI on CPU only. From #6 and #9 in
   [done/102](done/102-status-open-items-plan.md).
2. **Per-pipeline diffusion end-to-end smoke tests**: real weights, small resolution, a stored
   reference, and a loud failure when the checkpoint is missing. Item 4 of §4 in
   [done/2026-09-24-diffusion-perf-session-handoff.md](done/2026-09-24-diffusion-perf-session-handoff.md).
3. **Silent-no-op test sweep** (CLAUDE.md rule 12). Several real-weight tests search only
   `models/`, not `models/_models/`, and "pass" in 0.1 s without loading anything (Parakeet,
   Orpheus/SNAC and Z-Image were found by accident). Sweep every model-search helper against the
   real `models/_models/` contents; better, make a missing checkpoint skip visibly.
4. **`HybridGdnChunkedPrefill_MatchesSequentialPrefill` fails**, re-run with real weights on
   2026-09-27 and still failing: logits diverge at vocab idx 142707 (sequential 0.4995 vs chunked
   0.0924, tolerance 0.1025). See Phase 8 of [done/2026-09-25-hf-top-downloads-coverage-plan.md](done/2026-09-25-hf-top-downloads-coverage-plan.md).
6. **CPU greedy-decode non-determinism**: two sightings under CPU contention, neither reproduced.
8. **Stable Audio 3 APG padding masks**: **FIXED 2026-09-28**. Padded tokens masked from APG norm and dot product, orthogonal projection zeroed on padded tokens (`ApplyApg`), epsilon moved inside sqrt, and `ValidLatentTokens` / self-attention V-zeroing wired to `StableAudioMediumDiT`. Step 0 velocity cosine 0.999949; padded token final latent cosine 0.9999998 against `audiocpp_cli`. Attention masking over the padded tail remains a known gap. #11 in [done/102](done/102-status-open-items-plan.md).
9. **Youtu-VL text**: one 1024-token wikitext window is +5% PPL vs llama.cpp (others −2.1% to
    +0.8%). Diff per-token log-probs over wiki.test.raw [1024,2048) against `llama-server`.
    #5 in [done/102](done/102-status-open-items-plan.md).
11. **Classic LLaVA-1.5** (plain 32000-token vocab, no image token): **FIXED 2026-09-27**. The CLI
    now has a direct-splice path when `PlaceholderMarker` is absent from special tokens, tokenizing
    prompt text around the image marker and injecting soft tokens. Added Vicuna prompt formatting
    for LLaMA-2 backbones (`USER: <image>{prompt}\nASSISTANT:`). Fixed ViT patch/CLS layout in
    `LlavaVisionEncoder.cs` to match `llama.cpp`'s `clip_graph_llava::build` (patches at 0..575, CLS
    at 576, extracted at 1..576 dropping row 0). Pinned by `LlamaMtmdVisionParityTests.Llava15_Rainbow336_MatchesLlamaMtmdDebug`
    (sum -10587.12 vs -10596.39). End-to-end on `test-1.png` reads "The newspaper is the New York Times,
    and the main headline reads \"Men Walk on Moon.\"" matching `llama-mtmd-cli`. (Note: `llava_uhd` /
    anyres tiling remains unverified until a LLaVA-NeXT / 1.6 checkpoint is tested).
12. **CosyVoice 2 garbled endings — BLOCKED** on an independent reference: one recorded upstream
    `inference_zero_shot` run (input token ids, generated speech tokens, ideally per-step top-k),
    checked in as data. What is already ruled out: #10 in [done/102](done/102-status-open-items-plan.md).
13. **HunyuanVideo numeric verification — BLOCKED**: the vendored sd.cpp supports HunyuanVideo
    1.5 only; patched to load v1 it produces noise (1-step velocity cosine 0.187), so it is not an
    independent reference. Needs one v1 reference run (ComfyUI or diffusers) recorded as data, or
    an upstream C++ port with v1 support. The row stays ⚪ with visual-only evidence. #13 in
    [done/102](done/102-status-open-items-plan.md) lists the local sd.cpp patch.
14. **Small, known leftovers**: [1-correctness/bugstofix.md](1-correctness/bugstofix.md)
    (`SpeculativeDecoder` StepSampled/PLD latent defect, dtype/op drift notes); Apertus greedy
    re-check vs `llama-server --no-jinja`.

Not fixable on this machine (kept 🔴 in STATUS): MiMo-VL (upstream mmproj projects to 3584, the
text model wants 4096), Llama 4 vision (93 GB), MobileNetV5 (no checkpoint declares the projector).

## 2. Model coverage

1. **GLM-4.5 / 4.6 / 4.7 incl. Air (`glm4moe`)**: a top open family; GLM-4 dense already runs.
2. **Recurrent-state families, follow-ups** (Granite 4.0-H, Nemotron-H and LFM2 were admitted on
   CPU on 2026-09-27; #14, #15, #17 in [done/102](done/102-status-open-items-plan.md)):
   - Mamba-2 / short-conv support in the batched `PrefillCore` (prefill is token by token: Granite
     1B 17 tok/s vs llama.cpp ~110);
   - partial rewind of the recurrent state (`SupportsPartialRewind` is false, so the server's
     prefix cache is off for these models), and no zero KV rows for recurrent layers;
   - GPU paths;
   - untested variants: MoE Granite-H, MoE Nemotron-H, LFM2-MoE (`lfm2moe`), LFM2-VL/Audio, and the
     Nemotron-Nano-12B-v2-VL vision tower end to end; Falcon-H1 should come almost free.
3. ~~**Qwen3-VL**~~: done 2026-09-27 on CPU (IMROPE, deepstack, `qwen3vl` admitted; see STATUS). GPU image input remains.
4. **Gemma 4 E4B vision (`gemma4v`)**: encoder implemented, no STATUS row, never parity-checked.
   The oracle now exists (`llama-mtmd-debug` stage fingerprints, as used for Kimi/Youtu).
   [2-coverage/03-gemma4-e4b-vision-plan.md](2-coverage/03-gemma4-e4b-vision-plan.md).
5. **ACE-Step 1.5 Turbo**: V1 works end to end; needs numeric parity and a STATUS row. The
   `audio.cpp` head-to-head is blocked on the `acestep-5Hz-lm-1.7B` package.
   [2-coverage/064-acestep-implementation-plan.md](2-coverage/064-acestep-implementation-plan.md).
6. **Qwen3.5 MoE / Gated DeltaNet**: GDN state-lifecycle conformance tests (incl. retained
   sessions), then a benchmark. [2-coverage/02-qwen35moe-plan.md](2-coverage/02-qwen35moe-plan.md).
7. ~~**Parakeet TDT decode head**~~: done 2026-09-27 (`ParakeetTdtDecoder`, matches CrispASR on 4/4 LibriSpeech clips).
8. **ONNX pipelines**: SenseVoice and ONNX Paraformer work but are not wired into `stingray stt`;
    now that several ONNX pipelines exist, see whether a shared shape is worth extracting.
9. **Gemma 1 / Gemma 2**: no `ModelGraph` branch at all (no local checkpoint to verify with).
10. **Newer LTX families** (LTX-2.3 / 2.5), a later campaign.
11. **Missing GGML op kernels**: [2-coverage/050-ggml-op-coverage-gap-plan.md](2-coverage/050-ggml-op-coverage-gap-plan.md)
    (none blocks an admitted architecture today).
12. **Lower priority, not planned**: AI21 Jamba, Kimi Linear, RWKV-7, Arcee AFM, ServiceNow
    Apriel, Ant Ling (`bailingmoe2`), MiniMax-M2 (too large); DeepSeek-OCR v1 (no checkpoint).
13. **Out of scope for this PC**: DeepSeek-V3.2 / V4, alpha code never run on real weights.
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
5. Batched prefill for the Vulkan layer split (`VulkanLayerSplitForwardPass` prefills per token).
6. Raw-quant matvec bandwidth on Vulkan (gpt-oss decode is MXFP4 matvec-bound, ~18 GB/s here).
7. General multi-`VkBuffer` sharding for tensors beyond `maxStorageBufferRange`: undesigned, not
   needed by any current checkpoint ([done/084](done/084-vulkan-large-tensor-sharding-plan.md)).

**Diffusion** ([4-performance/diffusion/](4-performance/diffusion))
8. FLUX.2 GPU: Experiment 3 (production-shape GEMM ladder) and the `DoubleBlockGpu` row-offset
   audit; a true quantized GEMM (the `VulkanMatMulPath` "Path 2" seam is not implemented; it would
   help FLUX.1 too). [093](4-performance/diffusion/093-flux2-gpu-performance-optimization-plan.md).
9. FLUX.1 DiT gap to sd.cpp: 132.0 s vs the 99.8 s target (~1.32x) since the 2026-09-25 flash
   attention:
   [069](4-performance/diffusion/069-flux-vulkan-gemm-perf-handoff.md), Phase 6 of
   [094](4-performance/diffusion/094-diffusion-performance-plan.md).
10. HunyuanVideo: blocks are compute-bound at ~42% of the iGPU's fp32 peak (the GEMM kernel is the
    lever); the VAE mid-block causal attention is still scalar (measure its share first).
11. Cross-model DRY and perf-doc consistency: Phase 9 of [094](4-performance/diffusion/094-diffusion-performance-plan.md).
12. LTX-Video has no C++ comparison yet (sd.cpp's path for it is blocked).

**Audio** ([4-performance/audio/](4-performance/audio))
13. MiniMax-Music3 vocoder decode 29-33 s vs the reference's 13.1 s
    ([066](4-performance/audio/066-minimax-music3-future-plan.md)); flow-transformer GPU residency
    ([079](4-performance/audio/079-minimax-music3-gpu-residency-plan.md)).
14. MusicGen / AudioGen performance and DRY passes (CFG as a batch-2 GEMM, a T5 kernel shared with
    Parler) plus top-p sampling. "Known gaps" in [done/062](done/062-musicgen-implementation-plan.md).
15. CosyVoice3 ODE step count: the fewer-steps A/B (by ear plus a Whisper round trip) was never run;
    the default is still 10. Item 2 of [done/tts-performance-baseline-and-plan.md](done/tts-performance-baseline-and-plan.md).
16. Stable Audio 3: `IStableAudio3Engine` consolidation now that three variants exist (Sprint 3 of
    [done/065](done/065-stable-audio-3-medium-smallsfx-future-plan.md)).
17. TTS/ASR GPU residency, one engine at a time:
    [080](4-performance/audio/080-tts-asr-gpu-residency-checklist.md). The Vulkan `--backend`
    audits are closed ([done/052](done/052-vulkan-backend-for-tts-engines-plan.md),
    [done/053](done/053-vulkan-backend-audit-remaining-audio-engines.md)); Whisper, Parakeet and
    FunASR encoders are named future candidates there.

**Measurement gaps in `PerformanceLeague.md`**
18. Vision timings: Pixtral 12B and GLM-4.6V (gated; need an `HF_TOKEN` with the licence accepted),
    LLaVA-NeXT/OneVision, GLM-4V/OCR, Hunyuan-VL, Llama 4 vision.
19. Llama-4 Scout 17B-16E: no ratio vs llama.cpp (the ~93 GB download failed once; its state is
    unknown, and it does not fit this machine's 64 GB RAM).

## 9. Needs hardware this machine does not have

[9-external-hardware/90-external-hardware-work.md](9-external-hardware/90-external-hardware-work.md):
CUDA release evidence, Gemma 4 12B on CUDA, CUDA gate/up fusion, the CUDA graph default, ARM64.
Also waiting on hardware:
- CUDA passes apply `rope_freqs` for Gemma 4 only, so Llama-3.1-style models on CUDA are wrong at
  long context (fixed on Vulkan 2026-09-26).
- CUDA partial offload for the newer architectures (Vulkan `-g N` is done).
- Discrete-GPU measurements before changing any default: speculative decoding with a draft model
  on Vulkan, the UMT5 GPU path, gpt-oss on Vulkan.
