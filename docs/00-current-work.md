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
6. **CPU greedy-decode non-determinism**: two sightings under CPU contention, neither reproduced.
9. **Youtu-VL text**: one 1024-token wikitext window is +5% PPL vs llama.cpp (others −2.1% to
    +0.8%). Diff per-token log-probs over wiki.test.raw [1024,2048) against `llama-server`.
    #5 in [done/102](done/102-status-open-items-plan.md).
12. **CosyVoice 2 garbled endings — BLOCKED** on an independent reference: one recorded upstream
    `inference_zero_shot` run (input token ids, generated speech tokens, ideally per-step top-k),
    checked in as data. What is already ruled out: #10 in [done/102](done/102-status-open-items-plan.md).
13. **HunyuanVideo numeric verification — BLOCKED**: the vendored sd.cpp supports HunyuanVideo
    1.5 only; patched to load v1 it produces noise (1-step velocity cosine 0.187), so it is not an
    independent reference. Needs one v1 reference run (ComfyUI or diffusers) recorded as data, or
    an upstream C++ port with v1 support. The row stays ⚪ with visual-only evidence. #13 in
    [done/102](done/102-status-open-items-plan.md) lists the local sd.cpp patch.
14. **Remaining open items in [1-correctness/bugstofix.md](1-correctness/bugstofix.md)**: #22 stale
    numpy-golden vision parity tests (4 failing in the Vision run) and #15 LLaVA-NeXT/OneVision residuals
    (pixel-level and per-view embedding parity unverified). Closed 2026-10-01: #19/#20 (RWKV6/RWKV7 admitted)
    and #23 (DeepSeek-V2-Lite receipt; one decision left: whether F32 prefill stays its default).
    Also: Apertus greedy re-check vs `llama-server --no-jinja`.
15. **Stable Audio 3 Small quality**: our local SA3 checkpoints are the `-base` (pre-trained) models, which sound    worse than the post-trained releases. Waiting on the user's listening verdict on the audio.cpp post-trained    clips (`docs/audio-samples/sa3_small_*_POSTTRAINED_*`); if better, add GGUF loading (or gated safetensors via    `HF_TOKEN`) and an opt-in `pingpong` sampler (8 steps, CFG 1.0). Detail: [done/104](done/104-handover-2026-09-28.md) §2.

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
   - untested variants: MoE Nemotron-H, LFM2-VL/Audio, and the Nemotron-Nano-12B-v2-VL vision tower
     end to end; Falcon-H1 should come almost free.
8. **ONNX pipelines**: SenseVoice and ONNX Paraformer work but are not wired into `stingray stt`;
    now that several ONNX pipelines exist, see whether a shared shape is worth extracting.
9. **Gemma 1 / Gemma 2**: no `ModelGraph` branch at all (no local checkpoint to verify with).
10. **Newer LTX families** (LTX-2.3 / 2.5), a later campaign.
11. **Missing GGML op kernels**: none left; RWKV6 / RWKV7 closed 2026-10-01 (see
    [done/bugstofix-resolved-2026-10.md](done/bugstofix-resolved-2026-10.md); the rest of the old op gap: [done/17](done/17-ggml-op-coverage-verification-plan.md)).
12. **Lower priority, not planned**: AI21 Jamba, Kimi Linear, Arcee AFM, ServiceNow
    Apriel, Ant Ling (`bailingmoe2`), MiniMax-M2 (too large); DeepSeek-OCR v1 (no checkpoint).
13. **Out of scope for this PC**: DeepSeek-V3.2 / V4, alpha code never run on real weights.
    [2-coverage/058-deepseek-full-lineage-implementation-plan.md](2-coverage/058-deepseek-full-lineage-implementation-plan.md).
14. **TensorSharp coverage wave** (from 2026-10-02; the user's current focus, ahead of the items
    above while it runs).
    - **Done:** spin worker pool (opt-in), ggml Q8 cross-check, `Q1_0`/`Q2_0` verified, Bonsai2
      PRISM ported (gated). Muse-Glimmer **admitted 2026-10-03** (text, CPU only).
    - **Current, in order:** GLM-5.x (paged real-weight run, K-pool selection), DeepSeek V4 (paged run vs b10306),
      Qwen 3.8 Flash Next (implementation & 24 synthetic tests complete, paged Level 4 run pending), DiffusionGemma (implementation & 31 synthetic tests complete, Q4_K_M Level 4 run pending), MiniMax-H3, V4.1 (not attemptable
      here). One plan each in `2-coverage/`; the plans carry the honest state.
    - Ported families stay **not admitted and not advertised** until real-weight verified
      (status ladder and verification levels in the families todo).
    - New families beyond these get their own plan when someone takes them on.
    [2-coverage/2026-10-02-tensorsharp-takeaways-plan.md](2-coverage/2026-10-02-tensorsharp-takeaways-plan.md);
    per-family todo: [2-coverage/ported-families-todo.md](2-coverage/ported-families-todo.md).

## 3. Product and runtime

- **Architecture semantics in the descriptor** (adding a model = one component, no central edits): plan and live checklist in
  [2-coverage/2026-10-08-architecture-semantics-admission-plan.md](2-coverage/2026-10-08-architecture-semantics-admission-plan.md).

1. **Front door, steps 3-4** (steps 1-2 done: README, catalog, model home, `stingray setup`): starter manifest, README recipes
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
   [051](done/051-hotsession-capability-wiring-plan.md),
   [010](3-product-and-runtime/010-forward-pass-context-isolation-for-session-forking-plan.md).
5. **Releases**: follow [3-product-and-runtime/nuget-release-checklist.md](3-product-and-runtime/nuget-release-checklist.md).
6. **Parked** (useful as is, does not move the goal): DSpark speculative decoding, SafeTensors
   Phases 4-6, the ONNX-runtime path for engines that already have native ports.

## 4. Performance

Rules: dispatch proof, interleaved control/candidate samples, named-model end-to-end measurement,
numerical validation; no single-run result counts. An iGPU loss is not evidence against a GPU path
(CLAUDE.md rule 13). Do not reopen the closed Q4_K repacked-GEMM investigation. The cross-model
sweep is [4-performance/perf-sweep-plan.md](4-performance/perf-sweep-plan.md).

**CPU, LLM** (the closed CPU kernel programme is [done/05](done/05-cpu-architecture-kernel-opportunities.md))
1. SmolLM2 prefill at ~0.89x of llama.cpp: the Q4_K Path-2 GEMM is 65% of trunk time; then RoPE
   (scalar, ~3%) and attention (~6%). History: "SmolLM2 prefill" in
   [done/101](done/101-work-queue-after-coverage-plan.md).
2. Qwen3.6-35B-A3B prefill at 0.63x of llama.cpp (Phase 8 of
   [done/2026-09-25-hf-top-downloads-coverage-plan.md](done/2026-09-25-hf-top-downloads-coverage-plan.md)).
3. Image-token prefill in VLMs runs per token (~9 t/s here); a batched embedding prefill.
4a. Spin-then-park CPU worker pool (TensorSharp `CpuWorkerPool`) for the tiny-model decode gap
   (SmolLM2-135M ~0.55x, Qwen2.5-0.5B ~0.75x): an experiment the earlier `PersistentThreadPool`
   loss did not cover. §1 of
   [2-coverage/2026-10-02-tensorsharp-takeaways-plan.md](2-coverage/2026-10-02-tensorsharp-takeaways-plan.md).
4b. int8 alignment with ggml Q8_K (ADR-0003 known gap; per-token int8 matvec 1.2e-2 rel error per
   projection), with TensorSharp's managed Q8_K kernels as a second reference. Follow-ups under 10.2
   in [4-performance/perf-sweep-plan.md](4-performance/perf-sweep-plan.md).

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
13. MiniMax-Music3 flow-transformer GPU residency, which matters only on a real GPU (the vocoder gap
    closed 2026-09-28, [done/066](done/066-minimax-music3-future-plan.md)):
    [079](4-performance/audio/079-minimax-music3-gpu-residency-plan.md).
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
