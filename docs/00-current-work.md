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

1. **Vision features that exist only on the CPU/Vulkan `ForwardPass`.** Qwen3-VL image input
   (2D M-RoPE, deepstack) now runs on full Vulkan offload (2026-09-28, final-logit cosine ~0.9995 vs CPU,
   `Qwen3VlVulkanMRopeParityTests`). Still open: CUDA full and hybrid offload and the Vulkan layer split
   (unimplemented, CLI fails closed; CUDA cannot be verified here, bugstofix item 14 closed as not-a-bug);
   Qwen2.5-VL / PaddleOCR-VL / Granite 4.0 Vision deepstack on GPU are unverified; the server's image path
   (`InferenceEngine`, Gemma-style placeholders only). Plan:
   [1-correctness/14](1-correctness/14-qwenvl-gpu-image-input-parity-plan.md).
2. **Per-pipeline diffusion end-to-end smoke tests**: real weights, small resolution, a stored
   reference, and a loud failure when the checkpoint is missing. Item 4 of §4 in
   [done/2026-09-24-diffusion-perf-session-handoff.md](done/2026-09-24-diffusion-perf-session-handoff.md).
6. **CPU greedy-decode non-determinism**: two sightings under CPU contention, neither reproduced.
8. **Stable Audio 3 APG padding masks**: **FIXED 2026-09-28**. Padded tokens masked from APG norm and dot product, orthogonal projection zeroed on padded tokens (`ApplyApg`), epsilon moved inside sqrt, and `ValidLatentTokens` / self-attention V-zeroing wired to `StableAudioMediumDiT`. Step 0 velocity cosine 0.999949; padded token final latent cosine 0.9999998 against `audiocpp_cli`. Attention masking over the padded tail remains a known gap. #11 in [done/102](done/102-status-open-items-plan.md).
9. **Youtu-VL text**: one 1024-token wikitext window is +5% PPL vs llama.cpp (others −2.1% to
    +0.8%; whole-file 115.6 vs 114.1, +1.3%, in STATUS). Still the only unexplained window. Diff per-token log-probs over wiki.test.raw [1024,2048) against `llama-server`.
    #5 in [done/102](done/102-status-open-items-plan.md).
11. **Classic LLaVA-1.5** (plain 32000-token vocab, no image token): **FIXED 2026-09-27**. The CLI
    now has a direct-splice path when `PlaceholderMarker` is absent from special tokens, tokenizing
    prompt text around the image marker and injecting soft tokens. Added Vicuna prompt formatting
    for LLaMA-2 backbones (`USER: <image>{prompt}\nASSISTANT:`). Fixed ViT patch/CLS layout in
    `LlavaVisionEncoder.cs` to match `llama.cpp`'s `clip_graph_llava::build` (patches at 0..575, CLS
    at 576, extracted at 1..576 dropping row 0). Pinned by `LlamaMtmdVisionParityTests.Llava15_Rainbow336_MatchesLlamaMtmdDebug`
    (sum -10587.12 vs -10596.39). End-to-end on `test-1.png` reads "The newspaper is the New York Times,
    and the main headline reads \"Men Walk on Moon.\"" matching `llama-mtmd-cli`. (`llava_uhd` / anyres
    tiling is **open**: checkpoint obtained 2026-10-01, interim result is 5 views agreeing with semantically
    equal answers but exact parity not yet shown; bugstofix item 15,
    [1-correctness/15](1-correctness/15-llava-next-anyres-parity-plan.md)).
12. **CosyVoice 2 garbled endings — BLOCKED** on an independent reference: one recorded upstream
    `inference_zero_shot` run (input token ids, generated speech tokens, ideally per-step top-k),
    checked in as data. What is already ruled out: #10 in [done/102](done/102-status-open-items-plan.md).
13. **HunyuanVideo numeric verification — OPEN, not blocked**: the vendored sd.cpp supports
    HunyuanVideo 1.5 only; patched to load v1 it produces noise (1-step velocity cosine 0.187), so it is
    not an independent reference. An independent v1 reference exists: diffusers' first-party
    `HunyuanVideoPipeline` (also city96's v1 GGUF via ComfyUI-GGUF). Next step: one diffusers CPU run
    (256², 1 frame, fixed noise; bf16 transformer ~26 GB) recorded as checked-in data (noise,
    conditioning, 1-step velocity, 8-step latent), as done for #10; keep the script in the scratchpad
    (no new Python reference scripts in the repo). Then compare `HunyuanVideoModel` against it. The row
    stays ⚪ until then. #13 in [done/102](done/102-status-open-items-plan.md) lists the local sd.cpp patch.
14. **Dashboard of what is still open**: [1-correctness/bugstofix.md](1-correctness/bugstofix.md) is the
    source of truth (open: 15, 19, 20, 21).
15. **Small, known leftovers**: [1-correctness/bugstofix.md](1-correctness/bugstofix.md)
    (`SpeculativeDecoder` StepSampled/PLD latent defect, dtype/op drift notes); Apertus greedy
    re-check vs `llama-server --no-jinja`.

Not fixable on this machine (kept 🔴 in STATUS): MiMo-VL (upstream mmproj projects to 3584, the
text model wants 4096), Llama 4 vision (93 GB), MobileNetV5 (no checkpoint declares the projector).

## 2. Model coverage

1. **GLM-4.5 / 4.6 / 4.7 incl. Air (`glm4moe`): implemented, NOT admitted.** `glm4moe` is absent from
   `ModelCompatibility`'s allowlist, so the engine rejects it unless
   `STINGRAY_DIAGNOSTIC_ALLOW_UNSUPPORTED_ARCH=1`. Real-weight evidence exists (GLM-4.5-Air REAP 82B Q2_K,
   paired PPL 8.7753 default / 8.5956 with `STINGRAY_Q5K_DECODE_Q8K=1` vs llama.cpp 8.6125; bugstofix 09),
   but admission is gated on `docs/103` item 19 and a greedy-token receipt. GLM-4.7-Flash is `deepseek2`
   and already admitted (paired +1.3% PPL, characterized).
2. **Recurrent-state families, follow-ups** (Granite 4.0-H, Nemotron-H and LFM2 were admitted on
   CPU on 2026-09-27; #14, #15, #17 in [done/102](done/102-status-open-items-plan.md)):
   - Mamba-2 / short-conv support in the batched `PrefillCore` (prefill is token by token: Granite
     1B 17 tok/s vs llama.cpp ~110);
   - partial rewind of the recurrent state (`SupportsPartialRewind` is false, so the server's
     prefix cache is off for these models), and no zero KV rows for recurrent layers;
   - GPU paths;
   - untested variants: MoE Nemotron-H, LFM2-VL/Audio (MoE Granite-H was fixed 2026-09-28 and `lfm2moe` admitted 2026-10-01, see bugstofix 12/13), and the
     Nemotron-Nano-12B-v2-VL vision tower end to end; Falcon-H1 should come almost free.
3. ~~**Qwen3-VL**~~: done 2026-09-27 on CPU (IMROPE, deepstack, `qwen3vl` admitted; see STATUS) and on full Vulkan offload 2026-09-28. CUDA image input remains (needs hardware).
4. **Gemma 4 E4B vision (`gemma4v`)**: encoder and projector match `llama-mtmd-debug` (STATUS row,
   2026-09-27). Open: an end-to-end image answer through the Gemma 4 text model has not been compared yet
   (`docs/103` item 4). [done/03-gemma4-e4b-vision-plan.md](done/03-gemma4-e4b-vision-plan.md).
5. **ACE-Step 1.5 Turbo**: compared against `audio.cpp` from identical noise 2026-09-28 (STATUS row;
   8-step latent cosine 0.994, waveform 0.945 with a q8_0 DiT reference). Open: not wired into the CLI,
   the planner LM (lyrics-to-codes) path is not ported, and the bf16 bundle ceiling check.
   [done/064-acestep-implementation-plan.md](done/064-acestep-implementation-plan.md).
6. **Qwen3.5 MoE / Gated DeltaNet**: GDN state-lifecycle conformance tests (incl. retained
   sessions), then a benchmark (2026-10-01: GDN kernels now checked against an independent double-precision
   reference). [done/02-qwen35moe-plan.md](done/02-qwen35moe-plan.md).
7. ~~**Parakeet TDT decode head**~~: done 2026-09-27 (`ParakeetTdtDecoder`, matches CrispASR on 4/4 LibriSpeech clips).
8. **ONNX pipelines**: SenseVoice and ONNX Paraformer are wired into `stingray stt -m sensevoice|paraformer
    --model-file` (2026-09-27, `docs/103` item 5). Open question only: with several ONNX pipelines now,
    is a shared shape worth extracting?
9. **Gemma 1 / Gemma 2**: no `ModelGraph` branch at all (no local checkpoint to verify with).
10. **Newer LTX families** (LTX-2.3 / 2.5), a later campaign.
11. **Missing GGML op kernels**: [2-coverage/050-ggml-op-coverage-gap-plan.md](2-coverage/050-ggml-op-coverage-gap-plan.md)
    (none blocks an admitted architecture today).
12. **RWKV6 / RWKV7 CPU, generic `SOLVE_TRI`**: bugstofix items 19, 20, 21 (deferred, targets chosen:
    rwkv6-world-1b6, rwkv7-goose-world3-1b5; plan [1-correctness/17](1-correctness/17-ggml-op-coverage-verification-plan.md)).
13. **Lower priority, not planned**: AI21 Jamba, Kimi Linear, Arcee AFM, ServiceNow
    Apriel, Ant Ling (`bailingmoe2`), MiniMax-M2 (too large); DeepSeek-OCR v1 (no checkpoint).
14. **Out of scope for this PC**: DeepSeek-V3.2 / V4, alpha code never run on real weights.
    [2-coverage/058-deepseek-full-lineage-implementation-plan.md](2-coverage/058-deepseek-full-lineage-implementation-plan.md).

## 3. Product and runtime

1. **Front door, steps 3-4**: a starter manifest and README recipes around task commands. Step 2 is done
   2026-09-28 (`stingray setup`, `stingray models`, `ModelHome`, three-entry checksum-verified catalog). [3-product-and-runtime/103-front-door-design.md](3-product-and-runtime/103-front-door-design.md).
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

**CPU, LLM**
1. **CPU Q8 prefill is default-off since 2026-10-01** (exact numerical parity with llama.cpp, found on Granite-H
   small MoE). Cost measured on SmolLM2-1.7B Q4_K_M, 2,194-token prompt, 3 runs each: 81.0 t/s default vs
   224.2 t/s with `STINGRAY_CPU_PREFILL_Q8=1` (2.8x). Every dense model pays this; the 0.89x-of-llama.cpp
   figure below predates the flip and is stale until re-measured. Open: a Q8 prefill that keeps parity,
   or a per-architecture default ([1-correctness/13](1-correctness/13-granite4-h-small-moe-ppl-parity-plan.md)).
   Older note, SmolLM2 prefill at ~0.89x of llama.cpp: the Q4_K Path-2 GEMM is 65% of trunk time; then RoPE
   (scalar, ~3%) and attention (~6%). History: "SmolLM2 prefill" in
   [done/101](done/101-work-queue-after-coverage-plan.md).
2. Qwen3.6-35B-A3B prefill at 0.63x of llama.cpp (Phase 8 of
   [done/2026-09-25-hf-top-downloads-coverage-plan.md](done/2026-09-25-hf-top-downloads-coverage-plan.md)).
3. Image-token prefill in VLMs runs per token (~9 t/s here); a batched embedding prefill.
4. The CPU kernel programme: [done/05-cpu-architecture-kernel-opportunities.md](done/05-cpu-architecture-kernel-opportunities.md).

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
    ([066](done/066-minimax-music3-future-plan.md)); flow-transformer GPU residency
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
