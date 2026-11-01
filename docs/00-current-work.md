# Current work: outstanding items, quickest first

Merged 2026-11-01 from the former themed backlog (this file) and `done/103-quickest-first-plan-merged-2026-11-01.md` (the quick-first working order, now the order of this file). The order is the quick-first order below. Each item appears once. Items from the themed backlog that have no quick-list entry follow in section order under "Remaining backlog". Completed items are in [done/](done) and their evidence is in each model's STATUS row.

## Rules

1. **Order.** Take items in this order. Do not reorder because another view ranks something higher.
2. **If an item stalls,** write down the blocker precisely: in this file, and in `bugstofix.md` when
   it is a bug. Then move to the next item (CLAUDE.md).
3. **Evidence.** "Done" needs the check in the item's **Done when** line, with real weights and an
   independent reference where one applies. Then:
   - update the STATUS.md row;
   - update the [RUNNING.md](RUNNING.md) row (command, measured RAM and speed, dated);
   - commit.
4. **Heavy runs one at a time.** Two 30 GB processes at once runs this 64 GB machine out of memory,
   and concurrent runs make timings meaningless.
5. **Estimates are guesses** unless an item says it has been investigated. Record the real time
   taken when an item closes.

---

## About an hour

(none open; the item 3 closure is in done/bugstofix-resolved-2026-10.md item 10)

---

## A day or more

### 13. MoE variants of the recurrent families
- [ ] **13. MoE variants of the recurrent families**
  - [x] Admit Granite 4.0-H tiny/small (MoE) with PPL parity. Small DONE 2026-09-28 (top-k renormalisation fix; +1.2% residual at -c 2048 logged).
  - [ ] Admit Nemotron-H MoE (latent MoE, sigmoid gating) with PPL parity. No local checkpoint (2026-09-28); not downloaded (C: has no free space, K: ~10 GB).
  - [ ] Admit LFM2-MoE (`lfm2moe`) with PPL parity. BLOCKED 2026-09-28 (timeboxed): PPL +6.3% and per-token vs batched disagree by 9%; see bugstofix.
  - [ ] **Done when:** each variant admitted with PPL parity against reference.

### 16. Product items
- [ ] **16. Product items**: the four sub-items are the next four lines.
  - [ ] 2. **Configuration ownership**: source-tracked effective configuration beyond the static planning
   knobs, and an owner decision per obsolete-looking switch.
   [3-product-and-runtime/04-quality-of-life-improvements-plan.md](3-product-and-runtime/04-quality-of-life-improvements-plan.md).
  - [ ] 3. **Multi-model runtime**: execution-level fairness is built as an opt-in turn gate (Phase 6 slice 5, 2026-10-31; off by default, continuous-batching engines only, no throughput measurement) and
   durable session restore across a restart in multi-model mode.
   [3-product-and-runtime/032-multi-model-inference-runtime-plan.md](3-product-and-runtime/032-multi-model-inference-runtime-plan.md).
  - [ ] 4. **Sessions**: `Fork()` skill/instruction propagation (design question, wait for a real caller),
   per-session LoRA in the batched engine, and the unused `IForwardPass.CreateContext` (removed 2026-11-01 on the owner's decision, plan 010)
   (2026-10-31: nothing in `src/` calls it; session forking goes through `HotSessionRuntime.Fork` and the shared KV cache, and now has a concurrent real-model test).
   [051](done/051-hotsession-capability-wiring-plan.md),
   [010](3-product-and-runtime/010-forward-pass-context-isolation-for-session-forking-plan.md).
  - [ ] 5. **Releases**: follow [reference/nuget-release-checklist.md](reference/nuget-release-checklist.md).

---

## Can't be scheduled

### 17. CPU greedy non-determinism
- [ ] **17. CPU greedy non-determinism** (never reproduced; act only on a new sighting).
  - Detail: 6. **CPU greedy-decode non-determinism**: two sightings under CPU contention, neither reproduced.

### 18. Blocked on the user or on other hardware
- [ ] **18. Blocked items**: the four entries below are its sub-items.
  - 12. **CosyVoice 2 garbled endings — BLOCKED** on an independent reference: one recorded upstream
    `inference_zero_shot` run (input token ids, generated speech tokens, ideally per-step top-k),
    checked in as data. What is already ruled out: #10 in [done/102](done/102-status-open-items-plan.md).
  - 13. **HunyuanVideo numeric verification — BLOCKED**: the vendored sd.cpp supports HunyuanVideo
    1.5 only; patched to load v1 it produces noise (1-step velocity cosine 0.187), so it is not an
    independent reference. Needs one v1 reference run (ComfyUI or diffusers) recorded as data, or
    an upstream C++ port with v1 support. The row stays ⚪ with visual-only evidence. #13 in
    [done/102](done/102-status-open-items-plan.md) lists the local sd.cpp patch.
  - 18. Vision timings: Pixtral 12B and GLM-4.6V (gated; need an `HF_TOKEN` with the licence accepted),
    LLaVA-NeXT/OneVision, GLM-4V/OCR, Hunyuan-VL, Llama 4 vision.
  - 19. Llama-4 Scout 17B-16E: no ratio vs llama.cpp (the ~93 GB download failed once; its state is
    unknown, and it does not fit this machine's 64 GB RAM).

---

---

## Remaining backlog (themed, from the former 00-current-work)

## 1. Correctness of what is already claimed

- [ ] 1. **Remaining vision execution gaps.** CUDA image input still lacks text-decoder M-RoPE image
   positions and deepstack; the server's image path still handles Gemma-style placeholders only.
   Vulkan Qwen3-VL M-RoPE and deepstack are implemented: the real Qwen3VL-2B Q8_0 + mmproj
   `Qwen3VlVulkanMRopeParityTests` ran 2/2 tests in 54.457 s on 2026-10-20 (CPU/Vulkan parity,
   with no-M-RoPE and no-deepstack controls). Remaining CUDA verification is blocked on CUDA
   hardware (none available here); the server routes image input for Gemma 4 only (`ImageContent.cs`: PNG, at most 16 images per request, `<|image|>` placeholder; other VLMs are not routed, and a model without `SupportsImageInput` gets a 400).
   From #6 and #9 in [done/102](done/102-status-open-items-plan.md).
- [ ] 2. **Per-pipeline diffusion end-to-end smoke tests, remaining pipelines.** Done 2026-10-30 for the five image
   pipelines with local weights (SD 1.5, SDXL-Turbo, FLUX.1-schnell, Z-Image-Turbo, SD 3.5 Medium):
   `ImageCommandSmokeTests` runs the real CLI at 256x256 on the CPU and compares a fingerprint (channel means,
   luma std, 8x8 average hash) with `tests/OpenTail.Stingray.Tests.Cli/Baselines/ImageSmoke/`; a missing
   checkpoint fails under `STINGRAY_RUN_HEAVY_TESTS=1`. All five ran bit-identical to the reference, 219 s total.
   Not covered yet: FLUX.2, Qwen Image and the video pipelines (Wan, LTX-Video, HunyuanVideo), which are
   much slower here. The baselines are a regression guard (our own known-good output), not an independent
   reference. Origin: item 4 of §4 in
   [done/2026-09-24-diffusion-perf-session-handoff.md](done/2026-09-24-diffusion-perf-session-handoff.md).
- [ ] 14. **Remaining open items in [1-correctness/bugstofix.md](1-correctness/bugstofix.md)**: #15 LLaVA-NeXT/OneVision residuals
    (pixel-level and per-view embedding parity unverified).
    Closed 2026-10-09: #24 DeepSeek-V2-Lite greedy receipt vs Q3_K kernel (retained e7b7aa8a AVX2 Q3_K dot; characterized token-9 near-tie; explicit receipt in `DeepSeek2GreedyParityTests`; landed IQ4_NL folded MoE path) and #22 stale numpy-golden vision parity tests (retired in favor of active `LlamaMtmdVisionParityTests` oracles).
    Closed 2026-10-01: #19/#20 (RWKV6/RWKV7 admitted) and #23 (DeepSeek-V2-Lite receipt pinned to Q8 prefill).
- [ ] 15. **Stable Audio 3 Small quality**: our local SA3 checkpoints are the `-base` (pre-trained) models, which sound    worse than the post-trained releases. Waiting on the user's listening verdict on the audio.cpp post-trained    clips (`docs/audio-samples/sa3_small_*_POSTTRAINED_*`); if better, add GGUF loading (or gated safetensors via    `HF_TOKEN`) and an opt-in `pingpong` sampler (8 steps, CFG 1.0). Detail: [done/104](done/104-handover-2026-09-28.md) §2.

- [ ] 17. **Qwen3.8-27B (priority checkpoint): MTP self-speculation now wins on CPU: +12% (2.7 vs 2.4 t/s) at draft depth 1.**
    Use `--spec-type mtp --no-thinking --temp 0` (auto mode does not engage it: it also requires no history penalty).
    2026-10-08 history: MTP was silently unreachable from the CLI after the plan-driven refactor (fixed, `RunCommand.cs`);
    once reachable it was *slower* than plain (1.8-2.0 t/s). Measured cause: a verify step cost ~0.26 s + ~0.33 s per extra token
    (k=2 0.92 s, k=4 1.6-1.7 s; plain step 0.42 s) because nothing amortised. Reference ground truth (llama.cpp `-t 8`,
    same file, batch 1/2/4/8 = 2.43/3.44/4.05/4.27 t/s) showed the IQ quant is compute-bound there too, and a microbench
    (`tools/kernel-bench-cs iq3s-multi`, one thread, hot == streaming) showed ~570 of the 680 ns an IQ3_S row costs is weight
    *reconstruction*, repeated per activation. Fixes, each measured and committed:
    (1) shared-decode multi-input kernels (weight group reconstructed once, applied to 2 or 4 activations), bit-identical to
    N single dots: IQ3_S 1.85x/2.70x (N=2/4), IQ4_XS 1.66x/2.37x, plus IQ3_XXS and IQ2_S; verify (64 tok) 12.7 s -> 9.2 s;
    (2) batched GDN projections (`GdnBlockBatch`: qkv/z/ssm_out once for all k tokens, conv + recurrence stay sequential), 9.2 -> 7.4 s;
    (3) IQ3_XXS/IQ2_S kernels 7.4 -> 6.5 s; (4) groups with <=2 real tokens use the 2-input kernels (FFN, batched projections and
    lm_head; pinned bit-identical to the 4-input lanes for Q4_K/Q5_K/Q6_K/IQ formats), k=2 verify 0.92 s -> ~0.58 s (derived from the 2.7 t/s result: cycle 0.65 s minus 0.07 s draft);
    (5) CPU hybrid-GDN default draft depth 1 (`MtpDecoder.ResolveDraftN`).
    **Result (forced MTP, 96 tokens, repeated runs):** depth 1 **2.7 t/s** (76% accepted), depth 2 2.4, depth 3 2.2 (41% accepted;
    the adaptive probe, threshold 55%, then abandons it), plain 2.4. Generated text identical across kernel changes.
    **Remaining headroom:** a k=2 verify is still ~1.4x a plain step (floor ~1.0x = one DRAM pass of 12.2 GiB, ~0.33 s). Left on the
    table: attention layers still run per token (~0.05 s at k=2), lm_head Q5_K 2-input kernel is float (not Q8_K) arithmetic, and
    IQ2_XS/Q3_K/IQ2_XXS FFN tensors (~5% of FFN) have no shared-decode kernel. **Counter-example measured the same day: Qwen3.6-27B UD-Q3_K_XL is SLOWER with MTP (2.0-2.1 vs 2.6 t/s plain, all depths), see PerformanceLeague; cause not diagnosed (quant-mix coverage of the shared-decode kernels is the first suspect).** Evidence for depth 1 is Qwen3.8 only; other
    CPU hybrid-GDN MTP models are unmeasured. The `--spec-draft-n-max` help now states the real default (3, or 1 on the CPU hybrid-GDN pass; fixed 2026-10-29). Plain decode is at ~79% of the DRAM ceiling (~3 t/s), so +12% is most of what is available.
    Reference: llama.cpp k=2 batch costs 1.4 plain steps; ours is now in the same range.
    **Output equality (resolved 2026-10-08): MTP is exact.** With `--repeat-penalty 1.0`, plain greedy and MTP greedy produce
    byte-identical text over 96 tokens (batched verify, depth 1). The earlier apparent divergence ("planet’s surface" vs "planet,
    serving") was the CLI's default repetition penalty 1.1 being applied by plain decoding only: **an explicit `--spec-type mtp`
    skips the history-penalty check that auto mode makes, so it silently ignores `--repeat-penalty`.** Not a numerical bug; the CLI now prints a note when
    `--spec-type mtp` meets a history penalty (`RunCommand.Speculative.cs`). Single/2-input/4-input kernels are pinned bit-identical for
    Q4_K, Q5_K, Q6_K and the IQ formats (`IqSharedDecodeMultiInputTests`, finite data asserted).

Not fixable on this machine (kept 🔴 in STATUS): MiMo-VL (upstream mmproj projects to 3584, the
text model wants 4096), Llama 4 vision (93 GB), MobileNetV5 (no checkpoint declares the projector).

## 2. Model coverage

- [ ] 1. **GLM (`glm4moe`), follow-ups.** GLM-4.5-Air was admitted on 2026-10-04 (Q2_K, CPU and Vulkan layer split;
   STATUS row). Open: other quantisations and the GLM-4.6 / 4.7 checkpoints on real weights, and Vulkan offload
   beyond `-g 4` (every expert of a resident layer is uploaded, so device memory runs out). GLM-4 dense already runs.
- [ ] 2. **Recurrent-state families, follow-ups** (Granite 4.0-H, Nemotron-H and LFM2 were admitted on
   CPU on 2026-09-27; #14, #15, #17 in [done/102](done/102-status-open-items-plan.md)):
   - prefill is batched (`Mamba2Prefill`, `ShortConvPrefill` in `PrefillCore`), and since 2026-10-30 Q8_0 uses the
     int8 batched dispatch: Granite 4.0-H 1B 142 vs llama.cpp 115 t/s, 350M 518 vs 486 (PerformanceLeague, "Q8_0
     batched prefill scheduling"). Remaining: the conv and scan still walk tokens in order (0.65 s of 6.8 s);
   - partial rewind of the recurrent state (`SupportsPartialRewind` is false, so the server's
     prefix cache is off for these models), and no zero KV rows for recurrent layers;
   - GPU paths;
   - untested variants: LFM2-VL/Audio (MoE variants are item 13 above), and the Nemotron-Nano-12B-v2-VL vision tower
     end to end; Falcon-H1 should come almost free.
- [ ] 8. **ONNX pipelines**: SenseVoice and ONNX Paraformer are wired into `stingray stt` (`--model sensevoice` /
    `paraformer`). Open: now that several ONNX pipelines exist, see whether a shared shape is worth extracting.
- [ ] 10. **Newer LTX families** (LTX-2.3 / 2.5), a later campaign.
- [ ] 12. **Lower priority, not planned**: AI21 Jamba, Kimi Linear, Arcee AFM, ServiceNow
    Apriel, Ant Ling (`bailingmoe2`), MiniMax-M2 (too large); DeepSeek-OCR v1 (no checkpoint).
- [ ] 13. **Out of scope for this PC**: DeepSeek-V3.2 / V4, alpha code never run on real weights.
    [reference/058-deepseek-full-lineage-implementation-plan.md](reference/058-deepseek-full-lineage-implementation-plan.md).
- [ ] 14. **TensorSharp coverage wave** (from 2026-10-02; the user's current focus, ahead of the items
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

- [ ] **Architecture semantics in the descriptor** (adding a model = one component, no central edits): plan and live checklist in
  [2-coverage/2026-10-08-architecture-semantics-admission-plan.md](2-coverage/2026-10-08-architecture-semantics-admission-plan.md).

- [ ] 1. **Front door, steps 3-4** (steps 1-2 done: README, catalog, model home, `stingray setup`): starter manifest, README recipes
   around task commands. [3-product-and-runtime/2026-10-09-public-developer-surface-and-front-door-plan.md#front-door-design-merged-2026-10-30).
   2026-10-11 evidence: the README recipes around the task commands landed (root, package and CLI READMEs; their snippets build against the packed package), and `stingray setup chat` then `stingray chat` ran on a clean model home (5 to 6 s total). **Not yet run:** the CLI `speak` and `transcribe` recipes (`setup speak` asks the user to accept a voice data licence; the Speech and QuickStart samples exercised the same engines). Close this item once those two are run and recorded.
- [ ] 6. **Parked** (useful as is, does not move the goal): DSpark speculative decoding, SafeTensors
   Phases 4-6, the ONNX-runtime path for engines that already have native ports.

## 4. Performance

**Paused 2026-10-30 (owner decision).** To resume, start from
[4-performance/2026-10-30-performance-pickup.md](4-performance/2026-10-30-performance-pickup.md): ranked remaining gaps,
what was already tried, untested leads and the measuring rules.

Measurement rules: [reference/performance-measurement-rules.md](reference/performance-measurement-rules.md). An iGPU loss is not evidence against a GPU path
(CLAUDE.md rule 13). Do not reopen the closed Q4_K repacked-GEMM investigation. The cross-model
sweep is [4-performance/perf-sweep-plan.md](4-performance/perf-sweep-plan.md).

**CPU, LLM** (the closed CPU kernel programme is [done/05](done/05-cpu-architecture-kernel-opportunities.md))
   Before new kernel work, read [4-performance/2026-10-30-cpu-quantized-gemm-survey.md](4-performance/2026-10-30-cpu-quantized-gemm-survey.md):
   other projects' int8 GEMMs, and the tinyBLAS register-tile experiment that did not help here.
1. SmolLM2 prefill at ~0.89x of llama.cpp: the Q4_K Path-2 GEMM is 65% of trunk time; then RoPE
   (scalar, ~3%) and attention (~6%). History: "SmolLM2 prefill" in
   [done/101](done/101-work-queue-after-coverage-plan.md).
- [ ] 2. Qwen3.6-35B-A3B prefill at 0.63x of llama.cpp (Phase 8 of
   [done/2026-09-25-hf-top-downloads-coverage-plan.md](done/2026-09-25-hf-top-downloads-coverage-plan.md)).
- [ ] 3. Image-token prefill in VLMs. **Partly done; the "runs per token (about 9 t/s)" wording was stale** (rechecked in code 2026-10-31, not re-measured). The CPU `ForwardPass.PrefillEmbeddings` batches a whole image through the `PrefillCore` trunk, and the CLI's main image path (`RunCommand.Image.cs`) and two tests (`Granite4VisionE2ETests`, `Qwen3VlVulkanMRopeParityTests`) use it. Still per token: (a) the CLI's segment path (`segs`, the LLaVA-NeXT-style per-view layout); (b) the server's Gemma 4 path (`InferenceEngine`), where batching would change nothing today because the batched trunk excludes models with per-layer head dimensions (`_layerHeadDim is null`) and falls back to per-row `ForwardEmbedding`; (c) the Vulkan and CUDA passes, which have no batched embedding entry point (they loop `ForwardEmbedding`). A batched path is not numerically identical to the per-token one (batched prefill quantises activations differently, see [done/029](done/029-prefill-batch-composition-numerics-bug.md)), so any new caller needs a real-checkpoint output comparison, not just a speed number.
- [ ] 4b. int8 alignment with ggml Q8_K (ADR-0003 known gap; per-token int8 matvec 1.2e-2 rel error per
   projection; the Q3_K activation-format question is closed 2026-11-01, see the ADR-0003 addendum: Q8_KS in the batched path is deliberate and more precise; still open: a Q8_KS variant of the repacked Q4_Kx8 GEMM and the MoE activation items), with TensorSharp's managed Q8_K kernels as a second reference. Follow-ups under 10.2
   in [4-performance/perf-sweep-plan.md](4-performance/perf-sweep-plan.md).
  - Folded from todo.md (same numerics thread): Step 2: or a Q8_KS variant of the unified GEMM (keeps our decode precision; no reference
  implementation; the TensorSharp Q8_K kernels are the readable model).
  - Measure (same item): decode t/s, prefill t/s, PPL vs llama.cpp (SmolLM2: F32 0.03%, int8 0.69% today),
  decode-vs-prefill logit identity. Docs: ADR, League rows, perf-sweep-plan 10.2.

**GPU, LLM** ([4-performance/gpu/](4-performance/gpu))
- [ ] 6. Raw-quant matvec bandwidth on Vulkan (gpt-oss decode is MXFP4 matvec-bound, ~18 GB/s here).
- [ ] 7. General multi-`VkBuffer` sharding for tensors beyond `maxStorageBufferRange`: undesigned, not
   needed by any current checkpoint ([done/084](done/084-vulkan-large-tensor-sharding-plan.md)).

**Diffusion** ([4-performance/diffusion/](4-performance/diffusion))
- [ ] 8. FLUX.2 GPU: Experiment 3 (production-shape GEMM ladder) and the `DoubleBlockGpu` row-offset
   audit; the quantized-GEMM lever (corrected 2026-10-31: `VulkanMatMulPath` Path 2 is live for Q4_K and Q6_K LLM weights, and diffusion has quantized GEMMs in `SgemmQ*`; the open work is the FP16-tile project in the [GEMM plan](4-performance/gpu/vulkan-gemm/vulkan-gemm.md#part-2-fp16-tile-project-diffusion), which would help FLUX.1 too). [Part 3 (FLUX.2)](4-performance/diffusion/diffusion-performance.md#part-3-flux2-double-block-plan).
- [ ] 9. FLUX.1 DiT gap to sd.cpp: 132.0 s vs the 99.8 s target (~1.32x) since the 2026-09-25 flash
   attention:
   [Part 2 (FLUX.1 handoff)](4-performance/diffusion/diffusion-performance.md#part-2-flux1-vulkan-gemm-handoff), Phase 6 of
   [Part 1 (umbrella)](4-performance/diffusion/diffusion-performance.md#part-1-umbrella-plan).
- [ ] 10. HunyuanVideo: blocks are compute-bound at ~42% of the iGPU's fp32 peak (the GEMM kernel is the
    lever); the VAE mid-block causal attention is still scalar (measure its share first).
- [ ] 11. Cross-model DRY and perf-doc consistency: Phase 9 of [Part 1 (umbrella)](4-performance/diffusion/diffusion-performance.md#part-1-umbrella-plan).
- [ ] 12. LTX-Video has no C++ comparison yet (sd.cpp's path for it is blocked).

**Audio** ([4-performance/audio/](4-performance/audio))
- [ ] 13. MiniMax-Music3 flow-transformer GPU residency, which matters only on a real GPU (the vocoder gap
    closed 2026-09-28, [done/066](done/066-minimax-music3-future-plan.md)):
    [079](4-performance/audio/079-minimax-music3-gpu-residency-plan.md).
- [ ] 14. MusicGen / AudioGen performance and DRY passes (CFG as a batch-2 GEMM, a T5 kernel shared with
    Parler) plus top-p sampling. "Known gaps" in [done/062](done/062-musicgen-implementation-plan.md).
- [ ] 15. CosyVoice3 ODE step count: the fewer-steps A/B (by ear plus a Whisper round trip) was never run;
    the default is still 10. Item 2 of [done/tts-performance-baseline-and-plan.md](done/tts-performance-baseline-and-plan.md).
- [ ] 16. Stable Audio 3: `IStableAudio3Engine` consolidation now that three variants exist (Sprint 3 of
    [done/065](done/065-stable-audio-3-medium-smallsfx-future-plan.md)).
- [ ] 17. TTS/ASR GPU residency, one engine at a time:
    [080](4-performance/audio/080-tts-asr-gpu-residency-checklist.md). The Vulkan `--backend`
    audits are closed ([done/052](done/052-vulkan-backend-for-tts-engines-plan.md),
    [done/053](done/053-vulkan-backend-audit-remaining-audio-engines.md)); Whisper, Parakeet and
    FunASR encoders are named future candidates there.

- [ ] 20. **Last in the queue: why MTP loses on Qwen3.6-27B (UD-Q3_K_XL).** Measured 2026-10-08, CPU, 96 tokens, greedy: plain
    **2.6 t/s**, MTP **2.0-2.1 t/s** at every draft depth (accept 76% / 62% / 45% at depth 1 / 2 / 3), i.e. 0.77-0.81x of plain.
    The same engine on Qwen3.8-27B (same `qwen35` architecture and MTP head, different quant mix) wins by +12% (2.7 vs 2.4); see
    item 17 and `PerformanceLeague.md`. Not diagnosed. First steps: (a) diff the tensor-dtype histograms of the two files, especially
    the FFN and attention-projection types, and check which have no shared-decode 2-/4-input kernel (IQ2_XS, IQ2_XXS and Q3_K do
    not; IQ3_S, IQ4_XS, IQ3_XXS and IQ2_S do); (b) time one k=2 verify step on both files against a plain step (the Qwen3.8 numbers
    were 0.92 s -> ~0.58 s vs 0.42 s plain); (c) if a missing kernel explains it, add it; if not, check the MTP head's own quant and
    the acceptance rate per position. Until this is understood the CPU MTP draft-depth default (1) is evidenced by Qwen3.8 only, and
    `--spec-type mtp` should be compared with plain on the user's own file.
  - Folded from todo.md: 27B MTP validation: see [reference/adr-0002-hybrid-gdn-mtp-chunked-prefill.md](reference/adr-0002-hybrid-gdn-mtp-chunked-prefill.md) (open follow-ups; owner of the MTP thread).

## 5. Scout, admission tooling and MTP follow-ups (merged from todo.md, 2026-11-01)

Open items from the old `todo.md`, each kept once. Owners: the scout plan for 5a, the MTP ADR and the TensorSharp plan for 5b.

### 5a. Scout and admission tooling

- [ ] Admit ONE real, annoying model with the playbook and write down where scout helped / misled / was irrelevant, with times. Candidates: `ornith-ai/Ornith-1.5-35B-A3B-GGUF`
      (qwen35moe + an extra next-n head: does our loader ignore the unused `nextn` tensors?), a relabel, or the next item in `docs/00-current-work.md`. If it does not save time, fix scout before adding features.

- [ ] Quant picker cost on large-vocab families: ~11 MiB per file (248k-token vocabulary), so a 31-quant repo = 352 MiB / 177 requests. All quants of a repo share their metadata size roughly; learn it from the first file and fetch the exact range in one request. Low priority.

- [ ] **Signature harvest**: remote-scout single-file quants of ADMITTED architectures we have no local file for and `--emit-signature` them (Hub provenance, no download). `qwen3moe` is admitted but ranks against dense `qwen3` only because it has no signature. Prefer single files: split models have no single hash.

- [ ] **Demand-ranked backlog**: `api/models?filter=gguf&sort=downloads&expand[]=gguf&expand[]=downloads` returns architecture, 30-day downloads, gated and total size per repo in ONE call (verified). Aggregate by architecture, mark registered/admitted, rank the unregistered.
      Caveat: downloads are per repository over 30 days, not per file, not inference use. Bounded `--limit`, list-only, never a crawler.

- [ ] **Batch triage**: feed a repo list through remote scout; buckets: identical to an admitted family / identical structure under a new name (relabel candidates, the fast lane, still needs a golden) / near-parent with differences / nothing close / blocked (dtype, tokenizer, size). Machine-readable.

- [ ] **Compare semantics-changing metadata too** (not only tensors): a short curated list (rope scaling type, sliding window, expert gating...) shown as separate informational differences. Not keys that vary with fine-tuning (rope base). qwen3 vs qwen3vl are tensor-identical and differ only in metadata.

- [ ] **Scout output as the bug-report format**: a user whose model fails pastes the JSON (no paths, no weights); maintainer reproduces with `scout -r` at the pinned commit.

- [ ] **Shared goldens index** (file sha256 -> golden): goldens are tiny, hash-pinned, path-free. Checking one needs no llama.cpp; only capturing does. Lets scout say "verified for exactly this file".

- [ ] **Estimator**: MLA (DeepSeek2), hybrid/GDN and RWKV are Unknown (blocked); other MoE families unmeasured (only Phi-3.5-MoE: experts not repacked); only one machine calibrated; GPU placement deliberately not estimated (CLAUDE.md rule 13). Re-measure when the engine changes (the Q4_K repack copy is why a Q4_K_M file costs ~1.8x its size).

- [ ] **Leave-one-out** (`scripts/scout-leave-one-out.ps1`) again whenever signatures grow; still n=12. Observed split: <=11 differences = real near-relative, >=14 = nothing close.

- [ ] **Pretest script gaps**: Ctrl+C mid-run and a hard kill of PowerShell untested (child would survive a hard kill); stage 4 has no command; `admit-arch` exit 1 = fail OR tool error (script classifies by verdict line); split exit codes if other callers allow.

- [ ] Cap/expose index limits and rate behaviour for the Hub (retries, 429), and cache API responses by (repo, commit) if batch features arrive.

### 5b. MTP and numerics follow-ups

- [ ] ggml vs Stingray Q8 activation cross-check on identical rows (the long40 pos-541 projection
      inputs). P/Invoke the vendored b10306 `ggml-base.dll` (`quantize_row_q8_K_ref`,
      `dequantize_row_q8_K`) and `ggml_get_type_traits_cpu` vec_dot (q4_K/q6_K x q8_K).
      - Per 256-block: amax, scale, % blocks identical, differing q per block, max |dq|, dequant RMS
        vs x (ggml Q8_K vs our Q8_KS).
      - Four-way projection decomposition: FP64(W.x) oracle, FP64(W.deq(q_ggml)),
        FP64(W.deq(q_ours)), actual ggml vec_dot vs our dot. Report absolute error next to relative
        (cancellation rows).

- [ ] `MtpDecoder_GreedyParity_LlamaCpp` doc: already diverges from llama.cpp at token 1 on the
      exact path; re-baseline on the fast path when `Qwen3.6-27B-MTP-Q4_K_M.gguf` is available

- [ ] MoE audit and int8 follow-ups: see [1-correctness/2026-10-02-prefill-numerics-investigation.md](1-correctness/2026-10-02-prefill-numerics-investigation.md) (section "Follow-ups from the MoE audit").

- [ ] MTP draft acceptance, StrictSequential vs Optimized prefill, same prompts: accepted per
      round, acceptance by draft position, generated output (identical draft traces not required)

- [ ] llama.cpp b10306 comparison via `llama-server` pre-sampling `n_probs`

- [ ] Optional: name the strict path explicitly (e.g. a `StrictSequential` prefill mode) instead
      of only the env switch

---

## 9. Needs hardware this machine does not have

[9-external-hardware/90-external-hardware-work.md](9-external-hardware/90-external-hardware-work.md):
CUDA release evidence, Gemma 4 12B on CUDA, CUDA gate/up fusion, the CUDA graph default, ARM64.

---

## Thread owners

Items below are grouped by thread. Each thread has one owner doc that holds the full text; other docs point to it. Stage folders (`1-correctness/`, `2-coverage/`, `3-product-and-runtime/`, `4-performance/`) hold the detail.

- **MTP and hybrid-GDN** (owner: [reference/adr-0002](reference/adr-0002-hybrid-gdn-mtp-chunked-prefill.md)): items 17 (section 1) and 20 (section 4).
- **Quantised numerics: Q8_K, int8 and MoE activations** (owner: [1-correctness/2026-10-02-prefill-numerics-investigation.md](1-correctness/2026-10-02-prefill-numerics-investigation.md)): item 4b (section 4).
- **Vulkan GEMM for LLM prefill** (owner: [thread owner](4-performance/gpu/vulkan-gemm/vulkan-gemm.md)): no backlog item of its own; related to item 6 (section 4, GPU). The diffusion FP16-tile project is a separate thread (see the Qwen Image section of the [diffusion owner doc](4-performance/diffusion/diffusion-performance.md)).
- **Qwen Image and diffusion performance** (owner: [thread owner](4-performance/diffusion/diffusion-performance.md)): no backlog item for Qwen Image; items 8 and 9 (section 4, diffusion) are FLUX.
- **Batched MoE prefill** (owner: [thread owner](2-coverage/2026-10-03-batched-moe-prefill-plan.md)): item 5 (section 4, GPU).
- **Ported families** (owner: [2-coverage/ported-families-todo.md](2-coverage/ported-families-todo.md)): item 14 (section 2) and the status ladder.
- **First run, catalogue and public surface** (owner: [thread owner](3-product-and-runtime/2026-10-09-known-good-checkpoints-and-first-run-plan.md)): item 1 (section 3).
- **Gemma** (owner: [thread owner](2-coverage/2026-10-31-gemma-family-plan.md)): items 7 and 9 (sections 1 and 2).
- **Scout and architecture admission** (owner: [thread owner](3-product-and-runtime/2026-10-09-checkpoint-scout-and-ai-admission-plan.md)): open items are in section 5a below (merged from todo.md); the deferred phases are in the plan's thread-state section.
- **OpenTelemetry observability** (owner: [thread owner](3-product-and-runtime/2026-10-12-opentelemetry-observability-plan.md)): no backlog item of its own; its header was corrected on 2026-10-30; the acceptance items remain open.
- **Multi-model runtime** (owner: [thread owner](3-product-and-runtime/032-multi-model-inference-runtime-plan.md)): item 3 (section 3).
- **Documentation architecture** (owner: [thread owner](3-product-and-runtime/2026-10-09-documentation-architecture-plan.md)): no backlog item of its own.
- **Configuration and sessions** (owner: [thread owner](3-product-and-runtime/04-quality-of-life-improvements-plan.md)): no backlog item of its own; plan 010 is planning only.
- **CPU performance** (owner: [thread owner](4-performance/perf-sweep-plan.md)): no backlog item of its own; the pickup doc is the index.
- **Audio GPU residency** (owner: [thread owner](4-performance/audio/080-tts-asr-gpu-residency-checklist.md)): no backlog item of its own; discrete-GPU work is in section 9.

---

## Log

> Historical log entries from 2026-09-27 through 2026-09-28 are archived in [done/103-quickest-first-plan-closed-items.md](done/103-quickest-first-plan-closed-items.md).
