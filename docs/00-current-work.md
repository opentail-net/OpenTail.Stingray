# Current work: outstanding items, quickest first

Merged 2026-11-01 from the former themed backlog (this file) and `done/103-quickest-first-plan-merged-2026-11-01.md` (the quick-first working order, now the order of this file). The order is the quick-first order below. Each item appears once. Items from the themed backlog that have no quick-list entry follow in section order under "Remaining backlog". Completed items are in [done/](done) and their evidence is in each model's STATUS row.

## Open work (as of 2026-11-03)

As of 2026-11-05. Start with **Start here**: nothing in it waits on the owner. Items under **Needs a decision** need a scope call first. **Owner action** is separate and gates nothing above it. Evidence for finished items is in [done/open-work-evidence-2026-10-31-to-11-01.md](done/open-work-evidence-2026-10-31-to-11-01.md).
**Working rules:** one merged PR per item; `Tests.Core` before every merge; a perf claim needs a quiet machine, interleaved A/B, 3+ runs and equal output; a new item must name a failure, a user-visible limit, a measurable gain or a recurring cost. A stalled item is documented and the next one starts.

## Start here (no owner input needed)

- [ ] **2. Gemma 4 26B-A4B port, fused experts** (`ffn_gate_up_exps`, parallel dense FFN plus 128 experts). Runs on the CPU since 2026-11-03 (`Gemma4MoeGreedyParityTests`, STATUS row yellow). Open: greedy parity against llama.cpp holds on 2 prompts but 4 of 5 further 96-token prompts diverge (divergence index not measured), and there is no GPU path. Reuse `Diffusion/DiffusionGemma/DiffusionGemmaForwardPass.cs`, which already implements the layer. References on `E:\_models`: the stock bartowski Q4_K_M in `gemma4-26b-a4b` (stock llama.cpp b11556 runs it), and `diffusiongemma\dg-backbone-as-gemma4.gguf`. Done when a real-weight greedy and logit comparison against llama.cpp passes, timed, with a STATUS row. Not admitted before that.
  First step: measure the first divergence index on the four diverging prompts, against llama.cpp, before changing the layer.

## Needs a decision before work

- [ ] **6. Handoff gaps, Vulkan hybrid:** LongRoPE, sliding-window llama, contexts above 4096, Qwen2.5-7B / Coder / VL. No owner decision recorded. Each needs a passing parity row or a recorded refusal.
- [ ] **7. Handoff receipts, remaining families:** Phi-3 has no path for fused `attn_qkv` in the hybrid, so it is refused. Granite's scale fields are applied on the hybrid now (P2c, finished); its handoff receipt is not recorded yet. Gemma 2 stays Unverified because soft-capping is not on GPU (owner: no). Decide per family: add the path, or keep the refusal and record it.

## Evidence for the engine (added 2026-11-05, from the external review)

Items 1 to 5 are open and need no owner input. Pick them in order. Each finishes as a merged PR with its own test or report, and its own STATUS or backlog line. Do not start a model-family port, a UI, or a second backend.

- [x] **E1. Long-running inference stability.** Done 2026-11-05: `LongRunSoakTests` (opt-in, `STINGRAY_SOAK=1`) passed on real Qwen3-0.6B Q8_0 on CPU, 30 greedy repeats in one process, identical output, working set flat after warm-up. Repeated prompts, long contexts and several sessions, looking for memory growth, stale state, and KV-cache corruption over time. Done when a soak run at temperature 0 gives identical output on every repeat, and resident memory is flat after warm-up.
  First step: write a soak test on the fake engine in `tests/OpenTail.Stingray.Tests.Core` that loops N prompts and asserts identical output and flat allocations. Then the same loop against a real small model on CPU.
- [x] **E2. CPU performance regression gate.** Done 2026-11-05, as a local gate: `scripts/perf-gate.ps1` measures prefill and decode for qwen2.5-0.5b and 1.5b (median of 3) and fails on a drop of more than 15% against `scripts/perf-baseline.json`, per machine. Hosted CI has no model files, so it cannot run this gate; run it before a release or after a kernel change. Checked: unchanged code passes (exit 0); a baseline inflated 2x fails all four metrics (exit 1).
- [ ] **E3. Model admission reliability.** Scout, pull and admission against malformed, incomplete, incompatible and interrupted downloads. Failure messages must say what to do. Done when each failure class has a test that checks its message.
  First step: list the existing pull and download tests, then add the interrupted-download case first.
  Progress 2026-11-05: `ModelDownloaderResumeTests` (3 tests) covers a server that ignores Range (no doubled bytes), a 416 on a complete partial, and a 503 that keeps the partial for the next attempt. A mutation that appends on a 200 fails the first test. Already covered before: resume, hash mismatch, stale revision, offline (`ModelCatalogTests`), interrupted first run (`FirstRunTests`). Still open: the message wording for each failure class, and malformed or truncated listings from scout and pull.
- [ ] **E4. Memory and resource accounting.** Check that the RAM and VRAM the catalog and planner advertise match what the runtime peaks at, and that unloading and cancellation release memory. Done when a table of advertised against measured peak exists for the supported models.
  First step: pick one small CPU model from the catalog and measure its peak against its advertised figure.

  Measured 2026-11-05 on the Ryzen 7 5700G, CPU only (`-g 0`), Release CLI, greedy, peak working set of the process:

  | Model (catalog id) | Advertised | Measured peak | Tokens generated |
  |---|---|---|---|
  | qwen2.5-0.5b (Q4_K_M, 491 MB file) | about 1 GB | 835 MB | 512 (745 MB at 64) |
  | qwen2.5-1.5b (Q4_K_M, 1.07 GB file) | about 2.5 GiB | 2.04 GB | 512 |
  | qwen2.5-7b (Q4_K_M, first shard 3.8 GB) | about 9 GiB | 7.9 GB | 64 |

  All three advertised figures are at or above the measured peak. Still open: speech models (piper-lessac, whisper-base), unloading and cancellation releasing memory in the multi-model server, and the 7B run across all shards.

- [x] **E5. API compatibility corpus.** Done 2026-11-05: protocol-smoke now has 63 cases, including streaming, tool definitions, wrong shapes and malformed bodies on chat, messages and responses. Cancellation is not in the corpus (the harness reads whole bodies); `ClientDisconnectTests` covers it. Extend `protocol-smoke.json` with streaming, cancellation, tool-call and error-format cases for Chat Completions, Responses and Anthropic. Done when every case has a recorded status and error type.
  First step: list the Anthropic and Chat Completions streaming cases not yet in the corpus.

Deferred (not in this batch): Vulkan parity expansion, telemetry usefulness, catalogue provenance, documentation usability test. They stay in their plans.

## Owner action (does not block the list above)

- [ ] **Run once:** `pwsh scripts/qualify-tasks.ps1 -AcceptLicence`. It accepts the research-only Piper voice licence. Until then the speak and transcribe tasks show SKIP. The assistant cannot accept a licence for the owner.
- [ ] **Release 1.0.10:** not ready. No tag, no version bump, no pack for publication. Everything since 1.0.9 stays unpublished until the owner says otherwise.

## Pointers to work owned elsewhere

- **3. DiffusionGemma and Bonsai2 verdicts** (after item 2). Owned by [unverified-port-claims](1-correctness/2026-10-03-unverified-port-claims.md) (DiffusionGemma parity ladder, Bonsai2 PRISM) and [ported-families-todo](2-coverage/ported-families-todo.md). Stays internal (rule 14).
- **4. HunyuanVideo and Wan 2.2 findings.** Owned by [reconciliation plan, R2](1-correctness/2026-10-09-documentation-claim-reconciliation-plan.md) (owner decision: duplicate first). Per-variant boxes also in [diffusion-performance](4-performance/diffusion/diffusion-performance.md).
- **5. Gemma 2 beyond the 2B file.** Owned by [gemma family plan](2-coverage/2026-10-31-gemma-family-plan.md). Gemma 1 is done (STATUS row). Gemma 2 9B needs a download to E: or H:, never F:.
- **8. OpenTelemetry acceptance.** Owned by [OpenTelemetry plan](3-product-and-runtime/2026-10-12-opentelemetry-observability-plan.md) (its open acceptance boxes). Test files exist; map each box to a test, tick it, or write why it is not met.

## Blocked or parked

- **Release verification on a clean machine** (Q6): needs a second machine. The NativeAOT self-test in CI covers the AOT binary only.
- **CUDA and discrete-GPU work** (P2d, CUDA receipts, discrete-GPU measurements, the external-hardware queue): no NVIDIA hardware here. All CUDA items are out of scope for this file.
- **AVX-512 and ARM64 execution receipts:** no such machine. The claim is corrected in the docs instead.
- **TTS/ASR GPU residency and diffusion GPU performance:** every number is an iGPU measurement, and rule 13 forbids generalising it. They wait for a discrete GPU.
- **Ported-but-unverified families** (DeepSeek-V4, GLM-5.x, MiniMax-H3, DiffusionGemma, Qwen3.8-Flash-Next, the rest of `ported-families-todo.md`): they need checkpoints of 70 GB or more, or none exists here. Rule 14 keeps them internal.
- **Parked by owner decision:** the HTTP image endpoints stay 501 (`/v1/images/generations` and `/edits` are not connected to the real pipelines); Gemma 2 GPU soft-capping (no; Gemma 2 is CPU only); the Qwen3-Coder NaN hunt (canary only).
- **Qwen3-Coder NaN canary** (owner: stop hunting): keep one canary line in the Vulkan hybrid MoE test; confirm it exists, add it if not. Reopen the investigation only if the canary fires.

## Open boxes in other docs

These are not restated here. Each file is the owner of its list.

| Document | Open boxes | Note |
|---|---:|---|
| `docs/00-current-work.md` (this file) | 64 | the active backlog, including the open-work list above |
| `docs/4-performance/audio/080-tts-asr-gpu-residency-checklist.md` | 31 | rule 13: needs a discrete GPU |
| `docs/9-external-hardware/90-external-hardware-work.md` | 20 | no hardware here |
| `docs/4-performance/diffusion/diffusion-performance.md` | 19 | rule 13 |
| `docs/4-performance/perf-sweep-plan.md` | 16 | CPU items need a quiet machine |
| `docs/2-coverage/ported-families-todo.md` | 14 | rule 14 |
| `docs/4-performance/gpu/vulkan-gemm/vulkan-gemm.md` | 11 | rule 13 |
| `docs/4-performance/2026-10-10-inference-improvement-leads-plan.md` | 11 | |
| `docs/2-coverage/2026-10-03-deepseek-v4-review-plan.md` | 11 | no checkpoint fits |
| `docs/3-product-and-runtime/04-quality-of-life-improvements-plan.md` | 10 | |
| `docs/reference/nuget-release-checklist.md` | 9 | publishing is the owner's call |

The rest have 9 or fewer each. Find them with: `grep -rc "^\s*[-*+] \[ \]" docs`.


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
  - [ ] Admit Nemotron-H MoE (latent MoE, sigmoid gating) with PPL parity. No local checkpoint (2026-09-28); not downloaded (C: has no free space, K: ~10 GB).
  - [x] *(stale, closed 2026-11-03: `lfm2moe` was admitted 2026-10-01; the batched drift was flash-64 attention's online softmax, not a batched-path bug; see STATUS and [bugstofix item 12](done/bugstofix-resolved-2026-10.md))* Admit LFM2-MoE (`lfm2moe`) with PPL parity. BLOCKED 2026-09-28 (timeboxed): PPL +6.3% and per-token vs batched disagree by 9%; see bugstofix.
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
    **Re-measured 2026-11-03 (after the Q3_K shared-decode kernels and the Q3_K width-independence fix, plan 8 B7):** Qwen3.8-27B-UD-Q3_K_XL, CPU `-g 0`, 96-token request that stopped at 62 tokens, greedy, `--repeat-penalty 1.0 --no-thinking`, interleaved, 3 runs each, nothing else running: plain (`--spec-type none`) **2.4 t/s** in all runs, MTP at depth 1 **2.8 t/s** in all runs (88% of drafts accepted), about +17% (was 2.7 t/s, +12%, on 2026-10-08); text identical. The file has 12 Q3_K tensors, so it gained a little from the new kernels. The remaining tensors without a shared-decode kernel are the 4 IQ2_XS and 2 IQ2_XXS FFN tensors (a few percent of the FFN).
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
   **Run 2026-11-03 (plan 8 B5): `transcribe` recipe.** With `STINGRAY_MODEL_HOME` pointing at an empty folder, `stingray setup transcribe -y` downloaded and verified Whisper base (141.1 MiB) in 8.4 s and printed `Ready: whisper-base (transcribe)`; `stingray transcribe docs/audio-samples/piper-lessac-sample.wav` returned "Hello from Piperon Penial Tingad." in 1.20 s for 2.39 s of audio (2.0x real time), the known mis-hearing of the coined word "OpenTail Stingray" described in the audio-samples README. **Still not run: `speak`**, because `stingray setup speak` asks the user to accept the Piper voice data licence and that acceptance is the owner's to give (`--accept-licence` exists for it). Close this item after the owner runs or waives `speak`.
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
- [ ] 3. Image-token prefill in VLMs. **Partly done; the "runs per token (about 9 t/s)" wording was stale** (rechecked in code 2026-10-31, not re-measured). The CPU `ForwardPass.PrefillEmbeddings` batches a whole image through the `PrefillCore` trunk, and the CLI's main image path (`RunCommand.Image.cs`) and two tests (`Granite4VisionE2ETests`, `Qwen3VlVulkanMRopeParityTests`) use it. Still per token: (a) the CLI's segment path (`segs`, the LLaVA-NeXT-style per-view layout); (b) the server's Gemma 4 path (`InferenceEngine`), where batching would change nothing today because the batched trunk excludes models with per-layer head dimensions (`_layerHeadDim is null`) and falls back to per-row `ForwardEmbedding`; (c) the Vulkan and CUDA passes, which have no batched embedding entry point (they loop `ForwardEmbedding`). A batched path is not numerically identical to the per-token one (batched prefill quantises activations differently, see [done/029](done/029-prefill-batch-composition-numerics-bug.md)), so any new caller needs a real-checkpoint output comparison, not just a speed number.
- [ ] 4b. int8 alignment with ggml Q8_K (ADR-0003 known gap; per-token int8 matvec 1.2e-2 rel error per
   projection; the Q3_K activation-format question is closed 2026-11-01, see the ADR-0003 addendum: Q8_KS in the batched path is deliberate and more precise; still open: a Q8_KS variant of the repacked Q4_Kx8 GEMM and the MoE activation items), with TensorSharp's managed Q8_K kernels as a second reference. Follow-ups under 10.2
   in [4-performance/perf-sweep-plan.md](4-performance/perf-sweep-plan.md).
  - Folded from todo.md (same numerics thread): Step 2: or a Q8_KS variant of the unified GEMM (keeps our decode precision; no reference
  implementation; the TensorSharp Q8_K kernels are the readable model).
  - Measure (same item): decode t/s, prefill t/s, PPL vs llama.cpp (SmolLM2: F32 0.03%, int8 0.69% today),
  decode-vs-prefill logit identity. Docs: ADR, League rows, perf-sweep-plan 10.2.

**GPU, LLM** ([4-performance/gpu/](4-performance/gpu))
- [ ] 5. Batched prefill for partially offloaded dense models and the remaining per-token paths. **Rechecked 2026-10-31; the earlier wording ("`VulkanLayerSplitForwardPass` and the MoE path prefill per token") was wrong in both directions.** `GpuForwardPass` batches dense Q4_K/Q6_K models; `VulkanLayerSplitForwardPass` already hands long fresh prompts to the CPU batched pass and imports the KV (glm4moe and afmoe have receipts); the gap that was real was a **dense `llama`-architecture model with a partial offload (`-g N`, `HybridForwardPass`)**, which had no handoff receipt and prefilled token by token (SmolLM2-1.7B `-g 12`: 18 tok/s against 227 CPU-only). Closed 2026-10-31 by a `llama` receipt on the Vulkan hybrid: 18 -> 180 tok/s there and 7.7 -> 44-49 tok/s for Mistral-7B `-g 4` (see the STATUS row for the evidence, the Falcon3 `-g 8` deviation; the one-time 1.9 s warm-up of the first decode was moved to model load the same day, PR 73). Still per token: prompts under 32 tokens, `startPos > 0` (later chat turns), families without a receipt (dense llama on the full-GPU and layer-split paths, other dense architectures: Qwen2.5 has a hybrid receipt since 2026-10-31; Gemma 2 is CPU-only by design (its soft-capping is not in the GPU passes, so the planner keeps it on the CPU and says so); Granite and Gemma 3 are wrong on the Vulkan hybrid because it applies none of the model scale fields, so it now refuses them (plan 2 P2c); Phi-3 cannot run on the hybrid), and the MoE paths without a handoff receipt ([batched MoE plan](2-coverage/2026-10-03-batched-moe-prefill-plan.md), [Vulkan GEMM doc](4-performance/gpu/vulkan-gemm/vulkan-gemm.md)).
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
- 17. TTS/ASR GPU residency, one engine at a time (boxes live in the checklist, not here):
    [080](4-performance/audio/080-tts-asr-gpu-residency-checklist.md). The Vulkan `--backend`
    audits are closed ([done/052](done/052-vulkan-backend-for-tts-engines-plan.md),
    [done/053](done/053-vulkan-backend-audit-remaining-audio-engines.md)); Whisper, Parakeet and
    FunASR encoders are named future candidates there.

- [x] 20. **Last in the queue: why MTP loses on Qwen3.6-27B (UD-Q3_K_XL).** Measured 2026-10-08, CPU, 96 tokens, greedy: plain
    **2.6 t/s**, MTP **2.0-2.1 t/s** at every draft depth (accept 76% / 62% / 45% at depth 1 / 2 / 3), i.e. 0.77-0.81x of plain.
    The same engine on Qwen3.8-27B (same `qwen35` architecture and MTP head, different quant mix) wins by +12% (2.7 vs 2.4); see
    item 17 and `PerformanceLeague.md`. Not diagnosed. First steps: (a) diff the tensor-dtype histograms of the two files, especially
    the FFN and attention-projection types, and check which have no shared-decode 2-/4-input kernel (IQ2_XS, IQ2_XXS and Q3_K do
    not; IQ3_S, IQ4_XS, IQ3_XXS and IQ2_S do); (b) time one k=2 verify step on both files against a plain step (the Qwen3.8 numbers
    were 0.92 s -> ~0.58 s vs 0.42 s plain); (c) if a missing kernel explains it, add it; if not, check the MTP head's own quant and
    the acceptance rate per position. Until this is understood the CPU MTP draft-depth default (1) is evidenced by Qwen3.8 only, and
    `--spec-type mtp` should be compared with plain on the user's own file.
    **Update 2026-11-03 (plan 8 B7), steps (a) and (c) done, MTP still not a win on this file.** (a) Tensor-dtype histograms (`stingray list-tensors`): Qwen3.6-27B-UD-Q3_K_XL is a K-quant mix (Q4_K 185, Q3_K 136 tensors including 110 FFN gate/up, Q6_K 52, Q5_K 24, IQ3_S 10), Qwen3.8-27B-UD-Q3_K_XL an IQ mix (IQ4_XS 156, IQ3_S 111, Q8_0 98, IQ3_XXS 34, ...). `MatVec2In` and `MatVec4In` had no Q3_K case, so Q3_K fell back to one full weight pass per token (the first suspect above was right). (c) Added `DotQ3K_Q8K_2In`/`_4In` (weight block and scales loaded once, per-input arithmetic identical to `DotQ3K_Q8K`, so bit-identical to the single-input dot; `IqSharedDecodeMultiInputTests.Q3K_MatVec2In_And_MatVec4In_EqualIndependentMatVecs_Bitwise`, mutation-checked) and the two dispatch cases. **Measured** (CPU `-g 0`, 8 threads, Ryzen 7 5700G, 96-token request that stopped at 53 tokens, greedy, `--repeat-penalty 1.0 --no-thinking`, interleaved before/after, 3 runs each, nothing else running): plain decode (`--spec-type none`) **2.6 t/s** in all three runs; MTP before the change **2.0, 2.1, 2.1** t/s, after **2.3, 2.3, 2.3** t/s (about +12%), 83% of drafts accepted, generated text identical in all three configurations. So the shared-decode gap was real and is closed, but MTP is still about 0.88x of plain on this file: with 83% accepted a cycle yields 1.83 tokens, so it must cost under 1.83 plain steps (0.70 s) and costs about 0.80 s. Still open: where the remaining ~0.1 s per cycle goes (step (b), k=2 verify against a plain step per layer type: attention and GDN layers still run per token, the lm_head, Q4_K/Q5_K/Q6_K tensors already use the 2-input kernels). Note for later measurements: with `--repeat-penalty 1.0` the default `--spec-type auto` engages MTP on this model, so "plain" needs `--spec-type none`.
    **Update 2 2026-11-03 (plan 8 B7): closed, MTP now beats plain on this file.** The batched verify only uses the cheaper 2-input FFN for a group of at most two real tokens when `IsWidthIndependentMultiInput` holds for the layer's gate, up and down types (otherwise it pads the group to four inputs and computes two duplicate lanes); Q3_K was not on that list, so every k=2 verify ran the padded 4-input FFN on the 110 Q3_K gate/up tensors. Q3_K is now on it (its 2-input result is bit-identical to a 4-input lane and to the single-input dot: `TwoInput_Equals_FirstTwoLanesOfFourInput_Bitwise` and `Q3K_MatVec2In_And_MatVec4In_EqualIndependentMatVecs_Bitwise`). **Measured** (same setup as above, interleaved, 3 runs each, nothing else running, generated text identical to plain): MTP after the first fix 2.3, 2.2, 2.3 t/s; MTP now **2.7, 2.8, 2.7 t/s**; plain 2.6 t/s. So MTP is about +4 to +8% over plain on Qwen3.6-27B-UD-Q3_K_XL at draft depth 1 (83% accepted), against 2.0-2.1 t/s (0.8x) before 2026-11-03. The default depth 1 is therefore now evidenced on two files (Qwen3.8 and Qwen3.6).
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

- [ ] Cap/expose index limits and rate behaviour for the Hub, and cache API responses by (repo, commit) if batch features arrive. *(Retries done 2026-11-03, plan 8 B4: `ExternalHttpClient.GetAsync` retries 429 and 503 up to three times, waiting for `Retry-After` (capped at 30 s) or 1, 2, 4 s, then hands the last response to the caller; `ExternalHttpClientRetryTests`, mutation checked. Still open: exposing the limits in scout output and the (repo, commit) response cache, both only if batch features arrive.)*

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
