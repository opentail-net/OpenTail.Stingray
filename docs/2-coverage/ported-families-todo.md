# Families to port now, prove later (todo)

> Completed items moved to [done/ported-families-todo-completed-items-2026-10.md](../done/ported-families-todo-completed-items-2026-10.md) on 2026-10-30. This file keeps only open work.

User policy, 2026-10-02: port these families from their reference code now; prove them on real
checkpoints when there is capacity; meanwhile **don't admit and don't advertise them**. The
rules are in CLAUDE.md rule 14 and §4 of
[2026-10-02-tensorsharp-takeaways-plan.md](2026-10-02-tensorsharp-takeaways-plan.md). This file
is the working todo. When a family is ported, add a row to that plan's "Ported, not verified"
table.

Order (user): after the worker-pool experiment, the Q8_K alignment, and `Q1_0` + Bonsai2.

**Single copies (2026-11-08):** the per-family status table lives only in the tensorsharp plan's
"Ported, not verified" table; cross-cutting status, verification levels and the glm/qwen/deepseek
audit history live only in [2026-11-07-coverage-completion-programme-plan.md](2026-11-07-coverage-completion-programme-plan.md);
external sizes, licences and sources live only in [research/2026-10-31-coverage-facts.md](research/2026-10-31-coverage-facts.md).
Nothing is repeated here.

## References

The local llama.cpp source (`examples/llama.cpp/llama.cpp`, git-ignored) was pulled to upstream
**`bed0a8566`** (2026-10-02). It now contains `qwen4exp.cpp`, `muse-glimmer.cpp` and `glm5-next.cpp`
besides `glm-dsa.cpp` and `deepseek4.cpp`. The vendored **binaries** (`tools/llama.cpp`, b10306) are
unchanged and know only `glm-dsa` and `deepseek4`. Every existing receipt and League ratio is pinned
to b10306, so updating them is a separate, deliberate step (keep b10306 alongside, re-run key
receipts on both, then switch).

## Fits this host (summary; detail in the tensorsharp plan's table)

- Muse-Glimmer: yes (small quants). DiffusionGemma: yes (~13-17 GB). MiniMax-H3: yes (sequential loading).
- Qwen 3.8 Flash Next, GLM-5.x, DeepSeek V4: paged only. DeepSeek V4.1: no (does not fit the scratch disk).

## Status ladder (shared definition of done)

`NOT STARTED -> PORTED -> SYNTHETIC VERIFIED -> REAL-WEIGHT VERIFIED -> ADMITTED`

- **PORTED:** production code exists and loads the architecture (behind the not-admitted gate).
- **SYNTHETIC VERIFIED:** structural and specification tests on tiny synthetic models pass. This is
  not real-model evidence.
- **REAL-WEIGHT VERIFIED:** a real checkpoint against an independent implementation, with timed runs.
- **ADMITTED:** text LLMs go through `ModelCompatibility` + `admit-arch` + a STATUS row. **Media and
  diffusion pipelines (DiffusionGemma, MiniMax-H3) are promoted through their own pipeline/CLI
  registry**, not the LLM admission machinery.

PORTED is not SUPPORTED, and SYNTHETIC VERIFIED is not REAL-WEIGHT VERIFIED.

## Verification levels (say which one a check is)

1. **Structural:** shapes, loading, invariants.
2. **Specification:** a test-side reimplementation from the written spec. If the spec was written
   by reading TensorSharp, this catches transcription errors only; it is **not** an independent
   model reference.
3. **Independent implementation:** a second codebase, e.g. llama.cpp source/binaries or HF/PyTorch,
   run on the same synthetic or real inputs.
4. **Real-weight parity:** tokens, logits or intermediates on the real checkpoint.
5. **Admission:** level 4 + timed real-weight evidence + the registry/STATUS update.

Never call a check "independent" when its reference derives from the same source as the port.

**Scope:** this is the TensorSharp coverage wave (started 2026-10-02). Families beyond these six get
a new plan when someone decides to take them on; the wave isn't meant to grow by itself.

## Models that cannot be admitted on this host (2026-10-03)

Policy: a family whose checkpoint cannot be run here stays **written, tested synthetically, and not
exposed** (no admission, no catalog/CLI/STATUS entry). That is the correct end state, not a gap.

- **DeepSeek-V4.1 (`deepseek41`):** does not fit the scratch disk; not attempted. Stays PORTED + SYNTHETIC only.
- **DeepSeek-V3.2 (`deepseek32`):** no cheap shard option (see the programme plan).
- **Real-checkpoint order:** Muse-Glimmer (done 2026-10-03), then GLM-5.3, then DeepSeek-V4, then
  DiffusionGemma. Download into `E:\_models` (or `F:\_models`), verify against the independent
  implementation, then delete the checkpoint.

## Per-family checklist (every family)

- [ ] Read the primary reference first (llama.cpp `src/models/<arch>.cpp` when it exists), then
      TensorSharp's port as a second reading (CLAUDE.md rule 8).
- [ ] Port: hyperparameters, tensor set, forward pass, and tokenizer/chat template if new.
- [ ] Structural/synthetic tests only, named as such.
- [ ] `// <arch> — NOT admitted` block in `src/OpenTail.Stingray.Engine/ModelCompatibility.cs`:
      what was ported, from which reference (file and commit), and what verification is missing.
      *(Checked 2026-11-08: no family has this block; the NOT-admitted notes live in the `*Alpha.cs`
      header comments. Either add the blocks or change this rule.)*
- [ ] Row in the plan's "Ported, not verified" table (date, code location, reference).
- [ ] Ported code from TensorSharp goes into the TensorSharp section of `THIRD_PARTY_NOTICES.md`.
      *(Checked 2026-11-08: present for qwen4exp, MiniMax-H3 and Muse-Glimmer; not for glm-dsa or deepseek4.)*
- [ ] Nothing in STATUS.md, README, WHAT-YOU-CAN-DO, RUNNING or the catalogs.
- [ ] Later, for promotion (see the status ladder): a real checkpoint, an independent implementation
      (`stingray admit-arch` against llama.cpp for text LLMs; pipeline fixtures for media), timed
      real-weight runs, then the admission/registry entry and the STATUS row in the same pass.
- [ ] Each plan carries a **Deferred** list (vision, speculative heads, batched/GPU paths, ...) so the
      initial port doesn't grow.

## Families

### 1. Qwen 3.8 Flash Next (`qwen4exp`)
- **Design** (from TensorSharp's card):
  - hybrid MoE: GatedDeltaNet recurrent layers interleaved with full-attention layers, some behind
    Qwen Sparse Attention's indexer;
  - a PLE n-gram embedding block;
  - x4 hyper-connection streams;
  - 512-expert MoE;
  - image input through an mmproj.
- **Weights:** `unsloth/Qwen3.8-Flash-Next-GGUF` (~180B total; smallest quant ~72.5 GB; synthetic gate only on this host, see the programme plan for the paged run).
- **Reuse:** Stingray's `GdnKernels` (GDN FP32 recurrence), MoE primitives (512 experts, top-10 + 1 shared); distinct `GatedResidual` (HC) and QSA/PLE subsystems.
- **References:** llama.cpp `src/models/qwen4exp.cpp` (local source), TensorSharp `Models/Qwen4Exp`.
- [x] Real-weight verification, one prompt (paged execution, 2026-11-07; details in the programme plan).
- [ ] Broader real-weight verification before promotion: more fixed prompts, incremental decode, a
      prompt crossing a prefill/decode chunk boundary, QSA K-pool exercise, a documented tolerance.
      MTP and vision stay deferred.

### 2. GLM-5.x (`glm-dsa`, alias `glm_dsa`; GLM-5.3-Flash is `glm5next`)
- **Design:**
  - `glm-dsa` (GLM-5.2/5.3): 744B MoE (256 routed + 1 shared, top-8, sigmoid, scale 2.5; 3 leading dense) on MLA (64 heads, 512+64 latent row) + 32-head DSA indexer with orthonormal Sylvester Hadamard rotation (`PrismHadamard` FWHT); shared indexer layers reuse top-k.
  - `glm5next` (GLM-5.3-Flash): 320B hybrid trunk with 45 layers (34 KDA linear attention + 11 NoPE MLA/DSA in strict 3:1 schedule: `i % 4 != 3` -> KDA, `i % 4 == 3` -> MLA), 4-token K-pool indexer, x4 Sinkhorn mHC, 288+1 MoE with FP32 router, SwiGLU clamp at 10.
- **Weights:** `unsloth/GLM-5.3-GGUF` (UD-Q2_K_XL; does not fit 64 GB RAM; synthetic gate only on this host).
- **Reuse:** Stingray's `deepseek2` MLA, `deepseek32` indexer, `PrismHadamard` FWHT, and `deepseek4` mHC.
- **References:**
  - `glm-dsa`: llama.cpp `src/models/glm-dsa.cpp` (local source; vendored b10306 binaries serve as local oracle), HF `GlmMoeDsaForCausalLM`, TensorSharp `Models/GlmDsa`.
  - `glm5next`: current upstream llama.cpp `src/models/glm5-next.cpp` (`bed0a8566`), HF `Glm5NextForConditionalGeneration`, TensorSharp `Models/GlmDsa`.
- **Status:** the synthetic-correctness checklist for `glm-dsa` is complete (2026-11-08). Its full
  audit history (bugs fixed, tests added, the metadata-only real-weight check) is in the programme
  plan's "What changed" section and its Checklist, not repeated here.
- [ ] `glm-dsa` real-weight parity: out of reach on this host's disk budget; rescoped to hyperparameter parsing (done 2026-11-07).
- [ ] `glm5next` K-pool selection gating attention reads (Phase 2B of the programme plan).

### 3. Muse-Glimmer (`muse-glimmer`, alias `muse_glimmer`)
- **Design:**
  - text + image; thinking through an `assistant to=self` channel;
  - llama.cpp's standard SWA window predicate and a hard-coded 1e-8 (per TensorSharp's card);
  - DFlash speculative drafter (optional).
- **Status:** admitted 2026-10-03 (text, CPU only); see the Muse-Glimmer plan and the STATUS row.
- **Weights:** Muse-Glimmer-30B; GGUF sizes on the card are about 3-10 GB for the parts listed.
- **References:** TensorSharp `Models/MuseGlimmer` (cites `llama.cpp/src/models/muse-glimmer.cpp`,
  present in the local copy at `bed0a8566`).
- [ ] Port vision tower (deferred)

### 4. DiffusionGemma (`diffusion-gemma`, alias `diffusion_gemma`)
- **Design:**
  - block **text diffusion** on Gemma-4 MoE backbone (26B total, ~4B active: hidden 2816, dense FFN 2112 + 128 experts top-8);
  - 30 layers: 5 Full attention (Q=16, KV=2, dim=512) + 25 Sliding attention (Q=16, KV=8, dim=256, window=1024);
  - **Self-conditioning**: step 1 seed, $t > 1$ soft probability weighted embeddings through learned SC MLP injected into canvas;
  - Canvas denoising (256 tokens) with bidirectional canvas attention + persistent causal prefix KV cross-attention;
  - Sampler: temperature decay, nats entropy budget (0.1), greedy acceptance, categorical re-noising;
  - Block-autoregressive lifecycle: commit 256 tokens, causal re-prefill, next canvas.
- **Weights:** `google/diffusiongemma-26B-A4B-it` (HF); `unsloth/diffusiongemma-26B-A4B-it-GGUF` (Q4_K_M, see the research file for sizes).
- **Reuse:** Stingray's Gemma 4 primitives, MoE routing, and PagedKvCache.
- **References:** HF reference implementation, TensorSharp `Models/DiffusionGemma`, Unsloth / llama.cpp DiffusionGemma runner.
- [ ] Real-weight verification (Phase 9/13 ladder against llama.cpp and TensorSharp on `diffusiongemma-26B-A4B-it-Q4_K_M.gguf`)

### 5. MiniMax-H3 (video + native 32 kHz stereo audio)
- **Design:**
  - one DiT denoises a packed video+audio latent;
  - seven graphs: text encode, vision encode, DiT, and encode + decode for each of the two VAEs;
  - up to 15 s at 24 fps.
- **Weights:** `MiniMaxAI/MiniMax-H3`, `unsloth/MiniMax-H3-GGUF`, `Comfy-Org/MiniMax-H3` (sizes and licence in the research file).
- **Reuse:** Stingray's diffusion pipeline (Wan, HunyuanVideo, LTX) and audio VAE code. A diffusion project, not an LLM arch.
- **References:** TensorSharp `Models/MiniMaxH3`, upstream HF code.
- [ ] Real-checkpoint parity and remaining items: see the MiniMax-H3 plan's open checklist.

### 6. DeepSeek V4 / V4.1 Flash (V4 review + V4.1 distinct architecture)
- **Design:**
  - V4 Flash (`deepseek4`): 43 layers, hidden 4096, 64 heads, head dim 512, compression ratios 0/4/128, 256+1 MoE (top-6, sqrtsoftplus), x4 mHC, 64x128 indexer top-512, 1 NextN.
  - V4.1 Flash (`deepseek41`): distinct architecture with 40 layers, hidden 5120, compression ratios 0/1/2, 384+1 MoE (scale 1.5), dual ~196B Engram tables (layers 1 & 14, 4-gram, 99k compressed vocab), 8 index-source layers, candidate top-2048 x block 8, 3 NextN layers, DSpark (layers 37-39).
- **Weights:** V4 Flash Q2_K ~98.6 GB; V4.1 Flash ~335 GB (sizes and sources in the research file). Neither fits 64 GB RAM.
- **Reuse:** Stingray's existing `DeepSeek4ForwardPass`, `DeepSeek4CompressedState`, `DeepSeek4Alpha`.
- **References:**
  - V4: llama.cpp `src/models/deepseek4.cpp` (local source + vendored b10306 as local oracle).
  - V4.1: TensorSharp `Models/DeepSeek4/` (`DeepSeek41Model.cs`, `DeepSeek4CpuExecutor.V41.cs`, `Dsv41EngramData.cs`) validated under 103-case PyTorch oracle (100/103 passed).
- **Status and open phases:** see [2026-10-03-deepseek-v4-review-plan.md](2026-10-03-deepseek-v4-review-plan.md). Its inline `[x]` gate claims need the code check noted in the programme plan (no `NOT admitted` block in `ModelCompatibility.cs`).

## Open items moved from 058-deepseek-full-lineage-implementation-plan.md (2026-10-30)

- [ ] *(blocked 2026-11-03: no DeepSeek V1 checkpoint on C, E, F or H; only DeepSeek-V2-Lite and deepseek-ocr-2 are present.)* Port `deepseek` (V1) — plain dense transformer, no MLA/MoE. Expected near-free once `deepseek2`'s
  dense-layer code path (`leading_dense_block_count` layers) is confirmed working, since V1 is
  architecturally a subset.

- [x] *(done 2026-11-03: the `deepseek2-ocr` descriptor (`DeepSeek2Architectures.cs`) and the vision encoder (`DeepSeekOcr2VisionEncoder`, `UnifiedVisionPipeline`) reference none of the `DeepSeek4`/`DeepSeek32`/`DeepSeek41` types, which live in their own forward-pass classes; and on current main the real-weight SAM parity tests against `llama-mtmd-debug` pass: `DeepSeekOcr2_Rainbow1024_SamMatchesLlamaMtmdDebug` 14.8 s and `DeepSeekOcr2_Rainbow768Tile_SamMatchesLlamaMtmdDebug` 5.5 s with `mmproj-deepseek-ocr-2-q8_0.gguf`.)* Audit `deepseek2-ocr` only, to confirm nothing in Phases 0-2 silently affects the
  existing working vision branch. Not a rebuild.

---

## Research findings

Moved to [research/2026-10-31-coverage-facts.md](research/2026-10-31-coverage-facts.md) (sizes, licences, llama.cpp logprob fields, open checks). Not repeated here.
