# Families to port now, prove later (todo)

User policy, 2026-10-02: port these families from their reference code now; prove them on real
checkpoints when there is capacity; meanwhile **don't admit and don't advertise them**. The
rules are in CLAUDE.md rule 14 and §4 of
[2026-10-02-tensorsharp-takeaways-plan.md](2026-10-02-tensorsharp-takeaways-plan.md). This file
is the working todo. When a family is ported, add a row to that plan's "Ported, not verified"
table.

Order (user): after the worker-pool experiment, the Q8_K alignment, and `Q1_0` + Bonsai2.

## Per-family plans (one each, 2026-10-03)

References (updated 2026-10-03): the local llama.cpp source (`examples/llama.cpp/llama.cpp`,
git-ignored) was pulled to upstream **`bed0a8566`** (2026-10-02). It now contains `qwen4exp.cpp`,
`muse-glimmer.cpp` and `glm5-next.cpp` besides `glm-dsa.cpp` and `deepseek4.cpp`. The vendored
**binaries** (`tools/llama.cpp`, b10306) are unchanged and know only `glm-dsa` and `deepseek4`.
Every existing receipt and League ratio is pinned to b10306, so updating them is a separate,
deliberate step (keep b10306 alongside, re-run key receipts on both, then switch).

| Family | Plan | Fits this PC? | Second implementation | Port + synthetic |
|---|---|---|---|---|
| Muse-Glimmer | [plan](2026-10-03-muse-glimmer-port-plan.md) | yes (small quants) | llama.cpp `muse-glimmer.cpp` (source) + TensorSharp | **ADMITTED 2026-10-03 (text, CPU only).** `admit-arch` 8/8 exact; real-window 3,748-token prompt identical; Level 2 synthetic parity (`MuseGlimmerSyntheticTests`) plus level 4: `UD-Q4_K_XL` (Muse-Glimmer-30B) greedy output vs llama.cpp `bed0a8566` over 3 prompts is token-identical to the compared length except one 0.006-nat near-tie (" in" 0.7108 vs " proper" 0.7169 NLL). Prefill about 10x slower than llama.cpp (per-token for gated models). Remaining: batched prefill, vision tower, DFlash |
| Qwen 3.8 Flash Next | [plan](2026-10-03-qwen38-flash-next-port-plan.md) | paged only (72.5 GB; fits the scratch disk, not RAM) | llama.cpp `qwen4exp.cpp` (source) + TensorSharp | **Partial**: component tests + synthetic execution. Routed MoE added 2026-10-03 (`Qwen4ExpMoeRoutingTests`). **Missing:** RoPE in the QSA mixer, QSA indexer + K-pool block selection, PLE n-gram table. Forward pass refuses real configs (`indexer_top_k > 0`) until then |
| GLM-5.x | [plan](2026-10-03-glm5-port-plan.md) | paged only (GLM-5.3 Q2_K_XL ~236 GiB; fits the 279 GB scratch disk, not RAM) | `glm-dsa`: llama.cpp source + b10306 binaries; `glm5next`: llama.cpp source for the trunk, **not NextN** | **Synthetic execution** (`GlmDsaSyntheticTests`, `Glm5NextSyntheticTests`; finite-logit smoke tests, not parity). `glm5next`: HC-stream carry-over bug fixed 2026-10-03; K-pool sparse selection missing (throws past `indexer.top_k` keys); no regression test for the HC fix yet. `glm-dsa`: the b10306 llama.cpp synthetic comparison is not done |
| DiffusionGemma | [plan](2026-10-03-diffusiongemma-port-plan.md) | yes (~13-17 GB) | HF reference + TensorSharp (no llama.cpp) | **Component-tested only; real checkpoint REFUSED (2026-10-03: tensor names and Gemma-4 MoE layer do not match the port, no RoPE, prefill without attention; see docs/1-correctness).** (`DiffusionGemmaTests`). Fixed 2026-10-03: stability on argmax history, commit argmax not the re-noised canvas, exact soft-embedding sum. **Missing:** learned self-conditioning MLP (tensors not loaded), Gumbel-max candidate sampling unverified against the HF/vLLM source |
| MiniMax-H3 | [plan](2026-10-03-minimax-h3-port-plan.md) | yes (sequential loading, see plan) | upstream HF/PyTorch + TensorSharp | **Foundation only** (`MiniMaxH3Tests`): layout, schedulers, AdaLN, MM-RoPE, VAE decoders, synthetic pipeline. **Missing:** Qwen3-VL hidden-state extraction, 50-block real loader, VAE encode, conditioning modes |
| DeepSeek V4 / V4.1 (review) | [plan](2026-10-03-deepseek-v4-review-plan.md) | V4: paged only (~98.6 GB). V4.1: no (~335 GB, does not fit the scratch disk) | V4: llama.cpp source + b10306; V4.1: TensorSharp only | **V4: synthetic execution** (the forward test uses ratio 0 only, so CSA is unexercised). **V4.1: partial**, parsed + loaded but compressed attention (ratios 1/2), V4.1 indexer, YaRN and quantised Engram are not on the execution path; the forward pass now refuses such configs. Fixed 2026-10-03: `moe_intermediate_size` 2304, indexer defaults |

The estimates cover **port + synthetic verification only**. Real-checkpoint verification
(hardware-dependent), the closeout performance + DRY pass (CLAUDE.md rule 7) and promotion are
separate costs.

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

Label what a test suite actually proves; "Done" is not a label:
**structural** (metadata, shapes, construction) < **synthetic execution** (tiny model runs, finite
output) < **synthetic parity** (agrees with an independently written reference) < **reference
parity** (agrees with another implementation). A test that recomputes the expected value with the
same logic as the code under test proves nothing and should not exist.

**Scope:** this is the TensorSharp coverage wave (started 2026-10-02). Families beyond these six get
a new plan when someone decides to take them on; the wave isn't meant to grow by itself.

## Models that cannot be admitted on this host (2026-10-03)

Policy: a family whose checkpoint cannot be run here stays **written, tested synthetically, and not
exposed** (no admission, no catalog/CLI/STATUS entry). That is the correct end state, not a gap.

- **DeepSeek-V4.1 (`deepseek41`, ~335 GB):** does not fit even on the 279 GB scratch disk; not
  attempted. Stays PORTED + SYNTHETIC only.
- **Qwen 3.8 Flash Next (~72.5 GB), DeepSeek-V4 (~98.6 GB):** both fit `E:\_models` and can be paged on this host (slow is fine). DeepSeek-V4 is a real candidate (the vendored llama.cpp b10306 knows `deepseek4`); do it after GLM-5.3. Qwen cannot be run until QSA selection, RoPE and the PLE table are implemented (the forward pass refuses real configs).
- **GLM-5.3 (~236 GiB):** fits `E:\_models`; correctness-only runs (hours, paged from disk) are
  acceptable. Slowness is not a failure; a token/logit mismatch is.

Real-checkpoint order: Muse-Glimmer (done 2026-10-03), then GLM-5.3, then DeepSeek-V4. DiffusionGemma is blocked on a backbone rewrite (see its plan). Download into `E:\_models`, verify against the independent implementation, then delete the checkpoint.


## Per-family checklist (every family)

- [ ] Read the primary reference first (llama.cpp `src/models/<arch>.cpp` when it exists), then
      TensorSharp's port as a second reading (CLAUDE.md rule 8).
- [ ] Port: hyperparameters, tensor set, forward pass, and tokenizer/chat template if new.
- [ ] Structural/synthetic tests only, named as such.
- [ ] `// <arch> — NOT admitted` block in `src/OpenTail.Stingray.Engine/ModelCompatibility.cs`:
      what was ported, from which reference (file and commit), and what verification is missing.
- [ ] Row in the plan's "Ported, not verified" table (date, code location, reference).
- [ ] Ported code from TensorSharp goes into the TensorSharp section of `THIRD_PARTY_NOTICES.md`.
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
- **Weights:** `unsloth/Qwen3.8-Flash-Next-GGUF` (~180B total: 125B lang + 51B PLE + 4B MTP; smallest quant ~72.5 GB; does not fit this PC; synthetic gate only).
- **Reuse:** Stingray's `GdnKernels` (GDN FP32 recurrence), MoE primitives (512 experts, top-10 + 1 shared); distinct `GatedResidual` (HC) and QSA/PLE subsystems.
- **References:** llama.cpp `src/models/qwen4exp.cpp` (local source), TensorSharp `Models/Qwen4Exp`.
- [x] Port
- [x] Not-admitted block
- [x] Table row

### 2. GLM-5.x (`glm-dsa`, alias `glm_dsa`; GLM-5.3-Flash is `glm5next`)
- **Design:**
  - `glm-dsa` (GLM-5.2/5.3): 744B MoE (256 routed + 1 shared, top-8, sigmoid, scale 2.5; 3 leading dense) on MLA (64 heads, 512+64 latent row) + 32-head DSA indexer with orthonormal Sylvester Hadamard rotation (`PrismHadamard` FWHT); shared indexer layers reuse top-k.
  - `glm5next` (GLM-5.3-Flash): 320B hybrid trunk with 45 layers (34 KDA linear attention + 11 NoPE MLA/DSA in strict 3:1 schedule: `i % 4 != 3` -> KDA, `i % 4 == 3` -> MLA), 4-token K-pool indexer, x4 Sinkhorn mHC, 288+1 MoE with FP32 router, SwiGLU clamp at 10.
- **Weights:** `unsloth/GLM-5.3-GGUF` (UD-Q2_K_XL ~236.4 GiB, Q4 ~432 GB; does not fit 64 GB RAM; synthetic gate only).
- **Reuse:** Stingray's `deepseek2` MLA, `deepseek32` indexer, `PrismHadamard` FWHT, and `deepseek4` mHC.
- **References:**
  - `glm-dsa`: llama.cpp `src/models/glm-dsa.cpp` (local source; vendored b10306 binaries know it too and serve as local oracle), HF `GlmMoeDsaForCausalLM`, TensorSharp `Models/GlmDsa`.
  - `glm5next`: current upstream llama.cpp `src/models/glm5-next.cpp` (`bed0a8566`, includes K-pool, mHC, KDA recurrence, MTP), HF `Glm5NextForConditionalGeneration`, TensorSharp `Models/GlmDsa`.
- [x] Port `glm-dsa`
- [x] Port `glm5next`
- [x] Add not-admitted blocks
- [x] Add table rows
- [x] Specification tests passing (`GlmDsaSyntheticTests`, `Glm5NextSyntheticTests`)

### 3. Muse-Glimmer (`muse-glimmer`, alias `muse_glimmer`)
- **Design:**
  - text + image; thinking through an `assistant to=self` channel;
  - llama.cpp's standard SWA window predicate and a hard-coded 1e-8 (per TensorSharp's card);
  - DFlash speculative drafter (optional).
- **Weights:** Muse-Glimmer-30B; GGUF sizes on the card are about 3-10 GB for the parts listed.
- **References:** TensorSharp `Models/MuseGlimmer` (cites `llama.cpp/src/models/muse-glimmer.cpp`,
  which is present in the local copy at `bed0a8566`).
- [x] Port text tower
- [ ] Port vision tower (deferred)
- [x] Add not-admitted block
- [x] Add table row
- [x] Specification test (level 2) passing (`MuseGlimmerSyntheticTests`)

### 4. DiffusionGemma (`diffusion-gemma`, alias `diffusion_gemma`)
- **Design:**
  - block **text diffusion** on Gemma-4 MoE backbone (26B total, ~4B active: hidden 2816, dense FFN 2112 + 128 experts top-8);
  - 30 layers: 5 Full attention (Q=16, KV=2, dim=512) + 25 Sliding attention (Q=16, KV=8, dim=256, window=1024);
  - **Self-conditioning**: step 1 seed, $t > 1$ soft probability weighted embeddings through learned SC MLP injected into canvas;
  - Canvas denoising (256 tokens) with bidirectional canvas attention + persistent causal prefix KV cross-attention;
  - Sampler: temperature decay ($0.8 \to 0.408$ over 48 steps), nats entropy budget (0.1), greedy acceptance, categorical re-noising;
  - Block-autoregressive lifecycle: commit 256 tokens, causal re-prefill, next canvas.
- **Weights:** `google/diffusiongemma-26B-A4B-it` (HF), `unsloth/diffusiongemma-26B-A4B-it-GGUF` (Q4_K_M ~16.8 GB; fits 64 GB machine).
- **Reuse:** Stingray's Gemma 4 primitives, MoE routing, and PagedKvCache.
- **References:** HF reference implementation, TensorSharp `Models/DiffusionGemma`, Unsloth / llama.cpp DiffusionGemma runner.
- [x] Port (`src/OpenTail.Stingray.Diffusion/DiffusionGemma/*.cs`)
- [x] Add not-admitted block in `ModelCompatibility.cs`
- [x] Add table row
- [x] Specification tests passing (`DiffusionGemmaTests`)

### 5. MiniMax-H3 (video + native 32 kHz stereo audio)
- **Design:**
  - one DiT denoises a packed video+audio latent;
  - seven graphs: text encode, vision encode, DiT, and encode + decode for each of the two VAEs;
  - up to 15 s at 24 fps.
- **Weights:** `MiniMaxAI/MiniMax-H3`, `unsloth/MiniMax-H3-GGUF`, `Comfy-Org/MiniMax-H3`; parts
  about 12-35 GB.
- **Reuse:** Stingray's diffusion pipeline (Wan, HunyuanVideo, LTX) and audio VAE code. A
  diffusion project, not an LLM arch.
- **References:** TensorSharp `Models/MiniMaxH3`, upstream HF code.
- [x] Port (`src/OpenTail.Stingray.Diffusion/MiniMaxH3/*.cs`)
- [x] Add a not-advertised entry; diffusion has no `ModelCompatibility` allowlist, so keep it off
      the CLI's model list until verified
- [x] Add table row
- [x] 13 unit / pipeline tests passing (`MiniMaxH3Tests.cs`)

### 6. DeepSeek V4 / V4.1 Flash (V4 review + V4.1 distinct architecture)
- Stingray has V4 alpha code (`DeepSeek4*.cs`, plan [058](058-deepseek-full-lineage-implementation-plan.md)), never run.
- **Design:**
  - V4 Flash (`deepseek4`): 43 layers, hidden 4096, 64 heads, head dim 512, compression ratios 0/4/128, 256+1 MoE (top-6, sqrtsoftplus), x4 mHC, 64x128 indexer top-512, 1 NextN.
  - V4.1 Flash (`deepseek41`): distinct architecture with 40 layers, hidden 5120, compression ratios 0/1/2, 384+1 MoE (scale 1.5), dual ~196B Engram tables (layers 1 & 14, 4-gram, 99k compressed vocab), 8 index-source layers, candidate top-2048 x block 8, 3 NextN layers, DSpark (layers 37-39).
- **Weights:** V4 Flash Q2_K ~98.6 GB / 117 GB; V4.1 Flash Q2_K+Q5 ~335.4 GB (due to Engram). Neither fits 64 GB RAM.
- **Reuse:** Stingray's existing `DeepSeek4ForwardPass`, `DeepSeek4CompressedState`, `DeepSeek4Alpha`.
- **References:**
  - V4: llama.cpp `src/models/deepseek4.cpp` (local source + vendored b10306 as local oracle).
  - V4.1: TensorSharp `Models/DeepSeek4/` (`DeepSeek41Model.cs`, `DeepSeek4CpuExecutor.V41.cs`, `Dsv41EngramData.cs`) validated under 103-case PyTorch oracle (100/103 passed).
- [x] Review and fix existing V4 alpha (CSA ratio 4, 8-group output LoRA, Hadamard on indexer)
- [x] Implement V4.1 distinct architecture (`deepseek41`): ratios 0/1/2, 384+1 MoE, Engram subsystem (`DeepSeek41*.cs`)
- [x] Add not-admitted blocks and update table rows
- [x] Specification tests passing (`DeepSeek4AlphaTests`, `DeepSeek41AlphaTests`)
