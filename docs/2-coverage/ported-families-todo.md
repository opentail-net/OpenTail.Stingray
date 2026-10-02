# Families to port now, prove later (todo)

User policy, 2026-10-02: port these families from their reference code now; prove them on real
checkpoints when there is capacity; meanwhile **don't admit and don't advertise them**. The
rules are in CLAUDE.md rule 14 and §4 of
[2026-10-02-tensorsharp-takeaways-plan.md](2026-10-02-tensorsharp-takeaways-plan.md). This file
is the working todo. When a family is ported, add a row to that plan's "Ported, not verified"
table.

Order (user): after the worker-pool experiment, the Q8_K alignment, and `Q1_0` + Bonsai2.

## Per-family plans (one each, 2026-10-03)

| Family | Plan | Fits this PC? | Local independent reference | Effort |
|---|---|---|---|---|
| Muse-Glimmer (in progress) | [2026-10-03-muse-glimmer-port-plan.md](2026-10-03-muse-glimmer-port-plan.md) | yes (small quants) | none (TensorSharp only) | ~2-3 h text |
| Qwen 3.8 Flash Next | [2026-10-03-qwen38-flash-next-port-plan.md](2026-10-03-qwen38-flash-next-port-plan.md) | yes (~15 GB quant) | none (TensorSharp only) | ~1-2 days |
| GLM-5.x | [2026-10-03-glm5-port-plan.md](2026-10-03-glm5-port-plan.md) | no (320B-744B) | `glm-dsa`: llama.cpp source + vendored b10306 | ~2 days |
| DiffusionGemma | [2026-10-03-diffusiongemma-port-plan.md](2026-10-03-diffusiongemma-port-plan.md) | yes (~13-17 GB) | none (TensorSharp only) | ~1 day |
| MiniMax-H3 | [2026-10-03-minimax-h3-port-plan.md](2026-10-03-minimax-h3-port-plan.md) | yes (sequential loading) | TensorSharp pure-C# backend | ~3-5 days |
| DeepSeek V4 / V4.1 (review) | [2026-10-03-deepseek-v4-review-plan.md](2026-10-03-deepseek-v4-review-plan.md) | no (~340 GB) | llama.cpp source + vendored b10306 | ~0.5-1.5 days |

Each plan has the architecture spec as far as it is known, what Stingray reuses, phased work items,
and a verification ladder: a synthetic-model check against an independent reference first, then a
real checkpoint where it fits, then the normal admission path.

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
- [ ] Later, for admission: a real checkpoint, an independent reference (`stingray admit-arch`
      against llama.cpp where it supports the arch), timed real-weight runs, then the admission
      and the STATUS row in the same pass.

Local references: the llama.cpp source in `examples/llama.cpp/llama.cpp` (commit `3653e6d6d`,
2026-08-07) has `glm-dsa.cpp` and `deepseek4.cpp` but **not** `glm5next`, `qwen4exp` or
`muse-glimmer`. For those, TensorSharp is the only local reference, so a later independent
reference has to come from a newer llama.cpp or the upstream HF code before admission.

## Families

### 1. Qwen 3.8 Flash Next (`qwen4exp`)
- **Design** (from TensorSharp's card):
  - hybrid MoE: GatedDeltaNet recurrent layers interleaved with full-attention layers, some behind
    Qwen Sparse Attention's indexer;
  - a PLE n-gram embedding block;
  - x4 hyper-connection streams;
  - 512-expert MoE;
  - image input through an mmproj.
- **Weights:** `unsloth/Qwen3.8-Flash-Next-GGUF` (multi-shard); quants about 15-48 GB, so the
  smaller ones fit here.
- **Reuse:** Stingray's `HybridGdnForwardPass` (GDN), MoE and DeepSeek-style indexer work
  (`deepseek32` lightning indexer); hyper-connections overlap `deepseek4`'s.
- **References:** TensorSharp `Models/Qwen4Exp`. Not in the local llama.cpp.
- [ ] Port   - [ ] Not-admitted block   - [ ] Table row

### 2. GLM-5.x (`glm-dsa`, alias `glm_dsa`; GLM-5.3-Flash is `glm5next`)
- **Design:**
  - GLM-5.2/5.3 is a 744B MoE (256 routed + 1 shared, top-8) on DeepSeek Sparse Attention: MLA
    with weight absorption plus a lightning indexer;
  - GLM-5.3-Flash (`glm5next`) runs through the same executor in TensorSharp.
- **Weights:** `unsloth/GLM-5.3-GGUF` (about 765 GB; won't fit). GLM-5.3-Flash size not yet
  checked.
- **Reuse:** Stingray's `deepseek2` MLA and `deepseek32` indexer alpha code.
- **References:**
  - `glm-dsa`: llama.cpp `src/models/glm-dsa.cpp` (local), plus TensorSharp `Models/GlmDsa`. Its
    note: reproducing llama.cpp's indexer top-k restored 6/6 token parity.
  - `glm5next`: TensorSharp only. Its note: llama.cpp is not a valid reference for it.
- [ ] Port `glm-dsa`   - [ ] Port `glm5next`   - [ ] Not-admitted blocks   - [ ] Table rows

### 3. Muse-Glimmer (`muse-glimmer`, alias `muse_glimmer`)
- **Design:**
  - text + image; thinking through an `assistant to=self` channel;
  - llama.cpp's standard SWA window predicate and a hard-coded 1e-8 (per TensorSharp's card);
  - DFlash speculative drafter (optional).
- **Weights:** Muse-Glimmer-30B; GGUF sizes on the card are about 3-10 GB for the parts listed.
- **References:** TensorSharp `Models/MuseGlimmer` (cites `llama.cpp/src/models/muse-glimmer.cpp`,
  which is not in the local copy).
- [ ] Port (text first, vision second)   - [ ] Not-admitted block   - [ ] Table row

### 4. DiffusionGemma (`diffusion-gemma`, alias `diffusion_gemma`)
- **Design:**
  - block **text diffusion**, not autoregressive decode;
  - Gemma 4 trunk; image input through Gemma 4's vision tower.
- **Weights:** `google/diffusiongemma-26B-A4B-it` (HF), `unsloth/diffusiongemma-26B-A4B-it-GGUF`;
  about 13-17 GB.
- **Reuse:** Stingray's Gemma 4 path and vision tower; needs a new generation loop (a sampler over
  a canvas, not the token-by-token `InferenceEngine`).
- **References:** TensorSharp `Models/DiffusionGemma` (shard checked against a NumPy
  transcription of the HF reference); no llama.cpp output is recorded.
- [ ] Port   - [ ] Not-admitted block   - [ ] Table row

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
- [ ] Port   - [ ] Not-advertised entry (diffusion has no `ModelCompatibility` allowlist; keep it off
      the CLI's model list until verified)   - [ ] Table row

### 6. DeepSeek V4 / V4.1 Flash (review, not a new port)
- Stingray already has alpha code (`DeepSeek4*.cs`, plan
  [058](058-deepseek-full-lineage-implementation-plan.md)), never run.
- **Task:** review it against TensorSharp's pure-C# `DeepSeek4CpuExecutor` and llama.cpp
  `deepseek4.cpp` (local). Add `deepseek41` deltas if TensorSharp shows them. Checkpoints
  (about 340 GB) don't fit this PC.
- [ ] Review   - [ ] V4.1 deltas   - [ ] Update plan 058
