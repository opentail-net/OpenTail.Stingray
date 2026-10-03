# DeepSeek V4 / V4.1 review plan (`deepseek4`, `deepseek41`)

**Status (revised 2026-10-03):** V4 has reviewed alpha code (never compared with llama.cpp; its only test does not exercise CSA). V4.1 is **partial**: raw-attention trunk only, and it refuses real configs. Neither is verified. ([058](058-deepseek-full-lineage-implementation-plan.md))
This plan governs the review of existing V4 alpha code and the architectural definition of V4.1.
**Policy:** Both architectures remain **not admitted and not advertised** (CLAUDE.md rule 14).

---

## Architecture Lineage: V4 vs. V4.1

**V4.1 is not merely a set of "deltas" over V4; it is a distinct architecture** with its own GGUF architecture identifier (`deepseek41`; renaming or aliasing to `deepseek4` is invalid). It carries altered compression ratios, a 384-expert MoE, massive dual Engram tables, altered indexer schedules, 3 MTP layers, and DSpark.

### 1. DeepSeek-V4 Flash (`deepseek4`) Contract

| Parameter | Specification |
|---|---|
| Geometry | 43 layers, hidden dimension $D = 4096$, 64 attention heads, $\text{head\_dim} = 512$ |
| Compression Ratios | **0 / 4 / 128** (`[0, 0, 4, 128, ..., 4, 128, 0]`: raw, CSA, HCA) |
| MoE Backbone | 256 routed experts + 1 shared expert, $\text{top\_k} = 6$, $\text{sqrtsoftplus}$ routing, 3 hash-routing layers |
| Hyper-Connections | $\times 4$ mHC, 20 Sinkhorn iterations |
| Indexer & RoPE | $64 \times 128$ indexer, $\text{top\_k} = 512$, sliding window $128$, YaRN $16\times$ |
| NextN / MTP | 1 trailing NextN layer |

### 2. DeepSeek-V4.1 Flash (`deepseek41`) Contract

| Parameter | Specification |
|---|---|
| Geometry | 40 layers, hidden dimension $D = 5120$, 64 attention heads, $KV = 1$, $\text{head\_dim} = 512$ |
| Projections | `q_lora_rank = 1280`, `o_lora_rank = 1024` with 8 output groups |
| Compression Ratios | **0 / 1 / 2** (explicit array: `[0, 0, 2 x 18 layers, 1 x 20 layers, 0 x 3 layers]`) |
| MoE Backbone | 384 routed experts + 1 shared expert, $\text{top\_k} = 6$, $\text{sqrtsoftplus}$ routing, routed scale $= 1.5$ |
| Indexer & RoPE | 8 index-source layers, candidate $\text{top\_k} = 2048$, block size 8, final index $\text{top\_k} = 512$, sliding window $128$, YaRN with 64 RoPE dims |
| **Engram Subsystem** | Embedded N-gram memory on **layers 1 and 14**: ~384M embeddings each (~196.6B params in Q2/Q5 packaging), vocab size 16M, max N-gram 4, 8 heads $\times$ 256, compressed vocab 99,092 |
| **DSpark Subsystem** | Speculative drafting blocks on **layers 37, 38, 39**: block size 5, Markov rank 256, 128 routed experts, 3 experts/token, noise token ID 128799 |
| NextN / MTP | 3 NextN layers |

---

## References & Validation Oracles

- **V4 References**:
  - Primary: Local llama.cpp `src/models/deepseek4.cpp` (`bed0a8566`).
  - Local Oracle: Vendored `tools/llama.cpp` b10306 binaries know `deepseek4` and serve as an independent local oracle for synthetic GGUF checks.
- **V4.1 References**:
  - Primary: TensorSharp `Models/DeepSeek4/` (`DeepSeek41Model.cs`, `DeepSeek4CpuExecutor.V41.cs`, `Dsv41EngramData.cs`), verified under a 103-case PyTorch-derived validation protocol (100/103 passed).
  - Note: Stock upstream llama.cpp does **not** support `deepseek41` (distributed V4.1 GGUFs require specialized fork branches with expert streaming and disk-resident Engram). Stock `llama-server` cannot be claimed as an oracle for V4.1.

---

## Checkpoint Sizes & Hardware Reality

- **V4 Flash**: Q2_K base is ~98.6 GB (XL quants ~60–73 GiB; full ~117 GB).
- **V4.1 Flash**: Q2_K+Q5 Engram variant is **335.4 GB** (other variants 371–508 GB) due to the two ~196B Engram tables.
- **Neither fits a 64 GB machine**: Review, synthetic specification tests, and reference trace comparison are the only local goals.

---

## Existing V4 Alpha Status in Stingray

Per plan [058](058-deepseek-full-lineage-implementation-plan.md), Stingray's existing `DeepSeek4ForwardPass.cs` has:
- **Implemented**: Raw attention, HCA (ratio 128), basic mHC Sinkhorn, MoE routing, indexer scoring.
- **Gaps & Uncertainties to Resolve**:
  - CSA (ratio 4) is currently unimplemented.
  - Grouped output LoRA (8 groups) is unverified.
  - `rope_ext_back` semantics need alignment with `deepseek4.cpp`.
  - Non-rewindable compressed state must be upgraded to support standard cache rewind/decode.
  - Real-weight verification is unexecuted.

---

## Decomposed Review & Implementation Phases (honest state, 2026-10-03)

Legend: [x] done and checked, [~] code exists but is smoke-tested only or has known gaps, [ ] not done.
Tests: `DeepSeek4AlphaTests` / `DeepSeek41AlphaTests` (component and metadata checks) and `DeepSeek4SyntheticTests` /
`DeepSeek41SyntheticTests` (one finite-logit smoke test each). **No numeric oracle has run for either architecture.**

### Part 1: DeepSeek-V4 Review & Synthetic Verification
- [~] **Phase 0: V4 reference reconciliation.** A review pass changed the alpha code (CSA ratio 4, 8-group output LoRA,
  Hadamard on the indexer, per the commit and the hub). Not independently re-audited line by line; the code itself still calls
  its CSA overlap construction a "working hypothesis".
- [~] **Phase 1: Resolve V4 alpha uncertainties.** CSA ratio 4 and the 8-group output LoRA are in the code. Unverified:
  CSA overlap boundaries, indexer top-k selection, `rope_ext_back`, mHC equations, and the grouped-LoRA pointer arithmetic on
  quantised dtypes (offsets assume a dense stride).
- [ ] **Phase 2: V4 synthetic specification test.** The one synthetic test uses `compress_ratios = [0, 0]` and 2 output
  groups: **CSA is not exercised at all**. The planned tiny `deepseek4` GGUF against vendored llama.cpp b10306 was never run.
- [~] **Phase 3: V4 gate & audit record.** Gate [x] (`// deepseek4 - NOT admitted`). Plan 058's audit notes not updated.

### Part 2: DeepSeek-V4.1 Architecture Definition & Core Port
- [~] **Phase 10: V4.1 schema.** Loader, parser and tensor set exist. Corrected 2026-10-03 against the official
  `config.json`: `moe_intermediate_size` 2304 (was 2048) and `index_n_heads` 32 (was 64), now parsed from metadata.
  Official `compress_ratios` (0,0, then 2 x18, 1 x20, 0,0,0), `index_source_layer_ids` [2,8,14,20,24,28,32,36] and
  `engram_layer_ids` [1,14] agree with the code's defaults.
- [~] **Phase 11: V4.1 core forward pass.** Raw attention + mHC + MoE + grouped output only. **Compressed attention for
  ratios 1 and 2 is not implemented** (the ratio array was parsed and never used), and YaRN (factor 16, beta 32/1, original
  65536) is not applied. The forward pass now **refuses** configs with ratios > 0 or `rope.scaling.factor` > 1 before loading
  weights (`DeepSeek41AlphaTests` refusal tests). `output_hc_*` tensors are loaded but not consumed.
- [ ] **Phase 12: V4.1 indexer subsystem.** Tensors are loaded; the indexer (candidate 2048 x block 8, final top-512, 8
  index-source layers) is **not implemented**.
- [~] **Phase 13: V4.1 Engram.** Hashing, lookup and projection exist with unit tests, but the lookup reads raw **F32** (a
  non-F32 table now throws), and the hash function (multiplier 1000003, mod 99,092) is unverified against any reference.
- [ ] **Phase 14: V4.1 synthetic specification test.** One 2-layer smoke test; the 40-layer schedule, layers 1 and 14 Engram,
  ratio-1/2 layers, the index-source schedule and the TensorSharp 103-case oracle comparison are all untested.
- [~] **Phase 15: V4.1 gate.** `// deepseek41 - NOT admitted` block [x]; refusal gates [x].

### Deferred (Explicitly Out of Scope)
- **V4.1 DSpark**: Layers 37–39 speculative drafting is deferred.
- **V4.1 3-Layer MTP**: MTP heads deferred until trunk is stable.
- **Multimodal Vision**: 32-layer vision tower is deferred; initial milestone is text-only.
- **GPU Paths & Large-Memory Deployment**: Out of scope for local 64 GB machine.

---

## Verification Summary

| Model | Level 2 (Specification) | Level 3 (Independent Implementation) | Level 4 (Real Weights) | Admission (Level 5) |
|---|---|---|---|---|
| **V4** | Smoke test only (ratio 0, 2 groups); tiny-GGUF comparison not run | Vendored `llama.cpp` b10306 knows `deepseek4` | **Feasible**: Q2_K ~98.6 GB, paged from `E:_models`, hours-long correctness-only run (decision 2026-10-03: slowness is acceptable); do after GLM-5.3 | `admit-arch` + timed runs |
| **V4.1** | Smoke test only | TensorSharp 103-case oracle (needs a PyTorch/TensorSharp run) | **Not attempted: ~335 GB does not fit the 279 GB scratch disk.** Stays written, tested synthetically, not exposed | Not admissible on this host |

---

**Effort:** the original estimates covered port + smoke tests. Remaining for V4: a CSA-exercising synthetic test, the b10306 comparison, then the paged real-weight run. Remaining for V4.1: ratios 1/2 attention, the indexer, YaRN, dtype-aware Engram, an oracle comparison.
