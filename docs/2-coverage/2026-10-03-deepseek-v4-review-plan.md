# DeepSeek V4 / V4.1 review plan (`deepseek4`, `deepseek41`)

**Status:** Stingray has V4 **alpha code, never run** ([058](058-deepseek-full-lineage-implementation-plan.md)).
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

## Decomposed Review & Implementation Phases

### Part 1: DeepSeek-V4 Review & Synthetic Verification
- [ ] **Phase 0: V4 Reference Reconciliation**
  - Line-by-line audit of `DeepSeek4Alpha.cs`, `DeepSeek4ForwardPass.cs`, `DeepSeek4CompressedState.cs` against `deepseek4.cpp` (`bed0a8566`) and TensorSharp `DeepSeek4CpuExecutor.cs`.
- [ ] **Phase 1: Resolve V4 Alpha Uncertainties**
  - Wire CSA (ratio 4), verify 8-group output LoRA, align `rope_ext_back` and mHC equations.
- [ ] **Phase 2: V4 Synthetic Specification Test**
  - Construct tiny synthetic `deepseek4` GGUF.
  - Run level-3 independent verification against vendored `llama.cpp` b10306.
- [ ] **Phase 3: V4 Gate & Audit Record**
  - Retain `// deepseek4 — NOT admitted` block; update audit notes in plan 058.

### Part 2: DeepSeek-V4.1 Architecture Definition & Core Port
- [ ] **Phase 10: V4.1 Reference & GGUF Schema Lock**
  - Establish `deepseek41` GGUF architecture loader, metadata parser, and tensor set (`DeepSeek41Alpha.cs`, `DeepSeek41TensorSet.cs`).
- [ ] **Phase 11: V4.1 Core Forward Pass**
  - Implement 40-layer trunk, compression ratios **0 / 1 / 2**, 384+1 MoE (top-6, scale 1.5, sqrtsoftplus), and mHC.
- [ ] **Phase 12: V4.1 Indexer Subsystem**
  - 8 index-source layers, candidate pool 2048 $\times$ block 8, top-512 final selection.
- [ ] **Phase 13: V4.1 Engram Subsystem**
  - Layer 1 and layer 14 N-gram memory lookup, compressed-vocab hashing (99,092), memory-mapped tensor access strategy.
- [ ] **Phase 14: V4.1 Synthetic Specification Test**
  - Tiny synthetic `deepseek41` model; verify step trace parity against TensorSharp reference executor.
- [ ] **Phase 15: V4.1 Gate & Documentation**
  - Add `// deepseek41 — NOT admitted` block in `ModelCompatibility.cs`.

### Deferred (Explicitly Out of Scope)
- **V4.1 DSpark**: Layers 37–39 speculative drafting is deferred.
- **V4.1 3-Layer MTP**: MTP heads deferred until trunk is stable.
- **Multimodal Vision**: 32-layer vision tower is deferred; initial milestone is text-only.
- **GPU Paths & Large-Memory Deployment**: Out of scope for local 64 GB machine.

---

## Verification Summary

| Model | Level 2 (Specification) | Level 3 (Independent Implementation) | Level 4 (Real Weights) | Admission (Level 5) |
|---|---|---|---|---|
| **V4** | Synthetic tiny GGUF | Vendored `llama.cpp` b10306 | Deferred (>100 GB) | Needs large-RAM host + `admit-arch` |
| **V4.1** | Synthetic tiny GGUF | TensorSharp 103-case oracle | Deferred (>335 GB) | Needs custom deepseek41 runtime host |

---

**Effort:** ~1 day for V4 alpha reconciliation + b10306 synthetic verification; ~2 days for V4.1 core, ratios 0/1/2, and Engram architecture port. DSpark, vision, and real weights are separate.
