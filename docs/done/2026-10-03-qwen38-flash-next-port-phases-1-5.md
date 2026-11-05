# Qwen 3.8 Flash Next port plan: implemented Phases 1-5 (synthetic-verified)

> **ARCHIVED (2026-10-14).** Phases 1-5 of [../2-coverage/2026-10-03-qwen38-flash-next-port-plan.md](../2-coverage/2026-10-03-qwen38-flash-next-port-plan.md), split verbatim. All ticked: trunk, PLE, QSA RoPE, indexer/K-pool block selection, integration and the synthetic parity suite (24 tests). The live plan keeps the architecture reference, the status matrix and Phase 6 (72.5 GB real-weight qualification, still open; rule 14: not admitted).

### Phase 1 — Landed Trunk & Primitives (Complete)
- [x] **1.1 Specification & Hyperparameters Lock**:
  - GGUF metadata keys and tensor names matching `qwen4exp.cpp`.
  - `Qwen4ExpHyperparams` with full metadata extraction, including PLE hash arrays and compression ratios.
- [x] **1.2 GatedResidual (Hyper-Connections)**:
  - 4-stream grouped RMSNorm, 320 low-rank bottleneck, SiLU, stream-average collapse, $2\sigma$ injection.
  - Verified by `Qwen4ExpGatedResidualTests`.
- [x] **1.3 GDN + MoE Trunk (36 layers)**:
  - FP32 recurrence kernel via `GdnKernels.GdnRecurrenceDecode`.
  - 512 routed experts (top-10 + 1 shared expert, intermediate 640), softmax router, renormalised weights.
  - Verified by `Qwen4ExpMoeRoutingTests`.
- [x] **1.4 Basic PLE & QSA Component Helpers**:
  - Signed square root dot-product gating: `s = sum(key * query) / sqrt(d); gate = sigmoid(sgn(s) * sqrt(|s|))`.
  - Dilated depthwise causal 1D conv with recurrent state shift.
  - QSA interleaved Q+gate split and attention output gate (`SplitAndNormQGated`, `ApplyAttentionGate`).
  - Unit tests in `Tests.Core/Qwen4ExpAlphaTests.cs`.
- [x] **1.5 Guard & Gate**:
  - `hp.IndexerTopK > 0` constructor refusal guard in `Qwen4ExpForwardPass`.
  - Not-admitted block in `ModelCompatibility.cs` (CLAUDE.md rule 14).

---

### Phase 2 — PLE Subsystem Completion
*Objective: Replace the synthetic token-embedding feed in `ExecutePle` with the genuine n-gram hash row gather from the lazy PLE table.*

- [x] **2.1 Validate PLE Hash Metadata**:
  - Ensure `PleLayerMultipliers`, `PleHeadOffsets`, and `PleHeadVocabSizes` have matching lengths and validate bounds against total row count:
    `max(PleHeadOffsets[h] + PleHeadVocabSizes[h])`.
- [x] **2.2 Lazy Memory-Mapped PLE Row Store (`Qwen4ExpPleRowStore`)**:
  - Wrap the `per_layer_token_embd` tensor without loading the entire 28.8–51 GB table into managed arrays.
  - Expose random-access row dequantization: `ReadPleRow(long rowIndex, Span<float> dest)`.
- [x] **2.3 Stateful N-Gram Hasher (`Qwen4ExpPleHasher` / `PleNgramState`)**:
  - Maintain token history window across sequence steps.
  - Handle EOS reset semantics, missing predecessors, and sequence boundaries.
  - Compute the 64-bit n-gram hash per head via XOR of multiplied terms across the n-gram window:
    ```text
    mixed = token[t] * multiplier[0]
    mixed ^= token[t-1] * multiplier[1]
    ...
    row_h = offset[h] + (mixed % vocab_size[h])
    ```
- [x] **2.4 Connect Gathered Embeddings to `ExecutePle`**:
  - Gather and concatenate the multi-head PLE embeddings into `Span<float> pleEmb`.
  - Feed `pleEmb` into `PleKey` and `PleValue` projections, replacing the placeholder `tokEmbd` feed.
- [x] **2.5 Unit & Synthetic Tests**:
  - Test exact hash computation against known token sequence fixtures.
  - Test lazy row gather with small synthetic PLE table.
  - Test decode-step state continuity across multiple single-token forwards.

---

### Phase 3 — QSA RoPE (Two-Sided Rotary Application)
*Objective: Implement the full 4-section IMRoPE geometry across both the indexer and the main attention paths.*

- [x] **3.1 Shared IMRoPE Implementation**:
  - Implement 4-section IMRoPE matching `qwen4exp` rotary specifications (`RopeDimensionSections`).
- [x] **3.2 Indexer RoPE**:
  - Apply RoPE to pooled indexer keys using the position of the **first token in each pool**:
    `pos_pool = pool_index * kpool`.
  - Apply RoPE to indexer query using the current query token's position: `pos_q = position`.
- [x] **3.3 Main Attention RoPE**:
  - Apply RoPE to post-norm main attention Q (`AttnQNorm`) at `position`.
  - Apply RoPE to post-norm main attention K (`AttnKNorm`) at `position`.
- [x] **3.4 Unit Tests**:
  - Test RoPE numerical correctness at boundary positions and across section boundaries.

---

### Phase 4 — QSA Indexer & K-Pool Block Selection
*Objective: Implement raw indexer-K caching, block pooling, multi-head ReLU scoring, top-k pool selection with tail preservation, and sparse attention masking.*

- [x] **4.1 Dedicated QSA Indexer Cache (`QsaIndexerState`)**:
  - Maintain a separate raw indexer key cache per QSA layer (`index_k_proj` output), distinct from the attention KV cache.
  - Track pool completion state across prefill chunks and decode steps.
- [x] **4.2 K-Pool Formation & Tail Management**:
  - For complete blocks of $R$ (`kpool`, typically 4) tokens:
    Compute mean $\to$ RMSNorm (`IndexKNorm`) $\to$ RoPE at block start.
  - Maintain the active, incomplete tail block for tokens not yet forming a full pool.
  - Always retain the incomplete tail in the candidate selection set (`indexer_kpool_select_tail = true`).
- [x] **4.3 Indexer Query & Multi-Head ReLU Scoring**:
  - Project `index_q_proj` $\to$ RMSNorm (`IndexQNorm`) $\to$ RoPE at current position.
  - Score candidate pools:
    `score(pool) = (1 / sqrt(idxDim)) * sum_{h=0}^{H_idx-1} ReLU(dot(Q_h, K_pool))`.
- [x] **4.4 Top-K Pool Selection**:
  - Determine pool budget: `num_pools = indexer_top_k / kpool`.
  - Select top `num_pools` by score, plus all tokens in the incomplete tail.
- [x] **4.5 Sparse Attention Mask & Selective KV Walk**:
  - Translate selected pools into a token index bitmask / set.
  - In `ExecuteQsaMixer`, attend only over cached K/V tokens belonging to selected pools and the tail:
    unselected cells receive $-\infty$ (masked out); selected cells receive causal dot-product scores.
- [x] **4.6 Synthetic Tests**:
  - Test pool boundary straddling (e.g. prefill ending at position 3, decode resuming at 4).
  - Test sparse selection with synthetic KV caches where `top_k < total_tokens`.
  - Verify unselected tokens have zero attention weight on output.

---

### Phase 5 — Integration, Guard Removal & Synthetic Parity
- [x] **5.1 Wire Subsystems into `Qwen4ExpForwardPass`**:
  - Connect `Qwen4ExpPleHasher` + `Qwen4ExpPleRowStore` into layer 2 PLE.
  - Connect `QsaIndexerState` + IMRoPE + sparse attention into layers 3, 7, 11, ..., 47.
- [x] **5.2 Remove Refusal Guard**:
  - Remove `if (hp.IndexerTopK > 0) throw new NotSupportedException(...)` guard once indexer and PLE are functional.
- [x] **5.3 Synthetic Parity Test Suite**:
  - Extend `Qwen4ExpAlphaTests.cs` and `Qwen4ExpRoutedMoeTests.cs` with targeted structural parity tests:

  | Test | Target Subsystem | What It Catches |
  | :--- | :--- | :--- |
  | **PLE known-token fixture** | `PleHasher_KnownTokenSequence_CalculatesXorHashAndHeadIndices` | XOR vs addition, multipliers, and head index math |
  | **PLE n-gram head grouping** | `PleHasher_BigramVsTrigram_16HeadGroups` | Partitioning into bigram (heads 0–7) and trigram (heads 8–15) groups |
  | **PLE EOS / boundary semantics** | `PleHasher_BoundaryAndEosSemantics` | Missing predecessor padding, intervening EOS context cutoff, current EOS preservation |
  | **PLE concatenation geometry** | `Ple_Concatenation_AssemblyGeometry` | $16 \times 160$ head rows concatenated to 2560 (never averaged) |
  | **PLE chunked vs single-shot** | `PleHasher_ChunkedVsSingleShot_HistoryContinuity` | History continuity bugs and boundary truncation |
  | **QSA pool straddle** | `Qsa_PoolIndexerKeys_AveragesAndNormalizes` + multi-step forward | Block-state bugs when prefill/decode crosses $R$ boundary |
  | **QSA candidate & tail retention** | `Qsa_CandidateSelection_AlwaysRetainsActiveTail` | Unconditional preservation of incomplete tail block ($L=6$, $k=4$, tail $[4..5]$) |
  | **QSA selected vs unselected KV** | `Qsa_SelectTopKPools_ExcludesLowScoringPoolsAndRetainsTail` | Masking & index translation (unselected keys masked to $-\infty$) |
  | **RoPE Qwen3.8 geometry** | `RoPE_Qwen38Geometry_PartialRotaryFactor_Theta10M` | Partial rotary factor 0.25 (32 rotated, 96 passthrough), $\theta = 10\text{M}$ |
  | **RoPE known vector** | `Qwen4ExpRope_FourSectionImRope_RotatesSectionsCorrectly` | 4-section IMRoPE section boundaries and unrotated section 3 |
  | **Fused vs separate expert tensors** | `ExecuteMoe_FusedGateUpExperts_ProducesIdenticalOutputToSeparateTensors` | MoE layout compatibility (`ffn_gate_up_exps` vs separate) |
  | **End-to-end forward pass** | `Qwen4Exp_SyntheticForwardPass_RunsWithQsaSparseSelection_AndPleTable` | Full 4-stream HC + GDN + QSA + PLE integration |
  | **Reset / replay** | `ResetCache` test in `Qwen4ExpAlphaTests` | State leakage across consecutive forward passes |
