# Qwen 3.8 Flash Next port plan (`qwen4exp`)

**Status (revised 2026-10-03): IMPLEMENTED (SYNTHETICALLY VERIFIED, AWAITING REAL-WEIGHT QUALIFICATION).**
The full structural and semantic implementation (HC, GDN, MoE with fused/separate expert tensors, 16-head n-gram grouped PLE with 160-wide row concatenation, 4-section IMRoPE with partial rotary factor 0.25 and theta 10M, and QSA K-pooling/scoring with unconditional active tail retention) is complete and verified with synthetic tests. Real-weight qualification on the 72.5 GB UD-IQ1_S checkpoint is deferred.

**Policy:** port now, prove later; not admitted, not advertised (CLAUDE.md rule 14; [ported-families-todo](ported-families-todo.md)).

---

## Architecture & References

- **Primary reference**: llama.cpp `src/models/qwen4exp.cpp` (1,478 lines), in local source checkout (`bed0a8566`).
- **Secondary reference**: TensorSharp `docs/models/qwen38-flash-next.md`, `Models/Qwen4Exp/` (19 files, ~7.4k lines including CUDA/graph code, native QSA implementation, and QSA pool planner).
- **Upstream references**: transformers `configuration_qwen4_exp.py`, vLLM `config.py`, SGLang `configs/qwen4_exp.py`.

A hybrid MoE consisting of 48 layers in a strict 3:1 pattern (`full_attention_interval = 4`):
- `(i + 1) % 4 != 0` (36 layers): **GatedDeltaNet (GDN) linear attention + MoE**
- `(i + 1) % 4 == 0` (12 layers, i.e. 3, 7, 11, ..., 47): **Qwen Sparse Attention (QSA) + MoE**

### Architectural Components

1. **Hyper-connections (HC)**:
   - 4 parallel residual streams (`dsv4_hc_mult = 4`). Internal residual width is `4 × 2560 = 10240`. Block inputs/outputs are 2560.
   - Grouped RMSNorm with per-stream weights `[2560, 4]`.
   - Low-rank bottleneck of 320 (`hc_low_rank`): `down(10240 -> 320) -> SiLU(scale 1/4) -> up(320 -> 10240) -> sigmoid gate`.
   - Stream average collapse (`10240 -> 2560`) provides the block input.
   - Learned injection projection `w_inject` (`10240 -> 4`): block output is scaled by `2.0 * sigmoid(inject / 4)` and added to all 4 streams.
   - Note: This is an explicit `GatedResidual` mechanism. It is **not** DeepSeek-V4's Sinkhorn matrix hyper-connection (mHC).

2. **PLE (Position-Less Embedding) n-gram table & conv block**:
   - Injected at layer index 1 in GGUF zero-based indexing (`ple_layers = [1]`), corresponding to Hugging Face 1-based layer 2, which is a recurrent GDN layer.
   - ~320M-row / ~51B-parameter n-gram embedding table (`per_layer_token_embd`) mapped by 64-bit n-gram hashes (row count $\neq$ parameter count $\neq$ file size).
   - Upstream row geometry: `ple_head_dim = 160`, `n_heads = 16` (`ngram_size = 3`, 8 heads per n-gram).
   - **Row Concatenation**: Gathered head embeddings are $16 \times 160$ rows concatenated into a 2560-wide vector (`16 * 160 = 2560 = embed_dim`), which feeds `ple_key` and `ple_value`. They are **concatenated, never averaged**.
   - **N-Gram Head Grouping**: Heads are partitioned by n-gram length ($n = 2..\text{ngram\_size}$, base $= (n - 2) \times \text{heads\_per\_ngram}$). Heads 0–7 are bigrams ($n=2$); heads 8–15 are trigrams ($n=3$).
   - **EOS / Padding Semantics**: Missing predecessors before sequence start act as `eos_token_id`. An EOS anywhere in the predecessor window truncates context older than that EOS (padded with EOS). The current token being EOS does not erase its own context history.
   - Hash metadata arrays: `ple_layer_multipliers`, `ple_head_offsets`, `ple_head_vocab_sizes`.
   - 64-bit n-gram hash calculation: XOR of multiplied terms across the n-gram window (`mixed = (tok_0 * mult_0) ^ (tok_1 * mult_1) ^ ...`).
   - Gated query/key projection with signed square root scaling:
     `s = sum(key * query) / sqrt(d); gate = sigmoid(sgn(s) * sqrt(|s|))`.
   - Broadcast value with grouped RMSNorm.
   - Dilated depthwise causal 1D conv (`ple_conv1d`) with `dilation = ple_ngram_size`, `hist = (kern - 1) * dil`.
   - Stateful token history window surviving across chunked prefill and incremental decode calls.

3. **QSA (Qwen Sparse Attention)**:
   - Key pooling across blocks of `compress_ratio` cells (`indexer_kpool = 4`).
   - Dedicated indexer projections (`index_q_proj`, `index_k_proj`) and RMS norms.
   - **RoPE Geometry**: $\theta = 10{,}000{,}000$, `partial_rotary_factor = 0.25` ($\text{ropeDim} = 32$ of 128 head dim), sections `[11, 11, 10]`. Dimensions $\ge 32$ pass through unrotated (`sector = pair`, no modulo wrapping). Uses `SimdKernels.ApplyRoPECachedNeoxPartial`.
   - Two-sided RoPE: indexer Q and pooled-K RoPE; main attention Q and K RoPE using four-section IMRoPE.
   - Multi-head ReLU scoring: `score(pool, query) = (1 / sqrt(idxDim)) * sum_h ReLU(dot(Q_h, K_pool))`.
   - Top-k block selection (`indexer_top_k` / `compress_ratio` pools), with incomplete current tail preservation (`indexer_kpool_select_tail = true`).
   - Sparse attention mask mapping over KV cache: selected block cells $\to 0$, unselected cells $\to -\infty$.
   - Single Q projection producing interleaved `[q | gate]` per head, with `sigmoid(gate)` applied to attention output.

4. **MoE (Mixture of Experts)**:
   - 512 routed experts, top-10 routed experts per token.
   - 1 shared expert per token.
   - Expert intermediate size $d_{ff} = 640$ (`n_ff_exp = 640`).
   - Softmax over router logits, top-k selection, renormalised weights.
   - Both fused `ffn_gate_up_exps` and separate `ffn_gate_exps` / `ffn_up_exps` tensor layouts are supported and verified bit-identical.

5. **SSM Numerics**:
   - `mamba_ssm_dtype = float32`: GDN recurrent state and updates run strictly in FP32.

6. **MTP Head & Vision (Deferred)**:
   - Multi-token prediction and Qwen3.5-VL mmproj are explicitly out of scope for the text trunk.

---

## Checkpoint & Memory Architecture (Paged / Lazy)

- **Total Parameter Count**: ~180B total (125B language + 51B PLE n-gram table + 4B MTP).
- **Weights**: Smallest published GGUF (`UD-IQ1_S`) is **72.5 GB**. Standard quants range from 79 to 192 GB (BF16 is 354 GB).
- **Physical Host Context**: 64 GB host RAM + 279 GB scratch disk.
- **Lazy PLE Row Access**:
  - The PLE table (`per_layer_token_embd`) is ~320M rows / ~51B scalar parameters, and tens of GB on disk depending on representation (e.g. 28.8 GB on Q2_K_XL to 51 GB on 8-bit). Note: **row count $\neq$ parameter count $\neq$ file size**.
  - It **must not** be loaded into managed RAM via `new byte[]`.
  - Normal weight access is sequential-ish where OS mmap paging works naturally. PLE access is sparse/random row access across ~320M rows.
  - Follow TensorSharp and llama.cpp: keep the table memory-mapped on disk and read/dequantize only the specific gathered head rows required for each token.

---

## Current Status & Summary Matrix

The initial structural port, semantic implementations, and synthetic fixtures are complete. All unit and synthetic integration tests pass. The gate to production admission is real-weight qualification on the 72.5 GB checkpoint:

| Phase | Subsystem | Description | Status |
| :--- | :--- | :--- | :--- |
| **A** | PLE | Parse hash metadata arrays (`PleLayerMultipliers`, `PleHeadOffsets`, `PleHeadVocabSizes`) | **Implemented (synthetic verified)** |
| **B** | PLE | Memory-mapped PLE table row access via `PerLayerTokEmbd` | **Implemented (synthetic verified)** |
| **C** | PLE | Stateful n-gram hash window & row gather across decode calls (`Qwen4ExpPleHasher`, n-gram head grouping, EOS truncation) | **Implemented (synthetic verified)** |
| **D** | PLE | $16 \times 160$ row concatenation into 2560-wide vector feeding `ExecutePle` projection math | **Implemented (synthetic verified)** |
| **E** | QSA | Dedicated raw indexer-K cache separate from main KV cache | **Implemented (synthetic verified)** |
| **F** | QSA | K-pool layout, chunk-boundary completion state, and active tail retention | **Implemented (synthetic verified)** |
| **G** | QSA | Pooled-K arithmetic: block mean + RMSNorm | **Implemented (synthetic verified)** |
| **H** | QSA | Indexer Q RoPE and pooled-K RoPE ($\theta = 10\text{M}$, partial rotary 0.25, unrotated dims passthrough) | **Implemented (synthetic verified)** |
| **I** | QSA | Main attention Q & K four-section IMRoPE (`[11, 11, 10]`, partial rotary 0.25) | **Implemented (synthetic verified)** |
| **J** | QSA | Multi-head ReLU scoring + top-k block selection (`indexer_top_k / kpool`) | **Implemented (synthetic verified)** |
| **K** | QSA | Sparse attention mask generation & selected-block attention walk | **Implemented (synthetic verified)** |
| **L** | MoE | Fused `ffn_gate_up_exps` vs separate `ffn_gate_exps`/`ffn_up_exps` compatibility | **Implemented (synthetic verified)** |
| **M** | Tests | Chunk-boundary, incremental decode, K-pool tail retention, n-gram grouping, RoPE geometry, and MoE parity tests | **Implemented (synthetic verified)** |
| **N** | Gate | Remove `hp.IndexerTopK > 0` constructor refusal guard | **Implemented (synthetic verified)** |
| **O** | Validation | 72.5 GB UD-IQ1_S paged real-weight execution | **Deferred (awaiting download/run)** |
| **P** | Validation | External logit parity against llama.cpp / TensorSharp | **Deferred (awaiting download/run)** |

---

## Detailed Execution Phases

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

---

### Phase 6 — 72.5 GB Real-Weight Qualification (Deferred / Paged Run)
*Note: Gated on Phase 5 completion. The 72.5 GB UD-IQ1_S checkpoint does not fit in 64 GB host RAM, but fits the 279 GB scratch disk. Execution must be strictly paged and correctness-only.*

- [ ] **6.1 Sharded GGUF Loading & Tensor Inventory**:
  - Verify all split shards load via memory-mapped pointers.
  - Verify `per_layer_token_embd` remains mapped on disk without managed RAM allocation.
- [ ] **6.2 PLE Sparse Touch Verification**:
  - Instrument `Qwen4ExpPleRowStore` to assert that a single token decode touches only $H_{ple}$ rows (e.g. 8–32 rows), not scanning the 51 GB table.
- [ ] **6.3 Paged Correctness Run**:
  - Run short prompt (single token prefill + 5 decode steps).
  - Run prompt crossing a K-pool boundary ($L > 4$).
  - Run prompt crossing the sparse selection threshold ($L > \text{indexer\_top\_k}$).
- [ ] **6.4 External Parity Gate**:
  - Compare generated greedy token sequence and logit distributions against llama.cpp (`bed0a8566`) or TensorSharp.
  - Upon matching output: update status to **REAL-WEIGHT VERIFIED** and consider admission into `ModelCompatibility.cs`.

---

## Out of Scope / Deferred
- **Vision Tower**: Qwen3.5-VL mmproj, (T,H,W) IMRoPE, video decoding.
- **MTP (Multi-Token Prediction)**: Speculative decoding head (`nextn`).
- **Batched GPU Kernels**: CUDA/Vulkan graph paths.
