# Qwen 3.8 Flash Next (qwen4exp) - completed items 2026-11

Ticked and implemented rows moved from [2026-10-03-qwen38-flash-next-port-plan.md](../2-coverage/2026-10-03-qwen38-flash-next-port-plan.md) on 2026-11-08. Rows O and P stay there (real-weight run and external parity are still open).

## Summary matrix: implemented rows (synthetic verified)

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
