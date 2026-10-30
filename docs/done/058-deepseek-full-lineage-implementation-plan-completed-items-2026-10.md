# Completed items split from 058-deepseek-full-lineage-implementation-plan.md (2026-10-30)

Finished items moved verbatim out of [../reference/058-deepseek-full-lineage-implementation-plan.md](../reference/058-deepseek-full-lineage-implementation-plan.md). Open items stay there. Each block notes the section it came from.

> from: ## Progress: Phase 0 alpha, 2026-09-02
- [x] Implement raw/HCA/CSA attention (see the later CSA implementation update); real-weight
  correctness remains unverified.

> from: ### Overview Checklist
- [x] **Phase 2 — `deepseek2` (V2/V3/R1)**: Admitted and verified in `ModelCompatibility.cs` / `ForwardPass.cs`.

> from: ### Top-k masking and the raw+compressed attention concat (the "easy" remaining piece)
- [x] Compute `top_k` indices over the LID-compressed blocks (`build_lid_top_k`, deepseek4.cpp:
   608-703) — score = `sum_head(relu(indexer_q · indexer_k) * indexer_weight)` per compressed LID
   block, masked additively by causality, top-`indexer_top_k` selected. Already have
   `LightningIndexerScore`/`SelectTopKIndices` for this; need the indexer Q/K projections
   (`indexer_attn_q_b(qr)`, rope, Hadamard) and the per-block LID K read wired.

> from: ### Top-k masking and the raw+compressed attention concat (the "easy" remaining piece)
- [x] Build a combined key/value sequence: raw recent-token cache (as today) + ALL CSA-compressed
   blocks (as HCA already does for its own stream) — but the attention MASK for the compressed
   portion is not "attend to everything" (HCA's current behavior) — it is "attend only to the
   `indexer_top_k` blocks selected in step 1, `-inf` elsewhere" (`build_top_k_mask`,
   deepseek4.cpp:705-732). This is a straightforward per-position mask array, mechanically simple
   once the top-k indices exist.

> from: ### Top-k masking and the raw+compressed attention concat (the "easy" remaining piece)
- [x] Softmax-weighted-V sum exactly as today's raw/HCA attention already does, just against the
   masked combined sequence.

> from: ### Summary: ordered list of what to build, with dependencies
- [x] 2. **Raw-token-granularity overlap state store** — a new, `2*headDim`-wide, never-discarded
   per-token row list per layer (distinct from `DeepSeek4CompressedLayerState`'s per-block
   granularity), for BOTH the main CSA stream and the separate LID stream (two instances per CSA
   layer).

> from: ### Summary: ordered list of what to build, with dependencies
- [x] 3. **The prev/cur overlap gather** (the "concrete gather algorithm" section above), feeding the
   already-implemented `DeepSeek4Graph.CsaCompressBlock` — implement once, reuse for both the main
   CSA stream and the LID stream (same algorithm, different tensors).

> from: ### Summary: ordered list of what to build, with dependencies
- [x] 4. **Lightning indexer Q/K projection + top-k scoring wiring** — mostly assembling already-ported
   pieces (`LightningIndexerScore`, `SelectTopKIndices`) around the new LID compression stream
   from step 3.

> from: ### Summary: ordered list of what to build, with dependencies
- [x] 5. **`build_top_k_mask`-equivalent masking + the raw+compressed concat attention step** — the most
   mechanical piece, closely mirroring `RawAttention`'s existing structure and HCA's
   `GetKeyOrCompressed`.

> from: ### Summary: ordered list of what to build, with dependencies
- [x] 6. **Wire into `DeepSeek4ForwardPass`**: extend the constructor's ratio gate to accept 4, extend
   `RawAttention` (or split into a `CsaAttention` method) to call the above instead of throwing.

