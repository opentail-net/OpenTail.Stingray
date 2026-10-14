# DeepSeek full-lineage plan: Phase 0 alpha log (2026-09-02) and resolved Phase 2

> **ARCHIVED (2026-10-14).** Split verbatim out of [../2-coverage/058-deepseek-full-lineage-implementation-plan.md](../2-coverage/058-deepseek-full-lineage-implementation-plan.md). Part 1 is the dated alpha progress log for `deepseek4` (code written, never run on real weights); part 2 is Phase 2 (`deepseek2`), which is admitted and verified. The live plan keeps the open Phase 0/1/3 work and the CSA design.

## Part 1: Progress log, Phase 0 alpha, 2026-09-02

## Progress: Phase 0 alpha, 2026-09-02

Per explicit user direction ("write the deepseek4, fully... even an unproven version is a start...
we just need to be explicit that it's untested"), the algorithmic core of V4's mechanism set was
written as a self-contained, clearly-labeled alpha module rather than threaded through the live
GGUF-loading/dispatch pipeline blind (no way to compile-test against a real GGUF this session —
threading it through the 1200+/several-thousand-line `ModelGraph.cs`/`ForwardPass*.cs` machinery
without that risked silently corrupting working architectures):

- `src/OpenTail.Stingray.Engine/DeepSeek4Alpha.cs` — `DeepSeek4Hyperparams` (every GGUF key
  `load_arch_hparams` reads, deepseek4.cpp:18-77) and `DeepSeek4Graph` (hyper-connection Sinkhorn
  normalization + pre/post mixing, lightning-indexer scoring + top-k selection, CSA/HCA
  compressed-KV-block math, hash-layer routing lookup). Every method's XML doc cites the exact
  reference function and line range it was ported from. Builds clean, 0 warnings (this repo's
  `TreatWarningsAsErrors`).
- `tests/OpenTail.Stingray.Tests.Core/DeepSeek4AlphaTests.cs` — 8 synthetic tests, all passing.
  These confirm internal self-consistency only (Sinkhorn output is genuinely doubly-stochastic,
  ReLU/mask/top-k behave as coded) — **not** correctness against DeepSeek-V4's real behavior or
  the llama.cpp reference, which needs real weights this session doesn't have.
- `ModelCompatibility.cs` — added a `deepseek4` comment block (matching the existing `deepseek2`
  documentation pattern) recording that alpha code exists, is unwired, and must not be admitted.

**Update, same day — compressed-KV state cache + GGUF hyperparameter loader added**:
- `src/OpenTail.Stingray.Engine/DeepSeek4CompressedState.cs` — `DeepSeek4CompressedLayerState`
  (persist/read-back compressed KV+score blocks) and `DeepSeek4CompressedState` (the three
  per-layer streams: CSA, HCA, LID, allocated per layer according to that layer's compress
  ratio). **Deliberately a simplified substitute for the reference's actual state cache
  (`llama-kv-cache-dsv4.h`/`.cpp`), not a faithful port**: the reference's ~400-line class is
  mostly rollback/snapshot bookkeeping (`state_restore`/`state_snapshot`/`state_persist`, a
  "reserve_plan" mechanism, multi-stream sequence-copy support) built for llama.cpp's
  speculative-decode-with-rewind and multi-sequence batched serving. This port keeps only
  persist-then-read-back for a single decode sequence with no rewind — almost certainly
  sufficient for a first straight-line greedy-decode verification (one prompt, temp 0, no
  speculative decoding), explicitly NOT sufficient for wiring into this engine's
  `MtpDecoder`/`DSparkDecoder`/`ContinuousBatchingEngine` without further work. Flagged
  prominently in the file so nobody assumes rollback support exists.
- `DeepSeek4Hyperparams.FromGgufMetadata` (appended to `DeepSeek4Alpha.cs`) — reads every
  `deepseek4`-specific GGUF key using the exact `LLM_KV_*` key strings confirmed by grepping
  `llama-arch.cpp` directly (not assumed from naming convention). Pure metadata parsing, no
  tensor access — testable with a synthetic dictionary, same pattern `DeepSeek2Tests.cs` uses for
  the shared hyperparameter loader.
- Test suite grew to 15 (was 8), all passing: added coverage for the metadata loader (including
  the reference's own `compress_ratios`-too-short hard-error, deepseek4.cpp:58-60) and the
  compressed-state cache (persist/read-back round-trip, contiguous-range gather, per-layer
  CSA/HCA/LID allocation matching each layer's ratio, clear-on-reset).

**Update, same day — GGUF tensor resolution wired**:
- `src/OpenTail.Stingray.Engine/DeepSeek4TensorSet.cs` — `DeepSeek4TensorRef` (a zero-copy
  resolved-tensor wrapper, matching this codebase's existing private `TensorRef` pattern in
  `ForwardPass.Helpers.cs`/`HybridGdnForwardPass.cs`, duplicated here since those are `private` to
  their classes) and `DeepSeek4TensorSet.Load(GgufModel, DeepSeek4Hyperparams)` — resolves every
  deepseek4 tensor by name against a real `GgufModel`, tensor-for-tensor matching
  `load_arch_tensors` (deepseek4.cpp:79-178). **Deliberately does NOT touch
  `OpenTail.Stingray.Core.ModelGraph.cs` or any `ForwardPass*.cs`** — confirmed by reading
  `ForwardPass.cs` that tensor loading in this codebase is done per-architecture, inline, directly
  against `GgufModel`, not through a shared per-architecture registry, so this can live in its own
  file with zero risk to any existing architecture's loading path.
- Every GGUF tensor-name string was cross-checked against `llama-arch.cpp`'s actual
  `LLM_TENSOR_*` name table by direct grep, not assumed from the C++ enum's own naming — this
  caught several real naming mismatches a naive port would have gotten wrong: `LLM_TENSOR_ATTN_KV_NORM`
  maps to GGUF string `attn_kv_a_norm` (not `attn_kv_norm`), `ATTN_OUT_A`/`B` map to
  `attn_output_a`/`attn_output_b` (not `attn_out_a`/`b`), and the `HC_HEAD_*` tensors are the one
  hyper-connection tensor family that is NOT per-layer (`output_hc_fn`, no `blk.%d.` prefix) unlike
  every other `HC_*` tensor (`HC_ATTN_*`/`HC_FFN_*`, which are per-layer).
- Solution-wide build attempted to confirm no regressions: `src/OpenTail.Stingray.Engine` and the
  full solution both build clean for every project this session touched. One unrelated,
  pre-existing failure exists in `StableAudioPipeline.cs` (another session's in-progress work,
  visible in this session's starting git status) — not caused by, and not fixed by, this work.

**Update, same day — forward-pass dispatch wired (RAW ATTENTION ONLY)**:
- `src/OpenTail.Stingray.Engine/DeepSeek4ForwardPass.cs` — implements `IForwardPass`, driving
  embedding lookup, the full per-layer hyper-connection wrap (attn and FFN sides), MoE FFN (with
  hash-layer routing and sqrt-softplus gating — see below), the shared expert, and the final
  hc_head/output-norm/LM-head sequence, end to end for a real prompt.
- **Explicit, load-bearing scope limit**: only `compress_ratio==0` ("raw attention") layers are
  implemented. The constructor throws `NotSupportedException` immediately if ANY layer has a
  non-zero ratio (CSA/HCA) — checked eagerly rather than discovered mid-generation. Since CSA/HCA
  is DeepSeek-V4's headline mechanism, a real V4-Flash checkpoint (which almost certainly uses
  CSA/HCA on most layers) is NOT expected to construct successfully today. This class exists to
  prove the hyper-connection/MoE/tensor-loading wiring is structurally coherent and to give a real
  starting point for CSA/HCA attention, not to produce usable V4 output yet.
- Added `DeepSeek4Graph.SqrtSoftplusGate`/`SelectAndWeightExperts` (V4's loader hard-requires
  `LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS`, a gating function not previously ported anywhere
  in this codebase — every existing MoE architecture uses softmax or sigmoid) and
  `HyperConnectionHeadGate` (the single-gate, no-post/comb, no-Sinkhorn variant `build_hc_head`
  uses once at the very end of the trunk, distinct from the per-layer `HyperConnectionGate`).
- **Two specific pieces of the raw-attention port are high-risk and explicitly flagged
  in-file as not independently verified** (see the file's header for full detail):
  1. The raw path's `kv` projection is genuinely MQA-with-K==V — ONE cached
     `n_embd_head`-wide vector per token, used as both the attention key AND value (inferred by
     reading `build_raw_attention`'s call sites, not confirmed against an executed trace).
  2. The reference applies `ggml_rope_ext_back` (an inverse rotation) to the attention OUTPUT's
     rope-dim slice before recombining with the nope slice (deepseek4.cpp:1252-1258) — unusual;
     most architectures never touch RoPE after attention. Implemented here as "rotate by
     `-position`," which is what "_back" suggests, but `ggml_rope_ext_back`'s actual
     implementation was never read to confirm that interpretation.
  3. A third, smaller gap: the grouped output LoRA (`wo_a`/`wo_b` split by `OutputGroupCount`) is
     implemented as an UNGROUPED down-projection — correct only when `OutputGroupCount == 1`.
     Flagged in-file; needs the per-group reshape before trusting it for a checkpoint that
     declares more groups.
- All builds clean (`src/OpenTail.Stingray.Engine`, `tests/OpenTail.Stingray.Tests.Core`), 0
  warnings; the existing 15 DeepSeek4Alpha tests still pass unchanged (this addition didn't touch
  any function they exercise, only added new ones).

**Explicitly NOT done yet**:
- [ ] Run any real-weight verification — this has never executed even once, since there is no
  DeepSeek-V4 GGUF available this session. Every claim above is "believed correct from reading
  the reference," not measured.
- [x] Implement raw/HCA/CSA attention (see the later CSA implementation update); real-weight
  correctness remains unverified.
- [ ] Add rollback/multi-sequence support for the compressed-KV state cache.
- [ ] Add `mtp_only`/`trunk_only` conditional-tensor detection (deepseek4.cpp:93-95) — this port's
  loader treats every conditional tensor as simply optional.
- [ ] Add batched/packed prefill — `Prefill` is a plain per-token loop over `Forward`, not amortized.

**Update, same day — HCA (compress_ratio==128) attention added, CSA (ratio==4) still deferred**:
- `DeepSeek4ForwardPass`'s constructor now accepts `compress_ratio` 0 (raw) or 128 (HCA); ratio 4
  (CSA) still throws. Reason for the split: the compression projection's output width is
  `coff*n_embd_head` where `coff = (ratio==4 ? 2 : 1)` (deepseek4.cpp:129-136). HCA (`coff=1`) is
  one compressed row per raw token — structurally identical to the "one token in, one compressed
  row out" model already built. CSA (`coff=2`) produces TWO sub-rows per token via an overlapping-
  window scheme (`build_overlap_compressed_kv_from_state`, deepseek4.cpp:524-606) that could not
  be reverse-engineered with confidence in the time available; implementing it wrong seemed worse
  than leaving it explicitly unimplemented. CSA also needs the lightning indexer's top-k mask
  folded into its attention, which HCA does not. **A real V4-Flash checkpoint likely uses both
  ratios across its layers, so this class still cannot run such a checkpoint end-to-end** — it now
  covers two of three attention variants, not all three.
- Added, for HCA layers: per-token compressed-KV/score projection (`attn_comp_wkv`/
  `attn_comp_wgate` + the `attn_comp_ape` positional lookup table), block accumulation (buffers
  `ratio`=128 raw tokens, then finalizes via the already-ported `DeepSeek4Graph.HcaCompressBlock`
  and persists to `DeepSeek4CompressedState`), and attention over the concatenation of the raw
  recent-token cache with all persisted compressed blocks (matching `build_hca_attention`'s
  raw_k + hca_k concat ordering, deepseek4.cpp:832-833).
- One further simplification flagged in-file: a finalized HCA block's "score" companion is
  persisted as a zero vector, since HCA's non-overlapping scheme never reads a prior block's raw
  score again (only CSA's overlapping scheme would, and CSA isn't implemented) — only the
  finalized block's KV is ever consumed downstream, by attention.
- Builds clean, 0 warnings; all 15 existing tests still pass unchanged (no new tests added this
  round — the HCA logic depends on multi-token sequencing across `Forward` calls in ways that are
  harder to pin with a single synthetic case than the earlier per-call math was, and token budget
  was tight; a real synthetic multi-token test is a good next increment before touching real
  weights).

**Next steps, in order**: (1) a synthetic multi-token test exercising HCA's block-boundary logic
(e.g. confirm a block finalizes exactly at token 128, not 127 or 129, and that attention's key
count grows by exactly one after finalization); (2) CSA (ratio==4) attention, once the overlapping-
window `coff=2` scheme is understood with more confidence — worth a dedicated reading pass of
`build_overlap_compressed_kv_from_state` before attempting it blind; (3) get real ground-truth
intermediates the same way `docs/done/032-...md` did for deepseek2 — build DeepSeek-V4 in
llama.cpp's eval-callback tooling (if/when a small enough checkpoint or a synthetic tensor fixture
exists) and diff layer-by-layer against this port's own trace output, starting with the three
flagged high-risk details (MQA K==V, `rope_ext_back`, grouped output LoRA); (4) get a real
V4-Flash GGUF (open question below) and start the same kind of
ground-truth diffing methodology that eventually cracked real bugs in the `deepseek2`
investigation (`docs/done/032-...md`) — expect this alpha code to have real bugs surface the same
way once actual intermediate values exist to compare against.


## Part 2: Phase 2 — `deepseek2` (V2/V3/R1), resolved

### Phase 2 — `deepseek2` (V2/V3/R1): resolve or explicitly bound the open correctness question

By this point Phase 0/1 will have re-derived and re-verified the MLA/YaRN formula chain in a new
context (V4/V3.2), which may itself shed light on the V2-Lite routing-margin question. Then:

- Option (a): try a **native, larger DeepSeek-V2/V3 checkpoint** — the one lever doc 032
  identified but never tried (V2-Lite's routing-margin flatness may be specific to that
  checkpoint's size/training, not the architecture generally).
- Option (b): if (a) doesn't resolve it, accept the "inherent numerical sensitivity" finding as
  final, admit `deepseek2` behind `--allow-unverified-arch` with the limitation documented in
  `ModelCompatibility.cs` and `README.md`'s status matrix (per CLAUDE.md rule 10).
- Exit criteria: either a passing greedy-parity receipt against a real prompt, or an explicit,
  documented decision to ship known-limited behind the unverified-arch flag.

**Update, 2026-09-02 — Phase 2 formally resolved via option (b)**, per user direction to start
Phase 2 without a new checkpoint-download decision having been made for option (a):
- `--allow-unverified-arch` (`RunCommand.cs:328-331`, wired at `RunCommand.cs:904-913`) already
  exists in this codebase as a general escape hatch for any architecture the gate rejects — no new
  code was needed to make `deepseek2` runnable this way; it was usable the entire time doc 032's
  investigation ran. What Phase 2 actually closes out is the DECISION and the DOCUMENTATION: this
  is now the accepted, permanent resolution for `deepseek2` (not a placeholder pending option (a)),
  and `README.md`'s status matrix (per CLAUDE.md rule 10) now carries a sourced row for it, plus
  new rows for `deepseek4`/`deepseek32` reflecting their real current state (structurally complete
  alpha code, never executed).
- **Option (a) — the native/larger V2 or V3 checkpoint — remains untried and is NOT foreclosed.**
  This is a real, standing option if the user wants to revisit it later (it's the one lever doc
  032 identified but never tried, and could in principle overturn the "inherent" finding for a
  different checkpoint) — closing Phase 2 via option (b) is a decision to stop WAITING on it, not
  a claim that it's been ruled out. Needs an explicit download decision (checkpoint source/size)
  before it could be attempted, same as every other real-weight step in Phases 0-1.
- `ModelCompatibility.cs`'s existing `deepseek2` comment block already documented the closed
  investigation and named `--allow-unverified-arch` as the correct path if the user wants to ship
  known-wrong — no changes needed there; this update is additive (README) rather than corrective.
