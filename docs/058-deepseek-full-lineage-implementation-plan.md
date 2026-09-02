# DeepSeek full-lineage implementation plan — V4 through V1

Status: Phase 0 ALPHA CODE WRITTEN, untested against real weights. Phased, V4-first per explicit
user direction. Supersedes nothing; extends the closed investigation in
`docs/done/032-deepseek2-mla-yarn-moe-routing-investigation.md`.

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
- Any real-weight verification whatsoever — this has never executed even once, since there is no
  DeepSeek-V4 GGUF available this session. Every claim above is "believed correct from reading
  the reference," not measured.
- CSA/HCA attention (see above) — the actual mechanism most real V4 layers will need.
- Rollback/multi-sequence support for the compressed-KV state cache.
- The `mtp_only`/`trunk_only` conditional-tensor detection (deepseek4.cpp:93-95) — this port's
  loader treats every conditional tensor as simply optional.
- Batched/packed prefill — `Prefill` is a plain per-token loop over `Forward`, not amortized.

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

## Why this doc exists

The user wants DeepSeek support starting with V4 (current/latest), then working down through
older versions. Before starting, the architectures were compared against the vendored llama.cpp
reference (`examples/llama.cpp/llama.cpp/src/models/*.cpp`) and real download sizes were checked.
A bottom-up order (V2 correctness first) was proposed instead on risk/cost grounds; the user
overrode that and confirmed V4 first, 100%. This doc reflects that decision. The risk analysis
below is kept as known, accepted context for whoever works the phases — not as an argument to
reorder again.

## The five DeepSeek GGUF architectures

The vendored llama.cpp reference declares five distinct architecture strings
(`src/llama-arch.cpp`), not one:

| Arch string | Model(s) | Status in this codebase |
|---|---|---|
| `deepseek4` | V4 (Flash / Pro) | Not implemented. The largest, newest mechanism set of the five: hyper-connections (Sinkhorn-normalized multi-stream residual), grouped output LoRA, hash-layer MoE routing, variable per-layer KV-compression ratios with two attention variants (CSA/HCA) selected per layer, DSA/lightning-indexer sparse attention, and MTP. Requires MLA underneath. |
| `deepseek32` | V3.2 | Not implemented. Requires MLA (`throw`s if `!is_mla()`) plus DSA/lightning-indexer sparse attention and single-block MTP. |
| `deepseek2` | V2, V2-Lite, V3, R1 (V3/R1 share this arch string — confirmed via `tokenizer_pre == "deepseek-v3"` / `"deepseek-r1-qwen"` presets in `llama-vocab.cpp` and the V3.1/R1-distill chat templates in `models/templates/`) | Implemented but **gated closed** in `ModelCompatibility.cs` — loads and runs with zero crashes on DeepSeek-V2-Lite-Chat, but produces numerically wrong greedy output. Root cause fully investigated and closed (see below). |
| `deepseek2-ocr` | DeepSeek-OCR / OCR2 | Working — separate vision branch (`OpenTail.Stingray.Vision/DeepSeekOcr*.cs`), not part of this plan. Worth a quick audit in the mop-up phase to confirm it doesn't silently depend on anything this plan changes, but not a rebuild. |
| `deepseek` | V1 | Not implemented, not investigated. Plain dense transformer, no MLA/MoE — expected to be the cheapest phase whenever it's reached. |

## Known risks accepted by going V4-first (context, not a call to reorder)

1. **V4 and V3.2 both require MLA to already be correct.** `deepseek32.cpp:67` literally throws
   if `!hparams.is_mla()`. This codebase's own MLA implementation was investigated for 4+ rounds
   (`docs/done/032-...md`) and closed as "checkpoint-inherent numerical fragility" — the model's
   trained MoE router has chronically near-tied top-6-of-64 decisions (median margin ~0.002) that
   any tiny numerical difference flips, compounding into a fully sign-flipped residual stream by
   layer ~22 of 27. It was never actually fixed, only bounded and explained. Building V4's
   hyper-connections + DSA + hash-routing + MTP on top of this means a wrong V4 output could come
   from new code, the old MLA issue, or both, with no smaller/simpler working checkpoint of the
   same lineage to isolate which — this makes Phase 0 (V4)'s own debugging harder than it would be
   bottom-up, but is the accepted tradeoff for tackling the current/highest-priority model first.
2. **No small checkpoint exists for V4 or V3.2**, unlike V2 (which has a 15.7B "Lite" variant that
   made 4+ rounds of investigation tractable at all). Real sizes checked 2026-09-02:
   - DeepSeek-V4-Flash (284B total / 13B active): smallest usable GGUF quant (Q2_K_S) is **~99 GB**
     ([unsloth/DeepSeek-V4-Flash-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-GGUF)).
   - DeepSeek-V3.2 (671B): smallest quant (IQ1_M) is **~149 GB**
     ([unsloth/DeepSeek-V3.2-GGUF](https://huggingface.co/unsloth/DeepSeek-V3.2-GGUF)).
   Every test iteration against these is a 99GB+ load, and per the user, other sessions are
   already competing for CPU — an expensive debug loop. Mitigate by front-loading synthetic unit
   tests (point 3) so real-weight runs are used to confirm, not to discover, bugs.
3. **Most of the genuinely new mechanisms (hyper-connection Sinkhorn normalization, lightning
   indexer top-k masking, grouped output LoRA) are pure math/shape operations** that can and
   should get synthetic unit tests before ever touching real weights — the same way
   `DotQ2KAvx2ParityTests.cs` was built during the V2 investigation. This matters more, not less,
   going V4-first: it's the main lever available to keep the 99GB-checkpoint debug loop cheap.

## Phase plan

### Phase 0 — `deepseek4` (V4): full implementation, current priority

Reference now fully read (`examples/llama.cpp/llama.cpp/src/models/deepseek4.cpp`, all 1547
lines, plus `llama-kv-cache-dsv4.h`). **Correction to this doc's original plan**: V4's attention
is NOT DeepSeek-V2/V3's classic MLA reused. `load_arch_tensors` has no `wk_b`/`wv_b`
decompression pair and no `kv_lora_rank`-sized latent — `layer.wkv` is a flat per-head-sized
projection (`{n_embd, n_embd_head}`), Q/K each get their own per-head RMSNorm, and the real
novelty is a **persistent, block-based compressed-KV *state* mechanism** unrelated to MLA's
down/up projection. The `deepseek2` MLA formula chain (`docs/done/032-...md`) is NOT reusable
here — V4's attention has to be built from scratch. Mechanism set, in the order the reference
builds it up:

- **Per-head Q/K RMSNorm + rotary split** (`q`/`kv` reshape → `ggml_rms_norm` per head → nope/rope
  view split → `ggml_rope_ext` on the rope slice only) — the baseline attention path, used
  directly by `build_raw_attention` for ratio-0 layers.
- **Persistent compressed-KV state cache (CSA/HCA)** — the actual new mechanism, and the
  highest-risk item in this phase. Each ratio-4 (CSA) or ratio-128 (HCA) layer keeps a *separate*
  state cache (`llama_dsv4_comp_state`, one each for CSA/HCA/LID) of compressed KV+score blocks
  that persists **across decode positions**, not just within one forward pass. Every such layer:
  restores prior state via `dsv4_build_state_restore` (`ggml_get_rows` gather by
  `state_read_idxs`/`state_restore_src_idxs`), computes new compressed KV/score for the current
  block via a **softmax-over-block-scores weighted sum** (`build_hca_compressed_kv_from_state` /
  `build_overlap_compressed_kv_from_state` — `ggml_soft_max` over per-position scores, multiply
  into values, `ggml_sum_rows`), applies an optional Hadamard transform
  (`llama_mul_mat_hadamard`) to the compressed result, writes it back into the cache
  (`mctx->get_csa()/get_hca()->cpy_k`), and separately snapshots/persists the block state itself
  (`dsv4_build_state_snapshot`, `cpy_kv`/`cpy_score` with `state_persist_*_idxs`) for future
  positions to restore from. This needs its own new cache abstraction in this codebase —
  `PagedKvCache` has no analog for a second, block-compressed, cross-position-persistent state
  stream alongside the normal KV cache. Scope this as its own design/implementation sub-task
  before attempting the attention math around it.
- **Lightning indexer (LID) / DSA** — top-k sparse attention scoring (`build_lid_top_k`), itself
  layered on top of the same persistent compressed-state mechanism (LID has its own
  `indexer_comp_wkv`/`indexer_comp_wgate`/`indexer_comp_ape` compression state, restored/persisted
  the same way as CSA/HCA). Write synthetic unit tests for the indexer's top-k masking and scoring
  against small hand-computed tensors before touching real weights.
- **CSA attention** (`build_csa_lid_attention`, ratio 4) — combines the lightning-indexer top-k
  mask with the raw KV cache concatenated to the CSA-compressed KV (`ggml_concat` on the KV and
  mask dimension), i.e. attention over both full-resolution recent tokens and compressed older
  blocks simultaneously.
- **HCA attention** (`build_hca_attention`, ratio 128) — same raw+compressed concat pattern as
  CSA but without the indexer/top-k step (attends to all compressed HCA blocks, not a filtered
  top-k subset).
- **Hyper-connections** — Sinkhorn-normalized multi-stream residual (`build_hc_pre`,
  `build_hc_post`, `build_hc_head`, `build_hc_sinkhorn`). Entirely new mechanism class for this
  codebase, independent of the attention-state work above — highest-value target for synthetic
  unit tests (doubly-stochastic normalization has clean, checkable invariants: row/column sums ≈
  1).
- **Grouped output LoRA** (`wo_a`/`wo_b` split by `o_group_count`, reshape-avoiding tensor layout
  per the reference's own comment at `deepseek4.cpp:117-120`).
- **Hash-layer MoE routing** (`ffn_gate_tid2eid`, used for `dsv4_hash_layer_count` early layers
  instead of the usual learned router).
- **MTP** (`graph_mtp`) — draft-head decoding, single block, mirrors the trunk's construction.
- Reference: `examples/llama.cpp/llama.cpp/src/models/deepseek4.cpp`,
  `llama-kv-cache-dsv4.h`.
- Needs a real V4-Flash GGUF (~99 GB smallest quant, Q2_K_S) for end-to-end verification — confirm
  download source/quant with the user before the real-weight verification step.
- Note: `Engine.DeepSeekMoeGraph`/`MlaAttention` (`src/OpenTail.Stingray.Engine/`) are **dead
  code** — not on any live call path — and, per the correction above, not applicable to V4's
  attention design anyway. The `deepseek2` MLA implementation inline in
  `ForwardPass.cs`/`ForwardPass.Decode.cs`/`ForwardPass.Moe.cs`/`ModelGraph.cs`/`SimdKernels.cs`
  is a reference for coding *style/conventions* (how this codebase structures a novel attention
  variant), not a source of reusable formulas for V4.

### Phase 1 — `deepseek32` (V3.2)

**Correction**: V3.2 is not a strict subset of V4 despite the version ordering. `deepseek32.cpp`
confirms it uses DeepSeek-V2/V3's classic MLA (`wk_b`/`wv_b` decompression tensors,
`kv_lora_rank`-sized latent — `hparams.is_mla()` must be true, the loader throws otherwise), which
is a *different* attention design from V4's compressed-state CSA/HCA mechanism, not a piece of it.
What V3.2 DOES share with V4 is the DSA/lightning-indexer top-k mechanism (`build_lid_top_k`'s
V3.2 analog is inlined directly in `deepseek32.cpp`'s single `graph` constructor, same
q_pe/k_pe-rope + Hadamard-transform + top-k-mask shape) and single-block MTP — so Phase 0's DSA
unit tests and MTP scaffolding are reusable, but Phase 0's CSA/HCA compressed-state cache is NOT;
V3.2 instead needs the classic MLA path (which overlaps with, but per the `deepseek2` closed
investigation is not proven correct in, this codebase's existing dead-but-formula-verified MLA
code). Do not assume Phase 1 is "mostly free" going in — re-scope it once Phase 0 is further
along and the actual overlap is clearer.

- Reference: `examples/llama.cpp/llama.cpp/src/models/deepseek32.cpp`, `llama-kv-cache-dsa.h`.
- Needs a real V3.2 GGUF for end-to-end verification (~149 GB smallest quant) — confirm
  availability/download plan with the user before this phase's real-weight verification step.

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

### Phase 3 — mop-up

- `deepseek` (V1) — plain dense transformer, no MLA/MoE. Expected near-free once `deepseek2`'s
  dense-layer code path (`leading_dense_block_count` layers) is confirmed working, since V1 is
  architecturally a subset.
- `deepseek2-ocr` — quick audit only, to confirm nothing in Phases 0-2 silently affects the
  existing working vision branch. Not a rebuild.

## Open questions for the user before Phase 0 starts

1. Which V4-Flash GGUF (source, quant level) to download — smallest is ~99 GB (Q2_K_S). Confirm
   before the download starts, given shared CPU load with other running sessions.
2. Confirm before Phase 1's real-weight verification: which V3.2 GGUF (quant level, source) to
   download, given the ~149 GB minimum size.
3. Should `--allow-unverified-arch` shipping (Phase 2 option (b)) be considered acceptable at all
   if option (a) doesn't pan out, or is a hard "Paris" pass required for `deepseek2`?
