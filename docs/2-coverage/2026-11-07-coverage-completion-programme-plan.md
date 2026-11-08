# Model architecture coverage completion programme

> Source: a plan the owner got from ChatGPT, forwarded "not gospel but may be worth it"
> (2026-11-07). Filtered against the actual state of the code and checkpoints this session
> verified, not transcribed as-is. Each per-family deep dive still lives in its own existing
> plan/todo row -- this file is the cross-cutting index and the filtering notes, not a
> replacement for them.

## Objective

Move Stingray's existing alpha ports (`qwen4exp`, `glm-dsa`, `glm5next`, `deepseek4`) from
"loads, synthetic-tested" to "independently verified on real weights," without starting new
ports or a general engine refactor. Follows CLAUDE.md rule 14 throughout: nothing here is
admitted or advertised until it earns that.

## What changed since the original plan was written (read this before picking a phase)

- **Phase 1's premise (Qwen4Exp "incoherent text") is unconfirmed by anything in this repo's
  current tests.** `Qwen4ExpRealWeightsSmokeTests` (the only real-weight test that exists) checks
  finite logits and one decode step's embedding-sum against llama.cpp -- it has never generated
  multi-token text. The "incoherent continuation" claim traces to
  [ported-families-todo.md](ported-families-todo.md)'s Qwen row, from an earlier manual session,
  not a pinned regression. **Do not skip straight to root-causing** -- the plan's own step 1
  (reproduce with a fixed prompt, record the exact token IDs) is a real prerequisite here, not
  boilerplate.
- **DeepSeek V3.2 (`deepseek32`) and DeepSeek V4.1 Flash (`deepseek41`) stay out of scope for
  real-weight work**, confirmed today by trying: V4.1's smallest real quant (UD-IQ1_S) is ~217 GB
  across shards, V3.2's is ~171 GB, both exceed every disk this machine has free in one place.
  2026-11-07 added `DeepSeek41RealWeightsSmokeTests` (metadata-only -- the construction refusal on
  `compress_ratios`/YaRN runs purely off GGUF key/value metadata, confirmed against the real
  checkpoint's small first shard) and recorded `deepseek32` as having no cheap shard option at all
  (its first shard alone is ~45 GB regardless of quant). This matches the original plan's explicit
  V4.1 exclusion and extends the same reasoning to V3.2.
- **`glm-dsa`'s real-weight ceiling is lower than construction.** Its constructor loads every
  layer's tensors immediately with no metadata-only guard (unlike `deepseek4`/`deepseek41`), so
  only a metadata/hyperparameter check is possible without the full ~200+ GB checkpoint on disk.
  2026-11-07's `GlmDsaRealWeightsSmokeTests` did that: every value in `GlmDsaAlpha.cs`'s own spec
  comment (n_head=64, q_lora_rank=2048, kv_lora_rank=512, indexer_n_head=32, indexer_head_size=128,
  indexer_top_k=2048) matched the real GLM-5.2 checkpoint exactly. Phase 2A's "real-weight
  comparison" completion criterion is therefore not reachable on this machine's disk budget;
  rescope it to "hyperparameter parsing verified against real metadata" unless/until more disk is
  available.
- **`glm5next`'s K-pool gap is real and current** (verified 2026-11-07, not from memory):
  `Glm5NextForwardPass.cs:567-574` runs full attention over all cached KV and throws
  `NotSupportedException` once key count exceeds `indexer.top_k` -- confirms Phase 2B's premise
  exactly as written.
- **`deepseek4`'s YaRN refusal is real and current**, confirmed today
  (`DeepSeek4RealWeightsSmokeTests`, real checkpoint, `rope.scaling.factor=16`). Phase 3 is
  accurately scoped.
- **Phase 1 found a second, unrelated bug while reproducing the garbled-text claim**: `Qwen4ExpForwardPass.TruncateTo`
  violated `IForwardPass`'s documented contract (any length besides 0/current must throw
  `NotSupportedException` when `SupportsPartialRewind` is false) by silently no-opping instead.
  `InferenceEngine` never hit this path (it only calls an intermediate `TruncateTo` when
  `SupportsPartialRewind` is true), so it wasn't an observed failure, but it was a live landmine for
  any future caller. Fixed 2026-11-07 to throw like every sibling GDN-hybrid pass already does;
  regression test `TruncateTo_RejectsIntermediateLength_OnlyAcceptsZeroOrCurrent`.
- **Phase 2A got one small optimisation, not a scope change**: TensorSharp's `GlmDsaModel.Attention.cs`
  skips indexer query/scoring work while the cached key count is still `<= indexer_top_k` (selection
  can only ever be "all of them" in that regime). Ported into `GlmDsaForwardPass.RunFullIndexer`
  2026-11-07 -- output is unchanged, only wasted compute on short contexts is removed. This does not
  change Phase 2A's real-weight ceiling (still metadata-only, see above); it is a correctness-preserving
  perf fix on the synthetic-verified path, covered by `GlmDsa_IndexerTopKShortcut_MatchesFullScoringAcrossThreshold`.
- **Phase 3 found a real correctness bug, independent of the YaRN refusal**: `DeepSeek4ForwardPass.MoeFfn`'s
  hash-routed branch (`il < HashLayerCount`) assigned every selected expert a weight of exactly 1.0.
  Checked against both references (CLAUDE.md rule 8, not just TensorSharp) -- `build_moe_ffn`
  (llama-graph.cpp:2129-2146) and `deepseek4.cpp`'s hash-layer call site (1289-1310) confirm hash layers
  run the same router projection and sqrt-softplus gating as every other MoE layer; only the expert
  INDICES come from a hash lookup instead of top-k, the WEIGHT is still gathered from the real router
  probability. Fixed 2026-11-07, along with a second bug found in the same code: the non-hash branch
  added the DeepSeek-V3-style selection bias (`exp_probs_b`) to raw logits before gating and then both
  selected AND weighted from that biased result; the reference adds the bias after gating and only for
  selection, weighting always comes from the unbiased probs. Both share one fix
  (`DeepSeek4Graph.WeightSelectedExperts`). This doesn't touch Phase 3's YaRN scope (still refuses
  correctly, still the real next step) but is now also fixed on the same checkpoint/code path.
- **Phase 2A found the same bias-ordering bug in `glm-dsa`'s sigmoid MoE, confirmed 2026-11-08**:
  `GlmDsaForwardPass.MoeFfn` added `exp_probs_b` to raw logits before sigmoid gating and selected
  AND weighted experts from that biased result -- `build_moe_ffn` (shared by `glm-dsa.cpp` and
  `deepseek4.cpp`) adds the bias after gating, for selection only. Fixed the same way as the
  `deepseek4` fix above, reusing `DeepSeek4Graph.WeightSelectedExperts`
  (`MoeFfn_SigmoidGating_SelectionBiasNeverLeaksIntoTheGatheredWeight`). Also fixed a review finding
  on the indexer shortcut test's name (`GlmDsa_IndexerTopKShortcut_MatchesFullScoringAcrossThreshold`
  only checked finite logits, never an actual output comparison despite its name) -- renamed and
  added a real equivalence proof (`SelectTopKIndices_WhenTopKEqualsItemCount_AlwaysSelectsEveryIndexRegardlessOfScores`).
  Phase 2A's remaining synthetic-correctness checklist (MLA absorption, DSA scoring, Hadamard
  handling, shared-expert contribution against the llama.cpp source) is still open.
- **Phase 2A's MLA absorption and DSA scoring checks done 2026-11-08** (`MlaAttention`/`RunFullIndexer`
  against `glm-dsa.cpp`): MLA absorption matches the reference exactly (Qcur/Kcur concat order, the
  YaRN kq_scale pre-scale formula, wv_b decompression, and the NORM-type adjacent-pair RoPE
  convention, confirmed from `llama-model.cpp`'s `rope_type` table -- `GLM_DSA` uses NORM, not NEOX).
  DSA scoring found one real bug: the indexer's own RoPE (`indexer_q`/`indexer_k`) is YaRN-scaled in
  the reference with the exact same params as the main attention's RoPE, but this port was calling
  a plain, non-YaRN rotation there -- invisible on every checkpoint verified so far (YaRN inactive,
  where the two formulas are identical) but wrong once YaRN is active. Fixed by switching to the
  same `ApplyYarnRope` the main attention path already used, with a test proving the fix isn't a
  no-op (`YarnRope_DivergesFromPlainRopeOnlyWhenYarnIsActive`) and one exercising the fixed path
  end-to-end (`GlmDsa_FullIndexerScoring_WithYarnActive_ProducesFiniteLogits`). Hadamard
  rotation's own correctness was already covered by an existing invariant test.
- **Phase 2A's synthetic-correctness checklist is now complete (2026-11-08)**: shared-expert
  contribution checked against `glm-dsa.cpp:486-498` -- a plain SwiGLU added to the routed MoE
  output, no extra scale, matching `MoeFfn`'s shared-expert block exactly. No bug found. Added
  `SharedExpertSwiGlu_MatchesHandComputedValues` (hand-computed SiLU(x)*x values, not a
  same-logic recomputation) and `MoeFfn_SharedExpertContribution_SurvivesWhenRoutedExpertsAreZeroed`
  (zeroes every routed expert to provably isolate the shared contribution, confirms zeroing it
  changes the output). All five checklist items done: MLA absorption (no bug), DSA scoring (YaRN-RoPE
  bug fixed), Hadamard rotation (already covered), sigmoid MoE weighting (bias-ordering bug fixed),
  shared-expert contribution (no bug). Phase 2A's only remaining gap is real-weight parity, which
  stays out of reach on this machine's disk budget (see above).

## Qwen 3.8 (`qwen4exp`) reconciliation, 2026-11-08 (Phase 0.1 of the Qwen 3.8 correctness plan)

Checked against `Qwen4ExpForwardPass.cs` and the descriptor's `RefusalReason` (`Qwen4ExpArchitecture.cs`).

- **Fixed, verified in code:** the conv1d weight transpose (`LoadConv1dTransposed`-equivalent cache,
  `_convWeightTransposeCache`); SiLU on the GDN conv output (`conv_output_silu`); sigmoid, not SiLU, on the GDN output gate
  (`useSigmoidGate: true`); per-head L2 norm on Q and K before the recurrence. The two regression tests named in this
  file (`Layer0GdnMixerOutput_MatchesLlamaCppEvalCallback_AfterConv1dTransposeFix`,
  `TruncateTo_RejectsIntermediateLength_OnlyAcceptsZeroOrCurrent`) both exist in `Qwen4ExpRealWeightsSmokeTests.cs`.
- **Measurement points differ, not contradict:** the transpose fix was measured at the layer-0 **GDN mixer output**
  (matches llama.cpp within quantisation noise). The descriptor's later figure (layer-0 **post-GDN-and-FFN residual**
  still ~46% off) is a different measurement point. Both hold.
- **Not established:** full-stack numerical parity. Final logits are still off by order of magnitude per the descriptor;
  the layer-by-layer `l_last` trace still grows faster than llama.cpp's reference across the 48 layers.
- **Open, inconclusive:** the FFN injection gate (`hc_ffn_inject`). The tensor loads at full length with sane values; the
  per-stream gates diverge from llama.cpp beyond the known GDN imprecision, and one stream flips sign. Could be a
  remaining bug or amplified upstream error.
- **Paris continuation:** one prompt, 20 greedy tokens. Encouraging, not full-model correctness.
- **Admission:** unchanged. The architecture refusal stays. Qwen 3.8 is not admitted or advertised.
- **Debug output:** the layer-sum prints in `Qwen4ExpForwardPass.cs` are gated by `STINGRAY_Q4E_DEBUG=1`, but the
  gate is read from the environment inside the per-layer loop (lines 265, 354). Phase 0.2 replaces them with the
  structured diagnostic.

### Phase 0.2 progress, 2026-11-08: reference side

- Built `llama-eval-callback` from the pinned reference (`bed0a8566`, `build-ref`, Release, CPU).
- Checkpoint: `E:\_models\qwen38-flash-next\UD-IQ1_S`, shards 1-3 of 3 (62 GB). Prompt `"The capital of France is"`
  (5 tokens), one decode step, `-ngl 0`.
- The stock callback prints only edge values, so it cannot give full vectors. A local dump callback in the git-ignored
  `examples/eval-callback/eval-callback.cpp` writes the first occurrence of each named tensor as raw bytes plus a `.meta`
  file, under `$QWEN_DUMP_DIR`. Names are sanitised (`-` becomes `_`, so `l_last-47` is `l_last_47.bin`).
- Result: 1,671 tensor files, 1.1 GB, for the prompt pass. `l_last` is present for all 48 layers (`l_last_47`:
  `ne=2560,4,1,1`), `result_output` is present (`ne=248320,1,1,1`). Not committed: this is local tooling in a git-ignored
  reference checkout.
### Phase 0.2 progress, 2026-11-08: Stingray side and first comparison

- Stingray dump hook (`STINGRAY_Q4E_DUMP_DIR` + `STINGRAY_Q4E_DUMP_POS`, env-gated, off by default) writes
  `l_last_N`, `linear_attn_out_N`, `hc_combine_N`, `ffn_out_N`, `ffn_moe_logits_N`, `ffn_moe_weights_norm_N`,
  `ffn_moe_topk_N`, `ple_gate_N`, `ple_gated_value_N`, `ple_conv_out_N`, `ple_embd` and `result_output`. Driver:
  `QWEN_STINGRAY_DUMP_DIR=<dir>`, test `Qwen4ExpRefDumpTests` (skipped without that variable or the checkpoint).
- Comparison scripts live in the scratchpad, not the repo.
- **Bug fixed: PLE depthwise conv weight layout.** GGUF stores `blk.N.ple_conv1d.weight` as `[channel][tap]` (bf16),
  `StepDilatedConv` read `[tap][channel]`. Transposed at load (`Qwen4ExpPle.TransposeConvWeights`, cached per tensor).
  Effect on layer 1: `ple_conv_out-1` cosine 0.082 -> 0.99997; full logits cosine 0.975 -> 0.995, relative error
  22% -> 11%. Regression tests: `Qwen4ExpPleLayoutTests` (2 tests, passing).
- **Current measurement (5-token prompt, last position, against llama.cpp `bed0a8566`):**
  - `l_last`: layer 0 relative error 1.5% (cosine 0.9999); layer 4 5.8%; layer 24 13%; layer 47 14%.
  - Layer 0 GDN mixer output (`linear_attn_out_0`) relative error 2.5%; `ffn_out_0` 1.6%.
  - Layer 4 `hc_combine_4` (after the token mixer) 1.8%, but `ffn_out_4` (MoE) 26% (cosine 0.966).
  - Router logits agree to 0.3-1.1% relative. The top-10 expert set differs at the rank boundary in layers 0, 1,
    4 and 5 (near-ties, e.g. layer 4 ranks 7-9 swap experts 410/63/102). A flipped expert changes `ffn_out` a lot.
  - Logits: top-1 agrees (token 11751); relative error 11%, cosine 0.9947.
- **Open:** the 1.5-2.5% error already present at layer 0 is larger than expected for identical quantised weights.
  Candidate, not verified: llama.cpp quantises activations for the quantised matmuls and Stingray does not. The
  near-tie routing then amplifies this. Next step: find the source of the layer-0 error (the GDN mixer is the largest
  contributor) before any further change, and check whether the error is intrinsic (activation rounding) or a
  remaining port bug.
- **Narrowed further (same day):** the layer-0 error is inherited from the attention-side HC mix, not added by the
  GDN projections. Measured at position 4 (relative error, cosine):
  - `model_input_embed` (embedding): exact (0).
  - `hc_mixed_0` (GDN mixer input, after the attention HC mix): 1.08%, cosine 0.99995.
  - `linear_attn_qkv_mixed_0` 0.86%, `conv_output_raw_0` 0.87%, `conv_output_silu_0` 1.2%, `z_0` 0.79%,
    `alpha_0` 0.55%, raw beta vs `beta_0` 0.66%. All of these track the 1% input error.
  - The reference `beta-0` is not a sigmoid (range -3.7 to 7.7), so compare raw beta to it.
- **Next:** dump the intermediates inside `Qwen4ExpGatedResidual.Mix` (grouped RMS norm output = reference
  `hc_norm-0`, `hc_gate-0` and `hc_inject-0`), then find which step introduces the 1%. Candidate, not verified: the
  low-rank HC matmuls, or activation rounding in the reference's quantised kernels.
- Phase 0.3 (FFN injection gate) is still open. These numbers do not yet settle it.

### Parked by owner, 2026-11-08

- **Browser UI** (phase 7F of the Qwen 3.8 plan: a separate browser front end for the local server). Parked.
  This does not affect model image input. Image and video input for Qwen (phase 5) stays active.

## Phases (in order; do not start a later one before the current one's completion criteria)

| # | Family | Checkpoint available? | Status | What's actually left |
|---|---|---|---|---|
| 0 | Coverage baseline | n/a | **Already current** -- [ported-families-todo.md](ported-families-todo.md) and this doc's table above are the baseline; no separate pass needed right now | Re-check before each later phase starts, don't re-derive from scratch |
| 1 | `qwen4exp` (Qwen 3.8 Flash Next) | yes, paged, 62 GB | **ROOT-CAUSED AND FIXED 2026-11-07.** Reproduced the garbled-text claim, then bisected layer-by-layer against a freshly-rebuilt pinned llama.cpp reference (`qwen4exp.cpp` at `bed0a8566`, via `llama-eval-callback`) on the identical prompt/checkpoint: `hc_mixed` and the raw QKV projection matched within ~1%, `alpha`/`beta` projections matched almost exactly, but `conv_output_raw` was off by ~3.5x. Root cause: `blk.N.ssm_conv1d.weight` is stored by GGUF as `[channel][kernel]` (kernel innermost per `ggml_compute_forward_ssm_conv_f32`), the transpose of what `GdnKernels.CausalDepthwiseConv1dDecode`'s `weight[k*channels+c]` access pattern expects. `HybridGdnForwardPass` (Qwen 3.5 hybrid, already verified) already transposes this at load time (`LoadConv1dTransposed`); this port never did. Fixed with the same load-time transpose, cached per layer. After the fix: layer-0 raw GDN-mixer output sum moved from +30.5 (wrong sign, ~12x off) to -2.76, matching llama.cpp's -2.59 within quantisation noise; 20-token greedy continuation of `"The capital of France is"` now reads `"Paris.\nThe capital of France is Paris.\nIs the following sentence true?\n\n<think>\n"` -- correct and coherent. Regression test: `Layer0GdnMixerOutput_MatchesLlamaCppEvalCallback_AfterConv1dTransposeFix`. Also found and fixed the unrelated `TruncateTo` contract violation noted above the same day. | Remaining per the original plan: more fixed prompts (incremental decode, a prompt crossing a prefill/decode chunk boundary, one long enough to exercise QSA K-pool selection), and a documented tolerance for the real-weight comparison. MTP and vision remain deferred. Still alpha / not admitted (CLAUDE.md rule 14) until that broader verification is done |
| 2A | `glm-dsa` (GLM-5.2) | shard-1 metadata only (~9 MB of a much larger checkpoint) | **Synthetic-correctness checklist complete 2026-11-08** (MLA absorption, DSA scoring, Hadamard rotation, sigmoid MoE weighting, shared-expert contribution -- all checked against `glm-dsa.cpp`; two bugs found and fixed, three already correct); indexer top_k-shortcut optimisation ported 2026-11-07 (no behaviour change) | Real forward-pass/parity work needs the full checkpoint, which doesn't fit here -- that's Phase 2A's only remaining gap, and it's a disk constraint, not missing work. Drop the "real-weight comparison" completion criterion unless disk changes |
| 2B | `glm5next` (GLM-5.3-Flash) | paged, large (Q2_K_XL ~236 GB; have UD-IQ1_S shards) | K-pool selection doesn't drive attention yet (confirmed above) | Implement K-pool selection actually gating attention reads, remove the full-attention/throw shortcut, add the isolated numerical tests the original plan lists (KDA recurrence, Sinkhorn mHC, pool construction/selection, sequence isolation). This is real implementation work, not verification -- size it accordingly |
| 3 | `deepseek4` (DeepSeek-V4 Flash) | yes, paged, 86 GB | Refuses YaRN correctly; CSA unexercised. **MoE hash-routing weight bug fixed 2026-11-07** (see above) -- unrelated to YaRN, found while auditing a ChatGPT-sourced claim against the real reference | Implement YaRN rope scaling for real (factor 16, confirmed from the real checkpoint), remove the refusal only once verified against the pinned llama.cpp reference, then exercise non-zero `compress_ratios` (confirmed present: `[0,0,2,2,...]`-shaped) for the first real CSA test |
| 4 | Candidate reassessment | n/a | Not started | Survey Muse Spark / MiniMax M3.1 / others per the original plan's instructions; this is a report, not a port, and is cheap -- can run any time, doesn't block 1-3 |

## Explicitly out of scope (do not start without a new disk/scope decision)

- `deepseek41` (DeepSeek-V4.1 Flash) and `deepseek32` (DeepSeek-V3.2) real-weight work -- checkpoints
  too large for this machine (confirmed 2026-11-07, see above). Metadata-only checks already exist
  for `deepseek41`; `deepseek32` has none and no cheap path to get one.
- Any new model family port (Muse Spark, MiniMax M3.1, etc.) -- Phase 4 is assessment only.
- CUDA work, new GPU requirements, vision towers, MTP/speculative drafting, batching -- all
  explicitly out per the original plan and CLAUDE.md's existing scope rules.

## Checklist (state as of 2026-11-08)

- [x] Phase 0: coverage baseline current (`ported-families-todo.md` + the table above)
- [x] Phase 1, step 1: reproduce the Qwen4Exp garbled-text claim with a fixed prompt (2026-11-07; claim was real)
- [x] Phase 1 root cause and fix: conv1d weight transpose; regression test `Layer0GdnMixerOutput_MatchesLlamaCppEvalCallback_AfterConv1dTransposeFix`
- [x] Phase 1 side fix: `TruncateTo` contract violation (`TruncateTo_RejectsIntermediateLength_OnlyAcceptsZeroOrCurrent`)
- [ ] Phase 1 broader verification: more fixed prompts, incremental decode, a prefill/decode chunk-boundary prompt, QSA K-pool exercise, documented tolerance
- [x] Phase 2A: `glm-dsa` metadata-only real-weight check (`GlmDsaRealWeightsSmokeTests`, 2026-11-07)
- [x] Phase 2A: indexer top_k-shortcut optimisation (output unchanged, 2026-11-07)
- [x] Phase 2A: synthetic-correctness checklist complete 2026-11-08 (MLA absorption, DSA scoring, Hadamard rotation, sigmoid MoE weighting, shared-expert contribution); two bugs fixed, three already correct
- [ ] Phase 2A: real-weight parity. Rescoped to hyperparameter parsing (done); full parity blocked on disk, not missing work
- [ ] Phase 2B `glm5next`: K-pool selection gating attention reads; remove the full-attention/throw shortcut; isolated numerical tests (KDA recurrence, Sinkhorn mHC, pool construction/selection, sequence isolation). Real implementation work
- [x] Phase 3 side fix: `deepseek4` MoE hash-routing weight and selection-bias ordering (2026-11-07/08)
- [ ] Phase 3: implement YaRN rope scaling (factor 16) for real; remove the refusal only after verification against the pinned llama.cpp reference
- [ ] Phase 3: exercise non-zero `compress_ratios` (first real CSA test)
- [ ] Phase 4: candidate reassessment (report only; Muse Spark / MiniMax M3.1 / others)

## Recommended starting point

~~Phase 1, step 1 (reproduce the Qwen4Exp claim with a fixed prompt before touching any code).~~
Done 2026-11-07 -- the claim was real, root-caused and fixed. Next cheapest items, in order, before
taking on Phase 2B/3's real implementation work:
- [ ] 1. Phase 1's remaining broader-verification checklist (more fixed prompts, a prefill/decode chunk
   boundary, QSA K-pool exercise, a documented tolerance) -- no new disk or implementation, just
   more test cases against the checkpoint already on disk.
- [x] 2. Phase 2A's remaining synthetic-level correctness checklist (MLA absorption, indexer scoring,
   Hadamard rotation, sigmoid MoE routing, shared-expert contribution against the llama.cpp source) --
   done 2026-11-08 (see Checklist above).
- [ ] 3. Phase 2B (`glm5next` K-pool) or Phase 3 (`deepseek4` YaRN) only after 1-2, since both are real
   algorithm work, not verification, and should be sized and started deliberately rather than picked
   up as a side effect of something else.

### Phase 0.2 finding, 2026-11-08: the HC mix 1% is a quantised-activation mismatch

- Tensor types (raw GGUF header, shard 2): `blk.0.hc_attn_down` and `hc_attn_up` are Q8_0 (type 8);
  `hc_attn_inject` is BF16 (type 30); `hc_attn_norm` and `ffn_gate_inp` are F32; `attn_qkv` and `ssm_out` are Q5_K (type 13).
- `Qwen4ExpGatedResidual.Mix` reads the Q8_0 weights as float32 and dots them with a float32 activation. llama.cpp's
  quantised matmul rounds the activation to the Q8_1 int8 grid first (blocks of 32, d = amax/127).
- Experiment, env-gated (`STINGRAY_Q4E_HC_Q8ACT=1`, default off): round the matmul inputs (`xn` for down and inject,
  `lo` for up) the same way. Result at layer 0, last token:
  - `hc_mixed_0`: 1.08% -> 0.0048% relative error (cosine 0.999946 -> 1.000000).
  - `linear_attn_qkv_mixed_0` 0.29%, `conv_output_silu_0` 0.11%, `z_0` 0.027%, `alpha_0` 0.024%.
  - `final_output_0` (gated GDN output, before `ssm_out`): 0.30%.
  - `linear_attn_out_0` (after `ssm_out`): 2.2%. `l_last_0`: 1.43% (was 1.46% without the experiment).
- So the HC mix was a real numeric mismatch, but it is not the dominant source of the layer-0 residual error. The
  dominant step is now the `ssm_out` projection, which amplifies a 0.3% input error to 2.2%.
- Checked against the pinned reference: llama.cpp's Q8_K quantiser uses `iscale = -127/max` (ggml-quants.c:2791),
  the same as Stingray's `QuantizeRowToQ8K`. So Q8_K scale is not the cause.
- The HC experiment is not yet a default. A real fix would route the HC matmuls through the quantised kernel path
  (`SimdKernels.MatVec` with Q8_0 weights), with a regression test, once the whole chain is understood.
- **Next:** find why `ssm_out` (Q5_K, input `final_output`) amplifies the error. Check the Q5_K x Q8_K kernel against
  `ggml_vec_dot_q5_K_q8_K` in the pinned source, and dump `ssm_out` intermediates if needed.

### Phase 0.2 fix, 2026-11-08: Q5_K matvec numerics (Qwen path only)

- llama.cpp's ssm_out-0 output is reproduced to 5.6e-7 relative error by the Q5_K x Q8_K dot on llama.cpp's own
  input (Python check, llama.cpp Q5_K dequant formula and Q8_K quantiser). With f32 activations the error is 2.2%.
- Stingray's generic `SimdKernels.MatVec` keeps f32 activations for Q5_K. The Qwen path now calls
  `Qwen4ExpForwardPass.MatVecLlamaNumerics`, which uses `MatVecQ5K_Q8K` for Q5_K. Other dtypes already quantise
  like llama.cpp and are unchanged.
- Scope: this changes only the Qwen 3.8 forward pass. Other models still use the generic dispatch, so they are not affected.
  The generic Q5_K decode switch `STINGRAY_Q5K_DECODE_Q8K` is also unchanged. Its documented speed cost applies to other models,
  not to Qwen's correctness path. Speed of the Qwen path is not measured yet (rule 7 pass comes later).
- Regression test: `Qwen4ExpQ5KDispatchTests` (synthetic Q5_K weights; the dispatch must equal `MatVecQ5K_Q8K`, and the f32
  kernel must differ, so the test can detect a regression).
- Effect on the real checkpoint, layer 0 (last token, with the HC rounding experiment on): `linear_attn_out_0` 2.2% -> 0.36%,
  `l_last_0` 1.46% -> 0.37%, logits relative error 11% -> 6.5% (cosine 0.9979), top-1 still 11751 (matches reference).
- Still open: the MoE error from layer 1 on (`ffn_out` 2%, growing to 19-27% at layers 5-11). Routed experts computed in
  f32 give the same error, so it is not the expert matvec's activation rounding.
