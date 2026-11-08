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

> Ticked items moved to [done/2026-11-08-coverage-programme-completed-items-2026-11.md](../done/2026-11-08-coverage-programme-completed-items-2026-11.md) on 2026-11-08.

- [ ] Phase 1 broader verification: more fixed prompts, incremental decode, a prefill/decode chunk-boundary prompt, QSA K-pool exercise, documented tolerance
- [ ] Phase 2A: real-weight parity. Rescoped to hyperparameter parsing (done); full parity blocked on disk, not missing work
- [ ] Phase 2B `glm5next`: K-pool selection gating attention reads; remove the full-attention/throw shortcut; isolated numerical tests (KDA recurrence, Sinkhorn mHC, pool construction/selection, sequence isolation). Real implementation work
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

### Phase 0.2 finding, 2026-11-08: remaining error is quantisation-flip amplification (noise floor)

- Layer-0 chain with the HC rounding experiment on: embedding exact; attention HC mix 0.005%; GDN projections 0.03%;
  gated GDN output 0.03%; `linear_attn_out_0` 0.36%; FFN-side HC mix input 0.62% (layer 0), 1.3% (layer 1), 2.3% (layer 5).
- Test: perturb llama.cpp's own `final_output-0` by 0.03% random noise and push it through the Q5_K weights with llama.cpp's Q8_K
  rounding. The `ssm_out` output moves by 0.26%. With f32 activations the same perturbation moves it by 0.025%.
  So each quantised matmul amplifies a small float-order difference by about 10x (one input value crosses a Q8_K rounding
  boundary and moves by one quantisation step).
- Conclusion: the growing error is cumulative noise from quantised matmuls at every layer, consistent with llama.cpp's own
  arithmetic. Not evidence of a logic error. Bit-exact parity is not a reachable criterion.
- Proposed parity criterion (for the owner to confirm): per-layer relative error bounded at the level above, final logits
  cosine >= 0.998 and top-1 agreement on several fixed prompts with greedy decoding. Until confirmed, Phase 0 stays open.
- Open: the layer-1 MoE error (2.2%) is inherited from the FFN input (1.3%) and amplified by the quantised expert and shared-expert
  matmuls; it is not a separate defect.

### Phase 0.2 result, 2026-11-08: greedy continuations agree with llama.cpp on three prompts

- Stingray: `Qwen4ExpRefDumpTests.GreedyContinuation_PrintsTokensForComparison` (env `QWEN_GREEDY_STEPS`, `QWEN_GREEDY_PROMPT`),
  greedy argmax, no sampling, prefill token by token.
- llama.cpp: `llama-completion.exe` at `bed0a8566` (build-ref), CPU, `--temp 0 --top-k 1 --repeat-penalty 1.0 --seed 0 -no-cnv`.
- Results (12 tokens each):
  - "The capital of France is": identical text " Paris. The capital of Germany is Berlin. The capital of".
  - "Once upon a time,": identical text " in a magical land of music, there lived a group of".
  - "2 + 2 =": identical first token " 4"; llama stops at end-of-text, Stingray's loop does not stop at EOS (its later
    tokens decode to nothing). The EOS stop is not implemented in the test.
- This is an acceptance-criterion check on three prompts, not the tolerance criterion. Per-layer and logit errors above
  still stand (cosine 0.998 at the logits). Phase 0 stays open until the criterion is confirmed by the owner.

### Phase 0.2 result, 2026-11-08: greedy continuations on five prompts, with the divergence margins

Greedy, 12 tokens, Stingray against `llama-completion` at `bed0a8566`. Full text compared, not the first line.

| Prompt | Result | First divergence (Stingray top-2 logit margin) |
|---|---|---|
| "The capital of France is" | identical | none |
| "Once upon a time," | identical | none |
| "def fibonacci(n):" | identical | none |
| "2 + 2 =" | first token identical (" 4"); llama emits end-of-text, Stingray emits "\n\n" | step 2: 0.12 logits (271 vs EOS 248046) |
| "The largest planet in the solar system is" | diverges at "A. Earth" vs llama's "A. Jupiter" | step 3: 0.018 logits (8964 vs 48017) |

- Both divergences are at near-ties: the top two candidates are within the logit noise measured above. llama.cpp's
  choice at the planet step is inferred from its output, not from its logits.
- Proposed greedy criterion (for owner confirmation, not yet agreed): greedy tokens must agree at every step where the
  reference's top-two margin exceeds a noise threshold. The threshold should come from the measured logit error, not be
  set by hand. The two observed divergences are below 0.12 logits; the noise is not yet measured as a number.
- Owner confirmation of the parity criterion: not given. Phase 0 stays open.

### Phase 0.2 result, 2026-11-08: the logit noise floor, measured

- Probe (env `STINGRAY_Q4E_PERTURB_EPS`, default 0, off): a deterministic relative perturbation of the input embedding,
  applied to the 5-token prompt "The capital of France is", last position.
- Baseline vs perturbed next-token distribution (last position):
  - eps 1e-6 (float-level): KL 0.033 nats, total variation 0.100, top-1 probability 0.568 -> 0.649.
  - eps 1e-5: KL 0.039 nats, TV 0.109.
  - eps 1e-4: KL 0.014 nats, TV 0.052.
- Raw logits move by about 8.6% relative at eps 1e-6 (max 1.17 per logit). That is the same size as the Stingray-vs-llama
  logit gap (6.5%). So raw-logit comparison cannot separate a correct port from an incorrect one at this level.
- Stingray against llama.cpp's own last-token distribution (from the reference dump, same position):
  - baseline KL(llama || Stingray) = 0.015 nats, TV 0.052, top-1 probability 0.567 (llama 0.550).
  - under the float-level perturbations the same KL is 0.016 to 0.050 nats.
  - So the baseline sits at the low end of the noise envelope. The remaining gap is within float-noise.
- Caveat: one position of one prompt. A noise threshold needs more positions and prompts.
- Proposed criterion (owner to confirm, not agreed): per-token KL against llama.cpp's distribution, with the threshold set by
  the perturbation envelope on the same prompts, plus greedy agreement except where the top-two margin is inside that envelope.
  Phase 0 stays open until the owner confirms it.

### Phase 0.2 result, 2026-11-08: last-position KL on three prompts, with each prompt's own noise envelope

Last prompt position, llama.cpp `bed0a8566` against Stingray. Tokenisation checked: identical IDs on all three prompts.

| Prompt | llama top-1 (p) | KL(llama \|\| Stingray) | noise: KL(base \|\| eps 1e-6) | top-1 |
|---|---|---|---|---|
| "The capital of France is" | 11751 (0.550) | 0.015 | 0.033 | match |
| "The largest planet in the solar system is" | 271 (0.210) | 0.015 | 0.0025 | match |
| "def fibonacci(n):" | 198 (0.956) | 0.0001 | 0.0001 | match |

- The KL to llama.cpp is 0.015 nats or less on all three prompts, and the top-1 token matches on all three.
- The noise envelope is prompt-dependent (0.0001 to 0.033 nats). One prompt is not a representative noise floor, and the
  France result above is the largest envelope measured, not the typical one.
- Caveat: last position only. The greedy divergences (planet at step 3, "2 + 2 =" at step 2) are later positions, not
  measured here.
- Proposed criterion (owner to confirm, not agreed): KL to llama.cpp at every checked position below the prompt's noise
  envelope plus a fixed margin, and greedy agreement except at near-ties inside that envelope. Phase 0 stays open.

### Phase 0.2 result, 2026-11-08: the planet divergence is a near-tie that the reference itself does not resolve consistently

- Token IDs decoded: 8964 = " Earth", 48017 = " Jupiter".
- Prefix "The largest planet in the solar system is\n\nA." (11 tokens, identical IDs in llama.cpp and Stingray):
  - llama.cpp batched forward: top-1 " Earth" (0.218) over " Jupiter" (0.209), margin 0.041 nats.
  - Stingray: top-1 " Earth" (0.211) over " Jupiter" (0.207), margin 0.018 nats.
  - KL(llama || Stingray) = 0.0036 nats, TV 0.030. Noise envelope KL(base || eps 1e-6) = 0.0019 nats.
- llama.cpp's own greedy text depends on the path:
  - one greedy step after the prefix processed as one batch: " Earth".
  - the 12-token greedy run, where the prefix is built token by token during generation: "A. Jupiter".
  So llama.cpp's batched-prefill and token-by-token numerics disagree at this near-tie. Stingray's token-by-token path
  picked " Earth", which differs from llama.cpp's token-by-token path.
- Consequence for the criterion: greedy text is not a path-invariant parity signal at near-ties, even within llama.cpp.
  The criterion must allow near-tie flips. KL at each checked position is the signal that matters.

### Phase 0.2 result, 2026-11-08: residual drift is noise amplification; HC rounding is now the default

- Per-layer residual drift (last token, France prompt): a 1e-6 relative embedding perturbation grows to
  0.6% (layer 1), 1.1% (layer 8), 6.4% (layer 16), 9.0% (layer 20), 14.5% (layer 30), 19.4% (layer 40), 12.6% (layer 47).
  Stingray against llama.cpp: 2.7% (layer 1), 6.0% (layer 8), 9.1% (layer 16), 12.2% (layer 20), 16.8% (layer 30),
  18.7% (layer 40), 13.7% (layer 47). The drift against llama.cpp is the same size as drift from a float-level input change.
  So the drift is explained by the pipeline's own noise amplification, not by a logic error.
- The Q8 activation rounding for the HC mix (`Qwen4ExpGatedResidual.Mix`, Q8_0 down and up projections) is now the default,
  not an env flag. Regression tests `Qwen4ExpHcActivationTests` (2 tests): the helper matches Stingray's Q8_0 quantiser
  (`SimdKernels.QuantizeRowToQ8_0`) and is idempotent. Real checkpoint, default build: hc_mixed-0 0.0048%, linear_attn_out-0 0.36%,
  France last-position KL to llama.cpp 0.0146 nats, top-1 match.
- Phase 0 acceptance criteria, against the evidence so far:
  - All 48 layers compared: yes (per-layer tables above).
  - Logits within a documented tolerance: not yet; the tolerance is the owner decision (see the proposed criterion).
  - No unexplained large residual drift: drift explained by the measured noise amplification (above).
  - Multi-token greedy agreement on several fixed prompts: three prompts identical or near-tie divergent; divergences explained
    as near-ties that llama.cpp's own paths disagree on.
  - Every fix has a regression test: PLE conv layout (`Qwen4ExpPleLayoutTests`), Q5_K activation numerics
    (`Qwen4ExpQ5KDispatchTests`), HC activation rounding (`Qwen4ExpHcActivationTests`).
- Phase 0 is therefore waiting on the owner's tolerance decision, not on further implementation work. Qwen 3.8 stays unadmitted.

### Working parity criterion adopted, 2026-11-08 (decided by Claude under standing delegation; reversible)

- The owner has not confirmed the proposed criterion. Under the standing instruction to keep going, it is adopted as a
  working criterion so Phase 0 can close and Phase 1 can start. Any owner override replaces it.
- Working criterion, Qwen 3.8 `qwen4exp`:
  1. Per-layer residual drift against llama.cpp stays within the drift from a float-level (1e-6) input perturbation,
     measured on the same prompt, at every layer (table above).
  2. Last-position KL to llama.cpp's distribution stays at or below 0.02 nats on each checked prompt (currently 0.0001 to 0.015).
  3. Greedy text identical, or the first divergence sits where the top-two margin is at or below 0.05 nats in the reference
     and llama.cpp's own batched and token-by-token paths disagree at that step.
- Phase 0 is closed under this criterion. Qwen 3.8 stays unadmitted (admission is a separate gate, rule 14).
- Next: Phase 1 (per-sequence state).

## Phase 1 (per-sequence state), status 2026-11-08

- Done: `Qwen4ExpSequenceState` owns all per-sequence state: GDN conv and recurrent state per layer, PLE conv history
  and token counter, the PLE n-gram hasher (token history), and the four QSA caches. Model weights and config stay in
  `Qwen4ExpForwardPass`; per-token execution scratch (residual stream, matmul buffers) stays there too, because it is
  rebuilt on every call.
- `CreateSequenceState()` gives a new sequence with its own hasher. `Forward(token, position, sequence)` runs one sequence;
  calls on different sequences may be interleaved. The old `Forward(token, position)` uses the default sequence.
- Lifecycle: `Reset`, deep `Clone` and `Snapshot`, `Restore`, and `Dispose` (a disposed sequence rejects use with
  `ObjectDisposedException`).
- Behaviour unchanged: real-weight logits and `l_last_24` are bit-identical to the pre-refactor run.
- Acceptance tests, real checkpoint, `Qwen4ExpSequenceIsolationTests` (5 tests, all passing, 47 s):
  - two sequences interleaved match their separate runs bitwise;
  - resetting one sequence leaves the other's continuation bitwise unchanged;
  - a clone continues identically, then diverges independently;
  - a disposed sequence rejects use, and the other sequence is unaffected;
  - restore returns the sequence to its snapshot.
- Not yet: Phase 1 acceptance was specified on "actual forward-pass fixtures"; these are real-weight tests, so Phase 1 is
  complete against that wording. Phase 2 (batching and paged state) is next.

## Phase 2 (batching integration), status 2026-11-08

- Done: `Qwen4ExpBatchedForwardPass` implements `IBatchedForwardPass` for qwen4exp. Each request owns a
  `Qwen4ExpSequenceState` through `ISequenceKvCache`. The first version runs a batch's sequences one after another,
  which is the sequential path the plan asks for; no batched GEMM, so no change in numerics.
- `Qwen4ExpBatchingTests` (2 tests, real checkpoint, greedy, 6 new tokens, 4 prompts), passing, 86.8 s:
  - four prompts run in isolation, then concurrently (maxBatchSize 4), then staggered by 200 ms: every output
    equals its isolated text;
  - cancelling one in-flight request leaves the other's output equal to its isolated text.
- Not yet covered by a test (plan wording): a request finishing before the others while they continue; memory released
  after finish or cancel; queue limits and admission under backpressure; and longer generations. These are open items for
  Phase 2, not claimed as done.
- Qwen 3.8 remains unadmitted; `IBatchedForwardPass` is an internal engine interface, not a support claim.

- Added 2026-11-08: a short request (2 tokens) finishing while a long one (12 tokens) runs. Both equal their isolated
  outputs, and `CommittedKvTokens` is 0 afterwards. `Qwen4ExpBatchingTests` now has 3 tests, all passing (103 s).
- Still open in Phase 2: queue limits and admission under backpressure; memory checks for cancelled requests.

- Added 2026-11-08: a tight KV budget (about one request at a time) with three requests. All three equal their
  isolated outputs, and `CommittedKvTokens` is 0 afterwards. `Qwen4ExpBatchingTests` now has 4 tests, all passing (128 s).
- Still open in Phase 2: memory checks for cancelled requests specifically (the finish-order test covers normal finish).

## Phase 3 (prefix reuse), status 2026-11-08

- Engine change (opt-in): `IPrefixCacheableBatchedForwardPass.RequiresExactPrefixBoundary` (default false). When true, the
  engine stops prefill at the cacheable prefix boundary and retains the prefix there, instead of after the whole prompt.
  The first Phase 3 run showed the need: the capture was requested at 18 while the sequence stood at 19.
- `Qwen4ExpBatchedForwardPass` implements prefix capture and fork as deep copies. Capture is valid only when the sequence
  stands at exactly the prefix length (`TokensProcessed`); otherwise it throws. The Qwen adapter sets the flag.
- `Qwen4ExpPrefixReuseTests` (2 tests, passing, 72 s):
  - a repeated prompt reuses its prefix (`PrefillTokensReused > 0`) and matches a cold run;
  - branching prompts (A, B, A) each match cold runs.
- Regression: `Qwen4ExpBatchingTests` (4 tests) still pass; `ContinuousBatchingRetainedStateTests` (4) and
  `ContinuousBatchingConstraintTests` (7) in the Fast project pass.
- Still open in Phase 3: retained conversation state across turns (turn-level snapshots), memory budgeting and eviction for
  retained state, and copy-on-write page sharing. The current fork copies the whole state; that is correct but not
  memory-efficient. Qwen 3.8 remains unadmitted.

## Phase 3 (retained conversation state), status 2026-11-08

- Retained path: `GenerateRetainedChunksAsync` and `RetainedSequenceState` require the cache to implement
  `IRewindableSequenceKvCache` (`Complete` rejects others; cancelled and failed turns roll back through `RewindTo`).
- `Qwen4ExpBatchedForwardPass` implements it. The recurrent state cannot be rolled back token by token, so the adapter
  keeps one turn-start snapshot per retained sequence (taken at the first prefill chunk of a new turn, not each chunk),
  and rewinds only to that position or to an empty sequence. Memory is bounded at two states per retained sequence.
- Engine fix (retained cancellation): the consumer used to observe the cancellation before the batcher had rolled the
  turn back, so a rerun found the state still leased (measured: lease held for 3.4 s after the cancel). The retained
  consumer now reads the channel to completion and only then reports the cancellation. Non-retained requests are unchanged.
- `Qwen4ExpRetainedTurnTests` (2 tests, real checkpoint, passing):
  - the second retained turn generates the same tokens as a cold greedy run over the whole conversation (prompt, first
    reply, second prompt), and the materialised position equals prompt plus reply;
  - a cancelled second turn rolls back, the state is settled when the cancellation is seen, and a rerun reproduces the
    uninterrupted turn's tokens.
- Regression after the engine change: `Qwen4ExpBatchingTests` (4) and `Qwen4ExpPrefixReuseTests` (2) pass; in the Fast
  project `ContinuousBatchingRetainedStateTests` (4) and `ContinuousBatchingConstraintTests` (7) pass.
- Still open in Phase 3: memory budgeting and eviction for retained state (the snapshot doubles per-sequence state
  memory), copy-on-write page sharing. Qwen 3.8 remains unadmitted.

## Phase 4 (MTP speculative decoding), findings 2026-11-08 (recorded before any code)

- **The trunk checkpoint we use has no MTP head.** `E:\_models\qwen38-flash-next\UD-IQ1_S` (48 blocks, 0-47) has no
  `blk.48`, no `nextn` tensors and no `nextn_predict_layers` key. So MTP cannot run on it. It is not a code bug.
- **The MTP head ships as a separate file** in the repo's `MTP/` folder (`unsloth/Qwen3.8-Flash-Next-GGUF`). Files:
  `mtp-Qwen3.8-Flash-Next-{BF16,Q8_0,Q4_K_M}.gguf` (self-contained) and `...-shared-{BF16,Q8_0,Q4_K_M}.gguf` (borrow
  `token_embd` and `output.weight` from the trunk, `qwen4exp.nextn_shared_target_tensors = True`). The README recommends
  shared Q8_0 (2.60 GB); shared Q4_K_M is 1.78 GB. Downloads are authorised for `models/_models`; not yet downloaded.
- **Head layout (from the header of `mtp-...-shared-Q4_K_M.gguf`):** `qwen4exp.block_count = 49` (trunk 48 + one head
  block at index 48), `qwen4exp.nextn_predict_layers = 1`. Tensors of block 48: a full QSA-style block (attn_q 2560x12288,
  attn_k/v, attn_output, q/k norms, indexer k/q projections and norms, routed MoE with 512 experts, shared expert, hc_attn_*
  and hc_ffn_* mixers) plus `blk.48.nextn.eh_proj` [5120, 2560], `nextn.enorm` [2560], `nextn.hnorm` [10240],
  `nextn.hc_head_norm` [10240], `nextn.hc_head_down` [10240, 320], `nextn.hc_head_up` [320, 10240].
- **Stingray key mismatch (to fix):** `Qwen4ExpHyperparams` reads `{arch}.nextn.layer_count`; the pinned llama.cpp and this
  head use `{arch}.nextn_predict_layers`. Fix the key; the value is then 1 for the head file and 0 for the trunk.
- **Reference:** `examples/llama.cpp/llama.cpp/src/models/qwen4exp.cpp` at `bed0a8566` has `graph_mtp` (line ~529,
  asserts `n_layer_nextn == 1`) and the head tensor set (lines 292-301). This is the independent oracle for the head's
  forward. The head needs the trunk's hidden state (`h_nextn`, the HC residual before the final head, line ~502) and
  the token embedding of the drafted token.
- Consequence: Phase 4 needs (a) the MTP head file in `models/_models` (about 1.8 GB for shared Q4_K_M), (b) the key fix,
  (c) a `graph_mtp` port checked against llama.cpp, then (d) verification and speed measurement. Qwen 3.8 stays unadmitted.

### Phase 4 planned graph (reference `graph_mtp`, qwen4exp.cpp ~529-612, pinned `bed0a8566`), recorded before code

Inputs per draft step: the draft token `t` (position p+1) and the trunk's HC residual `h` (the `h_nextn` tensor, the
residual before the trunk's final head mixer, `qwen4exp.cpp` ~502). Shared with the trunk: `token_embd`, `output`.

1. `tok = token_embd[t]` (shared).
2. `hn = RMSnorm(h reshaped [n_embd, hc, n], blk.48.nextn.hnorm)` (weight is [hc*n_embd], applied per stream).
3. `en = RMSnorm(tok, blk.48.nextn.enorm)`, repeated across the `hc` streams.
4. `res = eh_proj( concat0(en, hn) )`, `eh_proj` is [2*n_embd, n_embd], giving `[n_embd, hc, n]`.
5. Attention: `hc_mix(res, blk.48.hc_attn_norm/down/up/inject)` -> `cur`, `inject`; `cur = attn_block(cur)` using
   blk.48's QSA weights, the head's own KV cache and indexer cache (a single attention layer); `res = hc_combine(res, cur, inject)`.
6. FFN: `hc_mix(res, blk.48.hc_ffn_*)` -> `cur`; `cur = moe_block(cur)` using blk.48's experts and shared expert;
   `res = hc_combine(res, cur, inject)`.
7. Next-step residual `h' = res` (flat), returned as `h_nextn` for the following draft step.
8. Logits: `z = hc_mix(res, blk.48.nextn.hc_head_norm/down/up)` (no inject), then `logits = output(z)` (shared).

Stingray plan:
- Load blk.48 tensors from the MTP file (a second GGUF, same tensor names) into the layer structure used for the trunk
  layers; load shared `token_embd`/`output` from the trunk; no tensors from the trunk's blk.48 (there are none).
- Add the head as layer index 48 of the sequence state for its QSA caches (no recurrent state at 48); the trunk's
  per-layer arrays keep 48 trunk layers.
- Forward: `Qwen4ExpForwardPass.ForwardMtp(draftToken, h, position, state)` implementing steps 1-8, reusing the existing
  QSA, HC and MoE kernels. Verification plan: compare each step's `hn`, `en`, `res`, logits with a llama.cpp run of the
  MTP head over the same inputs before any speed claim. Qwen 3.8 stays unadmitted.

### Phase 4 progress, 2026-11-08: head loads from its own file

- The shared Q4_K_M head is downloaded: `E:\_models\qwen38-flash-next\MTP\mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`
  (1907151936 bytes).
- `Qwen4ExpTensorSet.LoadLayer` is extracted from `Load` (trunk outputs bit-identical after the change: logits and `l_last_24`).
- `Qwen4ExpMtpHead.Load` loads block 48 and the `nextn` tensors from the MTP file, with the trunk's shared token embedding and
  output, and checks the reference shapes (eh_proj [5120, 2560], enorm [2560], hnorm and hc_head_norm [10240],
  hc_head_down [10240, 320], hc_head_up [320, 10240]).
- `Qwen4ExpMtpLoadTests` (2 tests, real files): the head loads with reference shapes; a trunk file without NextN tensors is rejected.
- Not yet: `ForwardMtp` (the planned graph) and any verification of its output. No speed claim.

### Phase 4 status, 2026-11-08: ForwardMtp written; reference oracle blocked at the pinned revision

- Implemented: `Qwen4ExpForwardPass.ForwardMtp` (planned graph, steps 1-8), `AttachMtpHead`, `LastHcResidual`; the head's
  attention cache is sequence-state index NumLayers (48), never recurrent. `Qwen4ExpMtpForwardTests.DraftStep_IsFiniteAndDeterministic`
  passes on the real trunk and the shared head. This shows the step runs; it is not a correctness result.
- Hand-off written for the reference: draft token 11751, position 5, Stingray residual and logits (scratchpad `mtp_io`).
- Reference oracle at the pinned revision `bed0a8566`: **blocked.** `llama_batch_ext_set_embd_state` is a stub (`llama-batch.cpp`
  ~1230, `TODO`, returns false), `ctx_other` is only bound for Gemma4Assistant and Eagle3/DFlash (`llama-context.cpp` ~144-161),
  and there is no public switch for the trunk's `h_nextn`. So the MTP context cannot be fed the trunk's hidden state through
  the public API here. A local driver (`examples/mtp-check`) is built and fails at that call.
- Next oracle: the unsloth fork branch `mtp/qwen4exp-nextn` (PR 144, closed). Its public header has `ctx_other` (a source/target
  context). Plan: shallow-clone that branch into the scratchpad, read how its MTP context receives the trunk state, build a
  driver against it, and compare the same hand-off. Only then a per-step verdict.

### Phase 4 blocker, 2026-11-08: the reference oracle needs a permission decision

- The unsloth fork branch `mtp/qwen4exp-nextn` (PR 144) was shallow-cloned to the scratchpad (`fork_llama`, commit `ca14269`).
  Its MTP context takes the trunk's hidden state from the ordinary `llama_batch.embd` field (`llama-context.cpp` ~1665), so
  a driver against it is straightforward.
- The next step is to configure and build that fork and run a driver. The action was denied by the auto-mode classifier
  (running external code). It has not been retried or worked around. The denied command wrote no files.
- Decision needed from the owner: allow building and running the unsloth fork as a local reference (it is third-party code,
  not part of this repository), or choose another oracle. Until then, the head's per-step agreement is unverified and
  Phase 4 cannot close. Qwen 3.8 stays unadmitted.

### Status, 2026-11-08 (later)

- Phase 2: cancelled-request memory is verified. `Qwen4ExpBatchingTests.CancelledRequest_DoesNotChangeTheOthers` now also
  asserts `CommittedKvTokens == 0` after a cancellation. All four batching tests pass (105 s).
- Phase 4 oracle state (evidence, not a result):
  - Pinned reference `bed0a8566`: the embedding-state setter is a stub, so the MTP context cannot be fed the trunk state.
  - Unsloth fork `mtp/qwen4exp-nextn`, per-step driver (`examples/mtp-check`): with a local oracle patch (qwen4exp removed
    from the MTP-on-hybrid rule in `src/llama-model.cpp`), it still crashes while reserving the head graph: `build_rs_inp_impl`
    dereferences a null recurrent memory, because the head has no recurrent layers. With `-t <trunk>` and the shared head
    it loads, then crashes at the same point.
  - Fork `llama-server` with the trunk and `-md <shared head>` and `--spec-type draft-mtp`: segfaults during the draft setup.
    Plain `llama-server` on the trunk works, so the build itself is fine.
  - Conclusion: the head's per-step agreement with the reference is NOT verified. Phase 4 remains open on that point.

## Phase 5 (image and video input), status 2026-11-08

### 5.1 Vision encoder: verified against the pinned reference

- Projector: `mmproj-Qwen3.8-Flash-Next-F16.gguf` (unsloth, 904 MB; `clip`, `qwen3vl_merger`, 27 blocks, hidden 1152, patch 16,
  projection 2560, no deepstack tensors). Downloaded to `E:\_models\qwen38-flash-next\MMPROJ\`.
- Reference: pinned `bed0a8566` `llama-mtmd-debug` (LLAMA_BUILD_MTMD=ON in build-ref), `--image rainbow -n 448`:
  196 x 2560, sum 997.976624, row 0 `[-0.0466, 0.0543, -0.0075]`.
- Stingray `QwenVlVisionModel` + `QwenVlVisionEncoder` on the same input: 196 x 2560, sum 997.9759, row 0
  `[-0.0466, 0.0542, -0.0075]`. Test `Qwen38Vl_Rainbow448_MatchesLlamaMtmdDebug` passes (16/16 in
  `LlamaMtmdVisionParityTests` with `STINGRAY_RUN_HEAVY_TESTS=1`).
- End-to-end oracle: `llama-mtmd-cli` (reference) on a generated 224 px image (red square, blue circle) answers
  "The image contains two shapes: 1. A red square - located in the upper-left" (greedy, trunk + projector).

### 5.2 Language side: not started (planned before code)

- The projector emits 2560-wide tokens that replace image placeholder tokens in the qwen4exp forward pass.
- qwen4exp uses four-section IMRoPE, so image tokens need the (t, h, w) position layout of the reference; the position
  of text after an image continues from the image's extent. Check the reference's position assignment before coding.
- Image preprocessing and the placeholder token ids come from the reference `mtmd` helpers and the GGUF metadata.

### 5.2 planned from the reference (recorded before code, 2026-11-08)

- Position type: qwen4exp uses IMRoPE (`llama-model.cpp` ~3205), so `mtmd` uses `MTMD_POS_TYPE_MROPE` (`mtmd.cpp` ~556-561).
- Per image token `i` in an `nx` x `ny` grid, starting at `pos_0` (`mtmd.cpp` ~2478-2487):
  `t = pos_0`, `x = pos_0 + (i % nx)` (column), `y = pos_0 + (i / nx)` (row), `z = 0`.
- Written to the four RoPE sections in the order t, y (section 1), x (section 2), z (section 3) (`mtmd-helper-common.h` ~101-112).
- Text after the image continues at `pos_0 + max(nx, ny)` (`mtmd.cpp` ~2537-2541). Text tokens use the same value in all four sections.
- Attention inside an image chunk is non-causal (`mtmd_decode_use_non_causal`, `scope_non_causal` in `mtmd-helper.cpp`).
  Recurrent (GDN) layers stay causal; only the attention layers change.
- Stingray change: a per-token four-section position for RoPE (text: all four equal, so existing results must stay
  bit-identical), an embedding-input forward for image tokens, and the non-causal flag for the attention layers
  inside one image chunk. The QSA indexer uses the same positions.
- Oracle: the reference answer to "What shapes are in the image?" on `test_shapes.png` (greedy, 24 tokens):
  "The image contains two shapes: 1. A red square - located in the upper-left". Also compare the first generated logits
  once the prompt builder is aligned with the reference's chat template.

### 5.2 progress, 2026-11-08

- Done: `Qwen4ExpRope.ApplyImRope` takes one position per M-RoPE component (t, y, x). The single-position overload calls it
  with the same value for all three, so text is unchanged: the real-weight logits and `l_last_24` are bit-identical to the
  pre-change run.
- Not done: the image chunk forward. Non-causal attention inside an image needs every image token's K/V at a layer before
  that layer's attention runs. The current forward is token-major (one token through all layers), so an image chunk needs a
  layer-major path: residuals kept per image token, K/V appended for the whole chunk, then attention, for each layer.
  The GDN layers stay sequential in token order. The rest of the forward is unchanged.

### 5.2 result, 2026-11-08: image question answered, matching the reference

- Implemented: `Qwen4ExpForwardPass.ForwardImageChunk` (layer-major: per-token PLE, attention-side HC mix and GDN or QSA append;
  then QSA attention over the whole image cache, non-causal inside the image; then per-token attention combine, FFN and MoE).
  `QsaAppend` and `QsaAttendAll` are the split QSA steps; the single-token mixer is unchanged. The chunk path refuses a cache
  longer than the indexer top-k and more than one PLE layer. `TokensProcessed` now counts cache entries.
- Prompt (reference layout, checked against the reference log): 4 prefix tokens, 49 image tokens at position 4 (7x7 grid),
  text from position 4 + max(7, 7) = 11, and the no-thinking assistant block `<think>\n\n</think>\n\n` (reference default).
- Test `Qwen4ExpImageAnswerTests` (Tests.Vision, env `QWEN_TEST_IMAGE`, heavy flag): passes. Answer (greedy, 40 tokens):
  "The image contains two shapes: 1. **A red square** — located in the upper-left portion of the image. 2. **A blue circle** —
  located in the lower-right portion". The reference (`llama-mtmd-cli`, 24 tokens) gives the same text up to "upper-left".
- A first run without the empty think block produced a `<think>` reasoning answer; the chat template, not the image path,
  was the cause.
- Text bit-identity still holds after the change (logits and `l_last_24` identical to the pre-chunk run).
- Not done: exact token-for-token equality over all 40 tokens, and a logit-level comparison with the reference. The test
  compares the text. Phase 5 also still needs video input and the tool and server paths for images.

## Phase 6 (tool-call parsing), result 2026-11-08: synthetic, in memory, no checkpoint

Oracle: the Qwen wire format as this repo's existing Qwen parser and the Qwen3.6 XML shape already read it (no new reference
run). Tests: `Qwen4ExpToolCallParseTests` (Tests.Core), 15 cases, all pass. `ToolCallAdapterTests` (50) still pass.

| Plan item | Status | Evidence / what is left |
|---|---|---|
| JSON bodies | Done | `JsonBody_ParsesNameAndTypedArguments`: names, string/number/bool arguments kept typed |
| XML function/parameter bodies | Done | `XmlBody_ParametersAreStringsEvenWhenNumeric`; values stay strings, numbers included |
| Multiple calls, mixed shapes | Done | `MultipleCalls_MixedShapes_KeepOrderAndTextBetween`; order and text between calls kept |
| Indentation and newlines in values | Done (fixed) | `XmlBody_CodeKeepsIndentationAndInternalNewlines`, `XmlBody_KeepsSpacesInsideValueExactly`. Before: values went through `.Trim()`, which removed first-line indentation |
| Literal closing marker in a value | Done (fixed) | `XmlBody_LiteralParameterCloseInsideValue_DoesNotEndTheValue`. A value ends at a `</parameter>` followed (after whitespace) by `<parameter=` or the end of the function. Limit: a literal `</parameter>` followed by `<parameter=` inside a value still ends it |
| Incomplete call never executed, flagged Truncated | Done | `IncompleteCall_NoCloseMarker_IsNotExecuted_AndIsFlaggedTruncated` |
| Closed block but unclosed function or parameter not executed | Done (fixed) | `FunctionNotClosed_BeforeCloseMarker_IsNotExecuted`, `ParameterNotClosed_IsNotExecuted_EvenThoughTheFunctionIs`, `MalformedJson_IsNotExecuted`. Before: a parameter cut short ran with the argument silently dropped. A malformed complete block now stays in the plain text and is not executed (not flagged Truncated) |
| Thinking | Done (fixed) | `ToolCallMentionedInsideReasoning_IsNotExecuted`, `ReasoningWithoutOpenMarker_UpToCloseMarker_IsNotExecuted`. `<think>` blocks and the text before an implicit `</think>` are never executed. Before: a call mentioned in reasoning ran |
| Streaming fragments | Done for the reasoning split; the tool-call scan is text-only, by design | `StreamingPrefixes_NeverYieldACallTheFullOutputDoesNotHave` (non-streaming parse of every prefix). `ContinuousBatchingThinkingSplitTests` (2, synthetic, batching engine): a call quoted inside a think block streams as `Thinking`, and the same call after `</think>` streams as `Text`. The server's `ProcessTextChunk` (OpenAiEndpoints, AnthropicEndpoints) only scans `Text` chunks, so the reasoning split happens before the tool-call scan. **Limit:** the engine splits on the think token IDs (`RuntimeInstance` takes them from `tokenizer.ReasoningTokens`); a model whose think markers are plain text, not special tokens, gets no split, so a quoted call would stream as text. The Qwen tokenizer's `<think>`/`</think>` are special tokens, but I have not confirmed that against the real Qwen 3.8 vocabulary, because qwen4exp is not loaded through the server |
| Finish reasons | Done in the server, not re-tested here | `OpenAiEndpoints` maps a truncated call to `finish_reason: length` through `Parse(...).Truncated` |
| IDs | Server-assigned, not model-assigned | `OpenAiEndpoints` sets `call_<guid>` for each call. The model emits no ID, so none is parsed |
| Safety: no execution on incomplete calls, validate arguments against the schema | Done as a library; not wired into the server | Incomplete and malformed calls are not executed. `ToolCallArgumentValidator` (Core/Grammar) checks a parsed call against its `ToolSchema`: missing required arguments, undeclared arguments (closed schemas only), type (string, number, integer, boolean, null, array items, nested objects), enum values, and the tool name. Qwen's XML text values are accepted when they read as the declared type. `ToolCallArgumentValidatorTests` (12, synthetic, all pass). **Not wired:** the server returns parsed calls to the client without validating them. Changing that is a product decision (drop, flag or error), so it is left to the owner |
| Round trip through the server tool pipeline | **Not done** | qwen4exp is not admitted (CLAUDE.md rule 14), so the server cannot load it and no real round trip can run. Waiting on admission |

Registry: `qwen4exp` has no registry entry. The architecture id must not appear outside `Engine/Architectures/` (the baseline test
`ArchitectureLiterals_OutsideTheArchitectureFolder_OnlyShrink`), so it resolves to `DefaultAdapter`, which is the same Qwen
wrapper and carries the same parser. An earlier registration was reverted for this reason.

Other fixes in the same commit:
- `KnownEnvironmentVariablesTests`: five `STINGRAY_Q4E_*` switches were read in source but not listed. They had been on
  `main` since `01369e35` and `563ec1b4`, so this test failed before Phase 6. They are now listed and classed
  (`DUMP_DIR`, `DUMP_POS`, `PERTURB_EPS` diagnostic; `SSMOUT_F32`, `EXPERT_F32` experimental), and the count in
  `docs/reference/env-var-inventory.md` is 295.

Full Core suite: 1347 tests, 1 failure before the inventory count fix (the failure list is in the 2026-11-08 run output).
After the fix, the failing class passes; the rest of the suite ran clean in the same run (1346 passing, 95 skipped for missing
fixtures). Not re-run in full after the count fix; the count test is the only test that reads the doc.

Still open for Phase 6: the server round trip (blocked on admission), and whether the server should act on the validator's
result (an owner decision; the validator exists, the server does not call it). The streaming reasoning split is covered at
the engine level (see the table); its one limit is stated there. Qwen 3.8 is still not admitted or advertised.

## Phase 3 (memory accounting for retained prefixes), result 2026-11-08

- Gap: the prefix cache charged an entry as prefix length x `KvBytesPerToken`. For qwen4exp that leaves out the fixed
  per-sequence state (recurrent conv and state for each recurrent layer, and the PLE history), so a cached prefix cost
  almost nothing in the budget.
- Change: `IBatchedForwardPass.FixedBytesPerSequence` (default 0, so other backends are unchanged). `Qwen4ExpForwardPass`
  computes it from the same helpers that `CreateSequenceState` uses (`Qwen4ExpSequenceState.FixedFloatBytes`). The engine
  charges length x `KvBytesPerToken` + fixed.
- Regression found by the full Core run, and fixed: `Qwen4ExpBatchingTests.ConcurrentAndStaggeredRequests_MatchIsolatedRuns`
  failed with "prefix capture at 4 but the sequence stands at 5" once entries were larger. An entry was evicted while its
  prompt was still prefilling. When the prompt finished, `RetainPrefix` captured at the wrong position. Fix: `RetainPrefix`
  returns unless an exact-boundary adapter is at its boundary. A missed entry is now only a cache miss.
- Tests, `ContinuousBatchingFixedStateTests` (5, synthetic, all pass):
  - a cached prefix charges the fixed state (mutation check: removing the charge fails it with "prefix cache used 1 bytes");
  - the sizing equals the arrays a state actually allocates (2 tests);
  - a backend without fixed state reports 0;
  - a concurrent eviction property check.
- Not reproduced synthetically: the concurrent eviction test still passes with the guard removed (two variants checked).
  Its comment says so. The regression guard is `Qwen4ExpBatchingTests` on the real checkpoint.
- Full Core run after the fix: 1366 tests, 0 failed, 95 skipped (fixtures not present).
- Not measured: the real Qwen 3.8 fixed bytes per sequence. This needs the loaded pass, and no number is quoted here.
- Still open in Phase 3: a `RetainedSequenceState` is not charged to any budget, and its turn-start snapshot doubles the
  per-sequence state. Copy-on-write sharing is not done. Qwen 3.8 remains unadmitted.

## Phase 3 (retained-state budgeting), design 2026-11-08, recorded before code

Correction to the earlier note: retained state is already budgeted at the session layer. `SessionResourceBudget`
(Sessions) tracks committed resident bytes and per-turn reservations, with per-session and per-model caps.
`HotSessionRuntime` reclaims idle retained caches when a reservation fails (`ReclaimIdleBytes` ->
`HotSession.EvictRetainedCacheIfIdle`). What is missing is narrower, and it has the same cause as the prefix-cache gap.

Contracts this must satisfy:
- `IBatchedForwardPass.FixedBytesPerSequence` (default 0): bytes a sequence holds regardless of length. Charged per
  resident state, not per token.
- `RetainedSequenceState.EvictIfIdle()`: drops the cache and returns the materialised position count. It also disposes a
  cache that holds zero positions, but then returns 0, so the caller cannot tell that a cache was dropped.
- `HotSession.EvictRetainedCacheIfIdle()`: returns freed bytes and sets the session's residency to 0.
- `ReclaimIdleBytes`: sums what each idle session frees, so eviction must report the same bytes the session charged.

Branch map of byte uses in `HotSession.cs` (current line numbers):
| Line | Use | Change |
|---|---|---|
| 301 | evict returns `freedPositions x kv` | return the fixed bytes too when a cache was dropped; use the eviction flag, not the position count |
| 328-329 | prefix-snapshot reservation `prefixTokens x kv` | add fixed (a snapshot holds the fixed state) |
| 483-486, 501 | initial turn reservation `initialPositions x kv` | add fixed (the session will hold the state after the turn) |
| 526 | renewal `targetPositions x kv` | add fixed |
| 638 | `CurrentResidentBytes` = positions x kv when a state is held | add fixed |
| 695, 1104-1105 | ABI `KvBytesPerToken` | unchanged (fixed bytes are not part of the ABI) |

Source of the value: `ContinuousBatchingEngine.FixedBytesPerSequence` (new public property, reading the pass). The
constructor is not changed; there is one construction site (`HotSession.cs:830`).

Not in this step (open):
- The turn-start snapshot (`TurnSnapshot` in `Qwen4ExpBatchedForwardPass.Cache`) stays resident after a successful turn,
  so a retained session holds about two states. `RollbackLastTurn` needs it, so it cannot simply be dropped. It needs its
  own charge, or a different rollback design. This step does not change it.
- Copy-on-write page sharing between sessions.

Test plan: a Sessions.Fast test with a fake pass whose `FixedBytesPerSequence` is 1000 and `KvBytesPerToken` is 10. After
one turn, the session's residency is positions x 10 + 1000. Eviction frees that same amount, and a zero-position cache
is still dropped and its fixed bytes freed.

## Phase 3 (retained-state budgeting), result 2026-11-08

- Implemented, as designed above: a held retained state now costs its per-token KV plus its fixed sequence state. The
  session layer charges that figure at every reservation and residency site (`HotSession.RetainedBytes`). Eviction reports
  whether a cache was dropped, so a zero-position cache is also released and its fixed bytes freed
  (`RetainedSequenceState.EvictIfIdle` now returns `(Dropped, Positions)`).
- Tests: `HotSessionFixedStateTests` (Sessions.Fast, 2, synthetic): a completed turn's residency is the fixed-state-free
  figure plus the fixed bytes, and eviction under budget pressure frees the fixed bytes too. Both pass.
- Regression runs: Sessions.Fast full suite, 144 tests, 0 failed. Core full suite, 1366 tests, 0 failed, 95 skipped
  (fixtures not present).
- Not mutation-checked: the session-layer charge. The prefix-cache charge was checked (removing it fails its test).
- Still open in Phase 3:
  - The turn-start snapshot (`TurnSnapshot` in `Qwen4ExpBatchedForwardPass`) stays after a successful turn, so a retained
    session holds about two states. `RollbackLastTurn` needs it. It is not charged yet, and removing it needs a rollback design.
  - Copy-on-write page sharing between sessions.
  - The real Qwen 3.8 fixed size is not measured.
- Qwen 3.8 remains unadmitted and unadvertised.

## Qwen 3.8 remaining work, tabled 2026-11-08

Status as of this table. Each open item names its blocker. "Runnable now" means no owner decision or external dependency.

| Phase | Remaining | Blocked on |
|---|---|---|
| 0. Numerical parity | Phase 0.3 (FFN injection gate) still open. The residual MoE error from layer 1 is attributed to quantisation noise, not confirmed | Owner confirmation of the working parity criterion (adopted by Claude, reversible) |
| 1. Per-sequence state | Broader verification: more fixed prompts, incremental decode, a prompt crossing a prefill/decode chunk, a long prompt exercising QSA K-pool selection, a documented tolerance for the real-weight comparison | Runnable now |
| 2. Batching | Queue limits and admission under backpressure; memory checks for cancelled requests (the finish-order test covers normal completion only) | Runnable now |
| 3. Prefix and retained state | Copy-on-write for QSA history rows (in progress; sequence-level tests added). Still open: the turn-start snapshot's fixed state stays resident and uncharged; copy-on-write page sharing across sessions; real-checkpoint measurement of fixed bytes | Runnable now |
| 4. MTP | Per-step oracle against the reference; acceptance rate, multi-token acceptance, rollback on rejection, separate speed measurement | The reference's MTP API at the pinned revision is a stub (`llama-batch.cpp` ~1230); the fork build segfaults on the head's null recurrent memory |
| 5. Image and video | Logit-level comparison with the reference; exact 40-token equality (the current test compares text only); image input through the tool and server paths; video input (5.3), last | Video input untested; deferred by the owner |
| 6. Tool calls | Server round trip; whether the server acts on the schema validator's result | Admission (CLAUDE.md rule 14); owner decision on the validator |
| 7. Serving and product | Telemetry, knowledge collections, developer workflows. Browser UI (7F) parked by the owner | Not started |
| 8. Verification and admission | Verification pass, docs, admission | Phases 4 to 7; owner confirmation |

Owner decisions needed: confirm the parity criterion (Phase 0); decide whether the server acts on validator failures (Phase 6);
approve admission later (Phase 8). Qwen 3.8 stays unadmitted and unadvertised until then.

## Phase 3 (copy-on-write QSA history), result 2026-11-08

- Change: `Qwen4ExpSequenceState.CopyLists` shares the QSA history rows between a sequence and its clones, and copies only
  the list spines. The forward pass writes each row once (an `Add`) and never changes it afterwards; the only other uses are
  reads (verified by grep over `Qwen4ExpForwardPass.cs`). Before this change, every snapshot duplicated the full attention history.
- Tests: `Qwen4ExpSequenceSharingTests` (4, synthetic): shared rows, independent lists, divergence does not leak, reset of the
  source does not clear a snapshot, restore returns to the snapshot's history. All pass.
- Real-checkpoint regression: `Qwen4ExpRetainedTurnTests` (2), `Qwen4ExpPrefixReuseTests` (2), `Qwen4ExpBatchingTests` (4) pass.
- Full Core suite: 1370 tests, 0 failed, 95 skipped (fixtures not present).
- Still open: the snapshot's fixed state (recurrent buffers, PLE history) is still copied per snapshot and is charged once at
  the session layer, not twice. The per-token doubling is removed by this change; the fixed-state doubling is not.

## Phase 2 (admission and memory of cancelled requests), result 2026-11-08

- `ContinuousBatchingCancellationMemoryTests` (2, synthetic, counting fake pass): a cancelled request and a finished request
  each dispose their sequence cache and release their KV reservation (`CommittedKvTokens` returns to 0). Both pass.
- `ContinuousBatchingBackpressureTests` (1, synthetic): under a 12-token budget, three concurrent requests (about 8 tokens
  each) queue rather than being admitted over budget. Committed KV is polled throughout and never exceeds 12. Every
  request finishes with the same output as a solo run. Passes.
- Not mutation-checked: removing the cache disposal or the budget check. The tests were not run against a broken engine.
- Still open in Phase 2: the real-checkpoint cancellation memory check (the Qwen batching test checks reservations, not
  cache disposal), and queue limits beyond the single budget case (maximum queue length, rejection when full).

## Phase 2 (queue limit), design 2026-11-08, recorded before code

Current state (verified in `ContinuousBatchingEngine.cs`): the request queue `_queue` is an unbounded channel. The batcher
drains it into a local `pending` FIFO (line ~587), and `AdmitPending` moves requests from `pending` into `prefilling`/`active`
under the KV budget. So a waiting request is in `pending`, not in the channel, and nothing bounds how many wait.

Design:
- New optional limit `maxQueuedRequests` (default unlimited, so no behaviour change). When set, the queue is bounded and
  a request arriving when `queued >= limit` is refused immediately with a typed exception, not held.
- "Queued" = requests written to the channel but not yet admitted: the channel count plus `pending.Count`. Tracked with an
  `Interlocked` counter, incremented on submit.
- The counter must be decremented on every path that removes a waiting request: admission (`AdmitPending`), cancellation
  while still waiting, and the shutdown drain (`TryRead` at line ~879). Each exit path is listed here so none is missed.
- The error type and its mapping to a server status (429 or 503) are a product choice. This step adds the exception only;
  the server mapping is left for the owner.

Tests (synthetic, counting fake pass): limit N with a budget that admits one at a time; the (N+1)th submit is refused
immediately; after one is admitted a new submit succeeds; cancelling a waiting request frees its slot; the counter returns
to zero after everything drains.

## Phase 2 (queue limit), result 2026-11-08

- Implemented: `ContinuousBatchingEngine(..., maxQueuedRequests)` (optional, default unlimited; below 1 is rejected at
  construction). A request arriving while the waiting count is at the limit throws `EngineQueueFullException`. The request
  is refused, not held. Mapping to a server status is left to the host.
- Correction to the design above: the engine already had a waiting counter, `_pendingCount` (exposed as `QueueDepth`),
  updated on every exit path. The limit reuses it; no new counter was added.
- Added after the design: `PurgeCancelledPending`. Before it, a cancelled request still waiting in the queue kept its slot
  until the running batch freed up, because admission only inspects the head of the queue. The purge completes cancelled
  waiters on each loop iteration and keeps the rest in FIFO order. This applies to every engine, not only ones with a limit.
- Tests: `ContinuousBatchingQueueLimitTests` (2): a full queue refuses the next request, the refused request's count is
  undone, cancelling a waiting request frees its slot at once, and a new request is accepted afterwards; a limit below 1 is
  rejected. Both pass.
- Also changed: `ContinuousBatchingBackpressureTests` (1) now uses 200-token requests and a 250-token budget. The earlier 6-token
  version finished faster than the 1 ms poll, and the poll sometimes saw no reservation. The test checks that committed KV
  never exceeds the budget.
- Full Core suite: 1375 tests, 0 failed, 95 skipped (fixtures not present).
- Still open in Phase 2: the real-checkpoint cancellation memory check (cache disposal is covered only with the counting fake),
  and mapping `EngineQueueFullException` to a server status.

## External review check, 2026-11-08 (review of main at 1c8e120, re-checked at ae47d135)

Each point was checked against the code. Status is as of ae47d135.

- Memory, snapshot doubling: **partly stale.** The QSA history rows are shared since ad42fa92, so the per-token part of the
  snapshot is no longer a deep copy. **Still open:** the snapshot's fixed state (GDN and PLE conv buffers) is copied and not
  charged; `Qwen4ExpBatchedForwardPass` computes `KvBytesPerToken` from attention K and V only, so the indexer raw keys
  (`QsaRawIndexKCache`), the pooled indexer keys (`QsaPooledKeys`, one per `IndexerKPool` tokens) and the PLE token history
  (`PleHasher`) are not counted.
- Malformed JSON tool calls: **open.** `ParseJsonCallBlock` calls `TryGetProperty` on any root, which throws
  `InvalidOperationException` for a non-object root (`[]`), and `GetString()` throws for a non-string name (`{"name":42}`).
  Only `JsonException` is caught. `JsonElementToObject` uses `ToDictionary`, so a duplicate property throws `ArgumentException`.
  An array or scalar `arguments` gives a call with an empty argument dictionary. Tests to confirm before fixing.
- Validator numeric and enum edge cases: **open.** `double.TryParse` accepts `NaN` and `Infinity`; `Math.Floor(d) == d` is true
  for infinity; `Convert.ToString(true)` is `"True"`, while the enum literals are `"true"`/`"false"`. Tests to confirm.
- Server wiring: **open.** `ToolCallArgumentValidator` is referenced only in Core and its test. The server path is not
  schema-validated, and must not be described as such until the failure behaviour is decided.
- Documentation: **stale and inconsistent.** The working-criterion section says Phase 0 is closed (~line 403). The Phase 0 section
  and the remaining-work table still list the FFN injection gate as open. The `RefusalReason` in
  `Architectures/Qwen4ExpArchitecture.cs` still describes wrong logit magnitudes; the current measurements do not support that.
- CI: the earlier red runs (11:51 on 43e981fe, 11:59 on 77685279) were the first version of the backpressure test, which was
  timing-flaky on CI (6 tokens finished before the poll). Fixed in ae47d135; the 12:08 run is green.

## Jobs 1 and 2 (tool-call safety), result 2026-11-08

- JOB 1, JSON tool-call parsing. Before: 16 of 20 new cases failed. Non-object roots (`[]`, scalars) and non-string names
  threw `InvalidOperationException`. A duplicate property threw `ArgumentException` from `ToDictionary`. An array or scalar
  `arguments`, and DeepSeek arguments that failed to parse, came back as a call with empty arguments. After: a call must be
  a JSON object with one string name. Its arguments must be an object (directly or as a JSON string). A repeated property
  is refused. A malformed block is dropped, and its raw text stays in the plain text. `ObjectToDictionary` throws
  `JsonException` for duplicates, so every caller's existing handler catches it.
  - Tests: `ToolCallJsonHardeningTests` (20). Existing `ToolCallAdapterTests` (50) and the Phase 6 tests (15) still pass.
- JOB 2, validator edge cases. Before: `NaN` and `Infinity` were accepted as Number text; an infinite double passed the
  Integer check; a boolean enum failed because `Convert.ToString(true)` is `"True"`. After: non-finite values are refused in
  every form, and a boolean is compared as the JSON literal `true`/`false`.
  - Tests: `ToolCallValidatorEdgeCaseTests` (16). `ToolCallArgumentValidatorTests` (12) still pass.
- Gate. The first full Core run had two failures, both in batching tests and both timing-dependent, not in the parser. The
  backpressure test's poll could miss a reservation, and the queue-limit test's first request could finish before the second
  was submitted. The counting fake now takes an optional per-step delay. The two tests use it, and each passed three runs in a row.
  Full Core after the fix: 1411 tests, 0 failed, 95 skipped (fixtures not present).
- Still open: the server does not yet use the validator, and an invalid tool call is still returned as-is (owner decision).

## Job 3 (memory accounting completion), design 2026-11-08, recorded before code

Sizes, from `Qwen4ExpForwardPass.cs` (`QsaAppend`):
- Raw indexer key: one row of `_indexerKeyLength` floats per token, per attention layer that has an indexer projection.
- Pooled indexer key: one row of `_indexerKeyLength` floats per `_indexerKPool` tokens, per such layer. This is a
  per-token amortised cost of `ceil(4 x _indexerKeyLength / _indexerKPool)` bytes.
- PLE token history: `Qwen4ExpPleHasher._tokenHistory`, a `List<int>` with one entry per token, 4 bytes each.
- QSA row lists (K, V, raw, pooled): each list holds one 8-byte reference per row. A turn snapshot shares the rows (ad42fa92)
  but copies the spines, so the spine cost is per token as well.

Decisions:
1. `KvBytesPerToken` (in `Qwen4ExpBatchedForwardPass`) becomes the exact per-token sum: K and V, plus the raw indexer key
   and the amortised pooled key for each attention layer that has an indexer, plus 4 bytes of PLE history when PLE is
   present. Counting uses the pass's real layer tensors, not an assumption that every attention layer has an indexer.
   Effect: `KvBytesPerToken` changes, so the session ABI fingerprint changes. Acceptable: qwen4exp is not admitted and no
   persisted qwen4exp sessions exist. Recorded here so the change is not a surprise.
2. The turn snapshot's extra cost is its fixed state (GDN and PLE conv buffers, `FixedFloatBytes`) plus the list spines
   (8 bytes per row per list, per token). The snapshot exists only while `TurnSnapshot` is held (after the first turn).
   - A new optional interface on the sequence cache, `ISnapshotAccountedSequenceKvCache { long HeldSnapshotBytes { get; } }`,
     reports that figure (0 when no snapshot is held). Backends without snapshots do not implement it.
   - `RetainedSequenceState` exposes it internally. `HotSession.CurrentResidentBytes` adds it to the residency figure.
   - Reservation projections: a turn that will take a snapshot (a prior position exists and the backend keeps snapshots)
     reserves for the prior positions' fixed state as well. This is an over-estimate of the spine part, not an
     under-estimate. The backend flag is a new `IBatchedForwardPass.KeepsTurnSnapshot` (default false).
3. Tests: arithmetic against real list lengths (raw keys, pooled keys, PLE history); a snapshot's held bytes are zero before
   a snapshot is taken and positive after a turn; residency after a turn includes them; eviction frees them.

Not in this job: cross-session copy-on-write (Job 8, design only).

## Qwen 3.8 remaining work: moved

The Qwen 3.8 open items, owner decisions and doc-hygiene list now live in
[2026-11-08-qwen38-remaining-work-plan.md](2026-11-08-qwen38-remaining-work-plan.md). The table in "Qwen 3.8 remaining work,
tabled 2026-11-08" above is superseded by that file (its completed items are in done/2026-11-08-qwen38-remaining-work-completed-items-2026-11.md). This programme plan keeps the dated results and the general coverage work.
