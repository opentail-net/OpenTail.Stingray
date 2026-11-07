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

## Phases (in order; do not start a later one before the current one's completion criteria)

| # | Family | Checkpoint available? | Status | What's actually left |
|---|---|---|---|---|
| 0 | Coverage baseline | n/a | **Already current** -- [ported-families-todo.md](ported-families-todo.md) and this doc's table above are the baseline; no separate pass needed right now | Re-check before each later phase starts, don't re-derive from scratch |
| 1 | `qwen4exp` (Qwen 3.8 Flash Next) | yes, paged, 62 GB | **ROOT-CAUSED AND FIXED 2026-11-07.** Reproduced the garbled-text claim, then bisected layer-by-layer against a freshly-rebuilt pinned llama.cpp reference (`qwen4exp.cpp` at `bed0a8566`, via `llama-eval-callback`) on the identical prompt/checkpoint: `hc_mixed` and the raw QKV projection matched within ~1%, `alpha`/`beta` projections matched almost exactly, but `conv_output_raw` was off by ~3.5x. Root cause: `blk.N.ssm_conv1d.weight` is stored by GGUF as `[channel][kernel]` (kernel innermost per `ggml_compute_forward_ssm_conv_f32`), the transpose of what `GdnKernels.CausalDepthwiseConv1dDecode`'s `weight[k*channels+c]` access pattern expects. `HybridGdnForwardPass` (Qwen 3.5 hybrid, already verified) already transposes this at load time (`LoadConv1dTransposed`); this port never did. Fixed with the same load-time transpose, cached per layer. After the fix: layer-0 raw GDN-mixer output sum moved from +30.5 (wrong sign, ~12x off) to -2.76, matching llama.cpp's -2.59 within quantisation noise; 20-token greedy continuation of `"The capital of France is"` now reads `"Paris.\nThe capital of France is Paris.\nIs the following sentence true?\n\n<think>\n"` -- correct and coherent. Regression test: `Layer0GdnMixerOutput_MatchesLlamaCppEvalCallback_AfterConv1dTransposeFix`. | Remaining per the original plan: more fixed prompts (incremental decode, a prompt crossing a prefill/decode chunk boundary, one long enough to exercise QSA K-pool selection), and a documented tolerance for the real-weight comparison. MTP and vision remain deferred. Still alpha / not admitted (CLAUDE.md rule 14) until that broader verification is done |
| 2A | `glm-dsa` (GLM-5.2) | shard-1 metadata only (~9 MB of a much larger checkpoint) | Synthetic-tested + metadata verified against real checkpoint | Real forward-pass/parity work needs the full checkpoint, which doesn't fit here. Rescope: finish synthetic-level correctness checks (MLA absorption, DSA indexer scoring, Hadamard rotation, sigmoid MoE routing, shared-expert contribution) against the llama.cpp source reference, same as the original plan's step 3, but drop the "real-weight comparison" completion criterion unless disk changes |
| 2B | `glm5next` (GLM-5.3-Flash) | paged, large (Q2_K_XL ~236 GB; have UD-IQ1_S shards) | K-pool selection doesn't drive attention yet (confirmed above) | Implement K-pool selection actually gating attention reads, remove the full-attention/throw shortcut, add the isolated numerical tests the original plan lists (KDA recurrence, Sinkhorn mHC, pool construction/selection, sequence isolation). This is real implementation work, not verification -- size it accordingly |
| 3 | `deepseek4` (DeepSeek-V4 Flash) | yes, paged, 86 GB | Refuses YaRN correctly; CSA unexercised | Implement YaRN rope scaling for real (factor 16, confirmed from the real checkpoint), remove the refusal only once verified against the pinned llama.cpp reference, then exercise non-zero `compress_ratios` (confirmed present: `[0,0,2,2,...]`-shaped) for the first real CSA test |
| 4 | Candidate reassessment | n/a | Not started | Survey Muse Spark / MiniMax M3.1 / others per the original plan's instructions; this is a report, not a port, and is cheap -- can run any time, doesn't block 1-3 |

## Explicitly out of scope (do not start without a new disk/scope decision)

- `deepseek41` (DeepSeek-V4.1 Flash) and `deepseek32` (DeepSeek-V3.2) real-weight work -- checkpoints
  too large for this machine (confirmed 2026-11-07, see above). Metadata-only checks already exist
  for `deepseek41`; `deepseek32` has none and no cheap path to get one.
- Any new model family port (Muse Spark, MiniMax M3.1, etc.) -- Phase 4 is assessment only.
- CUDA work, new GPU requirements, vision towers, MTP/speculative drafting, batching -- all
  explicitly out per the original plan and CLAUDE.md's existing scope rules.

## Recommended starting point

Phase 1, step 1 only (reproduce the Qwen4Exp claim with a fixed prompt before touching any code).
It's the cheapest way to find out whether this programme's highest-priority item is still real.
