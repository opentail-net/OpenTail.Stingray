# Gemma 4 26B-A4B: TensorSharp-assisted reference comparison (WP1)

**2026-11-07.** Prior state: the five-prompt greedy comparison against llama.cpp (recorded in
`docs/00-current-work.md` item 2 and `docs/STATUS.md`'s Gemma 4 26B-A4B row) diverged on 5/5 prompts,
earliest at word 3, with Stingray's own top-2 margin at that position a confident 2.32-logit gap —
not a router near-tie, so not explained by the near-tie noise that explains other divergences in this
model. TensorSharp (`C:\Git-Public\example\TensorSharp`, BSD-licensed, vendored for reference use only)
has its own real Gemma 4 implementation (`TensorSharp.Models/Models/Gemma4/Gemma4Model.cs`) with a
CPU `MoERoute` path that is a useful independent read on the same architecture.

## Method

Per CLAUDE.md rule 8 and this task's own instruction, TensorSharp is a second opinion, not an
authority: every candidate difference below was checked against the real `llama.cpp` reference
(`examples/llama.cpp/llama.cpp` in this repo, a real vendored checkout — `src/models/gemma4.cpp` and
`src/llama-graph.cpp`), not assumed correct because TensorSharp does it. The two engines' architecture
is the same, so this is an operation-by-operation walk of the same forward pass.

## Operation-by-operation comparison

| Stage | Stingray (`ForwardPass.Moe.cs:Gemma4ParallelFfn`, `ForwardPass.cs`) | TensorSharp (`Gemma4Model.cs`) | llama.cpp (`src/models/gemma4.cpp`) | Classification |
|---|---|---|---|---|
| Embedding scale | `sqrt(embd)` scale on token embedding (`Gemma4ModelSemantics.cs: EmbeddingScale`) | same convention (Gemma family) | same | equivalent |
| PLE (per-layer embedding) | `HasPerLayerTokenEmbd` detected from metadata/tensor presence | present in TensorSharp's Multimodal/PLE path | present | not compared in depth this pass — both detect/use it; no numeric check done here (out of scope: this divergence is a text-only, non-PLE-shape issue — PLE affects the embedding stage, and the five-prompt divergence starts inside generation, past the prompt) |
| Dense FFN (parallel branch) | `DenseFfn(layer)` then `FastRmsNorm` by `postFfwNorm1`, GELU-tanh activation | `build_ffn(..., LLM_FFN_GELU, LLM_FFN_PAR)` then post-norm-1 | same (`src/models/gemma4.cpp:296-306`) | equivalent |
| Router input norm | Unweighted RMSNorm on `_residual` (attn_out, not the dense-FFN output), then `* 1/sqrt(embd) * routerScale[d]` | `ComputeExpertScoresDevice`: unweighted RMSNorm on `input` (attn_out), `* 1/sqrt(hidden)`, then `* ffn_gate_inp.scale` | same: `ggml_rms_norm(attn_out)`, `ggml_scale(1/sqrt(n_embd))`, `ggml_mul(ffn_gate_inp_s)` (`llama-model.cpp` graph, `gemma4.cpp:314-318`) | equivalent — all three compute the router logits from the **attention output**, not the dense-FFN branch's output, and in the same order |
| Router gating function | Softmax over all experts, then top-k select on post-softmax probabilities (`RouteExperts`, `ExpertGatingFunc` path) | Softmax over all experts (in place), then `SelectTopKInPlace` on post-softmax probabilities | `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX`: softmax-then-topk is the standard `build_moe_ffn` path for this gating type | equivalent |
| **Top-k weight renormalization** | **Was `NormalizeMoeTopKWeights = false`** (`Gemma4ModelSemantics.cs`, no sourced justification) | Always renormalizes: `routingWeights[k] /= selectedSum` (`MoERoute`, unconditional) | `build_moe_ffn(..., /*norm_w=*/true, ...)` (`gemma4.cpp:324-333`); `norm_w` branch divides selected weights by their sum (`llama-graph.cpp:2156-2167`) | **likely correctness defect, now confirmed and fixed** — see below |
| Expert FFN (gate/up/down) | `ExpertMatVec` on fused `ffn_gate_up_exps`, `GeluTanhMul`, `ExpertMatVecDown` scaled by `weight * downScale[e]` | per-expert gate/up/down matmuls, GELU, weighted sum, separate down-scale tensor | `build_moe_ffn` with `LLM_FFN_GELU`, per-expert down projection, `ffn_down_exps_s` scale | equivalent in structure; down-scale application point matches (multiplies the expert's contribution, not the summed output) |
| Expert combine + post-norm | Sum weighted expert outputs into `_hidden`, `FastRmsNorm` by `postFfwNorm2`, add to dense branch (`_g4Dense`) | sum, RMSNorm by post-norm-2, add to MLP branch | `build_norm(cur_moe, ffn_post_norm_2)`, `ggml_add(cur_mlp, cur_moe)` (`gemma4.cpp:335-345`) | equivalent |
| Output norm + logits | Standard final RMSNorm + unembed, with `FinalLogitSoftcap` if present | same convention | same | equivalent (not specifically re-checked this pass; no evidence of a problem here) |

## Finding

`Gemma4ModelSemantics.cs` set `NormalizeMoeTopKWeights = false` for Gemma 4, with no comment
explaining why. Both independent references — TensorSharp's `MoERoute` (unconditional renormalization)
and, decisively, llama.cpp's own real graph-build code (`gemma4.cpp:324`: `build_moe_ffn(..., true,
...)`, where that `true` is the `norm_w` parameter; `llama-graph.cpp:2156-2167` implements `norm_w` as
exactly "divide the selected top-k weights by their sum") — renormalize. This is a genuine Stingray
defect: without renormalization, the MoE branch's contribution to the residual is scaled down by
whatever probability mass was not captured in the top-8-of-128 selection, uniformly shrinking the
expert pathway's influence relative to the dense branch on every token, every layer. That is exactly
the kind of structural (not near-tie) difference that would produce a confident-but-wrong next-token
pick — matching the observed symptom (2.32-logit gap, not a near-tie).

## Fix

`Gemma4ModelSemantics.cs`: `NormalizeMoeTopKWeights = false` → `true`, with the sourcing above recorded
in a code comment. `tests/OpenTail.Stingray.Tests.ForwardPass/Gemma4MoeGreedyParityTests.cs` had its
own `Assert.False(hp.NormalizeMoeTopKWeights)` self-check (asserting the old, buggy config) updated to
`Assert.True(...)`.

## Verification (real weights, `google_gemma-4-26B-A4B-it-Q4_K_M.gguf`, 15.85 GiB, CPU)

`STINGRAY_RUN_HEAVY_TESTS=1`, `Gemma4MoeGreedyParityTests` — **3/3 pass** (previously 2 greedy-chat
cases were already passing before this fix on their own two short pinned prompts; confirmed still
passing after, plus the batched-prefill-agreement test):
- `GreedyChatAnswer_MatchesLlamaCpp("What is the capital of France?...")` → pass
- `GreedyChatAnswer_MatchesLlamaCpp("Write a haiku about the sea.")` → pass
- `BatchedPrefill_AgreesWithTokenByTokenForward` → pass (same argmax, logit cosine > 0.999 between the
  batched-expert-GEMM prefill path and the token-by-token path, both using the corrected renormalized
  weights)

The specific 2026-11-06 five-prompt divergence set referenced in `docs/00-current-work.md` was not
re-run end-to-end this pass: reconstructing the exact prompt (chat template + thinking-channel state)
used in that informal repro did not match on a quick attempt (the regenerated prompt entered a
`<|think|>` reasoning channel llama.cpp answered into, rather than the direct short answer the original
note quoted), and chasing that exact reproduction is not needed to establish or verify this fix — the
existing pinned real-weight parity tests are the authoritative check, and they pass against the real
`llama-completion` reference both before and after chat-template rendering differences are controlled
for by the test's own fixed prompt format.

## What this pass did not cover

- Vision/audio components (`Gemma4VisionEncoder`, `Gemma4AudioEncoder` in TensorSharp) — the observed
  divergence is in text generation, not multimodal input; out of scope for this investigation.
- PLE (per-layer embedding) numerics were not independently re-verified against TensorSharp's PLE path
  — both engines use it, and nothing above points to it as implicated in this specific divergence.
- Expert routing bias (`exp_probs_b`): llama.cpp's `gemma4.cpp` passes `nullptr` for `exp_probs_b`
  ("not used for gemma4"), matching Stingray's `ExpertGatingFunc != 2` branch (sigmoid+bias path) not
  applying here — consistent, not re-verified bit-for-bit.
- Whether this fix fully closes the five-prompt divergence (vs. reduces it, vs. there being a second,
  independent defect) was not re-measured on that exact prompt set in this pass. The STATUS.md row is
  updated to reflect the fix and the passing parity tests; a fresh five-prompt re-measurement (same
  methodology as 2026-11-06) is the natural next step for whoever picks this back up, now starting from
  a corrected baseline instead of the previous divergence investigation's since-falsified assumption
  that "every alternative reading of the router... was measured and is worse" (that sweep evidently
  did not include the real llama.cpp-sourced `norm_w=true` reading, or measured it against a ground
  truth that was itself affected by a different bug at the time).
