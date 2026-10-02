# ADR-0002: optimized (chunked) prefill for hybrid-GDN models with an MTP head

**Status:** accepted (user decision)  
**Date:** 2026-10-02  
**Switch:** `STINGRAY_GDN_CHUNKED_PREFILL` (default on; `=0` forces the StrictSequential per-token path for every model)  
**Code:** `HybridGdnForwardPass.Prefill` / `PrefillChunked` (`src/OpenTail.Stingray.Engine/HybridGdnForwardPass.cs`)  
**Evidence:** [docs/1-correctness/2026-10-02-prefill-numerics-investigation.md](../1-correctness/2026-10-02-prefill-numerics-investigation.md)
(reproduction kit: `docs/4-performance/patches/2026-10-02-prefill-numerics/`)

## Context

Hybrid Gated-DeltaNet models (`qwen35` / `qwen35moe`: Qwen3.5/3.6/3.8, Ornith, Bonsai2) have two
CPU prefill paths:

- **StrictSequential** (formerly "exact"): one decode-forward per prompt token. It is
  bit-identical to decode, but prefill runs at the decode rate.
- **Optimized** (`PrefillChunked`): layer-major over the prompt, with three changes, none an
  algorithm change:
  - the chunked GDN recurrence (FlashQLA-style chunk_gated_delta_rule);
  - batched projections (a packed FP32 GEMM for 64 or more tokens);
  - a batched dense FFN / MoE.

Until this decision the optimized path was the default for every hybrid model **except those
with a native MTP head** (Qwen3.6-27B-MTP, Qwen3.8-27B, Qwen3.6-27B UD). The code comment cited
one test, `MtpDecoder_GreedyParity_LlamaCpp`: the chunked recurrence "is not bit-exact" and
flipped a knife-edge token against llama.cpp. Cost: Qwen3.8-27B prefill ran at its decode rate
(2.1 t/s, 0.07x of llama.cpp). Vulkan and CUDA batched prefill never had this exclusion.

## Evidence

The investigation was run on Ornith-1.0-9B (same `qwen35` trunk, no MTP head) and reviewed step by
step with an outside reviewer. Full record:
[investigation](../1-correctness/2026-10-02-prefill-numerics-investigation.md).

1. **The receipt is not a clean oracle.** On the StrictSequential path it already disagrees with
   llama.cpp at the first generated token, and realigns at `<think>`. Its checkpoint is not on the
   reference machine, so the flip was never re-verified and no margin was recorded.
2. **llama.cpp b10306's CPU GDN recurrence is sequential over tokens.** Our per-token path matches
   its *recurrence* order, but "exact" only ever meant "same as our own decode".
3. **Attribution matrix** (41 prompts, logits at every position, an FP64-recurrence reference D):
   - R vs A distance to D: median log-ratio of KL 0.00 [-0.04, +0.06]; R closer on 20/41 prompts.
   - Top-token flip rate vs D: A 1.61%, R 1.44%.
   - **Zero flips at certified positions** (top1-top2 margin m > 2 * max|delta logit|, a
     provable no-flip bound) across 8,197 positions x 8 comparisons.
   - Each single reordering (recurrence, projections, FFN) produces the same class of drift as A
     vs D itself (1.7-2.6% flips at 500-1200 tokens).
4. **Where error is born:** on identical inputs the recurrence error is at FP32 epsilon (sequential
   2-4e-7, chunked 1.3-3e-7, chunked marginally better). Hidden divergence then grows to 3-5e-2
   through 32 layers, at the same rate for both paths. That is downstream amplification.
5. **No prefill->decode discontinuity** after chunked prefill (16 teacher-forced decode steps at
   19-1200 tokens). At a `<think>` boundary token, **both** FP32 paths flip against D.
6. **The StrictSequential path is the least precise component measured.** At the worst-KL
   position, its per-token decode matvec (int8 activations, as llama.cpp's) has 1.2e-2 relative
   error per projection, against 9e-7 for the batched GEMM, both versus an FP64 dot of the
   dequantized weights. That is normal int8 noise: llama.cpp's own Q8_K is about 2.5x noisier on
   the same rows (investigation step 11).

**Terminology:** "exact" is retired. The paths are StrictSequential (A), ChunkedRecurrence (R),
Optimized (E) and DoubleGdnReference (D: FP64 recurrence only, int8 projections, a partial
reference). The numerical reference for one operation is an FP64 dot over the dequantized stored
weights and the supplied FP32 activations. **A reproducible computation is not necessarily a more
accurate one.**

## Decision

- The optimized prefill is the default for **all** hybrid-GDN models on CPU, MTP models included
  (the `!_hasMtp` exclusion is removed).
- StrictSequential stays available as a diagnostic control: `STINGRAY_GDN_CHUNKED_PREFILL=0`.
- MTP is an output capability, not a numerical reason to disable optimized prefill. Only
  demonstrated behavioural harm would earn that penalty.

## Consequences

- MTP models get the optimized prefill speed. Measured below.
- `MtpDecoder_GreedyParity_LlamaCpp` runs on the optimized path when its checkpoint is present. If
  it flips at the `<think>` boundary, record the margin and re-baseline the fixture as a documented
  knife-edge (its doc comment says so).
- `HybridGdnChunkedPrefill_MatchesSequentialPrefill` keeps its 0.9995 cosine floor, calibrated on
  one model; its doc comment points to the margin certificate as the principled check.

### Measured on an MTP checkpoint (Qwen3.8-27B UD-Q3_K_XL)

2026-10-02, CLI, 620-token prompt, 32 generated tokens, greedy, run back to back on an
otherwise idle machine, with llama-bench in the same session:

| Path | Prefill | Decode | Generated text |
|---|---|---|---|
| StrictSequential (`STINGRAY_GDN_CHUNKED_PREFILL=0`, the old MTP default) | 2.1 t/s | 2.1 t/s | (reference) |
| **Optimized (new default)** | **8.8 t/s (4.2x)** | 2.0 t/s | identical |
| llama.cpp b10306 `llama-bench` pp512 | 4.36 t/s | - | - |

On this MTP checkpoint, prefill goes from 0.48x to about 2.0x of llama.cpp (prompt lengths
differ: 620 vs 512). The likely reason: llama.cpp's fused CPU GDN op runs the recurrence
sequentially over tokens, while the optimized path chunks it and batches the projections.

## Rollback ("a really bad turn")

`STINGRAY_GDN_CHUNKED_PREFILL=0` restores StrictSequential prefill for every hybrid model, with no
rebuild. Re-gate MTP models only for **demonstrated behavioural harm**, any of:
- a systematic disadvantage of the optimized path against a higher-precision reference over a
  prompt corpus;
- a flip at a certified position (m > 2δ; that would also indicate a harness or kernel bug);
- a pathological layer;
- a prefill->decode discontinuity;
- measurably lower MTP draft acceptance or generation quality on the 27B validation.

## Open (validation, not blocking)

- 27B MTP validation with the same harness: boundary-sensitive, ~500 and ~2000 tokens; A vs E vs
  llama.cpp (`llama-server` pre-sampling `n_probs`); continuity.
- Paired MTP draft acceptance, StrictSequential vs optimized prefill.
- Re-baseline the receipt test when `Qwen3.6-27B-MTP-Q4_K_M.gguf` is available.
