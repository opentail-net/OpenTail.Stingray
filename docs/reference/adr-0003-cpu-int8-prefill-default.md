# ADR-0003: int8 activation prefill is the CPU default for dense models

**Status:** accepted (user decision)  
**Date:** 2026-10-02  
**Switch:** `STINGRAY_CPU_PREFILL_Q8` (default on; `=0` restores the per-token prefill path, bit-identical to decode)  
**Code:** `SimdKernels.Q8PrefillEnabled` (`src/OpenTail.Stingray.Cpu/SimdKernels.cs`)

## Context

CPU prompt processing (prefill) can multiply each quantized weight row against a batch of
prompt tokens in two ways:

- **Per-token path (`=0`, historically called "F32 prefill").** Each token goes through the same
  kernels as single-token decode, so prefill logits are bit-identical to a token-by-token run.
  The name is misleading: for K-quant weights the decode matvec itself quantizes activations
  to int8 (Q8_KS) before the dot. The ADR-0002 row check measured that at about 1e-2 relative
  error per projection, against about 1e-6 for an F32-activation GEMM. "Bit-identical to
  decode" here means reproducible, not more precise.
- **Batched int8 path (`=1`).** Activation rows are quantized per block (Q8_K / Q8_KS / Q8_0) once
  for the batch. Each weight row is read once and dotted against 4-8 tokens (`_4In`/`_8In`
  kernels, the repacked Q4_Kx8 GEMM). This is the technique llama.cpp's CPU backend uses for its
  own prompt processing: `src1` is converted to the weight type's `vec_dot_type` (Q8_K or Q8_0)
  and dotted in int8. It differs from the per-token path in how the activations are grouped
  and quantized, not in whether they are.

History:
- int8 prefill was the default until **2026-10-01**.
- Commit `7791e3c9` turned it off for "exact numerical parity". The trigger was the
  Granite-4-H-Small MoE investigation
  (`docs/done/13-granite4-h-small-moe-ppl-parity-plan.md`): int8 activation
  quantization in the batched MoE expert path accounted for all of the batched-vs-per-token NLL
  differences.
- The off default cost every dense model its prefill speed. The performance plan's evidence
  table (`docs/4-performance/perf-sweep-plan.md` 10.2) and the Gemma re-measure (Phase 6) put
  the decision to the user.

## Evidence (all measured on this repo's reference box, 8-core Ryzen 5700G)

| Model | llama.cpp PPL | F32 prefill PPL | int8 prefill PPL | Prefill t/s F32 -> int8 | Source |
|---|---|---|---|---|---|
| Qwen2.5-0.5B Q4_K_M | 12.0055 | 11.9702 | 11.9693 | 137 -> 309 (2.2x) | perf-sweep-plan 10.2 |
| SmolLM2-1.7B Q4_K_M | 6.9414 | 6.9437 (+0.03%) | 6.9891 (+0.69%) | 77.5 -> 227 (2.9x) | perf-sweep-plan 10.2 |
| Gemma-3-4B Q4_K_M | - | - | - | 44.0 -> 107.9 (0.39x -> 0.95x of llama.cpp) | PerformanceLeague 2026-10-02 |
| Gemma-4-12B Q4_K_M | - | - | - | 14.5 -> 36.4 (0.39x -> 0.97x of llama.cpp) | PerformanceLeague 2026-10-02 |
| Granite-4-H-Small (MoE) | 26.1080 | 26.5483 | 26.4155 | - | granite plan 13 (MoE int8 at the time) |

PPL: wiki.test.raw, `stingray perplexity --batched -g 0 -c 2048`, `[1024,+)` bucket, against
`llama-perplexity -c 2048 --chunks 1`.

Greedy parity: DeepSeek-V2-Lite matched llama-server for 16/16 tokens with int8 prefill and
departed at token 9 with F32 prefill (`DeepSeek2GreedyParityTests` pins int8 on). The Q3_K
kernel rewrite `e7b7aa8a` later flips that same knife-edge token on both paths; see
`docs/1-correctness/bugstofix.md` item 24, which is a separate decision.

## Decision

- `STINGRAY_CPU_PREFILL_Q8` defaults **on**.
- Routed MoE experts are **not** affected. They take int8 only when
  `STINGRAY_MOE_PREFILL_Q8=1` is also set (default off,
  `MoeBatchedExperts.Q8PrefillEnabled`), so the Granite MoE finding that motivated 2026-10-01
  stays fixed.
- All-control-token prompts keep their existing F32 exception in `ForwardPass`.

Why:
- 2.2-2.9x prefill on Q4_K dense models, and Gemma moves from 0.39x to 0.95-0.97x of llama.cpp.
- The quality cost is neutral to +0.69% PPL on the models measured.
- It is the same class of approximation llama.cpp ships by default.
- "Prefill bit-identical to decode" is an internal consistency property, not a fidelity
  guarantee. The same conclusion came out of the hybrid-GDN investigation (ADR-0002): FP32
  reduction-order and activation-rounding changes of this size sit inside the noise band the
  model already has.

## Consequences

- Prefill logits on dense CPU models are no longer bit-identical to a token-by-token decode run.
  Tests that pin the exact path set `SimdKernels.Q8PrefillEnabled = false` explicitly. These
  are `MatMulBatchedEquivalenceTests`, `ContinuousBatchingTests`, `PrefillDecodeSelfConsistencyTests`
  (its strict arm) and the MoE strict arms. The int8 arms already had calibrated budgets
  (`PrefillDecodeSelfConsistencyTests`, `OlmoeGreedyParityTests` 0.7137,
  `ApertusGreedyParityTests` ~3.3 < 5.0).
- The hybrid-GDN chunked prefill (`HybridGdnForwardPass`) and the RWKV batched paths also call
  `MatMulBatched(allowQ8: true)`, but neither has `ForwardPass`'s single-distinct-token guard
  (`IsSingleDistinctTokenPrompt`, the 2026-08-07 fix for int8 collapse on repeated-token prompts,
  `docs/done/cpu-prefill-quality-gate.md`).
  - **Hybrid-GDN, measured 2026-10-02** on Ornith-1.0-9B Q4_K_M: chunked prefill with int8 on is
    bit-identical to int8 off, on repeated-token prompts (" the" x8/x16/x48, "\n" x16, " " x16)
    and on prose (16/48 tokens). Both sit at cosine >= 0.9994 vs the per-token path, with the
    same argmax. There's no collapse, and ADR-0002's numbers are unaffected for Q4_K weights.
  - **RWKV is not verified:** no RWKV checkpoint is on this machine. Re-check the repeated-token
    case when one is available.
- Rows in `docs/RUNNING.md` / `PerformanceLeague.md` measured 2026-10-01..02 used the F32
  default. Re-measured rows are dated after this change.

## Known gap (follow-up, not a blocker)

Our int8 prefill is not numerically llama.cpp's. On SmolLM2 the F32 path matches llama.cpp PPL
to 0.03%, while ours with int8 is 0.69% off, although llama.cpp also quantizes activations.

**Cross-checked 2026-10-02** against the vendored ggml (b10306): `ggml_get_type_traits_cpu`
quantizer and `vec_dot` called directly, on identical rows of real SmolLM2-1.7B weights, against
an FP64 dot of the dequantized weights. Harness: `docs/4-performance/patches/2026-10-02-prefill-numerics/harness/ggmlx`.
- **Both engines' kernels are exact.** All of the error is activation quantization, which both
  apply by design.
- **ggml's Q8_K** (one scale per 256) has 2.1-3.5% activation error, and its Q4_K projections are
  0.6-4.5% off.
- **Our per-token decode Q8_KS** (one scale per 32) has 0.8-1.3% activation error, and its Q4_K
  projections are 0.3-1.6% off: about 2.5x more precise than llama.cpp.
- **Our Q6_K decode** is bit-identical to ggml's.
- **Our batched Q4_K prefill** (the repacked Q4_Kx8 GEMM, `RepackedGemm*.cs`) quantizes activations
  to ggml's own Q8_K, so it carries llama.cpp's activation-noise level. The per-token ("F32") path
  is actually the more precise one.

So the 0.69% isn't a precision defect in our int8 kernels, and its remaining source is still open.
Follow-ups:
- a Q8_KS variant of the repacked Q4_Kx8 GEMM (prefill more precise than llama.cpp at similar speed);
- the Q3_K batched path's format (it reportedly uses Q8_KS where decode uses Q8_K).

The follow-up list (quantize-once MoE dispatch, Q8_K alignment, then re-deciding
`STINGRAY_MOE_PREFILL_Q8`) is tracked in `docs/4-performance/perf-sweep-plan.md`.

## Rollback

`STINGRAY_CPU_PREFILL_Q8=0` restores the per-token prefill (bit-identical to decode) for the whole
process, with no rebuild needed. A code rollback is the one-line default in `SimdKernels.Q8PrefillEnabled`. Roll
back if a parity receipt or PPL check shows int8 prefill materially worse than F32 on a
supported model: worse than llama.cpp by clearly more than the +0.69% measured here, or a
greedy receipt that F32 passes and int8 fails for a reason other than a documented knife-edge.
