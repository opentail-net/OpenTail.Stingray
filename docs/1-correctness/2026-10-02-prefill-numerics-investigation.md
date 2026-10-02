# Prefill numerics investigation, 2026-10-02

**Question:** are the fast (chunked/batched) prompt-processing paths less accurate than the
per-token path, enough to justify slower defaults?

**Outcome:** no. The per-token path is reproducible, not more precise. Decisions taken:
- [ADR-0002](../reference/adr-0002-hybrid-gdn-mtp-chunked-prefill.md): chunked prefill allowed
  for hybrid-GDN models with an MTP head.
- [ADR-0003](../reference/adr-0003-cpu-int8-prefill-default.md): int8 prefill is the CPU default
  for dense models.

**Method** (reusable): [numerics-investigation-method.md](../reference/numerics-investigation-method.md).
**Code to reproduce:** [patches/2026-10-02-prefill-numerics](../4-performance/patches/2026-10-02-prefill-numerics/README.md).

Model for the experiments: Ornith-1.0-9B Q4_K_M (`qwen35` hybrid-GDN, same trunk as the
Qwen3.6/3.8-27B MTP checkpoints, no MTP head). The work was reviewed step by step with an
outside reviewer (ChatGPT); its main contributions are marked "(reviewer)".

## Starting point

`HybridGdnForwardPass.Prefill` excluded models with an MTP head from the chunked prefill path. The
code comment said the chunked recurrence "is not bit-exact", and that this flipped a knife-edge
token in `MtpDecoder_GreedyParity_LlamaCpp`, so MTP models "stay on the byte-exact per-token
scan". Cost: Qwen3.8-27B prefill ran at its decode rate (2.1 t/s, 0.07x of llama.cpp).
Separately, int8 prefill had been off by default since 2026-10-01 (`7791e3c9`) for the same kind
of "exact parity" reason.

## Steps

### 1. Evidence gathering (code and history)
- The gate cites one test. That test's checkpoint (`Qwen3.6-27B-MTP-Q4_K_M.gguf`) is not on this
  machine, so the flip was never re-verified after the gate went in, and no logit margin was ever
  recorded.
- The test already disagrees with llama.cpp **at the first generated token on the per-token path**
  (`<|im_end|>` vs `\n`). It realigns at `<think>` and compares 60 bytes.
- The gate's comment was stale: the chunked path also batches projections (and, from `a61f53ab`,
  the dense FFN), not only the recurrence. On Qwen3.6-35B the logit gap was unchanged with the
  recurrence swapped back to sequential (`e0558a87`).
- Vulkan and CUDA batched prefill never had the MTP exclusion.

### 2. llama.cpp's own recurrence (reviewer)
b10306 (vendored, `tools/llama.cpp/VERSION`) runs the fused CPU GDN op sequentially over tokens
within each head. So our per-token recurrence is the closer match to llama.cpp's *recurrence*,
but "exact" only ever meant "same as our own decode".

### 3. Attribution matrix design
- Temporary switches inside `PrefillChunked` toggle each change independently: recurrence (R),
  projections (P), FFN (F), all on (E = production).
- Reference A = per-token decode path, with logits captured at **every** prompt position
  (teacher-forced).
- **Sanity config S** (everything per-token inside the chunked path) must equal A bit for bit:
  it did, every prompt.
- Metrics per position: top-1/top-2 margin m, δ = max|Δ logit|, the certificate m > 2δ
  (reviewer: provably no flip), and a tight version m > |Δtop| + max|Δrest| (reviewer).
  Also argmax flip, cosine, KL and top-5 overlap.

### 4. First result: short prompts
On 13-24-token prompts, P and F are bit-identical to A: below 64 tokens the batched path uses
the same int8 kernel as decode, and the packed F32 GEMM starts at 64. So the recurrence alone
explains short-prompt drift, and it was large for a "reorder": δ up to 2.5, cosine 0.9958.

### 5. Higher-precision recurrence reference D
Added `XDblRec`: the GDN recurrence in FP64 (state in FP64), everything else as A.
- Finding: **A itself is as far from D as R is** (e.g. δ 1.10 vs 0.58, 0.75 vs 1.04, 1.30 vs 3.37).
- At a knife-edge position (short1, pos 7) the chunked path agreed with D and the per-token path
  didn't.

Reviewer: D is a "higher-precision recurrence reference", not truth; keep teacher-forced
comparison for diagnosis and generated behaviour for impact.

### 6. Full matrix, prompt as the statistical unit (reviewer)
41 prompts: 30 short chat prompts, 8 x 500 tokens and 3 x 1200 tokens of repo text. Bootstrap
CIs over prompts, not positions.

| Measure | Result |
|---|---|
| R vs A distance to D, median log-ratio of KL | **0.00 [-0.04, +0.06]**; R closer on 20/41 prompts |
| Top-token flip rate vs D | A 1.61%, R 1.44% |
| Flips at certified positions (m > 2δ) | **0** across 8,197 positions x 8 comparisons |
| Vulnerable positions (m <= 2δ) | ~28% (2δ), ~16% (tight) |
| A/R disagreements tie-broken by D | short R 5 / A 3, med R 37 / A 35, long R 31 / A 54 (3 prompts, correlated) |
| Flip rate vs A, med / long | chunked recurrence 1.82 / 2.42%, batched projections 2.25 / 2.64%, batched FFN 1.72 / 2.53%; A vs D itself 1.92 / 2.03% |

Every reordering produces the same class of drift. The only tilt toward A is on the 3 long
prompts, while the KL ratio on those same prompts is neutral (1.01x).

### 7. GDN probe: born in the recurrence, or amplified downstream? (reviewer)
At each GDN layer, inside the D run, also ran the FP32 sequential and FP32 chunked recurrences
**on identical inputs and state**:
- Local output error vs FP64: sequential 2-4e-7, chunked 1.3-3e-7, chunked vs sequential 2-4e-7.
  Final state: chunked also closer to FP64, e.g. 4.3e-7 vs 5.7e-7 at 500 tokens.
- End-to-end hidden divergence grows from ~1e-4 (layer 0) to 3-5e-2 (layer 31), at the same
  rate for sequential-vs-FP64 and chunked-vs-FP64.

So the drift is downstream amplification of FP32-epsilon differences. The chunked recurrence
is not the source, and is marginally more accurate.

### 8. Prefill -> decode continuity (reviewer)
Prefill N tokens with each path, then 16 teacher-forced single-token decode steps. Chunked-vs-
per-token δ at the last prompt position -> first decode step: 0.40 -> 1.02 (24 tokens),
0.49 -> 0.77 (19), 0.60 -> 0.42 (500), 0.81 -> 0.86 (1200). Per-token-vs-D at the same steps:
0.40 -> 1.10, 0.38 -> 0.73, 0.44 -> 0.41, 0.80 -> 0.93. No discontinuity at the handover. In the
24-token prompt the first decode step is the `<think>` boundary token, and **both** FP32 paths
flip against D there: the receipt's knife-edge is model-intrinsic FP32 noise.

### 9. The two KL outliers and the row check
Two isolated positions dominated the KL tails:
- long40 pos 541: KL 13.8; only batched-projection configs.
- long38 pos 751: every perturbation moved away from A.

To tell "GEMM bug" from "unstable point", each projection at row 541 (and control row 300) was
compared with an **FP64 dot of the dequantized weights and the actual FP32 activations**:

| Row | Batched GEMM (F32 activations) | Per-token decode matvec (int8 Q8_KS activations) |
|---|---|---|
| 541 | mean 9.2e-7, worst 1.5e-6 | mean 1.2e-2, worst 8.3e-2 |
| 300 | mean 9.2e-7, worst 1.3e-6 | mean 1.1e-2, worst 6.0e-2 |

The per-token "exact" path is the noisiest component measured, about 10^4 above the GEMM and the
recurrences, because the decode matvec quantizes activations (as llama.cpp's does). Position 541
is an unstable point where that int8 noise tips the prediction. This also means D (FP64
recurrence on top of int8 projections) was never full truth; E looked "farther from D" because
D inherits A's projection noise.

### 10. int8 default interaction
With int8 prefill on, the hybrid chunked path is **bit-identical** to int8 off (Q4_K): its
batched dispatch matches its per-token path for these weights. No repeated-token collapse (the
2026-08-07 failure, guarded in `ForwardPass` only), so the matrix numbers stand. RWKV is not
verified (no checkpoint).

### 11. ggml cross-check on identical rows (reviewer's request; done 2026-10-02)
The vendored b10306 ggml was called directly (`ggml-base.dll`, `ggml-cpu-haswell.dll`:
`ggml_get_type_traits_cpu` quantizer and `vec_dot`) on real SmolLM2-1.7B Q4_K/Q6_K rows, with
real layer-0 activations and synthetic outlier-heavy ones (amax/rms 25-90). Everything was
compared against an FP64 dot of the dequantized weights.

| | ggml Q8_K (1 scale/256) | Stingray decode Q8_KS (1 scale/32) |
|---|---|---|
| Activation quantization error | 2.1-3.5% | 0.8-1.3% |
| Q4_K projection error | 0.6-4.5% | 0.3-1.6% |
| Q6_K projection error | 2.1-13% | identical to ggml, bit for bit |
| Kernel error beyond quantization | 0 | 0 |

So the ~1e-2 per-projection error seen in step 9 is normal int8 activation noise. llama.cpp's own
is about 2.5x larger on the same rows. Our batched Q4_K prefill (repacked Q4_Kx8 GEMM) uses ggml's
Q8_K format. `Q1_0`/`Q2_0` dequantization is bit-identical to ggml's `to_float` on 786k random
values. Harness: `harness/ggmlx` in the reproduction kit.

## Conclusions

1. "Reproducible" is not "more accurate". The per-token path is a deterministic control, not
   ground truth. Retired term: "exact". Names now: StrictSequential (A), ChunkedRecurrence (R),
   Optimized (E), DoubleGdnReference (D).
2. Numerical reference, defined per operation only: an FP64 dot over the dequantized stored
   weights and the supplied FP32 activations.
3. No systematic disadvantage, no certified flips, no pathological layer, no handover jump.
   Those were the predefined "really bad turn" criteria, and none was met.

## Open / follow-up
- ggml vs Stingray Q8 activation quantization on identical rows (`quantize_row_q8_K_ref` via the
  vendored `ggml-base.dll`): is 1e-2 per projection normal int8 noise or a defect?
- 27B MTP validation: boundary, ~500, ~2000 tokens; A vs E vs llama.cpp; continuity; MTP draft
  acceptance.
- Re-baseline `MtpDecoder_GreedyParity_LlamaCpp` on the optimized path when its checkpoint is
  available.
