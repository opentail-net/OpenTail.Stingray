# CPU quantised GEMM: where else to look (survey, 2026-10-30)

**Why this exists.** On 2026-10-30, Q8_0 prefill got +31-66% from a scheduling change alone (PerformanceLeague,
"Q8_0 batched prefill scheduling"). That raised the question of what the best CPU int8 GEMMs do that ours do not.
llama.cpp is one reference, but not the only one, and on this problem not the most specialised. This page lists the
other sources, what each could give this engine, and how far the claims were checked. It is a reading list with a
recommended order, not a plan.

**Target machine for every judgement below:** Ryzen 7 5700G, Zen 3, AVX2 + FMA, **no AVX-VNNI, no AVX-512**. So the
int8 dot product is `vpmaddubsw` (u8 x s8 -> s16 pairs) + `vpmaddwd`, which is the instruction pair our
`DotQ8_0_Q8_0_*` kernels already use (sign trick: `|w|` x `sign(a, w)`). Results on Zen 4 / AVX-512 hosts do not
transfer directly.

## What we have today (for comparison)

| Format | Prefill kernel | Shape per inner call |
|---|---|---|
| Q4_K | repacked 8-row `Q4Kx8` GEMM (perf-loop iteration 42) | 8 rows x 1 token, row-interleaved weights |
| Q4_K, Q3_K, Q6_K | `TryMatMulBatchedQ8` `_8In` / `_4In` dots | 1 row x 8 or 4 tokens |
| Q4_0, Q5_0, **Q8_0** (since 2026-10-30) | `TryMatMulBatchedQ8` `_4In` dots | 1 row x 4 tokens |

All are tiled 16 rows x token group, with 512-token chunks. None keeps a 2-D (rows x tokens) tile of accumulators in
registers, which is what the faster kernels below do.

## Sources

| Source | Licence | What is relevant | Evidence (as checked) | Fit here |
|---|---|---|---|---|
| **llamafile tinyBLAS** (`ggml-cpu/llamafile/sgemm.cpp`, vendored in llama.cpp) | MIT (Mozilla Foundation header) | `tinyBLAS_Q0_AVX<block_q8_0, block_q8_0>` is what llama.cpp itself runs for Q8_0 / Q4_0 / Q5_0 prefill on AVX2. Register-tiled GEMM. With AVX2's 16 vector registers the main tiles are `gemm4xN<2>` (4 rows x 2 tokens) and `gemmMx4<2>` (2 x 4), with `<3,2>`, `<2,3>`, `<2,2>` and smaller for edges; the 3 x 3 and 4 x 4 tiles are the 32-register (AVX-512) branch. | Source read locally (`examples/llama.cpp`, b10306). Upstreaming note: "Q4_0 and Q8_0 weights should go ~40% faster on CPU". llamafile 0.7 release notes claim 30-500% prompt-eval gains, Q8_0 regressed on M2 Ultra. | **Tried, no gain here** (see below): our `_4In` dot already has the reuse tinyBLAS adds over ggml's `vec_dot`. |
| **ik_llama.cpp** (the K-quants author's fork) | MIT (GitHub API) | Row-interleaved repacked types: `_R4` (4 rows) and `Q8_K_R8` (8 rows), selected at load with `-rtr`. Author calls `Q8_K_R8` the fastest prompt-processing type. | Author's own llama-bench on Zen 4, 16 threads, pp512: Q8_0 **268 vs 148 t/s (1.81x)**, Q4_0 274 vs 154 (1.78x) over llama.cpp. AVX-512 ("fancy SIMD") was on, so the AVX2 gain will be smaller; not measured here. | **High for Q8_0 / Q4_0.** We already repack Q4_K to x8; the same idea for Q8_0 (`Q8_0x4`/`x8`) is the natural next step after tiling. |
| **ONNX Runtime MLAS** (`core/mlas/lib/sqnbitgemm*`, `MatMulNBits`) | MIT | Block-quantised weights x dynamically int8-quantised activations ("CompInt8", `accuracy_level 4`), with an explicit AVX2 `vpmaddubsw` + `vpmaddwd` fallback for hosts without VNNI (ours). Recent work quantises activations straight from fp16. | Commit descriptions only (no benchmark read). | **Medium.** A production-grade design reference for the no-VNNI path, not a drop-in. |
| **FBGEMM** (PyTorch) | BSD-style (GitHub reports `NOASSERTION`; check before copying code) | Weight packing for `vpmaddubsw`: B matrix rows interleaved by 4 for 32-bit accumulation (2 for 16-bit). | Header excerpt and the FBGEMM paper (arXiv 2101.05615); kernels not read. | **Medium.** Packing layout ideas. Its u8 x s8 model differs from ggml's symmetric blocks. |
| **oneDNN docs** ("nuances of int8 computations") | n/a (docs) | `vpmaddubsw` saturates its 16-bit pair sum; s8 x s8 paths do not guard against it. | Intel docs. | **Check, not port.** Our sign trick keeps both operands in [-127, 127], so a pair sum is at most 2 x 127 x 127 = 32,258 < 32,767: no saturation, as long as no quantiser emits -128. Worth a unit test. |
| **T-MAC** (Microsoft) | MIT | Lookup-table GEMM for **1-4 bit** weights (no dequantise / multiply). | Paper (arXiv 2407.00088): kernel up to 6.6x and 3.6x on average over llama.cpp, 2.8x end to end for 2-bit Llama-2-7B. AVX2 laptop: up to 5.3x for 2-bit mpGEMM at sequence 256. 8-bit is never measured. Optional fast aggregation raises NMSE about 2.5x. | **Not for Q8_0.** Relevant later for Q1_0/Q2_0, IQ2/IQ3 and BitNet-style weights. ONNX Runtime has an experimental T-MAC-style 2-bit path (`mlas.use_lut_gemm`). |
| **XNNPACK** (Google) | BSD-style (check) | QD8 (per-row dynamic int8 activations) x qb4w (blockwise int4) GEMM. | Upstream PRs add scalar qb4w kernels; no AVX2 qb4w kernel found. | **Low** on x86 desktop: its strength is ARM. |

## Tried: tinyBLAS register tiles for Q8_0 (2026-10-30), no gain, reverted

Both AVX2 tile shapes from `tinyBLAS_Q0_AVX` were ported into the Q8_0 batched dispatch. Each was bit-identical to our
dot (same `|w|` x `sign(a, w)`, maddubs + madd, per-block FMA by `dw*da`, one hsum), so output did not change.
Interleaved prefill A/B against the previous build, 3 runs each:

| Tile (rows x tokens) | granite-4.0-h-1b Q8_0, ctl / new t/s | Qwen3-0.6B Q8_0, ctl / new t/s |
|---|---|---|
| 4 x 2 (`gemm4xN<2>`), helper with `ref` accumulators | 145.4/129.0, 146.7/121.7, 148.4/126.7 | 359.0/342.5, 372.6/337.1, 365.9/290.6 |
| 4 x 2, accumulators as plain locals | 146.5/122.4, 145.3/119.7, 143.5/131.2 | 369.7/332.4, 368.3/333.2, 346.5/303.5 |
| 2 x 4 (`gemmMx4<2>`) | 143.2/141.7, 138.1/117.0, 122.5/114.8 | 287.8/309.2, 272.6/278.6, 289.3/286.4 |

The 4 x 2 tile is clearly slower. The 2 x 4 tile is within noise at best (the machine got busier during that run:
the control fell from about 143 to 122 t/s). **Why:** tinyBLAS's win in llama.cpp was over ggml's one-token
`vec_dot`. Our 1 row x 4 token `_4In` dot already reuses each weight block across four tokens. Per output it is bound by
the five-op integer chain (sign, maddubs, madd, convert, FMA), not by loads, which hit L1. A 4 x 2 tile saves loads
but doubles the per-output weight-side work (`abs` and the fp16 scale conversion: 0.5 per output instead of 0.25).
Both kernels were removed; do not retry this without a new idea about the ALU chain. Candidates: VNNI hosts, where
`vpdpbusd` replaces maddubs + madd, or precomputing `|w|` with a repack.

## Recommended order (revised after the tinyBLAS experiment)

1. **Row-interleaved Q8_0 repack** (ik_llama.cpp `_R4`/`_R8` style). A repack can store `|w|` and the sign separately
   and pre-convert the fp16 scales, which removes the weight-side ALU work that made the tinyBLAS tiles lose. It
   follows the pattern of our existing Q4_K x8 repack, which won 2.6x at its shape.
2. **Saturation unit test** for the `vpmaddubsw` sign trick (oneDNN caveat): an activation block that quantises to the
   extremes must not saturate.
3. **VNNI hosts:** a `vpdpbusd` path shortens the chain by one instruction (not testable on this machine).
4. **Later, separately:** T-MAC-style LUT kernels for the 1-2 bit formats, once those families matter.

Every step follows the CLAUDE.md performance rules: interleaved control/candidate runs, named models end to end,
output equality or a recorded numerics delta, and the PerformanceLeague row.

## Not verified here

No other project was built or benchmarked on this machine. The ik_llama.cpp and T-MAC numbers are the authors' own.
The MLAS and XNNPACK points come from commit and PR descriptions. The tinyBLAS tile shapes come from the local source.

## Sources

- [ik_llama.cpp discussion #164: CPU performance comparison with llama.cpp](https://github.com/ikawrakow/ik_llama.cpp/discussions/164)
- [llama.cpp commit "ggml : add llamafile sgemm (#6414)"](https://app.semanticdiff.com/gh/ggml-org/llama.cpp/commit/8cc91dc63c0df397d644a581b2cbeea74eb51ae0)
- [llamafile v0.7 release notes](https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.7)
- [The Register: llamafile performance gains (2024-04-03)](https://www.theregister.com/2024/04/03/llamafile_performance_gains/)
- [ONNX Runtime MLAS commit (CompInt8 AVX2 path)](https://app.semanticdiff.com/gh/microsoft/onnxruntime/commit/a4e58e8feb375f266a4e80c669d7b467a11ea0a3)
- [FBGEMM paper, arXiv 2101.05615](https://arxiv.org/pdf/2101.05615)
- [oneDNN: nuances of int8 computations](https://www.intel.com/content/www/us/en/docs/onednn/developer-guide-reference/2025-1/nuances-of-int8-computations.html)
- [T-MAC paper, arXiv 2407.00088](https://arxiv.org/html/2407.00088v2)
- [Arm learning path: XNNPACK QD8 kernels](https://learn.arm.com/learning-paths/mobile-graphics-and-gaming/integrate-kleidiai-kernel-xnnpack-qd8-f16-qc4w/)
