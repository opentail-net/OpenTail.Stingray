# IQ1_S / IQ1_M / IQ2_XS / IQ2_S coverage verification

**Status: closed.** No implementation gap was found. The remaining evidence limitation is the lack of a tractable real-weight IQ1_S/IQ1_M end-to-end receipt.

## Audit result

The four formats have format-specific decoders and `Dequantize.ToFloat32` dispatch. Their block layouts and sizes match the vendored ggml definitions:

| Format | Elements/block | Bytes/block | CPU matvec route |
| --- | ---: | ---: | --- |
| IQ1_S | 256 | 50 | `MatVecDequantFallback` |
| IQ1_M | 256 | 56 | `MatVecDequantFallback` |
| IQ2_XS | 256 | 74 | IQ2_XS scalar/AVX2 matvec kernels |
| IQ2_S | 256 | 82 | IQ2_S scalar/AVX2 matvec kernels |

All four are admitted by `ModelCompatibility.IsSupportedWeightDType`. IQ1_S and IQ1_M's Q8_K-paired dot routines are retained as correctness oracles, not wired into `MatVec`; the generic dequantized route is their supported execution path.

## Reference comparison

Compared complete literals in `IqCodebooks.cs` with `examples/ggml/src/ggml-common.h`: `iq1s_grid` (2048 entries), `iq2xs_grid` (512), `iq2s_grid` (1024), `ksigns_iq2xs` (128), and `kmask_iq2xs` (8) all match exactly, entry for entry. `Iq1sDelta` is `0.125f`, matching `IQ1S_DELTA` and `IQ1M_DELTA` in ggml.

Audited the four decoders against `examples/ggml/src/ggml-quants.c` and the block declarations in `ggml-common.h`. IQ1_S uses the global delta sign and 11-bit grid index; IQ1_M uses its embedded shared FP16 scale, two sub-scales per 32-element group, and four independent delta bits; IQ2_XS uses its 9-bit grid index and 7-bit sign lookup; IQ2_S uses its separate `qh` high index bits and direct sign bytes. Output ordering and block boundaries agree with the reference.

The vendored checkout is ggml v0.20.2, commit `8c63e70982c95ceb862e3a1073a2c1beef75d60a` (2026-08-18). The existing Qwen receipt cites a separately built llama.cpp reference `b10532-70aff2525`; these revisions are not assumed to be identical.

## Automated and real-weight evidence

Focused validation passed:

- 14 IQ correctness tests passed, including IQ1_S/IQ1_M independent dequantize-versus-Q8_K-dot checks, IQ2_XS/IQ2_S dequantized-dot checks, AVX2/scalar parity, codebook sanity, and IQ1_M fallback coverage.
- 42 `WeightDTypeProfile_OnlyAdvertisesExecutableFormats` cases passed, including all four supported dtype assertions.

The existing [Qwen3.8-27B UD-Q3_K_XL receipt](01-gguf-model-coverage-plan.md) remains the real-weight evidence for IQ2_XS and IQ2_S: the mixed-format checkpoint produced an exact 24-of-24-token greedy match with llama.cpp. It includes IQ2_XS and IQ2_S tensors alongside IQ3_XXS and IQ4_XS.

IQ1_S and IQ1_M have independent formula cross-checks and admission/fallback coverage, but no real-weight greedy-parity receipt. The known DeepSeek-V3.2 IQ1_M checkpoint is approximately 149 GB; downloading it solely to create a receipt is not warranted. A smaller suitable checkpoint may strengthen evidence later without reopening this implementation item.

## Existing implementation records

- [GGUF model coverage plan](01-gguf-model-coverage-plan.md) — implementation history and Qwen3.8 receipt.
- [CPU architecture/kernel opportunities](05-cpu-architecture-kernel-opportunities.md) — IQ1_S/IQ1_M implementation and dispatch decision.
- [Verification plan](../1-correctness/16-iq-formats-coverage-verification-plan.md) — audit scope and steps.
