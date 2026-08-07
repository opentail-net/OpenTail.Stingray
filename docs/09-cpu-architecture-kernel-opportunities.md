# CPU architecture coverage programme

**Status:** active backlog; the Q4_K repacked-GEMM and Flash64 reference-case investigation is closed.

## Ordered work

1. Flash attention for head dimensions 128 then 256, with isolated control and end-to-end parity.
2. Q6_K AVX2 dot path, requiring dispatch evidence and interleaved end-to-end measurement.
3. Native kernels for Q4_0, IQ4_NL, MXFP4, and other scalar-fallback formats.
4. Batched prefill for per-layer head dimensions and CPU MoE.
5. ARM64 NEON, dot-product, and i8mm coverage.
6. Exact Q3_K/Q2_K multi-input verification kernels.

Every item requires dispatch proof, isolated control/candidate samples, named-model end-to-end
measurement, and numerical validation. No single-run result is sufficient.

Historical evidence: [done/cpu-architecture-kernel-opportunities-2026-08.md](done/cpu-architecture-kernel-opportunities-2026-08.md).
