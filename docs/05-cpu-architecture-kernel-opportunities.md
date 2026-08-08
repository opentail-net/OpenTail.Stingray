# CPU architecture coverage programme

**Status:** active backlog; the Q4_K repacked-GEMM investigation, Flash64 reference case, and
the missing Flash 128/256 correctness route are closed. Performance acceptance for the new head
widths remains open.

## Ordered work

1. Measure Flash attention at 128/256 head widths against the materialised fallback, with
   interleaved isolated and real-model samples. The generic 64-query/KV tile now dispatches for
   dense 64/128/256 heads; the 64-wide special case remains on the hardcoded GEMM and 128/256 use
   the strided AVX2 microkernel. Qwen3-8B (headDim 128) Flash-vs-fallback parity passes; the 256
   GEMM shapes have an independent oracle, but no local dense hd256 model receipt exists yet.
2. Q6_K AVX2 performance-only investigation. Dispatch is complete: Q8-prefill resolves Q6_K to
   Q8_K activation plus 8/4/1-input dots, and F32 multi-input batching uses the 4/2-input paths.
   Focused equivalence coverage passes; any change needs an interleaved end-to-end win.
3. Native kernels for IQ4_NL, MXFP4, and other genuinely scalar-fallback formats. Q4_0 already
   has a fused CPU route, so it is not an implementation gap.
4. Batched prefill for per-layer head dimensions and CPU MoE.
5. ARM64 NEON, dot-product, and i8mm coverage (external hardware required).

Every performance item requires dispatch proof, isolated control/candidate samples, named-model
end-to-end measurement, and numerical validation. No single-run result is sufficient.

Historical evidence: [done/cpu-architecture-kernel-opportunities-2026-08.md](done/cpu-architecture-kernel-opportunities-2026-08.md).
