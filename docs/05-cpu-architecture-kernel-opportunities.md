> **Reprioritized 2026-08-08 — now last on the local runway.** Everything here is performance;
> none of it unlocks a model, and the goal now ranks model coverage above speed.
>
> **Item 3 is superseded in part.** Native kernels for IQ4_NL, MXFP4 and other scalar-fallback
> formats are a *follow-up* to §2 of [01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md),
> which first has to make the unimplemented IQ formats dequantize at all. Correctness admits the
> model; kernels only make it faster.

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

## Measurements — Ministral-8B-Instruct-2410 vs. llama.cpp (2026-08-28)

Collected incidentally while running the Ministral-8B greedy-parity receipt (see
[01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) / `ModelCompatibility.cs`'s
`mistral3`/`ministral` entries) — not itself acceptance evidence for any item above, just a data
point for when this list is picked back up. Prompt `"The capital of France is"`, `-n 64`,
`--temp 0 --repeat-penalty 1.0`, Q4_K_M, 36L/4096d/headDim=128, this machine (12-core AVX2, AMD
Radeon integrated GPU). 3 runs each side, CPU-only backend both sides (reference `llama-cli`/
`llama-completion` build has no GPU backend compiled in):

| Backend | Prefill (t/s) | Decode (t/s) |
|---|---|---|
| llama.cpp (CPU-only, build b10532-70aff2525) | 23.3 avg (21.4–25.2) | 9.25 avg |
| Stingray, `--device none` (CPU-only) | 13.1 avg (12.4–13.5) | 7.27 avg |
| Stingray, auto backend (picked CPU this run) | 13.2 avg | 6.67 avg |

Stingray trails by **~1.8x on prefill** and **~1.3x on decode** on this checkpoint/machine. Prefill
is the larger and more likely tractable gap (batched-GEMM kernel quality — relevant to item 4,
batched prefill), decode is memory-bandwidth-bound single-token generation and likely needs
sustained kernel work to close meaningfully.

Also observed, unexplained: Stingray's auto-selected GPU-hybrid path was not faster than its own
CPU-only path for this small model/prompt (an earlier uncontrolled run showed hybrid prefill at
5.7 t/s, *slower* than CPU-only) — plausibly PCIe upload/readback overhead dominating a short
8-token prompt on the integrated GPU, not investigated further. Worth a look whenever GPU-hybrid
dispatch heuristics are next touched.
