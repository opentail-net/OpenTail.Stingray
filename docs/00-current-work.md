# Current work

This is the active engineering backlog. It contains only work that is not yet proven or
product-complete. Historical investigations, experiments, and superseded plans live in [done](done).

## Ordered runway

Work the numbered files in this order unless a concrete defect changes the priority:

1. [01 — session-native inference runtime](01-session-native-inference-runtime-plan.md)
2. [02 — configuration and operator quality](02-quality-of-life-improvements-plan.md)
3. [03 — CPU prefill release hardening](03-cpu-prefill-plan.md)
4. [04 — Gemma 4 12B text validation](04-gemma4-12b-implementation-plan.md)
5. [05 — Gemma 4 E4B vision](05-gemma4-e4b-vision-plan.md)
6. [06 — Qwen3.5 MoE/GDN validation](06-qwen35moe-plan.md)
7. [07 — DSpark remaining integration](07-dspark-plan.md)
8. [08 — SafeTensors next formats/backends](08-safetensors-support-plan.md)
9. [09 — CPU architecture coverage](09-cpu-architecture-kernel-opportunities.md)

Hardware that cannot be validated on this PC is intentionally separated in
[90 — external hardware work](90-external-hardware-work.md). It is not part of the local runway.

## Priority 1 — release-quality and operator hardening

1. **CPU-dense session core is productized; close its release gate.** The capability-gated named
   lifecycle, durable KV/cursor state, bounded completed-operation replay, and real SmolLM2 GGUF
   cross-process proof (persist child exit → fresh runtime restore → greedy continuation →
   token-for-token replay) now pass. Remaining work is release packaging/CPU-runner evidence and
   the broader cache-conformance matrix, not another session API spike.
2. **Hardware release runners.** Attach reproducible CPU AVX2, CUDA dense, Vulkan, hybrid MoE,
   MTP/speculation, and real-session results to the release notes. Local CPU/Vulkan work stays
   here; NVIDIA/CUDA and ARM64-only receipts are tracked in
   [90-external-hardware-work.md](90-external-hardware-work.md).
3. **Capability fixtures.** Static inspect/plan golden fixtures now cover CPU backend selection,
   TurboQuant selection/fallback, KV-dtype applicability, batching/tool-grammar exclusion, MTP,
   and the unsupported restart-session verdict. Extend the set only where the loader actually
   executes the route; GPU and loader-route claims still require their hardware rows below.
4. **Configuration ownership.** Regenerate the drifted CLI/environment inventories from source,
   classify environment variables and host keys, retire stale bench switches, and extend
   source-tracked effective configuration beyond static planning knobs.

References: [release-quality-test-matrix.md](release-quality-test-matrix.md),
[recommended-configurations.md](recommended-configurations.md),
[02-quality-of-life-improvements-plan.md](02-quality-of-life-improvements-plan.md), and
[01-session-native-inference-runtime-plan.md](01-session-native-inference-runtime-plan.md).

## Priority 2 — correctness and architecture coverage

1. **Vulkan correctness validation.** The subgroup-width inventory, shared-memory replacement,
   hardware-backed F32 Wave64 seam, and real-model token-for-token CLI smoke are complete. The
   newly found per-layer V-cache stride bug showed where the coverage gap actually is, and it is
   worth stating precisely because the obvious reading is wrong. The Vulkan attention shader was
   measured correct at every shape tried, *including* production geometry (32/32/64, 32/8/128,
   32/32/128 — all exact), and the defect was format-independent: it was CPU-side, in
   `PagedKvCache`'s V striding. Adding more shader shapes or more Q4_K/q4_0 kernel coverage would
   not have caught it.

   The directly actionable cache-geometry gap is now closed: `PagedKvCacheTests` drives the real
   Gemma 12B geometry (8 KV heads, 256-wide SWA and 512-wide global heads) across a page boundary,
   and `PrefillAttentionSeamTests` pins the oracle-free position-zero invariant — attention output
   equals V, broadcast across every GQA/MQA group. The available real E4B q4_0 fixture also passes
   the CPU prefill/decode smoke on this corrected path. The dense 12B QAT q4_0 model is now
   acquired, and its real CPU coherent-prefill/decode guard passes on the corrected cache layout.
   The remaining work is its long-position/reference and CUDA/hybrid acceptance sequence, not another synthetic Vulkan shader
   shape. Do this before revisiting integrated-GPU memory detection or Vulkan performance. Treat
   Q8_0 only as a cross-format check when a supported deployment path needs it.
2. **Gemma 4 validation.** Dense 12B CPU coherence now also crosses the real 1,024-token SWA
   boundary. Finish CPU↔reference parity, then run CUDA/hybrid parity externally and exercise E4B
   vision plans on real fixtures.
   E4B text now has a real Q4_0 CPU CLI smoke (including the absent-shared-KV-norm variant),
   but that does not validate its unimplemented `gemma4v`/`gemma4a` encoders.
3. **Qwen3.5 MoE/GDN.** Use the authoritative tensor layout, not the superseded SSM plan.

## Priority 3 — CPU coverage

Do not reopen the closed Q4_K repacked-GEMM investigation. Flash attention now has a correctness-
gated 128/256-wide route: Qwen3-8B Q4_K_M (headDim 128) matches the materialised fallback, and
the two headDim-256 GEMM shapes pass an independent oracle. Its remaining work is controlled
performance measurement, including a dense hd256 model when one is available. Q6_K AVX2 dispatch
is also complete (Q8_K activation plus 8/4/1-input dots); it is performance-only, not a missing
correctness route. Focus next on scalar-fallback formats, per-layer-dimension batched prefill,
CPU MoE, additional scalar-fallback format coverage, and Q8-prefill accuracy. ARM64 work is in
[90-external-hardware-work.md](90-external-hardware-work.md). Q3_K batched dispatch and the
Q2_K >512-token production fallback are already covered by the focused Q8-prefill equivalence
suite; neither is an open correctness gap.

**Q6_K baseline (2026-08-07).** The checksum-guarded `kernel-bench-cs` harness at
`k=8192`, `rows=512`, `reps=12`, with `DOTNET_TC_QuickJitForLoops=0`, produced independent
best times of 0.1676, 0.1760, and 0.1845 ms (checksum `2363.599609`). This replaces the stale
0.2063-ms historical baseline but is not a new performance claim: any candidate must be
interleaved against this implementation in the same process and beat the observed run-to-run range.
The fused 2/4/8-input Q6_K route is already protected by
`MatMulBatchedQ8EquivalenceTests` (23/23 focused cases passed on 2026-08-07); the next Q6_K
slice is therefore performance-only, not missing correctness coverage.

See [09-cpu-architecture-kernel-opportunities.md](09-cpu-architecture-kernel-opportunities.md) and
[03-cpu-prefill-plan.md](03-cpu-prefill-plan.md).

## Archive rule

Move a document or section to done when its outcome is implemented and verified, or when a
measured negative result closes that line of investigation. Keep active documents short: decision,
remaining work, acceptance evidence, and links to retained history.
