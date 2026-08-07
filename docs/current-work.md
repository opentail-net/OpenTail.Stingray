# Current work

This is the active engineering backlog. It contains only work that is not yet proven or
product-complete. Historical investigations, experiments, and superseded plans live in [done](done).

## Priority 1 — release-quality and operator hardening

1. **Productize the proven CPU-dense restart lane.** The real SmolLM2 GGUF cross-process proof
   (persist child exit → fresh runtime restore → greedy continuation → token-for-token replay) now
   passes. Expose a minimal named-session lifecycle only for that capability-gated lane.
2. **Hardware release runners.** Attach reproducible CPU AVX2, CUDA dense, Vulkan, hybrid MoE,
   MTP/speculation, and real-session results to the release notes. This machine can supply AVX2
   and AMD Vulkan receipts; CUDA rows require a designated NVIDIA runner.
3. **Capability fixtures.** Static inspect/plan golden fixtures now cover CPU backend selection,
   TurboQuant selection/fallback, KV-dtype applicability, batching/tool-grammar exclusion, MTP,
   and the unsupported restart-session verdict. Extend the set only where the loader actually
   executes the route; GPU and loader-route claims still require their hardware rows below.
4. **Configuration ownership.** Regenerate the drifted CLI/environment inventories from source,
   classify environment variables and host keys, retire stale bench switches, and extend
   source-tracked effective configuration beyond static planning knobs.

References: [release-quality-test-matrix.md](release-quality-test-matrix.md),
[recommended-configurations.md](recommended-configurations.md),
[quality-of-life-improvements-plan.md](quality-of-life-improvements-plan.md), and
[session-native-inference-runtime-plan.md](session-native-inference-runtime-plan.md).

## Priority 2 — correctness and architecture coverage

1. **Vulkan correctness validation.** The subgroup-width inventory, shared-memory replacement,
   hardware-backed F32 Wave64 seam, and real-model token-for-token CLI smoke are complete. The
   newly found per-layer V-cache stride bug showed where the coverage gap actually is, and it is
   worth stating precisely because the obvious reading is wrong. The Vulkan attention shader was
   measured correct at every shape tried, *including* production geometry (32/32/64, 32/8/128,
   32/32/128 — all exact), and the defect was format-independent: it was CPU-side, in
   `PagedKvCache`'s V striding. Adding more shader shapes or more Q4_K/q4_0 kernel coverage would
   not have caught it.

   What was missing: **no test drove a model with per-layer head_dim through the CPU cache path
   end to end.** The coverage to add is that — plus the oracle-free invariant that caught it (at
   position 0, attention output must equal V, broadcast across each GQA group), which needs no
   reference backend and so names the faulty side on its own. Do this before revisiting
   integrated-GPU memory detection or Vulkan performance. Treat Q8_0 only as a cross-format check
   when a supported deployment path needs it.
2. **Gemma 4 validation.** Re-run dense 12B no-PLE validation after the per-layer V-cache stride
   fix, then exercise E4B vision plans on real fixtures.
   E4B text now has a real Q4_0 CPU CLI smoke (including the absent-shared-KV-norm variant),
   but that does not validate its unimplemented `gemma4v`/`gemma4a` encoders.
3. **Qwen3.5 MoE/GDN.** Use the authoritative tensor layout, not the superseded SSM plan.

## Priority 3 — CPU coverage

Do not reopen the closed Q4_K repacked-GEMM investigation. Focus on Flash attention at head
dimensions 128/256, Q6_K AVX2, scalar-fallback formats, per-layer-dimension batched prefill,
CPU MoE, ARM64, additional scalar-fallback format coverage, and Q8-prefill accuracy. Q3_K batched dispatch and the
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

See [cpu-architecture-kernel-opportunities.md](cpu-architecture-kernel-opportunities.md) and
[cpu-prefill-plan.md](cpu-prefill-plan.md).

## Archive rule

Move a document or section to done when its outcome is implemented and verified, or when a
measured negative result closes that line of investigation. Keep active documents short: decision,
remaining work, acceptance evidence, and links to retained history.
