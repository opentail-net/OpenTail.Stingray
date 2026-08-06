# Current work

This is the active engineering backlog. It contains only work that is not yet proven or
product-complete. Historical investigations, experiments, and superseded plans live in [done](done).

## Priority 1 — release-quality and operator hardening

1. **Real restart-continuation proof.** Run a real GGUF through two turns, persistence, process
   exit, a new process/runtime, restore, one greedy continuation, and token-for-token fresh replay.
2. **Hardware release runners.** Attach reproducible CPU AVX2, CUDA dense, Vulkan, hybrid MoE,
   MTP/speculation, and real-session results to the release notes.
3. **Capability fixtures.** Extend inspect/plan only where the loader actually executes the route;
   add golden model/backend/dtype/batching/speculation decision fixtures.
4. **Configuration ownership.** Classify environment variables and host keys; retire stale bench
   switches; extend source-tracked effective configuration beyond static planning knobs.

References: [release-quality-test-matrix.md](release-quality-test-matrix.md),
[recommended-configurations.md](recommended-configurations.md),
[quality-of-life-improvements-plan.md](quality-of-life-improvements-plan.md), and
[session-native-inference-runtime-plan.md](session-native-inference-runtime-plan.md).

## Priority 2 — correctness and architecture coverage

1. **Vulkan Wave64/GCN correctness.** Complete inventory, choose the safe replacement, add
   production-shape relative-error seam tests, then verify a real model.
2. **Gemma 4 validation.** Exercise dense 12B no-PLE, E4B, and vision plans on real fixtures.
3. **Qwen3.5 MoE/GDN.** Use the authoritative tensor layout, not the superseded SSM plan.

## Priority 3 — CPU coverage

Do not reopen the closed Q4_K repacked-GEMM investigation. Focus on Flash attention at head
dimensions 128/256, Q6_K AVX2, scalar-fallback formats, per-layer-dimension batched prefill,
CPU MoE, ARM64, and exact Q3_K/Q2_K multi-input verification.

See [cpu-architecture-kernel-opportunities.md](cpu-architecture-kernel-opportunities.md) and
[cpu-prefill-plan.md](cpu-prefill-plan.md).

## Archive rule

Move a document or section to done when its outcome is implemented and verified, or when a
measured negative result closes that line of investigation. Keep active documents short: decision,
remaining work, acceptance evidence, and links to retained history.
