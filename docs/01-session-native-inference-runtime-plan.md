# Session-native inference runtime — current work

**Status:** the narrow CPU-dense restart-continuation lane is proven on a real GGUF across two
processes; restart-safe sessions are not yet a supported CLI/server feature.

## Immediate work

1. Expose a minimal named-session lifecycle for the proven CPU-dense lane, with explicit
   capability refusal outside it. **Loader/DI seam complete:** `EnableSessions` now constructs
   and publishes `IServerSessionRuntime` only for CPU-dense GGUF batching. `POST`, `GET`, and
   `DELETE /v1/sessions` plus append-only `POST /v1/sessions/{id}/turns` now provide the hot
   named-session shell, including optimistic revisions and idempotency IDs. Durable restart
   lifecycle remains the next slice.
2. Add a backend/cache conformance matrix for hot reuse, rollback, persistence, and restart.
   Its persisted ABI must represent per-layer KV/head dimensions and V-region stride; a single
   model-level `headDim` silently corrupts Gemma-class mixed-dimension caches.
3. Exercise interrupted writes, corrupt packs, ABI mismatch, quotas, and eviction ownership.
4. Validate multi-model routing after the single-model exact lane is proven.

The proof is `HotSessionGreedyReplayTests.ColdSession_RealModel_CrossProcessRestore_MatchesFullGreedyReplay`:
one child process runs and persists a real SmolLM2 GGUF turn, then exits; a second process creates
a fresh runtime, restores the manifest/KV packs, runs a second greedy turn, and compares every
generated segment against fresh full replay. It passed in Release on 2026-08-07. No fake forward
pass, in-process restore, cache-byte comparison, or cursor-only replay would meet that bar. See
[release-quality-test-matrix.md](release-quality-test-matrix.md).

Historical record: [done/session-native-inference-runtime-plan-rev1.md](done/session-native-inference-runtime-plan-rev1.md),
[adr-0001-session-cache-lifecycle.md](adr-0001-session-cache-lifecycle.md), and
[done/milestone-0-state-topology-audit.md](done/milestone-0-state-topology-audit.md).
