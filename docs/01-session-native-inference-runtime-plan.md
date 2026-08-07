# Session-native inference runtime — current work

**Status:** the narrow CPU-dense restart-continuation lane is proven on a real GGUF across two
processes and is available from the server when `EnableSessions` and `SessionStorageDirectory`
are configured. It remains CPU-dense GGUF only; the operation-result ledger is hot-only and the
CLI has no named-session surface yet.

The server path itself is now acceptance-tested: a real CPU GGUF server creates a session through
HTTP, completes and persists a turn, shuts down, then a fresh server process restores that session
through `GET /v1/sessions/{id}` and completes its next turn. This is covered by
`SessionLifecycle_RealCpuGguf_RestoresAcrossServerRestart`.

## Immediate work

1. Expose a minimal named-session lifecycle for the proven CPU-dense lane, with explicit
   capability refusal outside it. **Loader/DI seam complete:** `EnableSessions` now constructs
   and publishes `IServerSessionRuntime` only for CPU-dense GGUF batching. `POST`, `GET`, and
   `DELETE /v1/sessions` plus append-only `POST /v1/sessions/{id}/turns` now provide the hot
   named-session shell, including optimistic revisions and idempotency IDs. Hot clients can
   reconnect after losing a turn response through `GET /v1/sessions/{id}/operations/{operationId}`
   and recover its retained result without rerunning it. With `SessionStorageDirectory`, completed
   turns persist and restore on demand. `/capabilities` explicitly reports that operation-result/
   idempotency persistence is unavailable: records and the lookup result are memory-only, so
   restart continuation must not be presented as durable retry replay. A bounded durable
   operation-result pack/journal is the next slice.
2. Add a backend/cache conformance matrix for hot reuse, rollback, persistence, and restart.
   Its persisted ABI must represent per-layer KV/head dimensions and V-region stride; a single
   model-level `headDim` silently corrupts Gemma-class mixed-dimension caches.
3. Exercise interrupted writes, corrupt packs, ABI mismatch, quotas, and eviction ownership.
   - Corrupt KV-pack restore is now covered at the cold-runtime boundary: both `Open` and
     `OpenOrCreate` refuse a damaged persisted pack rather than admitting partial state.
   - Re-persistence now writes a new generation of packs before atomically publishing its
     manifest. A crash before publication leaves the prior manifest and every referenced pack
     intact; stale generations are reclaimed only afterward.
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
