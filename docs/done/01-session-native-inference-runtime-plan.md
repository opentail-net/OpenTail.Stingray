> **ARCHIVED 2026-08-08.** The CPU-dense restart-continuation lane it scoped is implemented,
> capability-gated, and proven across two processes on a real SmolLM2 GGUF. Its four numbered
> items are all closed. **Carried forward:** sessions still have no CLI named-session surface,
> and the lane remains CPU-dense GGUF only — both are parked, not scheduled. See
> [../00-current-work.md](../00-current-work.md).

# Session-native inference runtime — current work

**Status:** the narrow CPU-dense restart-continuation lane is proven on a real GGUF across two
processes and is available from the server when `EnableSessions` and `SessionStorageDirectory`
are configured. It remains CPU-dense GGUF only; bounded completed operation results and
idempotency records now restore with the session, while the CLI has no named-session surface yet.

The server path itself is now acceptance-tested: a real CPU GGUF server creates a session through
HTTP, completes and persists a turn, shuts down, then a fresh server process restores that session
through `GET /v1/sessions/{id}` and completes its next turn. This is covered by
`SessionLifecycle_RealCpuGguf_RestoresAcrossServerRestart`.
That real-model cross-process proof was rerun successfully on 2026-08-07 after durable operation
replay was added.

## Immediate work

1. Expose a minimal named-session lifecycle for the proven CPU-dense lane, with explicit
   capability refusal outside it. **Loader/DI seam complete:** `EnableSessions` now constructs
   and publishes `IServerSessionRuntime` only for CPU-dense GGUF batching. `POST`, `GET`, and
   `DELETE /v1/sessions` plus append-only `POST /v1/sessions/{id}/turns` now provide the hot
   named-session shell, including optimistic revisions and idempotency IDs. Clients reconnect
   after losing a turn response through `GET /v1/sessions/{id}/operations/{operationId}` and
   recover its retained result without rerunning it. With `SessionStorageDirectory`, completed
   turns, their bounded completed-operation ledger, and KV state persist and restore on demand.
   A fresh server is acceptance-tested to look up and idempotently replay a pre-restart operation.
   `/capabilities` publishes this as available only for that CPU-dense persisted lane. The ledger
   is intentionally bounded by both record retention and a 1 MiB pack ceiling; old or oversized
   results can be pruned and must not be treated as an archival transcript.
2. **Done — cache conformance matrix.** Hot reuse, rollback, persistence/restart, corrupt packs,
   quotas/eviction, and per-model resource partitioning are covered in
   `../sessions-release-gate-matrix.md`. Its persisted ABI represents per-layer KV/head dimensions
   and V-region stride; a single model-level `headDim` would silently corrupt Gemma-class mixed-
   dimension caches.
3. **Done for the CPU-dense lane — interrupted writes, corrupt packs, ABI mismatch, quotas, and
   eviction ownership.**
   - Corrupt KV-pack and completed-operation-ledger restore are covered at the cold-runtime
     boundary: both `Open` and `OpenOrCreate` refuse a damaged persisted pack rather than
     admitting partial state or silently forgetting a replayable completed turn.
   - Re-persistence now writes a new generation of packs before atomically publishing its
     manifest. A crash before publication leaves the prior manifest and every referenced pack
     intact; stale generations are reclaimed only afterward.
4. **Done — multi-model resource routing.** The per-model budget suite prevents one model's
   resident/in-flight allocation from being charged to or consuming another model's partition.
   Cross-backend session support remains intentionally refused rather than inferred from this lane.

The proof is `HotSessionGreedyReplayTests.ColdSession_RealModel_CrossProcessRestore_MatchesFullGreedyReplay`:
one child process runs and persists two real SmolLM2 GGUF turns, then exits; a second process creates
a fresh runtime, restores the manifest/KV packs, runs a third greedy turn, and compares every
generated segment against fresh full replay. No fake forward
pass, in-process restore, cache-byte comparison, or cursor-only replay would meet that bar. See
[release-quality-test-matrix.md](../release-quality-test-matrix.md).

Historical record: [session-native-inference-runtime-plan-rev1.md](session-native-inference-runtime-plan-rev1.md),
[adr-0001-session-cache-lifecycle.md](../adr-0001-session-cache-lifecycle.md), and
[milestone-0-state-topology-audit.md](milestone-0-state-topology-audit.md).
