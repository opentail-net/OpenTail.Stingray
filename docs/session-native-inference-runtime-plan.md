# Session-native inference runtime — current work

**Status:** internal CPU-dense retained state and persistence seams exist; restart-safe sessions are
not yet a supported CLI/server feature.

## Immediate work

1. Add a real-GGUF, cross-process restart-continuation acceptance test.
2. Expose a minimal named-session lifecycle only after that proof exists.
3. Add a backend/cache conformance matrix for hot reuse, rollback, persistence, and restart.
4. Exercise interrupted writes, corrupt packs, ABI mismatch, quotas, and eviction ownership.
5. Validate multi-model routing after the single-model exact lane is proven.

No fake forward pass, in-process restore, cache-byte comparison, or cursor-only replay is a
restart-continuation proof. See [release-quality-test-matrix.md](release-quality-test-matrix.md).

Historical record: [done/session-native-inference-runtime-plan-rev1.md](done/session-native-inference-runtime-plan-rev1.md),
[adr-0001-session-cache-lifecycle.md](adr-0001-session-cache-lifecycle.md), and
[milestone-0-state-topology-audit.md](milestone-0-state-topology-audit.md).
