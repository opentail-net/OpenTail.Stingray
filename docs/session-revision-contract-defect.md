# Session `committed_revision` is not a usable concurrency token

**Found:** 2026-08-07, at the wire, with a reproduction. **Not fixed** — the correct fix changes
persisted-revision semantics and needs an owner decision. Recorded in full so it is not re-derived.

## The defect

Optimistic concurrency admits exactly one client pattern: read `committed_revision`, send it back as
`expected_revision`. Measured against a live (non-restored) session:

```
POST /v1/sessions                       -> committed_revision = 0
POST /v1/sessions/{id}/turns  (rev 0)   -> 200 OK
GET  /v1/sessions/{id}                  -> committed_revision = 6
POST /v1/sessions/{id}/turns  (rev 6)   -> 409  "Expected revision 6, but current revision is 1"
```

**The server advertises 6 and rejects 6, demanding 1.** Any client following the only workable
pattern fails on its second turn.

## Why

Two different quantities are both called "revision":

| | Meaning | Source |
|---|---|---|
| `HotSession.CommittedRevision` | cursor **position** count | `new(Cursor.AcceptedPositionCount)` |
| `InMemorySessionStore` `entry.Revision` | **turn** counter | `entry.Revision.Next()` per completed turn |

`RunTurnAsync` validates `expectedRevision` against the store's turn counter. The HTTP layer
publishes `HotSession.CommittedRevision` — the position count — as `committed_revision`. They agree
only when every turn accepts exactly one position.

## Why the existing restart test passed, and why that matters

`ServerTests.SessionLifecycle_RealCpuGguf_RestoresAcrossServerRestart` reads `committed_revision`
and posts it back, and it **passes**. That is not evidence the contract works: the persisted
revision is *also* written from the position count, so a restored session's store revision is set
to 6 and the round-trip closes. The old behaviour is self-consistent for restored sessions and
broken for live ones.

This is why the defect survived: one test exercised the path where the two wrong values agree.

## Why it is not fixed here

Both single-sided fixes were implemented and measured, and both are wrong:

1. **Fix `HotSession.CommittedRevision` to read the store** (`_store.Open(SessionId).CommittedRevision`).
   Live round-trip works; **two durable tests fail with `NotFound`**, because a cold-restored session
   is not in the hot store.
2. **Fix the endpoint to publish `snapshot.CommittedRevision`.** Live round-trip works, restore
   resolves correctly; **the restart test fails, expected 1 actual 6** — because the persisted
   revision is a position count, so the value jumps across a restart.

The tree is left at the original behaviour, green, rather than shipping either half-fix. Reverting
restores a known bug; shipping either alternative introduces a different one. Choosing between them
is not a judgement to make silently.

## What a correct fix requires

Decide what a persisted revision *means*, then make all three sources agree by construction:

- the store's turn counter,
- the value written to the manifest on eviction/persist,
- the value published as `committed_revision`.

That is an on-disk format decision — existing persisted sessions carry position counts in that
field, so a change needs either a migration or a version bump. `SessionStateCodec` /
`FileSessionManifest` are where the persisted value is written.

**Suggested direction:** make the store's turn counter authoritative everywhere, persist that, and
version the manifest so an old file's position-count revision is recognised rather than silently
trusted. Then `HotSession.CommittedRevision` should either be removed from the public surface or
renamed to say what it is (`AcceptedPositionCount`), since its current name invites exactly the
misuse that produced this.

## Reproduction

Insert into `SessionEndpointTests` (it needs the private `EnabledSessions`/`CreateClient` harness):
create a session, run one turn at `expected_revision = 0`, `GET` the session, then run a second turn
using the advertised value. Before any fix it returns `409`.
