# Session `committed_revision` is not a usable concurrency token

**Found:** 2026-08-07, at the wire, with a reproduction. **FIXED 2026-08-08** — see "Resolution"
at the end. The analysis is kept because it explains why the obvious single-sided fixes are wrong
and why one test passed over the defect for so long.

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


## Resolution (2026-08-08)

Fixed as the combined change this document called for. The store's turn counter is authoritative,
and all FOUR sources now agree by construction:

| Source | Before | After |
|---|---|---|
| Published `committed_revision` (`SessionEndpoints.ToResponse`) | `HotSession.CommittedRevision` (position count) | `snapshot.CommittedRevision` (turn counter) |
| Persisted manifest revision (`ColdSessionRuntime.EvictToDisk`) | `session.CommittedRevision` (position count) | store snapshot's turn counter |
| Restore seeding (`HotSession.RestoreCursor`) | `cursor.AcceptedPositionCount` | the manifest's persisted revision |
| Validation (`RunTurnAsync`) | store turn counter | unchanged — it was always the right one |

The fourth row is the one this document missed. `RestoreCursor` re-seeded the store from the cursor
position count on every import, so fixing only the publish and persist sides left restored sessions
still disagreeing. That is why the two half-fixes measured here each broke a durable test: neither
touched the seeding.

`HotSession.CommittedRevision` is renamed `AcceptedPositionCount` — it is not a concurrency token and
its old name is what invited publishing it as one.

**On-disk format:** manifest version 3. The layout is unchanged; only the meaning of the revision
field is. v1/v2 files are still read and are NOT migrated: the contract requires the sources to
agree, not to hold a particular number, so a legacy position count is simply a larger opaque seed
for the same monotonic counter — every value a client subsequently echoes comes from the store.

**Regression test:** `SessionEndpointTests.Turn_AdvertisedCommittedRevision_IsAcceptedAsExpectedRevision`
runs the read-then-echo pattern against a LIVE session over three turns and asserts a stale revision
still conflicts. Verified by mutation: publishing any other value fails it.
