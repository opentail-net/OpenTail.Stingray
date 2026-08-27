# 051 — Wiring HotSession's newly-ported capabilities into the live Server path

## Status (2026-08-27): DONE except LoRA (#6, real new engine work, tracked separately — see its
## section) and `OnTokenGenerated`/`ToolCallParser` (#5, deliberately not wired — the Server layer
## already does this independently and better). Everything else on this document's original list —
## skills/tool validation (#1), Metrics/Metadata/FinishReason/ToolCalls/AllowedChoices (found by a
## follow-up "is this actually reachable over HTTP, not just ported?" audit — see 1b), checkpoint/
## rollback (#2), session tree (#3), and suspend/resume (#4, reclassified from "not scheduled" once
## its real effort turned out to be trivial) — is wired into `/v1/sessions/*`. `Fork()` propagation
## of skills/metadata (an open design question, not a missing wiring step — see #1's scoping note)
## remains the one deliberately deferred item within what was otherwise closed.

## Background

`docs/028`'s Phase 0 audit (see `docs/done/028-inference-session-to-hotsession-migration-plan.md`)
covered a specific table of files and concluded two capabilities were genuinely novel
(`SessionMetadata`/`SessionMetrics`) and one was partial (`GenerationResult` bundling). Those three
are ported and documented in `docs/030-delete-inferencesession-todo.md`.

While re-verifying doc 030's "every genuinely novel capability has been ported" premise before
executing the `InferenceSession` deletion, a closer read of the full `IInferenceSession` interface
surface (not just doc 028's table) turned up seven more members Phase 0 never audited:
`ActiveLora`, `ToolProvider`/`ToolContext`/`ToolCallParser`/`ValidateToolCall`, `AttachSkill`/
`DetachSkill`/`AttachedSkills`, `OnTokenGenerated`, `CreateCheckpoint`/`Rollback`, `Tree`
(`ISessionTree`), and `SuspendAsync`/`ResumeAsync`.

**Investigation finding**: none of these seven are referenced anywhere in `src/OpenTail.Stingray.Server`,
`Server.Host`, or `Cli` — every reference is confined to `InferenceSession.cs`/`IInferenceSession.cs`
themselves and their own dedicated Sessions test files. The live OpenAI/Anthropic-compatible chat
endpoints (`OpenAiEndpoints.cs`, `AnthropicEndpoints.cs`) do tool-calling and streaming their own way,
entirely independent of the Sessions layer — the same "already exists at the Server layer,
architecture-independent" pattern doc 028 already used to classify `InferenceSessionGrammarExtensions.cs`
as redundant. `ActiveLora` is dead code even on `InferenceSession` itself: a bare `{ get; set; }`
auto-property never read by anything (the real, wired `LoraAdapter` property lives on
`ForwardPass`/`IForwardPass`, not `InferenceSession`).

Decision made 2026-08-27: port all seven capabilities onto `HotSession` (matching the rigor of the
two already-resolved items) rather than deleting them as unused, since dead-on-arrival code today
doesn't mean permanently unneeded — and plan how each becomes live. This document is that plan.

## What's ported (done, tested)

All in `src/OpenTail.Stingray.Sessions/HotSession.cs`, covered by
`tests/OpenTail.Stingray.Tests.Sessions.Fast/HotSessionCapabilityPortTests.cs`:

| Capability | HotSession surface | Real behavior today |
|---|---|---|
| Skills / tool validation | `AttachSkill`/`DetachSkill`/`AttachedSkills`/`ToolProvider`/`ToolContext`/`ValidateToolCall` | Fully functional, same two-source (skill tools + `ToolProvider`) check `InferenceSession` did. Nothing calls it yet. |
| `ToolCallParser` | Settable delegate property | Stored only — `HotSession` does not invoke it (chunk-based detection already happens at the Server layer independently; see "Tool calling" below). |
| Per-token notification | `OnTokenGenerated` event | Fires once per generated token **after the turn commits** (not mid-stream — `RunTurnAsync` only has committed tokens, unlike `InferenceSession`'s live per-token loop). Exceptions from listeners are isolated. |
| Checkpoint/rollback | `CreateCheckpoint()` / `RollbackAsync(HotSessionCheckpoint)` | Fully functional — real KV-cache rewind via `RetainedSequenceState.RollbackTo` (new), restores cursor and store revision. Can roll back to *any* earlier checkpoint, not just the last turn (that's what `RunTurnAsync`'s own internal failure-path rollback already did). In-memory only, like everything else `HotSession` has. |
| Suspend/resume | `SuspendAsync()` / `ResumeAsync()` / `IsSuspended` | `SuspendAsync` wraps the pre-existing `EvictRetainedCacheIfIdle` but throws if a turn is active (explicit caller intent, unlike the runtime's own silent-no-op idle reclaim). `ResumeAsync` is a documented no-op — the next `RunTurnAsync` re-prefills from `Cursor` automatically; there's no separate "resumed" state to enter. |
| Session tree / lineage | `HotSession.Tree` (`ISessionTree`), `ParentSessionId`, `HotSessionRuntime.Fork` now records lineage | Fully functional — `RootId`/`ParentId`/`Children` computed on demand from `HotSessionRuntime`'s session table; `CumulativeTreeMetrics` aggregates `ISessionMetrics` across a session and its live fork descendants. |
| LoRA | `HotSession.ActiveLora` (`LoraAdapter?`) | **Inert, by design** — matches what it did on `InferenceSession` (nothing). Real wiring needs new engine work; see below. |

## What "wiring into the live Server path" means, per capability

None of this is required before the `InferenceSession` deletion can proceed — the capabilities are
preserved on `HotSession` either way. This section is the follow-up plan for making each one
*reachable* from a real request, in priority order.

### 1. Skills / tool validation — DONE (2026-08-27)
Wired onto the raw Sessions API, not the OpenAI/Anthropic-compat chat endpoints — those already do
their own per-request tool-call parsing (`req.Tools`, redeclared every call) and duplicating that
with a session-scoped allow-list would just be two competing authorization models on one route.
`/v1/sessions/*` had zero tool-calling concept of any kind, so this is net-new surface, not
reconnecting something already expected to work:

- `POST /v1/sessions/{id}/skills` — attaches a `Skill` (name/description/tool names) via
  `HotSession.AttachSkill`.
- `GET /v1/sessions/{id}/skills` — lists `HotSession.AttachedSkills`.
- `DELETE /v1/sessions/{id}/skills/{name}` — `HotSession.DetachSkill`.
- `POST /v1/sessions/{id}/tool-calls/validate` — `{name, arguments}` → `HotSession.ValidateToolCall`.

**Deliberately scoped down, twice:**
- Only the `Tools` half of `ISkill` does anything here — `Instructions`/`Resources` are accepted
  and stored but never injected into a prompt, because the raw append-prompt turn API has no
  chat-template/prompt-composition step for them to land in. A skill attached this way is a pure
  tool allow-list, not a "prompt+" in the fuller sense `ISkill` supports elsewhere.
- No propagation through `Fork()`: attaching a skill to a parent session does **not** carry it to
  forked children (unlike `Metadata`, which `Fork` already copies). Deliberate choice — the
  inheritance semantics (copy-at-fork vs. live lookup via `Tree.RootId`, override vs. extend) are a
  real design question, not just a wiring detail, and nothing needed it yet. If a fork/consensus
  caller needs shared tool authorization across a tree, that decision has to be made explicitly
  first, not backed into silently.
- `ToolProvider`/`ToolCallParser` remain unwired, as this doc already recommended (see #5 below) —
  the validate endpoint only ever calls `ValidateToolCall` against whatever the caller supplies.

Tests: `SessionEndpointTests.cs` (`Skills_AttachListDetach_RoundTrips`,
`ValidateToolCall_AuthorizesOnlyToolsFromAttachedSkills`, `Skills_UnknownSession_Returns404`, and
the unavailable-lane route list).

**2026-08-27 follow-up — the "skill becomes a prompt" facility.** The scoping note above ("not a
'prompt+' in the fuller sense") turned out to matter: the actual goal is a UI letting someone pick
a skill from a registry (e.g. skills.sh) and have its instructions genuinely shape the model's
output, not just gate tool names. That facility now exists, split by API surface:

- **Chat-compat (`/v1/chat/completions`, `/v1/messages`) — the natural home, done.** Both requests
  gained a `skills: WireSkill[]` field (`SkillWireModels.cs` — `WireSkill`/`WireSkillTool`/
  `WireSkillInstruction`, shared with the Sessions endpoints below). `OpenAiEndpoints.ApplySkills`/
  `AnthropicEndpoints.ApplySkills` run first in each handler: every skill's `Instructions` fold into
  one system-message segment prepended ahead of the caller's own messages (Anthropic: prepended
  into `system`, ahead of whatever the caller already sent there), and every skill's `Tools` merge
  into the request's effective declared tools — reusing the existing `req.Tools`
  parse/authorize/render pipeline entirely unchanged. Zero new prompt-composition machinery; this
  is purely new input feeding the pipeline that already existed. Tests:
  `SkillPromptInjectionTests.cs`.
- **Sessions API (`/v1/sessions/*`) — real gap, now closed, deliberately narrower.**
  `SessionAttachSkillRequest` now also accepts `instructions`. `HotSession.AttachSkill` queues any
  instruction text onto `_pendingInstructionPreamble`; the *next* `RunTurnAsync` call prepends it to
  that turn's append-prompt text (both for KV-budget accounting and for what the engine actually
  generates from), then clears the queue — never repeated on a later turn, and never retroactive
  (a turn already committed to the KV cache cannot be rewritten to have "seen" a skill attached
  afterward; see the confirmed decision in the git history for why late-attach affects only future
  turns rather than being rejected). Test: `HotSessionCapabilityPortTests.HotSession_AttachSkillInstructions_PrependToNextTurnOnly`.

Still not done, deliberately: `Fork()` propagation (unchanged from the original scoping note above)
and `ToolProvider`/`ToolCallParser` wiring (see #5).

### 1b. Closed in the follow-up audit — Metrics/Metadata, FinishReason/ToolCalls, AllowedChoices (2026-08-27)
Prompted by "is everything ported onto HotSession actually wired in?" — the answer was no for three
capabilities that were fully functional on `HotSession` but invisible over `/v1/sessions` HTTP:

- **`ISessionMetrics`/`ISessionMetadata`.** `SessionResponse` (embedded in every session/turn/
  operation response) gained `metrics` (always present — prompt/generated tokens, prefill/
  generation seconds, tokens/sec, KV pages held) and `metadata` (omitted when empty, present when
  the host has called `HotSession.Metadata.Set` — there is no HTTP setter; `ISessionMetadata` is
  host-application state by design, not a client-facing concept, so this is read-only observability
  for an embedding host, not a new client feature). Metadata values are arbitrary
  `object?` (`ISessionMetadata.Set(string, object?)`) with no NativeAOT-safe general
  serialization, so they're stringified (`.ToString()`) for the wire rather than reflection-
  serialized — a caller that needs a structured value back should store a pre-serialized JSON
  string and parse it client-side.
- **`FinishReason`/`ToolCalls`.** `SessionTurnResponse` and `SessionOperationResponse` both gained
  `finish_reason` (`"completed"`/`"max_tokens"`/`"tool_call"`/`"context_limit"`/`"cancelled"`/
  `"failed"`) and `tool_calls` (omitted when empty). `GetOperation`'s replay path had no
  `HotSessionTurnResult` to read this from (it only has the stored `ResultChunks`), so
  `HotSessionTurnResult.DescribeOutcome` — previously `internal`, used only inside `RunTurnAsync`'s
  own replay branch — is now `public` so the endpoint layer can re-derive the same values from
  stored chunks the way the live-turn path already did.
- **`SamplingParams.AllowedChoices`.** `SessionTurnRequest` gained `allowed_choices: string[]?`,
  threaded through `SamplingParamsBuilder.Build`'s new `allowedChoices` parameter. Sessions-only —
  the chat-compat routes have no equivalent field and weren't asked for one; OpenAI/Anthropic's own
  wire protocols have no constrained-choice concept to map onto.

Tests: `SessionEndpointTests.cs` (`Session_ExposesMetricsAndFinishReason`,
`Session_ExposesMetadataSetInProcess`, `Turn_AllowedChoices_ConstrainsGeneration`).

### 2. Checkpoint/rollback — DONE (2026-08-27)
`POST /v1/sessions/{id}/checkpoints` → `HotSession.CreateCheckpoint()`, returning an opaque
`checkpoint_token` (the checkpoint's `SessionCursor`, encoded via the pre-existing
`SessionCursorCodec` — the same versioned binary envelope already used for durable cursor
persistence, so no new serialization format was needed) plus `committed_revision`.
`POST /v1/sessions/{id}/rollback` takes both fields back verbatim and calls
`HotSession.RollbackAsync`. In-memory only, as scoped originally — a token is valid only while the
same `HotSession` instance still holds the retained cache it was taken from; rolling back after
that cache has been evicted, or while a turn is active, is a 409
(`InvalidOperationException`/`NotSupportedException` from `RetainedSequenceState.RollbackTo`), and
a malformed/corrupt token is a 400. Tests: `Checkpoint_ThenRollback_RestoresTheEarlierRevision`,
`Rollback_MalformedToken_Returns400`.

### 3. Session tree / lineage — DONE (2026-08-27)
`GET /v1/sessions/{id}/tree` returns `root_id`/`parent_id`/`children`/`cumulative_metrics` straight
from `HotSession.Tree` (`ISessionTree`) — no new logic, `Tree` already computed all of this on
demand. Test: `Tree_RootSession_ReportsNoParentAndOwnMetrics`.

### 4. Suspend/resume — DONE (2026-08-27), reclassified
Originally scoped as "not scheduled" for lack of a concrete caller — but on reflection the actual
implementation is trivial (`SuspendAsync` is a one-line wrap around the pre-existing
`EvictRetainedCacheIfIdle`; `ResumeAsync` does no work at all), cheaper than #2 or #3, so it went in
too rather than waiting for demand that may never materialize just to save two thin endpoints.
`POST /v1/sessions/{id}/suspend` / `POST /v1/sessions/{id}/resume`; `SessionResponse` gained
`is_suspended`. Suspending mid-turn is a 409. Resume is honestly a no-op wire-for-wire with
`HotSession.ResumeAsync` — the response's `is_suspended` stays `true` until the session's next turn
actually re-prefills, which the test asserts explicitly rather than leaving implicit. Test:
`Suspend_ReleasesTheCache_AndResumeIsANoOpUntilTheNextTurn`.

### 5. `OnTokenGenerated` / `ToolCallParser` — likely NOT worth wiring
Both duplicate something the Server layer already does independently and better: real per-token
streaming already reaches the client via the chunk stream the endpoints consume directly (not
through a session-level event), and tool-call detection already happens per-request in
`OpenAiEndpoints.cs`/`AnthropicEndpoints.cs`. Wiring these would mean building a second, parallel
implementation of behavior that already exists elsewhere — the exact trap `InferenceSessionGrammarExtensions.cs`
fell into. Recommendation: leave both stored-but-unused unless a concrete caller need appears that
the Server-layer implementation can't already satisfy.

### 6. LoRA — real new engine work, not a wiring task
`ContinuousBatchingEngine`'s batched forward pass (`IBatchedForwardPass.BatchForwardMulti`/
`PrefillPackedMulti`) amortizes ONE shared set of model weights across every sequence in a batch in
a single matmul call — there is no per-sequence hook to apply a different LoRA delta per row.
Making `HotSession.ActiveLora` do something real requires new batched-kernel support (e.g. a
LoRA-aware batched matmul variant, or grouping same-adapter sequences into their own sub-batches)
before any session-level wiring is meaningful. This is a distinct, larger engine project — track it
separately if/when there's real demand for per-session LoRA switching; do not scope it into any
future "finish the wiring" pass on this document.

## Relationship to the InferenceSession deletion

This plan does not block `docs/030`'s deletion — the capabilities themselves are ported and
regression-tested regardless of whether anything in `Server` calls them yet. The deletion pass
(doc 030's "How to execute" steps) can proceed independently once its own fresh-grep re-check is
done.
