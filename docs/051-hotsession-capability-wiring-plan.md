# 051 — Wiring HotSession's newly-ported capabilities into the live Server path

## Status: skills / tool validation (item #1) wired, 2026-08-27. Checkpoint/rollback, session-tree
## endpoint, suspend/resume, LoRA remain unwired — see their sections below.

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

### 2. Checkpoint/rollback — medium effort, real value for the Sessions API
`SessionEndpoints.cs` has no checkpoint/rollback endpoints today. A natural fit:
`POST /sessions/{id}/checkpoints` → `HotSession.CreateCheckpoint()`,
`POST /sessions/{id}/rollback` → `HotSession.RollbackAsync`. Needs: a way to serialize
`HotSessionCheckpoint` across a request/response boundary (it's in-memory-only today, valid only
for the lifetime of the same `HotSession` instance — fine for same-process same-session use, not
for durable/cross-process checkpoint references without further work).

### 3. Session tree / lineage — low effort, observability value
`SessionEndpoints.cs` could expose `GET /sessions/{id}/tree` returning `RootId`/`ParentId`/
`Children`/`CumulativeTreeMetrics` — useful for anyone using `HotSessionRuntime.Fork` (e.g. the
consensus/voting pattern `ForkAndVoteTests.cs` exercised on the old architecture) to inspect a fork
tree's aggregate cost. No blocking dependency; smallest of the four real items.

### 4. Suspend/resume — low effort, mostly already covered
`HotSessionRuntime`'s own idle-pressure reclaim (docs/028 Phase 1) already does this automatically
under memory pressure. Explicit `SuspendAsync`/`ResumeAsync` would only matter for a caller that
wants to force-free a session's cache proactively (e.g. a client going idle for a known-long period)
— worth exposing as `POST /sessions/{id}/suspend` only if a real caller asks for it. Not scheduled.

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
