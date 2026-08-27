# 051 — HotSession capability wiring: remaining TODOs

## Status: most of this plan is done — see
## `docs/done/051-hotsession-capability-wiring-plan.md` for the full history (skills/tool
## validation, skill-instructions-as-prompt, Metrics/Metadata/FinishReason/ToolCalls/AllowedChoices,
## checkpoint/rollback, session tree, suspend/resume). What's left is genuinely open: two items
## deliberately not wired, and one design question the "done" work surfaced but didn't answer.

## 1. `Fork()` doesn't propagate skills or pending instructions

`HotSessionRuntime.Fork` copies `Metadata` entries from parent to each branch
(`HotSession.cs` — `foreach (var kv in parent.Metadata.GetEntries()) branch.Metadata.Set(...)`),
but does **not** copy `AttachedSkills`/`ToolProvider`/`ToolContext`/the pending instruction
preamble. Attach a skill to a parent session, then fork it (e.g. a consensus/voting pattern
spawning N branches), and every branch starts with an empty tool allow-list and no queued
instructions — the attachment silently doesn't reach the children.

This is a real design question, not a small fix, because there are two legitimate models and
picking wrong would need to be walked back later:

- **Copy-at-fork** (mirrors `Metadata`'s existing behavior): each branch gets an independent copy
  of the parent's attached skills at the moment of forking. Simple, consistent with `Metadata`,
  but a skill attached to the parent *after* forking doesn't reach already-existing children.
- **Live lookup via `Tree.RootId`**: `ValidateToolCall`/instruction injection consults the root
  session's current skill set on every call instead of a local copy. Always current, but changes
  `HotSession`'s per-instance independence assumption (a branch's authorization would depend on a
  *different* session object's mutable state) and needs a decision on override-vs-extend if a
  child also attaches its own skills.

**Do not implement either without a concrete caller.** Nothing today needs shared tool
authorization across a fork tree; deciding this speculatively risks the wrong shape. Revisit when
a real fork/consensus use case asks for it.

## 2. `OnTokenGenerated` / `ToolCallParser` — likely NOT worth wiring

Both duplicate something the Server layer already does independently and better: real per-token
streaming already reaches the client via the chunk stream the endpoints consume directly (not
through a session-level event), and tool-call detection already happens per-request in
`OpenAiEndpoints.cs`/`AnthropicEndpoints.cs`. Wiring these would mean building a second, parallel
implementation of behavior that already exists elsewhere — the exact trap
`InferenceSessionGrammarExtensions.cs` fell into (see `docs/done/028-inference-session-to-hotsession-migration-plan.md`).

**Recommendation: leave both stored-but-unused unless a concrete caller need appears that the
Server-layer implementation can't already satisfy.** Not scheduled.

## 3. LoRA — real new engine work, not a wiring task

`ContinuousBatchingEngine`'s batched forward pass (`IBatchedForwardPass.BatchForwardMulti`/
`PrefillPackedMulti`) amortizes ONE shared set of model weights across every sequence in a batch in
a single matmul call — there is no per-sequence hook to apply a different LoRA delta per row.
Making `HotSession.ActiveLora` do something real requires new batched-kernel support (e.g. a
LoRA-aware batched matmul variant, or grouping same-adapter sequences into their own sub-batches)
before any session-level wiring is meaningful.

**This is a distinct, larger engine project — track it separately if/when there's real demand for
per-session LoRA switching.** Not scheduled.
