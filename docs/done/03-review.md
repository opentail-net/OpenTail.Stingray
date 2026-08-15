> **ARCHIVED, 2026-08-15.** Implemented as an `IInferenceSession`/`InferenceSession`-era
> Sessions-layer feature (confirmed against source, not just this document's own claim). That
> whole lineage is being superseded by the `HotSession` architecture — see
> [028](028-inference-session-to-hotsession-migration-plan.md) for current migration status
> (Phases 1-3 done) and [030](../030-delete-inferencesession-todo.md) for the still-open
> deletion of the legacy `InferenceSession`/`InferenceRuntime` types once HotSession fully
> replaces them. Carried forward: nothing beyond what 028/030 already track in
> [00-current-work.md](../00-current-work.md).

---

This is a very good update. The previous review points have been acted on well, and the architecture is now much clearer.

In particular, I like the explicit ownership/mutability matrix, the OpenTail policy vs. Stingray execution boundary, and the documentation of the two execution modes. The session lifecycle, KV page accounting, fork/CoW tests, snapshot terminology and runtime/session registry are all moving in the right direction.

**Please keep this direction. I don't want a redesign.**

I found two important semantic issues worth fixing while the architecture is still being stabilised. They are both quite local.

### 1. One remaining issue: `Fork()` still shares the mutable `IForwardPass`

The documentation now correctly says:

> "Model & Vocab Sharing: Read-only operations against `IModel` / `IForwardPass` are multi-thread safe."

and the session model says session-specific mutable state is isolated.

However, `InferenceSession.Fork()` currently does:

```csharp
var childKv = _kvSequence.Fork();
return new InferenceSession(
    _kvCache,
    SessionId.New(),
    childKv,
    _tokenHistory,
    _forwardPass);
```

So parent and child still receive the **same `IForwardPass` instance**.

That matters because the session actually calls mutating operations on it:

```csharp
_forwardPass.Prefill(...)
_forwardPass.Forward(...)
_forwardPass.TruncateTo(...)
```

This means the logical KV sequence is forked correctly, but the actual forward-pass execution state may still be shared.

Please don't panic about this — the overall architecture is good. This is simply the last piece needed to make the ownership model completely honest.

### Desired result

We want:

```text
                 Shared immutable model
                         │
             ┌───────────┴───────────┐
             │                       │
       Session A context       Session B context
             │                       │
          KV A / CoW              KV B / CoW
```

Shared model weights are excellent.

Shared mutable per-session forward state is not.

Please inspect the existing `IForwardPass` implementation and choose the **smallest change consistent with its actual semantics**.

If the forward pass really contains mutable per-session KV/state, give each forked session its own execution context.

If there is already a natural clone/session-context/factory mechanism available, use that rather than inventing a large abstraction.

If the forward pass can be made genuinely stateless with the session state supplied externally, that is also a good solution — but don't refactor the entire engine just for this.

### Please add one regression test

Use a small stateful test/dummy forward pass which can demonstrate that:

1. parent is prefetched;
2. parent is forked;
3. child generates/prefills;
4. child's forward state changes;
5. parent's forward state remains unchanged;
6. rollback/truncation of the child does not alter the parent's forward state.

This test is more valuable than simply checking that the two sessions have different object references.

The key invariant is:

> **Forking may share immutable model resources and KV pages, but a fork must not share mutable execution state.**

---

### 2. Snapshot restore currently allocates KV pages but doesn't actually re-prefill them

This is a smaller but important implementation/documentation mismatch.

The documentation now says:

> "Physical KV cache pages are reconstructed via re-prefill upon session restore."

That's a good design for the current stage.

However, `RestoreFromSnapshot()` currently does:

```csharp
var newKv = _kvCache.AllocateSequence();
newKv.Append(_tokenHistory.Count);
```

and then:

```csharp
_forwardPass.TruncateTo(snapshot.Tokens.Count);
```

That reconstructs the **logical KV allocation**, but on a fresh session it doesn't appear to reconstruct the actual model KV state.

Compare this with `ResumeAsync()`, which correctly does:

```csharp
_forwardPass.Prefill(_tokenHistory.ToArray());
```

Please make snapshot restoration follow the same semantic path.

The desired behaviour is:

```text
snapshot
  ↓
restore token history
  ↓
allocate fresh logical KV sequence
  ↓
prefill the restored token history
  ↓
session is genuinely inference-ready
```

Please add/use a test with a small fake/stateful `IForwardPass` so that restore actually proves the model execution state was rebuilt, rather than merely checking token counts.

Again, **don't implement physical KV serialization**. The current token snapshot + re-prefill approach is perfectly reasonable. Just make the implementation match the documented contract.

---

### 3. One useful strengthening test

The current fork test is good:

```text
parent → fork → child append → parent remains unchanged
```

Please keep it.

I'd add one complementary test for the opposite direction:

```text
parent
  ↓
fork
  ↓
parent mutates
  ↓
child remains unchanged
```

That proves the CoW invariant symmetrically.

If the current page implementation already guarantees this and the test suite makes it obvious, this can be very small. No need for elaborate infrastructure.

---

### Everything else

I would **leave alone for now**.

In particular:

- the OpenTail/Stingray scheduler boundary is now correct;
- ContinuousBatchingEngine owning execution/batch coordination is appropriate;
- the runtime/session ownership model is much clearer;
- prefix-cache work looks useful;
- the CoW/page accounting is a substantial improvement;
- snapshot vs. actual KV persistence is now correctly distinguished;
- the execution-mode convergence documentation is useful;
- the invariant/golden tests are exactly the right direction.

So this is **not a request for another broad review cycle**.

Please make the two semantic fixes above, add the focused regression tests, run the full test suite, and then report what changed.

The overall direction is good. We're tightening the last ownership/state semantics rather than changing the architecture.