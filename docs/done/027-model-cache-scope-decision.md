> **ARCHIVED, 2026-08-15.** Implemented as designed (confirmed against source). No open
> remainder tracked separately from [00-current-work.md](../00-current-work.md).

---

# Why the Model-Serving Design Was Descoped — Reasoning for Review

## Purpose of this document

This is not a design document. It's a record of a scope decision, written to be handed to an
independent reviewer alongside `024-multi-model-serving-and-request-scheduling-plan.md`,
`025-shared-model-cache-phase1-plan.md`, and `026-shared-model-cache-phase2-eviction-plan.md`.
The question worth an outside opinion on isn't "is the multi-model design good" (024 already
got that review and was revised in response) — it's **"was narrowing implementation scope down
to just 025/026, and shelving the rest of 024, the right call?"**

## Sequence of events

1. Investigation into `Tests.ForwardPass` climbing to 59.5 GB memory on a 63 GB machine.
   Several hypotheses tested and ruled out (GC scheduling, mmap refcount leak, missing
   `Dispose()`, background-process contention) before landing on the actual cause: ~40 test
   files load ~15 different multi-gigabyte models with no coordination or reuse across files.
2. That observation was reframed, correctly, as evidence of a real gap in Stingray itself —
   the engine has no concept of more than one model, so there's no way to manage residency or
   schedule work across models, in production or in tests.
3. `024` was written: a full design for multi-model serving — residency management, async
   single-flight loading, a resource-budget abstraction (host + accelerator memory), a request
   scheduler with service-quantum and SLA-preemption logic, session/KV affinity, and a new
   `IInferenceService` API layer above the existing single-model `IInferenceEngine`.
4. `024` was sent for independent review. The review was substantively positive on direction
   (8/10 architecture) and caught two real correctness gaps in the first draft (eviction could
   destroy a model with live in-flight/session state; synchronous loading would block the
   scheduler on 20-50 GB loads with no request coalescing) plus several sound refinements
   (service quantum, `ModelSchedulingInfo` as a stable seam, promoting GPU/VRAM handling from
   an open question to a real design constraint). `024` was revised to incorporate all of it.
5. Before implementing anything, the question in this document's title was asked directly: is
   this over-engineered relative to the problem that actually motivated it?

## The over-engineering concern, stated plainly

Two different things got fused together across steps 2–4 above, and they have very different
justifications. More precisely, the investigation actually surfaced three distinct problems,
not one, and it matters which of them is actually demonstrated versus merely imaginable:

- **Problem A — the same model reopened repeatedly.** Ten files, one model, ten independent
  load/dispose cycles. Demonstrated, measured. `025` fixes exactly this.
- **Problem B — too many distinct models resident at once, unbounded.** Fifteen models, no
  eviction, so a long enough run accumulates all of them. Demonstrated, measured (this is the
  other half of the 59.5 GB number). `026` fixes exactly this.
- **Problem C — concurrent production requests actually competing for several different
  models at once**, needing scheduling to decide which one runs next. This is `024` territory.
  **There is no evidence problem C is a real, current requirement** — every existing
  deployment path is single-model by design, and nothing in this investigation demonstrated
  otherwise. That absence of evidence is precisely why shelving `024` is the right call, not a
  weaker version of "we might need it eventually."

For the two problems actually demonstrated (A and B), a reference-counted, capacity-bounded
shared cache is sufficient — worth stating that way rather than as "the test suite's problem
is just a shared cache," since the point isn't that the problem was small, it's that the
*demonstrated* portion of it doesn't require more than this. That's it: no scheduler, no
session awareness, no resource-budget taxonomy, no new API layer.

**The production feature (`024`, addressing problem C)** — full multi-model serving with
intelligent request scheduling. This is legitimate and the reviewed design for it is sound.
But it exists because of a reasoning chain (test bug → "there's no multi-model concept" →
"let's design one properly"), not because there is a current caller, deployment, or stated
requirement asking Stingray to serve more than one model at a time. Every existing entry
point — CLI, Server, `docs/`,
`OpenTailStingrayServerOptions` — is single-model by explicit design today, and nothing in
this investigation surfaced evidence that's about to change.

Building the full thing now means: `IResourceBudget`/`IResourceAdmissionController` with a
four-value `AdmissionResult` enum and a still-null `AcceleratorMemory` budget, for zero current
callers. A service-quantum-and-SLA-preemption scheduler, for zero current multi-model traffic
to schedule. Session-pin-on-eviction semantics, for a scenario (a session's model going cold
while other models are also resident) that can't occur today because only one model can ever
be resident. All well-designed, all justified *if* multi-model serving is actually needed —
none of it load-bearing for the bug that started this.

This is the textbook shape of speculative generality: correct-looking abstractions built ahead
of a real requirement, sized for a problem the codebase doesn't have yet. It's also explicitly
against this project's own stated engineering norms (from the house style this codebase is
written to): don't add abstractions beyond what the task requires, don't design for
hypothetical future requirements, don't build for a need that isn't there yet. `024`'s own
7-phase rollout makes the mismatch concrete: the actual, measured, already-reproduced bug
doesn't get fixed until Phase 7, gated behind building session integration and a resource
admission subsystem first.

## Decision

Decouple. Implement only `025` and `026` — a process-wide, reference-counted, capacity-bounded
`SharedModelCache` wrapping `GgufModel`, using the existing `SlruCache<TKey,TValue>` primitive
already proven by `ExpertSlotManager`. Wire it into the test suite, measure the result against
the original 59.5 GB observation, done.

`024` is not discarded — it's kept as a shelved, reviewed, implementation-ready design for the
day Stingray actually needs to serve more than one model. At that point it should be
re-validated (interfaces may drift from whatever `025`/`026` end up looking like in practice —
`024`'s own `ModelRuntime`/`ModelRuntimeManager` naming was written assuming the full-featured
version; `025`/`026` deliberately use a narrower `SharedModelCache`/`ModelHandle` shape instead
of pretending to be a subset of `024`'s types), but the hard design thinking — especially the
eviction-safety and session-affinity correctness constraints the reviewer caught — doesn't need
to be redone.

## What to sanity-check in review

Not "is 024 good" (already reviewed) and not "is 025/026 technically correct" (that's an
implementation-detail review once code exists) — specifically:

1. Is the line drawn between "measured problem" and "speculative production feature" in the
   right place? Is there a piece of `024` that actually *is* needed to fix the measured
   problem, that this decision incorrectly filed under "shelve for later"?
2. Is `025`/`026`'s soft-cap-under-refcount-pressure behavior (see `026`, "What happens when
   everything resident is in use") an acceptable simplification, or does deferring it to
   "shelved scheduler work" actually leave a real gap even for the test-suite use case?
3. Given `024` is being shelved rather than built, was writing the full, detailed version of it
   *before* asking "do we need all of this" the right sequence — or should the over-engineering
   question have been asked before that much design effort went in? (Asked honestly, not
   rhetorically: this is worth an outside opinion, not just self-assessment.)
