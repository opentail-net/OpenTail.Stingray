# OpenTail.Stingray — Plan 011

# Speculative Cascade Ensemble (PLD + Model Draft Pipeline)

## Objective

Extend the existing `SpeculativeDecoder` so that it can use **two speculative candidate sources in a cascade**:

```text
                    SpeculativeDecoder
                           │
                           ▼
                    Prompt Lookup (PLD)
                           │
                 ┌─────────┴─────────┐
                 │                   │
              usable              unusable
              candidates           / miss
                 │                   │
                 ▼                   ▼
             Verify             Small Draft Model
                 │                   │
                 │                   ▼
                 │               Verify
                 │                   │
                 └─────────┬─────────┘
                           ▼
                       Commit
```

The first speculative source is **Prompt Lookup / n-gram speculation** because it requires no neural forward pass.

If PLD cannot produce a useful candidate block, the decoder falls back to the existing **small neural draft model**.

The target model remains the authority.

This feature must **not change the correctness semantics of target-model verification, rollback, checkpointing, or session advancement**.

---

# 1. Important implementation principle

This plan is an extension of the existing speculative decoding architecture.

**Do not rewrite `SpeculativeDecoder`.**

First inspect the current implementation and identify:

* existing PLD/prompt lookup implementation
* existing draft-model implementation
* candidate representation
* verification path
* rollback/rewind logic
* checkpoint handling
* sampling behaviour
* greedy behaviour
* metrics
* session/KV interaction

Reuse these mechanisms wherever possible.

The goal is:

```text
existing speculative decoder
            +
candidate-source selection
            =
cascade speculative decoder
```

not:

```text
existing speculative decoder
            ↓
complete rewrite
```

---

# 2. Repository-first requirement

Before changing code, inspect the current repository.

Identify the actual implementations of:

```text
SpeculativeDecoder
Prompt Lookup / PLD
draft model
candidate generation
candidate verification
rollback / rewind
session state
KV sequence state
sampling
metrics
```

Do not assume filenames or APIs from this plan.

Adapt the implementation to the actual current Stingray architecture.

Document which existing classes correspond to each responsibility before making changes.

---

# 3. Preserve the existing target-model authority

The target model remains authoritative.

Neither PLD nor the draft model is permitted to directly commit tokens to the target session.

The conceptual flow is:

```text
Target Session
     │
     │ checkpoint
     ▼
Candidate generation
     │
     ├── PLD
     │
     └── Draft Model
     │
     ▼
Candidate block
     │
     ▼
Target verification
     │
     ├── accepted prefix
     │
     └── rejected suffix
     │
     ▼
Commit / rollback
```

The candidate generator must not bypass verification.

---

# 4. Candidate-source abstraction

If the existing code permits it cleanly, introduce a small internal abstraction for candidate sources.

Conceptually:

```csharp
interface ISpeculationSource
{
    SpeculationResult TryGenerate(...);
}
```

The exact API should follow the existing codebase.

Possible implementations:

```text
PromptLookupSpeculationSource
DraftModelSpeculationSource
```

The result should communicate at least:

```text
Did the source produce candidates?
Candidate tokens
Candidate count
Reason for failure / no candidates
Optional source metadata
```

Do not expose unnecessary implementation details.

---

# 5. Do not duplicate verification

There must remain exactly one target verification mechanism.

Avoid:

```text
PLD verifier
Draft verifier
```

Instead:

```text
PLD
  ↓
candidate block
  ↓
COMMON verifier

Draft Model
  ↓
candidate block
  ↓
COMMON verifier
```

This is important for correctness.

The verification result should mean exactly the same thing regardless of where the candidate block originated.

---

# 6. Cascade policy

The default policy is:

```text
1. Attempt PLD.
2. If PLD produces a usable candidate block, verify it.
3. If PLD cannot produce a usable candidate block, use the draft model.
4. Verify the draft-model candidates.
5. Commit according to the existing speculative-decoding semantics.
```

Do not fall back to the draft model merely because the target rejects a PLD candidate.

A rejected PLD candidate is a **verification result**, not necessarily a PLD-generation failure.

The distinction is important.

---

# 7. Define "usable PLD candidates"

Do not simply define:

```text
PLD found any match → use it
```

Instead define an explicit eligibility rule.

At minimum:

```text
candidate count >= configured minimum
```

where the minimum may initially be:

```text
1
```

if that matches the existing decoder semantics.

The implementation should allow a future policy such as:

```text
minimum candidate length
maximum candidate length
minimum expected acceptance rate
```

without requiring a rewrite.

Do not implement sophisticated prediction heuristics in this plan.

---

# 8. PLD candidate generation must be side-effect safe

PLD lookup should not mutate target inference state merely by generating candidates.

It should read:

```text
token history
prompt/context
existing session state as required
```

and produce:

```text
candidate tokens
```

It should not commit those tokens to the target KV cache.

---

# 9. Draft-model fallback

If PLD cannot produce a usable candidate block:

```text
PLD
 ↓
no usable candidates
 ↓
DraftModelSpeculationSource
 ↓
existing draft generation
```

Reuse the existing draft-model generation path.

Do not create a second draft-model implementation.

Do not change draft-model semantics unless required by the existing interfaces.

---

# 10. Preserve existing candidate length limits

The cascade must respect the existing speculative-decoding candidate limit.

For example:

```text
MaxDraftTokens
```

or whatever equivalent exists.

PLD must not bypass that limit.

If PLD naturally produces 50 candidates but the decoder allows 8:

```text
PLD
 ↓
50 candidates
 ↓
truncate to existing maximum
 ↓
verify
```

Use the existing candidate representation and limits.

---

# 11. PLD must not generate impossible candidates

The PLD implementation must continue respecting its existing rules around:

* token boundaries
* n-gram matching
* sequence continuity
* prompt/history boundaries
* maximum candidate length

Do not weaken PLD validation merely to increase hit rate.

Correct candidates matter more than raw candidate count.

---

# 12. Greedy decoding

The cascade must work correctly in greedy mode.

Test:

```text
Greedy
  ↓
PLD hit
  ↓
verification
```

and:

```text
Greedy
  ↓
PLD miss
  ↓
draft model
  ↓
verification
```

Existing greedy output must remain unchanged when cascade mode is disabled.

---

# 13. Sampling

The cascade must also work with the existing sampling path.

This is particularly important.

PLD candidates are deterministic/retrieved candidates, while the target model may be operating under:

```text
temperature
top-k
top-p
other existing sampling controls
```

Do not assume that a candidate retrieved from history is automatically valid under the sampling semantics.

The existing target verification mechanism must remain the authority.

Do not introduce a separate PLD sampling algorithm unless the existing architecture explicitly requires one.

---

# 14. Do not change target sampling semantics

The introduction of PLD must not change:

```text
temperature
top-k
top-p
seed handling
random number consumption
```

unless a documented existing speculative-decoding rule already requires it.

This is particularly important for reproducibility.

---

# 15. Checkpoint semantics

Before candidate generation, preserve the existing checkpoint behaviour.

Conceptually:

```text
Target session
     │
     ▼
checkpoint
     │
     ▼
candidate generation
     │
     ▼
verification
```

If the existing implementation already has a checkpoint/rewind abstraction, reuse it.

Do not create a parallel rollback system specifically for PLD.

---

# 16. Candidate source must be replaceable

The architecture should make this possible:

```text
SpeculativeDecoder
        │
        ▼
ISpeculationSource
        │
        ├── PLD
        ├── Draft Model
        └── future source
```

This leaves room for future strategies such as:

```text
grammar-based speculation
cache-based speculation
retrieval-based speculation
another small model
```

without modifying verification.

---

# 17. Cascade should be a policy

Conceptually:

```csharp
ISpeculationPolicy
```

or equivalent internal strategy.

Possible initial implementation:

```text
Cascade:
    Try PLD
    If usable → use PLD
    Else → draft model
```

Do not build a complex policy engine.

The important architectural distinction is:

```text
candidate source
        ≠
candidate selection policy
        ≠
candidate verification
```

---

# 18. Disabled-mode compatibility

The feature must be completely optional.

There should be a mode equivalent to:

```text
Cascade disabled
```

in which existing behaviour is preserved.

At minimum support:

```text
Draft model only
PLD only
PLD → Draft cascade
```

if the current architecture makes these modes practical.

---

# 19. Configuration

Introduce the smallest reasonable configuration surface.

Conceptually:

```text
SpeculativeCascadeOptions
{
    Enabled
    PreferPromptLookup
    MinimumPlannedCandidates
}
```

Use existing configuration conventions.

Avoid exposing internal implementation details.

Do not add dozens of tuning parameters.

---

# 20. Metrics

Add metrics that allow the cascade to be evaluated.

At minimum:

```text
PLDAttempts
PLDHits
PLDMisses
PLDCandidateTokens
PLDVerifiedTokens
PLDAcceptedTokens

DraftFallbacks
DraftCandidateTokens
DraftVerifiedTokens
DraftAcceptedTokens

TotalSpeculationAttempts
TotalSpeculatedTokens
TotalAcceptedSpeculatedTokens
```

Also track source usage:

```text
PLD chosen
Draft chosen
```

---

# 21. Measure acceptance rate separately

Do not combine PLD and draft-model acceptance statistics.

We need to know:

```text
PLD:
    candidates = X
    accepted = Y
    acceptance rate = Y/X

Draft:
    candidates = A
    accepted = B
    acceptance rate = B/A
```

This tells us whether the cascade actually improves speculation.

---

# 22. Measure fallback effectiveness

Important metric:

```text
PLD miss rate
```

and:

```text
tokens generated through PLD
tokens generated through draft model
```

The desired result is not merely:

> PLD finds lots of matches.

It is:

> PLD avoids unnecessary draft-model work without reducing overall generation quality or throughput.

---

# 23. Performance instrumentation

Measure:

```text
PLD lookup time
draft-model generation time
target verification time
total speculation overhead
tokens/sec
```

The critical comparison is:

```text
Draft-only
vs
PLD-only
vs
PLD → Draft
```

---

# 24. Do not assume PLD is always faster

The cascade should not become slower because PLD is invoked for every step when:

```text
PLD lookup cost
>
saved draft-model work
```

The initial implementation may always attempt PLD because the lookup should be extremely cheap.

But measure this.

If benchmarks show a pathological workload where PLD overhead dominates, the policy can later evolve.

Do not prematurely add complicated heuristics.

---

# 25. No KV representation changes

This plan must **not** introduce:

```text
INT8 KV
INT4 KV
TurboQuant
KV compression
new page formats
```

The existing KV representation remains unchanged.

This is a speculative candidate-source feature, not a KV-storage feature.

---

# 26. No scheduler redesign

Do not modify:

```text
continuous batching
scheduler
request queue
batch allocation
```

unless an existing API requires a minimal integration change.

The cascade operates inside the existing speculative-decoding execution path.

---

# 27. No session architecture redesign

Use the session/state abstractions produced by the previous plans.

Do not create:

```text
PLD session
Draft session
Cascade session
```

There is one target session.

Candidate sources operate against the existing speculative state mechanism.

---

# 28. Important distinction: PLD failure vs verification rejection

These are different events.

### PLD generation failure

```text
No usable candidate block
        ↓
fallback to draft model
```

### PLD verification rejection

```text
Candidate block exists
        ↓
target verifies
        ↓
some/all candidates rejected
```

Do **not** automatically run the draft model after a verification rejection unless a future policy explicitly chooses to do so.

The first implementation should keep these cases separate.

---

# 29. First implementation should be simple

The initial cascade should be:

```text
for each speculative iteration:

    candidates = PLD.TryGenerate()

    if candidates are usable:
        source = PLD
    else:
        candidates = DraftModel.Generate()
        source = DraftModel

    result = Target.Verify(candidates)

    CommitOrRollback(result)
```

Do not add:

```text
PLD confidence model
acceptance prediction
dynamic thresholds
multiple draft models
adaptive switching
```

yet.

---

# 30. Future extension point

The architecture should eventually allow:

```text
                    Candidate Cascade
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
            PLD        Draft Model    Other Source
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                       Verifier
```

But Plan 011 implements only:

```text
PLD → Draft Model
```

---

# 31. Testing strategy

Tests must be layered.

## A. Existing behaviour

With cascade disabled:

```text
existing tests pass unchanged
```

This is mandatory.

---

## B. PLD hit

Construct a deterministic token history where PLD is known to find a candidate.

Verify:

```text
PLD invoked
draft model NOT invoked
target verification invoked
correct output produced
```

---

## C. PLD miss

Construct a history where no valid PLD candidate exists.

Verify:

```text
PLD invoked
PLD returns no usable candidates
draft model invoked
target verification invoked
correct output produced
```

---

## D. PLD partial candidate

Create a case where PLD finds fewer candidates than the configured maximum.

Verify that the decoder handles the shorter block correctly.

---

## E. PLD candidate rejection

Force the target model to reject a PLD candidate.

Verify:

```text
target state remains correct
rollback is correct
next generated token is correct
```

Do not invoke draft fallback unless explicitly configured.

---

## F. Full acceptance

Construct a case where the target accepts all PLD candidates.

Verify:

```text
all accepted tokens committed
KV state correct
next target position correct
```

---

## G. Partial acceptance

Construct:

```text
candidate A
candidate B
candidate C
candidate D
```

where:

```text
A accepted
B accepted
C rejected
```

Verify that only the accepted prefix is committed and the decoder produces the correct next token according to existing speculative-decoding semantics.

---

# 32. Session-state tests

Because Stingray now has explicit session/KV state, test:

```text
PLD speculation
    ↓
verification
    ↓
session cursor
    ↓
KV sequence length
```

against the expected values.

After every speculative iteration:

```text
logical token position
KV length
session state
```

must agree.

---

# 33. Fork/COW compatibility

If session fork/COW is already available, add at least one integration test:

```text
Parent session
     │
     ├── child A
     └── child B
```

Run cascade speculation independently.

Verify that:

```text
A cannot modify B
B cannot modify A
parent remains correct
```

Do not redesign COW for this feature.

---

# 34. Checkpoint/rollback compatibility

Run:

```text
checkpoint
 ↓
PLD candidates
 ↓
verification rejection
 ↓
rollback
 ↓
continue generation
```

and compare against a non-speculative baseline.

The resulting target/session state must match the baseline within the existing defined semantics.

---

# 35. Deterministic regression test

Create a fixed test case:

```text
model
prompt
generation settings
seed
```

and record the expected output for:

```text
cascade disabled
```

Then enable:

```text
PLD → Draft cascade
```

For greedy decoding, the generated sequence should be identical.

For sampling, follow the existing speculative-decoding reproducibility rules and verify that the implementation does not introduce unintended RNG/state changes.

---

# 36. Benchmark suite

Benchmark three configurations:

```text
A. Normal / non-speculative
B. Draft-model speculative decoding
C. PLD → Draft cascade
```

Then test workloads including:

```text
code
JSON
repetitive prose
ordinary prose
creative text
long-context continuation
```

Record:

```text
tokens/sec
PLD hit rate
PLD acceptance rate
draft fallback rate
draft acceptance rate
target verification cost
```

---

# 37. Expected workload behaviour

The intended pattern is:

### Repetitive/code/JSON

```text
PLD hit
 ↓
cheap candidates
 ↓
verification
```

Many iterations should avoid invoking the draft model.

### Creative/unpredictable text

```text
PLD miss
 ↓
draft model
 ↓
verification
```

The system retains the intelligence of the neural draft model.

---

# 38. Success criteria

The feature is successful when:

* [ ] Existing speculative decoding still works unchanged with cascade disabled.
* [ ] PLD can supply candidate blocks.
* [ ] Draft model remains available as fallback.
* [ ] Target verification is shared between both sources.
* [ ] PLD never directly commits target state.
* [ ] Draft model never directly commits target state.
* [ ] Checkpoint/rollback semantics remain correct.
* [ ] Greedy generation remains correct.
* [ ] Sampling remains compatible with existing semantics.
* [ ] Session/KV cursor remains correct.
* [ ] COW/fork remains correct where supported.
* [ ] No KV storage representation changes are introduced.
* [ ] No scheduler redesign is required.
* [ ] Metrics distinguish PLD and draft behaviour.
* [ ] Benchmarks demonstrate whether the cascade actually improves throughput.
* [ ] PLD misses cleanly fall back to the draft model.
* [ ] PLD verification rejection does not accidentally corrupt state.
* [ ] Existing conformance/golden-logit tests remain green.

---

# 39. Implementation order

The coding AI should follow this exact order.

### STEP 1 — Audit

Inspect the existing:

```text
SpeculativeDecoder
PLD implementation
draft model
candidate representation
verification
rollback
sampling
metrics
session/KV APIs
```

Document the actual classes/methods involved.

### STEP 2 — Identify the smallest insertion point

Find the current point where the decoder obtains speculative candidates.

Change **that point**, rather than restructuring the decoder.

### STEP 3 — Extract candidate-source abstraction if useful

Only introduce an abstraction if it reduces duplication.

Do not create unnecessary architecture.

### STEP 4 — Wrap existing PLD

Expose existing PLD behaviour through the candidate-source interface.

Do not rewrite PLD.

### STEP 5 — Wrap existing draft model

Expose existing draft-model generation through the same interface.

Do not rewrite the draft model.

### STEP 6 — Implement cascade

Implement:

```text
PLD
 ↓
usable?
 ├── yes → verify
 └── no  → draft model → verify
```

### STEP 7 — Preserve common verification

Both sources must enter exactly the same verification path.

### STEP 8 — Add configuration

Support:

```text
existing draft-only mode
PLD-only mode if practical
PLD → Draft mode
```

### STEP 9 — Add metrics

Separate PLD and draft statistics.

### STEP 10 — Add correctness tests

Especially:

```text
hit
miss
partial
rejection
full acceptance
rollback
```

### STEP 11 — Run existing conformance suite

No existing inference/conformance regression is acceptable.

### STEP 12 — Benchmark

Compare:

```text
draft-only
PLD → Draft
```

and determine whether the cascade actually improves throughput.

---

# 40. Explicit non-goals

Do NOT implement in Plan 011:

* [ ] Dynamic KV quantization
* [ ] INT8/INT4 KV
* [ ] TurboQuant
* [ ] New KV page formats
* [ ] New scheduler
* [ ] New batching architecture
* [ ] New session architecture
* [ ] New speculative verification algorithm
* [ ] New sampling algorithm
* [ ] Multiple draft models
* [ ] Adaptive ML-based speculation policy
* [ ] PLD confidence prediction
* [ ] Automatic model selection
* [ ] Major refactoring of `SpeculativeDecoder`

If one of these appears necessary, stop and document why before expanding scope.

---

# 41. Final architectural target

The finished architecture should conceptually be:

```text
                         SpeculativeDecoder
                                │
                                ▼
                     SpeculationPolicy
                                │
                         PLD first
                                │
                    ┌───────────┴───────────┐
                    │                       │
               usable block             no block
                    │                       │
                    │                       ▼
                    │                 Draft Model
                    │                       │
                    └───────────┬───────────┘
                                │
                                ▼
                         Candidate Tokens
                                │
                                ▼
                       Target Verification
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
                 Accepted                Rejected
                    │                       │
                    └───────────┬───────────┘
                                ▼
                         Commit / Rollback
                                │
                                ▼
                           Target Session
                                │
                                ▼
                             KV Cache
```

The most important architectural rule is:

> **Candidate generation is replaceable; verification and state commitment are not.**

PLD and the draft model are merely different ways of proposing tokens.

The target model remains the authority.

---

# 42. Definition of the feature in one sentence

**Plan 011 adds a cheap prompt-lookup candidate source in front of the existing neural draft model, allowing Stingray to exploit repeated/code/JSON patterns without paying a draft-model forward pass while retaining the existing draft-model fallback for workloads where lookup speculation fails.**
