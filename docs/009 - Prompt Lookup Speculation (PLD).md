# Implementation Plan — Prompt Lookup Speculation (PLD) (`Plan 009`)

## Objective

Add **Prompt Lookup Speculation (PLD)** to OpenTail.Stingray.

PLD provides speculative draft tokens without loading a second draft model.

Instead of:

```text
Target model
     │
     │
Draft model ──→ K proposed tokens
     │
     ▼
Target verification
```

PLD uses tokens already present in the current session context:

```text
Session token history
        │
        ▼
N-gram lookup
        │
        ▼
candidate continuation
        │
        ▼
existing SpeculativeDecoder
        │
        ▼
target-model verification
```

The result should be:

- zero additional model weights;
- zero additional model inference for drafting;
- reuse of the existing `SpeculativeDecoder`;
- no duplicate acceptance/rollback logic;
- useful behaviour for repetitive prompts, code, documents, RAG context, JSON/tool calls and multi-turn conversations.

The implementation should be **small and conservative**. PLD is a proposal generator, not a second speculative-decoding engine.

---

# 1. Architectural principle

The most important design decision:

> **Do not modify or duplicate the existing speculative verification algorithm unless repository inspection proves an integration point is missing.**

Stingray already has:

```text
SpeculativeDecoder
 ├── BatchVerify()
 ├── greedy acceptance
 ├── stochastic acceptance
 ├── residual sampling
 ├── adaptive lookahead
 ├── rollback
 └── SpeculativeMetrics
```

PLD should provide:

```text
PromptLookupDecoder
        │
        ▼
IReadOnlyList<int> draftTokens
        │
        ▼
existing SpeculativeDecoder
```

Conceptually:

```csharp
var draft = promptLookup.BuildDraft(
    sessionTokens,
    currentPosition,
    lookahead);

return speculativeDecoder.Verify(
    draft,
    ...);
```

The exact API should follow the current implementation.

---

# 2. What PLD actually does

Suppose the session contains:

```text
The quick brown fox jumps over the lazy dog.
The quick brown
```

The current generation position is after:

```text
The quick brown
```

PLD searches earlier context for the same suffix:

```text
The quick brown
```

and finds:

```text
The quick brown fox jumps over...
```

It proposes:

```text
fox jumps over ...
```

The target model then verifies those tokens.

If the target model agrees:

```text
fox jumps over
```

can be accepted in a single verification pass.

If it disagrees:

```text
fox jumps [target says "the"]
```

the existing speculative rollback/acceptance logic handles the divergence.

PLD therefore does **not** assert that the copied text is correct.

It merely says:

> "This continuation already occurred in the context; perhaps the model will produce it again."

---

# 3. Core abstraction

Add a focused component, conceptually:

```csharp
public interface IPromptLookupDecoder
{
    IReadOnlyList<int> Propose(
        ReadOnlySpan<int> tokenHistory,
        int currentPosition,
        int maxDraftTokens);
}
```

Potential implementation:

```text
PromptLookupDecoder.cs
```

The exact interface is flexible.

If Stingray's existing speculative API has a better draft-provider abstraction, use that instead.

A good architecture would be:

```text
IDraftProvider
     │
     ├── ModelDraftProvider
     │
     └── PromptLookupDecoder
```

**However, do not create `IDraftProvider` merely for theoretical future use if it adds unnecessary plumbing.**

Inspect the current `SpeculativeDecoder` first.

If it already accepts draft tokens directly, `PromptLookupDecoder` can remain extremely small.

---

# 4. N-gram matching

The basic algorithm:

Given:

```text
history = [tokens...]
current position = P
N = lookup context length
K = maximum draft length
```

take the previous `N` tokens:

```text
history[P-N .. P]
```

and search earlier history for the same sequence.

For example:

```text
history:

A B C D E F G
        ↑
        current context = C D E

earlier match:
X Y C D E F G
      ↑
      match

draft:
F G
```

The matched continuation becomes the speculative draft.

---

# 5. Longest-match search

Prefer the longest useful N-gram match.

For example, if:

```text
N = 4
```

doesn't find a match, try:

```text
N = 3
N = 2
```

rather than immediately giving up.

Conceptually:

```text
for n = MaxN down to MinN:
    find previous occurrence of suffix[n]

    if found:
        return following tokens
```

This should remain bounded.

Do not perform an unbounded quadratic search over the entire context every generation step.

---

# 6. Lookahead limit

PLD must respect the existing speculative lookahead limit.

For example:

```text
MaxDraftTokens = 8
```

means:

```text
match continuation
      ↓
take at most 8 tokens
```

Do not let prompt lookup independently decide a larger K than `SpeculativeDecoder`.

The existing adaptive lookahead mechanism should remain authoritative where possible.

---

# 7. Current-generation boundary

Only search **tokens already available as context**.

Never search beyond the current generation cursor.

For example:

```text
tokens:

[prompt prompt prompt][generated generated][CURRENT]
                                      ↑
                                 searchable
```

The current position and future/uncommitted tokens must not accidentally become lookup sources.

This is particularly important when speculative decoding itself is active.

---

# 8. Do not search the speculative draft as committed history

Suppose:

```text
Committed history:
A B C D

PLD proposes:
E F G
```

Do not immediately index:

```text
E F G
```

as though it were confirmed prompt history.

Only committed tokens should become lookup candidates.

This prevents speculation from feeding itself recursively.

---

# 9. Candidate continuation bounds

If a match occurs near the end of the historical context:

```text
... A B C
        ↑ match
```

and there are only two following tokens:

```text
D E
```

return:

```text
D E
```

not an error.

The number of proposed tokens is:

```text
min(
    availableContinuation,
    MaxDraftTokens
)
```

---

# 10. Avoid proposing EOS accidentally

Decide explicitly how EOS is handled.

A sensible initial implementation is:

- allow EOS only if the existing speculative decoder already supports it safely;
- otherwise stop the PLD draft before EOS.

Do not invent special EOS semantics.

Follow Stingray's existing generation termination contract.

---

# 11. Multiple matches

There may be multiple occurrences of the same N-gram.

Example:

```text
A B C D E
X A B C F G
Y A B C D E
```

Both contain:

```text
A B C
```

Possible strategies include:

1. nearest previous occurrence;
2. longest continuation;
3. most recent occurrence;
4. score candidates by continuation frequency.

For the first implementation, use **nearest previous occurrence with the longest available continuation**, unless repository evidence suggests another choice.

The key is deterministic behaviour.

---

# 12. Why nearest previous occurrence is useful

In conversational/code contexts, the most recent occurrence is often more relevant:

```text
old example:
foo.Bar()

recent example:
foo.Bar(x)

current:
foo.Bar(
```

The recent continuation is generally a better proposal than something from thousands of tokens earlier.

This is a heuristic, not a correctness requirement.

The target model remains authoritative.

---

# 13. Repetition protection

PLD must not create pathological self-repetition.

Consider:

```text
A B C A B C A B C ...
```

A naive lookup may repeatedly propose the same continuation.

Add a simple safeguard if needed:

- do not select a match whose continuation would immediately reproduce the current generated suffix indefinitely;
- or rely on the existing generation repetition controls.

Prefer the latter if already present.

Do not introduce a second repetition-penalty system.

---

# 14. Token-level matching

PLD must operate on **token IDs**, not strings.

Do not do:

```csharp
string.Contains(...)
```

or repeatedly decode the whole prompt.

Use:

```text
int token ID
```

sequences.

This gives:

- exact tokenizer alignment;
- no Unicode ambiguity;
- no whitespace ambiguity;
- cheap comparison;
- compatibility with the existing KV/session token history.

---

# 15. Important tokenizer invariant

PLD operates on the **same tokenizer vocabulary that produced the session token history**.

Never compare token IDs from different tokenizers/models.

The session's tokenizer/model identity should therefore be the source of truth.

This also aligns with the namespace isolation already introduced for prefix caching.

---

# 16. Efficient lookup structure

A naive implementation might do:

```text
for every possible earlier position:
    compare N tokens
```

This is acceptable for an initial tiny implementation if contexts are small.

However, prefer a lightweight index if the existing architecture makes it straightforward.

Potential structure:

```text
N-gram key
    ↓
last occurrence
```

For example:

```csharp
Dictionary<NGramKey, int>
```

where:

```text
NGramKey = hash of N token IDs
```

But **do not introduce a complex persistent index unless measurements show the naive implementation is insufficient**.

The feature's initial purpose is functionality, not lookup micro-optimisation.

---

# 17. Hash collision safety

If a hash-based index is used:

> A hash match is only a candidate match.

Never assume:

```text
hash(A) == hash(B)
```

means:

```text
A == B
```

Verify the actual token IDs before accepting the match.

A simple robust design is:

```text
hash
 ↓
candidate position
 ↓
exact token comparison
 ↓
confirmed match
```

---

# 18. Context window

Only search tokens that remain inside the model's valid context window.

Do not create references to discarded context.

The lookup range should respect the existing session/model context limits.

Conceptually:

```text
lookupStart =
    max(0, currentPosition - ContextWindow);
```

Use existing context-window/session abstractions instead of introducing a second limit.

---

# 19. Prompt vs generated history

PLD should search the entire **committed token history**, not only the original prompt.

That is deliberate.

Useful repetitions can occur in:

- system prompts;
- user prompts;
- RAG documents;
- previous assistant turns;
- generated text already committed;
- code context.

For example:

```text
User:
Implement method Foo...

Assistant:
...

User:
Now implement Bar...

```

Previous generated content may provide excellent continuations.

Therefore call the source:

```text
Session token history
```

rather than:

```text
Prompt tokens
```

---

# 20. Interaction with prefix caching

PLD and the new Radix Prefix Cache solve different problems.

Prefix cache:

```text
previous session
     ↓
reuse physical KV
```

PLD:

```text
current session token history
     ↓
reuse textual/token continuation
```

They should coexist.

Do not make PLD depend on the prefix cache.

Do not store PLD matches in `RadixPrefixTree`.

---

# 21. Interaction with session branching

The new session multiverse should naturally work with PLD.

Example:

```text
Parent
  │
  ├── Branch A
  ├── Branch B
  ├── Branch C
  └── Branch D
```

Each branch has its own committed token history.

Therefore each branch's PLD lookup must operate on:

```text
that branch's token history
```

not the parent's mutable history.

The shared physical KV pages are irrelevant to the lookup algorithm.

---

# 22. Interaction with constrained sampling

PLD must work with the ConstraintEngine.

This is particularly valuable for JSON/tool calls.

Pipeline:

```text
PLD proposal
     ↓
existing speculative verification
     ↓
constraint validation
     ↓
accepted tokens
```

A proposed token sequence that violates the active constraint must not bypass constraint enforcement.

The target/constraint pipeline remains authoritative.

Do not implement a separate JSON validator inside PLD.

---

# 23. Interaction with existing SpeculativeDecoder

This is the central integration.

Existing:

```text
SpeculativeDecoder
```

already handles:

```text
draft
  ↓
BatchVerify
  ↓
accept/reject
  ↓
rollback
  ↓
residual sampling
```

PLD should simply become another source of draft tokens.

Conceptually:

```text
                ┌── Model Draft
                │
Draft Tokens ───┤
                │
                └── Prompt Lookup Draft
                           │
                           ▼
                  SpeculativeDecoder
                           │
                           ▼
                    Target model
```

If the existing API currently takes a draft model directly, refactor only enough to allow an already-tokenised draft proposal.

Do **not** duplicate `BatchVerify()`.

---

# 24. Adaptive lookahead

The existing adaptive lookahead should continue to work.

For example:

```text
PLD proposes K = 8
        ↓
only 3 accepted
        ↓
existing metrics reduce K
```

and:

```text
PLD proposes K = 4
        ↓
high acceptance
        ↓
existing mechanism increases K
```

PLD should not create a competing adaptive controller.

---

# 25. PLD metrics

Extend existing `SpeculativeMetrics` rather than introducing a separate metrics system.

Useful fields:

```text
PromptLookupAttempts
PromptLookupHits
PromptLookupMisses
PromptLookupProposedTokens
PromptLookupAcceptedTokens
PromptLookupAcceptanceRate
```

Most important:

```text
PromptLookupAcceptanceRate
```

This tells us whether PLD is actually useful for a workload.

Potentially also track:

```text
LookupSearchTokens
```

to identify pathological lookup costs.

---

# 26. No-match behaviour

If no N-gram match exists:

```text
PromptLookupDecoder.Propose()
        ↓
empty draft
```

The system must simply fall back to normal generation.

It must not:

- throw;
- stall;
- perform a fake verification pass;
- alter model logits.

Conceptually:

```text
No match
   ↓
normal target generation
```

---

# 27. Very short history

If the committed history is shorter than `MinN`:

```text
tokens < N
```

PLD should simply return no proposal.

Do not throw.

---

# 28. Draft validity

The PLD draft is always treated as speculative.

Never assume:

```text
copied from prompt = correct
```

The target model must verify it through the existing speculative mechanism.

This is essential for correctness.

---

# 29. API configuration

Expose PLD through existing generation options.

Conceptually:

```csharp
new GenerationOptions
{
    Speculation = new SpeculationOptions
    {
        EnablePromptLookup = true,
        PromptLookupMinNgram = 3,
        PromptLookupMaxNgram = 8,
        MaxDraftTokens = 8
    }
}
```

But inspect existing speculative configuration first.

Do not create a parallel configuration object if `SpeculativeDecoder` already has an appropriate options model.

---

# 30. Sensible defaults

Provide conservative defaults.

Conceptually:

```text
EnablePromptLookup = false initially
MinNgram = 3
MaxNgram = 8
MaxDraftTokens = existing speculative maximum
```

Whether PLD should ultimately default to enabled should be decided after correctness testing and benchmark evidence.

Do not make it automatically active merely because the feature exists.

---

# 31. Tests

Create:

```text
PromptLookupDecoderTests.cs
```

and integration tests for the actual speculative pipeline.

Mandatory tests:

### Test 1 — ExactNGramMatch

Given:

```text
A B C D E F
      ^
current context A B C
```

verify:

```text
draft = D E F
```

---

### Test 2 — LongestAvailableMatch

Given multiple N values, verify the longest valid match is preferred.

---

### Test 3 — NoMatch

Verify no matching N-gram returns an empty draft.

---

### Test 4 — ShortHistory

History shorter than minimum N returns no draft.

---

### Test 5 — EndOfHistory

A match with only two continuation tokens returns exactly two tokens.

---

### Test 6 — CurrentPositionBoundary

Verify PLD never searches tokens beyond the current committed position.

---

### Test 7 — ExactTokenMatching

Tokens with equivalent-looking decoded strings but different tokenisation must not be treated as equivalent.

---

### Test 8 — MultipleMatches

Verify deterministic selection when several identical N-grams exist.

---

### Test 9 — HashCollisionSafety

If a hash index is used, deliberately create a collision and verify exact token comparison prevents a false match.

---

### Test 10 — BranchIsolation

Fork a session and verify each branch searches its own token history.

---

### Test 11 — ConstraintCompatibility

Verify an invalid PLD proposal cannot bypass the active constraint.

---

### Test 12 — SpeculativeVerification

Verify PLD proposals pass through the existing `SpeculativeDecoder`.

Do not create a separate verification implementation.

---

### Test 13 — RejectionRollback

Force the target model to reject part of a PLD draft.

Verify existing rollback behaviour remains correct.

---

### Test 14 — AcceptanceMetrics

Verify PLD metrics accurately record:

```text
lookup hit
proposed tokens
accepted tokens
```

---

### Test 15 — NoMatchNormalGeneration

When PLD finds no match, normal generation still proceeds correctly.

---

# 32. Integration test

Add a real end-to-end test:

```text
prompt containing repeated text
        ↓
PLD
        ↓
draft tokens
        ↓
existing SpeculativeDecoder
        ↓
target verification
        ↓
generation
```

Verify:

1. PLD finds a proposal;
2. proposal reaches the existing speculative decoder;
3. output is correct;
4. rejected proposals are rolled back correctly;
5. generation terminates normally.

---

# 33. Important correctness test

Create a deliberately misleading prompt.

For example, the prompt should contain:

```text
A B C D
```

but the target model should prefer:

```text
A B C X
```

PLD proposes:

```text
D
```

The target verification must reject it.

This proves:

> PLD is a draft proposal mechanism, not a bypass around target-model inference.

---

# 34. Performance sanity test

This is not a performance-tuning task, but establish basic bounds.

Measure:

```text
normal generation
vs
PLD enabled
```

and record:

```text
draft lookup time
target verification time
tokens accepted
```

The lookup operation must not become so expensive that it consumes the savings from speculation.

If the naive search is too expensive for large contexts, only then introduce a lightweight N-gram index.

---

# 35. Allocation sanity

Avoid allocating strings during every lookup.

Preferred:

```text
token IDs
    ↓
integer comparisons
```

rather than:

```text
token IDs
    ↓
decode whole prompt
    ↓
string search
    ↓
re-tokenize
```

No re-tokenization should occur.

---

# 36. Cancellation

PLD lookup must respect the generation cancellation path.

If lookup itself becomes expensive on a large context, it must not ignore cancellation indefinitely.

If lookup is deliberately kept synchronous and bounded, document that it is bounded by:

```text
context length × maximum N
```

and remains small.

---

# 37. Thread safety

A `PromptLookupDecoder` should preferably be stateless or generation-local.

Do not maintain mutable global lookup state.

For branching:

```text
Branch A → lookup state A
Branch B → lookup state B
```

or derive lookup entirely from each branch's token history.

The latter is preferable if cheap enough.

---

# 38. Acceptance criteria

The feature is complete when:

- [ ] `PromptLookupDecoder` exists.
- [ ] It operates on token IDs.
- [ ] It searches committed session history.
- [ ] It finds repeated N-grams.
- [ ] It returns continuation tokens.
- [ ] It supports bounded N and K.
- [ ] It prefers an appropriate longest/recent match.
- [ ] It never searches beyond the committed cursor.
- [ ] It never treats speculative tokens as committed history.
- [ ] No match falls back cleanly to normal generation.
- [ ] Draft tokens feed the existing `SpeculativeDecoder`.
- [ ] Existing acceptance/rejection logic is reused.
- [ ] Existing rollback logic is reused.
- [ ] Existing adaptive lookahead remains authoritative.
- [ ] PLD works with session branching.
- [ ] PLD works with constrained generation.
- [ ] Prefix caching remains unaffected.
- [ ] PLD metrics integrate with existing speculative metrics.
- [ ] No second speculative verification implementation exists.
- [ ] No additional model weights are required.
- [ ] No prompt re-prefill is performed for lookup.
- [ ] Unit tests pass.
- [ ] End-to-end speculative test passes.
- [ ] Full Stingray test suite passes.
- [ ] Release build passes.

---

# 39. Definition of done

The resulting architecture should be:

```text
                    OpenTail.Stingray
                           │
                           ▼
                  InferenceSession
                           │
                           ▼
                Speculation controller
                     /           \
                    /             \
           Draft model             PLD
                │                   │
                │              token history
                │                   │
                └────────┬──────────┘
                         ▼
                 draft token sequence
                         │
                         ▼
                SpeculativeDecoder
                         │
                    BatchVerify
                         │
                         ▼
                   Target model
                         │
                         ▼
                 accepted tokens
```

PLD therefore becomes a **zero-model-cost draft provider** for the speculative infrastructure Stingray already possesses.

The fundamental invariant is:

> **Prompt lookup may propose tokens, but only the existing target-model speculative verification path may commit them.**

---

## Implementation flexibility

The code snippets and API names in this plan are **conceptual guidance, not requirements to reproduce them literally**.

Before implementation, inspect the current:

- `SpeculativeDecoder`;
- `InferenceSession`;
- `InferenceRuntime`;
- session token-history representation;
- speculative options/configuration;
- `SpeculativeMetrics`;
- prompt/context cursor;
- tokenizer API;
- existing adaptive lookahead implementation.

If the current `SpeculativeDecoder` already has a clean draft-token injection point, use it.

If it does not, make the **smallest architectural change necessary to expose one**.

Do not build a second speculative decoder.

Likewise, if a simple backward N-gram scan is sufficiently cheap for the current context sizes, use it. If the current context lengths make that expensive, introduce a lightweight token N-gram index.

**The ~150-line estimate is a target for simplicity, not a requirement. Correct integration with Stingray's existing speculative machinery is more important than line count.**