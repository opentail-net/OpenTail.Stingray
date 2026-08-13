# OpenTail.Stingray — Speculative Decoding Implementation Plan

## Objective

Add speculative decoding to OpenTail.Stingray using a smaller, faster **draft model** to propose multiple tokens and the main **target model** to verify them in a single forward pass.

The implementation must:

- preserve exact target-model semantics
- support deterministic greedy decoding first
- subsequently support stochastic sampling correctly
- reuse the existing Stingray session/KV infrastructure
- avoid copying the target model's entire KV cache
- integrate with continuous batching where practical
- expose useful performance metrics
- remain backend-independent
- work with CPU first
- allow CUDA/Vulkan optimisation later

The key performance objective is:

```text
Instead of:

Target → token 1
Target → token 2
Target → token 3
Target → token 4

Do:

Draft  → token 1 token 2 token 3 token 4
Target → verify all 4 in one forward operation
```

If the draft model predicts the target correctly, several target tokens can be accepted for approximately one target-model forward pass.

---

# 1. Important architectural prerequisite

Speculative decoding depends heavily on KV state management.

The preferred prerequisite is the new paged/logical KV architecture:

```text
IKvCache
    │
    └── IKvSequence
            │
            ├── committed KV
            │
            └── speculative KV
```

The speculative implementation must **not** deep-copy the target KV cache.

Bad:

```text
Target KV = 4 GB

Create speculation:
copy Target KV → another 4 GB
```

Good:

```text
Target KV
    │
    ├── committed pages
    │
    └── speculative pages
```

Speculative pages can be committed or released.

The target's existing committed KV remains untouched until verification succeeds.

---

# 2. Scope the implementation

Implement this in stages.

## Phase 1

Greedy speculative decoding.

## Phase 2

Deterministic sampling-compatible speculative decoding.

## Phase 3

Continuous-batching integration.

## Phase 4

Adaptive speculation length.

## Phase 5

Backend-specific optimisation.

Do **not** attempt all phases in one change.

---

# 3. Terminology

Use the following terminology consistently.

```text
Target model
    The model whose output defines the final result.

Draft model
    A smaller/faster model used to propose tokens.

Speculation window
    Number of tokens proposed by the draft model before target verification.

Proposal
    The sequence of tokens produced by the draft model.

Verification
    A target-model forward pass evaluating the proposed tokens.

Accepted tokens
    Draft tokens whose target probability satisfies the acceptance rule.

Rejected token
    First speculative token that fails verification.

Correction token
    Token sampled from the appropriate target residual distribution
    when using stochastic speculative decoding.

Committed KV
    Target KV state known to correspond to accepted tokens.

Speculative KV
    KV state that exists only for proposed tokens and can be committed or discarded.
```

---

# 4. Target public API

Do not expose implementation details through the normal generation API.

The preferred user-facing API should eventually look approximately like:

```csharp
var options = new SpeculativeDecodingOptions
{
    DraftModel = draftModel,
    MaxDraftTokens = 5
};

await foreach (var token in session.GenerateAsync(
    prompt,
    options))
{
    Console.Write(token);
}
```

Alternatively, if Stingray's existing architecture uses generation options objects, extend those rather than introducing an entirely separate API.

The API must remain compatible with ordinary generation.

Example:

```csharp
GenerateAsync(prompt)
```

must continue to work exactly as before.

---

# 5. Configuration object

Introduce something similar to:

```csharp
public sealed record SpeculativeDecodingOptions
{
    public required IModel DraftModel { get; init; }

    public int MaxDraftTokens { get; init; } = 5;

    public bool Enabled { get; init; } = true;

    public bool AdaptiveLength { get; init; } = false;
}
```

Do not expose internal KV checkpoint objects publicly unless necessary.

---

# 6. Model compatibility

Before enabling speculation, validate:

```text
Target tokenizer
Draft tokenizer
```

The safest first implementation requires:

```text
same tokenizer
```

because token IDs must be directly comparable.

Also validate:

```text
draft vocabulary == target vocabulary
```

If this is not true:

```text
disable speculative decoding
```

rather than attempting to translate tokens.

Do not implement cross-tokenizer speculative decoding in the first version.

---

# 7. Model compatibility checks

At minimum verify:

```text
Vocabulary size
Vocabulary/tokenizer identity
EOS token
BOS token where relevant
token ID compatibility
```

Optionally compare:

```text
chat template compatibility
```

but do not require identical model architecture.

The target and draft can be different architectures if they produce the same token vocabulary.

Example:

```text
Target:
Qwen3-8B

Draft:
Qwen3-0.6B
```

is valid if their tokenizer/vocabulary is compatible.

---

# 8. Core algorithm

The first implementation should use:

```text
draft K tokens
target verifies K tokens
accept/reject
commit accepted KV
rollback rejected KV
```

For example:

```text
Committed target sequence:

A B C D E

Draft proposes:

F G H I J
```

Target evaluates:

```text
F ✓
G ✓
H ✓
I ✗
```

Then:

```text
Committed:

A B C D E F G H
```

and the rejected portion:

```text
I J
```

is discarded.

---

# 9. Draft generation

The draft model generates up to:

```text
MaxDraftTokens
```

tokens.

Example:

```text
MaxDraftTokens = 5
```

The draft generates:

```text
T1 T2 T3 T4 T5
```

unless it encounters:

- EOS
- stop sequence
- grammar restriction
- maximum generation length
- another existing generation termination condition

Do not allow speculative decoding to bypass existing stop conditions.

---

# 10. Draft KV state

The draft model needs its own KV cache.

It must not use the target model's KV cache.

Conceptually:

```text
Session
│
├── TargetState
│     └── TargetKV
│
└── DraftState
      └── DraftKV
```

The draft state should track the same committed token history as the target.

When the target commits accepted tokens:

```text
Target:
commit accepted tokens

Draft:
retain the same token history
```

The draft's KV should therefore remain aligned with the target sequence.

---

# 11. Initialisation

When speculative decoding starts:

```text
Target:
prompt → target KV

Draft:
prompt → draft KV
```

Do not repeatedly process the prompt for every speculation cycle.

Both models must maintain their own session state.

---

# 12. Speculation loop

Conceptually:

```text
while generation not complete:

    draft proposes K tokens

    target verifies proposed tokens

    determine accepted tokens

    commit accepted target KV

    discard rejected speculative target KV

    emit accepted tokens

    if rejection occurred:
        emit correction token if required

    update draft state

    continue
```

This loop should be implemented in a dedicated component.

Suggested name:

```text
SpeculativeDecoder
```

Do not put the entire algorithm into `ContinuousBatchingEngine`.

---

# 13. Suggested internal interfaces

Introduce something conceptually like:

```csharp
public interface ISpeculativeDecoder
{
    ValueTask<SpeculativeResult> DecodeAsync(
        SpeculativeContext context,
        CancellationToken cancellationToken);
}
```

And:

```csharp
public sealed record SpeculativeResult
{
    public required ReadOnlyMemory<int> AcceptedTokens { get; init; }

    public int DraftTokens { get; init; }

    public int AcceptedDraftTokens { get; init; }

    public bool Rejected { get; init; }

    public bool Completed { get; init; }
}
```

Adapt naming to existing Stingray conventions.

---

# 14. Speculative context

Create an internal context containing:

```text
Target session
Draft session
Target KV checkpoint
Draft KV state
Sampling configuration
Maximum speculation length
Current generation position
Stop-condition state
```

Example:

```csharp
internal sealed class SpeculativeContext
{
    public required IModelSession Target { get; init; }

    public required IModelSession Draft { get; init; }

    public int MaxDraftTokens { get; init; }

    public required SamplingOptions Sampling { get; init; }
}
```

---

# 15. KV checkpoint

The target must create a checkpoint before speculative generation.

Conceptually:

```csharp
var checkpoint = targetKv.CreateCheckpoint();
```

Then:

```text
Target committed state
        │
        ▼
Create checkpoint
        │
        ▼
Target speculative tokens
```

After verification:

```text
accepted:
    Commit(checkpoint, acceptedTokens)

rejected:
    Rollback(checkpoint, acceptedTokens)
```

Do not physically copy all KV data.

---

# 16. Required KV operations

The KV abstraction should support operations equivalent to:

```csharp
IKvCheckpoint CreateCheckpoint();

void Commit(IKvCheckpoint checkpoint, int tokenCount);

void Rollback(IKvCheckpoint checkpoint, int tokenCount);
```

Alternatively:

```csharp
sequence.MarkSpeculative();

sequence.CommitSpeculative(tokenCount);

sequence.DiscardSpeculative();
```

The exact API is up to the KV implementation.

The semantics are mandatory.

---

# 17. Partial page handling

If page size is 32 tokens and the speculative sequence crosses a page boundary:

```text
Committed:
0–127

Speculative:
128–139
```

the implementation should allocate only the necessary speculative storage.

If the target accepts all 12:

```text
128–139 become committed
```

If it accepts only 5:

```text
128–132 committed
133–139 discarded
```

Do not copy the entire KV prefix.

---

# 18. Target verification

The target model should evaluate the proposed tokens in one batched forward pass whenever the backend permits.

Example:

```text
Context:
A B C D E

Proposals:
F G H I J
```

Target forward pass should process:

```text
F
G
H
I
J
```

as one verification operation.

Do not perform:

```text
Target(F)
Target(G)
Target(H)
Target(I)
Target(J)
```

individually.

That destroys the primary performance benefit.

---

# 19. Verification logits

The target must produce the probability distribution relevant to each proposed token.

For:

```text
F G H I J
```

the target provides:

```text
P_target(F | context)
P_target(G | context + F)
P_target(H | context + F + G)
P_target(I | context + F + G + H)
P_target(J | context + F + G + H + I)
```

The implementation must correctly align each proposed token with the logits generated immediately before it.

This is a critical correctness area.

---

# 20. Greedy mode

Implement greedy speculation first.

Draft:

```text
argmax(q)
```

Target:

```text
argmax(p)
```

Accept proposed tokens while:

```text
draftToken == targetArgmax
```

At the first mismatch:

```text
targetArgmax
```

becomes the next emitted token.

This version is intentionally simpler.

Do not implement stochastic acceptance in the first milestone.

---

# 21. Greedy algorithm example

Draft:

```text
A B C D E
```

Target predictions:

```text
A B C X Y
```

Result:

```text
A B C X
```

The draft tokens:

```text
A B C
```

are accepted.

`X` comes from the target.

`D` and `E` are discarded.

The next speculation starts after `X`.

---

# 22. Stochastic speculative decoding

After greedy mode is proven, implement proper sampling.

Let:

```text
p(x) = target probability
q(x) = draft probability
```

For proposed token `x`, accept with:

```text
min(1, p(x) / q(x))
```

using the same random source/seed semantics as Stingray's normal sampler.

If rejected, sample the correction token from the residual distribution.

Conceptually:

```text
r(x) = max(0, p(x) - q(x))
```

then normalise:

```text
r(x) / Σ r(x)
```

The implementation must follow the mathematically correct speculative sampling algorithm.

Do not approximate this as token equality.

---

# 23. Sampling configuration

Speculative decoding must respect the same sampling configuration as normal generation.

At minimum test:

```text
temperature
top-k
top-p
seed
EOS
max tokens
stop sequences
```

If Stingray supports additional processors:

```text
repetition penalty
frequency penalty
presence penalty
grammar
logit bias
```

each must be explicitly assessed for speculative compatibility.

Do not silently apply a processor to the target but not the draft if doing so changes the mathematical assumptions of the acceptance algorithm.

---

# 24. RNG handling

This is critical.

The speculative path must not accidentally consume random numbers differently from the normal path in a way that breaks deterministic expectations.

Create a dedicated RNG state for speculative sampling if necessary.

Document:

```text
same seed + same model + same options
```

must produce deterministic results when deterministic mode is requested.

---

# 25. EOS handling

If the draft produces EOS:

```text
stop drafting
```

The target must still verify the EOS proposal.

If target accepts EOS:

```text
generation complete
```

If target rejects EOS:

```text
continue normally
```

Do not allow draft EOS to terminate generation without target verification.

---

# 26. Stop sequences

Stop sequences must be checked against the actual emitted token stream.

Do not allow speculative tokens that are subsequently rejected to alter stop-sequence state.

Only committed tokens update persistent stop-condition state.

---

# 27. Grammar/tool constraints

Initially mark speculative decoding as unsupported when using complex constrained decoding unless compatibility has been demonstrated.

Do not produce subtly incorrect output.

For example:

```text
grammar enabled
        ↓
speculative decoding
        ↓
draft proposes invalid token
```

The system must either:

```text
apply the same constraint correctly to both models
```

or:

```text
fall back to ordinary decoding
```

Fallback is preferable to incorrect generation.

---

# 28. Continuous batching

Do not implement continuous-batch speculative decoding in Phase 1.

First support:

```text
batch size = 1
```

Then extend to multiple requests.

For batch speculation:

```text
Request A → draft 5
Request B → draft 5
Request C → draft 5
```

the target verification should ideally become:

```text
one batched verification
```

but requests may have different accepted lengths.

The scheduler therefore needs to support ragged speculative results.

---

# 29. Ragged acceptance

Example:

```text
Request A:
A B C D E
accepted 5

Request B:
A B X Y Z
accepted 2

Request C:
A Q R S T
accepted 1
```

The scheduler must be able to commit:

```text
A +5
B +2
C +1
```

without forcing all requests to the same length.

This should be treated as a scheduler concern rather than hidden inside the model.

---

# 30. Draft model execution

The draft model should use the same runtime infrastructure where possible.

Do not introduce a separate inference implementation.

For example:

```text
Stingray Target Model
Stingray Draft Model
        │
        ▼
same inference APIs
```

This ensures CPU/CUDA/Vulkan support can eventually apply to both.

---

# 31. Draft model loading

The speculative API should accept an already loaded draft model where possible.

Avoid:

```text
every generation request
    ↓
load draft model
```

Models should be loaded once and reused.

Example:

```csharp
var target = await Stingray.LoadAsync(targetPath);
var draft = await Stingray.LoadAsync(draftPath);

var session = target.CreateSession();

await session.GenerateAsync(
    prompt,
    new SpeculativeDecodingOptions
    {
        DraftModel = draft
    });
```

---

# 32. Memory accounting

Speculative decoding uses two KV caches.

Expose both:

```text
Target KV
Draft KV
```

in diagnostics.

Example:

```text
Target KV:
    2.4 GB

Draft KV:
    0.4 GB

Speculative pages:
    64 MB
```

Do not let speculative memory bypass Stingray's normal memory accounting.

---

# 33. Performance metrics

Add speculative-specific metrics:

```text
Draft tokens generated
Draft tokens accepted
Acceptance rate
Tokens accepted per verification
Verification passes
Draft forward passes
Target forward passes
Fallback count
Speculation windows
Average speculation length
```

Most important:

```text
Acceptance rate
```

and:

```text
Effective target tokens / target forward pass
```

---

# 34. Benchmark output

Add something like:

```text
SPECULATIVE DECODING
────────────────────────────

Target:
    Qwen3-8B Q4_K_M

Draft:
    Qwen3-0.6B Q4_K_M

Speculation:
    5 tokens

Acceptance:
    72.4%

Draft:
    31.2 tok/s

Target:
    5.8 tok/s

Effective:
    10.1 tok/s

Speedup:
    1.74x
```

Compare against:

```text
ordinary target-only generation
```

under identical conditions.

---

# 35. Adaptive speculation

After fixed speculation works, implement adaptive length.

Start with:

```text
K = 5
```

Track recent acceptance rate.

For example:

```text
acceptance > 80%
    → increase K

acceptance < 40%
    → decrease K
```

Do not overcomplicate this initially.

A simple bounded controller is sufficient:

```text
min K = 1
max K = 8
```

Later experiments can determine better values.

---

# 36. Draft/target mismatch

Speculative decoding becomes inefficient if the draft model is poor.

If:

```text
acceptance rate ≈ 0%
```

the system should detect this.

If acceptance remains poor for several windows:

```text
temporarily disable speculation
```

or reduce:

```text
K = 1
```

Do not spend more compute on drafting than it saves.

---

# 37. Automatic fallback

Fallback to normal generation when:

```text
draft model unavailable
tokenizers incompatible
grammar unsupported
sampling configuration unsupported
draft inference failure
target verification failure
acceptance consistently too low
memory pressure too high
```

Fallback must be transparent to the caller.

---

# 38. Error handling

If draft inference fails:

```text
discard speculative state
continue with target-only generation
```

Do not corrupt the target session.

If target verification fails:

```text
discard speculative state
fail or recover according to existing inference error policy
```

Never leave the target KV partially committed.

---

# 39. Transactional KV semantics

Treat each speculation window as a transaction:

```text
BEGIN
   draft
   target verification
   acceptance
COMMIT / ROLLBACK
```

The target's committed state must never contain an unverified speculative token.

This is one of the most important invariants.

---

# 40. Critical correctness invariants

Add tests for:

### Invariant 1 — target equivalence

Speculative greedy decoding must produce exactly the same token sequence as target-only greedy decoding.

```text
target-only:
A B C D E F G

speculative:
A B C D E F G
```

---

### Invariant 2 — rejection

Force a draft model to produce deliberately incorrect tokens.

Verify that the target's output remains identical to target-only decoding.

---

### Invariant 3 — full acceptance

Use identical target/draft models.

Every proposed token should be accepted.

Verify:

```text
accepted == proposed
```

and output matches target-only decoding.

---

### Invariant 4 — zero acceptance

Use a deliberately bad draft model.

Verify that output still matches target-only decoding.

---

### Invariant 5 — KV rollback

After a rejection:

```text
target KV token count
```

must equal the committed token count.

No rejected speculative tokens may remain.

---

### Invariant 6 — repeated speculation

Run hundreds of speculation windows.

Verify that:

```text
KV size
memory usage
token sequence
```

remain correct.

---

### Invariant 7 — page boundaries

Run speculation across:

```text
31 → 32
32 → 33
63 → 64
64 → 65
```

token boundaries.

---

### Invariant 8 — EOS

Test:

```text
draft predicts EOS
target accepts EOS

draft predicts EOS
target rejects EOS
```

---

### Invariant 9 — deterministic seed

Same:

```text
prompt
models
seed
sampling options
```

must produce deterministic output.

---

### Invariant 10 — cancellation

Cancel during:

```text
draft generation
target verification
```

and ensure no speculative KV remains allocated.

---

# 41. Golden-reference testing

For a small model, generate a known sequence using:

```text
target-only decoding
```

Store the expected result.

Then compare:

```text
ordinary decoding
speculative decoding
```

under identical conditions.

This should become part of the regression suite.

---

# 42. Benchmark matrix

At minimum benchmark:

```text
Target:
    small model
    medium model
    larger model

Draft:
    same-family small model

Speculation:
    K=2
    K=4
    K=6
    K=8

Backend:
    CPU
    CUDA if available
    Vulkan if available

Context:
    512
    2048
    8192
```

Measure:

```text
target tok/s
speculative tok/s
speedup
acceptance rate
draft overhead
memory
TTFT
```

---

# 43. CPU-first implementation

The first implementation should target:

```text
CPU
batch size 1
greedy decoding
```

This gives the cleanest environment for validating the algorithm.

Do not optimise SIMD kernels specifically for speculation initially.

The target verification should simply use the existing forward-pass implementation.

---

# 44. CUDA/Vulkan

Once CPU correctness is established:

```text
CUDA
Vulkan
```

should use the same speculative algorithm.

Only optimise:

```text
draft batching
target verification
KV transfers
```

after profiling.

Do not create separate speculative algorithms per backend.

---

# 45. Scheduler integration

After single-sequence speculation works, integrate with:

```text
ContinuousBatchingEngine
```

The scheduler should understand:

```text
request wants speculation
draft model
speculation window
```

but should not implement the acceptance algorithm.

Keep that in:

```text
SpeculativeDecoder
```

---

# 46. Suggested component architecture

Target structure:

```text
Inference
│
├── Generation
│
├── Sampling
│
├── Sessions
│
├── KV
│
└── Speculative
      ├── ISpeculativeDecoder
      ├── SpeculativeDecoder
      ├── SpeculativeContext
      ├── SpeculativeResult
      ├── SpeculativeVerifier
      ├── SpeculativeSampler
      └── SpeculativeMetrics
```

The exact folder/project names should follow existing Stingray conventions.

---

# 47. Do not duplicate sampling logic

If Stingray already has:

```text
Sampler
LogitProcessor
Temperature
TopK
TopP
```

reuse them.

Do not create:

```text
SpeculativeSampler
```

that independently reimplements the entire normal sampling pipeline unless mathematically necessary.

The speculative verifier should integrate with the existing sampling infrastructure.

---

# 48. Do not duplicate model execution

Do not create:

```text
SpeculativeTargetModel
SpeculativeDraftModel
```

as separate model implementations.

Use the existing model execution APIs.

Speculation is an orchestration strategy, not a new model architecture.

---

# 49. Commit semantics

After verification:

### Full acceptance

```text
Draft:
A B C D E

Target accepts:
A B C D E
```

Commit all speculative target KV.

### Partial acceptance

```text
Draft:
A B C D E

Target:
A B C X
```

Commit:

```text
A B C X
```

and discard speculative state corresponding to:

```text
D E
```

The target-generated correction token `X` becomes committed.

---

# 50. Do not emit rejected tokens

The output stream must only expose committed tokens.

Never emit:

```text
draft token
```

before target verification if doing so would make the public output incorrect.

Speculative decoding may internally generate tokens early, but external streaming must remain semantically correct.

---

# 51. Streaming considerations

Normal generation:

```text
generate token
emit token
```

Speculative generation:

```text
draft:
    generate privately

target:
    verify privately

commit:
    emit accepted tokens
```

Therefore a speculation window may create a slightly different latency profile.

Measure:

```text
time to first emitted token
```

and:

```text
tokens/sec after warm-up
```

separately.

---

# 52. Important performance tradeoff

Do not assume speculation is always faster.

Total cost is approximately:

```text
draft cost
+
target verification cost
```

versus:

```text
target cost × number of tokens
```

If the draft is too slow or acceptance is too low:

```text
speculation can be slower than ordinary decoding.
```

The implementation must therefore expose enough metrics to diagnose this.

---

# 53. Acceptance-rate experiment

Create a benchmark that records:

```text
K
accepted
rejected
acceptance %
target passes
draft passes
tokens generated
```

Example:

```text
K=5

Window 1: 5/5
Window 2: 4/5
Window 3: 2/5
Window 4: 5/5

Overall:
16/20 = 80%
```

This should guide adaptive speculation later.

---

# 54. Phase breakdown

## Phase 0 — investigation

Inspect:

```text
session implementation
ISequenceKvCache
new IKvCache implementation
forward pass
sampling
logits
ContinuousBatchingEngine
stop conditions
grammar
```

Produce a dependency map.

Do not modify code.

---

## Phase 1 — KV transaction support

Implement:

```text
checkpoint
commit
rollback
```

using the new KV architecture.

Test independently of speculation.

---

## Phase 2 — draft session

Add the ability to maintain:

```text
target session
draft session
```

with identical token histories.

---

## Phase 3 — greedy speculation

Implement:

```text
draft K
target verify K
accept matching prefix
target correction
commit/rollback
```

No stochastic sampling yet.

---

## Phase 4 — correctness suite

Implement all invariants listed above.

Do not proceed until:

```text
speculative output == target-only output
```

for all deterministic tests.

---

## Phase 5 — stochastic speculation

Implement mathematically correct acceptance/rejection sampling.

Test against target-only statistical distributions.

---

## Phase 6 — metrics

Add:

```text
acceptance rate
speedup
draft cost
target verification cost
```

---

## Phase 7 — adaptive speculation

Implement bounded dynamic K.

---

## Phase 8 — continuous batching

Integrate ragged accepted-token counts.

---

## Phase 9 — backend optimisation

Profile:

```text
CPU
CUDA
Vulkan
```

and optimise only where measured.

---

# 55. Statistical correctness tests

For stochastic speculative decoding, exact token-by-token equality is not necessarily expected.

Instead, run many samples with the same prompt and compare distributions.

For example:

```text
10000 generations
```

Compare:

```text
target-only distribution
```

against:

```text
speculative distribution
```

using an appropriate statistical test/tolerance.

The purpose is to demonstrate that speculation does not materially change the target distribution.

---

# 56. Performance acceptance criteria

Do not require speculation to be faster in every configuration.

For a well-matched target/draft pair:

```text
Acceptance rate:
    preferably > 60%

Effective speedup:
    > 1.2x
```

A good pairing may achieve considerably more.

If:

```text
speedup < 1.0x
```

the benchmark must clearly show that speculation is slower rather than hiding it.

---

# 57. Memory acceptance criteria

Speculation must not cause unbounded KV growth.

After every speculation cycle:

```text
committed KV
+
active speculative KV
```

must match expected token counts.

After rejection:

```text
discarded speculative KV
```

must become reusable.

Stress tests must demonstrate no memory leak.

---

# 58. Failure safety

The strongest invariant is:

> **A failed speculative operation must leave the target session exactly as it was immediately before the speculation window began.**

That means:

```text
target token count unchanged
target KV unchanged
sampling state unchanged where appropriate
stop-condition state unchanged
```

unless the operation successfully committed tokens.

---

# 59. Definition of done

The feature is complete only when:

- [ ] Draft model can be loaded independently.
- [ ] Target and draft tokenizer compatibility is validated.
- [ ] Target and draft maintain independent KV caches.
- [ ] Target speculative checkpoints exist.
- [ ] Speculative KV can be committed.
- [ ] Speculative KV can be rolled back.
- [ ] Draft can generate configurable K tokens.
- [ ] Target verifies K proposed tokens in one forward operation.
- [ ] Greedy speculative decoding works.
- [ ] Target-only and speculative greedy output are identical.
- [ ] Rejection works.
- [ ] Correction tokens work.
- [ ] EOS works correctly.
- [ ] Stop sequences remain correct.
- [ ] Sampling-compatible speculation works.
- [ ] RNG/seed semantics are tested.
- [ ] Unsupported constraints fall back safely.
- [ ] Continuous batching is eventually supported.
- [ ] Acceptance metrics exist.
- [ ] Speedup metrics exist.
- [ ] Memory metrics exist.
- [ ] No rejected KV remains committed.
- [ ] No memory leak occurs after repeated speculation.
- [ ] Page-boundary tests pass.
- [ ] CPU implementation works.
- [ ] CUDA implementation works where supported.
- [ ] Vulkan implementation works where supported.
- [ ] Existing non-speculative generation remains unchanged.
- [ ] NativeAOT build passes.
- [ ] Existing test suite passes.

---

# 60. Most important implementation instruction

**Do not implement speculative decoding as a giant modification to `ContinuousBatchingEngine`.**

Create a separate `SpeculativeDecoder` orchestration layer.

The architecture should be:

```text
                 Generation API
                       │
                       ▼
                SpeculativeDecoder
                 │             │
                 ▼             ▼
             Draft Model    Target Model
                 │             │
             Draft KV       Target KV
                               │
                         checkpoint
                               │
                         verification
                               │
                       ┌───────┴────────┐
                       ▼                ▼
                    COMMIT           ROLLBACK
                       │                │
                       └───────┬────────┘
                               ▼
                          output tokens
```

The speculative decoder decides **what to do**.

The models decide **how to execute inference**.

The KV cache decides **how state is stored, shared, committed and discarded**.

The sampler decides **how tokens are sampled**.

The scheduler decides **when requests run**.

Keep those responsibilities separate.

---

# 61. Final architectural goal

The ultimate Stingray architecture should allow:

```csharp
var target = await Stingray.LoadAsync(targetModel);
var draft = await Stingray.LoadAsync(draftModel);

var session = target.CreateSession();

await foreach (var token in session.GenerateAsync(
    prompt,
    new GenerationOptions
    {
        Speculative = new SpeculativeDecodingOptions
        {
            DraftModel = draft,
            MaxDraftTokens = 5,
            AdaptiveLength = true
        }
    }))
{
    Console.Write(token);
}
```

Internally:

```text
                    Target Session
                         │
                    committed KV
                         │
                  ┌──────┴──────┐
                  │             │
             checkpoint      Draft Session
                  │             │
                  │          draft KV
                  │             │
                  │        propose K tokens
                  │             │
                  └──────┬──────┘
                         ▼
                  Target verifies
                         │
               ┌─────────┴─────────┐
               ▼                   ▼
           accepted             rejected
               │                   │
             commit              discard
               │                   │
               └─────────┬─────────┘
                         ▼
                   next window
```

The critical property is that **speculation becomes a transaction over model state**, not a second copy of the model's context.

That is what makes the feature scalable to:

- paged KV
- session branching
- prefix sharing
- continuous batching
- adaptive speculation
- future speculative methods such as tree-based or multi-token prediction.