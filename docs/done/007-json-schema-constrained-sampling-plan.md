# Implementation Plan — Grammar & JSON Schema Constrained Sampling (`Plan 007`)

Implement a lightweight, incremental **Constraint Engine** for OpenTail.Stingray that can constrain token generation to a supplied JSON grammar or JSON Schema and guarantee that emitted output remains syntactically valid according to the active constraint.

The initial implementation should focus on:

1. JSON syntax constraints;
2. incremental state tracking;
3. token-level admissibility;
4. integration with the existing logits → sampling pipeline;
5. JSON Schema constraints where practical;
6. deterministic failure behaviour when no valid token exists.

The design must remain small and extensible so that general grammar/EBNF constraints can be added later without redesigning the sampling pipeline.

---

# 1. Goal

Allow an inference caller to specify:

```csharp
Constraint = JsonConstraint(...)
```

and guarantee that generated output cannot leave the permitted JSON language.

Conceptually:

```text
model logits
     │
     ▼
ConstraintEngine
     │
     ├── legal token → unchanged
     │
     └── illegal token → -∞
     │
     ▼
sampling
     │
     ▼
valid next token
```

The ConstraintEngine should operate **before sampling**, after the model has produced logits.

---

# 2. Important architectural decision

Do **not** implement this as a simple regex applied to the generated text.

For example, avoid a design like:

```csharp
if (text.EndsWith("{"))
    maskEverythingExcept(...);
```

That will quickly fail for:

- escaped strings;
- Unicode;
- numbers;
- nested objects;
- arrays;
- whitespace;
- partial tokens;
- tokenizer-specific token boundaries;
- multi-character tokens such as `"true"` or `"}]"`.

Instead use an **incremental parser/state machine**.

The core abstraction should conceptually be:

```csharp
public interface IConstraintState
{
    bool IsComplete { get; }

    bool IsValid { get; }

    ConstraintStateResult AcceptToken(
        ReadOnlySpan<int> tokenIds);

    bool IsTokenAllowed(
        int tokenId,
        ReadOnlySpan<float> logits);
}
```

The exact API may be improved after inspecting the existing Stingray sampling/tokenizer abstractions.

**Better existing concepts are explicitly allowed.**

If Stingray already has a natural logits processor, sampler hook, tokenizer abstraction, or generation-state object, integrate there rather than creating duplicate infrastructure.

---

# 3. Core abstractions

Introduce a small constraint layer, preferably in the engine/generation layer where it can be reused by sessions.

Potential structure:

```text
ConstraintEngine
      │
      ├── IConstraint
      │      │
      │      └── JsonConstraint
      │
      ├── JsonState
      │
      └── JsonSchemaConstraint
```

The precise names are not mandatory.

The architectural requirement is:

> **Constraints must be independent of the model itself and operate on token IDs/logits.**

---

# 4. Constraint interface

A useful conceptual contract is:

```csharp
public interface IGenerationConstraint
{
    void Reset();

    bool IsComplete { get; }

    bool IsValid { get; }

    void Apply(
        Span<float> logits,
        ReadOnlySpan<int> candidateTokenIds);

    void AcceptToken(int tokenId);
}
```

However, this may be improved.

A particularly important requirement is that the constraint engine must be able to determine whether a token is valid **without mutating the committed state**.

Conceptually:

```text
current state
     │
     ├── test token 123 → legal
     │                  ↓
     │              temporary state
     │
     └── test token 456 → illegal
```

Only the actually sampled token should commit the state transition.

---

# 5. JSON lexical state machine

Implement JSON syntax using explicit state rather than regex.

At minimum support states conceptually equivalent to:

```text
Start
Value
ObjectStart
ObjectKey
ObjectColon
ObjectValue
ObjectCommaOrEnd
ArrayStart
ArrayValue
ArrayCommaOrEnd
String
StringEscape
StringUnicodeEscape
Number
LiteralTrue
LiteralFalse
LiteralNull
Complete
Invalid
```

The actual state representation can be cleaner than this.

For example:

```csharp
enum JsonParserState
{
    ExpectValue,
    ExpectObjectKeyOrEnd,
    ExpectColon,
    ExpectObjectValue,
    ExpectObjectCommaOrEnd,
    ExpectArrayValueOrEnd,
    ExpectArrayCommaOrEnd,
    InString,
    InStringEscape,
    InUnicodeEscape,
    InNumber,
    InLiteral,
    Complete,
    Invalid
}
```

Use a stack for nested containers:

```text
{
  "items": [
      { ... },
      { ... }
  ]
}
```

The stack needs to remember whether each container is:

```text
Object
Array
```

and what it is currently expecting.

---

# 6. Strings

Strings require special handling.

The engine must distinguish:

```text
"hello"
```

from:

```text
"hello\"world"
```

and:

```text
"\u0041"
```

At minimum support standard JSON escapes:

```text
\"
\\
\/
\b
\f
\n
\r
\t
\uXXXX
```

Inside a JSON string:

```text
```

most punctuation is legal, while an unescaped `"` closes the string.

Do not incorrectly treat a tokenizer token boundary as a JSON character boundary.

A single model token may contain:

```text
"hello"
```

or:

```text
hello",
```

or several JSON characters.

The implementation must therefore reason about the **decoded token text**, not assume one token equals one JSON character.

---

# 7. Numbers

Implement JSON number recognition according to JSON syntax.

The state machine should distinguish:

```text
-
-0
-1
12
12.
12.3
12e
12e+
12e-2
```

and reject malformed forms such as:

```text
01
-
1.
1e
```

unless the next token completes the number legally.

Do not use culture-dependent number parsing.

JSON numbers are language syntax, not .NET locale numbers.

---

# 8. Literals

Support:

```text
true
false
null
```

incrementally.

For example:

```text
t
tr
tru
true
```

Only the correct continuation remains legal at each stage.

Likewise:

```text
f → fa → fal → fals → false
```

and:

```text
n → nu → nul → null
```

---

# 9. Whitespace

Support JSON whitespace:

```text
space
\t
\r
\n
```

outside strings.

Do not accept arbitrary Unicode whitespace as JSON whitespace.

Inside strings, whitespace is ordinary string content.

---

# 10. Token-level constraint application

The critical integration point is the existing generation pipeline.

Find the point conceptually equivalent to:

```text
Forward pass
    ↓
logits
    ↓
sampling
    ↓
selected token
```

Insert:

```text
Forward pass
    ↓
logits
    ↓
ConstraintEngine.Apply()
    ↓
sampling
    ↓
selected token
    ↓
ConstraintEngine.AcceptToken()
```

The constraint engine must not modify model weights or forward-pass behaviour.

---

# 11. Logit masking

Illegal tokens should be assigned negative infinity:

```csharp
logits[tokenId] = float.NegativeInfinity;
```

or use whatever existing masking convention Stingray's sampler uses.

Do not invent a second representation for impossible logits.

Prefer the existing sampler/logit-processing infrastructure if one exists.

---

# 12. Token text lookup

The ConstraintEngine needs to determine what text a token represents.

Use the existing tokenizer interface.

Conceptually:

```csharp
string tokenText = tokenizer.DecodeToken(tokenId);
```

But **do not assume this exact API exists**.

Inspect the repository and use the most efficient existing token-to-text mechanism.

Avoid repeatedly decoding the entire generated sequence.

Ideally:

```text
token ID
   ↓
token text
   ↓
constraint transition
```

---

# 13. Token-prefix handling

This is one of the most important implementation details.

Suppose the tokenizer contains a token representing:

```text
"foo":
```

The constraint engine must determine whether the entire token can be consumed from the current JSON state.

It must not merely inspect the first character.

Conceptually:

```text
Current JSON state
       │
       ▼
candidate token text = "\"foo\":"
       │
       ▼
simulate characters:
    "
    f
    o
    o
    "
    :
       │
       ▼
legal?
```

If every character is legal:

```text
token allowed
```

Otherwise:

```text
token masked
```

---

# 14. Candidate evaluation

Do not necessarily scan the entire vocabulary repeatedly if Stingray already has candidate filtering/top-K sampling.

Prefer the existing sampling architecture:

```text
logits
  ↓
existing candidate selection
  ↓
constraint filtering
  ↓
sampling
```

or, if correctness requires full-vocabulary masking:

```text
logits
  ↓
constraint mask
  ↓
existing sampler
```

Choose whichever integrates cleanly with the current sampler.

**Correctness takes precedence over micro-optimisation for this feature.**

---

# 15. State commit

After the sampler selects a token:

```text
selected token
      ↓
ConstraintEngine.AcceptToken(tokenId)
```

The state should then be committed.

Never mutate committed parser state while merely testing candidate tokens.

Use a temporary/copy state or a reversible transition mechanism.

---

# 16. Completion

When the root JSON value is complete:

```text
Complete
```

Only legal trailing whitespace should be accepted.

A subsequent non-whitespace token must be rejected.

For example:

```json
{"x":1}
```

is complete.

But:

```text
{"x":1} garbage
```

must not be accepted.

---

# 17. Early termination

When the constraint reaches:

```text
IsComplete == true
```

the generation layer should be able to stop naturally if the caller requests constrained completion.

This should not require generating an arbitrary additional token.

Use existing EOS/stop-sequence mechanisms where possible.

Potential behaviour:

```text
JSON complete
     ↓
stop generation
```

rather than:

```text
JSON complete
     ↓
force another token
     ↓
constraint rejects it
```

---

# 18. JSON Schema support

Build JSON syntax first.

Then layer schema constraints on top.

Conceptually:

```text
JSON Schema
     │
     ▼
SchemaConstraint
     │
     ▼
JSON Constraint
     │
     ▼
Token admissibility
```

The schema layer should be responsible for semantic constraints such as:

```json
{
  "type": "object",
  "properties": {
    "name": { "type": "string" },
    "age": { "type": "integer" }
  },
  "required": ["name", "age"]
}
```

It should not reimplement JSON lexical parsing.

---

# 19. Initial JSON Schema subset

Keep the first version deliberately bounded.

Support:

- `type`
- `object`
- `array`
- `properties`
- `required`
- `additionalProperties`
- `items`
- `enum`
- `const`
- basic string constraints where practical
- basic numeric constraints where practical

Do not attempt full JSON Schema immediately.

In particular, defer complex constructs such as:

```text
$ref
oneOf
anyOf
allOf
if/then/else
pattern
unevaluatedProperties
recursive schemas
```

unless the existing architecture makes them trivial.

The objective is **high-value tool-call schemas**, not a complete JSON Schema validator.

---

# 20. Schema object-property state

For an object schema, maintain:

```text
Schema node
     │
     ├── expected properties
     ├── properties already emitted
     ├── required properties remaining
     └── additionalProperties policy
```

Example:

```json
{
  "name": "Alice",
  "age": 42
}
```

After:

```json
{
  "name":
```

the constraint should know that the next value must satisfy:

```text
type = string
```

After:

```json
{
  "name": "Alice",
```

the next property must be one of the schema's permitted properties.

If:

```text
additionalProperties = false
```

unknown property names must be masked.

---

# 21. Required-property handling

When an object is about to close:

```text
}
```

the constraint must reject `}` if required properties remain.

For example:

```json
{
  "name": "Alice"
}
```

must be rejected if:

```json
"required": ["name", "age"]
```

because `age` has not yet appeared.

This is one of the key differences between **valid JSON** and **schema-valid JSON**.

---

# 22. Enum and const

For:

```json
{
  "type": "string",
  "enum": ["red", "green", "blue"]
}
```

the constraint should narrow the permitted string content.

For:

```json
{
  "const": "success"
}
```

only the token sequence representing `"success"` should remain legal.

Do not solve this by generating arbitrary JSON and validating afterwards.

The purpose of the feature is to prevent invalid continuations **during sampling**.

---

# 23. Impossible constraint detection

The engine must detect:

```text
all logits masked
```

rather than passing an all-`-∞` distribution into the sampler.

Return a specific failure such as:

```csharp
ConstraintViolationException
```

or an existing generation error type.

The exception/result should identify:

```text
constraint type
current state
reason
```

without dumping enormous model state.

---

# 24. No-legal-token invariant

Mandatory invariant:

> **At every constrained sampling step, either at least one legal token remains, or generation terminates with an explicit constraint failure.**

Never silently fall back to unconstrained sampling.

This is critical.

A fallback such as:

```csharp
if (allMasked)
    useOriginalLogits();
```

would completely undermine the guarantee.

---

# 25. Interaction with speculative decoding

The existing `SpeculativeDecoder` must remain compatible.

Constraint semantics should apply to speculative tokens too.

Conceptually:

```text
draft tokens
     ↓
constraint validation
     ↓
target verification
     ↓
accepted tokens
```

A draft token that violates the active constraint must not be accepted.

The target model's verification path remains authoritative.

Do not create a second constrained speculative decoder.

Reuse the existing speculative verification pipeline.

If implementing constraint support in speculative decoding is more invasive than expected, make it a clearly isolated follow-up rather than compromising correctness of normal constrained generation.

---

# 26. Interaction with prompt-lookup speculation

The existing prompt-lookup speculation must obey the same rule.

A prompt-derived proposed token is still a proposed generation token.

Therefore:

```text
prompt lookup proposal
        ↓
constraint admissibility
        ↓
verification
```

An invalid proposal must simply be rejected.

Again, do not duplicate constraint logic inside prompt lookup.

---

# 27. Session API

Expose the feature through the existing generation/session options rather than adding a special generation API.

Conceptually:

```csharp
new GenerationOptions
{
    Constraint = JsonConstraint.FromSchema(schema)
}
```

or:

```csharp
session.GenerateAsync(
    prompt,
    new GenerationOptions
    {
        Constraint = ...
    });
```

Use whatever options object already exists.

The constraint belongs to a **generation operation**, not permanently to the model.

---

# 28. Constraint lifetime

Each generation request should get an independent constraint state.

Do not share mutable parser state between:

```text
Session A generation
Session B generation
```

Likewise, if a session performs two independent constrained generations, their constraint states must not accidentally leak into one another.

---

# 29. Cancellation and errors

Constraint state must be discarded cleanly if generation is:

- cancelled;
- aborted;
- fails;
- reaches context limits;
- encounters a model/runtime error.

Do not leave the session permanently constrained after a failed generation.

---

# 30. Tests

Create:

```text
ConstraintEngineTests.cs
JsonConstraintTests.cs
JsonSchemaConstraintTests.cs
```

or follow the repository's existing test organisation.

At minimum implement the following tests.

### Test 1 — SimpleObject_IsValid

Generate:

```json
{"name":"Alice"}
```

and verify completion.

---

### Test 2 — InvalidPunctuation_IsMasked

At a state where `}` is illegal, verify its logit becomes:

```text
-∞
```

---

### Test 3 — NestedObjectsAndArrays

Verify:

```json
{"items":[{"id":1},{"id":2}]}
```

is accepted.

---

### Test 4 — StringEscapes

Verify escaped quotes and backslashes work.

---

### Test 5 — UnicodeEscape

Verify:

```text
\uXXXX
```

processing.

---

### Test 6 — NumberStateMachine

Accept valid:

```text
-12.5e+3
```

and reject malformed numeric continuations.

---

### Test 7 — LiteralStateMachine

Verify:

```text
true
false
null
```

are handled incrementally.

---

### Test 8 — MultiCharacterToken

Create a token representing multiple JSON characters and verify that the entire token is simulated before acceptance.

This test is particularly important.

---

### Test 9 — CompleteRootValue

Once the JSON root is complete:

```text
non-whitespace token → rejected
whitespace → accepted
```

---

### Test 10 — RequiredProperty

Schema requiring:

```text
name
age
```

must reject object completion when `age` is missing.

---

### Test 11 — AdditionalPropertiesFalse

Unknown property names must be rejected.

---

### Test 12 — Enum

Only schema enum values are permitted.

---

### Test 13 — SchemaType

A property declared:

```text
"type": "integer"
```

must not accept a string value.

---

### Test 14 — AllTokensRejected

Verify generation returns an explicit constraint failure rather than sampling an invalid token.

---

### Test 15 — ConstraintStateIsolated

Two simultaneous sessions/generations must not share parser state.

---

### Test 16 — CancellationDoesNotPoisonSession

Cancel constrained generation, then perform an unconstrained generation successfully.

---

### Test 17 — SpeculativeDecoderCompatibility

When speculative decoding is enabled, illegal draft tokens are not committed.

---

### Test 18 — PromptLookupCompatibility

Prompt-lookup speculative proposals still obey the active constraint.

---

# 31. Integration test — real generation

Add at least one integration test using the actual Stingray generation pipeline.

For example:

```text
prompt
  ↓
small test model
  ↓
JSON constraint
  ↓
generation
  ↓
parse output with System.Text.Json
```

The important assertion is:

```csharp
JsonDocument.Parse(output);
```

must succeed.

For schema generation, additionally validate the resulting structure against the expected schema semantics.

Do not rely exclusively on unit tests of the state machine.

---

# 32. Deterministic test mode

Where possible, use greedy generation or a deterministic seed for integration tests.

The constraint engine should be deterministic independently of sampling randomness.

The test should therefore prove:

```text
same logits + same constraint state
        ↓
same legal-token mask
```

---

# 33. Performance expectations

This is **not a performance-tuning project**.

The implementation should nevertheless avoid obviously pathological behaviour.

Do not:

- decode the entire generated sequence every token;
- parse the entire JSON output from scratch after every token;
- clone the complete parser state for every vocabulary token if a lightweight transition mechanism can be used;
- allocate large strings repeatedly.

A small token-text cache is acceptable if the existing tokenizer architecture makes that useful.

Correctness comes first.

---

# 34. Public API documentation

Document clearly:

```text
Constraint
```

means:

> The model is prevented from sampling tokens that violate the active constraint.

Also document the distinction:

```text
JSON constraint
    = syntactically valid JSON

JSON Schema constraint
    = JSON syntax + schema restrictions
```

Do not claim that arbitrary JSON Schema is supported if only the initial subset is implemented.

---

# 35. Acceptance criteria

The implementation is complete when:

- [ ] Constraint abstraction exists.
- [ ] JSON lexical state machine exists.
- [ ] Nested objects/arrays work.
- [ ] Strings and escapes work.
- [ ] Numbers work.
- [ ] `true`, `false`, and `null` work.
- [ ] Multi-character tokenizer tokens are handled correctly.
- [ ] Illegal logits are masked before sampling.
- [ ] Selected tokens update constraint state.
- [ ] Constraint state is never mutated merely by candidate testing.
- [ ] Completed JSON cannot accept trailing non-whitespace.
- [ ] All-masked distributions produce explicit failure.
- [ ] No unconstrained fallback exists.
- [ ] JSON Schema constraints build on the JSON parser rather than duplicating it.
- [ ] `type`, `properties`, `required`, `items`, `additionalProperties`, `enum`, and `const` are supported where implemented.
- [ ] Generation/session API exposes constraints cleanly.
- [ ] Constraint state is isolated per generation.
- [ ] Cancellation/error paths clean up constraint state.
- [ ] Speculative decoding remains correct.
- [ ] Prompt-lookup speculation remains correct.
- [ ] Unit tests pass.
- [ ] Real generation integration test passes.
- [ ] Full Stingray test suite passes.
- [ ] Release build passes.

---

# 36. Definition of done

The feature should make this possible:

```text
OpenTail Agent
      │
      ▼
"Call search tool"
      │
      ▼
GenerationOptions
      │
      └── JSON Schema constraint
                 │
                 ▼
             Stingray
                 │
           model logits
                 │
                 ▼
        ConstraintEngine
                 │
          illegal tokens
             → -∞
                 │
                 ▼
             sampler
                 │
                 ▼
          schema-valid JSON
                 │
                 ▼
          tool invocation
```

The key guarantee is:

> **The model may still choose the wrong semantic answer, but it cannot choose a token sequence that violates the active JSON syntax/schema constraints.**

---

## Implementation flexibility

The API names and code sketches above are **guidance, not a demand for literal implementation**.

The coding AI should inspect the current Stingray architecture first.

If it finds:

- an existing logits processor;
- an existing generation processor pipeline;
- a better tokenizer token-text API;
- an existing generation options object;
- an existing parser/state-machine abstraction;
- a better way to represent immutable constraint state;

it should **use that architecture instead**.

Do not introduce parallel abstractions merely to match this document.

The plan's required outcome is the invariant:

> **No token that violates the active constraint can be sampled or committed.**

The implementation may use a better design to achieve that outcome.