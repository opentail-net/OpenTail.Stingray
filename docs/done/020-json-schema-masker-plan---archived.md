> **ARCHIVED, 2026-08-15.** Implemented as an `IInferenceSession`/`InferenceSession`-era
> Sessions-layer feature (confirmed against source, not just this document's own claim). That
> whole lineage is being superseded by the `HotSession` architecture — see
> [028](028-inference-session-to-hotsession-migration-plan.md) for current migration status
> (Phases 1-3 done) and [030](../030-delete-inferencesession-todo.md) for the still-open
> deletion of the legacy `InferenceSession`/`InferenceRuntime` types once HotSession fully
> replaces them. Carried forward: nothing beyond what 028/030 already track in
> [00-current-work.md](../00-current-work.md).

---


# Implementation Plan — Dynamic Token-Level JSON Schema Masker

## Objective

Add native constrained JSON generation to Stingray's sampling pipeline.

The feature should allow a caller to provide either:

```csharp
await foreach (var chunk in session.GenerateAsync<ToolCallResult>(sampling))
{
    ...
}
```

or an equivalent JSON Schema/options API, causing Stingray to constrain generation **at token-selection time**.

The sampler must prevent tokens that would make the output impossible to remain valid JSON under the supplied schema.

The intended architecture is:

```text
C# DTO / JSON Schema
        │
        ▼
 Schema Compiler
        │
        ▼
Compact JSON Grammar / State Machine
        │
        ▼
Token Constraint Engine
        │
        ▼
Sampler logits
        │
        ▼
Only valid next tokens remain
        │
        ▼
Generation
```

The implementation should be:

- native C#;
- allocation-conscious;
- tokenizer-aware;
- sampler-integrated;
- reusable across sessions;
- independent of Python;
- independent of an external grammar process;
- optional.

Do **not** turn Stingray into a general-purpose JSON Schema validation framework.

---

# 1. First Inspect Stingray's Existing Sampling Architecture

Before changing anything, inspect:

```text
Sampler
SamplingParams
GenerateAsync()
tokenizer
vocabulary representation
logit buffer
token selection
AllowedChoices
tool calling
GenerationResult
streaming lifecycle
ModelCapabilities
```

The new feature must integrate with the existing sampler rather than creating a second generation pipeline.

In particular, identify the exact point where Stingray has:

```text
logits[tokenId]
```

and before the token is selected.

That is the ideal masking point.

---

# 2. Reuse Existing AllowedChoices Infrastructure

Stingray already has constrained choice functionality.

Do not create another independent "constraint" abstraction if the existing implementation can be generalized.

The desired relationship is:

```text
Sampling constraints
       │
       ├── AllowedChoices
       │
       └── JSON Schema Grammar
```

Both ultimately need to answer:

> **Which token IDs are legal at this generation step?**

If possible, introduce a small internal constraint interface rather than duplicating sampler logic.

For example, conceptually:

```csharp
interface ITokenConstraint
{
    void ApplyMask(...);
    void Advance(int tokenId);
    bool IsComplete { get; }
}
```

Use the actual Stingray naming conventions.

Do not expose unnecessary internals publicly.

---

# 3. Separate Schema Compilation from Token Masking

This is critical.

Do not make the sampler understand JSON Schema directly.

Instead:

```text
JSON Schema
    ↓
Schema Compiler
    ↓
Compiled Grammar
    ↓
Token Constraint State
    ↓
Sampler
```

The sampler should only know:

> "These token IDs are currently legal."

This keeps the hot path small.

---

# 4. JSON Schema Support Scope

Do not attempt full JSON Schema support initially.

Define a deliberately useful subset.

### Initial support should include

```text
type: object
type: array
type: string
type: integer
type: number
type: boolean
type: null

properties
required
additionalProperties
items

enum
const

minimum
maximum
minLength
maxLength

minItems
maxItems
```

Potentially:

```text
anyOf
oneOf
```

can be added if the state-machine architecture naturally supports them.

Do not make complex schema composition a prerequisite for v1.

---

# 5. Explicitly Define Unsupported Features

The compiler must reject unsupported constructs rather than silently pretending to support them.

For example:

```text
$ref
recursive schemas
complex pattern
format
unevaluatedProperties
dynamicRef
custom vocabularies
```

should either be:

- unsupported;
- explicitly handled;
- or represented as non-constraining validation.

Do not silently produce an incorrect grammar.

---

# 6. C# DTO Compilation

Provide a convenient path from a C# type to a schema/grammar.

Conceptually:

```csharp
GenerateAsync<ToolCallResult>(sampling)
```

should obtain a schema representation for `T`.

However, **do not use reflection on every token or every generation**.

Compilation should happen once.

```text
typeof(T)
    ↓
schema metadata
    ↓
compiled grammar
    ↓
generation
```

---

# 7. Schema Compilation Cache

Compiled schemas should be reusable.

Use a cache keyed by something equivalent to:

```text
Type + serializer/schema options
```

or:

```text
canonical schema hash
```

The cache must not accidentally reuse an incompatible grammar.

This is important for OpenTail, where the same DTO may be used repeatedly.

---

# 8. Do Not Promise "<1 ms" Blindly

Make this a performance target rather than a guaranteed architectural property.

Benchmark separately:

```text
schema parsing
schema compilation
grammar construction
token constraint preparation
```

For small schemas, sub-millisecond compilation should be a reasonable target.

Large schemas may legitimately take longer.

The important property is:

> **Compilation happens once, outside the token-generation hot path.**

---

# 9. Tokenizer-Aware Constraint Compilation

This is the technically difficult part.

JSON constraints operate on characters/bytes.

Stingray generates **tokens**.

Therefore the engine must understand:

```text
JSON grammar
        +
tokenizer vocabulary
```

A token is not necessarily:

```text
"{"
```

It may represent:

```text
"{\"name\":\""
```

or arbitrary fragments.

Therefore the compiler must determine which vocabulary tokens are compatible with the current grammar state.

---

# 10. Build Token Transition Tables

Precompute as much as possible.

Conceptually:

```text
Grammar State
      +
Token ID
      ↓
valid / invalid
      +
next grammar state
```

Avoid reparsing the complete token text on every generation step.

For a vocabulary of V tokens and grammar states G, a naive G×V structure may be too large.

Prefer a compact representation.

---

# 11. Avoid a Giant Dense Boolean Matrix

Do not automatically allocate:

```text
GrammarStates × VocabularySize
```

as a full matrix.

For a 128k+ vocabulary this can become wasteful.

Use compact representations such as:

```text
BitVector
sparse token lists
ranges
cached transition sets
```

depending on what works best with Stingray's vocabulary.

The objective is:

```text
small memory footprint
fast membership test
fast mask application
```

---

# 12. Token Mask Application

At each generation step:

```text
logits
  ↓
constraint
  ↓
mask invalid token IDs
  ↓
existing sampling
```

Do not modify the logits for valid tokens.

For invalid tokens, use the same mechanism already used elsewhere for impossible choices, ideally:

```text
logit = -∞
```

or Stingray's existing equivalent.

---

# 13. Preserve Existing Sampling Semantics

JSON masking should occur before token selection but should not otherwise change:

```text
temperature
top-k
top-p
min-p
repetition penalties
seeded randomness
```

The desired ordering should be established from Stingray's current sampler design.

The constraint should narrow the candidate set.

The normal sampler should then choose among legal candidates.

---

# 14. Constraint Ordering

Determine the correct ordering relative to:

```text
repetition penalties
temperature
top-k
top-p
AllowedChoices
```

Do not guess.

The invariant is:

> **An invalid token must never survive to the final sampling decision.**

Tests should verify this regardless of the precise ordering chosen.

---

# 15. JSON Lexical State Machine

The grammar needs to distinguish states such as:

```text
Start
InsideObject
ExpectingPropertyName
InsideString
AfterPropertyName
ExpectingValue
AfterValue
InsideArray
ExpectingArrayValue
AfterArrayValue
Complete
Invalid
```

Do not parse the complete output from scratch on every token.

Maintain incremental state.

---

# 16. String Handling

JSON strings are particularly important.

The engine must correctly handle:

```text
"
\"
\\
\n
\r
\t
\uXXXX
```

and determine whether a token continues a valid string.

A token containing an escaped quote must not accidentally terminate the JSON string.

Test this heavily.

---

# 17. Unicode Handling

JSON string generation must correctly handle Unicode.

Do not assume one generated token corresponds to one Unicode scalar.

The tokenizer may split UTF-8/Unicode sequences in ways that require careful handling.

The implementation should operate on the representation used by Stingray's tokenizer consistently.

---

# 18. Whitespace

JSON permits whitespace in specific lexical positions.

The grammar should allow valid whitespace without requiring the model to choose one exact formatting style.

For example:

```json
{"name":"Bob"}
```

and:

```json
{
  "name": "Bob"
}
```

should both be valid.

Do not over-constrain formatting unnecessarily.

---

# 19. Object Property State

For:

```json
{
  "name": "...",
  "age": 47
}
```

the grammar must know:

```text
which properties have already appeared
which properties remain
whether required properties are satisfied
whether another property is allowed
```

This is where the schema becomes more than ordinary JSON parsing.

---

# 20. Required Properties

At object completion:

```text
required properties ⊆ encountered properties
```

must be true.

For example:

```json
{
  "required": ["name", "age"]
}
```

must not allow:

```json
{"name":"Bob"}
```

to close the object.

The closing `}` token must therefore be masked until required fields are satisfied.

---

# 21. additionalProperties

Respect:

```json
"additionalProperties": false
```

This is particularly useful for agent tool calls.

If the schema says:

```json
{
  "properties": {
    "path": { "type": "string" }
  },
  "additionalProperties": false
}
```

then a generated:

```json
{"path":"x","unexpected":"bad"}
```

must be impossible.

---

# 22. Enum and Const

Integrate existing `AllowedChoices` concepts where possible.

For:

```json
"enum": ["read", "write", "delete"]
```

the grammar should constrain the string content accordingly.

For:

```json
"const": "approved"
```

only the exact value should be legal.

Do not treat enums merely as post-generation validation.

They should constrain token generation.

---

# 23. Numeric Constraints

Be careful here.

Lexical JSON validity and semantic numeric validity are different.

For example:

```json
"minimum": 10
```

cannot always be enforced by simply looking at the next token.

The state machine may need to track the generated numeric value.

Initial implementation should support straightforward numeric constraints where practical.

Do not compromise correctness merely to claim complete JSON Schema support.

---

# 24. Boolean and Null Values

These are relatively easy and should be strongly constrained.

For:

```json
"type": "boolean"
```

only:

```text
true
false
```

should be reachable.

For:

```json
"type": "null"
```

only:

```text
null
```

should be reachable.

---

# 25. Arrays

Support:

```text
items
minItems
maxItems
```

and correctly track:

```text
[
 value,
 value,
 value
]
```

The `]` token must be masked where:

```text
minItems
```

has not yet been satisfied.

---

# 26. Nested Structures

The state machine must support arbitrary practical nesting:

```json
{
  "user": {
    "address": {
      "city": "Ipswich"
    }
  }
}
```

Use a compact stack/state representation.

Do not recursively allocate objects for every generated token.

---

# 27. Zero-Allocation Hot Path

The generation path should avoid allocations.

In particular, do not create:

```text
JObject
JsonDocument
JsonElement
string
List<Token>
```

per token.

The runtime state should be represented using compact structures, spans, arrays, or pooled storage as appropriate.

---

# 28. Schema Compilation Can Allocate

Zero allocation is most important during generation.

It is acceptable for:

```text
schema compilation
grammar construction
cache population
```

to allocate.

Do not distort the architecture simply to make schema compilation allocation-free.

---

# 29. Constraint State Must Be Session/Generation Specific

The compiled schema is reusable.

The mutable grammar state is not.

Correct:

```text
CompiledSchema
       │
       ├── Generation A → ConstraintState A
       ├── Generation B → ConstraintState B
       └── Generation C → ConstraintState C
```

Never share mutable generation state between sessions or parallel branches.

---

# 30. Fork() Integration

This is particularly important for Stingray.

If a session is forked while constrained generation is active, the grammar state must not become accidentally shared and mutated.

Prefer:

```text
Compiled grammar = shared
Constraint state = branch-specific
```

The zero-copy KV architecture should remain unaffected.

---

# 31. Tool Calling Integration

This feature should eventually become the enforcement mechanism behind structured tool calls.

Conceptually:

```text
Tool definition
     ↓
JSON schema
     ↓
JsonSchemaGrammarMasker
     ↓
Sampler
     ↓
valid tool-call arguments
```

This is much stronger than:

```text
model emits JSON
     ↓
host parses
     ↓
host discovers invalid JSON
     ↓
repair/retry
```

The host should still validate the completed result as a defence-in-depth measure.

---

# 32. Structured GenerationResult Integration

Integrate cleanly with the existing `GenerationResult` work.

A constrained generation should report something like:

```text
finish reason = StructuredOutputComplete
```

or the project's equivalent.

Do not invent a parallel result type.

---

# 33. Failure Modes

The engine must detect:

### Impossible schema

```text
schema cannot produce any valid output
```

Fail before generation if detectable.

### No valid token

At some generation step:

```text
valid token set = ∅
```

Do not allow the sampler to select an invalid token.

Return a clear constraint failure.

### Maximum generation length reached

If the model hits the normal generation limit before completing the JSON:

```text
finish reason = Length
```

not "success".

---

# 34. Dead-End Detection

Where practical, detect impossible grammar states before masking.

For example:

```text
required property cannot be generated
```

because the tokenizer cannot express the required literal.

The engine should fail clearly rather than produce an endless sequence of masked tokens.

---

# 35. Tokenizer Compatibility

The feature must work against Stingray's actual tokenizer implementation.

Do not assume:

```text
one character = one token
```

and do not assume every JSON punctuation character has its own token.

The compiler must operate against the model's vocabulary.

---

# 36. ModelCapabilities Integration

If the existing `ModelCapabilities` feature is available, add an appropriate capability such as:

```text
StructuredOutput
JsonSchemaConstraints
```

if that distinction is useful.

Do not claim a model supports constrained generation merely because Stingray can technically mask its logits.

The actual requirement is primarily a runtime/tokenizer capability.

---

# 37. API Design

Keep the public API small.

Possible conceptual shape:

```csharp
var options = new SamplingParams
{
    JsonSchema = JsonSchema.FromType<ToolCallResult>()
};
```

or:

```csharp
await foreach (
    var chunk in session.GenerateAsync<ToolCallResult>(sampling))
{
}
```

Choose whichever integrates most naturally with existing `GenerateAsync()` overloads.

Do not create multiple competing APIs.

---

# 38. Generic GenerateAsync<T>

If:

```csharp
GenerateAsync<T>()
```

is implemented, make its purpose explicit:

> Generate text constrained to the JSON representation of `T`.

Do not imply that Stingray itself performs arbitrary object deserialization unless that is deliberately part of the API.

A clean separation may be:

```text
T
 ↓
JSON schema
 ↓
constraint
 ↓
generated JSON
```

with deserialization occurring after successful completion.

---

# 39. Streaming Semantics

Keep:

```csharp
await foreach (...)
```

streaming exactly as it is.

The constraint operates invisibly during token generation.

Consumers should not need to change their streaming code.

---

# 40. Benchmarking

Measure:

### Schema compilation

```text
small DTO
medium DTO
large nested DTO
```

### Token overhead

Compare:

```text
normal generation
vs
JSON constrained generation
```

Measure:

```text
tokens/sec
CPU time/token
allocations/token
```

### Masking cost

Measure the actual cost of:

```text
apply constraint
```

per token.

The target should be **very small overhead**, not a predetermined "0ms".

---

# 41. Correctness Tests

### Basic object

```json
{"name":"Bob"}
```

### Required properties

Cannot close until all required fields exist.

### Unknown properties

Rejected when `additionalProperties=false`.

### String escaping

Test:

```text
quotes
backslashes
Unicode
newlines
```

### Numbers

Test integer and floating-point output.

### Booleans

Only `true`/`false`.

### Null

Only `null`.

### Arrays

Test min/max items.

### Nested objects

Test multiple levels.

### Enums

Only permitted values.

### Const

Only exact value.

### Whitespace

Pretty and compact JSON.

### Malformed-token boundaries

Test vocabulary tokens that contain multiple JSON characters.

### Impossible schema

Fails cleanly.

### No valid next token

Fails cleanly.

---

# 42. Critical Adversarial Tests

Test cases specifically designed to break naive implementations:

```text
token contains:
    "\""
    "\\"
    "{"
    "}"
    "\":"
    "\",\""
    "\n"
```

Also test tokens that contain:

```text
valid JSON prefix + invalid suffix
```

The engine must not accept a token merely because its beginning is valid.

The **entire token** must be compatible with the grammar transition.

---

# 43. Fork Stress Test

Create:

```text
Root
 ├── Branch A
 ├── Branch B
 ├── Branch C
 └── Branch D
```

Generate constrained JSON independently.

Verify:

- compiled schema is safely shared;
- mutable constraint state is independent;
- branches cannot corrupt one another;
- KV COW remains unaffected.

---

# 44. Tool-Calling Test

Define a tool schema:

```json
{
  "type": "object",
  "properties": {
    "path": {
      "type": "string"
    },
    "recursive": {
      "type": "boolean"
    }
  },
  "required": ["path"],
  "additionalProperties": false
}
```

Verify the model cannot produce:

```json
{
  "path": "...",
  "recursive": true,
  "unexpected": 123
}
```

and cannot terminate the object without `path`.

---

# 45. Security / Safety Invariant

The constraint engine must guarantee:

> **No emitted token may take the generated output outside the compiled grammar.**

However, remember:

> Grammar correctness does not mean the generated content is semantically safe.

For example:

```json
{"path":"../../secret"}
```

can be perfectly valid according to a schema.

Host-level authorization and tool validation remain necessary.

This feature guarantees **structural correctness**, not permission.

---

# 46. Definition of Done

The feature is complete when:

1. JSON Schema can be compiled into a reusable grammar.
2. C# DTOs can produce equivalent constraints.
3. Compilation occurs outside the token-generation hot path.
4. Compiled schemas can be cached.
5. Mutable constraint state is generation-specific.
6. Tokenizer vocabulary is explicitly accounted for.
7. Invalid token IDs are masked before selection.
8. Existing sampling parameters continue to work.
9. JSON objects are constrained correctly.
10. Required properties are enforced.
11. `additionalProperties=false` is enforced.
12. Arrays are constrained.
13. Strings and escapes are handled correctly.
14. Primitive types are constrained.
15. Enum/const values are constrained.
16. Nested structures work.
17. Unsupported schema features fail explicitly.
18. Impossible grammars fail cleanly.
19. Zero-valid-token states fail cleanly.
20. Forked sessions remain independent.
21. Tool-call arguments can use the mechanism.
22. Streaming `GenerateAsync()` remains intact.
23. Generation results correctly report completion/failure.
24. Hot-path allocations remain effectively zero.
25. Token-generation overhead is benchmarked.
26. Existing Stingray tests remain green.
27. Adversarial tokenizer-boundary tests pass.

---

# Non-Goals

Do NOT:

- implement the entire JSON Schema specification in v1;
- build an external Python dependency;
- invoke an external grammar engine;
- build a general JSON parser into the sampler;
- allocate JSON DOM objects per token;
- modify KV-cache architecture;
- modify session persistence;
- create a separate generation loop;
- replace the existing sampler;
- claim zero latency without benchmarking;
- claim arbitrary C# types are automatically supported without schema limitations;
- treat grammar validity as authorization or security validation.

---

# Architectural Principle

The final design should be:

```text
                 JSON Schema / C# DTO
                          │
                          ▼
                 ┌─────────────────┐
                 │ Schema Compiler │
                 │  cold path      │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Compiled Grammar│
                 │ reusable/cache  │
                 └────────┬────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
          Session A   Session B   Session C
          state A     state B     state C
              │           │           │
              └───────────┼───────────┘
                          ▼
                     Tokenizer
                          │
                          ▼
                        logits
                          │
                          ▼
                JsonSchema Masker
                          │
                    mask invalid
                       tokens
                          │
                          ▼
                    Normal Sampler
                          │
                          ▼
                       token
```

The **compiler is the cold path**.

The **constraint state is tiny and generation-specific**.

The **mask operation is the hot path**.

And the sampler remains the single place where the final token decision happens.

## Most important implementation rule

Do not build this as:

```text
generate token
    ↓
parse JSON
    ↓
validate schema
    ↓
if invalid → repair/retry
```

That defeats the entire purpose.

Build it as:

```text
current grammar state
        +
candidate vocabulary
        ↓
legal token set
        ↓
mask logits
        ↓
normal sampling
        ↓
advance grammar state
        ↓
next token
```

That is the part that could make this a particularly good Stingray feature: **structured output becomes a property of the inference engine rather than another outer orchestration loop.**

And for OpenTail specifically, this pairs extremely well with the tool-context model you've been building: the host can expose the *right tool schema at the right moment*, and Stingray can enforce that the model's resulting arguments are structurally incapable of escaping that schema.