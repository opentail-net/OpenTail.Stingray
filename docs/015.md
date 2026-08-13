# Implementation Plan — Contextual Tool Contract & Harness-Style Tool Loop

## Objective

Evolve Stingray's tool-calling architecture into a **small, MCP-aligned, context-aware tool capability system**, while borrowing the best architectural ideas from Microsoft's Agent Framework Harness.

The implementation **must not become an agent framework** and must not pull Microsoft Agent Framework into Stingray.

The goal is:

> **Stingray owns the model-facing tool contract and tool-call continuation. The host/application owns tool discovery, execution, permissions and policy.**

This gives each inference session only the tools appropriate to its current context, rather than exposing the entire tool universe to the model.

Microsoft's Harness provides useful principles around multi-step execution, tool approval, context providers and scoped tools. We should **borrow those principles without importing the Harness itself**.

---

# 1. Establish the Canonical Tool Model

Create or consolidate a small internal tool contract in Stingray.

Do **not** create an elaborate MCP object hierarchy.

The model should follow the current MCP tool shape, particularly the modern MCP schema model:

```text
ToolDefinition
 ├── Name
 ├── Title             optional
 ├── Description       optional
 ├── InputSchema       JSON Schema
 ├── OutputSchema      optional
 └── Annotations       optional
       ├── ReadOnlyHint
       ├── DestructiveHint
       ├── IdempotentHint
       └── OpenWorldHint
```

A possible C# representation:

```csharp
public sealed record ToolDefinition(
    string Name,
    string? Description,
    JsonElement InputSchema,
    JsonElement? OutputSchema = null,
    ToolAnnotations? Annotations = null,
    string? Title = null);
```

Tool calls:

```csharp
public sealed record ToolCall(
    string Id,
    string Name,
    JsonElement Arguments);
```

Tool results:

```csharp
public sealed record ToolResult(
    string ToolCallId,
    JsonElement Content,
    bool IsError = false);
```

### Important

Do **not** put executable handlers inside `ToolDefinition`.

For example, do not make it:

```csharp
Func<...> Handler
```

The definition describes a capability. It does not execute it.

This keeps Stingray independent of whether the implementation eventually comes from:

* MCP
* OpenTail plugins
* C# functions
* HTTP
* databases
* remote APIs
* another mechanism

If equivalent types already exist in the repository, **reuse or refactor them rather than creating duplicates**.

---

# 2. Introduce `IToolProvider`

Introduce a small abstraction representing the tools available to an inference context.

Conceptually:

```csharp
public interface IToolProvider
{
    IReadOnlyList<ToolDefinition> GetTools(
        InferenceToolContext context);
}
```

However, this signature is **not mandatory**.

The coding agent should inspect the existing architecture and choose the cleanest idiomatic C# design.

The important concept is:

> `IToolProvider` supplies the **currently permitted capabilities** to an inference session.

It is **not** an MCP client.

---

# 3. Add Contextual Tool Selection

Introduce a lightweight context mechanism.

For example:

```csharp
public sealed record InferenceToolContext(
    string? Task,
    string? Mode,
    IReadOnlySet<string>? AllowedTools = null);
```

But again, do not blindly add this exact class if Stingray already has an appropriate context/session concept.

The key pipeline is:

```text
Tool universe
      ↓
context / policy
      ↓
tools exposed to this session
      ↓
model
```

### Example: Code Review

```text
read_file       ✓
search_code     ✓
git_diff        ✓
compile         ✓
run_tests       ✓

write_file      ✗
git_commit      ✗
git_push        ✗
shell           ✗
```

### Example: Implementation

```text
read_file       ✓
search_code     ✓
write_file      ✓
compile         ✓
run_tests       ✓

git_push        ✗
```

### Example: Release

```text
read_file       ✓
write_file      ✓
compile         ✓
test            ✓
commit          ✓
push            ✓
publish         ✓
```

**The model should not receive definitions for tools it is not currently permitted to use.**

Do not simply expose the entire tool universe and rely on model judgement.

---

# 4. Make Tool Providers Composable

Allow multiple providers without making Stingray responsible for their implementations.

Conceptually:

```text
IToolProvider
     │
     ├── OpenTailToolProvider
     ├── McpToolProvider       ← future, outside Stingray
     ├── LocalToolProvider
     └── ContextFilteredProvider
```

A simple composite provider may be useful:

```text
CompositeToolProvider
    ├── Provider A
    ├── Provider B
    └── Provider C
```

Then:

```text
Providers
   ↓
all available tools
   ↓
context policy
   ↓
deduplicate
   ↓
allowed tools
   ↓
InferenceSession
```

Do **not** introduce a dependency-injection framework merely to accomplish this.

Normal interfaces and composition are sufficient.

---

# 5. Integrate Tools at the Session Boundary

The tool set belongs to the **InferenceSession**, not globally to the model.

Possible designs include:

```csharp
session.ToolProvider = provider;
```

or:

```csharp
runtime.CreateSession(
    ...,
    tools: provider.GetTools(context));
```

The coding agent should choose the API that best fits the existing session architecture.

The invariant is:

> Two sessions using the same model must be able to have completely different tool sets.

For example:

```text
Session A
  Context = CodeReview
  Tools = read/search/compile/test

Session B
  Context = Implementation
  Tools = read/search/write/compile/test
```

---

# 6. Make Tool-Call Continuation First-Class

Borrow the best architectural idea from Microsoft's Harness:

> The tool loop should be a first-class continuation of inference, without making Stingray responsible for executing the tool.

The lifecycle should be:

```text
Generate
   ↓
Model emits ToolCall
   ↓
ToolCall returned to host
   ↓
HOST EXECUTES TOOL
   ↓
ToolResult supplied to Stingray
   ↓
same InferenceSession
   ↓
Generate continuation
```

Stingray should expose structured:

```text
ToolCall
```

and:

```text
ToolResult
```

events/data.

The host executes the actual tool.

Then the result is injected back into the same session.

This is particularly important because the existing Stingray session/KV architecture should allow the model to continue without unnecessarily rebuilding its inference state.

---

# 7. Add a Lightweight Loop-Round / Turn Concept

A small round/turn abstraction may improve the API.

For example:

```csharp
public sealed record InferenceRound(
    ...);
```

or:

```csharp
public sealed record InferenceTurn(
    ...);
```

Potential information:

```text
InferenceRound
 ├── generated content
 ├── ToolCalls
 ├── FinishReason
 └── optional metadata
```

Do **not** create a large `AgentLoop` framework.

The coding agent should inspect the existing generation result and streaming types and **extend them rather than creating parallel concepts**.

The important lifecycle is:

```text
Round 1
  Model → ToolCall

Round 2
  ToolResult → Model → ToolCall

Round 3
  ToolResult → Model → final answer
```

This should work naturally with:

* session persistence
* session branching
* speculative decoding
* constrained sampling
* streaming
* future agent orchestration

---

# 8. Feed Tool Schemas Into the Existing ConstraintEngine

Do **not** create a second tool-validation or grammar system.

The intended pipeline is:

```text
IToolProvider
      ↓
ToolDefinition
      ↓
JSON Schema
      ↓
ConstraintEngine
      ↓
constrained token generation
      ↓
ToolCall
```

The tool's `InputSchema` is the source of truth.

If the current ConstraintEngine supports only a subset of JSON Schema:

1. Preserve the original schema.
2. Compile the supported subset.
3. Fail clearly for unsupported constructs.
4. Do not attempt to implement the entire JSON Schema specification as part of this plan.
5. Leave a clean extension point for future schema support.

---

# 9. Borrow Harness-Style Approval Principles

Do not implement Microsoft's complete approval framework.

Borrow the principle:

```text
Tool
  ↓
Policy
  ↓
Allowed automatically?
  ├── yes → execute
  └── no  → approval required
```

Tool metadata should allow the host to distinguish between:

### Safe/read-only

```text
read_file
search
git_diff
```

### Potentially dangerous

```text
write_file
execute_command
git_commit
git_push
delete
send_email
```

The host/OpenTail should ultimately decide whether approval is required.

Stingray should **not** own the approval UI or policy engine.

Tool annotations such as read-only/destructive hints can help the host make that decision, but they must not be treated as a security boundary.

---

# 10. Security Invariant: Tool Availability Must Be Enforced

The model-facing tool list is not sufficient protection.

If a tool is not present in the current capability set:

```text
ToolDefinition
    ↓
NOT EXPOSED
```

And if the model somehow manually emits a call to that tool anyway:

```text
ToolCall
    ↓
capability validation
    ↓
REJECT
```

The host must never execute a tool merely because the model emitted its name.

Required invariant:

> **A tool call is executable only if the tool belongs to the capability set currently authorised for that inference context.**

---

# 11. Add Mandatory Tests

Add/extend tests covering the new boundary.

### `ToolProvider_ReturnsContextAppropriateTools`

A review context exposes only review-appropriate tools.

### `ToolProvider_ExcludesWriteToolsFromReview`

Verify write/destructive tools are absent.

### `DifferentSessions_CanHaveDifferentToolSets`

Two sessions using the same model can have independent capability sets.

### `ToolCall_UsesCanonicalShape`

Verify:

```text
id
name
arguments
```

are correctly represented.

### `ToolResult_CorrelatesByToolCallId`

A result cannot accidentally attach to another tool call.

### `ToolSchema_PreservedThroughProvider`

Verify the schema survives provider → Stingray without lossy conversion.

### `ToolCall_Continuation_PreservesSessionState`

After a tool result, generation continues using the existing session/KV state.

### `DisallowedTool_CannotBeExecuted`

A manually fabricated call to an unavailable tool is rejected.

### `ContextChange_ChangesVisibleTools`

Changing from:

```text
Review → Implementation
```

changes the available capability set appropriately.

### `EmptyToolSet_BehavesExactlyAsBefore`

Existing sessions with no tools must behave exactly as they did previously.

---

# 12. Keep MCP Outside Stingray

This plan intentionally does **not** implement:

* MCP client
* MCP server
* MCP transport
* MCP authentication
* MCP session management
* MCP tool execution

Those belong in OpenTail's **Connectors & Plugins** layer.

The relationship becomes:

```text
                     OpenTail
┌──────────────────────────────────────────────────┐
│                                                  │
│ MCP / Plugins / Connectors                       │
│ Tool discovery                                   │
│ Tool execution                                   │
│ Approval                                         │
│ Security / policy                                │
│                                                  │
│              IToolProvider                       │
└──────────────────────┬───────────────────────────┘
                       │
                 permitted tools
                       │
                       ▼
┌──────────────────── OpenTail.Stingray ───────────┐
│                                                  │
│                InferenceSession                  │
│                       │                          │
│                ToolDefinitions                   │
│                       │                          │
│                 ConstraintEngine                 │
│                       │                          │
│                     Model                        │
│                       │                          │
│                    ToolCall                      │
│                       │                          │
│             ToolResult injection                 │
│                       │                          │
│                continued generation              │
│                                                  │
│                existing KV state                 │
└──────────────────────────────────────────────────┘
```

This means OpenTail can eventually say:

> "Here are 12 tools from these MCP servers, but only 4 are permitted in this context."

Stingray does not need to know that MCP exists.

---

# 13. No Microsoft Agent Framework Dependency

**Do not reference or depend on Microsoft Agent Framework.**

Use it as architectural inspiration only.

Specifically borrow:

* contextual tool scoping
* approval concepts
* multi-round tool loops
* context providers
* capability reduction
* separation between agent orchestration and model execution

Do not import:

* Harness
* Agent Framework
* its dependency graph
* its memory system
* its planning system
* its orchestration framework

---

# 14. Minimise New Code and Dependencies

Before adding anything, inspect the current repository.

If equivalent types already exist:

```text
ToolDefinition
ToolCall
ToolResult
ToolSchema
ToolSchemaCompiler
ToolGrammarHelper
ForcedToolCallConstraint
```

**consolidate them rather than duplicating them.**

Expected new conceptual surface should remain small:

```text
IToolProvider
InferenceToolContext
canonical tool metadata
context filtering
tool-call continuation
```

No new heavyweight package should be required.

No dependency-injection framework should be introduced.

No new agent framework should be introduced.

---

# 15. Suggested File/Type Changes

The coding agent should determine exact paths from the current repository, but the likely shape is:

```text
src/OpenTail.Stingray.Engine/
    IToolProvider.cs
    ToolDefinition.cs
    ToolCall.cs
    ToolResult.cs
    InferenceToolContext.cs

    [existing]
    ConstraintEngine
    ToolSchemaCompiler
    ToolGrammarHelper

src/OpenTail.Stingray.Sessions/
    [modify]
    InferenceSession.cs
    [modify]
    InferenceRuntime.cs
    [modify]
    generation result / streaming types

tests/
    ToolProviderTests.cs
    ToolCapabilityTests.cs
    ToolContinuationTests.cs
```

Do not create files if existing types already provide the required responsibility.

---

# 16. Final Architecture Goal

The final system should provide:

```text
                    TOOL UNIVERSE
                         │
             ┌───────────┴───────────┐
             │                       │
        MCP Servers              OpenTail
             │                    Plugins
             └───────────┬───────────┘
                         │
                    Tool Providers
                         │
                         ▼
                 Context / Policy
                         │
                ┌────────┴────────┐
                │ permitted tools │
                └────────┬────────┘
                         │
                         ▼
                  InferenceSession
                         │
                    Tool schemas
                         │
                  ConstraintEngine
                         │
                         ▼
                       MODEL
                         │
                      ToolCall
                         │
                         ▼
                     OpenTail
                         │
                     execution
                         │
                    ToolResult
                         │
                         ▼
                  same session
                         │
                         ▼
                    continuation
```

## Core Principle

> **The model should see capabilities, not infrastructure.**

MCP can provide the universe of available capabilities.

OpenTail determines which capabilities are appropriate and authorised.

Stingray provides those capabilities to the model, constrains generation against their schemas, represents tool calls, accepts tool results, and continues inference using the existing session state.

---

## Implementation Guidance to the Coding AI

Do not implement this plan mechanically.

**Inspect the existing Stingray code first.**

In particular, look for existing:

* tool definitions
* tool schemas
* tool-call representations
* generation result types
* streaming events
* session input/output APIs
* constraint compilation
* forced tool calling
* existing provider/extension patterns

Where an existing abstraction already fits, extend it.

Where a better C# design naturally emerges from the existing architecture, **use the better design**. `IToolProvider` is the intended architectural concept, not a demand to force a particular method signature.

Avoid speculative infrastructure.

Avoid broad refactoring.

Avoid performance work.

Keep the implementation **small, composable, dependency-free and backwards compatible**.

The desired result is not a mini agent framework.

It is a **clean capability boundary around Stingray's inference loop** that can later be fed by MCP, OpenTail plugins, local functions or other tool systems without changing the inference engine again.

Notes:

We are building for/towards MCP 2.0 standard.

MCP should be "off" by default initially, but with parameter / setting to swtch it on.

NOTE: Microsoft Agent framework has been downloaded here so you can see it in detail - C:\Git-Public\OpenTail.Stingray\examples\agent-framework - it may help to understand the best way to implement some features - you can use it for research, and to copy anything you need (it's MIT licenced)