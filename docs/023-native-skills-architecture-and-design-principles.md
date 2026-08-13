# Native Skills, Instructions & Tools — Architecture & Design Principles

## Overview & Core Philosophy

Stingray treats **skills, instructions, and tools as fundamental, first-class inference concepts**, rather than application-level prompt conventions or unstructured text blocks.

Higher-level agent runtimes (such as OpenTail, Microsoft Agent Framework, or custom C# hosts) provide discovery, execution, permissions, and lifecycle management. However, they **do not need to translate a skill into an opaque block of prompt text** merely to use it with Stingray.

```
                           Stingray Engine
                                  │
                       Native Inference Model
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
          Skill              Instruction               Tool
            │                     │                     │
            └─────────────────────┼─────────────────────┘
                                  │
                           Inference Session
                                  │
             ┌────────────────────┼────────────────────┐
             ▼                    ▼                    ▼
          OpenTail         Agent Framework       Custom Host
```

---

## The Downstream Effect

### Without Native Concepts (Fragmented Integration)
```
Agent Framework / Application
     │
     ├── Invents custom skill model
     ├── Converts skill → prompt text
     ├── Converts tools → Stingray tools
     └── Wrestles with session/cache semantics
                    │
                    ▼
                 Stingray
```
*Result: Every integration must build its own translation layer, leading to duplicated prompt-parsing logic, fragile prefix-cache invalidation, and leaky abstractions.*

### With Native Concepts (Unified Engine Surface)
```
                    Stingray Engine
                       │
              Native Inference Model
                       │
        ┌──────────────┼──────────────┐
        │              │              │
      ISkill      IInstruction      ITool
        │              │              │
        └──────────────┼──────────────┘
                       │
                Inference Session
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
      OpenTail    Agent Framework   Custom Host
```
*Result: Runtimes map onto Stingray natively instead of fighting it. Skill instructions participate cleanly in KV prefix caching, and tools drive token-level `ToolGrammar` sampling constraints directly.*

---

## Architectural Boundary & Responsibility Matrix

Stingray explicitly separates **Inference Semantics** (owned by Stingray) from **Operational Semantics** (owned by the Host Application):

| Feature Domain | **Stingray (Inference Semantics)** | **Host Application (Operational Semantics)** |
| :--- | :--- | :--- |
| **Instructions** | Canonical context entry, composition pipeline (`System` → `Skill Instructions` → `Conversation`), prompt prefix hashing | Skill discovery, filesystem reading (`SKILL.md`), user formatting |
| **Tools** | Model callable surface declaration, `ToolGrammar` token-sampling constraints, tool call validation | Tool execution loop, script sandboxing, side-effect management |
| **Skills & Context** | Context identity, token budgeting, Paged KV Cache prefix sharing | Trust models, security policy, auto-approval dialogs |
| **Sessions** | Zero-copy session forking isolation (`Fork()`), KV page lifecycle | Session persistence, agent workflow loops, A2A/MCP protocols |

---

## Core C# Interfaces (`OpenTail.Stingray.Core`)

### `ISkill`
The composition root representing a declarative skill package:
```csharp
public interface ISkill
{
    string Name { get; }
    string? Description { get; }
    IReadOnlyList<IInstruction> Instructions { get; }
    IReadOnlyList<ITool> Tools { get; }
    IReadOnlyList<IResource> Resources { get; }
}
```

### `IInstruction`
Declarative instruction fragment contributed to prompt composition:
```csharp
public interface IInstruction
{
    string Content { get; }
    string? Name { get; }
}
```

### `ITool`
Declarative tool capability definition:
```csharp
public interface ITool
{
    string Name { get; }
    string? Description { get; }
}
```

---

## Key Benefits

1. **Zero-Copy Prefix Cache Reuse**: Attaching a skill generates a deterministic prompt prefix hash. Subsequent queries with identical active skill sets achieve **0 ms prefill recomputation**.
2. **Grammatical Tool Sampling**: Tools declared by attached skills automatically arm `ToolGrammar` token-level constraints, guaranteeing valid tool calls during sampling.
3. **Branching Session Isolation**: Calling `session.Fork()` deep-copies active skill attachments, ensuring child branch sessions can modify skill attachments without mutating parent or sibling state.
