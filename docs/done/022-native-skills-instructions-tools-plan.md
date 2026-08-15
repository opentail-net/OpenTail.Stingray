# Plan 013 — Native Skills, Instructions & Tools

## Objective

Introduce first-class `ISkill`, `IInstruction`, `ITool`, and resource abstractions into Stingray's inference layer so that a skill can contribute **instructions, executable tools, and contextual resources** to an inference session.

## Design Principle

Skills, instructions and tools are treated as **first-class inference concepts** rather than application-level prompt conventions.

Higher-level agent runtimes (OpenTail, Agent Framework, custom C# hosts) may provide discovery, execution, permissions, and lifecycle management, but they should not need to translate a skill into an opaque block of prompt text merely to use it with Stingray.

**Stingray owns the inference semantics of these concepts:**
- How instructions enter the effective context
- How tools enter the model's callable surface / grammar constraints
- How skills affect context identity
- How changes interact with sessions
- How skills interact with prefix/KV caching
- How their token cost participates in context budgeting

**The host owns the operational semantics:**
- Where skills come from
- Whether a skill is trusted
- How tools are executed
- Permissions and approval
- Persistence and installation

---

## 1. Design the Native Abstraction

Introduce:

```text
ISkill
IInstruction
ITool
IResource


with ISkill acting as the composition root.

Conceptually:

ISkill
 ├── IInstruction[]
 ├── ITool[]
 └── IResource[]

A skill therefore represents a coherent capability package, rather than merely a prompt.

Example:

SQL Analysis Skill
 ├── Instructions
 │    └── SQL analysis guidance
 │
 ├── Tools
 │    ├── execute_sql
 │    └── inspect_schema
 │
 └── Resources
      └── SQL Server reference
2. IInstruction

Create the smallest useful abstraction.

public interface IInstruction
{
    string Content { get; }
}

Initially avoid trying to model every possible agent-framework role.

The important property is that Stingray can distinguish instructional material from ordinary user conversation/context.

Potential metadata can be added where justified:

public interface IInstruction
{
    string Content { get; }
    string? Name { get; }
}

Do not prematurely introduce provider-specific concepts such as OpenAI system or developer roles unless the existing prompt architecture requires them.

3. ITool

Stingray already has tool-related concepts, so reuse existing tool abstractions wherever possible.

Do not create a competing tool system simply to make the new interfaces symmetrical.

The target should be approximately:

ITool
 ├── Name
 ├── Description
 └── Schema

The schema must remain compatible with Stingray's existing structured-output / tool-call / grammar machinery.

Important

ITool describes a tool.

It does not necessarily execute it.

Execution belongs to the caller/host.

Model
  ↓
tool call
  ↓
ITool definition
  ↓
host executes tool
  ↓
result returned to inference
4. IResource

Introduce a minimal resource abstraction for material that a skill can make available to inference.

Do not build a complete retrieval system into this plan.

For example:

public interface IResource
{
    string Name { get; }
    string ContentType { get; }
}

The resource should be capable of being resolved into content by the inference host/context layer.

This leaves room for:

file
document
URL
database result
memory record
generated context

without forcing Stingray to own those storage systems.

5. ISkill

Create the central abstraction:

public interface ISkill
{
    string Name { get; }

    string? Description { get; }

    IReadOnlyList<IInstruction> Instructions { get; }

    IReadOnlyList<ITool> Tools { get; }

    IReadOnlyList<IResource> Resources { get; }
}

Keep it declarative.

An ISkill should describe what it contributes to inference.

It should not:

execute scripts
access arbitrary files
perform network operations
request permissions
mutate the engine
manage sessions
6. Skill Attachment to Inference

Add a mechanism for attaching skills to an inference context/session.

Conceptually:

session.AttachSkill(skill);
session.DetachSkill(skill.Name);

or, preferably, if the existing session API suggests a cleaner design:

InferenceRequest
    .Skills

The implementation should follow Stingray's existing session architecture rather than creating a second session-state mechanism.

Required Behaviour

An inference operation should be able to determine:

Active skills
Active instructions
Active tools
Available resources
7. Instruction Composition

Create a deterministic composition pipeline:

System/context instructions
        +
Skill instructions
        +
Conversation
        +
User input

Do not simply concatenate strings throughout the codebase.

Create one canonical context-composition path.

For example:

ISkill[]
   ↓
IInstruction[]
   ↓
Instruction composer
   ↓
Inference context
   ↓
tokenisation

This becomes important later for prefix caching.

8. Tool Composition

When skills are attached:

Skill A
 ├── Tool A
 └── Tool B

Skill B
 └── Tool C

the inference request should expose:

A
B
C

to the model through Stingray's existing tool-call infrastructure.

Collision Handling

Define deterministic behaviour for:

Two skills expose "execute_sql"

Do not silently choose one.

Possible initial rule:

duplicate tool name → validation error

Later, namespacing can be added:

sql.execute
database.execute
9. Resource Handling

Do not automatically inject every resource into the prompt.

This is important.

A skill may contain:

50 reference documents

but that doesn't mean all 50 belong in context.

The initial resource contract should allow the host to decide which resources are materialised.

Therefore:

ISkill
 └── IResource[]
          │
          ▼
    Host/resource layer
          │
          ▼
   selected content
          │
          ▼
      inference

This keeps the door open for indexing/retrieval later.

10. Prefix-Cache Integration

This is one of the main reasons to make skills native.

Skill instructions can form part of the stable prompt prefix:

System
 +
Skill A instructions
 +
Skill B instructions
 +
Tool definitions
 +
conversation

The implementation should ensure that changing the active skill set changes the effective prompt identity.

For example:

Skill SQL v1

must not accidentally reuse a prefix generated for:

Skill SQL v2

The implementation should integrate with Stingray's existing prefix-cache/session mechanisms rather than creating a skill-specific cache.

11. Session Semantics

Define what happens when a skill is attached to an existing session.

Attach before generation
Session
 ↓
Attach skill
 ↓
Generate

Straightforward.

Attach after generation
Session
 ↓
Generate
 ↓
Attach skill
 ↓
Generate

Must correctly account for the newly introduced instructions/tools and invalidate/rebuild any incompatible prompt prefix.

Detach

Likewise:

Attach SQL skill
Generate
Detach SQL skill
Generate

must not accidentally retain the old skill instructions in the effective context.

12. Forked Sessions

This is especially important for Stingray's session model.

Test:

Base Session
     │
     ├── Child A + SQL skill
     │
     └── Child B + C# skill

The two sessions must have independent effective skill sets.

A fork should inherit skill state according to the same semantics as the rest of session state.

13. Serialization / Diagnostics

Add diagnostic visibility.

A session inspection should be able to report something like:

Session
 ├── Model: DeepSeek-V2-Lite
 ├── Context: 4,821 tokens
 ├── Skills:
 │    ├── sql-analysis
 │    └── coding
 ├── Tools:
 │    ├── execute_sql
 │    ├── inspect_schema
 │    └── search_code
 └── Instructions:
      ├── sql-analysis
      └── coding

This will be extremely useful when debugging local agents.

14. Validation

Introduce validation before inference.

Skill
Name present
No duplicate skill identity
Instruction
Content valid
Tool
Name valid
Schema valid
No duplicate names
Resource
Name valid
Content type valid

Errors should occur before generation, not halfway through a request.

15. No Execution Responsibility

Explicitly document this boundary:

ISkill
  │
  ├── describes instructions
  ├── describes tools
  └── describes resources

but:

ITool
  ↓
does NOT execute itself

The host owns execution.

This lets Stingray remain usable in:

standalone applications
OpenTail
test harnesses
other agent frameworks
custom C# applications

without dragging an agent runtime into the engine.

16. Backwards Compatibility

Existing inference calls must continue to work:

engine.GenerateAsync(prompt);

should remain valid.

Skills are additive:

engine.GenerateAsync(
    new InferenceRequest
    {
        Skills = [...]
    });

or whatever request abstraction already exists.

No existing caller should have to create a dummy skill/context object.

17. Testing

Create dedicated tests for:

Instruction
one instruction
multiple instructions
deterministic ordering
Skill
empty skill
instruction-only skill
tool-only skill
resource-only skill
mixed skill
Tools
tool exposure
duplicate names
schema validation
tool-call generation
Session
attach before generation
attach after generation
detach
fork
persistence if supported
Cache
skill prefix reuse
skill change invalidates incompatible prefix
identical skill set reuses prefix where appropriate
Isolation
Session A + Skill X
Session B + Skill Y

must never leak instructions/tools between sessions.

18. Reference Implementation

Create one very small built-in test skill:

EchoSkill

For example:

Name:
    echo

Instruction:
    Always prefix your response with "ECHO:"

Tool:
    echo_value(value)

This gives you a deterministic end-to-end test of:

ISkill
   ↓
IInstruction
   ↓
ITool
   ↓
Inference request
   ↓
model

without needing MCP, filesystem skills or another framework.

19. Do NOT Implement Yet

Explicitly leave these for future plans:

SKILL.md discovery
Skill installation
Skill marketplace
Script execution
Sandboxing
Permissions
MCP skill adapters
Microsoft Agent Framework adapter
A2A
Skill retrieval/indexing
Automatic skill selection by the model

Those can all consume ISkill later.

Definition of Done

The plan is complete when:

 ISkill exists as a native Stingray abstraction.
 IInstruction exists as a native Stingray abstraction.
 ITool integrates with the existing tool infrastructure rather than duplicating it.
 IResource exists as a lightweight resource abstraction.
 A skill can contribute instructions, tools and resources.
 Skills can be attached to inference/session context.
 Skill instructions participate in canonical prompt composition.
 Skill tools participate in existing tool-call/grammar machinery.
 Resources aren't blindly injected into context.
 Skill changes correctly interact with prefix caching.
 Skill state behaves correctly across session forks.
 Skill/tool collisions are deterministic.
 Existing inference APIs remain backwards compatible.
 No skill-specific execution/security framework is introduced.
 End-to-end tests demonstrate a real skill influencing inference.
 Documentation clearly defines the boundary between Stingray's native skill model and an external skill runtime.