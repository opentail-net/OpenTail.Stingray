namespace OpenTail.Stingray.Core;

/// <summary>
/// Represents a tool capability declared by a skill.
/// Describes tool metadata and schema without owning execution.
/// </summary>
public interface ITool
{
    /// <summary>
    /// Unique name of the tool.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Optional description of what the tool does.
    /// </summary>
    string? Description { get; }
}

/// <summary>
/// Represents an instructional prompt fragment contributed by a skill or context provider.
/// </summary>
public interface IInstruction
{
    /// <summary>
    /// The instructional text content to include in prompt composition.
    /// </summary>
    string Content { get; }

    /// <summary>
    /// Optional name or identifier for the instruction.
    /// </summary>
    string? Name { get; }
}

/// <summary>
/// Represents a declarative resource metadata reference contributed by a skill.
/// </summary>
public interface IResource
{
    /// <summary>
    /// Unique name of the resource.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Content type or MIME type of the resource (e.g. "text/plain", "application/json").
    /// </summary>
    string ContentType { get; }

    /// <summary>
    /// Optional text description of the resource.
    /// </summary>
    string? Description { get; }
}

/// <summary>
/// Native composition root representing a declarative skill package (instructions, tools, resources).
/// <para>
/// <b>Design Principle:</b> Skills, instructions, and tools are treated as first-class inference concepts
/// rather than application-level prompt conventions.
/// </para>
/// <para>
/// <b>Inference Semantics (Stingray):</b> Prompt composition, tool surface/grammar constraints, session prefix-caching, and context budgeting.
/// </para>
/// <para>
/// <b>Operational Semantics (Host):</b> Skill discovery, file loading, trust/permissions, tool execution, and agent orchestration.
/// </para>
/// </summary>
public interface ISkill
{
    /// <summary>
    /// Unique name of the skill.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Optional description of the skill's purpose and capabilities.
    /// </summary>
    string? Description { get; }

    /// <summary>
    /// Instructions contributed by this skill to prompt composition.
    /// </summary>
    IReadOnlyList<IInstruction> Instructions { get; }

    /// <summary>
    /// Tools declared by this skill for model function calling / grammar sampling.
    /// </summary>
    IReadOnlyList<ITool> Tools { get; }

    /// <summary>
    /// Resources declared by this skill.
    /// </summary>
    IReadOnlyList<IResource> Resources { get; }
}

/// <summary>
/// Immutable record implementation of <see cref="ITool"/>.
/// </summary>
public sealed record Tool(string Name, string? Description = null) : ITool;

/// <summary>
/// Immutable record implementation of <see cref="IInstruction"/>.
/// </summary>
public sealed record Instruction(string Content, string? Name = null) : IInstruction;

/// <summary>
/// Immutable record implementation of <see cref="IResource"/>.
/// </summary>
public sealed record Resource(string Name, string ContentType, string? Description = null) : IResource;

/// <summary>
/// Immutable record implementation of <see cref="ISkill"/>.
/// </summary>
public sealed record Skill(
    string Name,
    string? Description = null,
    IReadOnlyList<IInstruction>? Instructions = null,
    IReadOnlyList<ITool>? Tools = null,
    IReadOnlyList<IResource>? Resources = null) : ISkill
{
    public IReadOnlyList<IInstruction> Instructions { get; init; } = Instructions ?? Array.Empty<IInstruction>();
    public IReadOnlyList<ITool> Tools { get; init; } = Tools ?? Array.Empty<ITool>();
    public IReadOnlyList<IResource> Resources { get; init; } = Resources ?? Array.Empty<IResource>();
}
