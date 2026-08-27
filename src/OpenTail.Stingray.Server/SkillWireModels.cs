using System.Text.Json.Serialization;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Server;

/// <summary>
/// Wire shape for one <see cref="ITool"/> declared by a <see cref="WireSkill"/>. Shared by every
/// endpoint that accepts a skill: the OpenAI/Anthropic-compat chat routes (folded into the
/// rendered prompt + declared tools) and the raw Sessions API (folded into
/// <see cref="OpenTail.Stingray.Sessions.HotSession.ValidateToolCall"/>'s allow-list only, per
/// docs/051).
/// </summary>
public sealed record WireSkillTool(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("description")] string? Description = null);

/// <summary>Wire shape for one <see cref="IInstruction"/> declared by a <see cref="WireSkill"/>.</summary>
public sealed record WireSkillInstruction(
    [property: JsonPropertyName("content")] string Content,
    [property: JsonPropertyName("name")] string? Name = null);

/// <summary>
/// Wire shape of an <see cref="ISkill"/> passed to a request — e.g. fetched client-side from a
/// skill registry such as skills.sh and forwarded as-is. <c>Instructions</c> become part of the
/// rendered prompt (a system-message segment on the chat-compat routes; see docs/051 for the
/// Sessions API's narrower, deferred-injection handling), and <c>Tools</c> become both
/// declared/callable tools and (on the Sessions API) an authorization allow-list.
/// </summary>
public sealed record WireSkill(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("description")] string? Description = null,
    [property: JsonPropertyName("instructions")] WireSkillInstruction[]? Instructions = null,
    [property: JsonPropertyName("tools")] WireSkillTool[]? Tools = null);

public static class WireSkillExtensions
{
    /// <summary>Converts the wire shape to the native <see cref="Skill"/> record.</summary>
    public static Skill ToCoreSkill(this WireSkill w) => new(
        w.Name,
        w.Description,
        Instructions: (w.Instructions ?? []).Select(i => (IInstruction)new Instruction(i.Content, i.Name)).ToArray(),
        Tools: (w.Tools ?? []).Select(t => (ITool)new Tool(t.Name, t.Description)).ToArray());

    /// <summary>
    /// Concatenates every attached skill's <see cref="IInstruction.Content"/>, in order, separated
    /// by a blank line — the text that becomes a synthetic system-message segment on the
    /// chat-compat routes. Empty when no skill declares any instructions.
    /// </summary>
    public static string JoinInstructionText(this IEnumerable<WireSkill> skills) =>
        string.Join("\n\n", skills
            .SelectMany(s => s.Instructions ?? [])
            .Select(i => i.Content)
            .Where(c => !string.IsNullOrEmpty(c)));
}
