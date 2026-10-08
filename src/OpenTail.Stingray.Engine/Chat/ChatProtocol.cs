#nullable enable

using System.Text;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Request payload passed to a chat protocol's prompt renderer.
/// </summary>
public sealed record ChatRenderRequest(ChatHistory History, bool AddGenerationPrompt = true, string? SystemPrompt = null);

/// <summary>
/// Pluggable output parser for an architecture's token stream (e.g. tool extraction or reasoning separation).
/// </summary>
public interface IOutputParser
{
    string ParseChunk(string tokenText);
    string Flush();
}

/// <summary>
/// Architecture-specific chat protocol declaring fallback prompt formatting, thinking markers,
/// media placeholder behavior, and output parsing rules.
/// </summary>
public sealed class ChatProtocol
{
    public required string Id { get; init; }
    public Func<ChatRenderRequest, string>? Render { get; init; }
    public Action<ChatMessage, StringBuilder>? AppendMediaPlaceholders { get; init; }
    public string? ThinkingGrammarActivationTrigger { get; init; } = "</think>";
    public string? ThinkingBudgetEndToken { get; init; } = "</think>";
    public string? ThinkingBudgetClosingText { get; init; }
    public bool PromptAlwaysOpensThinking { get; init; }
    public bool CapsVideoFrames { get; init; }
    public Func<IOutputParser>? CreateOutputParser { get; init; }
}
