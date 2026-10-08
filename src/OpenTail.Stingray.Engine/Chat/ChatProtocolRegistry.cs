#nullable enable

using System.Text;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Registry mapping model architectures to their canonical wire chat protocols,
/// default renderers, thinking delimiters, and tool token boundaries.
/// </summary>
public static class ChatProtocolRegistry
{
    private static readonly Dictionary<string, ChatProtocol> s_byId = new(StringComparer.Ordinal);
    private static readonly ChatProtocol s_chatMl;

    static ChatProtocolRegistry()
    {
        s_chatMl = new ChatProtocol
        {
            Id = "chatml",
            Render = req =>
            {
                var sb = new StringBuilder();
                foreach (var msg in req.History)
                {
                    string role = msg.Role switch
                    {
                        AuthorRole.System => "system",
                        AuthorRole.User => "user",
                        AuthorRole.Assistant => "assistant",
                        AuthorRole.Tool => "tool",
                        _ => "user"
                    };
                    sb.Append($"<|im_start|>{role}\n{msg.Content}<|im_end|>\n");
                }
                if (req.AddGenerationPrompt)
                {
                    sb.Append("<|im_start|>assistant\n");
                }
                return sb.ToString();
            },
            ThinkingGrammarActivationTrigger = "</think>",
            ThinkingBudgetEndToken = "</think>",
        };

        var llama3 = new ChatProtocol
        {
            Id = "llama3",
            Render = req =>
            {
                var sb = new StringBuilder();
                sb.Append("<|begin_of_text|>");
                foreach (var msg in req.History)
                {
                    string role = msg.Role switch
                    {
                        AuthorRole.System => "system",
                        AuthorRole.User => "user",
                        AuthorRole.Assistant => "assistant",
                        AuthorRole.Tool => "tool",
                        _ => "user"
                    };
                    sb.Append($"<|start_header_id|>{role}<|end_header_id|>\n\n{msg.Content}<|eot_id|>");
                }
                if (req.AddGenerationPrompt)
                {
                    sb.Append("<|start_header_id|>assistant<|end_header_id|>\n\n");
                }
                return sb.ToString();
            },
        };

        var llama4 = new ChatProtocol
        {
            Id = "llama4",
            Render = req =>
            {
                var sb = new StringBuilder();
                sb.Append("<|begin_of_text|>");
                foreach (var msg in req.History)
                {
                    string role = msg.Role switch
                    {
                        AuthorRole.System => "system",
                        AuthorRole.User => "user",
                        AuthorRole.Assistant => "assistant",
                        AuthorRole.Tool => "tool",
                        _ => "user"
                    };
                    sb.Append($"<|start_header_id|>{role}<|end_header_id|>\n\n{msg.Content}<|eot_id|>");
                }
                if (req.AddGenerationPrompt)
                {
                    sb.Append("<|start_header_id|>assistant<|end_header_id|>\n\n");
                }
                return sb.ToString();
            },
        };

        var granite = new ChatProtocol
        {
            Id = "granite",
            Render = req =>
            {
                var sb = new StringBuilder();
                foreach (var msg in req.History)
                {
                    string role = msg.Role switch
                    {
                        AuthorRole.System => "system",
                        AuthorRole.User => "user",
                        AuthorRole.Assistant => "assistant",
                        AuthorRole.Tool => "tool",
                        _ => "user"
                    };
                    sb.Append($"<|start_of_role|>{role}<|end_of_role|>{msg.Content}<|end_of_text|>\n");
                }
                if (req.AddGenerationPrompt)
                {
                    sb.Append("<|start_of_role|>assistant<|end_of_role|>");
                }
                return sb.ToString();
            },
        };

        var gemma = new ChatProtocol
        {
            Id = "gemma",
            Render = req =>
            {
                var sb = new StringBuilder();
                foreach (var msg in req.History)
                {
                    string role = msg.Role switch
                    {
                        AuthorRole.System => "user",
                        AuthorRole.User => "user",
                        AuthorRole.Assistant => "model",
                        AuthorRole.Tool => "tool",
                        _ => "user"
                    };
                    sb.Append($"<start_of_turn>{role}\n{msg.Content}<end_of_turn>\n");
                }
                if (req.AddGenerationPrompt)
                {
                    sb.Append("<start_of_turn>model\n");
                }
                return sb.ToString();
            },
            ThinkingBudgetEndToken = null,
        };

        Register(s_chatMl);
        Register(llama3);
        Register(llama4);
        Register(granite);
        Register(gemma);
    }

    private static void Register(ChatProtocol protocol) => s_byId[protocol.Id] = protocol;

    // Llama model-family names callers have passed as an "architecture" (the GGUF architecture is just "llama"). They are not
    // registered descriptors, so they must not become admitted architectures; they only select the llama3 chat protocol.
    private static readonly HashSet<string> s_llama3FamilyNames = new(StringComparer.OrdinalIgnoreCase) { "llama3", "llama31", "llama32", "llama33" };

    /// <summary>
    /// Returns the matched chat protocol for an architecture, or the default ChatML protocol.
    /// </summary>
    public static ChatProtocol For(string? architecture)
    {
        string? id = ArchitectureRegistry.Find(architecture)?.ChatProtocolId
                     ?? (architecture is not null && s_llama3FamilyNames.Contains(architecture) ? "llama3" : null);
        return id is not null && s_byId.TryGetValue(id, out var protocol) ? protocol : s_chatMl;
    }
}
