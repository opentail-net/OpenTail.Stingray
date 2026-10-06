#nullable enable

using System.Text;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Registry mapping model architectures to their canonical wire chat protocols,
/// default renderers, thinking delimiters, and tool token boundaries.
/// </summary>
public static class ChatProtocolRegistry
{
    private static readonly Dictionary<string, ChatProtocol> s_byArch = new(StringComparer.OrdinalIgnoreCase);
    private static readonly ChatProtocol s_chatMl;

    static ChatProtocolRegistry()
    {
        s_chatMl = new ChatProtocol
        {
            Id = "chatml",
            Architectures = ["qwen", "qwen2", "qwen2moe", "qwen3", "qwen3moe", "qwen35", "qwen35moe", "deepseek2", "gpt-oss", "smollm3"],
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
            Architectures = ["llama", "llama3", "llama31", "llama32", "llama33"],
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
            Architectures = ["llama4"],
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
            Architectures = ["granite", "granitehybrid", "granitemoe"],
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
            Architectures = ["gemma", "gemma2", "gemma3", "gemma4"],
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

    private static void Register(ChatProtocol protocol)
    {
        foreach (var arch in protocol.Architectures)
        {
            s_byArch[arch] = protocol;
        }
    }

    /// <summary>
    /// Returns the matched chat protocol for an architecture, or the default ChatML protocol.
    /// </summary>
    public static ChatProtocol For(string? architecture)
    {
        if (architecture is not null && s_byArch.TryGetValue(architecture, out var protocol))
            return protocol;
        return s_chatMl;
    }
}
