namespace OpenTail.Stingray.Cli;

public sealed partial class RunCommand
{
    /// <summary>
    /// Process llama.cpp-style escape sequences in <paramref name="s"/>: <c>\n</c>, <c>\t</c>,
    /// <c>\r</c>, <c>\\</c>. Mirrors llama.cpp's <c>-e/--escape</c>. Unknown escapes and a
    /// trailing backslash are left exactly as written.
    /// </summary>
    internal static string ProcessEscapeSequences(string s)
    {
        if (s.IndexOf('\\', StringComparison.Ordinal) < 0) return s;

        var sb = new System.Text.StringBuilder(s.Length);
        for (int i = 0; i < s.Length; i++)
        {
            if (s[i] == '\\' && i + 1 < s.Length)
            {
                switch (s[i + 1])
                {
                    case 'n':  sb.Append('\n'); i++; continue;
                    case 't':  sb.Append('\t'); i++; continue;
                    case 'r':  sb.Append('\r'); i++; continue;
                    case '\\': sb.Append('\\'); i++; continue;
                }
            }
            sb.Append(s[i]);
        }
        return sb.ToString();
    }

    /// <summary>
    /// Resolves whether reasoning ("thinking") should be OFF for this run.
    /// Precedence: <c>--no-thinking</c> forces it off and wins if both flags are passed;
    /// <c>--thinking</c> forces it on; with neither, Gemma 4 defaults off (its stock instruct
    /// models aren't reasoning-trained — nothing in the GGUF metadata distinguishes a reasoning
    /// finetune from a stock model, so reasoning is opt-in there) while every other architecture
    /// defaults on. Pure and side-effect-free so the precedence is unit-testable.
    /// </summary>
    internal static bool ResolveThinkingOff(string arch, bool thinking, bool noThinking)
    {
        if (noThinking) return true;   // explicit off wins over a conflicting --thinking
        if (thinking)   return false;  // explicit on
        return ArchitectureRegistry.ThinkingDefaultOff(arch); // per-family default (Gemma 4)
    }

    /// <summary>
    /// Prepends BOS when the model asks for it (<c>tokenizer.ggml.add_bos_token</c>) and the prompt does not
    /// already start with it, as llama.cpp's <c>common_tokenize(..., add_special: true)</c> does. Raw prompts
    /// never contain it, and neither do chat templates that omit <c>bos_token</c> (DeepSeek-OCR2's is just the
    /// concatenated contents). Before 2026-09-27 this ran only for STINGRAY_RAW_PROMPT, so such templated
    /// prompts had no BOS at all.
    /// </summary>
    private static IReadOnlyList<int> EnsureBos(IReadOnlyList<int> tokens, GgufTokenizer tok)
    {
        if (!tok.AddBosToken || tok.BosTokenId < 0 || (tokens.Count > 0 && tokens[0] == tok.BosTokenId))
            return tokens;
        var withBos = new List<int>(tokens.Count + 1) { tok.BosTokenId };
        withBos.AddRange(tokens);
        return withBos;
    }

    internal static string FormatPrompt(string userMessage, string? systemPrompt, bool enableThinking = true, bool injectDefaultSystem = true)
    {
        // STINGRAY_RAW_PROMPT=1 bypasses the chat template entirely. Used for parity testing
        // against llama.cpp's --no-conversation mode (raw text completion). Not for normal use.
        if (Environment.GetEnvironmentVariable("STINGRAY_RAW_PROMPT") == "1")
            return userMessage;

        // Granite models (especially Granite 4.0 Vision): IBM's embedded GGUF Jinja template
        // unconditionally injects a default system prompt before user image tokens even when
        // none was requested, and indents user content by 8 spaces in render_content(x), causing
        // the vision decoder to predict <|end_of_text|> at token 0 (empty decode / early EOS).
        // Match llama.cpp's canonical LLM_CHAT_TEMPLATE_GRANITE_4_0 instead of the defective metadata Jinja.
        if (ArchitectureRegistry.FallbackChat(s_arch) == FallbackChatFormat.Granite)
        {
            var sbGranite = new System.Text.StringBuilder();
            if (!string.IsNullOrEmpty(systemPrompt))
                sbGranite.Append($"<|start_of_role|>system<|end_of_role|>{systemPrompt}<|end_of_text|>\n");
            sbGranite.Append($"<|start_of_role|>user<|end_of_role|>{userMessage}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>");
            return sbGranite.ToString();
        }

        // Use the model's own Jinja2 chat template when available (read from GGUF metadata).
        if (s_jinja != null)
        {
            // Qwen3 (dense) chat models behave poorly without a system message — they
            // end the turn after a few tokens for short prompts. The hardcoded fallback
            // path (below) injects this default; mirror it here for the same arch.
            // Note: qwen3moe is intentionally excluded — Qwen3-Coder appears to be
            // tuned to operate without a system prompt and gets HIGH-confidence on
            // <|endoftext|> when one is forced (logit ~29 vs ~14 with no system).
            string? effectiveSystemPrompt = systemPrompt
                ?? (injectDefaultSystem && s_arch is "qwen3" ? "You are a helpful assistant." : null);
            var messages = JinjaChatTemplate.BuildMessages(userMessage, systemContent: effectiveSystemPrompt);
            return s_jinja.Render(new Dictionary<string, object?>
            {
                ["messages"]             = messages,
                ["add_generation_prompt"] = true,
                ["tools"]                = (object?)s_tools,
                ["enable_thinking"]      = enableThinking,
            });
        }

        // Fallback: hardcoded templates for known architectures.
        var sb = new System.Text.StringBuilder();

        if (ArchitectureRegistry.FallbackChat(s_arch) == FallbackChatFormat.Llama4)
        {
            // Llama 4: <|begin_of_text|><|header_start|>role<|header_end|>\n\nmessage<|eot|>
            sb.Append("<|begin_of_text|>");
            if (systemPrompt is not null)
                sb.Append($"<|header_start|>system<|header_end|>\n\n{systemPrompt}<|eot|>");
            sb.Append($"<|header_start|>user<|header_end|>\n\n{userMessage}<|eot|>");
            sb.Append("<|header_start|>assistant<|header_end|>\n\n");
        }
        else if (ArchitectureRegistry.FallbackChat(s_arch) == FallbackChatFormat.Llama3)
        {
            if (!s_hasLlama3Headers)
            {
                // Vicuna (LLaVA-1.5): USER: {userMessage}\nASSISTANT:
                if (systemPrompt is not null)
                    sb.Append($"{systemPrompt}\n\n");
                sb.Append($"USER: {userMessage}\nASSISTANT:");
            }
            else
            {
                // Llama 3/3.1: <|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\nmessage<|eot_id|>
                sb.Append("<|begin_of_text|>");
                if (systemPrompt is not null)
                    sb.Append($"<|start_header_id|>system<|end_header_id|>\n\n{systemPrompt}<|eot_id|>");
                sb.Append($"<|start_header_id|>user<|end_header_id|>\n\n{userMessage}<|eot_id|>");
                sb.Append("<|start_header_id|>assistant<|end_header_id|>\n\n");
            }
        }
        else if (ArchitectureRegistry.FallbackChat(s_arch) == FallbackChatFormat.Granite)
        {
            // Granite: <|start_of_role|>role<|end_of_role|>message<|end_of_text|>\n
            if (systemPrompt is not null)
                sb.Append($"<|start_of_role|>system<|end_of_role|>{systemPrompt}<|end_of_text|>\n");
            sb.Append($"<|start_of_role|>user<|end_of_role|>{userMessage}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>");
        }
        else
        {
            // ChatML (Qwen, SmolLM, default): <|im_start|>role\nmessage<|im_end|>
            string? effectiveSystemPrompt = systemPrompt
                ?? (s_arch is "qwen3moe" or "qwen3" ? "You are a helpful assistant." : null);
            if (effectiveSystemPrompt is not null)
                sb.Append($"<|im_start|>system\n{effectiveSystemPrompt}<|im_end|>\n");
            sb.Append($"<|im_start|>user\n{userMessage}<|im_end|>\n<|im_start|>assistant\n");
        }

        return sb.ToString();
    }
}
