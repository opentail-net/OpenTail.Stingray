using OpenTail.Stingray.Cli;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>
/// The ChatML fallback wraps a template-less prompt in &lt;|im_start|&gt; markers. On a vocabulary without them (GPT-2,
/// StarCoder2) the model reads the markers as text, which the 2026-10-08 smoke sweep mistook for odd model output.
/// </summary>
public sealed class ChatMlFallbackWarningTests
{
    private static readonly IReadOnlyDictionary<string, int> s_noChatMl = new Dictionary<string, int> { ["<|endoftext|>"] = 50256 };
    private static readonly IReadOnlyDictionary<string, int> s_chatMl = new Dictionary<string, int> { ["<|im_start|>"] = 1, ["<|im_end|>"] = 2 };

    [Fact]
    public void BaseModel_WithoutTemplateOrChatMlTokens_Warns() =>
        Assert.True(RunCommand.UsesUnsupportedChatMlFallback("gpt2", hasChatTemplate: false, rawPrompt: false, s_noChatMl));

    [Fact]
    public void ChatMlVocabulary_DoesNotWarn() =>
        Assert.False(RunCommand.UsesUnsupportedChatMlFallback("qwen2", hasChatTemplate: false, rawPrompt: false, s_chatMl));

    [Fact]
    public void EmbeddedOrOverriddenTemplate_DoesNotWarn() =>
        Assert.False(RunCommand.UsesUnsupportedChatMlFallback("gpt2", hasChatTemplate: true, rawPrompt: false, s_noChatMl));

    [Fact]
    public void RawPromptMode_DoesNotWarn() =>
        Assert.False(RunCommand.UsesUnsupportedChatMlFallback("gpt2", hasChatTemplate: false, rawPrompt: true, s_noChatMl));

    [Fact]
    public void NonChatMlFallbackArchitecture_DoesNotWarn() =>
        Assert.False(RunCommand.UsesUnsupportedChatMlFallback("llama", hasChatTemplate: false, rawPrompt: false, s_noChatMl));
}
