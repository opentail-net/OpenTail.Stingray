using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

public sealed class ChatProtocolRegistryTests
{
    [Theory]
    [InlineData("qwen", "chatml")]
    [InlineData("qwen2", "chatml")]
    [InlineData("qwen3", "chatml")]
    [InlineData("qwen35", "chatml")]
    [InlineData("deepseek2", "chatml")]
    [InlineData("gpt-oss", "chatml")]
    [InlineData("smollm3", "chatml")]
    [InlineData("llama", "llama3")]
    [InlineData("llama3", "llama3")]
    [InlineData("llama31", "llama3")]
    [InlineData("llama4", "llama4")]
    [InlineData("granite", "granite")]
    [InlineData("granitehybrid", "granite")]
    [InlineData("granitemoe", "granite")]
    [InlineData("gemma", "gemma")]
    [InlineData("gemma2", "gemma")]
    [InlineData("gemma3", "gemma")]
    [InlineData("gemma4", "gemma")]
    public void For_KnownArchitecture_ReturnsExpectedProtocol(string architecture, string expectedProtocolId)
    {
        var protocol = ChatProtocolRegistry.For(architecture);
        Assert.NotNull(protocol);
        Assert.Equal(expectedProtocolId, protocol.Id);
    }

    [Theory]
    [InlineData("unknown_family")]
    [InlineData("")]
    [InlineData(null)]
    public void For_UnknownOrNull_ReturnsDefaultChatMl(string? architecture)
    {
        var protocol = ChatProtocolRegistry.For(architecture);
        Assert.NotNull(protocol);
        Assert.Equal("chatml", protocol.Id);
    }

    [Fact]
    public void ChatMl_Render_FormatsCorrectly_WithAndWithoutGenerationPrompt()
    {
        var protocol = ChatProtocolRegistry.For("qwen2");
        Assert.NotNull(protocol.Render);

        var history = new ChatHistory();
        history.AddSystemMessage("You are a helpful assistant.");
        history.AddUserMessage("Hello!");

        // Without generation prompt
        string renderedWithout = protocol.Render(new ChatRenderRequest(history, AddGenerationPrompt: false));
        Assert.Equal("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n", renderedWithout);

        // With generation prompt
        string renderedWith = protocol.Render(new ChatRenderRequest(history, AddGenerationPrompt: true));
        Assert.Equal("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n", renderedWith);
    }

    [Fact]
    public void Llama3_Render_FormatsHeadersAndEot()
    {
        var protocol = ChatProtocolRegistry.For("llama");
        Assert.NotNull(protocol.Render);

        var history = new ChatHistory();
        history.AddUserMessage("Hi");

        string rendered = protocol.Render(new ChatRenderRequest(history, AddGenerationPrompt: true));
        Assert.Contains("<|begin_of_text|>", rendered);
        Assert.Contains("<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>", rendered);
        Assert.EndsWith("<|start_header_id|>assistant<|end_header_id|>\n\n", rendered);
    }

    [Fact]
    public void Granite_Render_FormatsStartOfRole()
    {
        var protocol = ChatProtocolRegistry.For("granite");
        Assert.NotNull(protocol.Render);

        var history = new ChatHistory();
        history.AddUserMessage("Query");

        string rendered = protocol.Render(new ChatRenderRequest(history, AddGenerationPrompt: true));
        Assert.Contains("<|start_of_role|>user<|end_of_role|>Query<|end_of_text|>\n", rendered);
        Assert.EndsWith("<|start_of_role|>assistant<|end_of_role|>", rendered);
    }

    [Fact]
    public void Gemma_Render_MapsRolesToUserAndModel()
    {
        var protocol = ChatProtocolRegistry.For("gemma4");
        Assert.NotNull(protocol.Render);

        var history = new ChatHistory();
        history.AddSystemMessage("Instructions");
        history.AddAssistantMessage("Previous response");
        history.AddUserMessage("Next query");

        string rendered = protocol.Render(new ChatRenderRequest(history, AddGenerationPrompt: true));
        Assert.Contains("<start_of_turn>user\nInstructions<end_of_turn>\n", rendered);
        Assert.Contains("<start_of_turn>model\nPrevious response<end_of_turn>\n", rendered);
        Assert.Contains("<start_of_turn>user\nNext query<end_of_turn>\n", rendered);
        Assert.EndsWith("<start_of_turn>model\n", rendered);
        Assert.Null(protocol.ThinkingBudgetEndToken);
    }
}
