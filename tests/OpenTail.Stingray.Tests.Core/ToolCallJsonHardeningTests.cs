using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Phase 6 safety (Qwen 3.8 plan): malformed JSON tool calls are rejected. They must not throw, and must not come back as a
/// call with empty arguments. Covers the Qwen/Llama JSON path (<c>ParseJsonCallBlock</c>) and the DeepSeek path. Synthetic.
/// </summary>
public sealed class ToolCallJsonHardeningTests
{
    private static QwenToolCallAdapter Qwen() => new("qwen4exp");

    private static DeepSeekToolCallAdapter DeepSeek() => new();

    private static string QwenBlock(string json) => $"<tool_call>{json}</tool_call>";

    private static string DeepSeekBlock(string argsJson) =>
        "<|tool_calls_begin|><|tool_call_begin|>f<|tool_sep|>" + argsJson + "<|tool_call_end|><|tool_calls_end|>";

    [Theory]
    [InlineData("[]")]
    [InlineData("\"just a string\"")]
    [InlineData("42")]
    [InlineData("{\"name\":42,\"arguments\":{}}")]
    [InlineData("{\"name\":\"\",\"arguments\":{}}")]
    [InlineData("{\"name\":\"a\",\"name\":\"b\",\"arguments\":{}}")]
    [InlineData("{\"name\":\"a\",\"arguments\":[1,2]}")]
    [InlineData("{\"name\":\"a\",\"arguments\":5}")]
    [InlineData("{\"name\":\"a\",\"arguments\":\"[1]\"}")]
    [InlineData("{\"name\":\"a\",\"arguments\":{\"k\":1,\"k\":2}}")]
    public void Qwen_MalformedJsonCall_IsRejected_WithoutThrowing(string json)
    {
        var result = Qwen().Parse(QwenBlock(json));

        Assert.Empty(result.Calls);
    }

    [Fact]
    public void Qwen_ValidJsonCall_StillParses()
    {
        var result = Qwen().Parse(QwenBlock("{\"name\":\"get\",\"arguments\":{\"city\":\"Paris\"}}"));

        var call = Assert.Single(result.Calls);
        Assert.Equal("get", call.Name);
        Assert.Equal("Paris", call.Arguments["city"]);
    }

    [Fact]
    public void Qwen_ValidJsonCallWithEmptyArguments_StillParses()
    {
        var call = Assert.Single(Qwen().Parse(QwenBlock("{\"name\":\"ping\",\"arguments\":{}}")).Calls);

        Assert.Equal("ping", call.Name);
        Assert.Empty(call.Arguments);
    }

    [Theory]
    [InlineData("[]")]
    [InlineData("[1,2]")]
    [InlineData("\"text\"")]
    [InlineData("42")]
    [InlineData("{\"a\":1,\"a\":2}")]
    [InlineData("{\"a\":{\"b\":1,\"b\":2}}")]
    [InlineData("not json")]
    public void DeepSeek_MalformedArguments_RejectTheCall_WithoutThrowing(string argsJson)
    {
        var calls = DeepSeek().ParseBlock(DeepSeekBlock(argsJson));

        Assert.Empty(calls);
    }

    [Fact]
    public void DeepSeek_ValidArguments_StillParse()
    {
        var calls = DeepSeek().ParseBlock(DeepSeekBlock("{\"city\":\"Paris\"}"));

        var call = Assert.Single(calls);
        Assert.Equal("f", call.Name);
        Assert.Equal("Paris", call.Arguments["city"]);
    }
}
