using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Phase 6 acceptance (Qwen 3.8 plan): tool-call parsing for qwen4exp. Synthetic, in-memory, no checkpoint. Covers the
/// JSON and XML wire shapes, multiple calls, incomplete calls (never executed), argument text kept exactly as written,
/// reasoning text that mentions a call, and streaming prefixes that never yield a call the full output would not.
/// </summary>
public sealed class Qwen4ExpToolCallParseTests
{
    private static QwenToolCallAdapter Adapter() => new("qwen4exp");

    [Fact]
    public void Registry_QwenFamilyWrapperServesQwen4Exp()
    {
        // qwen4exp has no registry entry of its own (the architecture id must not appear outside the
        // Architectures folder); it resolves to the default Qwen wrapper, which carries the same parser.
        var adapter = ToolCallAdapterRegistry.Get("qwen4exp");
        Assert.IsType<QwenToolCallAdapter>(adapter);
        Assert.Same(ToolCallAdapterRegistry.DefaultAdapter, adapter);
    }

    [Fact]
    public void JsonBody_ParsesNameAndTypedArguments()
    {
        var raw = "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\",\"days\":3,\"metric\":true}}\n</tool_call>";
        var (plain, calls) = Adapter().Parse(raw);

        Assert.Equal("", plain);
        var call = Assert.Single(calls);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("Paris", call.Arguments["city"]);
        Assert.Equal(3L, call.Arguments["days"]);
        Assert.Equal(true, call.Arguments["metric"]);
    }

    [Fact]
    public void XmlBody_ParametersAreStringsEvenWhenNumeric()
    {
        var raw = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
        var (_, calls) = Adapter().Parse(raw);

        var call = Assert.Single(calls);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("Paris", call.Arguments["city"]);
        Assert.Equal("3", call.Arguments["days"]);
    }

    [Fact]
    public void XmlBody_CodeKeepsIndentationAndInternalNewlines()
    {
        // Only the single framing newline the template puts around a value is removed.
        var raw = "<tool_call>\n<function=run_code>\n<parameter=code>\n    def f(x):\n        return x + 1\n</parameter>\n</function>\n</tool_call>";
        var (_, calls) = Adapter().Parse(raw);

        var call = Assert.Single(calls);
        Assert.Equal("    def f(x):\n        return x + 1", call.Arguments["code"]);
    }

    [Fact]
    public void XmlBody_KeepsSpacesInsideValueExactly()
    {
        var raw = "<tool_call><function=echo><parameter=text>  two spaces  </parameter></function></tool_call>";
        var (_, calls) = Adapter().Parse(raw);

        Assert.Equal("  two spaces  ", Assert.Single(calls).Arguments["text"]);
    }

    [Fact]
    public void XmlBody_LiteralParameterCloseInsideValue_DoesNotEndTheValue()
    {
        var raw = "<tool_call>\n<function=write_doc>\n<parameter=doc>\nUse </parameter> to close.\n</parameter>\n</function>\n</tool_call>";
        var (_, calls) = Adapter().Parse(raw);

        Assert.Equal("Use </parameter> to close.", Assert.Single(calls).Arguments["doc"]);
    }

    [Fact]
    public void XmlBody_AngleBracketsInValueAreKept()
    {
        var raw = "<tool_call><function=eval><parameter=expr>a<b && c>d</parameter></function></tool_call>";
        var (_, calls) = Adapter().Parse(raw);

        Assert.Equal("a<b && c>d", Assert.Single(calls).Arguments["expr"]);
    }

    [Fact]
    public void MultipleCalls_MixedShapes_KeepOrderAndTextBetween()
    {
        var raw = "Checking both.<tool_call>\n<function=a><parameter=x>1</parameter></function>\n</tool_call>"
                + " and then <tool_call>{\"name\":\"b\",\"arguments\":{\"y\":\"2\"}}</tool_call> done.";
        var (plain, calls) = Adapter().Parse(raw);

        Assert.Equal("Checking both. and then  done.", plain);
        Assert.Equal(2, calls.Count);
        Assert.Equal("a", calls[0].Name);
        Assert.Equal("1", calls[0].Arguments["x"]);
        Assert.Equal("b", calls[1].Name);
        Assert.Equal("2", calls[1].Arguments["y"]);
    }

    [Fact]
    public void IncompleteCall_NoCloseMarker_IsNotExecuted_AndIsFlaggedTruncated()
    {
        var raw = "<tool_call>\n<function=get_weather>\n<parameter=city>\nPar";
        var result = Adapter().Parse(raw);

        Assert.Empty(result.Calls);
        Assert.True(result.Truncated);
    }

    [Fact]
    public void FunctionNotClosed_BeforeCloseMarker_IsNotExecuted()
    {
        var raw = "<tool_call><function=delete_file><parameter=path>/tmp/x</parameter></tool_call>";
        var result = Adapter().Parse(raw);

        Assert.Empty(result.Calls);
        Assert.Contains("<function=delete_file>", result.PlainText, StringComparison.Ordinal);
    }

    [Fact]
    public void ParameterNotClosed_IsNotExecuted_EvenThoughTheFunctionIs()
    {
        // The model closed the function but cut a parameter short; running it would drop the argument silently.
        var raw = "<tool_call><function=delete_file><parameter=path>/tmp/x</function></tool_call>";
        var result = Adapter().Parse(raw);

        Assert.Empty(result.Calls);
    }

    [Fact]
    public void MalformedJson_IsNotExecuted()
    {
        var result = Adapter().Parse("<tool_call>{\"name\":\"x\",</tool_call>");

        Assert.Empty(result.Calls);
    }

    [Fact]
    public void ToolCallMentionedInsideReasoning_IsNotExecuted()
    {
        var raw = "<think>\nI could call <tool_call>{\"name\":\"bad\",\"arguments\":{}}</tool_call> but I will not.\n</think>\n"
                + "<tool_call>\n{\"name\":\"good\",\"arguments\":{}}\n</tool_call>";
        var (_, calls) = Adapter().Parse(raw);

        Assert.Equal("good", Assert.Single(calls).Name);
    }

    [Fact]
    public void ReasoningWithoutOpenMarker_UpToCloseMarker_IsNotExecuted()
    {
        // The generation prompt already opened the think block, so only the close marker appears in the output.
        var raw = "Maybe <tool_call>{\"name\":\"bad\",\"arguments\":{}}</tool_call> would work.</think>\n"
                + "<tool_call>{\"name\":\"good\",\"arguments\":{}}</tool_call>";
        var (_, calls) = Adapter().Parse(raw);

        Assert.Equal("good", Assert.Single(calls).Name);
    }

    [Fact]
    public void StreamingPrefixes_NeverYieldACallTheFullOutputDoesNotHave()
    {
        var raw = "<think>plan</think>\nI will look.<tool_call>\n<function=read_file>\n<parameter=path>\n/etc/hosts\n</parameter>\n</function>\n</tool_call>"
                + "<tool_call>{\"name\":\"list\",\"arguments\":{\"dir\":\"/\"}}</tool_call>";
        var adapter = Adapter();
        var full = adapter.Parse(raw).Calls;
        Assert.Equal(2, full.Count);

        for (int n = 0; n <= raw.Length; n++)
        {
            var partial = adapter.Parse(raw[..n]).Calls;
            Assert.True(partial.Count <= full.Count, $"prefix {n} yielded more calls than the full output");
            for (int i = 0; i < partial.Count; i++)
            {
                Assert.Equal(full[i].Name, partial[i].Name);
                Assert.Equal(full[i].Arguments.Count, partial[i].Arguments.Count);
            }
        }
    }
}
