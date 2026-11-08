using OpenTail.Stingray.Core;
using OpenTail.Stingray.Core.Grammar;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Phase 6 safety rule (Qwen 3.8 plan): a parsed tool call is checked against the declared schema before a host acts on it.
/// Synthetic schemas, no checkpoint.
/// </summary>
public sealed class ToolCallArgumentValidatorTests
{
    private static ToolSchema WeatherSchema(bool open = false) => new(
        "get_weather",
        new ToolSchemaObject(
        [
            new ToolSchemaProperty("city", new ToolSchemaNode(JsonSchemaKind.String), Required: true),
            new ToolSchemaProperty("days", new ToolSchemaNode(JsonSchemaKind.Integer), Required: false),
            new ToolSchemaProperty("metric", new ToolSchemaNode(JsonSchemaKind.Boolean), Required: false),
            new ToolSchemaProperty("unit", new ToolSchemaNode(JsonSchemaKind.String, enumValues: ["C", "F"]), Required: false),
        ], open));

    private static ParsedToolCall Call(string name, params (string Key, object? Value)[] args) =>
        new(name, args.ToDictionary(a => a.Key, a => a.Value, StringComparer.Ordinal));

    [Fact]
    public void ValidCall_HasNoProblems()
    {
        var errors = ToolCallArgumentValidator.Validate(
            Call("get_weather", ("city", "Paris"), ("days", 3L), ("metric", true)), WeatherSchema());
        Assert.Empty(errors);
    }

    [Fact]
    public void XmlStyleTextValues_AreAcceptedWhenTheyReadAsTheDeclaredType()
    {
        var errors = ToolCallArgumentValidator.Validate(
            Call("get_weather", ("city", "Paris"), ("days", "3"), ("metric", "false")), WeatherSchema());
        Assert.Empty(errors);
    }

    [Fact]
    public void MissingRequiredArgument_IsReported()
    {
        var errors = ToolCallArgumentValidator.Validate(Call("get_weather", ("days", 3L)), WeatherSchema());
        Assert.Single(errors);
        Assert.Contains("city", errors[0], StringComparison.Ordinal);
        Assert.Contains("missing", errors[0], StringComparison.Ordinal);
    }

    [Fact]
    public void UndeclaredArgument_IsReported_WhenTheSchemaIsClosed()
    {
        var errors = ToolCallArgumentValidator.Validate(
            Call("get_weather", ("city", "Paris"), ("secret", "x")), WeatherSchema());
        Assert.Single(errors);
        Assert.Contains("secret", errors[0], StringComparison.Ordinal);
    }

    [Fact]
    public void UndeclaredArgument_IsAllowed_WhenTheSchemaIsOpen()
    {
        var errors = ToolCallArgumentValidator.Validate(
            Call("get_weather", ("city", "Paris"), ("extra", "x")), WeatherSchema(open: true));
        Assert.Empty(errors);
    }

    [Theory]
    [InlineData("three")]
    [InlineData("2.5")]
    public void IntegerArgument_RejectsNonIntegers(string value)
    {
        var errors = ToolCallArgumentValidator.Validate(
            Call("get_weather", ("city", "Paris"), ("days", value)), WeatherSchema());
        Assert.Single(errors);
        Assert.Contains("days", errors[0], StringComparison.Ordinal);
    }

    [Fact]
    public void BooleanArgument_RejectsOtherText()
    {
        var errors = ToolCallArgumentValidator.Validate(
            Call("get_weather", ("city", "Paris"), ("metric", "yes")), WeatherSchema());
        Assert.Single(errors);
    }

    [Fact]
    public void EnumArgument_RejectsValuesOutsideTheList()
    {
        var ok = ToolCallArgumentValidator.Validate(
            Call("get_weather", ("city", "Paris"), ("unit", "C")), WeatherSchema());
        var bad = ToolCallArgumentValidator.Validate(
            Call("get_weather", ("city", "Paris"), ("unit", "K")), WeatherSchema());
        Assert.Empty(ok);
        Assert.Single(bad);
        Assert.Contains("allowed", bad[0], StringComparison.Ordinal);
    }

    [Fact]
    public void ArrayItems_AreCheckedAgainstTheItemType()
    {
        var schema = new ToolSchema("sum", new ToolSchemaObject(
        [
            new ToolSchemaProperty("values",
                new ToolSchemaNode(JsonSchemaKind.Array, items: new ToolSchemaNode(JsonSchemaKind.Integer)), Required: true),
        ], open: false));

        Assert.Empty(ToolCallArgumentValidator.Validate(Call("sum", ("values", "[1,2,3]")), schema));
        Assert.Single(ToolCallArgumentValidator.Validate(Call("sum", ("values", "[1,\"x\",3]")), schema));
        Assert.Single(ToolCallArgumentValidator.Validate(Call("sum", ("values", "not json")), schema));
    }

    [Fact]
    public void WrongToolName_IsReported()
    {
        var errors = ToolCallArgumentValidator.Validate(Call("delete_file", ("city", "Paris")), WeatherSchema());
        Assert.Single(errors);
        Assert.Contains("delete_file", errors[0], StringComparison.Ordinal);
    }

    [Fact]
    public void ParsedQwenXmlCall_ValidatesEndToEnd()
    {
        // The parser's output is what a host validates: strings from XML, checked against the schema.
        var raw = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
                + "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
        var (_, calls) = new QwenToolCallAdapter("qwen4exp").Parse(raw);

        Assert.Empty(ToolCallArgumentValidator.Validate(Assert.Single(calls), WeatherSchema()));
    }
}
