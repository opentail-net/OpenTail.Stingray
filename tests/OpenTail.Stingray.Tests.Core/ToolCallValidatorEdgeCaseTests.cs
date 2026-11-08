using OpenTail.Stingray.Core;
using OpenTail.Stingray.Core.Grammar;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Phase 6 validator edge cases (Qwen 3.8 plan): non-finite numbers are not numbers or integers, and a boolean enum matches
/// the JSON literals <c>true</c>/<c>false</c>. Synthetic.
/// </summary>
public sealed class ToolCallValidatorEdgeCaseTests
{
    private static ToolSchema Schema(params ToolSchemaProperty[] props) =>
        new("t", new ToolSchemaObject(props, open: false));

    private static ParsedToolCall Call(params (string Key, object? Value)[] args) =>
        new("t", args.ToDictionary(a => a.Key, a => a.Value, StringComparer.Ordinal));

    private static ToolSchema NumberSchema() =>
        Schema(new ToolSchemaProperty("x", new ToolSchemaNode(JsonSchemaKind.Number), Required: true));

    private static ToolSchema IntegerSchema() =>
        Schema(new ToolSchemaProperty("n", new ToolSchemaNode(JsonSchemaKind.Integer), Required: true));

    [Theory]
    [InlineData("NaN")]
    [InlineData("Infinity")]
    [InlineData("-Infinity")]
    public void Number_RejectsNonFiniteText(string text)
    {
        Assert.Single(ToolCallArgumentValidator.Validate(Call(("x", text)), NumberSchema()));
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void Number_RejectsNonFiniteDoubles(double value)
    {
        Assert.Single(ToolCallArgumentValidator.Validate(Call(("x", value)), NumberSchema()));
    }

    [Fact]
    public void Number_AcceptsFiniteValues()
    {
        Assert.Empty(ToolCallArgumentValidator.Validate(Call(("x", 2.5)), NumberSchema()));
        Assert.Empty(ToolCallArgumentValidator.Validate(Call(("x", "2.5")), NumberSchema()));
    }

    [Theory]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(double.NaN)]
    [InlineData(2.5)]
    public void Integer_RejectsNonFiniteAndFractionalDoubles(double value)
    {
        Assert.Single(ToolCallArgumentValidator.Validate(Call(("n", value)), IntegerSchema()));
    }

    [Theory]
    [InlineData("Infinity")]
    [InlineData("NaN")]
    public void Integer_RejectsNonFiniteText(string text)
    {
        Assert.Single(ToolCallArgumentValidator.Validate(Call(("n", text)), IntegerSchema()));
    }

    [Fact]
    public void Integer_AcceptsWholeFiniteDoubles()
    {
        Assert.Empty(ToolCallArgumentValidator.Validate(Call(("n", 3.0)), IntegerSchema()));
    }

    [Fact]
    public void BooleanEnum_MatchesJsonLiterals()
    {
        var schema = Schema(new ToolSchemaProperty(
            "flag", new ToolSchemaNode(JsonSchemaKind.Boolean, enumValues: ["true", "false"]), Required: true));

        Assert.Empty(ToolCallArgumentValidator.Validate(Call(("flag", true)), schema));
        Assert.Empty(ToolCallArgumentValidator.Validate(Call(("flag", false)), schema));
        Assert.Empty(ToolCallArgumentValidator.Validate(Call(("flag", "true")), schema));
    }

    [Fact]
    public void BooleanEnum_RejectsValuesOutsideTheList()
    {
        var schema = Schema(new ToolSchemaProperty(
            "flag", new ToolSchemaNode(JsonSchemaKind.Boolean, enumValues: ["true"]), Required: true));

        Assert.Single(ToolCallArgumentValidator.Validate(Call(("flag", false)), schema));
    }
}
