using System.Globalization;
using System.Text.Json;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Core.Grammar;

/// <summary>
/// Checks a parsed tool call against the tool's declared argument schema (Phase 6 of the Qwen 3.8 plan, safety rule:
/// validate arguments before a host acts on them). Pure: it returns the problems and executes nothing.
///
/// <para>
/// Qwen's XML form carries every value as text, so a number or boolean written as <c>"3"</c> or <c>"true"</c> is
/// accepted when it reads as that type. A string value for an array or object is parsed as JSON and checked the same
/// way. An enum restricts the value's text to the listed values. A call to a different tool name is an error.
/// </para>
/// </summary>
public static class ToolCallArgumentValidator
{
    /// <summary>Problems with <paramref name="call"/> against <paramref name="schema"/>; empty when the call is valid.</summary>
    public static IReadOnlyList<string> Validate(ParsedToolCall call, ToolSchema schema)
    {
        ArgumentNullException.ThrowIfNull(schema);
        var errors = new List<string>();
        if (!string.Equals(call.Name, schema.Name, StringComparison.Ordinal))
        {
            errors.Add($"tool '{call.Name}' is not the declared tool '{schema.Name}'");
            return errors;
        }
        ValidateObject(call.Arguments, schema.Arguments, "", errors);
        return errors;
    }

    private static void ValidateObject(
        IReadOnlyDictionary<string, object?> value, ToolSchemaObject shape, string path, List<string> errors)
    {
        var declared = new Dictionary<string, ToolSchemaProperty>(StringComparer.Ordinal);
        foreach (var p in shape.Properties) declared[p.Name] = p;

        foreach (var p in shape.Properties)
            if (p.Required && !value.ContainsKey(p.Name))
                errors.Add($"{Where(path, p.Name)}: required argument is missing");

        foreach (var (key, v) in value)
        {
            if (declared.TryGetValue(key, out var prop))
                ValidateValue(v, prop.Value, Where(path, key), errors);
            else if (!shape.Open)
                errors.Add($"{Where(path, key)}: argument is not declared by the schema");
        }
    }

    private static void ValidateValue(object? value, ToolSchemaNode node, string path, List<string> errors)
    {
        if (node.EnumValues is { } allowed)
        {
            // Enum literals are JSON text, so a boolean is compared as true/false, not the .NET "True"/"False".
            string text = value switch
            {
                null => "null",
                bool b => b ? "true" : "false",
                _ => Convert.ToString(value, CultureInfo.InvariantCulture) ?? "",
            };
            if (!allowed.Contains(text, StringComparer.Ordinal))
                errors.Add($"{path}: value '{text}' is not one of the allowed values");
            return;
        }

        switch (node.Kind)
        {
            case JsonSchemaKind.Any:
                return;

            case JsonSchemaKind.String:
                if (value is not string)
                    errors.Add($"{path}: expected a string");
                return;

            case JsonSchemaKind.Number:
                // NaN and infinity are not JSON numbers, so they are refused in both forms.
                if (value is long) return;
                if (value is double dn && double.IsFinite(dn)) return;
                if (value is string s && double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out double parsed)
                    && double.IsFinite(parsed)) return;
                errors.Add($"{path}: expected a number");
                return;

            case JsonSchemaKind.Integer:
                if (value is long) return;
                if (value is double d && double.IsFinite(d) && Math.Floor(d) == d) return;
                if (value is string si && long.TryParse(si, NumberStyles.Integer, CultureInfo.InvariantCulture, out _)) return;
                errors.Add($"{path}: expected an integer");
                return;

            case JsonSchemaKind.Boolean:
                if (value is bool) return;
                if (value is string sb && (sb == "true" || sb == "false")) return;
                errors.Add($"{path}: expected true or false");
                return;

            case JsonSchemaKind.Null:
                if (value is null || (value is string sn && sn == "null")) return;
                errors.Add($"{path}: expected null");
                return;

            case JsonSchemaKind.Array:
                ValidateArray(value, node, path, errors);
                return;

            case JsonSchemaKind.Object:
                ValidateObjectValue(value, node, path, errors);
                return;

            default:
                return;
        }
    }

    private static void ValidateArray(object? value, ToolSchemaNode node, string path, List<string> errors)
    {
        object? model = AsObjectModel(value);
        if (model is not List<object?> items)
        {
            errors.Add($"{path}: expected an array");
            return;
        }
        if (node.Items is null) return;
        for (int i = 0; i < items.Count; i++)
            ValidateValue(items[i], node.Items, $"{path}[{i}]", errors);
    }

    private static void ValidateObjectValue(object? value, ToolSchemaNode node, string path, List<string> errors)
    {
        object? model = AsObjectModel(value);
        if (model is not Dictionary<string, object?> dict)
        {
            errors.Add($"{path}: expected an object");
            return;
        }
        if (node.Object is null) return;
        ValidateObject(dict, node.Object, path, errors);
    }

    /// <summary>A string that holds JSON is read as that JSON value; anything else is returned unchanged.</summary>
    private static object? AsObjectModel(object? value)
    {
        if (value is not string text) return value;
        try
        {
            using var doc = JsonDocument.Parse(text);
            return ToolCallParseHelpers.JsonElementToObject(doc.RootElement);
        }
        catch (JsonException)
        {
            return text;
        }
    }

    private static string Where(string path, string name) => path.Length == 0 ? name : $"{path}.{name}";
}
