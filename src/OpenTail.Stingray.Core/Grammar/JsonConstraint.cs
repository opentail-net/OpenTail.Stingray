
namespace OpenTail.Stingray.Core.Grammar;

/// <summary>
/// Factory helpers for building whole-response JSON syntax and JSON Schema constrained decoding instances
/// (`Plan 007` specification in `docs/007-json.md`).
/// </summary>
public static class JsonConstraint
{
    private static readonly ToolSchemaObject s_anyObjectSchema = CompileAnyObjectSchema();

    private static ToolSchemaObject CompileAnyObjectSchema()
    {
        using var doc = JsonDocument.Parse("""
        {
          "type": "object",
          "properties": {
            "value": {}
          }
        }
        """);
        return ToolSchema.FromOpenAiFunction("_any", doc.RootElement.Clone()).Arguments;
    }

    /// <summary>
    /// Creates a whole-response JSON constraint forcing the emitted tokens to satisfy the provided <see cref="ToolSchemaObject"/>.
    /// </summary>
    /// <param name="vocab">Model vocabulary view built from the active tokenizer.</param>
    /// <param name="schema">Compiled schema object root.</param>
    /// <param name="orderedProperties">When true, enforces strict property declaration ordering.</param>
    public static ITokenConstraint FromSchema(GrammarVocabulary vocab, ToolSchemaObject schema, bool orderedProperties = false)
    {
        ArgumentNullException.ThrowIfNull(vocab);
        ArgumentNullException.ThrowIfNull(schema);
        return new JsonSchemaOutputConstraint(vocab, schema, orderedProperties);
    }

    /// <summary>
    /// Creates a whole-response JSON constraint from a raw JSON Schema string (e.g. OpenAI function parameters schema).
    /// </summary>
    public static ITokenConstraint FromSchemaJson(GrammarVocabulary vocab, string jsonSchemaText, bool orderedProperties = false)
    {
        ArgumentNullException.ThrowIfNull(vocab);
        ArgumentException.ThrowIfNullOrEmpty(jsonSchemaText);
        using var doc = JsonDocument.Parse(jsonSchemaText);
        var schema = ToolSchema.FromOpenAiFunction("_schema", doc.RootElement.Clone()).Arguments;
        return new JsonSchemaOutputConstraint(vocab, schema, orderedProperties);
    }

    /// <summary>
    /// Creates a general whole-response JSON constraint enforcing syntactically valid JSON object (`{...}`) output.
    /// </summary>
    public static ITokenConstraint AnyJson(GrammarVocabulary vocab)
    {
        ArgumentNullException.ThrowIfNull(vocab);
        return new JsonSchemaOutputConstraint(vocab, s_anyObjectSchema, orderedProperties: false);
    }
}
