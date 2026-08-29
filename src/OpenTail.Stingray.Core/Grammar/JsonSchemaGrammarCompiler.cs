using System.Collections.Concurrent;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Text.Json.Nodes;

namespace OpenTail.Stingray.Core.Grammar;

/// <summary>
/// Fast, narrow, inference-focused compiler converting JSON Schema elements and C# DTO types into compact <see cref="GrammarStateMachine"/> instances.
/// </summary>
public static class JsonSchemaGrammarCompiler
{
    private static readonly ConcurrentDictionary<Type, GrammarStateMachine> s_typeCache = new();

    /// <summary>
    /// Compiles a C# DTO type into a cached <see cref="GrammarStateMachine"/>.
    /// </summary>
    public static GrammarStateMachine CompileForType<[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicProperties)] T>()
    {
        var targetType = typeof(T);
        return s_typeCache.GetOrAdd(targetType, _ =>
        {
            var properties = new List<SchemaPropertyNode>();
            foreach (var prop in targetType.GetProperties())
            {
                if (!prop.CanRead) continue;

                string propName = prop.Name;
                string typeName = GetJsonType(prop.PropertyType);
                List<string>? enumValues = null;

                if (prop.PropertyType.IsEnum)
                {
                    enumValues = new List<string>(Enum.GetNames(prop.PropertyType));
                }

                properties.Add(new SchemaPropertyNode(propName, typeName, IsRequiredType(prop.PropertyType), enumValues));
            }
            return new GrammarStateMachine(properties);
        });
    }

    /// <summary>
    /// Compiles a JSON Schema root object into a compact <see cref="GrammarStateMachine"/>.
    /// </summary>
    public static GrammarStateMachine Compile(JsonElement schemaElement)
    {
        var properties = CompileObjectProperties(schemaElement);
        return new GrammarStateMachine(properties);
    }

    private static List<SchemaPropertyNode> CompileObjectProperties(JsonElement schemaElement)
    {
        var properties = new List<SchemaPropertyNode>();

        if (schemaElement.ValueKind == JsonValueKind.Object)
        {
            var requiredSet = new HashSet<string>(StringComparer.Ordinal);
            if (schemaElement.TryGetProperty("required", out var reqElement) && reqElement.ValueKind == JsonValueKind.Array)
            {
                foreach (var item in reqElement.EnumerateArray())
                {
                    if (item.ValueKind == JsonValueKind.String && item.GetString() is { } reqName)
                    {
                        requiredSet.Add(reqName);
                    }
                }
            }

            if (schemaElement.TryGetProperty("properties", out var propsElement) && propsElement.ValueKind == JsonValueKind.Object)
            {
                foreach (var prop in propsElement.EnumerateObject())
                {
                    string propName = prop.Name;
                    string typeName = "string";
                    List<string>? enumValues = null;
                    List<SchemaPropertyNode>? childProps = null;
                    SchemaPropertyNode? arrayItemSchema = null;

                    if (prop.Value.TryGetProperty("type", out var typeElem) && typeElem.ValueKind == JsonValueKind.String)
                    {
                        typeName = typeElem.GetString() ?? "string";
                    }

                    if (prop.Value.TryGetProperty("enum", out var enumElem) && enumElem.ValueKind == JsonValueKind.Array)
                    {
                        enumValues = new List<string>();
                        foreach (var ev in enumElem.EnumerateArray())
                        {
                            if (ev.ValueKind == JsonValueKind.String && ev.GetString() is { } evStr)
                            {
                                enumValues.Add(evStr);
                            }
                        }
                    }

                    if (string.Equals(typeName, "object", StringComparison.OrdinalIgnoreCase))
                    {
                        childProps = CompileObjectProperties(prop.Value);
                    }

                    if (string.Equals(typeName, "array", StringComparison.OrdinalIgnoreCase) &&
                        prop.Value.TryGetProperty("items", out var itemsElem))
                    {
                        var itemsProps = CompileObjectProperties(itemsElem);
                        arrayItemSchema = new SchemaPropertyNode("item", "object", true, ChildProperties: itemsProps);
                    }

                    bool isRequired = requiredSet.Contains(propName);
                    properties.Add(new SchemaPropertyNode(propName, typeName, isRequired, enumValues, childProps, arrayItemSchema));
                }
            }
        }

        return properties;
    }

    /// <summary>
    /// Compiles a JSON Schema text string into a <see cref="GrammarStateMachine"/>.
    /// </summary>
    public static GrammarStateMachine CompileJson(string jsonSchemaText)
    {
        ArgumentException.ThrowIfNullOrEmpty(jsonSchemaText);
        using var doc = JsonDocument.Parse(jsonSchemaText);
        return Compile(doc.RootElement);
    }

    /// <summary>
    /// Benchmark helper returning schema compilation duration.
    /// </summary>
    public static GrammarStateMachine CompileWithBenchmark(JsonElement schemaElement, out TimeSpan compilationDuration)
    {
        var sw = Stopwatch.StartNew();
        var sm = Compile(schemaElement);
        sw.Stop();
        compilationDuration = sw.Elapsed;
        return sm;
    }

    private static string GetJsonType(Type t)
    {
        var underlying = Nullable.GetUnderlyingType(t) ?? t;
        if (underlying == typeof(string) || underlying.IsEnum) return "string";
        if (underlying == typeof(bool)) return "boolean";
        if (underlying == typeof(int) || underlying == typeof(long) || underlying == typeof(short)) return "integer";
        if (underlying == typeof(float) || underlying == typeof(double) || underlying == typeof(decimal)) return "number";
        if (underlying.IsArray || (underlying.IsGenericType && underlying.GetGenericTypeDefinition() == typeof(List<>))) return "array";
        return "object";
    }

    private static bool IsRequiredType(Type t)
    {
        return !t.IsGenericType || t.GetGenericTypeDefinition() != typeof(Nullable<>);
    }
}
