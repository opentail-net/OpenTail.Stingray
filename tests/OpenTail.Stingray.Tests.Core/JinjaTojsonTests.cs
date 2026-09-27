using System.Text.Json;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// <c>tojson</c> must format like llama.cpp's jinja engine (<c>common/jinja/value.cpp</c> <c>tojson</c> /
/// <c>value_to_json_internal</c>), which follows Python's <c>json.dumps</c>: <c>", "</c> between items and <c>": "</c>
/// after keys, and with <c>indent</c> one element per line with <c>","</c> item separators. Until 2026-09-27 ours wrote
/// compact <c>,</c> / <c>:</c>, so tool definitions rendered by templates like Qwen3's differed from llama.cpp's
/// prompt (docs/103 item 8).
/// </summary>
public sealed class JinjaTojsonTests
{
    private static string Render(string source, Dictionary<string, object?> ctx) => new JinjaChatTemplate(source).Render(ctx);

    private static Dictionary<string, object?> Sample() => new()
    {
        ["name"] = "get_weather",
        ["args"] = new List<object?> { 1L, "a", true, null },
        ["nested"] = new Dictionary<string, object?> { ["k"] = "v" },
    };

    [Fact]
    public void Tojson_Default_UsesPythonSeparators()
    {
        Assert.Equal("""{"name": "get_weather", "args": [1, "a", true, null], "nested": {"k": "v"}}""",
            Render("{{ x | tojson }}", new() { ["x"] = Sample() }));
    }

    [Fact]
    public void Tojson_Indent_OneElementPerLine()
    {
        string expected = "{\n    \"name\": \"get_weather\",\n    \"args\": [\n        1,\n        \"a\",\n        true,\n        null\n    ],\n    \"nested\": {\n        \"k\": \"v\"\n    }\n}";
        Assert.Equal(expected, Render("{{ x | tojson(indent=4) }}", new() { ["x"] = Sample() }));
    }

    [Fact]
    public void Tojson_EmptyContainers_AndEscapes()
    {
        var x = new Dictionary<string, object?>
        {
            ["e"] = new List<object?>(),
            ["o"] = new Dictionary<string, object?>(),
            ["s"] = "q\"b\\n\n\t\b\f",
        };
        Assert.Equal("""{"e": [], "o": {}, "s": "q\"b\\n\n\t\b\f"}""", Render("{{ x | tojson }}", new() { ["x"] = x }));
    }

    [Fact]
    public void Tojson_JsonElement_IsReserializedNotPassedThroughRaw()
    {
        using var doc = JsonDocument.Parse("""{"a":1,"b":[true,{"c":"d"}]}""");
        Assert.Equal("""{"a": 1, "b": [true, {"c": "d"}]}""", Render("{{ x | tojson }}", new() { ["x"] = doc.RootElement.Clone() }));
    }

    /// <summary>
    /// End to end on a real template: Qwen3-0.6B's own chat template with one tool must render byte for byte what
    /// <c>llama-server --jinja</c> <c>/apply-template</c> returned for the same request (captured 2026-09-27).
    /// </summary>
    [Fact]
    public void Qwen3Template_WithTool_MatchesLlamaServerApplyTemplate()
    {
        string? path = FindModel("Qwen3-0.6B-Q8_0.gguf");
        if (path is null) Assert.Skip("Qwen3-0.6B-Q8_0.gguf not found.");
        using var model = GgufModel.Open(path);
        var template = GgufTokenizer.FromGgufModel(model).ChatTemplate;
        Assert.NotNull(template);

        var tool = new Dictionary<string, object?>
        {
            ["type"] = "function",
            ["function"] = new Dictionary<string, object?>
            {
                ["name"] = "get_weather",
                ["description"] = "Get weather",
                ["parameters"] = new Dictionary<string, object?>
                {
                    ["type"] = "object",
                    ["properties"] = new Dictionary<string, object?>
                    {
                        ["city"] = new Dictionary<string, object?> { ["type"] = "string", ["description"] = "City name" },
                        ["units"] = new Dictionary<string, object?> { ["type"] = "string", ["enum"] = new List<object?> { "c", "f" } },
                    },
                    ["required"] = new List<object?> { "city" },
                },
            },
        };
        var ctx = new Dictionary<string, object?>
        {
            ["messages"] = new List<object?>
            {
                new Dictionary<string, object?> { ["role"] = "system", ["content"] = "You are helpful." },
                new Dictionary<string, object?> { ["role"] = "user", ["content"] = "Weather in Paris?" },
            },
            ["tools"] = new List<object?> { tool },
            ["add_generation_prompt"] = true,
        };

        const string expected = """
            <|im_start|>system
            You are helpful.

            # Tools

            You may call one or more functions to assist with the user query.

            You are provided with function signatures within <tools></tools> XML tags:
            <tools>
            {"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}, "units": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}}
            </tools>

            For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
            <tool_call>
            {"name": <function-name>, "arguments": <args-json-object>}
            </tool_call><|im_end|>
            <|im_start|>user
            Weather in Paris?<|im_end|>
            <|im_start|>assistant

            """;
        Assert.Equal(expected.ReplaceLineEndings("\n"), template!.Render(ctx));
    }

    private static string? FindModel(string file)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            foreach (var sub in new[] { "models", Path.Combine("models", "_models") })
            {
                var candidate = Path.Combine(dir, sub, file);
                if (File.Exists(candidate)) return candidate;
            }
            if (Directory.GetParent(dir) is not { } parent) break;
            dir = parent.FullName;
        }
        return null;
    }
}
