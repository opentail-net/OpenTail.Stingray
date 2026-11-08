using System.Linq;
using System.Text;
using System.Text.Json;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Core.Grammar;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Qwen 3.8 plan, Phase 6 (the round trip, the part that does not need the server): the real qwen4exp checkpoint, given a
/// declared tool, writes a tool call that the server's adapter parses and the schema validator accepts. Greedy, per-token,
/// the reference path. Local investigation only; the model is not admitted. Skipped when the trunk is absent.
/// </summary>
public sealed class Qwen4ExpToolCallModelTests
{
    private const string ToolJson =
        "{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get the current weather for a city.\"," +
        "\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}";

    private const string ParametersJson = "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}";

    private static string? FindModel() => ModelLocator.FindDeep("Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf");

    [Fact]
    public void RealModel_WritesAToolCall_ThatTheAdapterParsesAndTheSchemaAccepts()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        var prompt = new StringBuilder();
        prompt.Append("<|im_start|>system\nYou are a helpful assistant.\n\n# Tools\n\n");
        prompt.Append("You may call one or more functions to assist with the user query.\n\n");
        prompt.Append("You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n");
        prompt.Append(ToolJson).Append("\n</tools>\n\n");
        prompt.Append("For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n");
        prompt.Append("<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n");
        prompt.Append("<|im_start|>user\nWhat is the weather in Paris? Use the tool.<|im_end|>\n");
        prompt.Append("<|im_start|>assistant\n<think>\n\n</think>\n\n");

        using var model = GgufModel.Open(path!);
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        using var pass = new Qwen4ExpForwardPass(model, hp, tensors);
        var tok = GgufTokenizer.FromGgufModel(model);

        int[] ids = tok.Encode(prompt.ToString()).ToArray();
        Console.WriteLine($"TOOL_PROMPT_TOKENS {ids.Length}");

        var state = pass.CreateSequenceState();
        int pos = 0;
        ReadOnlySpan<float> logits = default;
        foreach (int id in ids) logits = pass.Forward(id, pos++, state);

        var generated = new List<int>();
        for (int s = 0; s < 120; s++)
        {
            int best = 0;
            for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
            if (tok.EogTokenIds.Contains(best)) break;
            generated.Add(best);
            logits = pass.Forward(best, pos++, state);
        }

        string raw = tok.Decode(generated);
        Console.WriteLine($"TOOL_RAW {raw}");

        var result = ToolCallAdapterRegistry.Get("qwen2").Parse(raw);
        Console.WriteLine($"TOOL_CALLS {string.Join(" | ", result.Calls.Select(c => c.Name))}");
        Assert.NotEmpty(result.Calls);

        var call = result.Calls[0];
        Assert.Equal("get_weather", call.Name);
        using var doc = JsonDocument.Parse(ParametersJson);
        var schema = ToolSchema.FromOpenAiFunction("get_weather", doc.RootElement.Clone());
        var errors = ToolCallArgumentValidator.Validate(call, schema);
        Assert.Empty(errors);
    }
}
