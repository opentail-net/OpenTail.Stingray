using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// TEMPORARY diagnostic while capturing the llama.cpp reference continuation for Granite. Not the
/// admission receipt — see GraniteGreedyParityTests once the reference lands, this file will be
/// replaced with the real parity test (or removed if this diagnostic proves unnecessary).
/// </summary>
public sealed class GraniteDiagnosticTests
{
    private const string ModelFile = "granite-3.3-2b-instruct-Q4_K_M.gguf";
    private static readonly int[] s_promptTokens = [1318, 18926, 432, 45600, 438];

    [Fact]
    public void Granite_DumpGreedyContinuation()
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this diagnostic.");

        using var model = GgufModel.Open(path!);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        Assert.Equal(s_promptTokens, tokenizer.Encode("The capital of France is"));

        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048);

        var logits = fwd.Prefill(s_promptTokens);
        var generated = new List<int>(24);
        int pos = s_promptTokens.Length;
        for (int i = 0; i < 24; i++)
        {
            int next = Sampler.Greedy(logits);
            generated.Add(next);
            if (i + 1 < 24) logits = fwd.Forward(next, pos++);
        }

        string continuation = tokenizer.Decode(generated);
        Assert.Fail($"CONTINUATION: [{continuation}]  IDS: [{string.Join(",", generated)}]");
    }

    private static string? FindModel()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var candidate = Path.Combine(dir, "models", ModelFile);
            if (File.Exists(candidate)) return candidate;
            if (Directory.GetParent(dir) is not { } parent) break;
            dir = parent.FullName;
        }
        var external = Path.Combine(@"E:\models", ModelFile);
        return File.Exists(external) ? external : null;
    }
}
