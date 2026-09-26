
namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// DeepSeek-V2-Lite-Chat (deepseek2, MLA + shared/routed MoE) greedy parity against llama-server
/// (vendored tools/llama.cpp, same GGUF, temperature 0, -fa off; captured 2026-09-26) on the
/// chat-templated prompt the CLI builds. Guards the two bugs that made this architecture emit
/// garbage: the shared expert's real width (2 x 1408) and MLA decode's in-place Q reorder, which
/// only showed from the 2nd decoded position on (hence the prefill/decode consistency check).
/// </summary>
public sealed class DeepSeek2GreedyParityTests : HeavyTestBase
{
    private const string ModelFile = "DeepSeek-V2-Lite-Chat.Q2_K.gguf";

    // "<｜begin▁of▁sentence｜>User: The capital of France is\n\nAssistant:"
    private static readonly int[] s_promptTokens = [100000, 5726, 25, 429, 6077, 280, 7239, 317, 185, 185, 77398, 25];

    [Fact]
    public void DeepSeek2_GreedyContinuation_MatchesLlamaServer()
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this parity receipt.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        Assert.Equal("deepseek2", Convert.ToString(model.Metadata["general.architecture"]));
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.Equal(2816, hp.SharedExpertIntermediateDim);

        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048);

        // " Paris is the capital of France.\n\nWell, I hope you are not"
        int[] expected = [8913, 317, 254, 6077, 280, 7239, 13, 185, 185, 6636, 11, 304, 3655, 340, 418, 441];
        var logits = fwd.Prefill(s_promptTokens);
        var generated = new List<int>(expected.Length);
        int pos = s_promptTokens.Length;
        for (int i = 0; i < expected.Length; i++)
        {
            int next = Sampler.Greedy(logits);
            generated.Add(next);
            if (i + 1 < expected.Length) logits = fwd.Forward(next, pos++);
        }
        Assert.Equal(expected, generated);
    }

    [Fact]
    public void DeepSeek2_DecodeStepwise_AgreesWithSinglePassPrefill()
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this consistency check.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        using var backend = new CpuBackend();

        foreach (int n in new[] { 2, 3, s_promptTokens.Length })
        {
            var seq = s_promptTokens[..n];
            float[] single, stepwise;
            using (var f = new Engine.ForwardPass(model, backend, hp, maxContextLength: 256))
                single = f.Prefill(seq).ToArray();
            using (var f = new Engine.ForwardPass(model, backend, hp, maxContextLength: 256))
            {
                ReadOnlySpan<float> l = default;
                for (int i = 0; i < n; i++) l = f.Forward(seq[i], i);
                stepwise = l.ToArray();
            }
            int a = Array.IndexOf(single, single.Max()), b = Array.IndexOf(stepwise, stepwise.Max());
            // Before the in-place Q-reorder fix these disagreed from n = 2 (maxDiff ~16 logits).
            Assert.True(a == b, $"n={n}: prefill argmax {a} vs stepwise decode argmax {b}");
        }
    }

    private static string? FindModel()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            foreach (var sub in new[] { "models", Path.Combine("models", "_models") })
            {
                var candidate = Path.Combine(dir, sub, ModelFile);
                if (File.Exists(candidate)) return candidate;
            }
            if (Directory.GetParent(dir) is not { } parent) break;
            dir = parent.FullName;
        }
        foreach (var external in new[] { @"E:\models", @"K:\_other_models" })
        {
            var candidate = Path.Combine(external, ModelFile);
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }
}
