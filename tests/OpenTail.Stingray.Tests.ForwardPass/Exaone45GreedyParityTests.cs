
namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// EXAONE 4.5 33B (exaone4, 64 layers) greedy parity against llama-server (vendored
/// tools/llama.cpp, same GGUF, raw completion, temperature 0, -c 4096, -fa off; captured
/// 2026-09-26). Exercises what the 1.2B receipt (Exaone4VerifyTemp) cannot: the 64-layer
/// checkpoint's 3:1 SWA pattern with RoPE ONLY on SWA layers (global layers are NoPE), plus the
/// post-norm-only trunk through batched prefill (PrefillCore used to crash on the missing
/// attn_norm tensor).
/// </summary>
public sealed class Exaone45GreedyParityTests : HeavyTestBase
{
    private const string ModelFile = "EXAONE-4.5-33B-Q4_K_M.gguf";

    private const string LongPrompt =
        "The history of the city of Paris stretches back more than two thousand years. It began as a small settlement of a Celtic tribe known as the Parisii on an island in the Seine river. The Romans conquered the area in 52 BC and built a town called Lutetia, with a forum, baths, temples and an amphitheatre. During the Middle Ages, Paris became one of the largest and most important cities in Europe, a centre of learning with its famous university, and home to the royal court of the kings of France. The construction of the cathedral of Notre-Dame began in 1163 and continued for almost two centuries. In the following centuries the city grew rapidly, and in 1789 it became the stage of the French Revolution, which changed the course of European history. In the nineteenth century, under Napoleon III, the prefect Baron Haussmann rebuilt much of the city, creating wide boulevards, parks and";

    [Fact]
    public void Exaone45_ShortPrompt_GreedyMatchesLlamaServer()
    {
        // " Paris.  \nThe capital of Germany is Berlin.  "
        AssertGreedy("The capital of France is", 5,
            [12229, 375, 33, 560, 1320, 7304, 670, 9625, 772, 20133, 375, 33]);
    }

    [Fact]
    public void Exaone45_LongPrompt_GreedyMatchesLlamaServer()
    {
        AssertGreedy(LongPrompt, 196,
            [8394, 12958, 375, 115476, 39623, 6791, 373, 12229, 925, 19848, 956, 9625, 2418, 118534, 373, 1224]);
    }

    private static void AssertGreedy(string prompt, int expectedPromptLen, int[] expected)
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this parity receipt.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        Assert.Equal("exaone4", Convert.ToString(model.Metadata["general.architecture"]));
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.True(hp.RopeOnlySwaLayers, "64-layer exaone4 must rope only its SWA layers");
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        var promptTokens = tokenizer.Encode(prompt);
        Assert.Equal(expectedPromptLen, promptTokens.Count);

        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 4096);

        var logits = fwd.Prefill(promptTokens);
        var generated = new List<int>(expected.Length);
        int pos = promptTokens.Count;
        for (int i = 0; i < expected.Length; i++)
        {
            int next = Sampler.Greedy(logits);
            generated.Add(next);
            if (i + 1 < expected.Length) logits = fwd.Forward(next, pos++);
        }

        Console.WriteLine(tokenizer.Decode(generated));
        Assert.Equal(expected, generated);
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
