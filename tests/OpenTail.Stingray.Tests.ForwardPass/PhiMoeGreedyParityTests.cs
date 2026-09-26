
namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Phi-3.5-MoE (phimoe) greedy parity against llama-server (vendored tools/llama.cpp, same GGUF,
/// raw completion, temperature 0, top_k 1, -fa off; captured 2026-09-26). Covers the three things
/// the engine got wrong before: RMSNorm + bias (not LayerNorm) on attn/ffn/output norms, the
/// LM-head output.bias, and LongRoPE — per-pair short/long factor tensors chosen by context size
/// (short at -c 4096, long at -c 8192) plus rope.scaling.attn_factor 1.19024 on cos/sin.
/// </summary>
public sealed class PhiMoeGreedyParityTests : HeavyTestBase
{
    private const string ModelFile = "Phi-3.5-MoE-instruct-Q3_K_M.gguf";

    private const string LongPrompt =
        "The history of the city of Paris stretches back more than two thousand years. It began as a small settlement of a Celtic tribe known as the Parisii on an island in the Seine river. The Romans conquered the area in 52 BC and built a town called Lutetia, with a forum, baths, temples and an amphitheatre. During the Middle Ages, Paris became one of the largest and most important cities in Europe, a centre of learning with its famous university, and home to the royal court of the kings of France. The construction of the cathedral of Notre-Dame began in 1163 and continued for almost two centuries. In the following centuries the city grew rapidly, and in 1789 it became the stage of the French Revolution, which changed the course of European history. In the nineteenth century, under Napoleon III, the prefect Baron Haussmann rebuilt much of the city, creating wide boulevards, parks and";

    /// <summary>Short factors (context 4096 = original_context_length).</summary>
    [Fact]
    public void PhiMoe_ShortFactors_GreedyMatchesLlamaServer()
    {
        // " Paris. Paris is the capital of France, a country located in Western Europe. It is known for its rich history,"
        AssertGreedy("The capital of France is", ctx: 4096, expectedPromptLen: 5,
        [
            3681, 29889, 3681, 338, 278, 7483, 310, 3444, 29892, 263, 4234, 5982,
            297, 10504, 4092, 29889, 739, 338, 2998, 363, 967, 8261, 4955, 29892,
        ]);
    }

    /// <summary>Long factors (context 8192 &gt; 4096), 214-token prompt through batched prefill.</summary>
    [Fact]
    public void PhiMoe_LongFactors_GreedyMatchesLlamaServer()
    {
        AssertGreedy(LongPrompt, ctx: 8192, expectedPromptLen: 214,
        [
            970, 13814, 29889, 20628, 29892, 3681, 338, 263, 5534, 4272, 29892, 411, 263, 8261, 9257, 29892,
            263, 16984, 4665, 322, 263, 19821, 363, 1616, 29892, 13460, 322, 2723, 275, 457, 29889, 13,
        ]);
    }

    private static void AssertGreedy(string prompt, int ctx, int expectedPromptLen, int[] expected)
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this parity receipt.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        Assert.Equal("phimoe", Convert.ToString(model.Metadata["general.architecture"]));
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        var promptTokens = tokenizer.Encode(prompt); // add_bos_token=false for this checkpoint
        Assert.Equal(expectedPromptLen, promptTokens.Count);

        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: ctx);

        var logits = fwd.Prefill(promptTokens);
        var generated = new List<int>(expected.Length);
        int pos = promptTokens.Count;
        for (int i = 0; i < expected.Length; i++)
        {
            int next = Sampler.Greedy(logits);
            generated.Add(next);
            if (i + 1 < expected.Length) logits = fwd.Forward(next, pos++);
        }

        Console.WriteLine($"ctx {ctx}: {tokenizer.Decode(generated)}");
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
