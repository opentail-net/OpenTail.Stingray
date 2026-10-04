namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Llama 4 Scout (<c>llama4</c>; Llama-4-Scout-17B-16E-Instruct Q3_K_M from unsloth, two shards: 16 experts, top-1 with sigmoid gating, one shared expert,
/// NoPE every 4th layer, parameter-free L2 QK-norm) teacher-forced against llama-server (vendored tools/llama.cpp, same GGUF, raw completion with BOS,
/// temperature 0, top_k 1, -fa off, CPU; captured 2026-10-04). Stage A of the handoff closure plan for <c>llama4</c>. Prompts stay far below 8192 tokens:
/// chunked attention and attention-temperature tuning are not implemented, and both are the identity below that length.
/// </summary>
public sealed class LlamaFourGreedyParityTests : HeavyTestBase
{
    private const string ModelFile = "Llama-4-Scout-17B-16E-Instruct-Q3_K_M-00001-of-00002.gguf";

    private const string LongPrompt =
        "The history of the city of Paris stretches back more than two thousand years. It began as a small settlement of a Celtic tribe known as the Parisii on an island in the Seine river. The Romans conquered the area in 52 BC and built a town called Lutetia, with a forum, baths, temples and an amphitheatre. During the Middle Ages, Paris became one of the largest and most important cities in Europe, a centre of learning with its famous university, and home to the royal court of the kings of France. The construction of the cathedral of Notre-Dame began in 1163 and continued for almost two centuries. In the following centuries the city grew rapidly, and in 1789 it became the stage of the French Revolution, which changed the course of European history. In the nineteenth century, under Napoleon III, the prefect Baron Haussmann rebuilt much of the city, creating wide boulevards, parks and";

    private const float ConfidentMargin = 1.5f;

    [Fact]
    public void Scout_ShortPrompt_TeacherForcedMatchesLlamaServer()
    {
        AssertTeacherForced("The capital of France is", expectedPromptLen: 6,
        [
            13796, 26, 13796, 373, 262, 5420, 5376, 393, 2265, 24, 24739, 24, 341, 11773, 26, 589, 427, 161580, 47214, 373, 262, 24093,
        ],
        [
            2.647f, 1.325f, 0.261f, 1.623f, 0.108f, 2.808f, 3.328f, 6.201f, 7.546f, 6.744f, 9.684f, 8.003f, 3.376f, 3.921f, 3.135f, 5.028f, 3.039f, 13.626f, 9.310f, 0.133f, 0.757f, 6.161f,
        ]);
    }

    [Fact]
    public void Scout_LongPrompt_BatchedPrefill_TeacherForcedMatchesLlamaServer()
    {
        AssertTeacherForced(LongPrompt, expectedPromptLen: 190,
        [
            70366, 24, 341, 262, 9888, 129838, 1829, 26, 608, 290, 121644, 18654, 24, 13796, 15068, 262, 17488, 323, 2265, 24, 24739, 341, 11773, 24, 341, 262, 9985, 323, 17025, 100540, 26, 21388,
        ],
        [
            0.428f, 0.814f, 2.203f, 0.930f, 0.008f, 2.372f, 2.588f, 1.278f, 0.095f, 1.503f, 1.417f, 4.346f, 1.410f, 4.104f, 1.593f, 1.251f, 0.613f, 0.415f, 0.593f, 0.440f, 4.685f, 1.026f, 1.907f, 2.850f, 1.828f, 0.862f, 1.415f, 6.374f, 1.796f, 1.557f, 0.504f, 0.860f,
        ]);
    }

    private static void AssertTeacherForced(string prompt, int expectedPromptLen, int[] reference, float[] margins)
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this parity receipt.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        Assert.Equal("llama4", Convert.ToString(model.Metadata["general.architecture"]));
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.Equal(16, hp.NumExperts);
        Assert.Equal(1, hp.NumActiveExperts);
        Assert.True(hp.UseSigmoidGating, "Llama 4 weights its single routed expert with a sigmoid");
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        var encoded = tokenizer.Encode(prompt);
        var promptTokens = tokenizer.AddBosToken ? new List<int> { tokenizer.BosTokenId }.Concat(encoded).ToList() : encoded.ToList(); // llama-server adds BOS
        Assert.Equal(expectedPromptLen, promptTokens.Count);

        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 1024);

        var logits = fwd.Prefill(promptTokens);
        int pos = promptTokens.Count, confident = 0, nearTieMisses = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            int ours = Sampler.Greedy(logits);
            if (margins[i] >= ConfidentMargin)
            {
                confident++;
                Assert.True(ours == reference[i], $"step {i}: ours {ours} vs llama-server {reference[i]} at a confident margin of {margins[i]} nats");
            }
            else if (ours != reference[i]) nearTieMisses++;
            if (i + 1 < reference.Length) logits = fwd.Forward(reference[i], pos++);
        }

        Console.WriteLine($"{confident} confident positions matched, {nearTieMisses} near-tie differences of {reference.Length}");
        Assert.True(confident >= 5, "too few confident positions to be evidence");
    }

    private static string? FindModel()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            foreach (var sub in new[] { "models", Path.Combine("models", "_models"), Path.Combine("models", "_models", "Q3_K_M") })
            {
                var candidate = Path.Combine(dir, sub, ModelFile);
                if (File.Exists(candidate)) return candidate;
            }
            if (Directory.GetParent(dir) is not { } parent) break;
            dir = parent.FullName;
        }
        foreach (var external in new[] { @"H:\_models\Q3_K_M", @"H:\_models", @"E:\models", @"K:\_other_models" })
        {
            var candidate = Path.Combine(external, ModelFile);
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }
}
