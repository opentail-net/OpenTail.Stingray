namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Arcee Trinity Mini (<c>afmoe</c>; arcee-ai/Trinity-Mini-GGUF Q4_K_M: 128 experts top-8 + 1 shared, sigmoid gating with bias, 2 leading dense layers,
/// sliding-window layers, attention output gate) teacher-forced against llama-server (vendored tools/llama.cpp, same GGUF, raw completion, temperature 0,
/// top_k 1, -fa off, CPU; captured 2026-10-04). Stage A of the handoff closure plan for the extension wave. The continuation is fed back token by token; our
/// argmax must equal llama-server's token wherever its top-1 over top-2 margin is at least <see cref="ConfidentMargin"/> nats. No prefix token
/// (add_bos_token=false), 5 tokens for the short prompt.
/// </summary>
public sealed class AfmoeGreedyParityTests : HeavyTestBase
{
    private const string ModelFile = "Trinity-Mini-Q4_K_M.gguf";

    private const string LongPrompt =
        "The history of the city of Paris stretches back more than two thousand years. It began as a small settlement of a Celtic tribe known as the Parisii on an island in the Seine river. The Romans conquered the area in 52 BC and built a town called Lutetia, with a forum, baths, temples and an amphitheatre. During the Middle Ages, Paris became one of the largest and most important cities in Europe, a centre of learning with its famous university, and home to the royal court of the kings of France. The construction of the cathedral of Notre-Dame began in 1163 and continued for almost two centuries. In the following centuries the city grew rapidly, and in 1789 it became the stage of the French Revolution, which changed the course of European history. In the nineteenth century, under Napoleon III, the prefect Baron Haussmann rebuilt much of the city, creating wide boulevards, parks and";

    private const float ConfidentMargin = 1.5f;

    [Fact]
    public void TrinityMini_ShortPrompt_TeacherForcedMatchesLlamaServer()
    {
        AssertTeacherForced("The capital of France is", expectedPromptLen: 5,
        [
            8849, 45, 405, 4533, 323, 6364, 351, 8849, 45, 405, 4533, 323, 6364, 351, 8849, 45, 405, 4533, 323, 6364, 351, 8849,
        ],
        [
            5.345f, 0.242f, 0.755f, 0.317f, 3.821f, 0.330f, 4.782f, 3.961f, 2.016f, 2.051f, 5.446f, 4.531f, 5.469f, 8.845f, 8.305f, 2.760f, 3.100f, 7.615f, 8.342f, 7.393f, 8.992f, 8.646f, 
        ]);
    }

    [Fact]
    public void TrinityMini_LongPrompt_BatchedPrefill_TeacherForcedMatchesLlamaServer()
    {
        AssertTeacherForced(LongPrompt, expectedPromptLen: 183,
        [
            1031, 8218, 45, 620, 296, 29082, 3883, 43, 8849, 493, 296, 4870, 323, 953, 1117, 15827, 43, 326, 1119, 296, 7353, 3295, 3562, 394, 3624, 290, 2486, 12233, 323, 296, 4913, 6953,
        ],
        [
            0.333f, 1.158f, 1.779f, 0.608f, 3.091f, 2.378f, 4.341f, 1.385f, 2.309f, 0.771f, 0.368f, 0.015f, 5.046f, 0.633f, 1.355f, 5.405f, 0.498f, 0.202f, 0.134f, 0.078f, 1.294f, 7.707f, 7.800f, 1.407f, 1.941f, 0.103f, 0.179f, 2.711f, 1.402f, 0.262f, 1.641f, 1.671f, 
        ]);
    }

    private static void AssertTeacherForced(string prompt, int expectedPromptLen, int[] reference, float[] margins)
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this parity receipt.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        Assert.Equal("afmoe", Convert.ToString(model.Metadata["general.architecture"]));
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
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
        Assert.True(confident >= 4, "too few confident positions to be evidence");
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
        foreach (var external in new[] { @"H:\_models", @"E:\models", @"K:\_other_models" })
        {
            var candidate = Path.Combine(external, ModelFile);
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }
}
