namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// GLM-4.5-Air (<c>glm4moe</c>; GLM-4.5-Air-Q2_K from unsloth: 128 experts, top-8, one shared expert, sigmoid gating with selection bias, leading dense layer)
/// teacher-forced against llama-server (vendored tools/llama.cpp, same GGUF, raw completion, temperature 0, top_k 1, -fa off, CPU; captured
/// 2026-10-04). Stage A of the handoff closure plan for <c>glm4moe</c>. The continuation is fed back token by token; our argmax must equal
/// llama-server's token wherever its top-1 over top-2 margin is at least <see cref="ConfidentMargin"/> nats. llama-server evaluates the prompt
/// without a prefix token for this GGUF (5 tokens for "The capital of France is"), and so do we.
/// </summary>
public sealed class GlmMoeGreedyParityTests : HeavyTestBase
{
    private const string ModelFile = "GLM-4.5-Air-Q2_K.gguf";

    private const string LongPrompt =
        "The history of the city of Paris stretches back more than two thousand years. It began as a small settlement of a Celtic tribe known as the Parisii on an island in the Seine river. The Romans conquered the area in 52 BC and built a town called Lutetia, with a forum, baths, temples and an amphitheatre. During the Middle Ages, Paris became one of the largest and most important cities in Europe, a centre of learning with its famous university, and home to the royal court of the kings of France. The construction of the cathedral of Notre-Dame began in 1163 and continued for almost two centuries. In the following centuries the city grew rapidly, and in 1789 it became the stage of the French Revolution, which changed the course of European history. In the nineteenth century, under Napoleon III, the prefect Baron Haussmann rebuilt much of the city, creating wide boulevards, parks and";

    private const float ConfidentMargin = 1.5f;

    [Fact]
    public void GlmAir_ShortPrompt_TeacherForcedMatchesLlamaServer()
    {
        AssertTeacherForced("The capital of France is", expectedPromptLen: 5,
        [
            12089, 13, 12089, 374, 279, 6722, 315, 9621, 13, 576, 6722, 315, 9621, 374, 12089, 13, 12089, 374, 279, 6722, 315, 9621,
        ],
        [
            3.182f, 0.927f, 0.161f, 2.849f, 1.126f, 0.322f, 0.429f, 1.976f, 1.505f, 0.334f, 3.038f, 2.903f, 3.288f, 2.806f, 3.202f, 1.219f, 0.814f, 5.241f, 5.940f, 8.517f, 6.981f, 6.853f,
        ]);
    }

    [Fact]
    public void GlmAir_LongPrompt_BatchedPrefill_TeacherForcedMatchesLlamaServer()
    {
        AssertTeacherForced(LongPrompt, expectedPromptLen: 190,
        [
            31209, 11, 323, 7086, 12089, 1181, 6481, 11089, 13, 576, 468, 3092, 301, 21884, 11, 5798, 369, 279, 220, 117786, 24, 4337, 594, 14578, 11, 6116, 279, 7735, 315, 279, 3283, 13,
        ],
        [
            2.005f, 0.230f, 2.177f, 0.419f, 0.497f, 0.069f, 0.674f, 0.577f, 1.102f, 0.072f, 0.603f, 9.324f, 10.754f, 5.669f, 0.186f, 0.479f, 0.606f, 6.276f, 1.480f, 8.021f, 11.156f, 0.477f, 1.616f, 5.369f, 3.744f, 1.531f, 1.160f, 1.699f, 5.448f, 0.703f, 4.651f, 1.955f,
        ]);
    }

    private static void AssertTeacherForced(string prompt, int expectedPromptLen, int[] reference, float[] margins)
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this parity receipt.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        Assert.Equal("glm4moe", Convert.ToString(model.Metadata["general.architecture"]));
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        Assert.Equal(128, hp.NumExperts);
        Assert.Equal(8, hp.NumActiveExperts);
        Assert.Equal(2, hp.ExpertGatingFunc);   // sigmoid probabilities, selected with exp_probs_b, weighted by the unbiased ones
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
