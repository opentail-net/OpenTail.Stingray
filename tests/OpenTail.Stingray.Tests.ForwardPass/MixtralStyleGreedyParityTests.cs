namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Mixtral-style MoE (<c>llama</c> architecture with <c>expert_count &gt; 0</c>; TinyLlama-4x1.1B-MoE Q4_K_M: 4 experts, top-2, routing weights
/// renormalised, no shared expert, Q4_K gate/up and Q6_K down) teacher-forced against llama-server (vendored tools/llama.cpp, same GGUF, raw
/// completion with BOS, temperature 0, top_k 1, -fa off, CPU; captured 2026-10-04). Stage A of the handoff closure plan for the
/// <c>llama+experts</c> family key. llama-server's continuation is fed back token by token; our argmax must equal its token wherever its
/// top-1 over top-2 margin is at least <see cref="ConfidentMargin"/> nats (the margin the LFM2 and Nemotron-H receipts use).
/// Found by this checkpoint: <c>NormalizeMoeTopKWeights</c> defaulted to false for <c>llama</c>, but llama.cpp's llama builder renormalises.
/// </summary>
public sealed class MixtralStyleGreedyParityTests : HeavyTestBase
{
    private const string ModelFile = "TinyLLama-4x1.1B-MoE.Q4_K_M.gguf";

    private const string LongPrompt =
        "The history of the city of Paris stretches back more than two thousand years. It began as a small settlement of a Celtic tribe known as the Parisii on an island in the Seine river. The Romans conquered the area in 52 BC and built a town called Lutetia, with a forum, baths, temples and an amphitheatre. During the Middle Ages, Paris became one of the largest and most important cities in Europe, a centre of learning with its famous university, and home to the royal court of the kings of France. The construction of the cathedral of Notre-Dame began in 1163 and continued for almost two centuries. In the following centuries the city grew rapidly, and in 1789 it became the stage of the French Revolution, which changed the course of European history. In the nineteenth century, under Napoleon III, the prefect Baron Haussmann rebuilt much of the city, creating wide boulevards, parks and";

    private const float ConfidentMargin = 1.5f;

    [Fact]
    public void MixtralStyle_ShortPrompt_TeacherForcedMatchesLlamaServer()
    {
        AssertTeacherForced("The capital of France is", expectedPromptLen: 6,
        [
            263, 4272, 297, 278, 7062, 310, 278, 4234, 29889, 13, 13, 29906, 29889, 450, 7483, 310, 13616, 338, 278, 7483, 310, 278,
        ],
        [
            0.115f, 0.984f, 0.552f, 2.036f, 0.554f, 0.512f, 0.304f, 3.087f, 0.280f, 0.089f, 1.029f, 0.013f, 3.613f, 1.640f, 1.137f, 2.140f, 0.311f, 0.571f, 0.990f, 4.233f, 2.049f, 0.043f,
        ]);
    }

    [Fact]
    public void MixtralStyle_LongPrompt_BatchedPrefill_TeacherForcedMatchesLlamaServer()
    {
        AssertTeacherForced(LongPrompt, expectedPromptLen: 215,
        [
            17161, 575, 29892, 322, 278, 13834, 15484, 316, 425, 1281, 2616, 311, 29889, 512, 278, 3252, 7268, 621, 6462, 29892, 278, 4272, 3897, 278, 4818, 310, 278, 3186, 29892, 322, 278, 3186,
        ],
        [
            0.076f, 5.506f, 0.314f, 1.940f, 0.303f, 0.199f, 0.106f, 0.840f, 3.130f, 1.042f, 7.753f, 7.792f, 0.022f, 0.113f, 0.054f, 0.420f, 0.395f, 8.941f, 5.841f, 2.825f, 0.888f, 3.117f, 0.059f, 0.076f, 0.148f, 3.063f, 1.994f, 1.246f, 1.229f, 0.061f, 0.817f, 0.509f,
        ]);
    }

    private static void AssertTeacherForced(string prompt, int expectedPromptLen, int[] reference, float[] margins)
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this parity receipt.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        Assert.Equal("llama", Convert.ToString(model.Metadata["general.architecture"]));
        Assert.True(Convert.ToInt32(model.Metadata["llama.expert_count"]) > 0, "this receipt is for the llama + experts (Mixtral-style) family");
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.Equal(4, hp.NumExperts);
        Assert.Equal(2, hp.NumActiveExperts);
        Assert.True(hp.NormalizeMoeTopKWeights, "Mixtral renormalises its top-2 routing weights");
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        var encoded = tokenizer.Encode(prompt);
        var promptTokens = tokenizer.AddBosToken ? new List<int> { tokenizer.BosTokenId }.Concat(encoded).ToList() : encoded.ToList(); // llama-server adds BOS
        Assert.Equal(expectedPromptLen, promptTokens.Count);

        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 512);

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
