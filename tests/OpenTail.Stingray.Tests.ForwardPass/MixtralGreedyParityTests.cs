namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Mixtral-style MoE (<c>llama</c> architecture with <c>expert_count &gt; 0</c>; Nous-Hermes-2-Mixtral-8x7B-DPO i1-Q4_K_S: 8 experts, top-2, routing weights
/// renormalised, no shared expert, Q4_K gate/up and Q6_K down) teacher-forced against llama-server (vendored tools/llama.cpp, same GGUF, raw
/// completion with BOS, temperature 0, top_k 1, -fa off, CPU; captured 2026-10-04). Stage A of the handoff closure plan for the
/// <c>llama+experts</c> family key. llama-server's continuation is fed back token by token; our argmax must equal its token wherever its
/// top-1 over top-2 margin is at least <see cref="ConfidentMargin"/> nats (the margin the LFM2 and Nemotron-H receipts use).
/// Found by this checkpoint: <c>NormalizeMoeTopKWeights</c> defaulted to false for <c>llama</c>, but llama.cpp's llama builder renormalises.
/// </summary>
public sealed class MixtralGreedyParityTests : HeavyTestBase
{
    private const string ModelFile = "Nous-Hermes-2-Mixtral-8x7B-DPO.i1-Q4_K_S.gguf";

    private const string LongPrompt =
        "The history of the city of Paris stretches back more than two thousand years. It began as a small settlement of a Celtic tribe known as the Parisii on an island in the Seine river. The Romans conquered the area in 52 BC and built a town called Lutetia, with a forum, baths, temples and an amphitheatre. During the Middle Ages, Paris became one of the largest and most important cities in Europe, a centre of learning with its famous university, and home to the royal court of the kings of France. The construction of the cathedral of Notre-Dame began in 1163 and continued for almost two centuries. In the following centuries the city grew rapidly, and in 1789 it became the stage of the French Revolution, which changed the course of European history. In the nineteenth century, under Napoleon III, the prefect Baron Haussmann rebuilt much of the city, creating wide boulevards, parks and";

    private const float ConfidentMargin = 1.5f;

    [Fact]
    public void Mixtral8x7b_ShortPrompt_TeacherForcedMatchesLlamaServer()
    {
        AssertTeacherForced("The capital of France is", expectedPromptLen: 6,
        [
            264, 2990, 369, 349, 2651, 354, 871, 9689, 28725, 3340, 28725, 304, 5679, 28723, 5465, 349, 264, 2990, 369, 349, 2173, 302,
        ],
        [
            0.244f, 0.862f, 0.381f, 0.507f, 0.574f, 2.178f, 2.515f, 0.364f, 0.291f, 0.099f, 1.136f, 1.755f, 1.349f, 2.946f, 0.285f, 1.426f, 0.252f, 0.783f, 2.587f, 0.464f, 0.343f, 6.870f,
        ]);
    }

    [Fact]
    public void Mixtral8x7b_LongPrompt_BatchedPrefill_TeacherForcedMatchesLlamaServer()
    {
        AssertTeacherForced(LongPrompt, expectedPromptLen: 209,
        [
            4524, 4585, 28725, 304, 1287, 302, 272, 2990, 28809, 28713, 1080, 8376, 2533, 17181, 654, 4429, 1938, 456, 3216, 28725, 2490, 272, 413, 2728, 301, 19895, 28725, 272, 6389, 18924, 420, 1331,
        ],
        [
            0.997f, 10.934f, 0.196f, 2.194f, 0.039f, 1.789f, 0.530f, 2.465f, 0.715f, 9.949f, 1.033f, 2.718f, 0.377f, 7.160f, 0.251f, 0.653f, 1.798f, 2.283f, 0.412f, 0.672f, 2.565f, 7.871f, 0.638f, 7.711f, 10.527f, 3.459f, 0.836f, 2.771f, 0.458f, 5.679f, 2.690f, 13.571f,
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
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        Assert.Equal(8, hp.NumExperts);
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

    private static string? FindModel() => OpenTail.Stingray.Engine.Verification.ModelLocator.FindOrReport(ModelFile);
}
