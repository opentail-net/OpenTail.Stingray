namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Hunyuan-A13B-Instruct (<c>hunyuan-moe</c>; DevQuasar Q3_K_S: 64 routed experts + one shared expert, QK-norm after RoPE) teacher-forced against llama-server
/// (vendored tools/llama.cpp, same GGUF, raw completion, temperature 0, top_k 1, -fa off, CPU; captured 2026-10-04). Stage A of the handoff closure plan
/// for the extension wave. The continuation is fed back token by token; our argmax must equal llama-server's token wherever its top-1 over top-2 margin is at
/// least <see cref="ConfidentMargin"/> nats. llama-server evaluates the short prompt without a prefix token (5 tokens), and so do we.
/// </summary>
public sealed class HunyuanMoeGreedyParityTests : HeavyTestBase
{
    private const string ModelFile = "tencent.Hunyuan-A13B-Instruct.Q3_K_S.gguf";

    private const string LongPrompt =
        "The history of the city of Paris stretches back more than two thousand years. It began as a small settlement of a Celtic tribe known as the Parisii on an island in the Seine river. The Romans conquered the area in 52 BC and built a town called Lutetia, with a forum, baths, temples and an amphitheatre. During the Middle Ages, Paris became one of the largest and most important cities in Europe, a centre of learning with its famous university, and home to the royal court of the kings of France. The construction of the cathedral of Notre-Dame began in 1163 and continued for almost two centuries. In the following centuries the city grew rapidly, and in 1789 it became the stage of the French Revolution, which changed the course of European history. In the nineteenth century, under Napoleon III, the prefect Baron Haussmann rebuilt much of the city, creating wide boulevards, parks and";

    private const float ConfidentMargin = 1.5f;

    [Fact]
    public void HunyuanA13B_ShortPrompt_TeacherForcedMatchesLlamaServer()
    {
        AssertTeacherForced("The capital of France is", expectedPromptLen: 5,
        [
            12366, 627, 791, 6864, 315, 9822, 374, 12366, 627, 14196, 19884, 40, 1205, 311, 3350, 264, 13325, 5429, 430, 5097, 264, 1160,
        ],
        [
            0.819f, 0.379f, 2.321f, 8.088f, 10.010f, 2.051f, 7.996f, 3.453f, 2.725f, 0.277f, 2.095f, 0.116f, 0.163f, 3.077f, 1.147f, 4.465f, 0.359f, 0.024f, 3.601f, 0.839f, 2.604f, 0.431f,
        ]);
    }

    [Fact]
    public void HunyuanA13B_LongPrompt_BatchedPrefill_TeacherForcedMatchesLlamaServer()
    {
        AssertTeacherForced(LongPrompt, expectedPromptLen: 196,
        [
            220, 16, 17, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        ],
        [
            0.855f, 0.160f, 1.170f, 1.138f, 2.359f, 1.592f, 3.895f, 2.548f, 3.032f, 3.617f, 1.359f, 4.713f, 3.306f, 2.746f, 5.894f, 4.739f, 4.895f, 4.031f, 3.075f, 2.769f, 3.246f, 3.819f, 3.887f, 4.643f, 5.140f, 5.095f, 4.962f, 5.940f, 6.169f, 6.676f, 6.951f, 6.295f,
        ]);
    }

    private static void AssertTeacherForced(string prompt, int expectedPromptLen, int[] reference, float[] margins)
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this parity receipt.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        Assert.Equal("hunyuan-moe", Convert.ToString(model.Metadata["general.architecture"]));
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
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

    private static string? FindModel() => OpenTail.Stingray.Engine.Verification.ModelLocator.FindOrReport(ModelFile);
}
