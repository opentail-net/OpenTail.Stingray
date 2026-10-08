namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Qwen2-MoE (<c>qwen2moe</c>, Qwen1.5-MoE-A2.7B-Chat Q4_K_M: 60 routed experts top-4, un-normalised top-k, a 5632-wide shared expert with a
/// sigmoid gate, Q8_0 down experts) teacher-forced parity against llama-server (vendored tools/llama.cpp, same GGUF, raw completion, temperature 0,
/// top_k 1, -fa off, CPU; captured 2026-10-04). Stage A of the handoff closure plan: the CPU path must be right before its K/V is handed over.
/// llama-server's continuation is fed back token by token; our argmax must equal its token wherever its top-1 over top-2 margin is at least
/// <see cref="ConfidentMargin"/> nats (free-running greedy diverged at a 0.05-nat near-tie; 1.5 nats is the margin the LFM2 and Nemotron-H receipts use).
/// Two bugs were found by this checkpoint: the routed-expert width fell back to the dense width (this GGUF has no expert_feed_forward_length)
/// and the shared expert's sigmoid gate (ffn_gate_inp_shexp) was never applied.
/// </summary>
public sealed class Qwen2MoeGreedyParityTests : HeavyTestBase
{
    private const string ModelFile = "Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf";

    private const string LongPrompt =
        "The history of the city of Paris stretches back more than two thousand years. It began as a small settlement of a Celtic tribe known as the Parisii on an island in the Seine river. The Romans conquered the area in 52 BC and built a town called Lutetia, with a forum, baths, temples and an amphitheatre. During the Middle Ages, Paris became one of the largest and most important cities in Europe, a centre of learning with its famous university, and home to the royal court of the kings of France. The construction of the cathedral of Notre-Dame began in 1163 and continued for almost two centuries. In the following centuries the city grew rapidly, and in 1789 it became the stage of the French Revolution, which changed the course of European history. In the nineteenth century, under Napoleon III, the prefect Baron Haussmann rebuilt much of the city, creating wide boulevards, parks and";

    private const float ConfidentMargin = 1.5f;

    [Fact]
    public void Qwen2Moe_ShortPrompt_TeacherForcedMatchesLlamaServer()
    {
        AssertTeacherForced("The capital of France is", expectedPromptLen: 5,
        [
            2130, 563, 624, 32, 13, 24081, 198, 33, 13, 19846, 198, 34, 13, 21718, 198, 35, 13, 12095, 198, 102349, 510, 35,
        ],
        [
            0.221f, 0.315f, 2.806f, 2.198f, 8.229f, 0.074f, 6.116f, 7.594f, 14.730f, 0.448f, 9.660f, 10.838f, 13.212f, 1.736f, 10.234f, 8.629f, 14.353f, 6.555f, 8.611f, 7.384f, 0.110f, 4.850f,
        ]);
    }

    [Fact]
    public void Qwen2Moe_LongPrompt_BatchedPrefill_TeacherForcedMatchesLlamaServer()
    {
        AssertTeacherForced(LongPrompt, expectedPromptLen: 196,
        [
            501, 31473, 13, 576, 220, 16, 24, 15, 15, 4337, 594, 14588, 7117, 3786, 279, 1429, 10847, 5440, 315, 279, 882, 11, 323, 12095, 6116, 264, 12261, 315, 6481, 487, 11, 448,
        ],
        [
            1.082f, 0.503f, 0.431f, 0.672f, 0.669f, 1.457f, 0.036f, 0.405f, 9.786f, 1.791f, 0.367f, 1.919f, 0.394f, 0.399f, 1.084f, 0.309f, 1.042f, 0.677f, 1.335f, 4.565f, 0.487f, 1.065f, 1.767f, 0.733f, 1.266f, 0.296f, 0.047f, 0.918f, 0.579f, 0.202f, 0.067f, 1.408f,
        ]);
    }

    private static void AssertTeacherForced(string prompt, int expectedPromptLen, int[] reference, float[] margins)
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this parity receipt.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        Assert.Equal("qwen2moe", Convert.ToString(model.Metadata["general.architecture"]));
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        Assert.Equal(1408, hp.ExpertIntermediateDim);
        Assert.Equal(5632, hp.SharedExpertIntermediateDim);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        var promptTokens = tokenizer.Encode(prompt);
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
