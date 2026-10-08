namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Llama 4 Scout past its 8192-token attention chunk: a 9430-token prompt (the first 42000 characters of scripts/kvarn-gate/wiki.test.raw, cut at the
/// last newline) is prefilled in one batch, then 24 greedy tokens are teacher-forced against llama-server (same GGUF, CPU, BOS added; captured
/// 2026-10-04). Chunked attention (RoPE layers read only their own 8192-token chunk) and attention temperature tuning (NoPE layers) both change
/// the result beyond position 8192; below it they are the identity, which the short-prompt tests cannot see.
/// </summary>
public sealed class LlamaFourLongContextParityTests : HeavyTestBase
{
    private const string ModelFile = "Llama-4-Scout-17B-16E-Instruct-Q3_K_M-00001-of-00002.gguf";
    private const float ConfidentMargin = 1.5f;

    [Fact]
    public void Scout_PastTheChunkBoundary_TeacherForcedMatchesLlamaServer()
    {
        string? wiki = FindRepoFile(Path.Combine("scripts", "kvarn-gate", "wiki.test.raw"));
        Assert.SkipWhen(wiki is null, "scripts/kvarn-gate/wiki.test.raw is required for this parity receipt.");
        string text = File.ReadAllText(wiki!);
        text = text[..Math.Min(42000, text.Length)];
        text = text[..text.LastIndexOf('\n')];

        AssertTeacherForced(text, expectedPromptLen: 9430,
        [
            8352, 24, 10388, 1102, 423, 4102, 939, 15402, 428, 2290, 6790, 651, 18027, 651, 33052, 1102, 24723, 270, 590, 1897, 3420, 2290, 10388, 1102,
        ],
        [
            0.508f, 1.319f, 0.206f, 3.208f, 1.850f, 11.428f, 7.974f, 10.875f, 14.087f, 5.213f, 1.640f, 1.444f, 2.486f, 0.612f, 1.471f, 0.797f, 2.042f, 9.458f, 9.044f, 11.338f, 12.840f, 4.725f, 0.054f, 1.683f,
        ]);
    }

    private static string? FindRepoFile(string relative)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, relative);
            if (File.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }
        return null;
    }

    private static void AssertTeacherForced(string prompt, int expectedPromptLen, int[] reference, float[] margins)
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this parity receipt.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        Assert.Equal("llama4", Convert.ToString(model.Metadata["general.architecture"]));
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        Assert.Equal(16, hp.NumExperts);
        Assert.Equal(1, hp.NumActiveExperts);
        Assert.True(hp.UseSigmoidGating, "Llama 4 weights its single routed expert with a sigmoid");
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        var encoded = tokenizer.Encode(prompt);
        var promptTokens = tokenizer.AddBosToken ? new List<int> { tokenizer.BosTokenId }.Concat(encoded).ToList() : encoded.ToList(); // llama-server adds BOS
        Assert.Equal(expectedPromptLen, promptTokens.Count);

        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 9728);

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
