
namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Shared teacher-forced parity check against a recorded llama-server continuation: the reference tokens are fed
/// back in and our argmax must equal llama.cpp's token wherever llama.cpp's top-1 led its top-2 by more than
/// <paramref name="minMargin"/> nats. Unlike a free-running greedy comparison it does not fork at the first close
/// pair, so it stays meaningful on low-bit quantisations and for models whose top-2 are often close.
/// </summary>
internal static class TeacherForcedParity
{
    public static void Assert(string modelFile, string expectedArch, IReadOnlyList<int> promptTokens,
        int[] reference, double[] llamaMargins, double minMargin = 1.5, int minChecked = 8)
    {
        var path = FindModel(modelFile);
        Xunit.Assert.SkipWhen(path is null, $"{modelFile} is required for this parity receipt.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        Xunit.Assert.Equal(expectedArch, Convert.ToString(model.Metadata["general.architecture"]));
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        var input = new List<int>(promptTokens);
        input.AddRange(reference[..^1]);
        int firstPredicting = promptTokens.Count - 1;
        var argmax = new int[reference.Length];

        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 1024);
        fwd.PrefillWithPerPositionLogits(input, 0, (pos, logits) =>
        {
            int j = pos - firstPredicting;
            if (j >= 0 && j < reference.Length) argmax[j] = Sampler.Greedy(logits);
        });

        int checkedCount = 0;
        for (int j = 0; j < reference.Length; j++)
        {
            if (llamaMargins[j] <= minMargin) continue;
            checkedCount++;
            Xunit.Assert.True(argmax[j] == reference[j],
                $"position {j}: ours {argmax[j]} ({tokenizer.Decode([argmax[j]])}) vs llama.cpp {reference[j]} " +
                $"({tokenizer.Decode([reference[j]])}), llama margin {llamaMargins[j]}");
        }
        Xunit.Assert.True(checkedCount >= minChecked, $"only {checkedCount} confident positions");
    }

    /// <summary>Tokenises <paramref name="prompt"/> the way llama-server's raw completion does: BOS first when the
    /// model's metadata asks for it.</summary>
    public static IReadOnlyList<int> Tokenize(string modelFile, string prompt)
    {
        var path = FindModel(modelFile);
        if (path is null) return [];
        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path);
        var tokenizer = GgufTokenizer.FromGgufModel(modelHandle.Model);
        var ids = tokenizer.Encode(prompt).ToList();
        if (tokenizer.AddBosToken && tokenizer.BosTokenId >= 0 && (ids.Count == 0 || ids[0] != tokenizer.BosTokenId))
            ids.Insert(0, tokenizer.BosTokenId);
        return ids;
    }

    public static string? FindModel(string modelFile)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            foreach (var sub in new[] { "models", Path.Combine("models", "_models") })
            {
                var candidate = Path.Combine(dir, sub, modelFile);
                if (File.Exists(candidate)) return candidate;
            }
            if (Directory.GetParent(dir) is not { } parent) break;
            dir = parent.FullName;
        }
        return null;
    }
}
