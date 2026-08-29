
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real structural test for <see cref="QwenTtsCodePredictorGeneration.GenerateAcousticCodes"/>:
/// a real Talker forward step produces a real semantic code and real
/// <see cref="ForwardPass.LastHidden"/>, which then drives the real Code Predictor's 15-step
/// acoustic depth-expansion. Not yet golden-verified against a numeric oracle -- checks the
/// full real chain runs to completion on real weights and produces a real, in-range,
/// non-degenerate 15-code acoustic sequence, matching this session's established first-pass bar
/// for a from-scratch generation loop.
/// </summary>
public sealed class QwenTtsCodePredictorGenerationTests : HeavyTestBase
{
    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void GenerateAcousticCodes_RealWeights_ProducesInRangeNonDegenerateCodeSequence()
    {
        string? modelPath = FindRepoFile("models/qwen-talker-0.6b-base-Q8_0.gguf");
        Assert.SkipUnless(modelPath != null, "models/qwen-talker-0.6b-base-Q8_0.gguf not found");

        using var rawModel = GgufModel.Open(modelPath!);

        // Real Talker step producing c0 + its real LastHidden -- mirrors the tail of
        // QwenTtsTalkerGeneration's own decode loop (a real Forward step, not Prefill alone,
        // per this session's confirmed LastHidden constraint).
        using var talkerSource = new QwenTtsTalkerTensorSource(rawModel, numLayers: 28);
        var talkerHp = ModelHyperparams.FromGgufMetadata(talkerSource.Metadata);
        using var talkerBackend = new CpuBackend();
        using var talkerFwd = new ForwardPass(talkerSource, talkerBackend, talkerHp);

        var prompt = new int[] { 0, 1, 2 };
        _ = talkerFwd.Prefill(prompt);
        var c0Logits = talkerFwd.Forward(2, prompt.Length);
        int c0 = ArgMax(c0Logits);
        var talkerLastHidden = talkerFwd.LastHidden.ToArray();

        Assert.InRange(c0, 0, 3071); // real codec vocab size

        var codePredWeights = QwenTtsCodePredictorGeneration.Weights.Load(rawModel);
        var codes = QwenTtsCodePredictorGeneration.GenerateAcousticCodes(rawModel, codePredWeights, numLayers: 5, c0, talkerLastHidden, new System.Random(42));

        Assert.Equal(QwenTtsCodePredictorGeneration.NumAcousticCodebooks, codes.Length);
        foreach (var c in codes)
            Assert.InRange(c, 0, QwenTtsCodePredictorGeneration.AcousticVocabSize - 1);

        Assert.True(new System.Collections.Generic.HashSet<int>(codes).Count > 1, "generated acoustic codes look degenerate (all-identical)");
    }

    private static int ArgMax(System.ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestVal = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }
}
