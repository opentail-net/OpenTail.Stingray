using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.MiniMaxMusic3;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion.MiniMaxMusic3;

public sealed class MiniMaxMusic3GlobalModelStepPairParityTests
{
    private static string? FindRepoDir(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void ForwardIncrementalStepPair_MatchesIndependentForwardIncremental()
    {
        string? weightsDir = FindRepoDir("models/minimax-music3/language_model");
        Assert.SkipUnless(weightsDir != null, "models/minimax-music3/language_model/ not found");

        using var loader = SafetensorsLoader.OpenDirectory(weightsDir!);
        using var model = new MiniMaxMusic3GlobalModel(loader);

        int hidden = MiniMaxMusic3Config.LanguageModelHiddenSize;
        var condEmb = new float[hidden];
        var uncondEmb = new float[hidden];
        var rng = new Random(42);
        for (int i = 0; i < hidden; i++)
        {
            condEmb[i] = (float)rng.NextDouble() * 0.1f;
            uncondEmb[i] = (float)rng.NextDouble() * 0.1f;
        }

        // 1. Run independent
        var condCacheIndep = new MiniMaxMusic3GlobalKvCache(MiniMaxMusic3Config.LanguageModelNumLayers);
        var uncondCacheIndep = new MiniMaxMusic3GlobalKvCache(MiniMaxMusic3Config.LanguageModelNumLayers);

        var (condHiddenIndep, condLogitsIndep) = model.ForwardIncrementalWithEmbedding(condEmb, condCacheIndep);
        var (uncondHiddenIndep, uncondLogitsIndep) = model.ForwardIncrementalWithEmbedding(uncondEmb, uncondCacheIndep);

        // 2. Run paired
        var condCachePair = new MiniMaxMusic3GlobalKvCache(MiniMaxMusic3Config.LanguageModelNumLayers);
        var uncondCachePair = new MiniMaxMusic3GlobalKvCache(MiniMaxMusic3Config.LanguageModelNumLayers);

        var (condHiddenPair, uncondHiddenPair, condLogitsPair, uncondLogitsPair) = model.ForwardIncrementalStepPair(
            condEmb,
            uncondEmb,
            condCachePair,
            uncondCachePair);

        // Verify hidden states
        for (int i = 0; i < hidden; i++)
        {
            Assert.True(Math.Abs(condHiddenIndep[0][i] - condHiddenPair[i]) < 1e-4f,
                $"Cond hidden mismatch at {i}: indep={condHiddenIndep[0][i]}, pair={condHiddenPair[i]}");
            Assert.True(Math.Abs(uncondHiddenIndep[0][i] - uncondHiddenPair[i]) < 1e-4f,
                $"Uncond hidden mismatch at {i}: indep={uncondHiddenIndep[0][i]}, pair={uncondHiddenPair[i]}");
        }

        // Verify logits for audio candidate tokens
        int audioStart = MiniMaxMusic3Config.AudioEndTokenId;
        int audioCount = (MiniMaxMusic3Config.AudioCodeOffset + MiniMaxMusic3Config.SemanticVocabSize) - audioStart;
        for (int i = audioStart; i < audioStart + audioCount; i++)
        {
            Assert.True(Math.Abs(condLogitsIndep[i] - condLogitsPair[i]) < 1e-3f,
                $"Cond logits mismatch at {i}: indep={condLogitsIndep[i]}, pair={condLogitsPair[i]}");
            Assert.True(Math.Abs(uncondLogitsIndep[i] - uncondLogitsPair[i]) < 1e-3f,
                $"Uncond logits mismatch at {i}: indep={uncondLogitsIndep[i]}, pair={uncondLogitsPair[i]}");
        }
    }
}
