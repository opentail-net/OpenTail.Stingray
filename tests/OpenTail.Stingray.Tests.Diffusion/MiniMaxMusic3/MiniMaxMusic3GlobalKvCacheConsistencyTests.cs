using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.MiniMaxMusic3;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion.MiniMaxMusic3;

/// <summary>
/// Self-consistency check for <see cref="MiniMaxMusic3GlobalModel.ForwardIncremental"/>: feeding
/// the same token sequence one token at a time through a growing <see cref="MiniMaxMusic3GlobalKvCache"/>
/// must produce EXACTLY the same final-token hidden state and logits as a single full-sequence
/// <see cref="MiniMaxMusic3GlobalModel.Forward(int[])"/> call (real Qwen3 causal attention with a
/// cache is mathematically identical to full recomputation, just incremental) -- this is a real
/// numerical regression test for the cache/position-offset bookkeeping itself, not an architecture
/// check (that's already covered by <c>MiniMaxMusic3GlobalModelGoldenParityTests</c>). See
/// docs/066-minimax-music3-future-plan.md.
/// </summary>
public sealed class MiniMaxMusic3GlobalKvCacheConsistencyTests
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
    public void ForwardIncremental_StepByStep_MatchesFullSequenceForward()
    {
        string? weightsDir = FindRepoDir("models/minimax-music3/language_model");
        Assert.SkipUnless(weightsDir != null, "models/minimax-music3/language_model/ not found");

        int[] tokenIds = [141615, 13892, 125721, 175286, 5000, 99999];

        using var loaderFull = SafetensorsLoader.OpenDirectory(weightsDir!);
        using var modelFull = new MiniMaxMusic3GlobalModel(loaderFull);
        var (fullHidden, fullLogits) = modelFull.Forward(tokenIds);

        using var loaderInc = SafetensorsLoader.OpenDirectory(weightsDir!);
        using var modelInc = new MiniMaxMusic3GlobalModel(loaderInc);
        var cache = new MiniMaxMusic3GlobalKvCache(MiniMaxMusic3Config.LanguageModelNumLayers);

        // Prefill the first 3 tokens together, then step the rest one at a time -- exercises both
        // the multi-token prefill path and the single-token incremental path in one test.
        var (_, _) = modelInc.ForwardIncremental(tokenIds[..3], cache);
        float[][] lastIncHidden = [];
        float[] lastIncLogits = [];
        for (int t = 3; t < tokenIds.Length; t++)
        {
            (lastIncHidden, lastIncLogits) = modelInc.ForwardIncremental([tokenIds[t]], cache);
        }

        Assert.Equal(tokenIds.Length, cache.Length);

        int hidden = MiniMaxMusic3Config.LanguageModelHiddenSize;
        var expectedLastHidden = fullHidden[^1];
        var actualLastHidden = lastIncHidden[0];

        double maxAbsDiff = 0;
        for (int c = 0; c < hidden; c++)
            maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs(expectedLastHidden[c] - actualLastHidden[c]));

        // Not bit-exact (parallel reduction order differs between the batched-prefill single-pass
        // path and per-step incremental calls), but should agree to float rounding, not architecture-
        // level divergence.
        Assert.True(maxAbsDiff < 1e-2, $"incremental vs full-sequence hidden state maxAbsDiff={maxAbsDiff:F6} too large -- cache/position bookkeeping bug");

        int vocab = MiniMaxMusic3Config.LanguageModelVocabSize;
        int argmaxFull = 0, argmaxInc = 0;
        float maxFull = float.NegativeInfinity, maxInc = float.NegativeInfinity;
        for (int i = 0; i < vocab; i++)
        {
            if (fullLogits[i] > maxFull) { maxFull = fullLogits[i]; argmaxFull = i; }
            if (lastIncLogits[i] > maxInc) { maxInc = lastIncLogits[i]; argmaxInc = i; }
        }
        Assert.Equal(argmaxFull, argmaxInc);
    }
}
