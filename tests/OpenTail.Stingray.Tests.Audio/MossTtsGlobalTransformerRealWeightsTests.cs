using OpenTail.Stingray.Audio.MossTts;
using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real-weight smoke test for MOSS-TTS-Nano's global transformer: loads the real
/// `moss-tts-nano-100m-q8_0.gguf` checkpoint and runs a real forward pass over a small synthetic
/// text-only prompt, confirming finite output and (as a real, non-trivial invariant a broken
/// wiring would very likely violate) that argmax next-text-token predictions differ across a few
/// different short prompts rather than collapsing to one constant token -- the same collapse
/// signature `docs/audio-review-progress.md` documents as the real Fun-ASR-Nano LLM-wiring bug.
/// Not a numeric golden-parity check (no captured reference trace for this checkpoint yet) --
/// see the progress doc for that gap.
/// </summary>
public sealed class MossTtsGlobalTransformerRealWeightsTests : HeavyTestBase
{
    private static string? FindRepoFile(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static List<MossTtsGlobalRow> TextOnlyPrompt(IEnumerable<int> textIds)
    {
        var pad = MossTtsGlobalTransformerWeights.AudioPadTokenId;
        return textIds
            .Select(id => new MossTtsGlobalRow(id, Enumerable.Repeat(pad, MossTtsGlobalTransformerWeights.NumCodebooks).ToArray()))
            .ToList();
    }

    [Fact]
    public void ForwardLastHidden_OnRealCheckpoint_ProducesFiniteVaryingLogits()
    {
        string? path = FindRepoFile("models/_models/moss-tts-nano/MOSS-TTS-Nano-100M-GGUF/moss-tts-nano-100m-q8_0.gguf");
        Assert.SkipUnless(path != null, "moss-tts-nano checkpoint not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        var w = new MossTtsGlobalTransformerWeights(source);

        int imStart = MossTtsGlobalTransformerWeights.ImStartTokenId;

        var promptA = TextOnlyPrompt([imStart, 100, 200, 300]);
        var promptB = TextOnlyPrompt([imStart, 500, 600, 700, 800]);

        var hiddenA = MossTtsGlobalTransformer.ForwardLastHidden(w, promptA);
        var hiddenB = MossTtsGlobalTransformer.ForwardLastHidden(w, promptB);

        Assert.All(hiddenA, v => Assert.True(float.IsFinite(v)));
        Assert.All(hiddenB, v => Assert.True(float.IsFinite(v)));

        var logitsA = MossTtsGlobalTransformer.TextLogits(w, hiddenA);
        var logitsB = MossTtsGlobalTransformer.TextLogits(w, hiddenB);
        Assert.All(logitsA, v => Assert.True(float.IsFinite(v)));

        int ArgMax(float[] x)
        {
            int best = 0;
            for (int i = 1; i < x.Length; i++) if (x[i] > x[best]) best = i;
            return best;
        }

        int argmaxA = ArgMax(logitsA);
        int argmaxB = ArgMax(logitsB);

        // Real invariant, not a guess: two meaningfully different prompts should not collapse to
        // the exact same predicted next token on a real, correctly-wired 100M LM -- this is the
        // same collapse signature that isolated Fun-ASR-Nano's real LLM-wiring bug this session.
        Assert.NotEqual(argmaxA, argmaxB);

        // Causal invariant: a longer prompt sharing a prefix must reproduce the prefix's earlier
        // hidden states exactly (no lookahead leakage).
        var prefixOnly = TextOnlyPrompt([imStart, 100, 200]);
        var prefixHidden = MossTtsGlobalTransformer.ForwardAll(w, prefixOnly);
        var fullHidden = MossTtsGlobalTransformer.ForwardAll(w, promptA);
        for (int i = 0; i < prefixOnly.Count; i++)
        {
            for (int d = 0; d < MossTtsGlobalTransformerWeights.HiddenDim; d++)
                Assert.Equal(prefixHidden[i][d], fullHidden[i][d], precision: 3);
        }
    }
}
