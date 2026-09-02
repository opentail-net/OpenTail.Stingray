using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.AceStep.Conditioning;
using OpenTail.Stingray.Diffusion.AceStep.Text;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion.AceStep;

/// <summary>
/// First real-weight smoke test for ACE-Step's condition encoder (<see cref="AceStepConditionEncoder"/>):
/// real `turbo.safetensors` condition-encoder tensors (text projector + 8-layer lyric encoder) plus
/// the real Qwen3-Embedding-0.6B GGUF (both for text hidden states AND the raw token-embedding table
/// used for lyric lookup -- see docs/064-acestep-implementation-plan.md's "Corrections and
/// confirmations"). Non-degeneracy receipt (finite, non-trivial, shape-correct, sensitive to its
/// lyric input), not yet a numeric golden-parity test against a real `diffusers` reference run.
/// </summary>
public sealed class AceStepConditionEncoderTests
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
    public void Forward_RealWeights_ProducesNonDegenerateCondition()
    {
        string? turboPath = FindRepoFile("models/acestep-v15/turbo.safetensors");
        string? ggufPath = FindRepoFile("models/qwen3-embedding-0.6b/qwen3-embedding-0.6b-q8_0.gguf");
        Assert.SkipUnless(turboPath != null, "models/acestep-v15/turbo.safetensors not found");
        Assert.SkipUnless(ggufPath != null, "models/qwen3-embedding-0.6b/qwen3-embedding-0.6b-q8_0.gguf not found");

        using var loader = SafetensorsLoader.Open(turboPath!);
        var weights = AceStepConditionEncoderWeights.Load(loader);

        using var textEncoder = new AceStepQwen3TextEncoder(ggufPath!);

        string prompt =
            "# Instruction\nFill the audio semantic mask based on the given conditions:\n\n" +
            "# Caption\nA cinematic orchestral soundtrack with deep drums\n\n" +
            "# Metas\n- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: 30 seconds\n<|endoftext|>\n";
        var textHidden = textEncoder.Encode(prompt);

        int[] lyricTokenIds = textEncoder.Tokenize("The rain falls softly on the empty street");
        var embedTable = textEncoder.TokenEmbeddingTable;

        var condition = AceStepConditionEncoder.Forward(weights, textHidden, lyricTokenIds, embedTable);

        Assert.Equal(lyricTokenIds.Length + textHidden.Length, condition.Length);
        foreach (var row in condition)
        {
            Assert.Equal(2048, row.Length); // real ACE-Step DiT hidden_size
            foreach (var v in row)
                Assert.True(float.IsFinite(v), "condition contains NaN/Inf -- degenerate output");
        }

        double sumSq = 0;
        int count = 0;
        foreach (var row in condition)
            foreach (var v in row) { sumSq += (double)v * v; count++; }
        double rms = Math.Sqrt(sumSq / count);
        Assert.True(rms > 1e-3, $"condition RMS ({rms}) is near-zero -- likely a wiring bug");

        // Real sensitivity check: different lyrics should produce a different packed condition
        // (specifically, the lyric-derived rows -- a wiring bug that ignored lyricTokenIds would
        // pass the shape/finiteness checks above but be silently constant here).
        int[] otherLyricTokenIds = textEncoder.Tokenize("Thunder crashes over the mountain peaks");
        var otherCondition = AceStepConditionEncoder.Forward(weights, textHidden, otherLyricTokenIds, embedTable);

        double diff = 0;
        int n = Math.Min(lyricTokenIds.Length, otherLyricTokenIds.Length);
        for (int t = 0; t < n; t++)
            for (int d = 0; d < 2048; d++)
                diff += Math.Abs(condition[t][d] - otherCondition[t][d]);
        Assert.True(diff > 1e-2, "different lyrics produced (near-)identical lyric-encoded condition rows -- likely a wiring bug");
    }
}
