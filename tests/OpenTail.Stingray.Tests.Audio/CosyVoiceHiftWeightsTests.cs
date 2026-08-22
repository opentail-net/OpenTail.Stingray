using System.IO;
using OpenTail.Stingray.Audio.CosyVoice;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real-weights coverage for CosyVoice2's HiFT vocoder weight loader, in particular the
/// PyTorch weight_norm fold (`parametrizations.weight.original0/1` -> plain conv weight) --
/// see docs/audio-review-progress.md's CosyVoice section. The vocoder's forward DSP pass
/// itself is not ported yet (deliberately not rushed -- see that doc entry), so this only
/// covers the loader.
/// </summary>
public sealed class CosyVoiceHiftWeightsTests : HeavyTestBase
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
    public void Weights_LoadAndFoldWeightNorm_CorrectShapesFiniteNonDegenerate()
    {
        string? path = FindRepoFile("models/cosyvoice2_hift.safetensors");
        Assert.SkipUnless(path != null, "models/cosyvoice2_hift.safetensors not found");

        using var w = new CosyVoiceHiftWeights(path!);

        // conv_pre: [512, 80, 7] folded weight.
        Assert.Equal(512 * 80 * 7, w.ConvPreWeight.Length);
        Assert.Equal(512, w.ConvPreBias.Length);
        // conv_post: [18, 64, 7] (18 = n_fft+2 = 16+2).
        Assert.Equal(18 * 64 * 7, w.ConvPostWeight.Length);

        Assert.Equal(3, w.UpWeight.Length);
        Assert.Equal(512 * 256 * 16, w.UpWeight[0].Length);
        Assert.Equal(256 * 128 * 11, w.UpWeight[1].Length);
        Assert.Equal(128 * 64 * 7, w.UpWeight[2].Length);

        Assert.Equal(9, w.ResBlocks.Length); // 3 stages x 3 kernels
        Assert.Equal(3, w.SourceResBlocks.Length);

        foreach (var v in w.ConvPreWeight) Assert.False(float.IsNaN(v) || float.IsInfinity(v));
        foreach (var v in w.UpWeight[0]) Assert.False(float.IsNaN(v) || float.IsInfinity(v));

        // Sanity: a weight_norm fold should not collapse everything to zero (would indicate a
        // sign/shape bug in the per-output-channel norm computation).
        float maxAbs = 0f;
        foreach (var v in w.ConvPreWeight) maxAbs = MathF.Max(maxAbs, MathF.Abs(v));
        Assert.True(maxAbs > 1e-4f, "folded conv_pre weight looks degenerate (all near zero)");
    }
}
