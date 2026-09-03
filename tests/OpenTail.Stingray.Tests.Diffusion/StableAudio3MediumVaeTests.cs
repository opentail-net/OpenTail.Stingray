using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.StableAudio;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// First real-weight smoke test for Stable Audio 3 Medium's SAME-L VAE decoder
/// (<see cref="SameLargeVae"/>). Real weights, synthetic latent -- see docs/057-stable-audio-3-
/// implementation-plan.md's Sprint 5 section for the real windowed-attention derivation. Non-
/// degeneracy receipt (finite, non-silent, correctly-shaped audio), not yet a numeric golden-parity
/// test.
/// </summary>
public sealed class StableAudio3MediumVaeTests
{
    private static string? FindRepoDir(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath.Replace('/', Path.DirectorySeparatorChar));
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Decode_RealWeights_ProducesFiniteNonSilentAudio()
    {
        string? dir = FindRepoDir("models/stable-audio-3-medium-base");
        Assert.SkipUnless(dir != null, "models/stable-audio-3-medium-base not found");

        using var weights = SafetensorsLoader.OpenDirectory(dir!);
        using var vae = SameLargeVae.FromLoader(weights);

        const int latentDim = 256;
        int latentSeqLen = 20; // short on purpose -- real 16x downsampling, keep wall-clock low

        var rng = new Random(0);
        var latent = new float[latentSeqLen * latentDim];
        for (int i = 0; i < latent.Length; i++) latent[i] = (float)(rng.NextDouble() * 0.4 - 0.2);

        var pcm = vae.Decode(latent, latentSeqLen);

        // n * Stride(16) upsampled "frames", each unpatchified into PatchSize(256) raw samples,
        // interleaved across AudioChannels(2).
        Assert.Equal(latentSeqLen * 16 * 256 * 2, pcm.Length);
        foreach (var v in pcm) Assert.True(float.IsFinite(v), "PCM contains NaN/Inf");

        double sumSq = 0;
        foreach (var v in pcm) sumSq += (double)v * v;
        double rms = Math.Sqrt(sumSq / pcm.Length);
        Assert.True(rms > 1e-6, $"PCM RMS ({rms}) is near-silent -- likely a wiring bug");
    }
}
