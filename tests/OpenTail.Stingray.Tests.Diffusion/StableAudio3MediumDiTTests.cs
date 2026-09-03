using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.StableAudio;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// First real-weight smoke test for Stable Audio 3 Medium's DiT (<see cref="StableAudioMediumDiT"/>),
/// the real differential-attention variant -- see docs/057-stable-audio-3-implementation-plan.md's
/// "Medium — real archaeology" section. Real weights (9.22GB `model.safetensors`), synthetic
/// (real-shaped) condition tokens -- no real T5Gemma/VAE wiring yet (Sprint 4's own exit criterion
/// per docs/065: "latent generation only, no audio yet is fine at this stage"). Non-degeneracy
/// receipt (finite, non-degenerate, timestep-sensitive output), not yet a numeric golden-parity
/// test against a real Python reference run.
/// </summary>
public sealed class StableAudio3MediumDiTTests
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
    public void Forward_RealWeights_ProducesFiniteTimestepSensitiveOutput()
    {
        string? ditDir = FindRepoDir("models/stable-audio-3-medium-base");
        Assert.SkipUnless(ditDir != null, "models/stable-audio-3-medium-base not found");

        using var weights = SafetensorsLoader.OpenDirectory(ditDir!);
        using var dit = StableAudioMediumDiT.FromLoader(weights);

        const int ioChannels = 256;
        const int condTokenDim = 768;
        int seqLen = 8;
        int nCond = 4;

        var rng = new Random(0);
        var latent = new float[seqLen * ioChannels];
        for (int i = 0; i < latent.Length; i++) latent[i] = (float)(rng.NextDouble() * 0.2 - 0.1);

        var condTokens = new float[nCond * condTokenDim];
        for (int i = 0; i < condTokens.Length; i++) condTokens[i] = (float)(rng.NextDouble() * 0.2 - 0.1);

        var secondsTotalRaw = new float[condTokenDim];
        for (int i = 0; i < secondsTotalRaw.Length; i++) secondsTotalRaw[i] = (float)(rng.NextDouble() * 0.2 - 0.1);

        var out1 = dit.Forward(latent, seqLen, condTokens, nCond, secondsTotalRaw, timestep: 0.3f);
        Assert.Equal(seqLen * ioChannels, out1.Length);
        foreach (var v in out1) Assert.True(float.IsFinite(v), "output contains NaN/Inf");

        double sumSq = 0;
        foreach (var v in out1) sumSq += (double)v * v;
        double rms = Math.Sqrt(sumSq / out1.Length);
        Assert.True(rms > 1e-4, $"output RMS ({rms}) is near-zero -- likely a wiring bug");

        // Real sensitivity check: a different timestep should produce measurably different output
        // (confirms the AdaLN timestep-conditioning path is genuinely wired, not silently dropped --
        // the same style of check used for every other DiT in this project).
        var out2 = dit.Forward(latent, seqLen, condTokens, nCond, secondsTotalRaw, timestep: 0.8f);
        double diff = 0;
        for (int i = 0; i < out1.Length; i++) diff += Math.Abs(out1[i] - out2[i]);
        Assert.True(diff > 1e-2, "different timesteps produced (near-)identical output -- likely a wiring bug");
    }
}
