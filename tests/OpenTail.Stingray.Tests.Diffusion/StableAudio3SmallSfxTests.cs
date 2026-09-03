using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.StableAudio;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// First real-weight smoke test for Stable Audio 3 Small SFX. Per docs/057-stable-audio-3-
/// implementation-plan.md's "SA3_MODEL_MATRIX" section: real archaeology found Small SFX's DiT and
/// VAE (`taae_v2`) configs are BYTE-IDENTICAL to the already golden-verified Small Music runtime,
/// and its T5Gemma text encoder is the literal same checkpoint (identical sha256) -- so this test
/// exercises the EXISTING <see cref="StableAudioPipeline"/>/<see cref="StableAudioDiT"/>/
/// <see cref="AcousticVae"/> code, completely unmodified, just pointed at the real SFX
/// `model.safetensors` (a genuinely different fine-tune, same tensor shapes: 685 tensors, matching
/// Small Music's own real tensor count). Non-degeneracy receipt (finite, non-silent, real
/// SFX-appropriate prompt), not a numeric golden-parity test (no real Python SFX reference dump
/// produced this session) -- the DiT/VAE math itself is already golden-verified via the Small
/// Music fixtures, so this test's job is only to confirm the SFX checkpoint's own real weights load
/// and produce sane output, not to re-verify math already covered.
/// </summary>
public sealed class StableAudio3SmallSfxTests
{
    private const string DitDirRelative = "models/stable-audio-3-small-sfx-base";
    private const string T5GemmaDirRelative = "models/stable-audio-3-t5gemma";

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
    public void Generate_RealSfxWeights_ProducesFiniteNonSilentAudio()
    {
        string? ditDir = FindRepoDir(DitDirRelative);
        string? t5gemmaDir = FindRepoDir(T5GemmaDirRelative);
        Assert.SkipUnless(ditDir != null, "models/stable-audio-3-small-sfx-base not found");
        Assert.SkipUnless(t5gemmaDir != null, "models/stable-audio-3-t5gemma not found");

        using var ditWeights = SafetensorsLoader.OpenDirectory(ditDir!);
        using var textEncoderWeights = SafetensorsLoader.OpenDirectory(t5gemmaDir!);
        using var pipeline = new StableAudioPipeline(ditWeights, textEncoderWeights, t5gemmaDir);

        var pcm = pipeline.Generate(new StableAudioRequest
        {
            Prompt = "a glass bottle shattering on a hard floor",
            DurationSeconds = 3f,
            Steps = 8, // short on purpose for a smoke test
            CfgScale = 6.0f,
            Seed = 1234,
            OutputPath = "",
        });

        Assert.True(pcm.Length > 0, "generated zero samples");
        foreach (var v in pcm) Assert.True(float.IsFinite(v), "PCM contains NaN/Inf");

        double sumSq = 0;
        foreach (var v in pcm) sumSq += (double)v * v;
        double rms = Math.Sqrt(sumSq / pcm.Length);
        Assert.True(rms > 1e-6, $"generated audio RMS ({rms}) is near-silent -- likely a wiring bug");
    }
}
