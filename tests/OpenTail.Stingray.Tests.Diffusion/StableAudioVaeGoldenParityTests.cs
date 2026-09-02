using OpenTail.Stingray.Diffusion.StableAudio;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Numeric parity check of <see cref="AcousticVae"/>'s decoder path against a golden run of the
/// real `stable_audio_3.models.autoencoders.SAMEDecoder`, loaded with the real
/// `stabilityai/stable-audio-3-small-music-base` checkpoint's bundled VAE weights (ungated,
/// downloaded locally to `models/stable-audio-3-small-music-base/` -- see
/// docs/057-stable-audio-3-implementation-plan.md). The fixture exercises the real dual-window
/// differential-attention resampling mechanism (chunked + shift-window passes, `DynamicTanh` norm,
/// weight-normalized 3-tap mapping conv) at real checkpoint shapes on a small (4-frame) latent
/// sequence. Real eval-time noise sources (`mask_noise`, bottleneck noise-regularization) were
/// disabled when generating the fixture to keep the comparison deterministic -- see
/// <see cref="AcousticVae"/>'s class doc for why this is a safe, tiny-magnitude simplification.
/// </summary>
public sealed class StableAudioVaeGoldenParityTests
{
    private const string DitDirRelative = "models/stable-audio-3-small-music-base";
    private const int LatentSeqLen = 4;

    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath.Replace('/', Path.DirectorySeparatorChar));
            if (File.Exists(p) || Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static string? FindGoldenDir()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "tests", "OpenTail.Stingray.Tests.Diffusion", "TestData", "StableAudioVaeGolden");
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static float[] ReadFloats(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var arr = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, arr, 0, bytes.Length);
        return arr;
    }

    private static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        return (float)(dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12));
    }

    [Fact]
    public void AcousticVae_Decode_MatchesRealSAMEDecoderReference()
    {
        string? ditDir = FindRepoFile(DitDirRelative);
        string? goldenDir = FindGoldenDir();
        if (ditDir is null || goldenDir is null) return; // skip: needs local VAE weights + fixtures

        using var st = SafetensorsLoader.OpenDirectory(ditDir);
        using var vae = AcousticVae.FromLoader(st);

        var latents = ReadFloats(Path.Combine(goldenDir, "latents.bin"));
        var goldenPcm = ReadFloats(Path.Combine(goldenDir, "pcm.bin"));

        var pcm = vae.Decode(latents, LatentSeqLen);

        Assert.Equal(goldenPcm.Length, pcm.Length);
        float cos = CosineSimilarity(pcm, goldenPcm);
        Assert.True(cos > 0.99f, $"AcousticVae decode cosine-sim too low: {cos}");
    }

    /// <summary>
    /// Numeric parity check of <see cref="AcousticVae"/>'s encoder path (patchify → real
    /// `SAMEEncoder` → `SoftNormBottleneck.encode`) against a golden run of the real
    /// `stable_audio_3.models.autoencoders.SAMEEncoder` on the same fixed-seed synthetic raw audio,
    /// same real checkpoint weights, same eval-time-noise-disabled convention as the decode test
    /// above.
    /// </summary>
    [Fact]
    public void AcousticVae_Encode_MatchesRealSAMEEncoderReference()
    {
        string? ditDir = FindRepoFile(DitDirRelative);
        string? goldenDir = FindGoldenDir();
        if (ditDir is null || goldenDir is null) return; // skip: needs local VAE weights + fixtures

        using var st = SafetensorsLoader.OpenDirectory(ditDir);
        using var vae = AcousticVae.FromLoader(st);

        var pcm = ReadFloats(Path.Combine(goldenDir, "encode_pcm.bin"));
        var goldenLatents = ReadFloats(Path.Combine(goldenDir, "encode_latents.bin"));

        var latents = vae.Encode(pcm, pcm.Length / 2);

        Assert.Equal(goldenLatents.Length, latents.Length);
        float cos = CosineSimilarity(latents, goldenLatents);
        Assert.True(cos > 0.99f, $"AcousticVae encode cosine-sim too low: {cos}");
    }
}
