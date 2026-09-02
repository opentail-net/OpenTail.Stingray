using OpenTail.Stingray.Diffusion.StableAudio;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Numeric parity check of <see cref="StableAudioDiT"/> against a golden forward step dumped from
/// the real `stable_audio_3.models.dit.DiffusionTransformer` reference, loaded with the real
/// `stabilityai/stable-audio-3-small-music-base` checkpoint (ungated, downloaded locally to
/// `models/stable-audio-3-small-music-base/` -- see docs/057-stable-audio-3-implementation-plan.md).
/// The fixture reuses the real T5Gemma golden encode's output (with real learned padding-embedding
/// substitution applied) concatenated with a real `NumberConditioner` `seconds_total` embedding as
/// the cross-attention context, and a small fixed-seed synthetic latent -- exercising every real
/// mechanism (RoPE partial rotary, QK-RMSNorm, 6-way AdaLN, cross-attn V-zeroing, SwiGLU FFN,
/// memory tokens) at a size small enough to keep the reference-generation script fast.
/// </summary>
public sealed class StableAudioDiTGoldenParityTests
{
    private const string DitDirRelative = "models/stable-audio-3-small-music-base";
    private const int SeqLen = 8;
    private const int NCond = 25;
    private const int CondTokenDim = 768;

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
            var p = Path.Combine(dir, "tests", "OpenTail.Stingray.Tests.Diffusion", "TestData", "StableAudioDiTGolden");
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
    public void StableAudioDiT_MatchesRealReference_OnFixedSyntheticLatent()
    {
        string? ditDir = FindRepoFile(DitDirRelative);
        string? goldenDir = FindGoldenDir();
        if (ditDir is null || goldenDir is null) return; // skip: needs local DiT weights + fixtures

        using var st = SafetensorsLoader.OpenDirectory(ditDir);
        using var dit = StableAudioDiT.FromLoader(st);

        var latent = ReadFloats(Path.Combine(goldenDir, "latent.bin"));
        var condTokens = ReadFloats(Path.Combine(goldenDir, "cond_tokens.bin"));
        var secondsTotalRaw = ReadFloats(Path.Combine(goldenDir, "seconds_total_raw.bin"));
        var goldenVelocity = ReadFloats(Path.Combine(goldenDir, "velocity.bin"));

        Assert.Equal(NCond, condTokens.Length / CondTokenDim);

        // Note: the real reference this fixture was generated from never passed a
        // cross_attn_cond_mask to model.forward() either -- the real model UNCONDITIONALLY
        // discards it before it would ever reach attention anyway (see StableAudioDiT.Forward's
        // doc comment), so this comparison already reflects real production behavior exactly.
        var velocity = dit.Forward(latent, SeqLen, condTokens, NCond, secondsTotalRaw, timestep: 0.5f);

        Assert.Equal(goldenVelocity.Length, velocity.Length);
        float cos = CosineSimilarity(velocity, goldenVelocity);
        Assert.True(cos > 0.99f, $"StableAudioDiT cosine-sim too low: {cos}");
    }
}
