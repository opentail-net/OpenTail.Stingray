using OpenTail.Stingray.Audio.MusicGen;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric golden verification for <see cref="MusicGenTextEncoder"/> (the real, bundled
/// stock t5-base encoder used for MusicGen's text conditioning) -- compares against
/// `scratch-llamacpp-ref/musicgen_t5_encoder_golden.py`, which loads the real, already-local
/// `models/musicgen-small/musicgen-small.safetensors` `text_encoder.*` tensor tree directly via
/// safetensors and computes the real 12-layer T5 encoder math (relative-position-bias
/// self-attention with no 1/sqrt(head_dim) scaling, RMSNorm-only T5LayerNorm, plain ReLU FFN)
/// in numpy, transcribed from the real `transformers` `modeling_t5.py`.
///
/// Closes gap 1/3 of the "not just the decoder" MusicGen numeric-parity closure requested
/// 2026-09-04 (decoder-only golden parity was previously confirmed in
/// <see cref="MusicGenDecoderGoldenParityTests"/> but used a FAKE all-0.05 T5 stand-in --
/// this test is the first real numeric check of the actual T5 encoder math itself).
/// </summary>
public sealed class MusicGenTextEncoderGoldenParityTests : HeavyTestBase
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
    public void Forward_RealWeights_MatchesGoldenOutput()
    {
        string? modelPath = FindRepoFile("models/musicgen-small/musicgen-small.safetensors");
        Assert.SkipUnless(modelPath != null, "models/musicgen-small/musicgen-small.safetensors not found");

        string? idsPath = FindRepoFile("scratch-llamacpp-ref/musicgen_t5_encoder_golden_ids.txt");
        string? hiddenPath = FindRepoFile("scratch-llamacpp-ref/musicgen_t5_encoder_golden_hidden.txt");
        Assert.SkipUnless(idsPath != null && hiddenPath != null,
            "golden MusicGen T5 encoder files not found (re-run scratch-llamacpp-ref/musicgen_t5_encoder_golden.py)");

        var tokenIds = Array.ConvertAll(File.ReadAllText(idsPath!).Trim().Split(','), int.Parse);

        var hiddenLines = File.ReadAllText(hiddenPath!).Split('\n');
        var dims = hiddenLines[0].Trim().Split(',');
        int goldenT = int.Parse(dims[0]);
        int goldenDim = int.Parse(dims[1]);
        var goldenParts = hiddenLines[1].Trim().Split(',');
        Assert.Equal(goldenT * goldenDim, goldenParts.Length);
        var golden = new float[goldenT * goldenDim];
        for (int i = 0; i < golden.Length; i++) golden[i] = float.Parse(goldenParts[i]);

        Assert.Equal(tokenIds.Length, goldenT);
        Assert.Equal(MusicGenConfig.TextDModel, goldenDim);

        using var loader = SafetensorsLoader.Open(modelPath!);
        var weights = MusicGenTextEncoderWeights.Load(loader);

        var output = MusicGenTextEncoder.Forward(weights, tokenIds);

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < goldenT; i++)
        {
            for (int d = 0; d < goldenDim; d++)
            {
                float a = output[i][d];
                float b = golden[i * goldenDim + d];
                dot += a * b;
                normA += a * a;
                normB += b * b;
            }
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.True(cosine > 0.999, $"cosine similarity {cosine} too low vs golden MusicGen T5 encoder output");
    }
}
