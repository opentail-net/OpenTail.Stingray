
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric golden verification for <see cref="FunAsrRealMelExtractor"/> (see docs/audio-
/// review-progress.md's FunASR section) -- compares against `scratch-llamacpp-ref/
/// funasr_golden_frontend.py`, which uses the REAL `torchaudio.compliance.kaldi.fbank`
/// function directly (not a reimplementation) plus the real `apply_lfr`/`apply_cmvn` functions
/// copied verbatim from the real `funasr` package.
/// </summary>
public sealed class FunAsrRealMelExtractorTests : HeavyTestBase
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

    private static float[] ParseCsv(string path, int expectedLength)
    {
        var parts = File.ReadAllText(path).Trim().Split(',');
        Assert.Equal(expectedLength, parts.Length);
        var arr = new float[expectedLength];
        for (int i = 0; i < expectedLength; i++) arr[i] = float.Parse(parts[i]);
        return arr;
    }

    [Fact]
    public void Extract_RealWeights_MatchesGoldenFrontendOutput()
    {
        string? modelPath = FindRepoFile("models/paraformer-q8.gguf");
        Assert.SkipUnless(modelPath != null, "models/paraformer-q8.gguf not found");
        string? pcmPath = FindRepoFile("scratch-llamacpp-ref/funasr_golden_frontend_pcm.txt");
        string? featsPath = FindRepoFile("scratch-llamacpp-ref/funasr_golden_frontend_feats.txt");
        Assert.SkipUnless(pcmPath != null && featsPath != null,
            "golden frontend files not found (re-run scratch-llamacpp-ref/funasr_golden_frontend.py)");

        const int numSamples = 32000; // 2s @ 16kHz
        var pcm = ParseCsv(pcmPath!, numSamples);

        var lines = File.ReadAllText(featsPath!).Split('\n');
        var dims = lines[0].Trim().Split(',');
        int goldenT = int.Parse(dims[0]);
        int goldenDim = int.Parse(dims[1]);
        var goldenFlat = ParseCsv2(lines[1], goldenT * goldenDim);

        using var w = new FunAsrWeights(modelPath!);
        var extractor = new FunAsrRealMelExtractor();
        var feats = extractor.Extract(pcm, w.CmvnShift, w.CmvnScale);

        Assert.Equal(goldenT, feats.Length);
        Assert.Equal(goldenDim, feats[0].Length);

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < goldenT; i++)
        {
            for (int d = 0; d < goldenDim; d++)
            {
                float a = feats[i][d];
                float b = goldenFlat[i * goldenDim + d];
                dot += a * b;
                normA += a * a;
                normB += b * b;
            }
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.True(cosine > 0.99, $"cosine similarity {cosine} too low vs golden frontend output");
    }

    private static float[] ParseCsv2(string csv, int expectedLength)
    {
        var parts = csv.Trim().Split(',');
        Assert.Equal(expectedLength, parts.Length);
        var arr = new float[expectedLength];
        for (int i = 0; i < expectedLength; i++) arr[i] = float.Parse(parts[i]);
        return arr;
    }
}
