using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Xtts;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real numeric golden verification for <see cref="XttsMelExtractor.ForConditioningCloning"/>
/// (the REAL `wav_to_mel_cloning` conditioning-encoder mel frontend, n_fft=2048/hop=256/win=1024 --
/// distinct from the DVAE-path n_fft=1024 config already verified by
/// <see cref="XttsMelExtractorTests"/>), against `scratch-llamacpp-ref/xtts_cloning_mel_golden.py`'s
/// real output on the same reference audio clip.</summary>
public sealed class XttsCloningMelExtractorTests : HeavyTestBase
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

    private static float[] ReadCsv(string path) =>
        Array.ConvertAll(File.ReadAllText(path).Trim().Split(','), float.Parse);

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
    }

    [Fact]
    public void ExtractMel_RealAudio_MatchesGoldenOracle()
    {
        string? melStatsPath = FindRepoFile("models/xtts-v2/mel_stats.safetensors");
        Assert.SkipUnless(melStatsPath != null, "models/xtts-v2/mel_stats.safetensors not found");
        string? wavPath = FindRepoFile("scratch-llamacpp-ref/xtts_cloning_mel_golden_wav.txt");
        string? outPath = FindRepoFile("scratch-llamacpp-ref/xtts_cloning_mel_golden_output.txt");
        Assert.SkipUnless(wavPath != null && outPath != null, "golden cloning-mel files not found (re-run scratch-llamacpp-ref/xtts_cloning_mel_golden.py)");

        using var loader = SafetensorsLoader.Open(melStatsPath!);
        float[] melStats = loader.ReadF32("mel_stats");

        float[] wav = ReadCsv(wavPath!);

        var extractor = XttsMelExtractor.ForConditioningCloning();
        float[] mel = extractor.ExtractMel(wav, melStats);

        float[] golden = ReadCsv(outPath!);
        Assert.Equal(golden.Length, mel.Length);
        foreach (float v in mel)
        {
            Assert.False(float.IsNaN(v), "mel output must not contain NaN");
            Assert.False(float.IsInfinity(v), "mel output must not contain Infinity");
        }
        double cosine = Cosine(mel, golden);
        Assert.True(cosine > 0.99, $"mel output cosine {cosine} too low vs golden");
    }
}
