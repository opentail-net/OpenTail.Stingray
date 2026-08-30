using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Xtts;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real numeric golden verification for <see cref="XttsMelExtractor"/>, against
/// `scratch-llamacpp-ref/xtts_mel_extractor_golden.py`'s real `dvae_wav_to_mel` output on a real
/// reference audio clip already in this repo. Confirms the HTK (not Slaney-scale) mel-frequency
/// conversion this port deliberately used instead of copying `CosyVoiceMelExtractor`'s different
/// formula.</summary>
public sealed class XttsMelExtractorTests : HeavyTestBase
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
        string? wavPath = FindRepoFile("scratch-llamacpp-ref/xtts_mel_golden_wav.txt");
        string? outPath = FindRepoFile("scratch-llamacpp-ref/xtts_mel_golden_output.txt");
        Assert.SkipUnless(wavPath != null && outPath != null, "golden mel files not found (re-run scratch-llamacpp-ref/xtts_mel_extractor_golden.py)");

        using var loader = SafetensorsLoader.Open(melStatsPath!);
        float[] melStats = loader.ReadF32("mel_stats");

        // Golden wav is already real 22050Hz mono float samples (the golden script did the
        // resample/downmix itself) -- feed directly, no WavReader/AudioResampler round-trip
        // needed to isolate the mel math itself.
        float[] wav = ReadCsv(wavPath!);

        var extractor = new XttsMelExtractor();
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
