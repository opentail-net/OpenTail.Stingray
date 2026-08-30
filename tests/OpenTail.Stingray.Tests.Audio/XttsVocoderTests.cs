using OpenTail.Stingray.Audio.Xtts;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real numeric golden verification for <see cref="XttsVocoder"/> (the real FiLM-conditioned
/// HiFi-GAN vocoder, `hifigan_decoder.waveform_decoder`), against
/// `scratch-llamacpp-ref/xtts_vocoder_golden.py`'s real `HifiganGenerator.forward` output on a fixed
/// synthetic `[1,1024,20]` latent input and a fixed synthetic `[1,512]` d-vector (isolating this
/// stage, matching the established staged-verification pattern used for every prior XTTS-v2
/// piece).</summary>
public sealed class XttsVocoderTests : HeavyTestBase
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
        Array.ConvertAll(File.ReadAllText(path).Trim(',').Split(','), float.Parse);

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
    }

    [Fact]
    public void Forward_SyntheticInput_MatchesGoldenOracle()
    {
        string? weightsPath = FindRepoFile("models/xtts-v2/model.safetensors");
        Assert.SkipUnless(weightsPath != null, "models/xtts-v2/model.safetensors not found");
        string? xPath = FindRepoFile("scratch-llamacpp-ref/xtts_vocoder_golden_x.txt");
        string? gPath = FindRepoFile("scratch-llamacpp-ref/xtts_vocoder_golden_g.txt");
        string? outPath = FindRepoFile("scratch-llamacpp-ref/xtts_vocoder_golden_out.txt");
        Assert.SkipUnless(xPath != null && gPath != null && outPath != null, "golden vocoder files not found (re-run scratch-llamacpp-ref/xtts_vocoder_golden.py)");

        using var loader = SafetensorsLoader.Open(weightsPath!);
        var weights = new XttsVocoderWeights(loader, "hifigan_decoder.waveform_decoder");

        float[] x = ReadCsv(xPath!);
        float[] g = ReadCsv(gPath!);
        const int t = 20;
        Assert.Equal(XttsVocoderWeights.InChannels * t, x.Length);
        Assert.Equal(XttsVocoderWeights.CondChannels, g.Length);

        float[] waveform = XttsVocoder.Forward(weights, x, t, g);
        float[] golden = ReadCsv(outPath!);

        Assert.Equal(golden.Length, waveform.Length);
        foreach (float v in waveform)
        {
            Assert.False(float.IsNaN(v), "waveform must not contain NaN");
            Assert.False(float.IsInfinity(v), "waveform must not contain Infinity");
        }
        double cosine = Cosine(waveform, golden);
        Assert.True(cosine > 0.99, $"vocoder waveform cosine {cosine} too low vs golden");
    }
}
