using OpenTail.Stingray.Audio.Xtts;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real numeric golden verification for <see cref="XttsHifiDecoder"/> (the real
/// `HifiDecoder.forward`: two chained linear-interpolation upsample stages feeding the vocoder),
/// against `scratch-llamacpp-ref/xtts_hifidecoder_golden.py`'s real output on a fixed synthetic
/// `[1,15,1024]` latent input and a fixed synthetic `[1,512]` d-vector.</summary>
public sealed class XttsHifiDecoderTests : HeavyTestBase
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
        string? latentsPath = FindRepoFile("scratch-llamacpp-ref/xtts_hifidecoder_golden_latents.txt");
        string? gPath = FindRepoFile("scratch-llamacpp-ref/xtts_hifidecoder_golden_g.txt");
        string? outPath = FindRepoFile("scratch-llamacpp-ref/xtts_hifidecoder_golden_out.txt");
        Assert.SkipUnless(latentsPath != null && gPath != null && outPath != null, "golden hifi decoder files not found (re-run scratch-llamacpp-ref/xtts_hifidecoder_golden.py)");

        using var loader = SafetensorsLoader.Open(weightsPath!);
        var weights = new XttsVocoderWeights(loader, "hifigan_decoder.waveform_decoder");

        float[] latents = ReadCsv(latentsPath!);
        float[] g = ReadCsv(gPath!);
        const int tIn = 15;
        Assert.Equal(XttsVocoderWeights.InChannels * tIn, latents.Length);
        Assert.Equal(XttsVocoderWeights.CondChannels, g.Length);

        float[] waveform = XttsHifiDecoder.Forward(weights, latents, tIn, g);
        float[] golden = ReadCsv(outPath!);

        Assert.Equal(golden.Length, waveform.Length);
        foreach (float v in waveform)
        {
            Assert.False(float.IsNaN(v), "waveform must not contain NaN");
            Assert.False(float.IsInfinity(v), "waveform must not contain Infinity");
        }
        double cosine = Cosine(waveform, golden);
        Assert.True(cosine > 0.99, $"hifi decoder waveform cosine {cosine} too low vs golden");
    }
}
