using OpenTail.Stingray.Audio.Xtts;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real numeric golden verification for <see cref="XttsDvaeDecoder"/>, against
/// `scratch-llamacpp-ref/xtts_dvae_decoder_golden.py`'s real `coqui-ai-TTS` `DiscreteVAE.decode`
/// output on a fixed deterministic code sequence.</summary>
public sealed class XttsDvaeDecoderTests : HeavyTestBase
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
    public void Decode_RealWeights_MatchesGoldenOracle()
    {
        string? weightsPath = FindRepoFile("models/xtts-v2/dvae.safetensors");
        Assert.SkipUnless(weightsPath != null, "models/xtts-v2/dvae.safetensors not found (run scratch-llamacpp-ref/xtts_convert_to_safetensors.py)");
        string? codesPath = FindRepoFile("scratch-llamacpp-ref/xtts_dvae_golden_codes.txt");
        string? outPath = FindRepoFile("scratch-llamacpp-ref/xtts_dvae_golden_output.txt");
        Assert.SkipUnless(codesPath != null && outPath != null, "golden DVAE files not found (re-run scratch-llamacpp-ref/xtts_dvae_decoder_golden.py)");

        var weights = new XttsDvaeWeights(weightsPath!);
        var codes = Array.ConvertAll(File.ReadAllText(codesPath!).Trim().Split(','), int.Parse);

        float[] output = XttsDvaeDecoder.Decode(weights, codes);
        float[] golden = ReadCsv(outPath!);

        Assert.Equal(golden.Length, output.Length);
        foreach (float v in output)
        {
            Assert.False(float.IsNaN(v), "DVAE decoder output must not contain NaN");
            Assert.False(float.IsInfinity(v), "DVAE decoder output must not contain Infinity");
        }
        double cosine = Cosine(output, golden);
        Assert.True(cosine > 0.99, $"DVAE decoder output cosine {cosine} too low vs golden");
    }
}
