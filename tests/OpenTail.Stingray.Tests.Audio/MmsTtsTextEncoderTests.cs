using OpenTail.Stingray.Audio.MmsTts;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real numeric golden verification for <see cref="MmsTtsTextEncoder"/>, against
/// `scratch-llamacpp-ref/mms_tts_golden.py`'s real HuggingFace `transformers.VitsModel` output on
/// the deterministic input "hello world".</summary>
public sealed class MmsTtsTextEncoderTests : HeavyTestBase
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
    public void Forward_RealWeights_MatchesGoldenOracle()
    {
        string? modelDir = FindRepoFile("models/mms-tts-eng/model.safetensors");
        Assert.SkipUnless(modelDir != null, "models/mms-tts-eng/model.safetensors not found");
        string? idsPath = FindRepoFile("scratch-llamacpp-ref/mms_tts_golden_input_ids.txt");
        string? hiddenPath = FindRepoFile("scratch-llamacpp-ref/mms_tts_golden_encoder_hidden.txt");
        string? muPath = FindRepoFile("scratch-llamacpp-ref/mms_tts_golden_mu.txt");
        Assert.SkipUnless(idsPath != null && hiddenPath != null && muPath != null,
            "golden MMS-TTS files not found (re-run scratch-llamacpp-ref/mms_tts_golden.py)");

        string configPath = FindRepoFile("models/mms-tts-eng/config.json")!;
        var config = MmsTtsConfig.Load(configPath);
        var weights = new MmsTtsWeights(modelDir!, config);

        var ids = Array.ConvertAll(File.ReadAllText(idsPath!).Trim().Split(','), int.Parse);
        var (encoderHidden, mu, _) = MmsTtsTextEncoder.Forward(weights, ids);

        var goldenHidden = ReadCsv(hiddenPath!);
        var goldenMu = ReadCsv(muPath!);

        Assert.Equal(goldenHidden.Length, encoderHidden.Length);
        Assert.Equal(goldenMu.Length, mu.Length);

        double cosHidden = Cosine(encoderHidden, goldenHidden);
        double cosMu = Cosine(mu, goldenMu);
        Assert.True(cosHidden > 0.99, $"encoderHidden cosine {cosHidden} too low vs golden");
        Assert.True(cosMu > 0.99, $"mu cosine {cosMu} too low vs golden");
    }
}
