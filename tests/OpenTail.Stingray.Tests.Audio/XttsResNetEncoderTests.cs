using OpenTail.Stingray.Audio.Xtts;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real numeric golden verification for <see cref="XttsResNetEncoder"/> (the real
/// ResNet-SE speaker encoder), against `scratch-llamacpp-ref/xtts_resnet_encoder_golden.py`'s
/// real `ResNetSpeakerEncoder.forward` output on real reference audio.</summary>
public sealed class XttsResNetEncoderTests : HeavyTestBase
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
    public void Forward_RealAudio_MatchesGoldenOracle()
    {
        string? weightsPath = FindRepoFile("models/xtts-v2/model.safetensors");
        Assert.SkipUnless(weightsPath != null, "models/xtts-v2/model.safetensors not found");
        string? wavPath = FindRepoFile("scratch-llamacpp-ref/xtts_speaker_mel_golden_wav.txt");
        string? outPath = FindRepoFile("scratch-llamacpp-ref/xtts_resnet_encoder_golden_output.txt");
        Assert.SkipUnless(wavPath != null && outPath != null, "golden ResNet encoder files not found (re-run scratch-llamacpp-ref/xtts_resnet_encoder_golden.py)");

        using var loader = SafetensorsLoader.Open(weightsPath!);
        var weights = new XttsResNetWeights(loader, "hifigan_decoder.speaker_encoder");

        float[] wav = ReadCsv(wavPath!);
        float[] preemph = XttsSpeakerMelExtractor.Preemphasis(wav);
        var melExtractor = new XttsSpeakerMelExtractor();
        float[] mel = melExtractor.ExtractMel(preemph);
        int t = mel.Length / XttsSpeakerMelExtractor.NumMels;

        float[] embedding = XttsResNetEncoder.Forward(weights, mel, t);
        float[] golden = ReadCsv(outPath!);

        Assert.Equal(golden.Length, embedding.Length);
        foreach (float v in embedding)
        {
            Assert.False(float.IsNaN(v), "speaker embedding must not contain NaN");
            Assert.False(float.IsInfinity(v), "speaker embedding must not contain Infinity");
        }
        double cosine = Cosine(embedding, golden);
        Assert.True(cosine > 0.99, $"speaker embedding cosine {cosine} too low vs golden");
    }
}
