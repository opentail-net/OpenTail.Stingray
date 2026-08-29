
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Verifies MeloTTS's use of the shared <see cref="VitsLengthRegulator"/> (token-rate mu/logs
/// bridged to frame-rate z_p via per-token durations) against real ONNX weights and real
/// onnxruntime golden output (scratch-llamacpp-ref/melo_golden_lengthreg.py). Uses OUR OWN
/// (already golden-verified, see MeloTextEncoderTests) TextEncoder output for mu/logs, but the
/// REAL golden per-token durations (`/Ceil_output_0`) and REAL golden frame-rate noise draw
/// (`/RandomNormalLike_output_0`), so this isolates length-regulator correctness end-to-end
/// without needing to re-derive sdp_ratio-blended durations ourselves.
/// </summary>
public sealed class MeloLengthRegulatorTests : HeavyTestBase
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

    private static float[] ReadNpyFloat32(string path)
    {
        byte[] data = File.ReadAllBytes(path);
        if (data[0] != 0x93 || Encoding.ASCII.GetString(data, 1, 5) != "NUMPY")
            throw new InvalidDataException($"Not a .npy file: {path}");
        byte major = data[6];
        int headerLen, headerStart;
        if (major == 1) { headerLen = data[8] | (data[9] << 8); headerStart = 10; }
        else { headerLen = data[8] | (data[9] << 8) | (data[10] << 16) | (data[11] << 24); headerStart = 12; }
        int dataStart = headerStart + headerLen;
        int floatCount = (data.Length - dataStart) / 4;
        var result = new float[floatCount];
        Buffer.BlockCopy(data, dataStart, result, 0, floatCount * 4);
        return result;
    }

    private static double CosineSimilarity(float[] a, float[] b)
    {
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            normA += (double)a[i] * a[i];
            normB += (double)b[i] * b[i];
        }
        return dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
    }

    [Fact]
    public void MeloLengthRegulator_RealOnnxWeights_MatchesOnnxGoldenZp()
    {
        string? modelPath = FindRepoFile("models/melotts-zh_en.onnx");
        string? durPath = FindRepoFile("scratch-llamacpp-ref/melo_golden_lengthreg/Ceil_output_0.npy");
        string? noisePath = FindRepoFile("scratch-llamacpp-ref/melo_golden_lengthreg/RandomNormalLike_output_0.npy");
        string? zpPath = FindRepoFile("scratch-llamacpp-ref/melo_golden_lengthreg/Add_2_output_0.npy");
        if (modelPath is null || durPath is null || noisePath is null || zpPath is null) return;

        var weights = new MeloOnnxWeights(modelPath);

        int[] tokens = [1, 5, 10, 20, 30, 40, 50, 2];
        int[] tones = [0, 1, 2, 3, 4, 5, 6, 0];
        const int speakerId = 0;
        int tTokens = tokens.Length;

        var (_, mu, logs) = MeloTextEncoder.Forward(weights, tokens, tones, speakerId);

        float[] durFloats = ReadNpyFloat32(durPath);
        Assert.Equal(tTokens, durFloats.Length);
        var durations = new int[tTokens];
        for (int i = 0; i < tTokens; i++) durations[i] = (int)durFloats[i];

        float[] noise = ReadNpyFloat32(noisePath);
        const float noiseScale = 0.6f; // matches melo_golden_lengthreg.py's noise_scale

        var (zp, tFrames, _) = VitsLengthRegulator.ExpandWithDurations(mu, logs, weights.HiddenDim, tTokens, durations, noise, noiseScale);

        float[] goldenZp = ReadNpyFloat32(zpPath);
        Assert.Equal(goldenZp.Length, zp.Length);
        Assert.Equal(goldenZp.Length, weights.HiddenDim * tFrames);

        foreach (float v in zp) Assert.False(float.IsNaN(v) || float.IsInfinity(v), "zp must be finite");

        double cosine = CosineSimilarity(zp, goldenZp);
        Assert.True(cosine > 0.99, $"Length-regulator z_p cosine similarity {cosine} too low vs golden ONNX output.");
    }
}
