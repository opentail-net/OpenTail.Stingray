
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Verifies MeloTTS's HiFi-GAN Generator (`dec`) against real ONNX weights and real onnxruntime
/// golden output (scratch-llamacpp-ref/melo_golden_generator.py). Feeds the golden flow output
/// directly (`/flow/flows.0/Concat_output_0`, == dec's actual input since the mask is all-ones
/// for our unpadded test input), isolating generator correctness from flow correctness.
/// </summary>
public sealed class MeloGeneratorTests : HeavyTestBase
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
    public void MeloGenerator_RealOnnxWeights_MatchesOnnxGoldenWaveform()
    {
        string? modelPath = FindRepoFile("models/melotts-zh_en.onnx");
        string? zPath = FindRepoFile("scratch-llamacpp-ref/melo_golden_generator/flow_flows.0_Concat_output_0.npy");
        string? yPath = FindRepoFile("scratch-llamacpp-ref/melo_golden_generator/y.npy");
        if (modelPath is null || zPath is null || yPath is null) return;

        var weights = new MeloOnnxWeights(modelPath);

        float[] z = ReadNpyFloat32(zPath); // [1,192,T] channel-first
        int t = z.Length / weights.HiddenDim;

        const int speakerId = 0;
        var g = new float[weights.GinChannels];
        Array.Copy(weights.EmbGWeight, speakerId * weights.GinChannels, g, 0, weights.GinChannels);

        float[] waveform = MeloGenerator.Forward(weights, z, t, g);

        float[] goldenWaveform = ReadNpyFloat32(yPath);
        Assert.Equal(goldenWaveform.Length, waveform.Length);

        foreach (float v in waveform) Assert.False(float.IsNaN(v) || float.IsInfinity(v), "waveform must be finite");

        double cosine = CosineSimilarity(waveform, goldenWaveform);
        Assert.True(cosine > 0.99, $"Generator waveform cosine similarity {cosine} too low vs golden ONNX output.");
    }
}
