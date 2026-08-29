
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Verifies MeloTTS's TransformerCouplingBlock flow (reverse direction) against real ONNX
/// weights and real onnxruntime golden output (scratch-llamacpp-ref/melo_golden_flow.py). Feeds
/// the golden-dumped z_p (`/Add_2_output_0`, the frame-rate prior-latent sample the length
/// regulator would otherwise produce) directly, isolating flow correctness from length-regulator
/// correctness -- same bisection philosophy as MeloDurationPredictorTests.
/// </summary>
public sealed class MeloFlowTests : HeavyTestBase
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
        int headerLen;
        int headerStart;
        if (major == 1)
        {
            headerLen = data[8] | (data[9] << 8);
            headerStart = 10;
        }
        else
        {
            headerLen = data[8] | (data[9] << 8) | (data[10] << 16) | (data[11] << 24);
            headerStart = 12;
        }
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
    public void MeloFlow_RealOnnxWeights_MatchesOnnxGoldenOutput()
    {
        string? modelPath = FindRepoFile("models/melotts-zh_en.onnx");
        string? zpPath = FindRepoFile("scratch-llamacpp-ref/melo_golden_flow/Add_2_output_0.npy");
        string? outPath = FindRepoFile("scratch-llamacpp-ref/melo_golden_flow/flow_flows.0_Concat_output_0.npy");
        if (modelPath is null || zpPath is null || outPath is null) return;

        var weights = new MeloOnnxWeights(modelPath);

        float[] zp = ReadNpyFloat32(zpPath); // [1, 192, T_frame] channel-first
        int t = zp.Length / weights.HiddenDim;
        Assert.Equal(weights.HiddenDim * t, zp.Length);

        const int speakerId = 0;
        var g = new float[weights.GinChannels];
        Array.Copy(weights.EmbGWeight, speakerId * weights.GinChannels, g, 0, weights.GinChannels);

        float[] z = MeloFlow.Reverse(weights, zp, t, g);

        float[] goldenZ = ReadNpyFloat32(outPath);
        Assert.Equal(goldenZ.Length, z.Length);

        foreach (float v in z) Assert.False(float.IsNaN(v) || float.IsInfinity(v), "flow output must be finite");

        double cosine = CosineSimilarity(z, goldenZ);
        Assert.True(cosine > 0.99, $"Flow output cosine similarity {cosine} too low vs golden ONNX output.");
    }
}
