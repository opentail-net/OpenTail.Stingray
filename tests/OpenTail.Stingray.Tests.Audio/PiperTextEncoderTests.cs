
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Verifies PiperTextEncoder (VITS enc_p: embedding + windowed relative-position Transformer +
/// proj to mu/logs) against real ONNX weights and real onnxruntime golden output
/// (scratch-llamacpp-ref/piper_golden_textenc.py's `/enc_p/proj/Conv_output_0`, [1,384,T]).
/// </summary>
public sealed class PiperTextEncoderTests : HeavyTestBase
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
    public void PiperTextEncoder_RealOnnxWeights_MatchesOnnxGoldenOutput()
    {
        string? modelPath = FindRepoFile("models/en_US-lessac-medium.onnx");
        string? goldenPath = FindRepoFile("scratch-llamacpp-ref/piper_golden_textenc/enc_p_proj_Conv_output_0.npy");
        if (modelPath is null || goldenPath is null) return;

        var weights = new PiperOnnxWeights(modelPath);

        // Matches piper_golden_textenc.py's fixed input_ids exactly.
        int[] tokens = [1, 0, 25, 0, 32, 0, 41, 0, 38, 0, 2];

        var (_, mu, logs) = PiperTextEncoder.Forward(weights, tokens);

        int t = tokens.Length;
        int dim = weights.HiddenDim;
        Assert.Equal(dim * t, mu.Length);
        Assert.Equal(dim * t, logs.Length);

        float[] golden = ReadNpyFloat32(goldenPath); // [1, 384, T] = [mu; logs] concatenated on channel dim
        Assert.Equal(2 * dim * t, golden.Length);

        var goldenMu = new float[dim * t];
        var goldenLogs = new float[dim * t];
        Array.Copy(golden, 0, goldenMu, 0, dim * t);
        Array.Copy(golden, dim * t, goldenLogs, 0, dim * t);

        double muCosine = CosineSimilarity(mu, goldenMu);
        double logsCosine = CosineSimilarity(logs, goldenLogs);
        Assert.True(muCosine > 0.999, $"TextEncoder mu cosine similarity {muCosine} too low vs golden ONNX output.");
        Assert.True(logsCosine > 0.999, $"TextEncoder logs cosine similarity {logsCosine} too low vs golden ONNX output.");
    }
}
