using System;
using System.IO;
using System.Text;
using OpenTail.Stingray.Audio.Kokoro;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio.Fast;

public sealed class KokoroF0NtrainTests : HeavyTestBase
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

    /// <summary>
    /// Verifies ProsodyPredictor.F0Ntrain (predictor.shared BiLSTM + F0/N AdainResBlk1d stacks
    /// + F0_proj/N_proj) in isolation: feeds the real onnxruntime-computed `en` tensor
    /// (scratch-llamacpp-ref/kokoro_golden_f0n.py's `/encoder/MatMul_output_0`, [1,640,42])
    /// directly into KokoroProsodyPredictor.F0Ntrain, bypassing the not-yet-chained
    /// DurationEncoder/alignment stages (each already verified independently), and compares
    /// against the golden F0_proj/N_proj outputs ([1,1,84] -- the F0/N stacks upsample x2).
    /// </summary>
    [Fact]
    public void KokoroF0Ntrain_RealWeights_MatchesOnnxGoldenOutput()
    {
        string? modelPath = FindRepoFile("models/kokoro-82m-q8_0.gguf");
        string? enPath = FindRepoFile("scratch-llamacpp-ref/kokoro_golden_f0n/encoder_MatMul_output_0.npy");
        string? goldenF0Path = FindRepoFile("scratch-llamacpp-ref/kokoro_golden_f0n/encoder_F0_proj_Conv_output_0.npy");
        string? goldenNPath = FindRepoFile("scratch-llamacpp-ref/kokoro_golden_f0n/encoder_N_proj_Conv_output_0.npy");
        if (modelPath is null || enPath is null || goldenF0Path is null || goldenNPath is null) return;

        using var weights = new KokoroWeights(modelPath);

        var predictorStyle = new float[weights.StyleDim]; // ref_s[:,128:] is all-zero for this golden fixture (see KokoroProsodyPredictorTests)
        float[] en = ReadNpyFloat32(enPath);
        int frames = en.Length / (weights.HiddenDim + weights.StyleDim);
        Assert.Equal(42, frames);

        var (f0Curve, nCurve) = KokoroProsodyPredictor.F0Ntrain(weights, en, frames, predictorStyle);

        float[] goldenF0 = ReadNpyFloat32(goldenF0Path);
        float[] goldenN = ReadNpyFloat32(goldenNPath);
        Assert.Equal(84, f0Curve.Length);
        Assert.Equal(84, nCurve.Length);
        Assert.Equal(goldenF0.Length, f0Curve.Length);
        Assert.Equal(goldenN.Length, nCurve.Length);

        double f0Cosine = CosineSimilarity(f0Curve, goldenF0);
        double nCosine = CosineSimilarity(nCurve, goldenN);
        Assert.True(f0Cosine > 0.998, $"F0Ntrain F0_pred cosine similarity {f0Cosine} too low vs golden ONNX output.");
        Assert.True(nCosine > 0.998, $"F0Ntrain N_pred cosine similarity {nCosine} too low vs golden ONNX output.");
    }
}
