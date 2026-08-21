using System;
using System.IO;
using System.Text;
using OpenTail.Stingray.Audio.Kokoro;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio.Fast;

public sealed class KokoroProsodyPredictorTests : HeavyTestBase
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
    /// Verifies DurationEncoder (`d`) and predictor.lstm+duration_proj+sigmoid-sum (pred_dur,
    /// pre-round/clamp) against a real onnxruntime run of models/kokoro-v1.0.onnx
    /// (scratch-llamacpp-ref/kokoro_golden_durenc.py), chained on top of the already-verified
    /// BERT encoder stage. Uses the fixed style vector matching the golden script's
    /// (style[0,:5] = [0.1,-0.2,0.3,0.05,-0.1], rest zero) and speed=1.0.
    /// </summary>
    [Fact]
    public void KokoroProsodyPredictor_RealWeights_MatchesOnnxGoldenOutput()
    {
        string? modelPath = FindRepoFile("models/kokoro-82m-q8_0.gguf");
        string? goldenDPath = FindRepoFile("scratch-llamacpp-ref/kokoro_golden_durenc/encoder_predictor_text_encoder_Concat_4_output_0.npy");
        string? goldenDurSumPath = FindRepoFile("scratch-llamacpp-ref/kokoro_golden_durenc/encoder_predictor_ReduceSum_output_0.npy");
        if (modelPath is null || goldenDPath is null || goldenDurSumPath is null) return;

        using var weights = new KokoroWeights(modelPath);

        int[] inputIds = [0, 50, 83, 54, 156, 57, 135, 3, 16, 65, 156, 0];
        int t = inputIds.Length;

        // Matches kokoro_golden*.py's fixed style vector: style[0,:5] = [0.1,-0.2,0.3,0.05,-0.1],
        // rest zero, with s = ref_s[:, StyleDim:] (the predictor-facing half of the 256-dim vector).
        var fullStyle = new float[2 * weights.StyleDim];
        fullStyle[0] = 0.1f;
        fullStyle[1] = -0.2f;
        fullStyle[2] = 0.3f;
        fullStyle[3] = 0.05f;
        fullStyle[4] = -0.1f;
        var predictorStyle = new float[weights.StyleDim];
        Array.Copy(fullStyle, weights.StyleDim, predictorStyle, 0, weights.StyleDim);

        float[] lastHidden = KokoroBertEncoder.Forward(weights, inputIds);
        float[] dEn = KokoroBertEncoder.ProjectToWorkingDim(weights, lastHidden, t);

        float[] d = KokoroProsodyPredictor.EncodeDuration(weights, dEn, predictorStyle, t);
        float[] goldenD = ReadNpyFloat32(goldenDPath);
        Assert.Equal(t * (weights.HiddenDim + weights.StyleDim), goldenD.Length);
        double dCosine = CosineSimilarity(d, goldenD);
        // Threshold is slightly relaxed vs the BERT stage's 0.999: `d` is BERT's already-~0.999
        // output run through 3 more stacked LSTM+AdaLayerNorm blocks, so Q8_0 quantization noise
        // compounds further. Verified this isn't a structural bug by checking per-timestep cosine
        // similarity (uniformly 0.998-0.9996 across all 12 tokens, no outlier/discontinuity that
        // would indicate e.g. a wrong channel range or an off-by-one in the AdaLN style re-concat).
        Assert.True(dCosine > 0.998, $"DurationEncoder `d` cosine similarity {dCosine} too low vs golden ONNX output.");

        float[] durations = KokoroProsodyPredictor.PredictDurations(weights, d, t);
        float[] goldenDurSums = ReadNpyFloat32(goldenDurSumPath);
        Assert.Equal(t, goldenDurSums.Length);
        double durCosine = CosineSimilarity(durations, goldenDurSums);
        Assert.True(durCosine > 0.999, $"Predicted duration sums cosine similarity {durCosine} too low vs golden ONNX output.");
    }
}
