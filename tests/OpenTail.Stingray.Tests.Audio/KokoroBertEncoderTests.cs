using System;
using System.IO;
using System.Text;
using OpenTail.Stingray.Audio.Kokoro;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

public sealed class KokoroBertEncoderTests
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

    /// <summary>Minimal .npy reader for float32 arrays (version 1.0 header), enough to read scratch-llamacpp-ref/kokoro_golden_bert/*.npy.</summary>
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

    /// <summary>
    /// Verifies the BERT stage (embeddings -> 12x shared ALBERT layer) against a real
    /// onnxruntime run of models/kokoro-v1.0.onnx (scratch-llamacpp-ref/kokoro_golden_bert.py),
    /// using the exact same fixed input_ids. This is the first numerically-verified stage of
    /// the Kokoro forward pass rewrite -- see docs/audio-review-progress.md.
    /// </summary>
    [Fact]
    public void KokoroBertEncoder_RealWeights_MatchesOnnxGoldenOutput()
    {
        string? modelPath = FindRepoFile("models/kokoro-82m-q8_0.gguf");
        string? goldenEmb = FindRepoFile("scratch-llamacpp-ref/kokoro_golden_bert/encoder_bert_embeddings_LayerNorm_LayerNormalization_output_0.npy");
        string? goldenFinal = FindRepoFile("scratch-llamacpp-ref/kokoro_golden_bert/encoder_bert_encoder_albert_layer_groups.0_albert_layers.0_full_layer_layer_norm_11_LayerNormalization_output_0.npy");
        if (modelPath is null || goldenEmb is null || goldenFinal is null) return;

        using var weights = new KokoroWeights(modelPath);

        int[] inputIds = [0, 50, 83, 54, 156, 57, 135, 3, 16, 65, 156, 0];
        int t = inputIds.Length;

        float[] lastHidden = KokoroBertEncoder.Forward(weights, inputIds);

        float[] golden = ReadNpyFloat32(goldenFinal);
        Assert.Equal(t * weights.BertHiddenSize, golden.Length);

        double maxAbsDiff = 0;
        double sumAbsDiff = 0;
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < golden.Length; i++)
        {
            double diff = Math.Abs(lastHidden[i] - golden[i]);
            maxAbsDiff = Math.Max(maxAbsDiff, diff);
            sumAbsDiff += diff;
            dot += (double)lastHidden[i] * golden[i];
            normA += (double)lastHidden[i] * lastHidden[i];
            normB += (double)golden[i] * golden[i];
        }
        double meanAbsDiff = sumAbsDiff / golden.Length;
        double cosineSim = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));

        // The reference model (kokoro-v1.0.onnx) is FP32; our GGUF weights are Q8_0-quantized,
        // so some divergence across 12 stacked transformer layers is expected and NOT a bug.
        // Cosine similarity is the primary correctness signal (near-1.0 means "same computation,
        // quantization noise", far from 1.0 means "wrong computation").
        Assert.True(cosineSim > 0.999, $"BERT last_hidden_state cosine similarity {cosineSim} too low vs golden ONNX output (maxAbsDiff={maxAbsDiff}, meanAbsDiff={meanAbsDiff}) -- suggests an architecture bug, not just Q8_0 quantization noise.");
    }
}
