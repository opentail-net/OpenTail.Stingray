using System;
using System.IO;
using System.Text;
using OpenTail.Stingray.Audio.Piper;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Verifies PiperHifiGanDecoder against real ONNX weights and real onnxruntime golden output.
/// Feeds the flow's golden output directly (scratch-llamacpp-ref/piper_golden_flow/flow_flows.0_Concat_output_0.npy)
/// so this test isolates "is the decoder correct" from upstream stages, which are separately
/// verified in PiperFlowTests/PiperDurationPredictorTests/PiperTextEncoderTests. Compares against
/// the full graph's final waveform output (scratch-llamacpp-ref/piper_golden_flow/output.npy),
/// since the decoder is the last stage in the pipeline.
/// </summary>
public sealed class PiperHifiGanDecoderTests : HeavyTestBase
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
    public void PiperHifiGanDecoder_RealOnnxWeights_MatchesOnnxGoldenWaveform()
    {
        string? modelPath = FindRepoFile("models/en_US-lessac-medium.onnx");
        string? flowOutPath = FindRepoFile("scratch-llamacpp-ref/piper_golden_flow/flow_flows.0_Concat_output_0.npy");
        string? goldenWavPath = FindRepoFile("scratch-llamacpp-ref/piper_golden_flow/output.npy");
        if (modelPath is null || flowOutPath is null || goldenWavPath is null) return;

        var weights = new PiperOnnxWeights(modelPath);

        float[] flowOut = ReadNpyFloat32(flowOutPath); // [1,192,58] channel-first
        int t = flowOut.Length / weights.HiddenDim;

        float[] wav = PiperHifiGanDecoder.Forward(weights, flowOut, t);

        float[] goldenWav = ReadNpyFloat32(goldenWavPath);
        Assert.Equal(goldenWav.Length, wav.Length);

        foreach (float v in wav)
        {
            Assert.False(float.IsNaN(v), "waveform must not contain NaN");
            Assert.False(float.IsInfinity(v), "waveform must not contain Infinity");
        }

        double cosine = CosineSimilarity(wav, goldenWav);
        Assert.True(cosine > 0.99, $"HiFi-GAN decoder waveform cosine similarity {cosine} too low vs golden ONNX output.");
    }
}
