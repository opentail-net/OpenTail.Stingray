using System;
using System.IO;
using System.Text;
using OpenTail.Stingray.Audio.Piper;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Verifies PiperDurationPredictor (VITS StochasticDurationPredictor, reverse/inference path)
/// against real ONNX weights and real onnxruntime golden output
/// (scratch-llamacpp-ref/piper_golden_sdp.py). Since the SDP is noise-driven, the golden dump
/// also captures the exact raw N(0,1) draw the ONNX graph used
/// (/dp/RandomNormalLike_output_0), so this test feeds that exact noise in rather than a fresh
/// random draw -- isolating "is the flow math correct" from "does the RNG match" (same technique
/// used for the NSF/noise-driven vocoder stages in Kokoro/Chatterbox).
/// </summary>
public sealed class PiperDurationPredictorTests : HeavyTestBase
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
    public void PiperDurationPredictor_RealOnnxWeights_MatchesOnnxGoldenLogw()
    {
        string? modelPath = FindRepoFile("models/en_US-lessac-medium.onnx");
        string? noisePath = FindRepoFile("scratch-llamacpp-ref/piper_golden_sdp/dp_RandomNormalLike_output_0.npy");
        string? logwPath = FindRepoFile("scratch-llamacpp-ref/piper_golden_sdp/dp_Split_output_0.npy");
        if (modelPath is null || noisePath is null || logwPath is null) return;

        var weights = new PiperOnnxWeights(modelPath);

        // Matches piper_golden_sdp.py's fixed input_ids exactly.
        int[] tokens = [1, 0, 25, 0, 32, 0, 41, 0, 38, 0, 2];
        int t = tokens.Length;

        var (encoderHidden, _, _) = PiperTextEncoder.Forward(weights, tokens);

        float[] noise = ReadNpyFloat32(noisePath); // [1,2,T] raw N(0,1), channel-first
        Assert.Equal(2 * t, noise.Length);
        float noiseScaleW = 0.8f; // matches piper_golden_sdp.py's scales[2]

        float[] logw = PiperDurationPredictor.Predict(weights, encoderHidden, t, noise, noiseScaleW);

        float[] goldenLogw = ReadNpyFloat32(logwPath);
        Assert.Equal(t, goldenLogw.Length);
        Assert.Equal(t, logw.Length);

        foreach (float v in logw)
        {
            Assert.False(float.IsNaN(v), "logw must not contain NaN");
            Assert.False(float.IsInfinity(v), "logw must not contain Infinity");
        }

        double cosine = CosineSimilarity(logw, goldenLogw);
        Assert.True(cosine > 0.99, $"StochasticDurationPredictor logw cosine similarity {cosine} too low vs golden ONNX output. (logw={string.Join(",", logw)}, golden={string.Join(",", goldenLogw)})");
    }
}
