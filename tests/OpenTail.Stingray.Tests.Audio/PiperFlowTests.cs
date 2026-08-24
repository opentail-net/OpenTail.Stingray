using System;
using System.IO;
using System.Text;
using OpenTail.Stingray.Audio.Piper;
using OpenTail.Stingray.Audio.Primitives;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Verifies VitsLengthRegulator + PiperFlow (VITS ResidualCouplingBlock, reverse/inference path)
/// against real ONNX weights and real onnxruntime golden output
/// (scratch-llamacpp-ref/piper_golden_flow.py). Feeds the exact golden noise draw and golden logw
/// (from PiperDurationPredictorTests' own golden dump) so the test isolates "is the flow math
/// correct" from "does the RNG/SDP match" -- same technique used throughout this session.
/// </summary>
public sealed class PiperFlowTests : HeavyTestBase
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
    public void PiperLengthRegulatorAndFlow_RealOnnxWeights_MatchesOnnxGoldenOutput()
    {
        string? modelPath = FindRepoFile("models/en_US-lessac-medium.onnx");
        string? sdpNoisePath = FindRepoFile("scratch-llamacpp-ref/piper_golden_sdp/dp_RandomNormalLike_output_0.npy");
        string? ceilPath = FindRepoFile("scratch-llamacpp-ref/piper_golden_flow/Ceil_output_0.npy");
        string? flowNoisePath = FindRepoFile("scratch-llamacpp-ref/piper_golden_flow/RandomNormalLike_output_0.npy");
        string? zpPath = FindRepoFile("scratch-llamacpp-ref/piper_golden_flow/Add_output_0.npy");
        string? flowOutPath = FindRepoFile("scratch-llamacpp-ref/piper_golden_flow/flow_flows.0_Concat_output_0.npy");
        if (modelPath is null || sdpNoisePath is null || ceilPath is null || flowNoisePath is null || zpPath is null || flowOutPath is null) return;

        var weights = new PiperOnnxWeights(modelPath);

        int[] tokens = [1, 0, 25, 0, 32, 0, 41, 0, 38, 0, 2];
        int tTokens = tokens.Length;
        int dim = weights.HiddenDim;

        var (_, mu, logs) = PiperTextEncoder.Forward(weights, tokens);

        // Uses the golden ONNX-computed durations directly (rather than our own SDP's logw) so
        // this test isolates "is the length regulator + flow correct" from "does our SDP match" --
        // the SDP itself is separately verified in PiperDurationPredictorTests.
        float[] goldenCeil = ReadNpyFloat32(ceilPath);
        Assert.Equal(tTokens, goldenCeil.Length);
        var durations = new int[tTokens];
        for (int i = 0; i < tTokens; i++) durations[i] = (int)goldenCeil[i];

        float noiseScale = 0.667f; // scales[0]
        float[] flowNoise = ReadNpyFloat32(flowNoisePath);

        var (zp, tFrames, _) = VitsLengthRegulator.ExpandWithDurations(mu, logs, dim, tTokens, durations, flowNoise, noiseScale);

        float[] goldenZp = ReadNpyFloat32(zpPath);
        Assert.Equal(goldenZp.Length, zp.Length);
        double zpCosine = CosineSimilarity(zp, goldenZp);
        Assert.True(zpCosine > 0.999, $"Length regulator z_p cosine similarity {zpCosine} too low vs golden ONNX output.");

        float[] flowOut = PiperFlow.Reverse(weights, zp, tFrames);

        float[] goldenFlowOut = ReadNpyFloat32(flowOutPath);
        Assert.Equal(goldenFlowOut.Length, flowOut.Length);

        foreach (float v in flowOut)
        {
            Assert.False(float.IsNaN(v), "flow output must not contain NaN");
            Assert.False(float.IsInfinity(v), "flow output must not contain Infinity");
        }

        double cosine = CosineSimilarity(flowOut, goldenFlowOut);
        Assert.True(cosine > 0.99, $"ResidualCouplingBlock reverse output cosine similarity {cosine} too low vs golden ONNX output.");
    }

    /// <summary>Isolated wall-clock benchmark for <see cref="PiperFlow.Reverse"/> alone, reusing the same golden z_p/tFrames input as the correctness test above.</summary>
    [Fact]
    public void PiperFlow_RealOnnxWeights_PerfBenchmark()
    {
        string? modelPath = FindRepoFile("models/en_US-lessac-medium.onnx");
        string? ceilPath = FindRepoFile("scratch-llamacpp-ref/piper_golden_flow/Ceil_output_0.npy");
        string? flowNoisePath = FindRepoFile("scratch-llamacpp-ref/piper_golden_flow/RandomNormalLike_output_0.npy");
        Assert.SkipUnless(modelPath != null && ceilPath != null && flowNoisePath != null,
            "Piper ONNX model or golden flow input files not found");

        var weights = new PiperOnnxWeights(modelPath!);

        int[] tokens = [1, 0, 25, 0, 32, 0, 41, 0, 38, 0, 2];
        int tTokens = tokens.Length;
        int dim = weights.HiddenDim;

        var (_, mu, logs) = PiperTextEncoder.Forward(weights, tokens);

        float[] goldenCeil = ReadNpyFloat32(ceilPath!);
        var durations = new int[tTokens];
        for (int i = 0; i < tTokens; i++) durations[i] = (int)goldenCeil[i];

        float noiseScale = 0.667f;
        float[] flowNoise = ReadNpyFloat32(flowNoisePath!);
        var (zp, tFrames, _) = VitsLengthRegulator.ExpandWithDurations(mu, logs, dim, tTokens, durations, flowNoise, noiseScale);

        PiperFlow.Reverse(weights, zp, tFrames); // warmup

        const int samples = 5;
        var msSamples = new double[samples];
        for (int s = 0; s < samples; s++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            PiperFlow.Reverse(weights, zp, tFrames);
            sw.Stop();
            msSamples[s] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(msSamples);
        double median = msSamples[samples / 2];
        double mean = 0; foreach (var v in msSamples) mean += v; mean /= samples;
        string resultLine = $"samples_ms=[{string.Join(", ", Array.ConvertAll(msSamples, v => v.ToString("F1")))}] mean_ms={mean:F2} median_ms={median:F2}";
        File.WriteAllText(Path.Combine(Path.GetTempPath(), "piper_flow_bench_result.txt"), resultLine + Environment.NewLine);
    }
}
