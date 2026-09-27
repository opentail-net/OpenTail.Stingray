using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OpenTail.Stingray.Tests.Embeddings;

/// <summary>
/// Chronos-2 against an independent oracle: the OpenSTEF fp32 ONNX export of <c>amazon/chronos-2</c>
/// (<c>OpenSTEF/chronos-2-onnx</c>, <c>chronos-2.onnx</c>, dynamic shapes; its README reports max deviation
/// 1.86e-05 vs torch), run through ONNX Runtime. Same univariate series and horizon on both sides; every quantile is
/// compared.
/// </summary>
public sealed class Chronos2OnnxParityTests
{
    private static float Noisy(int t)
    {
        float s = 10f + 0.02f * t + 3f * MathF.Sin(2f * MathF.PI * t / 24f) + 0.3f * MathF.Sin(t * 1.7f);
        uint h = (uint)t * 0x9E3779B9u + 0x7F4A7C15u;
        h ^= h >> 16; h *= 0x85EBCA6Bu; h ^= h >> 13; h *= 0xC2B2AE35u; h ^= h >> 16;
        return s + 1.7f * ((h >> 8) / 16777216f - 0.5f) * 2f;
    }

    [Fact]
    public void Chronos2_MatchesOnnxExport()
    {
        string? dir = BertEncoderOnnxParityTests.FindRepoDir("models/_models/hf/amazon__chronos-2");
        string? onnxDir = BertEncoderOnnxParityTests.FindRepoDir("models/_models/hf/OpenSTEF__chronos-2-onnx");
        Assert.SkipUnless(dir != null && onnxDir != null && File.Exists(Path.Combine(onnxDir, "chronos-2.onnx")), "chronos-2 or its ONNX export not found");

        const int n = 480, h = 672; // the export fixes the horizon at 42 output patches x 16
        var x = Enumerable.Range(0, n).Select(Noisy).ToArray();

        using var model = Chronos2Model.Load(dir!);
        var ours = model.Predict([x], h)[0];                       // [21 * h]

        using var session = new InferenceSession(Path.Combine(onnxDir!, "chronos-2.onnx"));
        foreach (var kv in session.InputMetadata) Console.WriteLine($"[chronos2-onnx] in  {kv.Key} {kv.Value.ElementType.Name} [{string.Join(",", kv.Value.Dimensions)}]");
        foreach (var kv in session.OutputMetadata) Console.WriteLine($"[chronos2-onnx] out {kv.Key} [{string.Join(",", kv.Value.Dimensions)}]");
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("context", new DenseTensor<float>(x, [1, n])),
            NamedOnnxValue.CreateFromTensor("group_ids", new DenseTensor<long>(new long[] { 0 }, [1])),
            NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<float>(Enumerable.Repeat(1f, n).ToArray(), [1, n])),
            NamedOnnxValue.CreateFromTensor("future_covariates", new DenseTensor<float>(new float[h], [1, h])),
            NamedOnnxValue.CreateFromTensor("future_covariates_mask", new DenseTensor<float>(new float[h], [1, h])),
        };
        using var results = session.Run(inputs);
        var outT = results.First().AsTensor<float>();
        Console.WriteLine($"[chronos2-onnx] output dims [{string.Join(",", outT.Dimensions.ToArray())}]");
        int nq = outT.Dimensions[1], hOut = outT.Dimensions[2];
        Assert.Equal(21, nq);

        double maxAbs = 0, maxRef = 0;
        for (int k = 0; k < nq; k++)
            for (int t = 0; t < h; t++)
            {
                float r = outT[0, k, t];
                maxAbs = Math.Max(maxAbs, Math.Abs(ours[k * h + t] - r));
                maxRef = Math.Max(maxRef, Math.Abs(r));
            }
        Console.WriteLine($"[chronos2-onnx] horizon {hOut}; ours vs ONNX over 21 x {h}: maxAbs {maxAbs:E3} (max |ref| {maxRef:F2}); median t0 ours {ours[10 * h]:F4} onnx {outT[0, 10, 0]:F4}");
        // Measured 2026-09-27: maxAbs 2.54e-4 on values up to 45.7 (~5.5e-6 relative).
        Assert.True(maxAbs < 1e-4 * maxRef, $"maxAbs {maxAbs}");

        // Multivariate: the target and a 6-step-lagged series in ONE group (group attention across series).
        var lagged = Enumerable.Range(0, n).Select(t => Noisy(t - 6)).ToArray();
        var oursMv = model.Predict([x, lagged], h, groupIds: [0, 0]);
        var mvInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("context", new DenseTensor<float>(x.Concat(lagged).ToArray(), [2, n])),
            NamedOnnxValue.CreateFromTensor("group_ids", new DenseTensor<long>(new long[] { 0, 0 }, [2])),
            NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<float>(Enumerable.Repeat(1f, 2 * n).ToArray(), [2, n])),
            NamedOnnxValue.CreateFromTensor("future_covariates", new DenseTensor<float>(new float[2 * h], [2, h])),
            NamedOnnxValue.CreateFromTensor("future_covariates_mask", new DenseTensor<float>(new float[2 * h], [2, h])),
        };
        using var mvResults = session.Run(mvInputs);
        var mvT = mvResults.First().AsTensor<float>();
        double mvMax = 0, mvRef = 0, soloVsGroup = 0;
        for (int row = 0; row < 2; row++)
            for (int k = 0; k < nq; k++)
                for (int t = 0; t < h; t++)
                {
                    float r = mvT[row, k, t];
                    mvMax = Math.Max(mvMax, Math.Abs(oursMv[row][k * h + t] - r));
                    mvRef = Math.Max(mvRef, Math.Abs(r));
                    if (row == 0) soloVsGroup = Math.Max(soloVsGroup, Math.Abs(r - outT[0, k, t]));
                }
        Console.WriteLine($"[chronos2-onnx] multivariate group of 2: maxAbs {mvMax:E3} (max |ref| {mvRef:F2}); group changes the target forecast by up to {soloVsGroup:F3} in ONNX");
        Assert.True(mvMax < 1e-4 * mvRef, $"multivariate maxAbs {mvMax}");
        Assert.True(soloVsGroup > 1e-3, "the group input should change the target forecast (else this case tests nothing)");
    }

    /// <summary>
    /// Chronos-Bolt encoder against the light-curve ONNX export of <c>amazon/chronos-bolt-small</c>
    /// (<c>light-curve/chronos-bolt-small</c>, encoder only: instance norm, patching, input residual block, T5 encoder).
    /// Compares the per-patch hidden states (<c>sequence</c>) and their mean (<c>mean</c>). The decoder and quantile
    /// head are not in that export; the decoder is the shared, ONNX-verified <c>T5Model</c>.
    /// </summary>
    [Fact]
    public void ChronosBolt_EncoderMatchesOnnxExport()
    {
        string? dir = BertEncoderOnnxParityTests.FindRepoDir("models/_models/hf/amazon__chronos-bolt-small");
        string? onnxDir = BertEncoderOnnxParityTests.FindRepoDir("models/_models/hf/light-curve__chronos-bolt-small");
        Assert.SkipUnless(dir != null && onnxDir != null && File.Exists(Path.Combine(onnxDir, "chronos-bolt-small.onnx")), "chronos-bolt-small or its ONNX export not found");

        const int n = 480;                               // a multiple of 16: the export expects NaN-padded patches
        var x = Enumerable.Range(0, n).Select(Noisy).ToArray();
        using var model = ChronosBoltModel.Load(dir!);
        var ours = model.EncodeHidden(x, out int patches);

        using var session = new InferenceSession(Path.Combine(onnxDir!, "chronos-bolt-small.onnx"));
        using var results = session.Run([NamedOnnxValue.CreateFromTensor("context", new DenseTensor<float>(x, [1, n]))]);
        var seq = results.First(r => r.Name == "sequence").AsTensor<float>();
        var mean = results.First(r => r.Name == "mean").AsTensor<float>();
        int d = seq.Dimensions[2];
        Assert.Equal(patches, seq.Dimensions[1]);

        double maxAbs = 0, maxRef = 0, meanMax = 0;
        for (int p = 0; p < patches; p++)
            for (int c = 0; c < d; c++)
            {
                maxAbs = Math.Max(maxAbs, Math.Abs(ours[p * d + c] - seq[0, p, c]));
                maxRef = Math.Max(maxRef, Math.Abs(seq[0, p, c]));
            }
        for (int c = 0; c < d; c++)
        {
            double m = 0;
            for (int p = 0; p < patches; p++) m += ours[p * d + c];
            meanMax = Math.Max(meanMax, Math.Abs(m / patches - mean[0, c]));
        }
        Console.WriteLine($"[bolt-onnx] {patches} patches x {d}: sequence maxAbs {maxAbs:E3} (max |ref| {maxRef:F3}); mean-pool maxAbs {meanMax:E3}");
        Assert.True(maxAbs < 1e-3 * maxRef, $"sequence maxAbs {maxAbs}");
        Assert.True(meanMax < 1e-3 * maxRef, $"mean maxAbs {meanMax}");
    }
}
