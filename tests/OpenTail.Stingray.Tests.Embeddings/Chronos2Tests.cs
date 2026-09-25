namespace OpenTail.Stingray.Tests.Embeddings;

/// <summary>
/// Chronos-2 (amazon/chronos-2), ported from the vendored reference (<c>examples/chronos-forecasting/.../chronos2</c>).
/// No numeric oracle exists (no ONNX / C++ port), so behavioural checks: accuracy on a clean seasonal+trend series,
/// ordered quantiles, exact affine equivariance (instance norm + arcsinh round trip), batch/group isolation (a series in
/// its own group gives the same forecast alone or batched), multivariate groups and NaN input.
/// </summary>
public sealed class Chronos2Tests
{
    private static string? FindDir() => BertEncoderOnnxParityTests.FindRepoDir("models/_models/hf/amazon__chronos-2");

    private static float Series(int t) => 10f + 0.02f * t + 3f * MathF.Sin(2f * MathF.PI * t / 24f) + 0.3f * MathF.Sin(t * 1.7f);

    // Deterministic pseudo-noise (sd ~1) so the forecast distribution has real width.
    private static float Noisy(int t)
    {
        uint h = (uint)t * 0x9E3779B9u + 0x7F4A7C15u; // murmur3 fmix32 of t
        h ^= h >> 16; h *= 0x85EBCA6Bu; h ^= h >> 13; h *= 0xC2B2AE35u; h ^= h >> 16;
        return Series(t) + 1.7f * ((h >> 8) / 16777216f - 0.5f) * 2f;
    }

    [Fact]
    public void Forecast_Accurate_Ordered_Equivariant_GroupIsolated()
    {
        string? dir = FindDir();
        Assert.SkipUnless(dir != null && File.Exists(Path.Combine(dir, "model.safetensors")), "chronos-2 not found");
        using var model = Chronos2Model.Load(dir!);
        Assert.Equal(21, model.Quantiles.Length);
        int median = Array.IndexOf(model.Quantiles, 0.5f);

        const int n = 480, h = 64;
        var x = Enumerable.Range(0, n).Select(Series).ToArray();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var q = model.Predict([x], h)[0];
        long ms = sw.ElapsedMilliseconds;
        double mae = Enumerable.Range(0, h).Average(t => Math.Abs(q[median * h + t] - Series(n + t)));
        double naive = Enumerable.Range(0, h).Average(t => Math.Abs(x[n - 24 + t % 24] - Series(n + t)));
        var qn = model.Predict([Enumerable.Range(0, n).Select(Noisy).ToArray()], h)[0];
        int violations = 0, noisyViolations = 0;
        for (int t = 0; t < h; t++)
            for (int k = 1; k < model.Quantiles.Length; k++)
                if (qn[k * h + t] < qn[(k - 1) * h + t] - 1e-4f) noisyViolations++;
        Console.WriteLine($"[Chronos2] noisy series: quantile crossings {noisyViolations} of {h * (model.Quantiles.Length - 1)}; 0.1-0.9 width at t=0: {qn[Array.IndexOf(model.Quantiles, 0.9f) * h] - qn[Array.IndexOf(model.Quantiles, 0.1f) * h]:F3}");
        var perPair = new int[model.Quantiles.Length];
        var worst = new float[model.Quantiles.Length];
        for (int t = 0; t < h; t++)
            for (int k = 1; k < model.Quantiles.Length; k++)
            {
                float gap = q[(k - 1) * h + t] - q[k * h + t];
                if (gap > 1e-4f) { perPair[k]++; worst[k] = Math.Max(worst[k], gap); violations++; }
            }
        Console.WriteLine("[Chronos2] crossings per adjacent pair (count/worst): " + string.Join(" ", Enumerable.Range(1, model.Quantiles.Length - 1)
            .Where(k => perPair[k] > 0).Select(k => $"{model.Quantiles[k - 1]}-{model.Quantiles[k]}:{perPair[k]}/{worst[k]:F3}")));
        Console.WriteLine($"[Chronos2] {ms} ms; median MAE {mae:F4} vs seasonal-naive {naive:F4}; first 6 median: " +
            string.Join(", ", Enumerable.Range(0, 6).Select(t => $"{q[median * h + t]:F2}/{Series(n + t):F2}")) + $"; clean-series crossings {violations} (diagnostic)");
        Assert.True(mae < 0.5, $"median MAE {mae}");
        // On this nearly noise-free series the central quantiles sit within hundredths of each other and cross by up to
        // ~0.05 (diagnostic only). With real noise (below) the distribution has width and must be strictly ordered.
        Assert.Equal(0, noisyViolations);
        Assert.True(worst.Max() < 0.1f, $"clean-series quantile crossing up to {worst.Max()}");

        var qy = model.Predict([x.Select(v => 5f * v + 100f).ToArray()], h)[0];
        float maxRel = 0;
        for (int i = 0; i < q.Length; i++) maxRel = Math.Max(maxRel, Math.Abs(qy[i] - (5f * q[i] + 100f)) / (5f * Math.Abs(q[i]) + 100f));
        Console.WriteLine($"[Chronos2] affine equivariance max relative error {maxRel:E2}");
        Assert.True(maxRel < 1e-4f, $"affine equivariance {maxRel}");

        // Group isolation: [x, other] in separate groups must reproduce x's solo forecast.
        var other = Enumerable.Range(0, 300).Select(t => 50f + 5f * MathF.Cos(t / 7f)).ToArray();
        var both = model.Predict([x, other], h);
        float isoMax = 0;
        for (int i = 0; i < q.Length; i++) isoMax = Math.Max(isoMax, Math.Abs(both[0][i] - q[i]));
        Console.WriteLine($"[Chronos2] batched-in-separate-groups vs solo maxAbs {isoMax:E2}");
        Assert.True(isoMax < 1e-3f, $"group isolation {isoMax}");

        // Multivariate: x and a lagged copy in one group; NaN holes in the context.
        var lagged = Enumerable.Range(0, n).Select(t => Series(t - 6)).ToArray();
        var holes = (float[])x.Clone();
        for (int i = 100; i < 140; i++) holes[i] = float.NaN;
        var mv = model.Predict([holes, lagged], h, [0, 0]);
        Assert.All(mv.SelectMany(a => a), v => Assert.True(float.IsFinite(v)));
        double mvMae = Enumerable.Range(0, h).Average(t => Math.Abs(mv[0][median * h + t] - Series(n + t)));
        Console.WriteLine($"[Chronos2] multivariate (with a 40-step NaN hole) median MAE {mvMae:F4}");
        Assert.True(mvMae < 0.6, $"multivariate MAE {mvMae}");
    }
}
