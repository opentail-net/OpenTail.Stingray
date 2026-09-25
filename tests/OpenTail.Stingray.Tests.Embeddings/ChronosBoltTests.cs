namespace OpenTail.Stingray.Tests.Embeddings;

/// <summary>
/// Chronos-Bolt (amazon/chronos-bolt-small). The T5 backbone is ONNX-verified separately (<see cref="T5Tests"/>); the
/// Chronos-specific parts are ported from the vendored reference (<c>examples/chronos-forecasting</c>). No numeric
/// oracle exists yet (no ONNX export or C++ port of Chronos-Bolt was found), so these are behavioural checks: accuracy on
/// a clean seasonal+trend series, ordered quantiles, exact affine equivariance (instance norm round trip), NaN input and
/// the long-horizon path.
/// </summary>
public sealed class ChronosBoltTests
{
    private static string? FindDir() => BertEncoderOnnxParityTests.FindRepoDir("models/_models/hf/amazon__chronos-bolt-small");

    private static float Series(int t) => 10f + 0.02f * t + 3f * MathF.Sin(2f * MathF.PI * t / 24f) + 0.3f * MathF.Sin(t * 1.7f);

    [Fact]
    public void Forecast_SeasonalTrend_IsAccurate_And_QuantilesOrdered()
    {
        string? dir = FindDir();
        Assert.SkipUnless(dir != null, "chronos-bolt-small not found");
        using var model = ChronosBoltModel.Load(dir!);
        Assert.Equal(64, model.PredictionLength);
        Assert.Equal(9, model.Quantiles.Length);

        const int n = 480, h = 64;
        var context = Enumerable.Range(0, n).Select(Series).ToArray();
        var q = model.Forward(context);
        int median = Array.IndexOf(model.Quantiles, 0.5f);
        double mae = 0, naive = 0;
        for (int t = 0; t < h; t++)
        {
            float truth = Series(n + t);
            mae += Math.Abs(q[median * h + t] - truth);
            naive += Math.Abs(context[n - 24 + t % 24] - truth); // seasonal-naive baseline
        }
        mae /= h;
        naive /= h;
        int violations = 0;
        for (int t = 0; t < h; t++)
            for (int k = 1; k < model.Quantiles.Length; k++)
                if (q[k * h + t] < q[(k - 1) * h + t] - 1e-4f) violations++;
        Console.WriteLine($"[ChronosBolt] median MAE {mae:F4} vs seasonal-naive {naive:F4} (amplitude 3); first 8 median: " +
            string.Join(", ", Enumerable.Range(0, 8).Select(t => $"{q[median * h + t]:F2}/{Series(n + t):F2}")) + $"; quantile order violations {violations}");
        Assert.True(mae < 0.5, $"median MAE {mae}");
        Assert.Equal(0, violations);
    }

    [Fact]
    public void Forecast_IsAffineEquivariant_HandlesNaN_AndLongHorizon()
    {
        string? dir = FindDir();
        Assert.SkipUnless(dir != null, "chronos-bolt-small not found");
        using var model = ChronosBoltModel.Load(dir!);
        var x = Enumerable.Range(0, 300).Select(Series).ToArray();
        var fx = model.Forward(x);
        var fy = model.Forward(x.Select(v => 5f * v + 100f).ToArray());
        float maxRel = 0;
        for (int i = 0; i < fx.Length; i++) maxRel = Math.Max(maxRel, Math.Abs(fy[i] - (5f * fx[i] + 100f)) / (5f * Math.Abs(fx[i]) + 100f));
        Console.WriteLine($"[ChronosBolt] affine equivariance max relative error {maxRel:E2}");
        Assert.True(maxRel < 1e-4f, $"affine equivariance {maxRel}");

        var holes = (float[])x.Clone();
        for (int i = 50; i < 70; i++) holes[i] = float.NaN;
        holes[0] = float.NaN;
        var fh = model.Forward(holes);
        Assert.All(fh, v => Assert.True(float.IsFinite(v)));

        var longQ = model.Predict(x, 150);
        Assert.Equal(9 * 150, longQ.Length);
        Assert.All(longQ, v => Assert.True(float.IsFinite(v)));
        for (int t = 0; t < 64; t++) Assert.Equal(fx[4 * 64 + t], longQ[4 * 150 + t]);
        double mae = Enumerable.Range(64, 86).Average(t => Math.Abs(longQ[4 * 150 + t] - Series(300 + t)));
        Console.WriteLine($"[ChronosBolt] long horizon (150): median MAE over steps 64-149 = {mae:F4}");
        Assert.True(mae < 1.0, $"long-horizon MAE {mae}");
    }
}

