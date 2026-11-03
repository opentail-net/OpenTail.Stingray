using System.Runtime.InteropServices;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Prefill numerics investigation, open boxes "ggml vs Stingray Q8 activation quantization on identical rows" (1e-2 per projection: normal
/// int8 noise or a defect?). Stingray quantises an activation row to Q8_KS (eight scales per 256 values, one per 32-value sub-block);
/// ggml's CPU kernels use Q8_K (one scale per 256). On the rows that matter (amax/rms of 25-90 from a few outlier channels) a finer scale can
/// only help, so Stingray's activation error must not exceed ggml's. The test also pins the in-test port of ggml's
/// <c>quantize_row_q8_K_ref</c> against the real function in the vendored ggml-base.dll when that library is present.
/// </summary>
public sealed unsafe class Q8ActivationQuantizationTests
{
    private const int BlockSize = 256;

    // ggml quantize_row_q8_K_ref: one scale per 256 values, scale taken from the signed value of largest magnitude.
    private static void GgmlQ8K(float[] x, out float[] d, out sbyte[] qs)
    {
        int nb = x.Length / BlockSize;
        d = new float[nb];
        qs = new sbyte[x.Length];
        for (int b = 0; b < nb; b++)
        {
            float max = 0f, amax = 0f;
            for (int j = 0; j < BlockSize; j++)
            {
                float ax = MathF.Abs(x[b * BlockSize + j]);
                if (ax > amax) { amax = ax; max = x[b * BlockSize + j]; }
            }
            if (amax == 0f) continue;
            float iscale = -127.0f / max;
            for (int j = 0; j < BlockSize; j++)
            {
                int v = (int)MathF.Round(iscale * x[b * BlockSize + j], MidpointRounding.ToEven);
                qs[b * BlockSize + j] = (sbyte)Math.Min(127, v);
            }
            d[b] = 1.0f / iscale;
        }
    }

    private static float[] GgmlDequant(float[] d, sbyte[] qs)
    {
        var y = new float[qs.Length];
        for (int i = 0; i < y.Length; i++) y[i] = d[i / BlockSize] * qs[i];
        return y;
    }

    private static float[] StingrayDequant(float[] x)
    {
        int nb = x.Length / BlockSize;
        var scratch = new byte[nb * 32 + nb * 256 + nb * 32];
        fixed (float* px = x)
        fixed (byte* ps = scratch)
        {
            SimdKernels.QuantizeRowToQ8KS(px, x.Length, ps);
            float* d = (float*)ps;
            sbyte* qs = (sbyte*)(ps + nb * 32);
            var y = new float[x.Length];
            for (int i = 0; i < y.Length; i++) y[i] = d[i / 32] * qs[i];
            return y;
        }
    }

    private static double RelRms(float[] reference, float[] approx)
    {
        double err = 0, norm = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            double e = reference[i] - approx[i];
            err += e * e; norm += (double)reference[i] * reference[i];
        }
        return Math.Sqrt(err / norm);
    }

    // Gaussian row with a few large outlier channels, scaled so amax/rms is about the requested ratio.
    private static float[] OutlierRow(Random rng, int n, double amaxOverRms, int outliers)
    {
        var x = new float[n];
        for (int i = 0; i < n; i++)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
            x[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
        for (int k = 0; k < outliers; k++) x[rng.Next(n)] = (float)(amaxOverRms * (rng.Next(2) == 0 ? 1 : -1) * (0.7 + 0.3 * rng.NextDouble()));
        return x;
    }

    [Theory]
    [InlineData(25.0)]
    [InlineData(50.0)]
    [InlineData(90.0)]
    public void Q8KS_ActivationError_IsNoWorseThanGgmlQ8K_OnOutlierRows(double amaxOverRms)
    {
        var rng = new Random(11);
        double sumKs = 0, sumK = 0;
        const int Rows = 64;
        for (int r = 0; r < Rows; r++)
        {
            var x = OutlierRow(rng, 2048, amaxOverRms, outliers: 3);
            GgmlQ8K(x, out var d, out var qs);
            double errK = RelRms(x, GgmlDequant(d, qs));
            double errKs = RelRms(x, StingrayDequant(x));
            Assert.True(errKs <= errK * 1.0001, $"row {r}: Q8_KS {errKs:E3} worse than Q8_K {errK:E3}");
            sumKs += errKs; sumK += errK;
        }
        // Not vacuous: the outliers make ggml's single scale visibly coarse, and the finer scales gain a clear margin.
        Assert.True(sumK / Rows > 1e-3, $"mean Q8_K error {sumK / Rows:E3} is too small to say anything");
        Assert.True(sumKs / Rows < 0.9 * sumK / Rows, $"Q8_KS mean {sumKs / Rows:E3} is not clearly better than Q8_K {sumK / Rows:E3}");
    }

    [Fact]
    public void InTestGgmlPort_MatchesTheVendoredQuantizeRowQ8KRef()
    {
        string? dll = FindGgmlBase();
        Assert.SkipUnless(dll is not null, "vendored ggml-base.dll not found (set STINGRAY_LLAMACPP_DIR or run on a machine with tools/llama.cpp)");
        Assert.True(NativeLibrary.TryLoad(dll!, out nint lib), "ggml-base.dll did not load");
        Assert.True(NativeLibrary.TryGetExport(lib, "quantize_row_q8_K_ref", out nint fn), "quantize_row_q8_K_ref is not exported");
        var quantize = (delegate* unmanaged[Cdecl]<float*, byte*, long, void>)fn;

        var rng = new Random(5);
        for (int trial = 0; trial < 50; trial++)
        {
            var x = OutlierRow(rng, 1024, 20 + trial, outliers: 2);
            GgmlQ8K(x, out var d, out var qs);
            const int BlockBytes = 4 + 256 + 32;   // block_q8_K: float d, int8 qs[256], int16 bsums[16]
            var native = new byte[x.Length / BlockSize * BlockBytes];
            fixed (float* px = x)
            fixed (byte* pn = native)
                quantize(px, pn, x.Length);
            for (int b = 0; b < d.Length; b++)
            {
                int off = b * BlockBytes;
                Assert.Equal(d[b], BitConverter.ToSingle(native, off));
                for (int j = 0; j < BlockSize; j++) Assert.Equal(qs[b * BlockSize + j], (sbyte)native[off + 4 + j]);
            }
        }
    }

    private static string? FindGgmlBase()
    {
        string? env = Environment.GetEnvironmentVariable("STINGRAY_LLAMACPP_DIR");
        if (env is not null && File.Exists(Path.Combine(env, "ggml-base.dll"))) return Path.Combine(env, "ggml-base.dll");
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "tools", "llama.cpp", "ggml-base.dll");
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }
}
