using OpenTail.Stingray.Diffusion;

namespace OpenTail.Stingray.Tests.Diffusion;

// Guards DiffusionOps.Linear's GEMM-backed batched path against a plain scalar reference.
public sealed class DiffusionOpsLinearTests
{
    [Fact]
    public void Linear_MatchesScalarReference()
    {
        var rng = new Random(11);
        foreach (var (n, inD, outD, useBias) in new[] {
            (1, 64, 48, true), (2, 7, 5, true), (3, 33, 17, false), (77, 768, 96, true), (256, 128, 1000, false), (9, 513, 257, true) })
        {
            var x = new float[n * inD]; var w = new float[outD * inD];
            for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
            float[]? b = null;
            if (useBias) { b = new float[outD]; for (int i = 0; i < outD; i++) b[i] = (float)(rng.NextDouble() - 0.5); }

            var got = DiffusionOps.Linear(x, w, b, n, inD, outD);

            float maxDiff = 0, maxRef = 0;
            for (int r = 0; r < n; r++)
                for (int o = 0; o < outD; o++)
                {
                    double s = b?[o] ?? 0;
                    for (int i = 0; i < inD; i++) s += (double)x[r * inD + i] * w[o * inD + i];
                    maxDiff = MathF.Max(maxDiff, MathF.Abs((float)s - got[r * outD + o]));
                    maxRef = MathF.Max(maxRef, MathF.Abs((float)s));
                }
            Assert.True(maxDiff <= 1e-4f * MathF.Max(1f, maxRef), $"n={n} {inD}->{outD}: maxDiff={maxDiff:E2}");
        }
    }
}
