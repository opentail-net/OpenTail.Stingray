using System.Runtime.InteropServices;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Tests.Diffusion;

// Guards PackedSgemmF32 (panel-packed 6x16 FP32 GEMM) against a double-precision reference,
// including ragged m (not a multiple of 6 / Mc), ragged n (not a multiple of 16) and k spanning
// several Kc blocks with a partial last block.
public sealed unsafe class PackedSgemmF32Tests
{
    [Fact]
    public void Gemm_MatchesDoubleReference()
    {
        if (!PackedSgemmF32.IsSupported) return;
        var rng = new Random(5);
        foreach (var (m, k, n, useBias) in new[] {
            (1, 64, 16, true), (5, 7, 5, false), (6, 256, 32, true), (13, 300, 50, true),
            (97, 513, 257, false), (200, 1536, 64, true), (7, 9, 1, true) })
        {
            var x = new float[m * k]; var w = new float[n * k]; var b = new float[n];
            for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
            for (int i = 0; i < n; i++) b[i] = (float)(rng.NextDouble() - 0.5);
            var got = new float[m * n];

            fixed (float* px = x, pw = w, pb = b, po = got)
            {
                float* packed = PackedSgemmF32.PackWeights(pw, n, k);
                try { PackedSgemmF32.Gemm(po, px, packed, useBias ? pb : null, m, n, k); }
                finally { NativeMemory.AlignedFree(packed); }
            }

            float maxDiff = 0, maxRef = 0;
            for (int r = 0; r < m; r++)
                for (int o = 0; o < n; o++)
                {
                    double s = useBias ? b[o] : 0;
                    for (int i = 0; i < k; i++) s += (double)x[r * k + i] * w[o * k + i];
                    maxDiff = MathF.Max(maxDiff, MathF.Abs((float)s - got[r * n + o]));
                    maxRef = MathF.Max(maxRef, MathF.Abs((float)s));
                }
            Assert.True(maxDiff <= 1e-4f * MathF.Max(1f, maxRef), $"m={m} k={k} n={n}: maxDiff={maxDiff:E2}");
        }
    }
}
