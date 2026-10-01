using System.Runtime.InteropServices;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Size-sweep differential tests for the core elementwise / reduction kernels. SIMD loops with scalar tails are where
/// off-by-one and over-read bugs live, and the usual tests use "nice" sizes. Every size 0..130 plus awkward larger ones,
/// at four alignments, against double-precision references. Each buffer is followed by NaN (so any read past the end
/// contaminates the result) and bracketed by canaries (so any write outside [0,n) is caught).
/// </summary>
public sealed unsafe class SimdKernelsSizeSweepTests
{
    private const int Pad = 32;
    private const uint NanPattern = 0x7FC01234;       // NaN tail: any READ past the end poisons a reduction
    private const uint FinitePattern = 0x449A5000;    // 1234.5f tail: any WRITE past the end changes it (NaN+NaN keeps its bits, so a NaN tail cannot show that)
    private const uint PrePattern = 0x3F9E3779;

    private static readonly int[] Sizes = Enumerable.Range(0, 131)
        .Concat([255, 256, 257, 383, 511, 512, 513, 767, 1000, 1023, 1024, 1025, 2047, 2049, 4095, 4096, 4097])
        .ToArray();

    /// <summary>A float buffer of exactly n elements, 4 possible alignments, NaN after the end and canaries before it.</summary>
    private sealed class Buf : IDisposable
    {
        private readonly float* _raw;
        public readonly float* P;
        public readonly int N;
        private readonly uint _post;
        public Buf(int n, int shift, Random rnd, float lo = -2f, float hi = 2f, bool nanTail = false)
        {
            N = n;
            _post = nanTail ? NanPattern : FinitePattern;
            long total = Pad + shift + n + Pad;
            _raw = (float*)NativeMemory.AlignedAlloc((nuint)(total * sizeof(float)), 64);
            for (int i = 0; i < Pad + shift; i++) ((uint*)_raw)[i] = PrePattern;
            P = _raw + Pad + shift;
            for (int i = 0; i < n; i++) P[i] = lo + (float)rnd.NextDouble() * (hi - lo);
            for (int i = 0; i < Pad; i++) ((uint*)(P + n))[i] = _post;
        }
        public float[] ToArray() { var a = new float[N]; for (int i = 0; i < N; i++) a[i] = P[i]; return a; }
        public void AssertCanaries(string what)
        {
            for (int i = 0; i < Pad; i++)
            {
                Assert.True(((uint*)(P + N))[i] == _post, $"{what}: wrote past the end (n={N}, +{i})");
                Assert.True(((uint*)(P - 1 - i))[0] == PrePattern, $"{what}: wrote before the start (n={N}, -{i + 1})");
            }
        }
        public void Dispose() => NativeMemory.AlignedFree(_raw);
    }

    private static IEnumerable<(int n, int shift)> Cases()
    {
        foreach (int n in Sizes)
            for (int shift = 0; shift < 4; shift++)
                yield return (n, shift);
    }

    private static void Close(double expected, double actual, double tol, string what)
    {
        Assert.True(double.IsFinite(actual) && Math.Abs(expected - actual) <= tol,
            $"{what}: expected {expected:R}, got {actual:R} (tol {tol:E2})");
    }

    [Fact]
    public void DotF32_AllSizes()
    {
        var rnd = new Random(1);
        foreach (var (n, shift) in Cases())
        {
            using var a = new Buf(n, shift, rnd, nanTail: true); using var b = new Buf(n, (shift + 1) & 3, rnd, nanTail: true);
            double sum = 0, mag = 0;
            for (int i = 0; i < n; i++) { sum += (double)a.P[i] * b.P[i]; mag += Math.Abs((double)a.P[i] * b.P[i]); }
            float got = SimdKernels.DotF32(a.P, b.P, n);
            Close(sum, got, 1e-5 * mag + 1e-6, $"DotF32 n={n} shift={shift}");
            a.AssertCanaries("DotF32 a"); b.AssertCanaries("DotF32 b");
        }
    }

    [Fact]
    public void DotF32_2In_And_4In_AllSizes_MatchDotF32()
    {
        var rnd = new Random(2);
        foreach (var (n, shift) in Cases())
        {
            using var a0 = new Buf(n, shift, rnd, nanTail: true); using var a1 = new Buf(n, shift, rnd, nanTail: true);
            using var a2 = new Buf(n, shift, rnd, nanTail: true); using var a3 = new Buf(n, shift, rnd, nanTail: true);
            using var b = new Buf(n, (shift + 2) & 3, rnd, nanTail: true);
            double mag = 0; for (int i = 0; i < n; i++) mag += Math.Abs((double)a0.P[i] * b.P[i]);
            double tol = 1e-5 * (mag + 1) + 1e-6;
            SimdKernels.DotF32_2In(a0.P, a1.P, b.P, n, out float s0, out float s1);
            Close(SimdKernels.DotF32(a0.P, b.P, n), s0, tol, $"DotF32_2In[0] n={n}");
            Close(SimdKernels.DotF32(a1.P, b.P, n), s1, tol, $"DotF32_2In[1] n={n}");
            SimdKernels.DotF32_4In(a0.P, a1.P, a2.P, a3.P, b.P, n, out float t0, out float t1, out float t2, out float t3);
            Close(SimdKernels.DotF32(a0.P, b.P, n), t0, tol, $"DotF32_4In[0] n={n}");
            Close(SimdKernels.DotF32(a3.P, b.P, n), t3, tol, $"DotF32_4In[3] n={n}");
        }
    }

    [Fact]
    public void DotF32Bf16_AllSizes()
    {
        var rnd = new Random(3);
        foreach (var (n, shift) in Cases())
        {
            using var a = new Buf(n, shift, rnd, nanTail: true);
            var raw = (ushort*)NativeMemory.AlignedAlloc((nuint)((n + 2 * Pad) * sizeof(ushort)), 64);
            try
            {
                for (int i = 0; i < n + 2 * Pad; i++) raw[i] = i < n ? (ushort)(BitConverter.SingleToUInt32Bits((float)(rnd.NextDouble() * 4 - 2)) >> 16) : (ushort)0x7FC1;
                double sum = 0, mag = 0;
                for (int i = 0; i < n; i++)
                {
                    float bf = BitConverter.UInt32BitsToSingle((uint)raw[i] << 16);
                    sum += (double)a.P[i] * bf; mag += Math.Abs((double)a.P[i] * bf);
                }
                Close(sum, SimdKernels.DotF32Bf16(a.P, raw, n), 1e-5 * mag + 1e-6, $"DotF32Bf16 n={n} shift={shift}");
            }
            finally { NativeMemory.AlignedFree(raw); }
        }
    }

    [Fact]
    public void AddScaleWeightedAdd_AllSizes()
    {
        var rnd = new Random(4);
        foreach (var (n, shift) in Cases())
        {
            using var d = new Buf(n, shift, rnd); using var s = new Buf(n, (shift + 1) & 3, rnd);
            var d0 = d.ToArray(); var s0 = s.ToArray();
            SimdKernels.AddInPlace(d.P, s.P, n);
            for (int i = 0; i < n; i++) Close(d0[i] + s0[i], d.P[i], 1e-6 * (Math.Abs(d0[i]) + Math.Abs(s0[i])) + 1e-7, $"AddInPlace n={n} i={i}");
            d.AssertCanaries("AddInPlace dst"); s.AssertCanaries("AddInPlace src");

            using var x = new Buf(n, shift, rnd); var x0 = x.ToArray();
            SimdKernels.ScaleInPlace(x.P, 0.37f, n);
            for (int i = 0; i < n; i++) Close(x0[i] * 0.37f, x.P[i], 1e-6 * Math.Abs(x0[i]) + 1e-7, $"ScaleInPlace n={n} i={i}");
            x.AssertCanaries("ScaleInPlace");

            using var w = new Buf(n, shift, rnd); var w0 = w.ToArray();
            SimdKernels.WeightedAddInPlace(w.P, s.P, 0.61f, n);
            for (int i = 0; i < n; i++) Close(w0[i] + 0.61 * s.P[i], w.P[i], 1e-6 * (Math.Abs(w0[i]) + Math.Abs(s.P[i])) + 1e-7, $"WeightedAddInPlace n={n} i={i}");
            w.AssertCanaries("WeightedAddInPlace");
        }
    }

    [Fact]
    public void SiLuMul_AllSizes()
    {
        var rnd = new Random(5);
        foreach (var (n, shift) in Cases())
        {
            using var g = new Buf(n, shift, rnd, -8f, 8f); using var u = new Buf(n, (shift + 3) & 3, rnd);
            var g0 = g.ToArray();
            SimdKernels.SiLuMul(g.P, u.P, n);
            for (int i = 0; i < n; i++)
            {
                double expected = g0[i] / (1.0 + Math.Exp(-g0[i])) * u.P[i];
                Close(expected, g.P[i], 1e-4 * Math.Abs(expected) + 1e-5, $"SiLuMul n={n} i={i}");
            }
            g.AssertCanaries("SiLuMul gate"); u.AssertCanaries("SiLuMul up");
        }
    }

    private static double[] SoftmaxRef(float[] x)
    {
        double max = x.Where(v => float.IsFinite(v)).Select(v => (double)v).DefaultIfEmpty(0).Max();
        var e = x.Select(v => double.IsNegativeInfinity(v) ? 0.0 : Math.Exp(v - max)).ToArray();
        double s = e.Sum();
        return e.Select(v => v / s).ToArray();
    }

    [Fact]
    public void SoftmaxInPlace_AllSizes_IncludingMaskedAndHugeValues()
    {
        var rnd = new Random(6);
        foreach (var (n, shift) in Cases())
        {
            if (n == 0) continue;
            foreach (int variant in new[] { 0, 1, 2 })
            {
                using var x = new Buf(n, shift, rnd, -20f, 20f, nanTail: (n + shift + variant) % 2 == 0);   // alternate: NaN tail (reads) and finite tail (writes)
                if (variant == 1 && n > 1) for (int i = 0; i < n; i += 3) x.P[i] = float.NegativeInfinity;   // causal-mask style
                if (variant == 1 && n > 1) x.P[n - 1] = 1.5f;                                                // at least one finite
                if (variant == 2) x.P[rnd.Next(n)] = 800f;                                                    // would overflow a naive exp
                var x0 = x.ToArray();
                SimdKernels.SoftmaxInPlace(x.P, n);
                var expected = SoftmaxRef(x0);
                double total = 0;
                for (int i = 0; i < n; i++)
                {
                    Close(expected[i], x.P[i], 2e-5, $"Softmax n={n} variant={variant} i={i}");
                    total += x.P[i];
                }
                Close(1.0, total, 1e-4, $"Softmax sum n={n} variant={variant}");
                x.AssertCanaries($"Softmax variant={variant}");
            }
        }
    }

    [Fact]
    public void RmsNorm_AllSizes_AndInPlace()
    {
        var rnd = new Random(7);
        foreach (var (n, shift) in Cases())
        {
            if (n == 0) continue;
            foreach (bool wide in new[] { false, true })
            {
                using var inp = new Buf(n, shift, rnd, -3f, 3f, nanTail: true); using var w = new Buf(n, (shift + 1) & 3, rnd, 0.5f, 1.5f);
                using var o = new Buf(n, (shift + 2) & 3, rnd);
                double ss = 0; for (int i = 0; i < n; i++) ss += (double)inp.P[i] * inp.P[i];
                double scale = 1.0 / Math.Sqrt(ss / n + 1e-5);
                if (wide) SimdKernels.RmsNormWide(o.P, inp.P, w.P, n, 1e-5f); else SimdKernels.RmsNorm(o.P, inp.P, w.P, n, 1e-5f);
                for (int i = 0; i < n; i++)
                    Close(inp.P[i] * scale * w.P[i], o.P[i], 1e-4 * Math.Abs(inp.P[i] * scale * w.P[i]) + 1e-5, $"RmsNorm wide={wide} n={n} i={i}");
                o.AssertCanaries("RmsNorm out");

                // in place (output == input), as the engine calls it on the residual stream
                var inp0 = inp.ToArray();
                if (wide) SimdKernels.RmsNormWide(inp.P, inp.P, w.P, n, 1e-5f); else SimdKernels.RmsNorm(inp.P, inp.P, w.P, n, 1e-5f);
                for (int i = 0; i < n; i++)
                    Close(inp0[i] * scale * w.P[i], inp.P[i], 1e-4 * Math.Abs(inp0[i] * scale * w.P[i]) + 1e-5, $"RmsNorm in-place wide={wide} n={n} i={i}");
            }
        }
    }

    [Fact]
    public void LayerNorm_AllSizes_WithAndWithoutBias()
    {
        var rnd = new Random(8);
        foreach (var (n, shift) in Cases())
        {
            if (n == 0) continue;
            foreach (bool withBias in new[] { true, false })
            {
                using var inp = new Buf(n, shift, rnd, -3f, 3f, nanTail: true); using var w = new Buf(n, (shift + 1) & 3, rnd, 0.5f, 1.5f);
                using var b = new Buf(n, (shift + 2) & 3, rnd); using var o = new Buf(n, shift, rnd);
                double mean = 0; for (int i = 0; i < n; i++) mean += inp.P[i]; mean /= n;
                double var = 0; for (int i = 0; i < n; i++) var += (inp.P[i] - mean) * (inp.P[i] - mean); var /= n;
                double inv = 1.0 / Math.Sqrt(var + 1e-5);
                SimdKernels.LayerNorm(o.P, inp.P, w.P, withBias ? b.P : null, n, 1e-5f);
                for (int i = 0; i < n; i++)
                {
                    double expected = (inp.P[i] - mean) * inv * w.P[i] + (withBias ? b.P[i] : 0.0);
                    Close(expected, o.P[i], 1e-4 * Math.Abs(expected) + 1e-5, $"LayerNorm bias={withBias} n={n} i={i}");
                }
                o.AssertCanaries("LayerNorm out");
            }
        }
    }

    // ---- activations: normal and extreme magnitudes (a vector exp that overflows or returns garbage shows up out here) ----

    private static float[] ActivationInputs(int n, Random rnd)
    {
        float[] special = [0f, 1f, -1f, 0.5f, -0.5f, 5f, -5f, 20f, -20f, 87f, -87f, 88.5f, -88.5f, 100f, -100f, 1000f, -1000f, 10000f, -10000f];
        var x = new float[n];
        for (int i = 0; i < n; i++)
            x[i] = i % 3 == 0 ? special[rnd.Next(special.Length)] : (float)(rnd.NextDouble() * 24 - 12);
        return x;
    }

    private static void Load(float* dst, float[] src) { for (int i = 0; i < src.Length; i++) dst[i] = src[i]; }
    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
    private static double GeluTanh(double x) => 0.5 * x * (1.0 + Math.Tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));

    [Fact]
    public void Activations_AllSizes_NormalAndExtremeValues()
    {
        var rnd = new Random(9);
        foreach (var (n, shift) in Cases())
        {
            if (n == 0) continue;
            var inputs = ActivationInputs(n, rnd);
            double Tol(double expected) => 1e-4 * Math.Abs(expected) + 2e-5;

            using (var x = new Buf(n, shift, rnd)) { Load(x.P, inputs); SimdKernels.SigmoidInPlace(x.P, n);
                for (int i = 0; i < n; i++) Close(Sigmoid(inputs[i]), x.P[i], 1e-4, $"SigmoidInPlace n={n} x={inputs[i]}"); x.AssertCanaries("Sigmoid"); }

            using (var o = new Buf(n, shift, rnd)) using (var inp = new Buf(n, (shift + 1) & 3, rnd)) { Load(inp.P, inputs); SimdKernels.GeluF32(o.P, inp.P, n);
                for (int i = 0; i < n; i++) Close(GeluTanh(inputs[i]), o.P[i], Tol(GeluTanh(inputs[i])), $"GeluF32 n={n} x={inputs[i]}"); o.AssertCanaries("GeluF32"); }

            using (var x = new Buf(n, shift, rnd)) { Load(x.P, inputs); SimdKernels.GeluInPlace(x.P, n);
                for (int i = 0; i < n; i++) Close(GeluTanh(inputs[i]), x.P[i], Tol(GeluTanh(inputs[i])), $"GeluInPlace n={n} x={inputs[i]}"); x.AssertCanaries("GeluInPlace"); }

            using (var g = new Buf(n, shift, rnd)) using (var u = new Buf(n, (shift + 2) & 3, rnd, -3f, 3f)) using (var o = new Buf(n, shift, rnd))
            {
                Load(g.P, inputs); SimdKernels.GeluTanhMul(g.P, u.P, o.P, n);
                for (int i = 0; i < n; i++) { double e = GeluTanh(inputs[i]) * u.P[i]; Close(e, o.P[i], Tol(e), $"GeluTanhMul n={n} x={inputs[i]}"); }
                o.AssertCanaries("GeluTanhMul out");
            }

            using (var x = new Buf(n, shift, rnd)) { Load(x.P, inputs); SimdKernels.GeluQuickInPlace(x.P, n);
                for (int i = 0; i < n; i++) { double e = inputs[i] * Sigmoid(1.702 * inputs[i]); Close(e, x.P[i], Tol(e), $"GeluQuickInPlace n={n} x={inputs[i]}"); } x.AssertCanaries("GeluQuick"); }

            using (var x = new Buf(n, shift, rnd)) { Load(x.P, inputs); SimdKernels.ReluSqrInPlace(x.P, n);
                for (int i = 0; i < n; i++) { double e = Math.Max(0.0, inputs[i]); e *= e; Close(e, x.P[i], 1e-5 * e + 1e-6, $"ReluSqrInPlace n={n} x={inputs[i]}"); } x.AssertCanaries("ReluSqr"); }

            // exact (erf) GELU: vector path vs the scalar path of the same A&S formula, and sane at the extremes
            var span = new float[n]; Array.Copy(inputs, span, n);
            ErfGelu.InPlace(span);
            for (int i = 0; i < n; i++)
            {
                double scalar = ErfGelu.Apply(inputs[i]);
                Close(scalar, span[i], 1e-4 * Math.Abs(scalar) + 1e-5, $"ErfGelu vector vs scalar n={n} x={inputs[i]}");
                if (inputs[i] > 10) Close(inputs[i], span[i], 1e-4 * inputs[i], $"ErfGelu(x) -> x for large x n={n}");
                if (inputs[i] < -10) Close(0, span[i], 1e-4, $"ErfGelu(x) -> 0 for very negative x n={n}");
            }
        }
    }
}
