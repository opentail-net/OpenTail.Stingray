using System.Runtime.InteropServices;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Structural sweep of <c>SimdKernels.MatVec</c> for the common quantized formats: random valid blocks, the dequantized matrix
/// as the float reference, and awkward row counts (the 2-row / 4-row kernels and the parallel threshold have row-remainder paths)
/// and widths. Activation quantization makes the kernels differ from a float dot by about a percent, so the tolerance is set
/// to catch STRUCTURAL errors (wrong block, dropped tail row, scale applied to the wrong sub-block), which show up as errors
/// comparable to the result's own magnitude, not rounding noise.
/// </summary>
public sealed unsafe class MatVecSizeSweepTests
{
    // fp16 scale fields inside one block, per format: random bytes there can be NaN/Inf, which would poison the reference.
    private static readonly (DType Type, int[] HalfOffsets)[] Formats =
    [
        (DType.Q4_0, [0]), (DType.Q4_1, [0, 2]), (DType.Q5_0, [0]), (DType.Q5_1, [0, 2]), (DType.Q8_0, [0]),
        (DType.Q2_K, [80, 82]), (DType.Q3_K, [108]), (DType.Q4_K, [0, 2]), (DType.Q5_K, [0, 2]), (DType.Q6_K, [208]),
        (DType.IQ4_NL, [0]), (DType.IQ4_XS, [0]),
    ];

    private static readonly int[] Rows = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 63, 64, 65, 130, 257];
    private static readonly int[] Cols = [256, 512, 768, 1280, 2304];

    private static byte[] RandomMatrix(DType type, int rows, int cols, int[] halfOffsets, Random rnd)
    {
        int bpb = DTypeInfo.BytesPerBlock(type), bs = DTypeInfo.BlockSize(type);
        long blocks = (long)rows * (cols / bs);
        var data = new byte[blocks * bpb];
        rnd.NextBytes(data);
        for (long b = 0; b < blocks; b++)
            foreach (int off in halfOffsets)
            {
                // a modest positive scale, so the dequantized values stay O(1) and finite
                ushort bits = BitConverter.HalfToUInt16Bits((Half)(0.01 + rnd.NextDouble() * 0.05));
                data[b * bpb + off] = (byte)bits;
                data[b * bpb + off + 1] = (byte)(bits >> 8);
            }
        return data;
    }

    [Fact]
    public void MatVec_StructuralSweep_AllFormats_AllRowRemainders()
    {
        var rnd = new Random(20261001);
        var worst = new SortedDictionary<string, double>();
        foreach (var (type, halfOffsets) in Formats)
        {
            foreach (int cols in Cols)
            {
                foreach (int rows in Rows)
                {
                    byte[] w = RandomMatrix(type, rows, cols, halfOffsets, rnd);
                    var deq = new float[(long)rows * cols];
                    Dequantize.ToFloat32(w, deq, type, deq.Length);
                    var input = new float[cols];
                    for (int i = 0; i < cols; i++) input[i] = (float)(rnd.NextDouble() * 2 - 1);

                    foreach (bool allowQ8 in new[] { true, false })
                    {
                        var output = new float[rows + 16];
                        for (int i = rows; i < output.Length; i++) output[i] = float.NaN;     // canary: rows beyond `rows` must stay untouched
                        fixed (byte* wp = w) fixed (float* ip = input) fixed (float* op = output)
                            SimdKernels.MatVec(op, wp, ip, rows, cols, type, allowQ8);

                        for (int i = rows; i < output.Length; i++)
                            Assert.True(float.IsNaN(output[i]), $"{type} rows={rows} cols={cols}: wrote output[{i}] past the last row");

                        for (int r = 0; r < rows; r++)
                        {
                            double expected = 0, mag = 0;
                            for (int c = 0; c < cols; c++)
                            {
                                double p = (double)deq[(long)r * cols + c] * input[c];
                                expected += p; mag += Math.Abs(p);
                            }
                            double err = Math.Abs(expected - output[r]) / (mag + 1e-9);
                            string key = $"{type}";
                            worst[key] = Math.Max(worst.GetValueOrDefault(key), err);
                            Assert.True(double.IsFinite(output[r]) && err < 0.08,
                                $"{type} rows={rows} cols={cols} allowQ8={allowQ8} row={r}: expected {expected:G6}, got {output[r]:G6} (error {err:P1} of sum|w*x|)");
                        }
                    }
                }
            }
        }
        foreach (var kv in worst) Console.WriteLine($"[MatVecSweep] {kv.Key,-8} worst error / sum|w*x| = {kv.Value:P2}");
    }

    private static double[] Reference(float[] deq, int rows, int cols, float[] input, out double[] mag)
    {
        var exp = new double[rows]; mag = new double[rows];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
            {
                double p = (double)deq[(long)r * cols + c] * input[c];
                exp[r] += p; mag[r] += Math.Abs(p);
            }
        return exp;
    }

    private static void CheckRows(string label, double[] expected, double[] mag, float[] got, int offset)
    {
        for (int r = 0; r < expected.Length; r++)
        {
            double err = Math.Abs(expected[r] - got[offset + r]) / (mag[r] + 1e-9);
            Assert.True(float.IsFinite(got[offset + r]) && err < 0.08,
                $"{label} row={r}: expected {expected[r]:G6}, got {got[offset + r]:G6} (error {err:P1} of sum|w*x|)");
        }
    }

    [Fact]
    public void MultiInputAndBatchedEntryPoints_AgreeWithTheDequantizedReference()
    {
        var rnd = new Random(77);
        int[] rowSet = [1, 3, 8, 17, 65, 130];
        int[] colSet = [256, 1280];
        int[] batchSet = [1, 2, 3, 4, 5, 7, 8, 9, 16, 33];
        foreach (var (type, halfOffsets) in Formats)
        foreach (int cols in colSet)
        foreach (int rows in rowSet)
        {
            byte[] w = RandomMatrix(type, rows, cols, halfOffsets, rnd);
            byte[] w2 = RandomMatrix(type, rows, cols, halfOffsets, rnd);
            var deq = new float[(long)rows * cols]; Dequantize.ToFloat32(w, deq, type, deq.Length);
            var deq2 = new float[(long)rows * cols]; Dequantize.ToFloat32(w2, deq2, type, deq2.Length);
            string tag = $"{type} rows={rows} cols={cols}";

            // ---- MatVecDual: two weight matrices, one input
            var x = new float[cols]; for (int i = 0; i < cols; i++) x[i] = (float)(rnd.NextDouble() * 2 - 1);
            var o1 = new float[rows]; var o2 = new float[rows];
            fixed (byte* wp = w) fixed (byte* w2p = w2) fixed (float* xp = x) fixed (float* o1p = o1) fixed (float* o2p = o2)
                SimdKernels.MatVecDual(o1p, wp, o2p, w2p, xp, rows, cols, type, type);
            CheckRows($"MatVecDual[0] {tag}", Reference(deq, rows, cols, x, out var m1), m1, o1, 0);
            CheckRows($"MatVecDual[1] {tag}", Reference(deq2, rows, cols, x, out var m2), m2, o2, 0);

            // ---- MatVec2In / MatVec4In: several inputs, one weight matrix
            var xs = Enumerable.Range(0, 4).Select(_ => { var v = new float[cols]; for (int i = 0; i < cols; i++) v[i] = (float)(rnd.NextDouble() * 2 - 1); return v; }).ToArray();
            var outs = Enumerable.Range(0, 4).Select(_ => new float[rows]).ToArray();
            fixed (byte* wp = w) fixed (float* a = xs[0]) fixed (float* b = xs[1]) fixed (float* c = xs[2]) fixed (float* d = xs[3])
            fixed (float* p0 = outs[0]) fixed (float* p1 = outs[1]) fixed (float* p2 = outs[2]) fixed (float* p3 = outs[3])
            {
                SimdKernels.MatVec2In(p0, p1, wp, a, b, rows, cols, type);
                CheckRows($"MatVec2In[0] {tag}", Reference(deq, rows, cols, xs[0], out var q0), q0, outs[0], 0);
                CheckRows($"MatVec2In[1] {tag}", Reference(deq, rows, cols, xs[1], out var q1), q1, outs[1], 0);
                SimdKernels.MatVec4In(p0, p1, p2, p3, wp, a, b, c, d, rows, cols, type);
                for (int k = 0; k < 4; k++)
                    CheckRows($"MatVec4In[{k}] {tag}", Reference(deq, rows, cols, xs[k], out var qm), qm, outs[k], 0);
            }

            // ---- MatMulBatched: n inputs through the batched entry point (output [n, rows], canary after it)
            foreach (int n in batchSet)
            {
                var inp = new float[(long)n * cols]; for (long i = 0; i < inp.Length; i++) inp[i] = (float)(rnd.NextDouble() * 2 - 1);
                var outp = new float[(long)n * rows + 16];
                for (long i = (long)n * rows; i < outp.Length; i++) outp[i] = float.NaN;
                fixed (byte* wp = w) fixed (float* ip = inp) fixed (float* op = outp)
                    SimdKernels.MatMulBatched(op, wp, ip, n, rows, cols, type);
                for (long i = (long)n * rows; i < outp.Length; i++)
                    Assert.True(float.IsNaN(outp[i]), $"MatMulBatched {tag} n={n}: wrote past the end of the output");
                for (int t = 0; t < n; t++)
                {
                    var xt = new float[cols]; Array.Copy(inp, (long)t * cols, xt, 0, cols);
                    CheckRows($"MatMulBatched {tag} n={n} token={t}", Reference(deq, rows, cols, xt, out var mt), mt, outp, t * rows);
                }
            }
        }
    }
}
