using System.Diagnostics;
using System.Runtime.InteropServices;
using OpenTail.Stingray.Cpu;

/// <summary>One-thread IQ4_XS check + timing of the production shared-decode 2In/4In dots vs N separate single-input dots.</summary>
internal static unsafe class Iq4XsMultiBench
{
    public static int Run(string[] args)
    {
        int cols = args.Length > 0 ? int.Parse(args[0]) : 5120;
        int nb = cols / 256, rowBytes = nb * 136, q8Bytes = SimdKernels.Q8KScratchBytes(cols);
        var rng = new Random(777);
        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)(64 * rowBytes), 64);
        for (int i = 0; i < 64 * rowBytes; i++) w[i] = (byte)rng.Next(256);
        for (int r = 0; r < 64; r++) for (int b = 0; b < nb; b++)
            *(ushort*)(w + r * rowBytes + b * 136) = BitConverter.HalfToUInt16Bits((Half)(0.5f + (float)rng.NextDouble()));
        byte*[] q8 = new byte*[4];
        float* tmp = (float*)NativeMemory.AlignedAlloc((nuint)(cols * 4), 64);
        for (int t = 0; t < 4; t++)
        {
            for (int i = 0; i < cols; i++) tmp[i] = (float)(rng.NextDouble() * 2 - 1);
            q8[t] = (byte*)NativeMemory.AlignedAlloc((nuint)q8Bytes, 64);
            SimdKernels.QuantizeRowToQ8K(tmp, cols, q8[t]);
        }
        double maxErr = 0;
        for (int r = 0; r < 64; r++)
        {
            byte* row = w + r * rowBytes;
            SimdKernels.DotIq4Xs_Q8K_4In(row, q8[0], q8[1], q8[2], q8[3], cols, out float a, out float b, out float c, out float d);
            SimdKernels.DotIq4Xs_Q8K_2In(row, q8[0], q8[1], cols, out float e, out float f);
            float[] got = { a, b, c, d };
            for (int t = 0; t < 4; t++)
            {
                float refv = SimdKernels.DotIq4Xs_Q8K(row, q8[t], cols);
                maxErr = Math.Max(maxErr, Math.Abs(refv - got[t]) / (Math.Abs(refv) + 1e-3));
            }
            maxErr = Math.Max(maxErr, Math.Abs(SimdKernels.DotIq4Xs_Q8K(row, q8[0], cols) - e) / (Math.Abs(e) + 1e-3));
            maxErr = Math.Max(maxErr, Math.Abs(SimdKernels.DotIq4Xs_Q8K(row, q8[1], cols) - f) / (Math.Abs(f) + 1e-3));
        }
        Console.WriteLine($"IQ4_XS cols={cols} correctness max rel err = {maxErr:E2}");
        if (maxErr > 1e-4) return 2;
        double a1 = T(() => { for (int i = 0; i < 1; i++) { } }, 0);
        double A1 = Time(w, q8, cols, 'A', 1), A2 = Time(w, q8, cols, 'A', 2), A4 = Time(w, q8, cols, 'A', 4);
        double S2 = Time(w, q8, cols, 'S', 2), S4 = Time(w, q8, cols, 'S', 4);
        Console.WriteLine($"A N=1 {A1:F0} ns | A N=2 {A2:F0} | S N=2 {S2:F0} ({A2 / S2:F2}x) | A N=4 {A4:F0} | S N=4 {S4:F0} ({A4 / S4:F2}x)   ns/row, hot");
        return 0;
    }

    private static double T(Action a, int _) { a(); return 0; }

    private static double Time(byte* w, byte*[] q8, int cols, char kind, int n)
    {
        double best = double.MaxValue; float sink = 0;
        for (int trial = 0; trial < 8; trial++)
        {
            long t0 = Stopwatch.GetTimestamp(); long calls = 0;
            while (Stopwatch.GetTimestamp() - t0 < 0.4 * Stopwatch.Frequency)
            {
                for (int r = 0; r < 64; r++)
                {
                    byte* row = w + r * ((cols / 256) * 136);
                    if (kind == 'A') for (int t = 0; t < n; t++) sink += SimdKernels.DotIq4Xs_Q8K(row, q8[t], cols);
                    else if (n == 2) { SimdKernels.DotIq4Xs_Q8K_2In(row, q8[0], q8[1], cols, out float x, out float y); sink += x + y; }
                    else { SimdKernels.DotIq4Xs_Q8K_4In(row, q8[0], q8[1], q8[2], q8[3], cols, out float x, out float y, out float z, out float v); sink += x + v; }
                }
                calls += 64;
            }
            double ns = (Stopwatch.GetTimestamp() - t0) * 1e9 / Stopwatch.Frequency / calls;
            if (trial >= 3 && ns < best) best = ns;
        }
        GC.KeepAlive(sink); return best;
    }
}
