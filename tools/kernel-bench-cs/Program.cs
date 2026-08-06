using System.Diagnostics;
using System.Runtime.InteropServices;
using OpenTail.Stingray.Cpu;

return KernelBench.Run(args);

internal static unsafe class KernelBench
{
    private const int SuperBlock = 256;
    private const int Q6KBytes = 210;

    public static int Run(string[] args)
    {
        if (Environment.GetEnvironmentVariable("DOTNET_TC_QuickJitForLoops") != "0")
        {
            Console.Error.WriteLine("Refusing to run: set DOTNET_TC_QuickJitForLoops=0 (tiered JIT invalidates these numbers).");
            return 1;
        }

        if (!TryParse(args, out int k, out int rows, out int reps, out string? error))
        {
            Console.Error.WriteLine(error);
            Console.Error.WriteLine("Usage: OpenTail.Stingray.KernelBench [k=8192] [rows=512] [reps=8]");
            return 1;
        }

        int blocks = k / SuperBlock;
        nuint alignment = 64;
        float* input = null;
        float* rowSource = null;
        byte* weights = null;
        byte* activation = null;
        try
        {
            input = (float*)NativeMemory.AlignedAlloc((nuint)(k * sizeof(float)), alignment);
            rowSource = (float*)NativeMemory.AlignedAlloc((nuint)(k * sizeof(float)), alignment);
            weights = (byte*)NativeMemory.AlignedAlloc((nuint)(rows * blocks * Q6KBytes), alignment);
            activation = (byte*)NativeMemory.AlignedAlloc((nuint)SimdKernels.Q8KScratchBytes(k), alignment);
            if (input is null || rowSource is null || weights is null || activation is null)
                throw new OutOfMemoryException("NativeMemory.AlignedAlloc returned null.");

            for (int i = 0; i < k; i++) input[i] = Synth(i);
            SimdKernels.QuantizeRowToQ8K(input, k, activation);
            for (int row = 0; row < rows; row++)
            {
                int offset = checked(row * 7919);
                for (int i = 0; i < k; i++) rowSource[i] = Synth(checked(i + offset));
                QuantizeRowToQ6K(rowSource, weights + (long)row * blocks * Q6KBytes, k);
            }

            float checksum = RunDots(weights, activation, k, rows);
            Console.WriteLine($"k={k} rows={rows} reps={reps}  (QK_K={SuperBlock}, {blocks} blocks/row)");
            Console.WriteLine($"checksum q6_k = {checksum:F6}");

            BenchResult result = Bench(() => RunDots(weights, activation, k, rows), reps, warmup: 3);
            Console.WriteLine($"q6_k best={result.BestMs:F4} ms  mean={result.MeanMs:F4} ms  sd={result.StandardDeviationMs:F4}  ({rows / result.BestMs / 1000d:F1} Mdot/s)");
            Console.WriteLine("Q6_K uses the same Q8_K activation format as llama.cpp; compare checksum before timing.");
            return 0;
        }
        finally
        {
            if (activation is not null) NativeMemory.AlignedFree(activation);
            if (weights is not null) NativeMemory.AlignedFree(weights);
            if (rowSource is not null) NativeMemory.AlignedFree(rowSource);
            if (input is not null) NativeMemory.AlignedFree(input);
        }
    }

    private static bool TryParse(string[] args, out int k, out int rows, out int reps, out string? error)
    {
        k = args.Length > 0 && int.TryParse(args[0], out int parsedK) ? parsedK : 8192;
        rows = args.Length > 1 && int.TryParse(args[1], out int parsedRows) ? parsedRows : 512;
        reps = args.Length > 2 && int.TryParse(args[2], out int parsedReps) ? parsedReps : 8;
        error = null;
        if (args.Length > 3) error = "Only Q6_K is implemented: Q4_K is intentionally withheld until its distinct activation format has a byte-identical weight-input path.";
        else if (k <= 0 || k % SuperBlock != 0) error = $"k must be a positive multiple of {SuperBlock}.";
        else if (rows <= 0 || reps <= 0) error = "rows and reps must be positive.";
        return error is null;
    }

    private static float Synth(int i) => (float)(Math.Sin((double)i * 0.017) * 2.0 + Math.Cos((double)i * 0.0031));

    private static float RunDots(byte* weights, byte* activation, int k, int rows)
    {
        int rowBytes = (k / SuperBlock) * Q6KBytes;
        float sum = 0;
        for (int row = 0; row < rows; row++) sum += SimdKernels.DotQ6K_Q8K(weights + (long)row * rowBytes, activation, k);
        return sum;
    }

    // The delegate invocation happens once per whole rows sweep, outside the measured dot loop.
    // It therefore does not change the kernel comparison while keeping the harness readable.
    private static BenchResult Bench(Func<float> action, int reps, int warmup)
    {
        for (int i = 0; i < warmup; i++) _ = action();
        var samples = new double[reps];
        for (int i = 0; i < reps; i++)
        {
            long start = Stopwatch.GetTimestamp();
            _ = action();
            samples[i] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        }
        double mean = samples.Average();
        double variance = samples.Select(x => (x - mean) * (x - mean)).Average();
        return new(samples.Min(), mean, Math.Sqrt(variance));
    }

    private static void QuantizeRowToQ6K(float* input, byte* output, int cols)
    {
        Span<sbyte> levels = stackalloc sbyte[SuperBlock];
        Span<float> scales = stackalloc float[16];
        for (int block = 0; block < cols / SuperBlock; block++)
        {
            float* source = input + block * SuperBlock;
            byte* destination = output + block * Q6KBytes;
            float maxScale = 0, maxAbsScale = 0;
            for (int group = 0; group < 16; group++)
            {
                float scale = MakeQxQuants(source + group * 16, levels.Slice(group * 16, 16));
                scales[group] = scale;
                if (MathF.Abs(scale) > maxAbsScale) { maxAbsScale = MathF.Abs(scale); maxScale = scale; }
            }
            new Span<byte>(destination, Q6KBytes).Clear();
            if (maxAbsScale < 1e-15f) continue;

            float inverseScale = -128f / maxScale;
            WriteHalf(destination + 208, 1f / inverseScale);
            for (int group = 0; group < 16; group++)
                ((sbyte*)destination)[192 + group] = (sbyte)Math.Min(127, NearestInt(inverseScale * scales[group]));

            for (int group = 0; group < 16; group++)
            {
                float d = ReadHalf(destination + 208) * ((sbyte*)destination)[192 + group];
                if (d == 0) continue;
                for (int i = 0; i < 16; i++) levels[group * 16 + i] = (sbyte)(Math.Clamp(NearestInt(source[group * 16 + i] / d), -32, 31) + 32);
            }
            for (int baseIndex = 0; baseIndex < SuperBlock; baseIndex += 128)
            for (int lane = 0; lane < 32; lane++)
            {
                byte q1 = (byte)(levels[baseIndex + lane] & 0x0f); byte q2 = (byte)(levels[baseIndex + lane + 32] & 0x0f);
                byte q3 = (byte)(levels[baseIndex + lane + 64] & 0x0f); byte q4 = (byte)(levels[baseIndex + lane + 96] & 0x0f);
                int ql = baseIndex / 2;
                destination[ql + lane] = (byte)(q1 | (q3 << 4));
                destination[ql + 32 + lane] = (byte)(q2 | (q4 << 4));
                destination[128 + baseIndex / 4 + lane] = (byte)((levels[baseIndex + lane] >> 4) | ((levels[baseIndex + lane + 32] >> 4) << 2) | ((levels[baseIndex + lane + 64] >> 4) << 4) | ((levels[baseIndex + lane + 96] >> 4) << 6));
            }
        }
    }

    private static float MakeQxQuants(float* input, Span<sbyte> levels)
    {
        float max = 0, maxAbs = 0;
        for (int i = 0; i < 16; i++) if (MathF.Abs(input[i]) > maxAbs) { maxAbs = MathF.Abs(input[i]); max = input[i]; }
        if (maxAbs < 1e-15f) { levels.Clear(); return 0; }
        float inverse = -32f / max, sumLx = 0, sumL2 = 0;
        for (int i = 0; i < 16; i++)
        {
            int level = Math.Clamp(NearestInt(inverse * input[i]), -32, 31);
            levels[i] = (sbyte)(level + 32);
            float weight = input[i] * input[i]; sumLx += weight * input[i] * level; sumL2 += weight * level * level;
        }
        float scale = sumL2 != 0 ? sumLx / sumL2 : 0;
        float best = scale * sumLx;
        for (int candidate = -9; candidate <= 9; candidate++)
        {
            if (candidate == 0) continue;
            inverse = -(32 + 0.1f * candidate) / max; sumLx = sumL2 = 0;
            for (int i = 0; i < 16; i++) { int level = Math.Clamp(NearestInt(inverse * input[i]), -32, 31); float weight = input[i] * input[i]; sumLx += weight * input[i] * level; sumL2 += weight * level * level; }
            if (sumL2 > 0 && sumLx * sumLx > best * sumL2)
            {
                for (int i = 0; i < 16; i++) levels[i] = (sbyte)(Math.Clamp(NearestInt(inverse * input[i]), -32, 31) + 32);
                scale = sumLx / sumL2; best = scale * sumLx;
            }
        }
        return scale;
    }

    private static int NearestInt(float value) => (BitConverter.SingleToInt32Bits(value + 12_582_912f) & 0x007f_ffff) - 0x0040_0000;
    private static void WriteHalf(byte* destination, float value) => *(ushort*)destination = BitConverter.HalfToUInt16Bits((Half)value);
    private static float ReadHalf(byte* source) => (float)BitConverter.UInt16BitsToHalf(*(ushort*)source);
    private readonly record struct BenchResult(double BestMs, double MeanMs, double StandardDeviationMs);
}
