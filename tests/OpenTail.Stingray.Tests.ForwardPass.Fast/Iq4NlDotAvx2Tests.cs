using System.Runtime.InteropServices;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// The AVX2 IQ4_NL x Q8_0 row dot (pshufb codebook lookup, ggml's structure) against the scalar reference. The integer part is exact and the only difference
/// is float accumulation order, so results must agree to a tight relative tolerance, including odd block counts (scalar tail) and the shortest AVX2 row.
/// </summary>
public sealed unsafe class Iq4NlDotAvx2Tests
{
    [Theory]
    [InlineData(64)]     // 2 blocks: one AVX2 pair, no tail
    [InlineData(96)]     // 3 blocks: pair + scalar tail
    [InlineData(1024)]
    [InlineData(4128)]   // 129 blocks, odd
    public void Avx2Dot_MatchesScalarReference(int cols)
    {
        if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported) return;
        var rng = new Random(cols);
        var input = new float[cols];
        for (var i = 0; i < cols; i++) input[i] = (float)(rng.NextDouble() * 2 - 1) * (rng.Next(32) == 0 ? 6f : 1f);
        var bytesPerRow = cols / 32 * 18;
        var scratchBytes = SimdKernels.Q8_0ScratchBytes(cols);
        var scratch = (byte*)NativeMemory.AlignedAlloc((nuint)scratchBytes, 64);
        var row = (byte*)NativeMemory.AlignedAlloc((nuint)bytesPerRow, 64);
        try
        {
            fixed (float* inp = input) SimdKernels.QuantizeRowToQ8_0(inp, cols, scratch);
            var worst = 0.0;
            for (var trial = 0; trial < 200; trial++)
            {
                var buf = new byte[bytesPerRow];
                rng.NextBytes(buf);
                for (var b = 0; b < cols / 32; b++)   // sane fp16 scale per block (random bytes can be inf/NaN)
                {
                    var h = BitConverter.HalfToUInt16Bits((Half)(0.002f + 0.004f * (float)rng.NextDouble()));
                    buf[b * 18] = (byte)h; buf[b * 18 + 1] = (byte)(h >> 8);
                }
                buf.AsSpan().CopyTo(new Span<byte>(row, bytesPerRow));
                var fast = SimdKernels.DotIq4Nl_Q8_0_Avx2(row, scratch, cols / 32);
                var slow = SimdKernels.DotIq4Nl_Q8_0_Scalar(row, scratch, cols);
                worst = Math.Max(worst, Math.Abs(fast - slow) / (Math.Abs(slow) + 1e-3));
            }
            Assert.True(worst < 1e-4, $"worst relative difference {worst:E2}");
        }
        finally { NativeMemory.AlignedFree(scratch); NativeMemory.AlignedFree(row); }
    }
}
