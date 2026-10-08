using System.Runtime.InteropServices;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// The IQ3_S / IQ4_XS shared-decode multi-input dots (MTP batched verify, docs/00-current-work.md item 17) reconstruct each
/// weight group once and apply it to 2 or 4 activations. A token's result must not depend on how many activations ran with it,
/// so every output is asserted bit-for-bit equal to the single-input production dot. Also checks the MatVec2In/MatVec4In
/// dispatch (the parallel row sweep) against N independent MatVec calls.
/// </summary>
public sealed unsafe class IqSharedDecodeMultiInputTests
{
    private const int Cols = 1024; // 4 super-blocks
    private const int Rows = 96;   // >= MinRowsForParallel so the dispatchers take the multi-input sweep

    [Theory]
    [InlineData(DType.IQ3_S, 110)]
    [InlineData(DType.IQ4_XS, 136)]
    [InlineData(DType.IQ3_XXS, 98)]
    [InlineData(DType.IQ2_S, 82)]
    public void MatVec2In_And_MatVec4In_EqualIndependentMatVecs_Bitwise(DType dtype, int bytesPerBlock)
    {
        Assert.True(SimdKernels.HasSharedDecodeMultiInput(dtype, Cols));
        int nb = Cols / 256;
        var rng = new Random(2026);
        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)(Rows * nb * bytesPerBlock), 64);
        float* x = (float*)NativeMemory.AlignedAlloc((nuint)(4 * Cols * sizeof(float)), 64);
        float* want = (float*)NativeMemory.AlignedAlloc((nuint)(4 * Rows * sizeof(float)), 64);
        float* got = (float*)NativeMemory.AlignedAlloc((nuint)(4 * Rows * sizeof(float)), 64);
        try
        {
            for (int i = 0; i < Rows * nb * bytesPerBlock; i++) w[i] = (byte)rng.Next(256);
            for (int r = 0; r < Rows; r++)
                for (int b = 0; b < nb; b++)
                    *(ushort*)(w + (r * nb + b) * bytesPerBlock) = BitConverter.HalfToUInt16Bits((Half)(0.25f + (float)rng.NextDouble()));
            for (int i = 0; i < 4 * Cols; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

            for (int t = 0; t < 4; t++)
                SimdKernels.MatVec(want + t * Rows, w, x + t * Cols, Rows, Cols, dtype);

            SimdKernels.MatVec2In(got, got + Rows, w, x, x + Cols, Rows, Cols, dtype);
            for (int i = 0; i < 2 * Rows; i++)
                Assert.True(BitConverter.SingleToInt32Bits(want[i]) == BitConverter.SingleToInt32Bits(got[i]),
                    $"{dtype} MatVec2In differs at {i}: {want[i]} vs {got[i]}");

            SimdKernels.MatVec4In(got, got + Rows, got + 2 * Rows, got + 3 * Rows, w,
                x, x + Cols, x + 2 * Cols, x + 3 * Cols, Rows, Cols, dtype);
            for (int i = 0; i < 4 * Rows; i++)
                Assert.True(BitConverter.SingleToInt32Bits(want[i]) == BitConverter.SingleToInt32Bits(got[i]),
                    $"{dtype} MatVec4In differs at {i}: {want[i]} vs {got[i]}");
        }
        finally
        {
            NativeMemory.AlignedFree(w); NativeMemory.AlignedFree(x);
            NativeMemory.AlignedFree(want); NativeMemory.AlignedFree(got);
        }
    }

    [Theory]
    [InlineData(DType.Q4_K)]
    [InlineData(DType.Q8_0)]
    [InlineData(DType.IQ2_XS)]
    public void NonSharedDtypes_ReportNoSharedDecodeKernel(DType dtype) =>
        Assert.False(SimdKernels.HasSharedDecodeMultiInput(dtype, Cols));

    // Routing safety for the batched verify: a group with at most two real tokens may use the 2-input kernel instead of a
    // padded 4-input one only if a token's result does not depend on the width. Pinned per dtype so a future kernel change
    // that breaks the property fails here, not as a silent greedy flip.
    [Theory]
    [InlineData(DType.Q4_K, 144)]
    [InlineData(DType.Q5_K, 176)]
    [InlineData(DType.Q6_K, 210)]
    [InlineData(DType.IQ3_S, 110)]
    [InlineData(DType.IQ4_XS, 136)]
    public void TwoInput_Equals_FirstTwoLanesOfFourInput_Bitwise(DType dtype, int bytesPerBlock)
    {
        int nb = Cols / 256;
        var rng = new Random(77);
        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)(Rows * nb * bytesPerBlock), 64);
        float* x = (float*)NativeMemory.AlignedAlloc((nuint)(4 * Cols * sizeof(float)), 64);
        float* a = (float*)NativeMemory.AlignedAlloc((nuint)(4 * Rows * sizeof(float)), 64);
        float* b = (float*)NativeMemory.AlignedAlloc((nuint)(2 * Rows * sizeof(float)), 64);
        try
        {
            for (int i = 0; i < Rows * nb * bytesPerBlock; i++) w[i] = (byte)rng.Next(256);
            for (int r = 0; r < Rows; r++)
                for (int blk = 0; blk < nb; blk++)
                    *(ushort*)(w + (r * nb + blk) * bytesPerBlock) = BitConverter.HalfToUInt16Bits((Half)(0.25f + (float)rng.NextDouble()));
            for (int i = 0; i < 4 * Cols; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
            SimdKernels.MatVec4In(a, a + Rows, a + 2 * Rows, a + 3 * Rows, w, x, x + Cols, x + 2 * Cols, x + 3 * Cols, Rows, Cols, dtype);
            SimdKernels.MatVec2In(b, b + Rows, w, x, x + Cols, Rows, Cols, dtype);
            for (int i = 0; i < 2 * Rows; i++)
                Assert.True(BitConverter.SingleToInt32Bits(a[i]) == BitConverter.SingleToInt32Bits(b[i]),
                    $"{dtype}: 2-input differs from 4-input lane at {i}: {b[i]} vs {a[i]}");
        }
        finally
        {
            NativeMemory.AlignedFree(w); NativeMemory.AlignedFree(x); NativeMemory.AlignedFree(a); NativeMemory.AlignedFree(b);
        }
    }
}
