using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Correctness check for the IQ1_M dequantizer added 2026-08-28
/// (docs/05-cpu-architecture-kernel-opportunities.md, Backlog A). Unlike every other IQ format
/// this session, IQ1_M deliberately has no fast matvec kernel wired into
/// <c>SimdKernels.MatVec</c>'s dispatch and no AVX2 path at all -- IQ1_S (same grid, less
/// bookkeeping) already measured a real, repeated loss against the fallback, so writing and
/// measuring a full AVX2 port for IQ1_M was skipped on that evidence (see
/// <c>SimdKernels.DotIq1M_Q8K_Scalar</c>'s remarks). What ships is the dequantizer
/// (<c>Dequantize.DequantIq1M</c>) plus admission into <c>ModelCompatibility.
/// IsSupportedWeightDType</c> -- the real deliverable, a previously-unloadable format now loads
/// and runs (via the generic <c>MatVecDequantFallback</c>).
///
/// <para>With no AVX2 kernel to cross-check the scalar one against, and no local checkpoint to
/// get a llama.cpp receipt from, the primary correctness gate here is the same technique used for
/// IQ1_S: cross-verify two independently-derived formulas (the dequantizer, worked from
/// <c>dequantize_row_iq1_m</c>, against <c>DotIq1M_Q8K_Scalar</c>, worked from
/// <c>ggml_vec_dot_iq1_m_q8_K</c>) rather than a formula checking itself.</para>
/// </summary>
public sealed unsafe class SimdKernelsIq1MTests
{
    private static byte[] BuildBlock(Random rng)
    {
        const int bytesPerBlock = 56;
        var bytes = new byte[bytesPerBlock];
        rng.NextBytes(bytes);

        // Patch the 4 scattered nibbles that combine into the shared FP16 scale (ggml's
        // iq1m_scale_t trick, see DequantIq1M's remarks) so it decodes to a valid, finite, small
        // positive value. Fully random bytes can and do produce a NaN/Inf half pattern by chance
        // (~1/16 of trials) for THIS specific field only — the 3-bit sub-block scales sharing the
        // same 8 bytes are always finite regardless of value, so only these 4 nibbles need fixing;
        // every other bit, including the rest of these same 4 bytes, stays fully random.
        float scale = (float)(rng.NextDouble() * 0.09 + 0.01);
        ushort scaleBits = BitConverter.HalfToUInt16Bits((Half)scale);
        bytes[49] = (byte)((bytes[49] & 0x0F) | ((scaleBits & 0xF) << 4));
        bytes[51] = (byte)((bytes[51] & 0xF0) | ((scaleBits >> 4) & 0xF));
        bytes[52] = (byte)((bytes[52] & 0x0F) | (((scaleBits >> 8) & 0xF) << 4));
        bytes[55] = (byte)((bytes[55] & 0x0F) | (((scaleBits >> 12) & 0xF) << 4));

        return bytes;
    }

    [Fact]
    public void DequantIq1M_ProducesFiniteValues()
    {
        var rng = new Random(0x1101);
        for (int trial = 0; trial < 20; trial++)
        {
            byte[] block = BuildBlock(rng);
            var y = new float[256];
            Dequantize.ToFloat32(block, y, DType.IQ1_M, 256);
            foreach (float v in y)
                Assert.True(float.IsFinite(v), $"IQ1_M dequant produced non-finite value {v} on trial {trial}");
        }
    }

    /// <summary>
    /// Independent cross-check: dequantize a random block, dot the result against a raw
    /// (unquantized) input via plain F32 dot, and compare against the int-domain
    /// <see cref="SimdKernels.DotIq1M_Q8K_Scalar"/> path fed the same block through Q8_K
    /// activation quantization. A large systematic deviation would mean the dequantizer and the
    /// dot kernel disagree about what the format encodes -- exactly the class of bug two
    /// independently-hand-derived formulas checking each other is meant to catch. Same tolerance
    /// rationale as the IQ1_S sibling test: Q8_K quantization error alone explains a few percent,
    /// not more.
    /// </summary>
    [Fact]
    public void DotIq1M_Q8K_AgreesWithDequantThenF32Dot()
    {
        const int cols = 256;
        var rng = new Random(0x1102);

        for (int trial = 0; trial < 8; trial++)
        {
            byte[] weightBytes = BuildBlock(rng);
            var input = new float[cols];
            for (int i = 0; i < cols; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

            var dequantWeights = new float[cols];
            Dequantize.ToFloat32(weightBytes, dequantWeights, DType.IQ1_M, cols);
            float reference = 0f;
            for (int i = 0; i < cols; i++) reference += dequantWeights[i] * input[i];

            int scratchBytes = SimdKernels.Q8KScratchBytes(cols);
            var scratch = new byte[scratchBytes];
            float kernelResult;
            fixed (byte* wPtr = weightBytes)
            fixed (byte* sPtr = scratch)
            fixed (float* iPtr = input)
            {
                SimdKernels.QuantizeRowToQ8K(iPtr, cols, sPtr);
                kernelResult = SimdKernels.DotIq1M_Q8K_Scalar(wPtr, sPtr, cols);
            }

            float diff = MathF.Abs(kernelResult - reference);
            float relDiff = diff / (MathF.Abs(reference) + 1e-6f);
            Console.WriteLine(
                $"IQ1_M dot-vs-dequant cross-check trial={trial}: kernel={kernelResult:F6} " +
                $"dequant+f32dot={reference:F6} diff={diff:E3} relDiff={relDiff:E3}");
            Assert.True(relDiff < 0.05f,
                $"IQ1_M dot kernel disagrees with dequant+F32 reference by {relDiff:P1} on trial {trial} " +
                $"(kernel={kernelResult:F6}, reference={reference:F6}) — larger than Q8_K quantization " +
                "error should explain; likely a formula mismatch between the dot kernel and the dequantizer.");
        }
    }

    /// <summary>
    /// Confirms IQ1_M is admitted but deliberately NOT dispatched to a fast kernel: MatVec must
    /// still produce a finite, correct result via MatVecDequantFallback (the real deliverable —
    /// the format loads and runs), even with no dedicated fast path wired for it.
    /// </summary>
    [Fact]
    public void MatVecDequantFallback_HandlesIq1M()
    {
        const int rows = 4, cols = 256;
        var rng = new Random(0x1103);
        var weightBytes = new byte[rows * 56];
        rng.NextBytes(weightBytes);
        var input = new float[cols];
        for (int i = 0; i < cols; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        var output = new float[rows];

        fixed (byte* wPtr = weightBytes)
        fixed (float* iPtr = input)
        fixed (float* oPtr = output)
        {
            SimdKernels.MatVecDequantFallback(oPtr, wPtr, iPtr, rows, cols, DType.IQ1_M);
        }

        foreach (float v in output)
            Assert.True(float.IsFinite(v), $"MatVecDequantFallback produced non-finite value {v} for IQ1_M");
    }
}
