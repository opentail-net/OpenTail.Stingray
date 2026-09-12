namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Verifies <see cref="OpenTail.Stingray.Core.FastVectorTypeConverter.ConvertF32ToQ8_0"/> against
/// a real matvec, ahead of perf-sweep Phase 1.2 (docs/perf-sweep-plan.md -- quantizing
/// Voxtral-Mini-4B-Realtime's weights from F32 to Q8_0). This converter's own doc comment flags it
/// as previously removed from every product path for being unverified ("a quantizer that runs and
/// is subtly wrong is worse than either") -- this test is that verification, done once, before
/// wiring it into any real pipeline.
/// </summary>
public sealed unsafe class ConvertF32ToQ8_0VerificationTests
{
    [Fact]
    public void ConvertF32ToQ8_0_QuantizedMatVec_MatchesF32MatVecWithinToleratedQuantizationError()
    {
        const int rows = 64;
        const int cols = 320; // multiple of 32, real Voxtral hidden-size-scale shape
        var rng = new Random(unchecked((int)0xC0FFEE));

        var weightsF32 = new float[rows * cols];
        for (var i = 0; i < weightsF32.Length; i++) weightsF32[i] = (float)(rng.NextDouble() * 2 - 1);

        var input = new float[cols];
        for (var i = 0; i < cols; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        var weightsQ8 = new byte[rows * (cols / 32) * 34];
        for (var r = 0; r < rows; r++)
            OpenTail.Stingray.Core.FastVectorTypeConverter.ConvertF32ToQ8_0(
                weightsF32.AsSpan(r * cols, cols), weightsQ8.AsSpan(r * (cols / 32) * 34, (cols / 32) * 34));

        var expected = new float[rows];
        var actual = new float[rows];
        fixed (float* pOutF32 = expected, pW = weightsF32, pIn = input, pOutQ8 = actual)
        fixed (byte* pWQ8 = weightsQ8)
        {
            OpenTail.Stingray.Cpu.SimdKernels.MatVecF32(pOutF32, pW, pIn, rows, cols);
            OpenTail.Stingray.Cpu.SimdKernels.MatVecQ8_0(pOutQ8, pWQ8, pIn, rows, cols);
        }

        // Q8_0 is a real lossy 8-bit-per-weight quantization (llama.cpp's own format), so
        // per-row relative error is a noisy quantity -- a single row can exceed any fixed
        // per-row bound by chance even for a genuinely correct quantizer (dot-product
        // cancellation on random data). Aggregate RMS relative error across all rows is the
        // statistically meaningful check: it stays small (~1-2%) for a correct converter and
        // blows up by 10x+ for a genuinely broken one (wrong scale, wrong byte layout,
        // off-by-one block indexing).
        double sumSqErr = 0, sumSqExpected = 0;
        for (var r = 0; r < rows; r++)
        {
            double diff = actual[r] - expected[r];
            sumSqErr += diff * diff;
            sumSqExpected += (double)expected[r] * expected[r];
        }
        double rmsRelError = Math.Sqrt(sumSqErr / sumSqExpected);
        Assert.True(rmsRelError < 0.05,
            $"RMS relative error {rmsRelError:P2} across {rows} rows exceeds the 5% quantization-noise budget -- likely a real converter bug, not sampling noise.");
    }
}
