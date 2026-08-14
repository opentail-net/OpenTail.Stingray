using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Correctness gate for <see cref="SimdKernels.TryMatMulBatchedDualQ8"/> (perf-loop-progress.md
/// iteration 13): shares one Q8 quantization pass across two weight matrices (gate+up) instead
/// of <see cref="SimdKernels.TryMatMulBatchedQ8"/> being called twice. The quantize/dot8/dot4/dot1
/// delegates themselves are completely unchanged (same `TryResolveQ8Dispatch`) -- only the
/// dispatch/scratch-reuse STRUCTURE differs, so unlike a genuinely new arithmetic kernel, the
/// correctness bar here is exact agreement with calling the already-deeply-trusted
/// `TryMatMulBatchedQ8` twice on the same inputs (bit-identical expected, not just tolerance --
/// same quantize call, same dot kernel, same floating-point operations, just fewer redundant
/// calls to reach them).
/// </summary>
public sealed unsafe class MatMulBatchedDualQ8Tests : HeavyTestBase
{
    [Theory]
    [InlineData(1, 2048, 2048)]   // batch=1 edge case
    [InlineData(2, 2048, 2048)]   // batch=2, exercises the tail-only path (no groupsOf8/4)
    [InlineData(3, 2048, 2048)]   // batch=3, same tail-only path, different remainder
    [InlineData(4, 2048, 2048)]   // batch=4, exactly one groupsOf4, zero tail -- real repro shape
    [InlineData(5, 2048, 2048)]   // batch=5, exercises groupsOf4=1 + 1 tail
    [InlineData(7, 2048, 2048)]   // batch=7, groupsOf4=1 + 3 tail -- real repro shape
    [InlineData(9, 2048, 2048)]   // batch>8, exercises the groupsOf8+groupsOf4+tail split
    [InlineData(17, 2048, 2048)]  // batch=17, groupsOf8=2 + groupsOf4=0 + 1 tail
    [InlineData(256, 2048, 2048)] // real QKV/O-shape batch
    [InlineData(256, 8192, 2048)] // real FFN gate/up shape (SmolLM2: rows=intermDim, cols=embDim)
    [InlineData(4, 8192, 2048)]   // small batch AT the real FFN gate/up shape -- exact repro combo
    [InlineData(7, 8192, 2048)]   // small batch AT the real FFN gate/up shape -- exact repro combo
    [InlineData(600, 64, 256)]    // exercises the >512-token chunking path
    public void TryMatMulBatchedDualQ8_MatchesTwoSeparateTryMatMulBatchedQ8Calls(int batchSize, int rows, int cols)
    {
        var rng = new Random(97531);
        int bytesPerRow = (cols / 256) * 144; // Q4_K
        byte[] weights1 = new byte[(long)rows * bytesPerRow];
        byte[] weights2 = new byte[(long)rows * bytesPerRow];
        rng.NextBytes(weights1);
        rng.NextBytes(weights2);
        for (int w = 0; w < 2; w++)
        {
            var weights = w == 0 ? weights1 : weights2;
            for (int r = 0; r < rows; r++)
            {
                int off = r * bytesPerRow;
                var dBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
                var dminBits = BitConverter.GetBytes((Half)(0.001 + rng.NextDouble() * 0.05));
                weights[off] = dBits[0]; weights[off + 1] = dBits[1];
                weights[off + 2] = dminBits[0]; weights[off + 3] = dminBits[1];
            }
        }

        float[] input = new float[(long)batchSize * cols];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] expected1 = new float[(long)batchSize * rows];
        float[] expected2 = new float[(long)batchSize * rows];
        float[] actual1 = new float[(long)batchSize * rows];
        float[] actual2 = new float[(long)batchSize * rows];

        fixed (byte* w1 = weights1)
        fixed (byte* w2 = weights2)
        fixed (float* inp = input)
        fixed (float* e1 = expected1)
        fixed (float* e2 = expected2)
        fixed (float* a1 = actual1)
        fixed (float* a2 = actual2)
        {
            bool ok1 = SimdKernels.TryMatMulBatchedQ8(e1, w1, inp, batchSize, rows, cols, DType.Q4_K);
            bool ok2 = SimdKernels.TryMatMulBatchedQ8(e2, w2, inp, batchSize, rows, cols, DType.Q4_K);
            Assert.True(ok1 && ok2, "Baseline TryMatMulBatchedQ8 calls must succeed for Q4_K.");

            bool okDual = SimdKernels.TryMatMulBatchedDualQ8(a1, w1, a2, w2, inp, batchSize, rows, cols, DType.Q4_K);
            Assert.True(okDual, "TryMatMulBatchedDualQ8 must succeed for Q4_K.");

            for (int i = 0; i < expected1.Length; i++)
                Assert.Equal(expected1[i], actual1[i]); // bit-identical: same quantize+dot, no tolerance needed
            for (int i = 0; i < expected2.Length; i++)
                Assert.Equal(expected2[i], actual2[i]);
        }
    }

    [Fact]
    public void TryMatMulBatchedDualQ8_ReturnsFalse_ForUnsupportedDtype()
    {
        const int rows = 64, cols = 256, batchSize = 4;
        byte[] weights1 = new byte[rows * cols * sizeof(float)];
        byte[] weights2 = new byte[rows * cols * sizeof(float)];
        float[] input = new float[batchSize * cols];
        float[] out1 = new float[batchSize * rows];
        float[] out2 = new float[batchSize * rows];

        fixed (byte* w1 = weights1)
        fixed (byte* w2 = weights2)
        fixed (float* inp = input)
        fixed (float* o1 = out1)
        fixed (float* o2 = out2)
        {
            // Float32 has no Q8 dot support (per TryResolveQ8Dispatch's default branch) -- must
            // return false, not write garbage, matching TryMatMulBatchedQ8's own contract.
            bool ok = SimdKernels.TryMatMulBatchedDualQ8(o1, w1, o2, w2, inp, batchSize, rows, cols, DType.Float32);
            Assert.False(ok);
        }
    }
}
