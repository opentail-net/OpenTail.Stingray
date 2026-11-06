using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// <see cref="CudaTensorParallelSharding"/> is deliberately pure (no CUDA dependency) so its
/// alignment/correctness-sensitive arithmetic can be checked here without the GPU hardware this
/// project doesn't have — see docs/4-performance/gpu/2026-11-06-cuda-tensor-parallelism-plan.md.
/// </summary>
public sealed class CudaTensorParallelShardingTests
{
    [Fact]
    public void ColumnParallelSlice_F32_SlicesContiguousRows()
    {
        // rows=4, cols=8, F32: each row is 32 bytes (8 floats). tp=2 -> rank 0 gets rows 0-1, rank 1 rows 2-3.
        const long rows = 4, cols = 8;
        var data = new byte[rows * cols * sizeof(float)];
        var floats = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(data.AsSpan());
        for (int i = 0; i < floats.Length; i++) floats[i] = i;

        var rank0 = CudaTensorParallelSharding.ColumnParallelSlice(data, rows, cols, DType.Float32, rank: 0, tpDegree: 2, "t");
        var rank1 = CudaTensorParallelSharding.ColumnParallelSlice(data, rows, cols, DType.Float32, rank: 1, tpDegree: 2, "t");

        Assert.Equal(2 * (int)cols * sizeof(float), rank0.Length);
        Assert.Equal(2 * (int)cols * sizeof(float), rank1.Length);
        var rank0Floats = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(rank0);
        var rank1Floats = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(rank1);
        Assert.Equal([0f, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], rank0Floats.ToArray());
        Assert.Equal([16f, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], rank1Floats.ToArray());
    }

    [Fact]
    public void ColumnParallelSlice_TpDegreeOne_ReturnsTheWholeTensorUnchanged()
    {
        // The N=1 path must be provably identical to no sharding at all (plan requirement).
        var data = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var slice = CudaTensorParallelSharding.ColumnParallelSlice(data, rows: 2, cols: 4, DType.Int8, rank: 0, tpDegree: 1, "t");
        Assert.Equal(data, slice.ToArray());
    }

    [Fact]
    public void ColumnParallelSlice_RowsNotDivisible_Throws()
    {
        var data = new byte[3 * 4 * sizeof(float)];
        var ex = Assert.Throws<InvalidOperationException>(() =>
            CudaTensorParallelSharding.ColumnParallelSlice(data, rows: 3, cols: 4, DType.Float32, rank: 0, tpDegree: 2, "odd_rows"));
        Assert.Contains("odd_rows", ex.Message);
    }

    [Fact]
    public void RowParallelSlice_Q8_0_ExtractsBlockAlignedColumnsPerRow()
    {
        // rows=2, cols=64 (two Q8_0 blocks of 32 cols each, 34 bytes/block). tp=2 -> each rank gets
        // exactly one block per row. Fill each block with a distinctive byte value so misrouting a
        // block (wrong row, wrong rank) is visible immediately.
        const long rows = 2, cols = 64;
        const int bytesPerBlock = 34; // DTypeInfo.BytesPerBlock(Q8_0)
        var data = new byte[rows * 2 * bytesPerBlock];
        // row 0: block A = 0xA0, block B = 0xB0. row 1: block A = 0xA1, block B = 0xB1.
        Array.Fill(data, (byte)0xA0, 0 * bytesPerBlock, bytesPerBlock);
        Array.Fill(data, (byte)0xB0, 1 * bytesPerBlock, bytesPerBlock);
        Array.Fill(data, (byte)0xA1, 2 * bytesPerBlock, bytesPerBlock);
        Array.Fill(data, (byte)0xB1, 3 * bytesPerBlock, bytesPerBlock);

        var rank0 = CudaTensorParallelSharding.RowParallelSlice(data, rows, cols, DType.Q8_0, rank: 0, tpDegree: 2, "t");
        var rank1 = CudaTensorParallelSharding.RowParallelSlice(data, rows, cols, DType.Q8_0, rank: 1, tpDegree: 2, "t");

        Assert.Equal(rows * bytesPerBlock, rank0.Length);
        Assert.Equal(rows * bytesPerBlock, rank1.Length);
        // rank 0 = block A of every row: [0xA0 x34, 0xA1 x34]
        Assert.All(rank0.AsSpan(0, bytesPerBlock).ToArray(), b => Assert.Equal(0xA0, b));
        Assert.All(rank0.AsSpan(bytesPerBlock, bytesPerBlock).ToArray(), b => Assert.Equal(0xA1, b));
        // rank 1 = block B of every row: [0xB0 x34, 0xB1 x34]
        Assert.All(rank1.AsSpan(0, bytesPerBlock).ToArray(), b => Assert.Equal(0xB0, b));
        Assert.All(rank1.AsSpan(bytesPerBlock, bytesPerBlock).ToArray(), b => Assert.Equal(0xB1, b));
    }

    [Fact]
    public void RowParallelSlice_NotBlockAligned_Throws()
    {
        // cols=32 is exactly one Q8_0 block; splitting it across 2 ranks would split the block.
        var data = new byte[1 * 34];
        var ex = Assert.Throws<InvalidOperationException>(() =>
            CudaTensorParallelSharding.RowParallelSlice(data, rows: 1, cols: 32, DType.Q8_0, rank: 0, tpDegree: 2, "one_block"));
        Assert.Contains("one_block", ex.Message);
        Assert.Contains("block", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void AllReduceSum_SumsElementwise()
    {
        float[][] ranks = [[1f, 2f, 3f], [10f, 20f, 30f], [100f, 200f, 300f]];
        var sum = CudaTensorParallelSharding.AllReduceSum(ranks);
        Assert.Equal([111f, 222f, 333f], sum);
    }

    [Fact]
    public void AllReduceSum_MismatchedLengths_Throws()
    {
        float[][] ranks = [[1f, 2f], [1f, 2f, 3f]];
        Assert.Throws<InvalidOperationException>(() => CudaTensorParallelSharding.AllReduceSum(ranks));
    }

    [Fact]
    public void AllReduceSum_EmptyInput_Throws()
    {
        Assert.Throws<ArgumentException>(() => CudaTensorParallelSharding.AllReduceSum([]));
    }
}
