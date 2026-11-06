using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Pure weight-sharding and AllReduce math for CUDA tensor parallelism
/// (<see cref="CudaTensorParallelGroup"/>), kept free of any CUDA/GPU dependency so it can be unit
/// tested on this machine without hardware — see
/// docs/4-performance/gpu/2026-11-06-cuda-tensor-parallelism-plan.md. The GPU-touching layer above
/// this (device-indexed <c>CudaBackend</c> instances, actual kernel dispatch) cannot be run or tested
/// here (no NVIDIA hardware); this class is deliberately where the alignment/correctness-sensitive
/// arithmetic lives instead, so at least that part is real, checked logic rather than unverified code.
///
/// <para>Megatron-style tensor parallelism splits a weight matrix one of two ways (TensorSharp's
/// documented convention, matched here): <b>column-parallel</b> splits the OUTPUT dimension (GGUF
/// ne1, the row count of a row-major tensor) — each rank gets a contiguous, disjoint run of whole
/// rows, which is a zero-copy byte-range slice because GGUF tensors are row-major. <b>Row-parallel</b>
/// splits the INPUT/contraction dimension (GGUF ne0, the column count) — each rank needs part of
/// EVERY row, which is not contiguous in memory and must be copied out block by block, respecting the
/// quantization format's block size so a block is never split across ranks.</para>
/// </summary>
public static class CudaTensorParallelSharding
{
    /// <summary>
    /// Validates that <paramref name="rows"/> divides evenly across <paramref name="tpDegree"/> ranks.
    /// Column-parallel splitting needs no block alignment (whole rows are atomic), but it does need
    /// an even rank count or one rank silently gets a different shape than the others.
    /// </summary>
    public static long RowsPerRank(long rows, int tpDegree, string tensorName)
    {
        if (tpDegree < 1) throw new ArgumentOutOfRangeException(nameof(tpDegree), tpDegree, "Tensor-parallel degree must be >= 1.");
        if (rows % tpDegree != 0)
            throw new InvalidOperationException(
                $"Tensor '{tensorName}' has {rows} output rows, which does not divide evenly across {tpDegree} tensor-parallel ranks.");
        return rows / tpDegree;
    }

    /// <summary>
    /// Validates that <paramref name="cols"/> divides evenly across <paramref name="tpDegree"/> ranks
    /// AND that each rank's share is itself a whole number of quantization blocks (a block is never
    /// split across ranks — splitting one would corrupt its shared scale/min, not just its payload).
    /// </summary>
    public static long ColsPerRank(long cols, DType dtype, int tpDegree, string tensorName)
    {
        if (tpDegree < 1) throw new ArgumentOutOfRangeException(nameof(tpDegree), tpDegree, "Tensor-parallel degree must be >= 1.");
        if (cols % tpDegree != 0)
            throw new InvalidOperationException(
                $"Tensor '{tensorName}' has {cols} input columns, which does not divide evenly across {tpDegree} tensor-parallel ranks.");
        long colsPerRank = cols / tpDegree;
        int blockSize = DTypeInfo.BlockSize(dtype);
        if (colsPerRank % blockSize != 0)
            throw new InvalidOperationException(
                $"Tensor '{tensorName}' ({dtype}, block size {blockSize}): {colsPerRank} columns per rank at tp={tpDegree} " +
                $"is not a whole number of quantization blocks. Splitting a block across ranks would corrupt its shared scale.");
        return colsPerRank;
    }

    /// <summary>
    /// Column-parallel shard: a contiguous, zero-copy-able byte range covering
    /// <c>rank</c>'s disjoint run of whole output rows (<paramref name="rows"/> is the tensor's ne1).
    /// </summary>
    public static ReadOnlySpan<byte> ColumnParallelSlice(
        ReadOnlySpan<byte> rowMajorData, long rows, long cols, DType dtype, int rank, int tpDegree, string tensorName)
    {
        long rowsPerRank = RowsPerRank(rows, tpDegree, tensorName);
        long rowByteSize = DTypeInfo.ByteSize(cols, dtype);
        long startByte = rank * rowsPerRank * rowByteSize;
        long shardBytes = rowsPerRank * rowByteSize;
        if (startByte + shardBytes > rowMajorData.Length)
            throw new InvalidOperationException(
                $"Tensor '{tensorName}': computed shard range [{startByte}, {startByte + shardBytes}) exceeds the " +
                $"tensor's {rowMajorData.Length} bytes (rows={rows}, cols={cols}, dtype={dtype}).");
        return rowMajorData.Slice((int)startByte, (int)shardBytes);
    }

    /// <summary>
    /// Row-parallel shard: for every one of <paramref name="rows"/> rows, copies out <c>rank</c>'s
    /// block-aligned sub-range of that row's columns into a new, tightly packed buffer (not a
    /// zero-copy view — each row's shard is non-contiguous in the source).
    /// </summary>
    public static byte[] RowParallelSlice(
        ReadOnlySpan<byte> rowMajorData, long rows, long cols, DType dtype, int rank, int tpDegree, string tensorName)
    {
        long colsPerRank = ColsPerRank(cols, dtype, tpDegree, tensorName);
        long rowByteSize = DTypeInfo.ByteSize(cols, dtype);
        long shardRowByteSize = DTypeInfo.ByteSize(colsPerRank, dtype);
        long startByteInRow = rank * shardRowByteSize;

        if ((long)rows * rowByteSize != rowMajorData.Length)
            throw new InvalidOperationException(
                $"Tensor '{tensorName}': expected {rows * rowByteSize} bytes (rows={rows}, cols={cols}, dtype={dtype}), got {rowMajorData.Length}.");

        var result = new byte[rows * shardRowByteSize];
        for (long r = 0; r < rows; r++)
        {
            var srcRow = rowMajorData.Slice((int)(r * rowByteSize + startByteInRow), (int)shardRowByteSize);
            srcRow.CopyTo(result.AsSpan((int)(r * shardRowByteSize), (int)shardRowByteSize));
        }
        return result;
    }

    /// <summary>
    /// Host-staged AllReduce-sum: element-wise sum of every rank's partial result, correctness-first
    /// (no P2P/NCCL — see the type doc). This is the mechanism <see cref="CudaTensorParallelGroup"/>
    /// uses after a row-parallel projection (attention output, FFN down) to reconverge the sharded
    /// partials into the single value every rank needs for its residual add.
    /// </summary>
    public static float[] AllReduceSum(ReadOnlySpan<float[]> perRankPartials)
    {
        if (perRankPartials.Length == 0) throw new ArgumentException("At least one rank's partial result is required.", nameof(perRankPartials));
        int length = perRankPartials[0].Length;
        var sum = new float[length];
        foreach (var partial in perRankPartials)
        {
            if (partial.Length != length)
                throw new InvalidOperationException(
                    $"AllReduceSum: rank partial lengths disagree ({partial.Length} vs {length}) -- every rank must produce the same-shaped partial.");
            for (int i = 0; i < length; i++)
                sum[i] += partial[i];
        }
        return sum;
    }
}
