using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cuda;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Coordinates <paramref name="Degree"/> <see cref="CudaBackend"/> instances, one per GPU, for local
/// (single-node) tensor parallelism. <b>Never run on real multi-GPU hardware</b> -- this development
/// machine has no NVIDIA GPU at all, so nothing below this class's own unit-testable pieces
/// (<see cref="CudaTensorParallelSharding"/>) has been exercised against real CUDA. See
/// docs/4-performance/gpu/2026-11-06-cuda-tensor-parallelism-plan.md for scope and the "done" bar
/// this feature is held to instead of the project's usual real-weights bar.
///
/// <para>Each rank's backend is created and must be driven from its own dedicated thread: CUDA's
/// "current device" (<c>cudaSetDevice</c>) is thread-local, and <see cref="CudaBackend.Create(int,
/// SgemmPrecision?)"/> sets it once per thread at creation time rather than before every call. This
/// group does not attempt P2P or NCCL-style device-to-device collectives (TensorSharp's own stated
/// design, which this plan's Stage 1c explicitly defers): <see cref="AllReduceSumAcrossRanks"/> stages
/// every rank's partial result through host memory, which is correct regardless of P2P/NVLink
/// availability and is the only collective this class's logic can actually be checked here
/// (<see cref="CudaTensorParallelSharding.AllReduceSum"/> is unit tested).</para>
///
/// <para><b>Rank execution is sequential, not concurrent</b>, in this first slice: a forward pass
/// built on this group runs rank 0's shard of a layer, then rank 1's, ... rather than driving all
/// ranks' GPUs at once from separate threads. This gives up tensor parallelism's latency benefit
/// (meaningless to claim here anyway, with nothing to measure it on) in exchange for a dramatically
/// smaller, more reviewable surface than concurrent per-rank dispatch would need. Stage 1c's fused,
/// concurrent execution is explicitly out of scope for this plan.</para>
/// </summary>
public sealed class CudaTensorParallelGroup : IDisposable
{
    private readonly CudaBackend[] _backends;
    private bool _disposed;

    public int Degree => _backends.Length;

    /// <summary>
    /// Creates <paramref name="degree"/> ranks, one <see cref="CudaBackend"/> per device ordinal
    /// 0..degree-1. Each backend is created on its own thread so <c>cudaSetDevice</c>'s thread-local
    /// scoping pins it to that device for its whole lifetime (see the type doc).
    /// </summary>
    public CudaTensorParallelGroup(int degree, SgemmPrecision? precision = null)
    {
        if (degree < 2)
            throw new ArgumentOutOfRangeException(nameof(degree), degree, "Tensor parallelism needs at least 2 ranks; degree 1 is the ordinary single-GPU path.");

        var backends = new CudaBackend[degree];
        var exceptions = new Exception?[degree];
        var threads = new Thread[degree];
        for (int rank = 0; rank < degree; rank++)
        {
            int r = rank;
            threads[r] = new Thread(() =>
            {
                try { backends[r] = CudaBackend.Create(r, precision); }
                catch (Exception ex) { exceptions[r] = ex; }
            });
            threads[r].IsBackground = true;
            threads[r].Start();
        }
        foreach (var t in threads) t.Join();

        var failures = exceptions.Where(e => e is not null).ToArray();
        if (failures.Length > 0)
        {
            foreach (var b in backends) b?.Dispose();
            throw new AggregateException($"Failed to create {degree} tensor-parallel CUDA ranks.", failures!);
        }

        _backends = backends;
    }

    /// <summary>The backend for a given rank (0..<see cref="Degree"/>-1).</summary>
    public CudaBackend Backend(int rank)
    {
        if ((uint)rank >= (uint)_backends.Length)
            throw new ArgumentOutOfRangeException(nameof(rank), rank, $"Rank must be in [0, {_backends.Length}).");
        return _backends[rank];
    }

    /// <summary>
    /// Host-staged AllReduce-sum across every rank's own copy of a same-shaped buffer: downloads each
    /// rank's <paramref name="perRankBuffers"/> entry, sums on the host
    /// (<see cref="CudaTensorParallelSharding.AllReduceSum"/>), then re-uploads the sum into every
    /// rank's buffer in place. Call this after a row-parallel projection (attention output, FFN down)
    /// so every rank ends up with the full, reconverged result for its residual add.
    /// </summary>
    public void AllReduceSumAcrossRanks(Tensor[] perRankBuffers, int elementCount)
    {
        if (perRankBuffers.Length != Degree)
            throw new ArgumentException($"Expected one buffer per rank ({Degree}), got {perRankBuffers.Length}.", nameof(perRankBuffers));

        var partials = new float[Degree][];
        for (int rank = 0; rank < Degree; rank++)
        {
            partials[rank] = new float[elementCount];
            _backends[rank].Download(perRankBuffers[rank], partials[rank]);
        }

        float[] sum = CudaTensorParallelSharding.AllReduceSum(partials);

        for (int rank = 0; rank < Degree; rank++)
            _backends[rank].UploadInto(perRankBuffers[rank], sum);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var backend in _backends) backend.Dispose();
    }
}
