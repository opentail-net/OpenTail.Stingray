using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cuda;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// A per-sequence GPU KV cache for CUDA continuous batching (issue #190, step 1). Unlike the
/// single owned cache hardwired into <see cref="CudaForwardPass"/>, this is a detachable bundle
/// of per-layer device K/V tensors that the continuous-batching engine allocates once per
/// admitted sequence (via <c>CudaForwardPass.CreateCache</c>) and threads back through the
/// batched forward-pass calls as an opaque <see cref="ISequenceKvCache"/>.
///
/// The tensors are allocated by the forward pass, which owns the geometry (per-layer head_dim,
/// SWA ring sizing, k_eq_v / shared-KV aliasing, and the KV dtype) — this type just holds them
/// and frees the non-aliased ones on dispose, mirroring <see cref="CudaForwardPass"/>'s own
/// KV-cache teardown. <see cref="Length"/> is the logical token count the forward pass advances
/// as it appends; it lives here (not on the forward pass) so each sequence tracks its own.
/// </summary>
internal sealed class CudaSequenceKvCache : ISequenceKvCache, IKvSequence
{
    private static long s_sequenceIdCounter;

    private readonly CudaBackend _gpu;
    private readonly long _sequenceId;
    private KvPageId[] _virtualPages = Array.Empty<KvPageId>();

    /// <summary>Per-layer key cache. Aliased layers (see <see cref="_aliasedLayers"/>) share the
    /// source layer's tensor instance, exactly as the forward pass's owned cache does.</summary>
    public Tensor[] K { get; }

    /// <summary>Per-layer value cache (aliased layers share the source layer's tensor).</summary>
    public Tensor[] V { get; }

    /// <summary>Layers whose K/V alias a source layer (Gemma 4 shared-KV tail) — never freed here.</summary>
    private readonly IReadOnlySet<int> _aliasedLayers;

    /// <summary>Physical KV rows stored so far (the forward pass reads/advances this). Equals the
    /// logical token count unless SnapKV evicted this sequence's prompt, in which case it is the
    /// post-compaction count and <see cref="EvictedCount"/> bridges back to logical positions.</summary>
    public int Length;

    /// <summary>Logical-minus-physical delta from a SnapKV prefill eviction (issue #196): the
    /// number of prompt positions dropped during compaction. The batched decode maps a logical
    /// position <c>p</c> to the physical slot <c>p - EvictedCount</c> for KV append + attention,
    /// while RoPE keeps the logical <c>p</c> (the kept K rows retain their original RoPE phase).
    /// 0 for a sequence that was never evicted (physical slot == logical position).</summary>
    public int EvictedCount;

    private bool _disposed;

    public CudaSequenceKvCache(CudaBackend gpu, Tensor[] k, Tensor[] v, IReadOnlySet<int> aliasedLayers)
    {
        ArgumentNullException.ThrowIfNull(gpu);
        ArgumentNullException.ThrowIfNull(k);
        ArgumentNullException.ThrowIfNull(v);
        ArgumentNullException.ThrowIfNull(aliasedLayers);
        _gpu = gpu;
        K = k;
        V = v;
        _aliasedLayers = aliasedLayers;
        Length = 0;
        _sequenceId = System.Threading.Interlocked.Increment(ref s_sequenceIdCounter);
    }

    public long SequenceId => _sequenceId;
    public int TokenCount => Length;
    public int CapacityTokens => K.Length > 0 && K[0].Shape.Dims.Length > 0 ? (int)K[0].Shape.Dims[0] : int.MaxValue;
    public int PageSize => 32;
    public int PageCount => KvPageMath.GetRequiredPageCount(Length, 32);

    public ReadOnlySpan<KvPageId> Pages
    {
        get
        {
            int req = PageCount;
            if (_virtualPages.Length != req)
            {
                var newPages = new KvPageId[req];
                for (int i = 0; i < req; i++) newPages[i] = new KvPageId(i);
                _virtualPages = newPages;
            }
            return _virtualPages;
        }
    }

    public void Append(int tokenCount)
    {
        if (tokenCount < 0) throw new ArgumentOutOfRangeException(nameof(tokenCount));
        if (tokenCount > CapacityTokens - Length)
            throw new InvalidOperationException($"CUDA KV cache capacity exceeded: requested {Length + tokenCount}, capacity {CapacityTokens}.");
        Length += tokenCount;
    }

    public void TruncateTo(int tokenCount)
    {
        if (tokenCount < 0 || tokenCount > Length) throw new ArgumentOutOfRangeException(nameof(tokenCount));
        Length = tokenCount;
    }

    public IKvSequence Fork() => ForkAt(Length);
    public IKvSequence ForkAt(int tokenCount)
    {
        throw new NotSupportedException($"CudaSequenceKvCache does not support zero-copy forking without CpuKvCache or paged CUDA allocator (requested position {tokenCount}).");
    }
    public void Release() => Dispose();
    public void Clear() { Length = 0; EvictedCount = 0; }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int i = 0; i < K.Length; i++)
        {
            // Skip aliased layers: their handles are owned by the source layer, which frees
            // them on its own iteration. A second Free() on the same device pointer would be a
            // CUDA double-free (same guard as CudaForwardPass.Dispose).
            if (_aliasedLayers.Contains(i)) continue;
            _gpu.Free(K[i]);
            _gpu.Free(V[i]);
        }
    }
}
