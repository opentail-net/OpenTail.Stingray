using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Paged KV cache: dynamically allocates fixed-size page blocks as tokens are appended,
/// eliminating pre-allocation of the full context window.
///
/// Memory model:
/// - Pages are lazily allocated (NativeMemory) on first write to each block.
/// - <see cref="TruncateTo"/> is a "soft" truncate — it moves the length pointer without freeing pages.
///   Pages beyond the new length are reused on the next write (used during batched prefill
///   per-layer resets and speculative decoding rewinds).
/// - <see cref="Reset"/> returns all allocated page slots to the warm pool for reuse across requests.
///
/// All layers share the same physical slot index per block, so a single free list manages
/// all layer pages together. Total memory at any point:
///   allocatedBlocks × numLayers × PageSize × kvDim × 2 × sizeof(float)
/// </summary>
public sealed unsafe class PagedKvCache : IRewindableSequenceKvCache, IPersistableSequenceKvCache
{
    /// <summary>
    /// When <c>STINGRAY_KV_DTYPE=bf16</c>, K and V are rounded to BF16 precision on write while
    /// still being STORED as F32.
    /// </summary>
    /// <remarks>
    /// <para>This exists to gate the quality question independently of the storage change. Narrowing
    /// the cache to 16-bit is a two-part proposition — "is BF16 precision good enough?" and "is a
    /// 16-bit store worth implementing?" — and only the first needs answering before the second is
    /// worth the work. Rounding on write reproduces the exact numerical effect of a BF16 cache with
    /// no change to layout, readers, prefix capture, or snapshot/rewind.</para>
    ///
    /// <para>It therefore saves no memory and is <b>not</b> the shipping form; it is a measurement
    /// scaffold. The real change stores 2 bytes and widens on read (a zero-extend plus a 16-bit
    /// shift — BF16 is the top half of an F32, so no F16C is needed, which matters because .NET
    /// exposes none).</para>
    ///
    /// <para>Rounding is round-to-nearest-even, matching what a real conversion would do. Truncation
    /// would have been a more conservative bound but would understate quality and risk rejecting a
    /// change that is actually fine.</para>
    ///
    /// <para><b>Cost of the storage change — the earlier figure here was measured in the wrong
    /// regime and is corrected (2026-08-04).</b> This remark used to state that BF16 attention is
    /// ~8% SLOWER because it is "ALU/load-port bound, not bandwidth bound", on the strength of a
    /// scratchpad benchmark over ONE head at 2431 positions, headDim 64. That working set is
    /// 622 KB — it fits in L2, so it measured the in-cache regime, where the finding is broadly
    /// right (BF16 is roughly neutral there). Decode does not run in that regime: it streams every
    /// head of every layer, 1.11 GiB at 3K context on SmolLM2, ~35x this machine's L3. Re-measured
    /// with the working set above L3, BF16 is <b>1.44-1.50x FASTER</b>, because the halved DRAM
    /// traffic dominates the widen ops (<c>vpmovzxwd</c> + <c>vpslld</c>). The per-8-float uop
    /// count in the old note is accurate; the conclusion drawn from it was not.</para>
    ///
    /// <para>See <see cref="Bf16StoreRequested"/> for the real 2-byte store. Memory saving is
    /// unchanged: 3.0 GiB -> 1.5 GiB at 8192 context for this model.</para>
    /// </remarks>
    private static readonly bool s_kvBf16 = string.Equals(
        Environment.GetEnvironmentVariable("STINGRAY_KV_DTYPE"), "bf16", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// The real 2-byte KV store: <c>STINGRAY_KV_STORE=bf16</c>. Pages hold BF16, so the cache
    /// costs half the bytes AND moves half the bytes; readers widen on load.
    /// </summary>
    /// <remarks>
    /// <para>Distinct from <see cref="s_kvBf16"/>, which rounds to BF16 precision while still
    /// storing F32 — that flag is a quality scaffold and buys zero bytes. This one changes the
    /// layout. Both can be on; the rounding is then redundant but harmless.</para>
    ///
    /// <para><b>Why this is worth a layout change.</b> Decode is memory-bound: measured end-to-end
    /// on SmolLM2-1.7B-Instruct-Q4_K_M, throughput matches a constant ~26-28 GB/s across a 2.1x
    /// change in bytes moved per token (26.2 t/s at 0.98 GiB short-context, 13.2 t/s at 2.09 GiB
    /// with 3031 tokens of context). That model has no GQA (32 KV heads for 32 heads), so KV is
    /// 384 KiB/token and is <b>53% of all bytes read</b> at 3K context. Halving it predicts ~1.36x
    /// decode. Arithmetic stays fp32 throughout — only the store narrows.</para>
    ///
    /// <para><b>Scope.</b> Only the dense CPU attention path (<c>ForwardPass</c>) reads BF16 pages
    /// natively. Every other reader — hybrid, CUDA-hybrid, SnapKV, TurboQuant — still expects
    /// <see cref="KeyAt"/>'s F32 pointer, so those accessors throw rather than silently reinterpret
    /// 2-byte data as 4-byte. That is the honest failure: a wrong-format read would produce
    /// plausible garbage, which is the expensive kind of bug.</para>
    ///
    /// <para>Read by the DENSE CPU forward pass only, which passes it to the constructor. It is
    /// deliberately not consulted here: the hybrid, CUDA-hybrid and Vulkan-hybrid passes also
    /// construct a <see cref="PagedKvCache"/>, and if this were a global default they would silently
    /// get BF16 pages and then throw on their first <see cref="KeyAt"/>. An env var must not be able
    /// to break a model family that has no BF16 reader — opting in is the caller's decision.</para>
    /// </remarks>
    public static bool Bf16StoreRequested { get; } = string.Equals(
        Environment.GetEnvironmentVariable("STINGRAY_KV_STORE"), "bf16", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// <c>STINGRAY_KV_STORE=auto</c>: start F32 and narrow the cache to BF16 once the sequence is
    /// long enough that the halved DRAM traffic outweighs the widen ops.
    /// </summary>
    /// <remarks>
    /// The crossover was measured on SmolLM2-1.7B-Q4_K_M (no GQA, so KV is 384 KiB/token): BF16
    /// decode is 0.83x at 286 tokens, 1.02x at 1058, 1.23x at 1989, 1.34x at 4002, 1.47x at 5314.
    /// Below the crossover the KV fits in L3, halving DRAM traffic buys nothing, and the widen ops
    /// are pure cost. This mode exists because the format is otherwise fixed at construction, where
    /// the only signal available is the CONFIGURED context (<c>-c</c>) — which defaults to the model
    /// maximum and says nothing about how much of it a request will use.
    /// </remarks>
    public static bool Bf16AutoRequested { get; } = string.Equals(
        Environment.GetEnvironmentVariable("STINGRAY_KV_STORE"), "auto", StringComparison.OrdinalIgnoreCase);

    /// <summary>Sequence length at which auto mode narrows. Default 1024 — the measured crossover.</summary>
    public static int Bf16AutoMinTokens { get; } =
        int.TryParse(Environment.GetEnvironmentVariable("STINGRAY_KV_BF16_MIN_TOKENS"), out int v) && v > 0
            ? v : 1024;

    private bool _bf16Store;
    private readonly bool _autoBf16;
    private static long s_bf16Conversions;

    /// <summary>Number of caches that have narrowed to BF16 in auto mode. Lets a null A/B result be
    /// distinguished from "the conversion never fired" — a failure mode that has cost real time here.</summary>
    public static long Bf16Conversions => Interlocked.Read(ref s_bf16Conversions);

    /// <summary>True when pages hold BF16 rather than F32. Readers must use
    /// <see cref="Bf16KeyAt"/>/<see cref="Bf16ValueAtHead"/>; the F32 accessors throw.</summary>
    public bool IsBf16Store => _bf16Store;

    /// <summary>Counts BF16-rounded appends, so a null quality result can be distinguished from
    /// "the rounding never ran" — a failure mode that has cost real time on this codebase.</summary>
    private static long s_kvBf16Appends;

    /// <summary>Number of K/V vectors written through the BF16 rounding path.</summary>
    public static long Bf16RoundedAppends => Interlocked.Read(ref s_kvBf16Appends);

    /// <summary>Round-to-nearest-even conversion of an F32 to BF16 precision, returned as F32.</summary>
    private static float ToBf16Precision(float f)
    {
        uint u = BitConverter.SingleToUInt32Bits(f);
        // NaN must not be turned into a different NaN payload or an infinity by the rounding add.
        if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x007FFFFFu) != 0) return f;
        u += 0x7FFFu + ((u >> 16) & 1u);
        return BitConverter.UInt32BitsToSingle(u & 0xFFFF0000u);
    }

    /// <summary>Number of KV positions stored per page block.</summary>
    public const int PageSize = 16;

    private readonly int _numLayers;
    private readonly int _kvDim;
    private readonly int _numKvHeads;
    private readonly int _headDim;

    // Per-layer V head-plane stride. See the constructor's layerHeadDim parameter; null = uniform.
    private readonly int[]? _layerHeadDim;
    private readonly int _maxBlocks;
    // Not readonly: auto mode narrows the pages in place once the sequence is long enough to profit,
    // which halves the element width and therefore this.
    private nuint _pageBytes;

    // Per-layer pool: Pool[layer][slot] → pointer to PageSize * kvDim * 2 floats (keys then values).
    // Pointers are null until the slot is first allocated (lazy allocation). A pool can be held
    // alive by immutable prefix snapshots after the request that built it has retired.
    private sealed class NativePagePool
    {
        public readonly float*[][] Pages;
        private int _references = 1;

        /// <summary>True when a fork or snapshot also holds these pages. Narrowing them in place
        /// would rewrite memory another cache is reading as F32, so conversion must refuse.</summary>
        public bool IsShared => Volatile.Read(ref _references) > 1;

        public NativePagePool(int numLayers, int maxBlocks)
        {
            Pages = new float*[numLayers][];
            for (int l = 0; l < numLayers; l++)
                Pages[l] = new float*[maxBlocks];
        }

        public NativePagePool AddRef()
        {
            Interlocked.Increment(ref _references);
            return this;
        }

        public void Release(int numLayers, int maxBlocks)
        {
            if (Interlocked.Decrement(ref _references) != 0) return;
            for (int l = 0; l < numLayers; l++)
                for (int s = 0; s < maxBlocks; s++)
                    if (Pages[l][s] != null)
                        NativeMemory.Free(Pages[l][s]);
        }
    }

    private readonly NativePagePool _pool;
    private float*[][] Pool => _pool.Pages;

    // A forked cache reads these leading, page-aligned blocks from a retained source pool.
    // Writes to one of those blocks first materialise a private copy (copy-on-write).
    private NativePagePool? _sharedPrefixPool;
    private int[]? _sharedPrefixSlots;
    private int _sharedPrefixBlocks;
    private bool[] _ownedBlocks = Array.Empty<bool>();

    // Block table: _blockTable[layer][blockIdx] = physical slot index in _pool[layer].
    private int[][] _blockTable;

    // High-water mark: number of blocks currently in the block table.
    // Only grows via Append; never shrinks (TruncateTo is a soft operation).
    // Reset sets this to 0 and pushes all slots back to _warmPool.
    private int _allocatedBlocks;

    // Slot allocator: warm pool returns recently-freed slots; _nextFreshSlot allocates never-used ones.
    private readonly Stack<int> _warmPool;
    private int _nextFreshSlot;

    // Current logical position count (can be < _allocatedBlocks * PageSize after TruncateTo).
    private int _length;

    // SnapKV (issue #51): when prefill eviction is active, _length becomes the
    // *slot* count after compaction while _logicalLength stays at the original
    // (pre-eviction) prompt length. The two diverge only after Compact() is
    // called — outside of SnapKV they track each other exactly.
    //
    //   _length         : how many K/V vectors are physically stored, == slot
    //                     index of the next append.
    //   _logicalLength  : the absolute position the next decode token sits at,
    //                     used for RoPE so cached K's (RoPE'd at their
    //                     original positions) and the incoming query share the
    //                     same reference frame.
    private int _logicalLength;

    private bool _disposed;

    /// <param name="bf16Store">Store K/V as BF16 rather than F32 — half the bytes, half the DRAM
    /// traffic. Only pass true from a forward pass that has BF16 readers (today: the dense CPU
    /// <c>ForwardPass</c>); every other reader's accessors throw. See <see cref="Bf16StoreRequested"/>.</param>
    /// <param name="autoBf16">Start F32 and narrow to BF16 once <see cref="Length"/> reaches
    /// <see cref="Bf16AutoMinTokens"/>. Same reader requirements as <paramref name="bf16Store"/>.</param>
    /// <param name="layerHeadDim">Per-layer head_dim, for models that vary it by layer (Gemma 4:
    /// 256 on SWA layers, 512 on global ones). <paramref name="headDim"/> then sizes the pages at
    /// the MAXIMUM, while each layer reads and writes its V head planes at its OWN stride. Pages
    /// are pooled per layer, so no two layers ever share one and a per-layer stride is safe.
    /// Null (the default) means every layer uses <paramref name="headDim"/>.</param>
    public PagedKvCache(int numLayers, int numKvHeads, int headDim, int maxBlocks = 8192,
                        bool bf16Store = false, bool autoBf16 = false, int[]? layerHeadDim = null)
    {
        _autoBf16 = autoBf16 && !bf16Store;
        _numLayers = numLayers;
        _layerHeadDim = layerHeadDim;
        _kvDim = numKvHeads * headDim;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _maxBlocks = maxBlocks;

        // Page layout, keys:   [PageSize][kvDim]           — read whole-row by the score pass.
        // Page layout, values: [numKvHeads][PageSize][headDim] — TRANSPOSED within the page.
        //
        // Values are read one KV head at a time (the weighted-V sum), so a row-major V region made
        // each head touch headDim floats out of every kvDim, i.e. 256 B out of every 2 KB here.
        // That defeats the prefetcher and, at long context, thrashes the L2 DTLB: one layer's V
        // sweep scatters ~830 KB of useful data across ~13 MB. Transposing within the page turns a
        // head's 16 per-page reads into one contiguous headDim*PageSize run (4 KB here).
        // Measured worth +17.2% CPU decode (perf loop iterations 49-52). Total page size is
        // unchanged — this is a permutation of the same V region, not extra memory.
        // The `* 2` is K-region + V-region, not a dtype width; the element width is the last factor.
        _bf16Store = bf16Store;
        _pageBytes = (nuint)(PageSize * _kvDim * 2 * (_bf16Store ? sizeof(ushort) : sizeof(float)));

        _pool = new NativePagePool(numLayers, maxBlocks);

        _blockTable = new int[numLayers][];
        for (int l = 0; l < numLayers; l++)
            _blockTable[l] = Array.Empty<int>();

        _warmPool = new Stack<int>(64);
        _nextFreshSlot = 0;
    }

    public int Length => _length;
    public int KvDim => _kvDim;

    /// <summary>
    /// Absolute position the next decode token will sit at, == <see cref="Length"/>
    /// unless SnapKV (issue #51) eviction has been applied. After
    /// <see cref="Compact"/>, this stays at the original prompt length while
    /// <see cref="Length"/> drops to the surviving slot count; downstream
    /// callers should use <see cref="LogicalLength"/> for RoPE on new tokens
    /// so the cached RoPE'd K's stay in the right reference frame.
    /// </summary>
    public int LogicalLength => _logicalLength;

    int IRewindableSequenceKvCache.LogicalPosition => _logicalLength;

    bool IRewindableSequenceKvCache.CanRewindTo(int logicalPosition) =>
        !_disposed && _logicalLength == _length && logicalPosition >= 0 && logicalPosition <= _logicalLength;

    void IRewindableSequenceKvCache.RewindTo(int logicalPosition)
    {
        if (!((IRewindableSequenceKvCache)this).CanRewindTo(logicalPosition))
            throw new NotSupportedException(
                "PagedKvCache can exactly rewind retained session state only when logical and physical positions agree.");
        TruncateTo(logicalPosition);
    }

    /// <summary>Maximum sequence length this cache can hold (slot pool limit).</summary>
    public int MaxSeqLen => _maxBlocks * PageSize;

    // ── Slot management ──────────────────────────────────────────────────

    private int AllocSlot()
    {
        if (_warmPool.Count > 0) return _warmPool.Pop();
        if (_nextFreshSlot >= _maxBlocks)
            throw new InvalidOperationException(
                $"PagedKvCache exhausted: max {_maxBlocks} blocks × {PageSize} positions = {MaxSeqLen} tokens");
        return _nextFreshSlot++;
    }

    private void EnsureBlockTableCapacity(int blockIdx)
    {
        for (int l = 0; l < _numLayers; l++)
        {
            if (_blockTable[l].Length <= blockIdx)
                Array.Resize(ref _blockTable[l], Math.Max(blockIdx + 8, _blockTable[l].Length * 2 + 8));
        }
    }

    private float* GetPage(int layer, int slot)
    {
        if (Pool[layer][slot] == null)
            Pool[layer][slot] = (float*)NativeMemory.AllocZeroed(_pageBytes);
        return Pool[layer][slot];
    }

    private void EnsureOwnedCapacity(int blockIdx)
    {
        if (_ownedBlocks.Length <= blockIdx)
            Array.Resize(ref _ownedBlocks, Math.Max(blockIdx + 8, _ownedBlocks.Length * 2 + 8));
    }

    private void AllocateOwnedBlock(int blockIdx)
    {
        EnsureBlockTableCapacity(blockIdx);
        EnsureOwnedCapacity(blockIdx);
        int slot = AllocSlot();
        for (int l = 0; l < _numLayers; l++)
            _blockTable[l][blockIdx] = slot;
        _ownedBlocks[blockIdx] = true;
    }

    private void EnsureWritableBlock(int blockIdx, int layer)
    {
        if (blockIdx >= _sharedPrefixBlocks || _ownedBlocks[blockIdx]) return;
        if (layer != 0)
            throw new InvalidOperationException(
                "PagedKvCache shared-prefix writes must begin at layer 0 so the page can be copied-on-write.");

        int sharedSlot = _sharedPrefixSlots![blockIdx];
        AllocateOwnedBlock(blockIdx);
        for (int l = 0; l < _numLayers; l++)
        {
            float* src = _sharedPrefixPool!.Pages[l][sharedSlot];
            if (src is null) continue;
            float* dst = GetPage(l, _blockTable[l][blockIdx]);
            new ReadOnlySpan<byte>(src, checked((int)_pageBytes))
                .CopyTo(new Span<byte>(dst, checked((int)_pageBytes)));
        }
    }

    private float* ReadPage(int layer, int blockIdx)
    {
        if (blockIdx < _sharedPrefixBlocks && !_ownedBlocks[blockIdx])
            return _sharedPrefixPool!.Pages[layer][_sharedPrefixSlots![blockIdx]];
        return Pool[layer][_blockTable[layer][blockIdx]];
    }

    /// <summary>
    /// Creates a cache sharing exactly <paramref name="prefixLength"/> immutable, full pages.
    /// The fork has private pages for all subsequent tokens; a rare rewrite into the shared
    /// range is copied-on-write. This is intentionally page-aligned so normal decode never
    /// needs to copy a partially-filled tail page.
    /// </summary>
    public PagedKvCache ForkSharedPrefix(int prefixLength)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(PagedKvCache));
        if (prefixLength < 0 || prefixLength > _length || prefixLength % PageSize != 0)
            throw new ArgumentOutOfRangeException(nameof(prefixLength),
                "A shared prefix must be a page-aligned length within this cache.");

        // The fork must match this cache's element width: it shares pages by reference and
        // copy-on-writes them with a raw byte copy sized by _pageBytes.
        var fork = new PagedKvCache(_numLayers, _kvDim, 1, _maxBlocks, _bf16Store)
        {
            _sharedPrefixBlocks = prefixLength / PageSize,
            _length = prefixLength,
            _logicalLength = prefixLength,
        };
        fork._sharedPrefixPool = (_sharedPrefixPool ?? _pool).AddRef();
        fork._sharedPrefixSlots = new int[fork._sharedPrefixBlocks];
        for (int b = 0; b < fork._sharedPrefixBlocks; b++)
            fork._sharedPrefixSlots[b] = b < _sharedPrefixBlocks && !_ownedBlocks[b]
                ? _sharedPrefixSlots![b]
                : _blockTable[0][b];
        fork._allocatedBlocks = fork._sharedPrefixBlocks;
        fork._ownedBlocks = new bool[Math.Max(8, fork._sharedPrefixBlocks + 8)];
        return fork;
    }

    // ── Public API ────────────────────────────────────────────────────────

    /// <summary>
    /// Write key and value vectors for <paramref name="layer"/> at position <see cref="Length"/>.
    /// Layer 0 must be appended first for each token to trigger block allocation.
    /// </summary>
    public void Append(int layer, ReadOnlySpan<float> key, ReadOnlySpan<float> value)
    {
        int blockIdx = _length / PageSize;
        int offset = _length % PageSize;

        if (blockIdx >= _allocatedBlocks)
        {
            if (layer != 0)
                throw new InvalidOperationException(
                    "PagedKvCache.Append: layer 0 must be appended first to allocate a new block");
            AllocateOwnedBlock(blockIdx);
            _allocatedBlocks++;
        }
        else
        {
            EnsureWritableBlock(blockIdx, layer);
        }

        float* page = GetPage(layer, _blockTable[layer][blockIdx]);
        WriteKv(layer, page, offset, key, value);
    }

    /// <summary>
    /// Reserves a new block slot in the per-layer block table without allocating any page.
    /// Use this for hybrid architectures where layer 0 does not call <see cref="Append"/>
    /// (e.g. qwen35moe, whose layer 0 is a recurrent Gated DeltaNet block that stores no KV).
    /// Call once per token when crossing a <see cref="PageSize"/> boundary, BEFORE any layer
    /// in the trunk calls Append for the current token. Per-layer pages remain null until
    /// each layer's first KV write triggers lazy allocation.
    /// </summary>
    public void ReserveBlock()
    {
        int blockIdx = _length / PageSize;
        if (blockIdx >= _allocatedBlocks)
        {
            AllocateOwnedBlock(blockIdx);
            _allocatedBlocks++;
        }
    }

    /// <summary>
    /// Write key and value vectors at an explicit <paramref name="position"/> regardless
    /// of <see cref="Length"/>. Used by the MTP batched verify path (issue #30) where two
    /// tokens at adjacent positions share the same cache: token 1's per-layer Append fires
    /// at <c>startPos</c>, token 2's at <c>startPos + 1</c>, both before either token
    /// bumps <see cref="Length"/>. The block table must already cover the page containing
    /// <paramref name="position"/>; call <see cref="ReserveBlockAt"/> on token 1's first
    /// layer for each new page.
    /// </summary>
    public void AppendAt(int layer, int position, ReadOnlySpan<float> key, ReadOnlySpan<float> value)
    {
        int blockIdx = position / PageSize;
        int offset = position % PageSize;

        if (blockIdx >= _allocatedBlocks)
            throw new InvalidOperationException(
                $"PagedKvCache.AppendAt({position}): block {blockIdx} not reserved " +
                $"(allocated={_allocatedBlocks}). Call ReserveBlockAt before the first " +
                "AppendAt that crosses a page boundary.");

        EnsureWritableBlock(blockIdx, layer);
        float* page = GetPage(layer, _blockTable[layer][blockIdx]);
        WriteKv(layer, page, offset, key, value);
    }

    /// <summary>
    /// Reserve the block containing <paramref name="position"/> if not already allocated.
    /// Mirrors <see cref="ReserveBlock"/> but for an explicit position rather than the
    /// current <see cref="Length"/>. Used by the batched verify path to make room for
    /// two tokens that may straddle a page boundary.
    /// </summary>
    public void ReserveBlockAt(int position)
    {
        int blockIdx = position / PageSize;
        if (blockIdx >= _allocatedBlocks)
        {
            EnsureBlockTableCapacity(blockIdx);
            // Stretch the high-water mark to include any skipped blocks too — the cache
            // expects contiguous block ids 0.._allocatedBlocks-1 with valid slots.
            for (int b = _allocatedBlocks; b <= blockIdx; b++)
                AllocateOwnedBlock(b);
            _allocatedBlocks = blockIdx + 1;
        }
    }

    /// <summary>
    /// Narrows every allocated page from F32 to BF16 in one pass and flips the cache's format.
    /// </summary>
    /// <remarks>
    /// <para>Called from <see cref="IncrementPosition"/>, i.e. on a token boundary with every
    /// layer's K/V for the current token already written. That placement is what makes it safe:
    /// readers latch the format once per attention call, and no attention call is in flight here.</para>
    ///
    /// <para><b>Refuses when the pages are shared</b> — a shared-prefix fork reads these very pages
    /// as F32 and copy-on-writes them with a raw byte copy sized by <c>_pageBytes</c>. Narrowing
    /// underneath it would be silent corruption rather than a throw, so a cache that is sharing or
    /// being shared stays F32 for its whole life. The cost of that conservatism is one missed
    /// optimisation on prefix-reuse sessions, which is the right side to err on.</para>
    ///
    /// <para>Every page in the pool is converted, not just the currently-allocated blocks: pages
    /// returned to the warm pool keep their allocation, and leaving those at the old width would
    /// hand a full-size F32 page back to a BF16 cache on the next <see cref="Reset"/>.</para>
    /// </remarks>
    private void ConvertPagesToBf16()
    {
        if (_bf16Store || _sharedPrefixBlocks > 0 || _sharedPrefixPool is not null || _pool.IsShared)
            return;

        int elemsPerPage = PageSize * _kvDim * 2;   // K region + V region, same element order in both formats.
        nuint newBytes = (nuint)(elemsPerPage * sizeof(ushort));

        for (int l = 0; l < _numLayers; l++)
        {
            float*[] layerPages = Pool[l];
            for (int slot = 0; slot < layerPages.Length; slot++)
            {
                float* src = layerPages[slot];
                if (src is null) continue;
                ushort* dst = (ushort*)NativeMemory.Alloc(newBytes);
                for (int i = 0; i < elemsPerPage; i++) dst[i] = ToBf16Bits(src[i]);
                layerPages[slot] = (float*)dst;
                NativeMemory.Free(src);
            }
        }

        _pageBytes = newBytes;
        _bf16Store = true;
        Interlocked.Increment(ref s_bf16Conversions);
    }

    /// <summary>Advances the logical length. Call once per token after all layers are appended.</summary>
    public void IncrementPosition()
    {
        if (_autoBf16 && !_bf16Store && _length + 1 >= Bf16AutoMinTokens)
            ConvertPagesToBf16();

        _length++;
        _logicalLength++;
    }

    /// <summary>Returns a pointer to the key vector at <paramref name="position"/> for <paramref name="layer"/>.</summary>
    /// <exception cref="InvalidOperationException">The cache is in BF16 store mode, where pages hold
    /// 2-byte elements. Reinterpreting those as F32 yields plausible garbage rather than an error,
    /// so this throws instead; use <see cref="Bf16KeyAt"/>.</exception>
    public float* KeyAt(int layer, int position)
    {
        if (_bf16Store) throw new InvalidOperationException(Bf16AccessorMessage(nameof(KeyAt)));
        float* page = ReadPage(layer, position / PageSize);
        return page + (long)(position % PageSize) * _kvDim;
    }

    private static string Bf16AccessorMessage(string member) =>
        $"PagedKvCache.{member} is unavailable when STINGRAY_KV_STORE=bf16: pages hold 2-byte " +
        $"elements. Use Bf16{member} (widening on load), or unset STINGRAY_KV_STORE. Only the " +
        "dense CPU attention path reads BF16 pages natively today.";

    /// <summary>BF16-store counterpart of <see cref="KeyAt"/>. Elements are the top 16 bits of the
    /// F32, so widening is a zero-extend plus a 16-bit left shift.</summary>
    public ushort* Bf16KeyAt(int layer, int position)
    {
        ushort* page = (ushort*)ReadPage(layer, position / PageSize);
        return page + (long)(position % PageSize) * _kvDim;
    }

    /// <summary>BF16-store counterpart of <see cref="ValueAtHead"/>. Same transposed
    /// <c>[numKvHeads][PageSize][headDim]</c> V layout — only the element width differs.</summary>
    /// <summary>
    /// The V head-plane stride for one layer. Only the TRANSPOSED V region needs this: the K
    /// region is row-major and its readers already index with the caller's per-layer head_dim.
    /// Taking it from <see cref="_headDim"/> instead made every KV head above the first read
    /// unwritten memory on models with per-layer head_dim (Gemma 4 SWA layers: 256 vs 512).
    /// </summary>
    private int HeadDimOf(int layer) => _layerHeadDim is null ? _headDim : _layerHeadDim[layer];

    public ushort* Bf16ValueAtHead(int layer, int position, int kvHead)
    {
        ushort* page = (ushort*)ReadPage(layer, position / PageSize);
        return page + (long)PageSize * _kvDim
             + ((long)kvHead * PageSize + position % PageSize) * HeadDimOf(layer);
    }

    /// <summary>
    /// Returns a pointer to one KV head's <paramref name="headDim"/>-float value slice at
    /// <paramref name="position"/>. Values are stored transposed within the page
    /// (<c>[numKvHeads][PageSize][headDim]</c>), so a whole <c>kvDim</c>-wide value row is NOT
    /// contiguous — there is deliberately no row-returning overload. Consecutive positions for a
    /// fixed head ARE contiguous, which is the point of the layout.
    /// </summary>
    /// <exception cref="InvalidOperationException">The cache is in BF16 store mode — see
    /// <see cref="KeyAt"/>. Use <see cref="Bf16ValueAtHead"/>.</exception>
    public float* ValueAtHead(int layer, int position, int kvHead)
    {
        if (_bf16Store) throw new InvalidOperationException(Bf16AccessorMessage(nameof(ValueAtHead)));
        float* page = ReadPage(layer, position / PageSize);
        return page + (long)PageSize * _kvDim
             + ((long)kvHead * PageSize + position % PageSize) * HeadDimOf(layer);
    }

    /// <summary>Scatters a contiguous kvDim-wide value row into the transposed V region of a page.</summary>
    /// <summary>
    /// Single write point for a token's K and V, so the BF16 precision gate (<see cref="s_kvBf16"/>)
    /// applies identically to <see cref="Append"/> and <see cref="AppendAt"/>. Having two copies of
    /// this would be a numerics boundary between the ordinary and the speculative/batched-verify
    /// paths — the defect class this codebase keeps re-learning.
    /// </summary>
    private void WriteKv(int layer, float* page, int offset,
                         ReadOnlySpan<float> key, ReadOnlySpan<float> value)
    {
        if (_bf16Store)
        {
            WriteKvBf16(layer, page, offset, key, value);
            return;
        }

        float* keyDst = page + (long)offset * _kvDim;
        if (!s_kvBf16)
        {
            key[.._kvDim].CopyTo(new Span<float>(keyDst, _kvDim));
            ScatterValue(layer, page, offset, value);
            return;
        }

        float* kb = stackalloc float[_kvDim];
        float* vb = stackalloc float[_kvDim];
        for (int i = 0; i < _kvDim; i++)
        {
            kb[i] = ToBf16Precision(key[i]);
            vb[i] = ToBf16Precision(value[i]);
        }
        new ReadOnlySpan<float>(kb, _kvDim).CopyTo(new Span<float>(keyDst, _kvDim));
        ScatterValue(layer, page, offset, new ReadOnlySpan<float>(vb, _kvDim));
        Interlocked.Increment(ref s_kvBf16Appends);
    }

    /// <summary>
    /// BF16-store write. Same page geometry as the F32 path — keys row-major at
    /// <c>offset * kvDim</c>, values transposed at <c>kvDim*PageSize + (h*PageSize + offset)*headDim</c>
    /// — with 2-byte elements, so every offset is in <see cref="ushort"/> units.
    /// </summary>
    private void WriteKvBf16(int layer, float* page, int offset,
                             ReadOnlySpan<float> key, ReadOnlySpan<float> value)
    {
        ushort* p = (ushort*)page;
        ushort* keyDst = p + (long)offset * _kvDim;
        for (int i = 0; i < _kvDim; i++) keyDst[i] = ToBf16Bits(key[i]);

        ushort* vBase = p + (long)PageSize * _kvDim;
        for (int h = 0; h < _numKvHeads; h++)
        {
            ushort* dst = vBase + ((long)h * PageSize + offset) * _headDim;
            int src = h * _headDim;
            for (int d = 0; d < _headDim; d++) dst[d] = ToBf16Bits(value[src + d]);
        }
        Interlocked.Increment(ref s_kvBf16Appends);
    }

    /// <summary>
    /// Round-to-nearest-even F32 -> BF16, returned as the raw 16 bits. Mirrors
    /// <see cref="ToBf16Precision"/> so the quality scaffold and the real store agree bit for bit.
    /// </summary>
    private static ushort ToBf16Bits(float f)
    {
        uint u = BitConverter.SingleToUInt32Bits(f);
        // A NaN must stay a NaN. Truncating the mantissa can clear every surviving mantissa bit and
        // silently produce an infinity, so force one back on rather than let the rounding add run.
        if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x007FFFFFu) != 0)
            return (ushort)((u >> 16) | 0x0040u);
        u += 0x7FFFu + ((u >> 16) & 1u);
        return (ushort)(u >> 16);
    }

    /// <summary>Widens one BF16 store element back to F32 — the inverse of <see cref="ToBf16Bits"/>.</summary>
    private static float FromBf16Bits(ushort v) => BitConverter.UInt32BitsToSingle((uint)v << 16);

    /// <summary>BF16-store counterpart of <see cref="GatherValue"/>, widening as it gathers.</summary>
    private void GatherValueBf16(float* page, int offset, Span<float> dst)
    {
        ushort* vBase = (ushort*)page + (long)PageSize * _kvDim;
        for (int h = 0; h < _numKvHeads; h++)
        {
            ushort* src = vBase + ((long)h * PageSize + offset) * _headDim;
            int o = h * _headDim;
            for (int d = 0; d < _headDim; d++) dst[o + d] = FromBf16Bits(src[d]);
        }
    }

    /// <summary>BF16-store counterpart of <see cref="ScatterValue"/>, narrowing as it scatters.</summary>
    private void ScatterValueBf16(int layer, float* page, int offset, ReadOnlySpan<float> value)
    {
        int hd = HeadDimOf(layer);
        ushort* vBase = (ushort*)page + (long)PageSize * _kvDim;
        for (int h = 0; h < _numKvHeads; h++)
        {
            ushort* dst = vBase + ((long)h * PageSize + offset) * hd;
            int o = h * hd;
            for (int d = 0; d < hd; d++) dst[d] = ToBf16Bits(value[o + d]);
        }
    }

    private void ScatterValue(int layer, float* page, int offset, ReadOnlySpan<float> value)
    {
        int hd = HeadDimOf(layer);
        float* vBase = page + (long)PageSize * _kvDim;
        for (int h = 0; h < _numKvHeads; h++)
            value.Slice(h * hd, hd)
                 .CopyTo(new Span<float>(vBase + ((long)h * PageSize + offset) * hd, hd));
    }

    /// <summary>Gathers one position's value row out of the transposed V region into a contiguous span.</summary>
    private void GatherValue(int layer, float* page, int offset, Span<float> dst)
    {
        int hd = HeadDimOf(layer);
        float* vBase = page + (long)PageSize * _kvDim;
        for (int h = 0; h < _numKvHeads; h++)
            new ReadOnlySpan<float>(vBase + ((long)h * PageSize + offset) * hd, hd)
                .CopyTo(dst.Slice(h * hd, hd));
    }

    /// <summary>
    /// Soft truncate: sets the logical length to <paramref name="length"/> without freeing pages.
    /// Positions ≥ <paramref name="length"/> will be overwritten by subsequent appends.
    /// Used during per-layer prefill resets and speculative decoding rewinds.
    /// </summary>
    public void TruncateTo(int length)
    {
        _length = length;
        _logicalLength = length;
    }

    /// <summary>
    /// Full reset: returns all allocated page slots to the warm pool for reuse.
    /// Use at the start of a new request when the KV prefix cannot be reused.
    /// </summary>
    public void Reset()
    {
        // All layers share the same slot index per block — push each slot once.
        for (int b = 0; b < _allocatedBlocks; b++)
            if (b < _ownedBlocks.Length && _ownedBlocks[b])
                _warmPool.Push(_blockTable[0][b]);
        _allocatedBlocks = 0;
        _length = 0;
        _logicalLength = 0;
    }

    /// <summary>
    /// SnapKV (issue #51) compaction: keep only the K/V vectors at the positions
    /// listed in <paramref name="keepPositions"/> (sorted ascending, all within
    /// <c>[0, Length)</c>); discard the rest. After compaction the cache holds
    /// <c>keepPositions.Length</c> entries in slots <c>[0, keepPositions.Length)</c>;
    /// <see cref="LogicalLength"/> is left at the pre-compaction value so RoPE on
    /// subsequent decode tokens stays in the original position frame.
    /// </summary>
    /// <remarks>
    /// Uniform-across-layers eviction only: the same keep set applies to every
    /// layer because the block table is shared. Per-layer eviction (true SnapKV)
    /// is a follow-up; for the common case where attention sparsity patterns
    /// correlate across layers this is the right tradeoff for a first cut.
    ///
    /// The implementation copies survivors into a staging buffer one page at a
    /// time, then writes them back contiguously into the existing pages. This
    /// in-place style avoids allocating a parallel cache and works regardless
    /// of how non-contiguous the keep set is.
    /// </remarks>
    public void Compact(ReadOnlySpan<int> keepPositions)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(PagedKvCache));
        if (_sharedPrefixBlocks > 0)
            throw new NotSupportedException("Compact is not supported on a shared-prefix cache.");
        int K = keepPositions.Length;
        if (K > _length)
            throw new ArgumentException(
                $"Compact: keep count {K} exceeds current Length {_length}.",
                nameof(keepPositions));
        // Validate sorted + bounds. The selector contract guarantees this; the
        // assert here catches caller bugs (it's cheap relative to the layer loop).
        int prev = -1;
        for (int i = 0; i < K; i++)
        {
            int p = keepPositions[i];
            if (p < 0 || p >= _length)
                throw new ArgumentOutOfRangeException(nameof(keepPositions),
                    $"keepPositions[{i}]={p} is outside [0,{_length}).");
            if (p <= prev)
                throw new ArgumentException(
                    $"keepPositions must be strictly increasing; got {prev} then {p} at index {i}.",
                    nameof(keepPositions));
            prev = p;
        }

        if (K == _length)
        {
            // No-op compaction (every position kept). Skip the staging dance.
            return;
        }

        int preCompactLength = _logicalLength;

        // Stage one layer at a time: read survivors into a contiguous host
        // buffer, then write them back into slot 0..K-1 of the same layer.
        // Buffer size: K × kvDim × 2 floats. For K=2048, kvDim=1024 (Qwen3-8B-
        // class), that's 16 MiB — fine for a one-shot post-prefill op.
        nuint stageBytes = (nuint)((long)K * _kvDim * 2 * sizeof(float));
        float* stage = (float*)NativeMemory.Alloc(stageBytes);
        try
        {
            for (int l = 0; l < _numLayers; l++)
            {
                // Pull survivors into the stage buffer (K rows × kvDim K-then-V layout).
                for (int i = 0; i < K; i++)
                {
                    int srcPos = keepPositions[i];
                    int srcSlot = _blockTable[l][srcPos / PageSize];
                    float* srcPage = Pool[l][srcSlot];
                    if (srcPage == null)
                    {
                        // Layer's page was never allocated for that block — write
                        // zeros to the stage row. This happens for hybrid GDN
                        // models on non-attention layers; the slot exists but no
                        // K/V was ever appended. Compaction should leave them
                        // empty rather than crash.
                        for (int j = 0; j < _kvDim * 2; j++) stage[(long)i * _kvDim * 2 + j] = 0f;
                        continue;
                    }
                    int srcOff = srcPos % PageSize;
                    float* dstKey = stage + (long)i * _kvDim * 2;
                    float* dstVal = dstKey + _kvDim;
                    // The stage buffer is always F32; a BF16 store widens into it here and narrows
                    // again on write-back, so survivor selection and ordering below are identical
                    // for both formats.
                    if (_bf16Store)
                    {
                        ushort* srcKeyB = (ushort*)srcPage + (long)srcOff * _kvDim;
                        for (int j = 0; j < _kvDim; j++) dstKey[j] = FromBf16Bits(srcKeyB[j]);
                        GatherValueBf16(srcPage, srcOff, new Span<float>(dstVal, _kvDim));
                    }
                    else
                    {
                        float* srcKey = srcPage + (long)srcOff * _kvDim;
                        new ReadOnlySpan<float>(srcKey, _kvDim)
                            .CopyTo(new Span<float>(dstKey, _kvDim));
                        // Stage rows stay row-major so the survivor order logic below is unchanged;
                        // only the in-page representation is transposed.
                        GatherValue(l, srcPage, srcOff, new Span<float>(dstVal, _kvDim));
                    }
                }

                // Write survivors back into slot 0..K-1 of this layer. We touch
                // each destination page through GetPage so any never-allocated
                // pages get materialised on demand (matters when the source
                // survivors came from a higher slot than was previously
                // populated for this layer).
                int writeBlocks = (K + PageSize - 1) / PageSize;
                for (int i = 0; i < K; i++)
                {
                    int dstBlk = i / PageSize;
                    int dstOff = i % PageSize;
                    int dstSlot = _blockTable[l][dstBlk];
                    float* dstPage = GetPage(l, dstSlot);
                    float* srcKey = stage + (long)i * _kvDim * 2;
                    float* srcVal = srcKey + _kvDim;
                    if (_bf16Store)
                    {
                        ushort* dstKeyB = (ushort*)dstPage + (long)dstOff * _kvDim;
                        for (int j = 0; j < _kvDim; j++) dstKeyB[j] = ToBf16Bits(srcKey[j]);
                        ScatterValueBf16(l, dstPage, dstOff, new ReadOnlySpan<float>(srcVal, _kvDim));
                    }
                    else
                    {
                        float* dstKey = dstPage + (long)dstOff * _kvDim;
                        new ReadOnlySpan<float>(srcKey, _kvDim)
                            .CopyTo(new Span<float>(dstKey, _kvDim));
                        ScatterValue(l, dstPage, dstOff, new ReadOnlySpan<float>(srcVal, _kvDim));
                    }
                }

                // Free pages beyond the compacted prefix. These slots return to
                // the warm pool for the *next* request's prefill — the current
                // request can still grow into them via Append for decode tokens.
                // We DON'T free the native pages here because the slot may be
                // reused immediately; freeing would force a re-alloc.
            }

            // Free trailing block-table entries whose slots are no longer used
            // for any kept position. Push their slots back to the warm pool so
            // decode appends reuse them (or future Reset can.)
            int compactedBlocks = (K + PageSize - 1) / PageSize;
            for (int b = compactedBlocks; b < _allocatedBlocks; b++)
            {
                // All layers share the slot index per block — push once.
                _warmPool.Push(_blockTable[0][b]);
            }
            _allocatedBlocks = compactedBlocks;
            _length = K;
            _logicalLength = preCompactLength;
        }
        finally
        {
            NativeMemory.Free(stage);
        }
    }

    /// <summary>
    /// Bookkeeping-only counterpart of <see cref="Compact"/> for callers (e.g.
    /// <see cref="CudaHybridGdnForwardPass"/>) that store the actual KV payload
    /// outside this cache and only use it for slot/position accounting. Drops
    /// <see cref="Length"/> to <paramref name="K"/>, keeps <see cref="LogicalLength"/>
    /// at its current value (so RoPE on subsequent decode tokens stays in the
    /// pre-eviction position frame), and returns trailing block slots to the
    /// warm pool. Never touches per-layer page memory.
    /// </summary>
    public void CompactLengthOnly(int K)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(PagedKvCache));
        if (K < 0 || K > _length)
            throw new ArgumentOutOfRangeException(nameof(K), K,
                $"K must be in [0, Length={_length}].");
        int compactedBlocks = (K + PageSize - 1) / PageSize;
        for (int b = compactedBlocks; b < _allocatedBlocks; b++)
            _warmPool.Push(_blockTable[0][b]);
        _allocatedBlocks = compactedBlocks;
        _length = K;
        // _logicalLength stays untouched — that's the whole point.
    }

    // ── IPersistableSequenceKvCache ──────────────────────────────────────────

    /// <summary>Elements (K region + V region) in one page, independent of element width.</summary>
    private int ElementsPerPage => PageSize * _kvDim * 2;

    /// <summary>
    /// Serialises the live pages into a self-describing PKVC stream (magic, version, dimensions,
    /// element format, lengths, then one flag+page per block per layer).
    /// </summary>
    /// <remarks>
    /// <para><b>Not thread-safe against a running turn.</b> This class synchronises nothing, and the
    /// export walks <c>_length</c>, the block table and every page. The caller must have quiesced the
    /// session first — a concurrent <c>Append</c> would produce a stream whose header disagrees with
    /// its body. <c>ColdSessionRuntime.EvictToDisk</c> enforces that.</para>
    ///
    /// <para>Only blocks covering <c>_length</c> are written. <c>_allocatedBlocks</c> can exceed that
    /// after a rewind — <see cref="TruncateTo"/> is a soft operation that keeps pages allocated — and
    /// those blocks hold nothing a restore needs. At 256 KB per layer per block that dead weight is
    /// the difference between fitting a storage bound and not.</para>
    /// </remarks>
    public byte[] ExportKvState()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(PagedKvCache));

        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true);

        w.Write(0x504B5643u); // Magic: PKVC
        w.Write((ushort)1);   // Format version
        w.Write(_numLayers);
        w.Write(_kvDim);
        w.Write(_numKvHeads);
        w.Write(_headDim);
        w.Write(_bf16Store);
        w.Write(_length);
        w.Write(_logicalLength);
        int liveBlocks = Math.Min(_allocatedBlocks, (_length + PageSize - 1) / PageSize);
        w.Write(liveBlocks);

        for (int b = 0; b < liveBlocks; b++)
        {
            for (int l = 0; l < _numLayers; l++)
            {
                float* pagePtr = ReadPage(l, b);
                if (pagePtr != null)
                {
                    w.Write((byte)1);
                    var span = new ReadOnlySpan<byte>(pagePtr, (int)_pageBytes);
                    w.Write(span);
                }
                else
                {
                    w.Write((byte)0);
                }
            }
        }

        w.Flush();
        return ms.ToArray();
    }

    public void ImportKvState(ReadOnlySpan<byte> kvStateBytes)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(PagedKvCache));
        // magic(4) + version(2) + 4 dims(16) + bf16 flag(1) + length/logicalLength/blocks(12) = 35.
        // A cache with no allocated blocks exports exactly this and must round-trip.
        const int PkvcHeaderBytes = 35;
        if (kvStateBytes.Length < PkvcHeaderBytes)
            throw new InvalidDataException("Invalid PKVC state stream: buffer too small for header.");

        fixed (byte* p = kvStateBytes)
        {
            using var ms = new UnmanagedMemoryStream(p, kvStateBytes.Length);
            using var r = new BinaryReader(ms, Encoding.UTF8, leaveOpen: true);

            uint magic = r.ReadUInt32();
            if (magic != 0x504B5643u)
                throw new InvalidDataException($"Invalid PKVC magic header: 0x{magic:X8}.");

            ushort version = r.ReadUInt16();
            if (version != 1)
                throw new InvalidDataException($"Unsupported PKVC version: {version}.");

            int numLayers = r.ReadInt32();
            int kvDim = r.ReadInt32();
            int numKvHeads = r.ReadInt32();
            int headDim = r.ReadInt32();
            bool bf16Store = r.ReadBoolean();
            int length = r.ReadInt32();
            int logicalLength = r.ReadInt32();
            int allocatedBlocks = r.ReadInt32();

            if (numLayers != _numLayers || kvDim != _kvDim || numKvHeads != _numKvHeads || headDim != _headDim)
                throw new InvalidOperationException(
                    $"PagedKvCache dimensions mismatch on state import. Configured: {numLayers}/{kvDim}/{numKvHeads}/{headDim}, Expected: {_numLayers}/{_kvDim}/{_numKvHeads}/{_headDim}.");

            // Element formats may legitimately differ. Under STINGRAY_KV_STORE=auto a cache narrows
            // to BF16 once it passes the crossover, so any session long enough to be worth evicting
            // exports bf16=true — while the cache restoring it starts F32 and has not yet crossed its
            // own threshold. Refusing that would fail on exactly the sessions cold storage exists for,
            // so convert instead. Page geometry is identical in both formats; only element width
            // differs, and BF16 is the top 16 bits of the F32.
            bool needsConversion = bf16Store != _bf16Store;
            nuint sourcePageBytes = needsConversion
                ? (nuint)(PageSize * _kvDim * 2 * (bf16Store ? sizeof(ushort) : sizeof(float)))
                : _pageBytes;

            Reset();

            for (int b = 0; b < allocatedBlocks; b++)
            {
                ReserveBlockAt(b * PageSize);
                for (int l = 0; l < _numLayers; l++)
                {
                    byte flag = r.ReadByte();
                    if (flag == 1)
                    {
                        float* pagePtr = GetPage(l, _blockTable[l][b]);
                        byte[] pageData = r.ReadBytes((int)sourcePageBytes);
                        if ((nuint)pageData.Length != sourcePageBytes)
                            throw new InvalidDataException("Truncated page data in PKVC stream.");

                        fixed (byte* src = pageData)
                        {
                            if (!needsConversion)
                            {
                                Buffer.MemoryCopy(src, pagePtr, (long)_pageBytes, (long)_pageBytes);
                            }
                            else if (_bf16Store)
                            {
                                // F32 stream -> BF16 pages: round each element to nearest even.
                                float* f = (float*)src;
                                ushort* dst = (ushort*)pagePtr;
                                for (int i = 0; i < ElementsPerPage; i++) dst[i] = ToBf16Bits(f[i]);
                            }
                            else
                            {
                                // BF16 stream -> F32 pages: widen each element.
                                ushort* b16 = (ushort*)src;
                                float* dst = pagePtr;
                                for (int i = 0; i < ElementsPerPage; i++) dst[i] = FromBf16Bits(b16[i]);
                            }
                        }
                    }
                }
            }

            _length = length;
            _logicalLength = logicalLength;
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _sharedPrefixPool?.Release(_numLayers, _maxBlocks);
        _pool.Release(_numLayers, _maxBlocks);
    }
}
