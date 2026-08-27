using System.Runtime.CompilerServices;
using System.Threading.Channels;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Core.Grammar;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Continuous batching inference engine: admits multiple concurrent requests and decodes them
/// in lock-step batches, amortizing weight reads across N sequences per decode step.
///
/// Flow:
///   1. Caller enqueues requests via <see cref="GenerateChunksAsync"/>.
///   2. Background batcher admits pending requests under a KV token budget (issue #183 Gap 3:
///      admission backpressure — a burst of long prompts queues instead of exhausting memory).
///   3. Admitted prompts prefill in chunks interleaved with decode steps (issue #183 Gap 1:
///      a long prompt no longer stalls every active sequence). When several prompts are
///      prefilling at once, their chunks run as one packed forward pass
///      (<see cref="ForwardPass.PrefillPackedMulti"/>, issue #183 Gap 2) so weight reads are
///      amortized across prompts exactly like decode batching.
///   4. All active sequences are decoded together in a single <see cref="ForwardPass.BatchForwardMulti"/> call.
///   5. Sequences that hit EOS or max tokens are retired; their caches are returned to the pool.
///
/// Not supported for MoE models or when TurboQuant KV cache is enabled.
///
/// Reasoning support: when constructed with <c>thinkTokenId</c> / <c>endThinkTokenId</c>,
/// the engine splits each sequence's output into <see cref="GenerateChunkKind.Thinking"/>
/// and <see cref="GenerateChunkKind.Text"/> chunks. Per-sequence state tracks the current
/// reasoning mode and uses independent UTF-8 decoders so multi-byte chars in either stream
/// reassemble cleanly.
/// </summary>
public sealed class ContinuousBatchingEngine : IInferenceEngine, IContinuousBatchingObservability, IDisposable
{
    // A request may emit at most one chunk per generated token plus usage, stop and two UTF-8
    // decoder flush chunks. The transport remains non-blocking for the batcher, while this bound
    // prevents an abandoned reader from accumulating an unbounded response in its channel.
    private const int MaxBufferedOutputChunks = 4_096;
    private readonly IBatchedForwardPass _fwd;
    private readonly IKvCache _kvCache;
    private readonly bool _ownsKvCache;
    private readonly ITokenizer _tokenizer;
    private readonly int _maxBatchSize;
    private readonly int _thinkTokenId;
    private readonly int _endThinkTokenId;
    private readonly bool _thinkingEnabled;

    // Issue #183 Gap 1: tokens of prompt prefilled per batcher iteration. Between chunks
    // every active sequence advances one decode step. 0 disables chunking (a prompt
    // prefills in one blocking call — the pre-#183 behavior). SnapKV also disables
    // chunking: its prefill eviction only runs on a fresh full-prompt prefill.
    private readonly int _prefillChunkTokens;
    private readonly bool _chunkingEnabled;
    private readonly IPrefixCacheableBatchedForwardPass? _prefixFwd;
    private readonly long _prefixCacheBytes;
    private long _prefixCacheUsedBytes;
    private long _prefillTokensReused;
    private long _crossSessionPrefixTokensShared;
    private long _prefixCacheHits;
    private long _prefixCacheMisses;
    private long _prefixCacheHitTokens;
    private long _prefixCacheMissMatchedTokens;
    private int _prefixCacheLastHitLength;
    private int _prefixCacheLastMissLongestMatch;
    private long _prefixCacheEvictions;
    private int _prefixCacheEntryCount;
    // Accessed only by the batcher thread. MRU is at index 0.
    private readonly List<PrefixCacheEntry> _prefixCache = [];

    // Issue #183 Gap 3: max total KV tokens committed across admitted sequences. Each
    // sequence reserves promptTokens + MaxNewTokens (clamped to MaxSeqLen) at admission
    // and releases the reservation when it retires. long.MaxValue = unlimited.
    private readonly long _kvTokenBudget;
    private long _committedTokens; // batcher-thread only

    private readonly Channel<PendingRequest> _queue =
        Channel.CreateUnbounded<PendingRequest>(new UnboundedChannelOptions { SingleWriter = false, SingleReader = true });
    private readonly Task _batcherTask;
    private volatile bool _disposed;

    // Observability counters (updated via Interlocked).
    private int _pendingCount;
    private int _activeCount;
    private long _batchedArgmaxSteps;
    private long _batchedFullLogitsSteps;
    private long _batchedArgmaxSequences;
    private long _batchedFullLogitsSequences;

    private sealed class PendingRequest(string prompt, SamplingParams sp, CancellationToken ct, Channel<GenerateChunk> output,
        string? canonicalHistoryPrefix, RetainedSequenceState? retainedState = null,
        ISequenceKvCache? retainedCache = null, int turnStartPosition = 0,
        Func<int, bool>? reservationRenewer = null)
    {
        public readonly string Prompt = prompt;
        public readonly SamplingParams Sp = sp;
        public readonly CancellationToken Ct = ct;
        public readonly Channel<GenerateChunk> Output = output;
        public readonly string? CanonicalHistoryPrefix = canonicalHistoryPrefix;
        public readonly RetainedSequenceState? RetainedState = retainedState;
        public ISequenceKvCache? RetainedCache = retainedCache;
        public readonly int TurnStartPosition = turnStartPosition;
        public readonly Func<int, bool>? ReservationRenewer = reservationRenewer;
        public int[]? Tokens; // memoized tokenization (admission may retry under backpressure)
    }

    private sealed class PrefixCacheEntry(int[] tokens, ISequenceKvCache cache, long bytes)
    {
        public readonly int[] Tokens = tokens;
        public readonly ISequenceKvCache Cache = cache;
        public readonly long Bytes = bytes;
    }

    /// <summary>A sequence whose prompt is being prefilled chunk-by-chunk.</summary>
    private sealed class PrefillingSeq
    {
        public required PendingRequest Req;
        public required int[] Tokens;
        public required ISequenceKvCache Cache;
        public required long ProjectedTokens; // KV budget reservation, released on retire
        public RetainedSequenceState? RetainedState;
        public int TurnStartPosition;
        public int StartPosition;
        public int Consumed;                  // prompt tokens prefilled so far
        public int InitialConsumed;           // reused immutable prefix; mixed snapshots are not promoted
        public int CacheablePrefixLength;     // aligned canonical/request prefix to retain after prefill
        public Func<int, bool>? ReservationRenewer;
    }

    private sealed class ActiveSeq
    {
        public required int CurrentToken;
        public required int Position;       // position at which CurrentToken will be decoded
        public required ISequenceKvCache Cache;
        public required SamplingParams Sp;
        public required Channel<GenerateChunk> Output;
        public required System.Collections.Immutable.ImmutableArray<int> StopIds;
        public required Random Rng;
        public required CancellationToken Ct;
        public required long ProjectedTokens;
        public RetainedSequenceState? RetainedState;
        public int TurnStartPosition;
        public List<int>? GeneratedTokenIds;
        public Func<int, bool>? ReservationRenewer;
        /// <summary>
        /// Set when the rolling reservation cannot fund the token this step would produce. The
        /// step still runs so the pending token materialises; the newly sampled token is then
        /// discarded unlogged and unemitted via the `done` disjunction below.
        /// </summary>
        public bool StopAfterStep;
        // Per-sequence grammar/FSM constraint (issue #374/#377). Null = unconstrained decoding for
        // this sequence (the common case). When non-null the batcher advances it with every emitted
        // token via Accept and, while it reports IsConstraining, masks this sequence's logits row via
        // Filter before sampling — forcing tool-call arguments to satisfy the schema. The instance is
        // built per request (ChatTemplateRenderer.BuildToolArgumentConstraint) and is single-request.
        public ITokenConstraint? Constraint;
        // Per-sequence constrained-choice state (SamplingParams.AllowedChoices). Null = no choice
        // constraint (the common case). When non-null the batcher masks this sequence's logits row
        // to the trie's currently-allowed continuation tokens before sampling, advances it with the
        // chosen token via Advance, and stops the sequence the moment IsComplete is true — mirrors
        // the choice-constraint semantics the now-deleted InferenceSession implemented directly.
        public TokenChoiceTrie.ChoiceState? ChoiceState;
        /// <summary>
        /// Set when the choice constraint was already complete after the very first sampled token
        /// (e.g. "YES" is itself an allowed choice). That token's KV row is not yet materialised —
        /// materialising it still requires one more decode step, like the rolling-reservation
        /// off-by-one below — so this behaves like <see cref="StopAfterStep"/>: the step runs, the
        /// token it produces is discarded unlogged/unemitted, and the sequence retires as a natural
        /// completion (no truncation flag), not a stop token or a budget cutoff.
        /// </summary>
        public bool ChoiceAlreadyComplete;
        public int TokenCount;
        // Per-sequence stateful UTF-8 decoders: reassembles multi-byte characters
        // split across tokens (CJK, emoji, smart quotes). Independent decoders for
        // the answer stream and reasoning stream so neither pollutes the other.
        public Utf8StreamDecoder TextDec = new();
        public Utf8StreamDecoder ThinkDec = new();
        public bool InThinking;
        public int ThinkingCount;  // tokens accumulated in the current <think> block (resets on each <think> open)
        public List<int>? RecentTokens;
    }

    /// <param name="fwd">Forward pass implementation. Owned externally — caller disposes.</param>
    /// <param name="tokenizer">Tokenizer matching the model vocabulary.</param>
    /// <param name="modelId">Human-readable model identifier returned in API responses.</param>
    /// <param name="maxBatchSize">Maximum concurrent decode sequences. Clamped to ≥ 1.</param>
    /// <param name="thinkTokenId">
    /// Token ID of the model's <c>&lt;think&gt;</c> marker, or <c>-1</c> if the model has no reasoning
    /// stream. When <c>-1</c>, all chunks are emitted as <see cref="GenerateChunkKind.Text"/>.
    /// </param>
    /// <param name="endThinkTokenId">
    /// Token ID of the model's <c>&lt;/think&gt;</c> marker, or <c>-1</c>. Must be paired with
    /// a non-negative <paramref name="thinkTokenId"/> to enable reasoning-stream splitting.
    /// </param>
    /// <param name="prefillChunkTokens">
    /// Prompt tokens prefilled per batcher iteration (issue #183 Gap 1); active sequences
    /// decode one step between chunks. <c>0</c> = prefill each prompt in one blocking call.
    /// <c>-1</c> = auto (issue #189): <c>64</c> when the forward pass's dequant-once weight
    /// cache covers the model — small chunks are then nearly free, so a low decode-stall
    /// chunk is safe — otherwise <c>256</c> to keep the per-chunk re-dequant amortized.
    /// </param>
    /// <param name="kvBudgetBytes">
    /// KV-memory budget gating admission (issue #183 Gap 3). <c>0</c> = auto (half of
    /// available system RAM), negative = unlimited, positive = explicit byte budget.
    /// </param>
    /// <param name="prefixCacheBytes">
    /// Retained immutable prefix-KV budget. <c>0</c> disables it, negative is unlimited, and the
    /// default is 256 MiB. Only backends implementing <see cref="IPrefixCacheableBatchedForwardPass"/>
    /// use it; CUDA deliberately opts out until it has ref-counted device pages.
    /// </param>
    public ContinuousBatchingEngine(
        IBatchedForwardPass fwd,
        ITokenizer tokenizer,
        string modelId,
        int maxBatchSize = 8,
        int thinkTokenId = -1,
        int endThinkTokenId = -1,
        int prefillChunkTokens = -1,
        long kvBudgetBytes = 0,
        long prefixCacheBytes = 256L * 1024 * 1024,
        IKvCache? kvCache = null)
    {
        ArgumentNullException.ThrowIfNull(fwd);
        ArgumentNullException.ThrowIfNull(tokenizer);
        _fwd = fwd;
        _tokenizer = tokenizer;
        ModelId = modelId;
        _maxBatchSize = Math.Max(1, maxBatchSize);
        _thinkTokenId = thinkTokenId;
        _endThinkTokenId = endThinkTokenId;
        _thinkingEnabled = thinkTokenId >= 0 && endThinkTokenId >= 0;

        // -1 = auto (issue #189): a small chunk minimizes decode stall but normally collapses
        // prefill throughput (per-chunk weight re-dequant); pick it only when the dequant-once
        // cache covers the model so small chunks re-pay no dequant. Otherwise keep 256.
        _prefillChunkTokens = prefillChunkTokens >= 0
            ? prefillChunkTokens
            : (fwd.PrefillDequantCacheActive ? 64 : 256);
        _chunkingEnabled = _prefillChunkTokens > 0 && !fwd.SnapKvEnabled;
        _prefixFwd = fwd as IPrefixCacheableBatchedForwardPass;
        _prefixCacheBytes = _prefixFwd is null || fwd.SnapKvEnabled
            ? 0
            : prefixCacheBytes < 0 ? long.MaxValue : prefixCacheBytes;

        long budgetBytes = kvBudgetBytes switch
        {
            < 0 => long.MaxValue,
            0 => GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / 2,
            _ => kvBudgetBytes,
        };
        _kvTokenBudget = budgetBytes == long.MaxValue
            ? long.MaxValue
            : Math.Max(1, budgetBytes / Math.Max(1, fwd.KvBytesPerToken));

        if (kvCache != null)
        {
            _kvCache = kvCache;
            _ownsKvCache = false;
        }
        else
        {
            // This cache is not consulted by real admission control (see the KvCache property
            // doc) — it exists only for callers that want to observe/compose with it directly.
            // Since nothing here writes to it, it must never be sized off _kvTokenBudget: the
            // default "auto" budget (half of available system RAM, see kvBudgetBytes above) is
            // a large but finite number, not long.MaxValue, so a budget-scaled placeholder here
            // eagerly allocates on the order of half the machine's RAM (CpuKvCache's constructor
            // is O(totalPages) — a billion-page cache does a billion-iteration enqueue loop in
            // the constructor itself) for a cache nothing ever reads from by default. A small
            // fixed placeholder is correct regardless of what the real budget resolves to.
            const int totalPages = 4_096;
            _kvCache = new CpuKvCache(totalPages, pageSizeTokens: 32, bytesPerToken: Math.Max(1, fwd.KvBytesPerToken));
            _ownsKvCache = true;
        }

        _batcherTask = Task.Run(BatcherLoop);
    }

    public string ModelId { get; }

    /// <summary>
    /// A paged KV cache instance sized from the same budget as real admission control, exposed for
    /// callers that want to observe/compose with it directly. <b>Not currently consulted by this
    /// engine's own admission control</b> — that runs entirely on <see cref="_committedTokens"/> /
    /// <see cref="_kvTokenBudget"/>, tracked independently. Reading <see cref="IKvCache.GetStatistics"/>
    /// here does not reflect this engine's real traffic.
    /// </summary>
    public IKvCache KvCache => _kvCache;

    /// <summary>Number of requests queued but not yet being generated.</summary>
    public int QueueDepth => _pendingCount;

    /// <summary>Number of requests currently being prefilled or decoded.</summary>
    public int ActiveRequests => _activeCount;

    /// <summary>Decode iterations that used the device-side greedy argmax tail.</summary>
    public long BatchedArgmaxSteps => Interlocked.Read(ref _batchedArgmaxSteps);

    /// <summary>Decode iterations that had to download full logits for sampling, constraints, or forced tokens.</summary>
    public long BatchedFullLogitsSteps => Interlocked.Read(ref _batchedFullLogitsSteps);

    /// <summary>Total sequence rows processed through the device-side greedy argmax tail.</summary>
    public long BatchedArgmaxSequences => Interlocked.Read(ref _batchedArgmaxSequences);

    /// <summary>Total sequence rows processed through the full-logits fallback path.</summary>
    public long BatchedFullLogitsSequences => Interlocked.Read(ref _batchedFullLogitsSequences);

    /// <summary>Total KV token budget gating admission (issue #183 Gap 3).</summary>
    public long KvTokenBudget => _kvTokenBudget;

    /// <summary>Backend-reported KV bytes per logical token, used by session residency planning.</summary>
    public long KvBytesPerToken => _fwd.KvBytesPerToken;

    /// <summary>Maximum logical sequence length accepted by the active backend.</summary>
    public int MaxSequenceLength => _fwd.MaxSeqLen;

    /// <summary>
    /// docs/028 Phase 2: forks up to <paramref name="maxPrefixLength"/> leading positions off
    /// <paramref name="source"/>'s retained cache for a sibling session to seed from, using this
    /// engine's own <see cref="IPrefixCacheableBatchedForwardPass"/> — the same mechanism this
    /// engine's cross-request prefix cache already relies on for non-retained admissions, now
    /// reused for cross-session sharing rather than duplicated. Returns null when the active
    /// backend doesn't support prefix forking at all (the CPU forward pass does; CUDA
    /// deliberately opts out); <see cref="RetainedSequenceState.TryForkSharedPrefix"/> covers the
    /// remaining reasons (source in use, nothing retained, below one alignment block).
    /// </summary>
    internal (ISequenceKvCache Cache, int Length)? TryForkSharedPrefix(RetainedSequenceState source, int maxPrefixLength)
    {
        if (_prefixFwd is null) return null;
        var forked = source.TryForkSharedPrefix(_prefixFwd, maxPrefixLength);
        if (forked is { } result) Interlocked.Add(ref _crossSessionPrefixTokensShared, result.Length);
        return forked;
    }

    /// <inheritdoc/>
    public bool IsContinuousBatching => true;

    /// <summary>Configured upper bound for each incremental prefill work item.</summary>
    public int PrefillChunkTokens => _prefillChunkTokens;

    /// <summary>KV tokens currently reserved by admitted requests.</summary>
    public long CommittedKvTokens => Volatile.Read(ref _committedTokens);

    /// <summary>Configured prefix-cache capacity in bytes; <see cref="long.MaxValue"/> means unlimited.</summary>
    public long PrefixCacheBudgetBytes => _prefixCacheBytes;

    /// <summary>Bytes occupied by retained immutable prefix snapshots.</summary>
    public long PrefixCacheUsedBytes => Volatile.Read(ref _prefixCacheUsedBytes);

    /// <summary>Number of retained immutable prefix snapshots.</summary>
    public int PrefixCacheEntries => Volatile.Read(ref _prefixCacheEntryCount);

    /// <summary>Successful prefix-cache lookups.</summary>
    public long PrefixCacheHits => Interlocked.Read(ref _prefixCacheHits);

    /// <summary>Prefix-cache lookups which found no reusable prefix.</summary>
    public long PrefixCacheMisses => Interlocked.Read(ref _prefixCacheMisses);

    /// <summary>
    /// docs/028 Phase 2: total tokens shared via <see cref="TryForkSharedPrefix"/> (cross-session
    /// forking, driven by <c>HotSessionRuntime.CreateWithSharedPrefixHint</c>) — deliberately a
    /// separate counter from <see cref="PrefixCacheHits"/>/<see cref="PrefillTokensReused"/>, which
    /// track this engine's own LRU prefix cache for non-retained admissions, a related but
    /// distinct mechanism.
    /// </summary>
    public long CrossSessionPrefixTokensShared => Interlocked.Read(ref _crossSessionPrefixTokensShared);

    /// <summary>Prefix snapshots evicted to honour the configured cache capacity.</summary>
    public long PrefixCacheEvictions => Interlocked.Read(ref _prefixCacheEvictions);

    /// <summary>
    /// Total tokens served from retained prefixes — i.e. prefill work skipped. Hit COUNT alone
    /// does not say this: one hit reusing 4000 tokens and one reusing 8 are both "a hit".
    /// </summary>
    public long PrefixCacheHitTokens => Interlocked.Read(ref _prefixCacheHitTokens);

    /// <summary>Tokens reused by the most recent prefix-cache hit.</summary>
    public int PrefixCacheLastHitLength => Volatile.Read(ref _prefixCacheLastHitLength);

    /// <summary>
    /// Divergence point of the most recent MISS: how many leading tokens matched the best
    /// eligible retained prefix before the sequences differed. A miss with a long match means the
    /// retained prefixes are cut at the wrong granularity, not that the prompt was unrelated —
    /// two situations the miss counter cannot tell apart.
    /// </summary>
    public int PrefixCacheLastMissLongestMatch => Volatile.Read(ref _prefixCacheLastMissLongestMatch);

    /// <summary>Cumulative <see cref="PrefixCacheLastMissLongestMatch"/>: tokens that matched a
    /// retained prefix but were re-prefilled anyway. This is the recoverable-work estimate.</summary>
    public long PrefixCacheMissMatchedTokens => Interlocked.Read(ref _prefixCacheMissMatchedTokens);

    /// <summary>Whether this backend can retain and fork immutable KV-prefix pages.</summary>
    public bool PrefixCacheEnabled => _prefixCacheBytes > 0;

    /// <inheritdoc/>
    public long PrefillTokensReused => Interlocked.Read(ref _prefillTokensReused);

    /// <summary>
    /// Back-compat string-stream view of <see cref="GenerateChunksAsync"/>: yields only
    /// user-facing answer text, suppressing reasoning chunks. Equivalent to the default
    /// interface-method implementation on <see cref="IInferenceEngine"/> but callable
    /// directly on the concrete type.
    /// </summary>
    public async IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        SamplingParams sp,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        await foreach (var c in GenerateChunksAsync(prompt, sp, ct).WithCancellation(ct).ConfigureAwait(false))
        {
            if (c.Kind == GenerateChunkKind.Text)
                yield return c.Text;
        }
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<GenerateChunk> GenerateChunksAsync(
        string prompt,
        SamplingParams sp,
        [EnumeratorCancellation] CancellationToken ct = default,
        string? canonicalHistoryPrefix = null)
    {
        ValidateOutputBound(sp);
        // Grammar-constrained decoding (issue #374) is wired into the batched sampler (issue #377):
        // sp.Constraint flows through admission onto the per-sequence ActiveSeq, where the batcher
        // masks that sequence's logits row while it is constraining and advances it on every token.

        var channel = Channel.CreateUnbounded<GenerateChunk>(
            new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });

        Interlocked.Increment(ref _pendingCount);
        try
        {
            await _queue.Writer.WriteAsync(new PendingRequest(prompt, sp, ct, channel, canonicalHistoryPrefix), ct)
                .ConfigureAwait(false);
        }
        catch
        {
            // Disposed concurrently (writer completed) or caller-cancelled before the
            // write landed — undo the counter so QueueDepth doesn't drift.
            Interlocked.Decrement(ref _pendingCount);
            throw;
        }

        await foreach (var chunk in channel.Reader.ReadAllAsync(ct).ConfigureAwait(false))
            yield return chunk;
    }

    /// <summary>
    /// Generates one turn while retaining its backend-private cache in <paramref name="state"/>
    /// for a later appended turn. <paramref name="prompt"/> is an append-only suffix: callers
    /// own the authoritative execution log and must not repeat tokens already materialized by the
    /// state. The current implementation is intentionally limited to caches that implement
    /// <see cref="IRewindableSequenceKvCache"/>; unsupported backends fail closed.
    /// </summary>
    public async IAsyncEnumerable<GenerateChunk> GenerateRetainedChunksAsync(
        string prompt,
        SamplingParams sp,
        RetainedSequenceState state,
        [EnumeratorCancellation] CancellationToken ct = default,
        Func<int, bool>? reservationRenewer = null)
    {
        ArgumentNullException.ThrowIfNull(state);
        ValidateOutputBound(sp);
        var lease = state.Reserve();
        var channel = Channel.CreateUnbounded<GenerateChunk>(
            new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });

        Interlocked.Increment(ref _pendingCount);
        try
        {
            await _queue.Writer.WriteAsync(
                new PendingRequest(prompt, sp, ct, channel, canonicalHistoryPrefix: null, state,
                    lease.Cache, lease.TurnStartPosition, reservationRenewer), ct).ConfigureAwait(false);
        }
        catch
        {
            Interlocked.Decrement(ref _pendingCount);
            state.Fail(lease.Cache);
            throw;
        }

        await foreach (var chunk in channel.Reader.ReadAllAsync(ct).ConfigureAwait(false))
            yield return chunk;
    }

    private static void ValidateOutputBound(SamplingParams sampling)
    {
        if (sampling.MaxNewTokens > MaxBufferedOutputChunks - 4)
            throw new ArgumentOutOfRangeException(nameof(sampling),
                $"MaxNewTokens exceeds the non-blocking output limit of {MaxBufferedOutputChunks} chunks.");
    }

    private async Task BatcherLoop()
    {
        var pending = new Queue<PendingRequest>();
        var prefilling = new List<PrefillingSeq>();
        var active = new List<ActiveSeq>(_maxBatchSize);
        var tokensBuf = new int[_maxBatchSize];
        var posBuf = new int[_maxBatchSize];
        var cacheBuf = new ISequenceKvCache[_maxBatchSize];

        // An exception the per-request handlers didn't isolate (e.g. a backend failure
        // inside BatchForwardMulti) must not kill the batcher silently: without the
        // catch + finally-drain below, every in-flight caller would hang forever on a
        // never-completed channel and the per-sequence caches would leak.
        Exception? fatal = null;
        try
        {
        while (!_disposed)
        {
            // Issue #302: rebind the backend's thread-affine context (CUDA) before any forward
            // work this iteration. The loop's `await WaitToReadAsync().ConfigureAwait(false)` on
            // the idle path can resume the continuation on a different thread-pool thread than the
            // one that ran the previous step, so a single bind before the loop wouldn't hold. The
            // call is a no-op on CPU/Vulkan and free after the first call per thread.
            _fwd.BindToCurrentThread();

            // Pull everything queued so far into the local pending queue. Channels have
            // no peek, and admission backpressure needs to inspect the head request's
            // size without consuming it — hence the local FIFO.
            while (_queue.Reader.TryRead(out var queued))
                pending.Enqueue(queued);

            AdmitPending(pending, prefilling, active);

            if (active.Count == 0 && prefilling.Count == 0)
            {
                // No active work — wait for a new request. (Admission always starts the
                // head request when nothing is running, so pending is empty here.)
                try
                {
                    bool hasMore = await _queue.Reader.WaitToReadAsync().ConfigureAwait(false);
                    if (!hasMore) break;
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                continue;
            }

            // Advance prefilling prompts by one chunk, then give every active sequence
            // one decode step — the interleave that keeps decode from stalling.
            if (prefilling.Count > 0)
                RunPrefillStep(prefilling, active);

            // Rolling reservation renewal.
            //
            // THE OFF-BY-ONE THAT MAKES THIS SUBTLE: the decode loop always carries exactly one
            // sampled-but-unmaterialised token. A token sampled at step k has its KV written at
            // step k+1, when it is fed back as CurrentToken. So entering a step at Position P, the
            // step writes KV at P (materialising P+1 positions) and PRODUCES a token that will
            // occupy P+1 and only materialise on the following step — needing P+2.
            //
            // Renewing for P+1 and stopping immediately therefore leaves the pending token logged
            // and ALREADY EMITTED to the consumer but never materialised, so the log has to drop
            // it to keep accepted == materialized — and the caller has then seen text for a token
            // the execution log does not contain, which breaks the replay contract §8.1 rests on.
            //
            // Instead: ask for P+2 — "can I afford the token this step would produce?" — and on
            // failure mark the sequence to stop AFTER the step. The step still runs, materialising
            // the pending token, and the newly sampled token is discarded before it is logged or
            // emitted, exactly as the EOS and max-token paths already do. Nothing is emitted that
            // is not also logged, and no log surgery is needed.
            //
            // Entering a step at P, the reservation already covers P+1: either from the initial
            // reservation (prompt + 1) or from the previous step's successful renewal of P+2.
            for (int i = 0; i < active.Count; i++)
            {
                var seq = active[i];
                if (seq.ReservationRenewer is not null && !seq.ReservationRenewer(seq.Position + 2))
                    seq.StopAfterStep = true;
            }

            if (active.Count == 0)
                continue;

            // Build batched inputs
            int n = active.Count;
            for (int i = 0; i < n; i++)
            {
                tokensBuf[i] = active[i].CurrentToken;
                posBuf[i] = active[i].Position;
                cacheBuf[i] = active[i].Cache;
            }

            // Batched decode step (shares weight reads across N sequences). When EVERY active
            // sequence is plain greedy this step (temp ≤ 0, not force-closing </think>), take the
            // on-device argmax tail (#205/#206): it returns just the per-seq (token, logit) and
            // skips the full N×vocab logits D2H + host split. A single sampled or force-closing
            // seq reverts the whole step to the full-logits path (the argmax buffer can't sample).
            // A sequence that is actively constraining (issue #377) also needs the full logits row to
            // mask, so it likewise reverts the step — but only WHILE constraining: a constraint-bearing
            // request keeps the fast path in its unconstrained spans (its tokens still flow through
            // Accept below from the argmax tail). With no constraint anywhere this is byte-identical.
            bool allGreedy = _fwd.SupportsBatchedGpuArgmax;
            for (int i = 0; i < n && allGreedy; i++)
            {
                var s = active[i];
                bool forcedClose = _thinkingEnabled && s.InThinking && s.Sp.MaxThinkingTokens > 0
                                   && s.ThinkingCount >= s.Sp.MaxThinkingTokens && _endThinkTokenId > 0;
                // The raw GPU argmax has no visibility of per-sequence token history.
                // Presence/frequency/repetition penalties can therefore change even a
                // temperature-zero choice, so those requests must use full logits.
                if (s.Sp.Temperature > 0f || s.Sp.HasHistoryPenalty || forcedClose || s.Constraint is { IsConstraining: true }
                    || s.ChoiceState is not null)
                    allGreedy = false;
            }

            float[][]? logitsBatch = null;
            (int Token, float Logit)[]? argmaxBatch = null;
            if (allGreedy)
            {
                argmaxBatch = _fwd.BatchForwardMultiArgmax(tokensBuf[..n], posBuf[..n], cacheBuf[..n]);
                Interlocked.Increment(ref _batchedArgmaxSteps);
                Interlocked.Add(ref _batchedArgmaxSequences, n);
            }
            else
            {
                logitsBatch = _fwd.BatchForwardMulti(tokensBuf[..n], posBuf[..n], cacheBuf[..n]);
                Interlocked.Increment(ref _batchedFullLogitsSteps);
                Interlocked.Add(ref _batchedFullLogitsSequences, n);
            }

            // Process results in reverse order so RemoveAt indices stay valid
            for (int i = n - 1; i >= 0; i--)
            {
                var seq = active[i];

                // Force-inject </think> on budget overrun. Mirrors InferenceEngine /
                // RunCommand.DecodeLoop: the forced close token threads through the
                // boundary branch below and is fed back as the next CurrentToken so the
                // model continues from its post-think state on the next batched step.
                int next;
                if (argmaxBatch is not null)
                {
                    // All-greedy fast path: the on-device argmax already picked each token (the
                    // gate above excluded any sampled / force-closing seq from this branch).
                    next = argmaxBatch[i].Token;
                }
                else if (_thinkingEnabled && seq.InThinking && seq.Sp.MaxThinkingTokens > 0
                    && seq.ThinkingCount >= seq.Sp.MaxThinkingTokens && _endThinkTokenId > 0)
                {
                    next = _endThinkTokenId;
                }
                else
                {
                    // While this sequence's constraint is inside a constrained region, sample from the
                    // masked logits (grammar-forbidden tokens set to -inf); otherwise sample the raw
                    // logits exactly as the unconstrained path would (issue #377). The gate above kept
                    // a constraining seq off the argmax fast path, so logitsBatch is populated here.
                    // A choice constraint masks first (in place — the row is this step's own fresh
                    // array), then the grammar constraint's own filter applies on top, matching the
                    // order ActivateSeq uses for the first token.
                    var row = logitsBatch![i];
                    if (seq.ChoiceState is not null)
                        seq.ChoiceState.MaskLogits(row);
                    var rowLogits = seq.Constraint is { IsConstraining: true } ctr
                        ? ctr.Filter(row)
                        : (ReadOnlySpan<float>)row;
                    var spWithHistory = seq.RecentTokens is { Count: > 0 }
                        ? seq.Sp with { PreviousTokens = seq.RecentTokens }
                        : seq.Sp;
                    next = seq.Sp.Temperature <= 0f && !spWithHistory.HasHistoryPenalty
                        ? Sampler.Greedy(rowLogits)
                        : Sampler.Sample(rowLogits, spWithHistory, seq.Rng);
                }

                // Advance the grammar constraint with the just-chosen token (every token, including the
                // argmax-tail and force-close paths, so it can detect a tool-call boundary and
                // begin/end constraining). No-op when this sequence has no constraint.
                seq.Constraint?.Accept(next);
                bool isChoiceDone = false;
                if (seq.ChoiceState is not null)
                {
                    if (!seq.ChoiceState.Advance(next))
                        throw new InvalidOperationException($"Sampled token {next} is not permitted by choice constraint.");
                    isChoiceDone = seq.ChoiceState.IsComplete;
                }
                seq.RecentTokens?.Add(next);

                bool stoppedByStopToken = seq.StopIds.Contains(next);
                bool cancelled = seq.Ct.IsCancellationRequested;
                // isChoiceDone does NOT end the turn on this same iteration: `next` was chosen but
                // its own KV row is not yet materialised (off-by-one, see the reservation-renewal
                // comment above), so it still needs to flow through the normal emit-and-advance path
                // below before the sequence can retire. Marking StopAfterStep there defers the
                // actual stop to the following step, exactly like a reservation running out.
                bool done = stoppedByStopToken
                    || seq.TokenCount >= seq.Sp.MaxNewTokens
                    || seq.StopAfterStep
                    || cancelled
                    || seq.ChoiceAlreadyComplete;

                if (done)
                {
                    // Only "not a stop token and not cancelled" can mean the MaxNewTokens
                    // disjunct fired — the three conditions are mutually exhaustive above.
                    //
                    // ORDER IS LOAD-BEARING: RetireSeq publishes the turn outcome
                    // (RetainedSequenceState.LastTurn); FlushAndComplete completes the output
                    // channel, which releases the consumer's `await foreach`. HotSession reads
                    // LastTurn the instant it resumes, so completing the channel first is a race
                    // it loses whenever the scheduler runs it before RetireSeq — surfacing as
                    // "The retained engine completed without a retained turn outcome" and a Failed
                    // operation on a turn that actually succeeded.
                    //
                    // Retiring first is safe: HotSession serialises turns behind its own gate and
                    // cannot begin another until RunTurnAsync returns, which is after the foreach
                    // ends — i.e. strictly after the channel completes below.
                    RetireSeq(seq, rollback: cancelled);
                    // Budget exhaustion is reported on its own flag. It used to fall into the
                    // max-tokens disjunct, telling the caller "you hit your token limit" when the
                    // truth was "the server could not fund another token" — opposite remedies.
                    bool byBudget = seq.StopAfterStep && !stoppedByStopToken && !cancelled && !seq.ChoiceAlreadyComplete;
                    FlushAndComplete(seq,
                        truncatedByMaxTokens: !stoppedByStopToken && !cancelled && !byBudget && !seq.ChoiceAlreadyComplete,
                        truncatedByResourceBudget: byBudget);
                    active.RemoveAt(i);
                }
                else
                {
                    seq.GeneratedTokenIds?.Add(next);
                    if (isChoiceDone)
                    {
                        // `next` completes the allowed choice and is emitted normally below; there
                        // is nothing left to mask/advance, and one more decode step is still needed
                        // to materialise `next`'s own KV row before the sequence can retire (see the
                        // `done` comment above).
                        seq.ChoiceState = null;
                        seq.ChoiceAlreadyComplete = true;
                    }
                    // Counter update mirrors RunCommand.DecodeLoop: reset on each <think>
                    // open, otherwise increment whenever seq.InThinking was true on entry to
                    // this step. That includes the </think> boundary token — so N reasoning
                    // tokens trip the force-close on step N+1.
                    if (_thinkingEnabled && next == _thinkTokenId) seq.ThinkingCount = 0;
                    else if (seq.InThinking) seq.ThinkingCount++;

                    // Reasoning boundary tokens are ALWAYS consumed and never emitted. State flips
                    // only on a valid transition, so a malformed double-open or an orphan close is
                    // swallowed rather than leaking its literal marker as text — e.g. a bare Gemma 4
                    // <channel|> close with no preceding open (issue #304).
                    if (_thinkingEnabled && next == _thinkTokenId)
                    {
                        if (!seq.InThinking)
                        {
                            var textTail = seq.TextDec.Flush();
                            if (textTail.Length > 0)
                                seq.Output.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Text, textTail));
                            seq.InThinking = true;
                        }
                    }
                    else if (_thinkingEnabled && next == _endThinkTokenId)
                    {
                        if (seq.InThinking)
                        {
                            var thinkTail = seq.ThinkDec.Flush();
                            if (thinkTail.Length > 0)
                                seq.Output.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Thinking, thinkTail));
                            seq.InThinking = false;
                        }
                    }
                    else
                    {
                        var bytes = _tokenizer.DecodeBytes(next);
                        if (seq.InThinking)
                        {
                            var chunk = seq.ThinkDec.Append(bytes);
                            if (chunk.Length > 0)
                                seq.Output.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Thinking, chunk));
                        }
                        else
                        {
                            var chunk = seq.TextDec.Append(bytes);
                            if (chunk.Length > 0)
                                seq.Output.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Text, chunk));
                        }
                    }

                    seq.CurrentToken = next;
                    seq.Position++;
                    seq.TokenCount++;
                }
            }
        }
        }
        catch (Exception ex)
        {
            fatal = ex;
        }
        finally
        {
            // Drain: complete everything still in flight — with the fatal exception when
            // the loop died, so callers observe the failure instead of hanging. Pull any
            // requests still sitting in the channel first (written between our last
            // TryRead pass and the writer completing).
            while (_queue.Reader.TryRead(out var stranded))
                pending.Enqueue(stranded);
            foreach (var req in pending)
            {
                Interlocked.Decrement(ref _pendingCount);
                req.Output.Writer.TryComplete(fatal);
                req.RetainedState?.Fail(req.RetainedCache);
            }
            pending.Clear();
            foreach (var p in prefilling)
            {
                p.Req.Output.Writer.TryComplete(fatal);
                if (p.RetainedState is { } state && fatal is null)
                    state.RollbackAndComplete(p.Cache, p.TurnStartPosition);
                else
                {
                    p.RetainedState?.Fail(p.Cache);
                    if (p.RetainedState is null) p.Cache.Dispose();
                }
                Interlocked.Decrement(ref _activeCount);
            }
            prefilling.Clear();
            foreach (var seq in active)
            {
                // Engine shutdown mid-generation (e.g. Dispose()) cuts these sequences off
                // involuntarily, not via a natural stop token — report it as truncated so
                // the client doesn't see a false "stop"/"end_turn" for incomplete content.
                if (fatal is null) FlushAndComplete(seq, truncatedByMaxTokens: true);
                else seq.Output.Writer.TryComplete(fatal);
                if (seq.RetainedState is { } state && fatal is null)
                    state.RollbackAndComplete(seq.Cache, seq.TurnStartPosition);
                else
                {
                    seq.RetainedState?.Fail(seq.Cache);
                    if (seq.RetainedState is null) seq.Cache.Dispose();
                }
                Interlocked.Decrement(ref _activeCount);
            }
            active.Clear();
        }
    }

    /// <summary>
    /// Moves requests from the local pending queue into the prefilling set while batch
    /// slots are free and the KV token budget allows (issue #183 Gap 3). FIFO: a
    /// too-large head request blocks the queue until running sequences retire — except
    /// when nothing is running, where it is always admitted so a single oversized
    /// request can't deadlock the engine (it then fails in prefill if it truly can't fit).
    /// </summary>
    private void AdmitPending(Queue<PendingRequest> pending, List<PrefillingSeq> prefilling, List<ActiveSeq> active)
    {
        while (pending.Count > 0 && active.Count + prefilling.Count < _maxBatchSize)
        {
            var req = pending.Peek();
            if (req.Ct.IsCancellationRequested)
            {
                pending.Dequeue();
                Interlocked.Decrement(ref _pendingCount);
                req.Output.Writer.TryComplete();
                ReturnUnstartedRetainedState(req);
                continue;
            }

            int[] tokens;
            try
            {
                tokens = req.Tokens ??= _tokenizer.Encode(req.Prompt).ToArray();
            }
            catch (Exception ex)
            {
                pending.Dequeue();
                Interlocked.Decrement(ref _pendingCount);
                req.Output.Writer.TryComplete(ex);
                ReturnUnstartedRetainedState(req);
                continue;
            }
            if (tokens.Length == 0)
            {
                pending.Dequeue();
                Interlocked.Decrement(ref _pendingCount);
                req.Output.Writer.TryComplete();
                ReturnUnstartedRetainedState(req);
                continue;
            }

            long projected = Math.Min(
                req.TurnStartPosition + tokens.Length + Math.Max(0L, req.Sp.MaxNewTokens), _fwd.MaxSeqLen);
            if (_committedTokens + projected > _kvTokenBudget
                && (active.Count > 0 || prefilling.Count > 0))
            {
                break; // backpressure: re-evaluated next iteration as sequences retire
            }

            pending.Dequeue();
            Interlocked.Decrement(ref _pendingCount);

            ISequenceKvCache? cache = req.RetainedCache;
            int cacheablePrefixLength = req.RetainedState is null ? GetCacheablePrefixLength(req, tokens) : 0;
            int reused = 0;
            try
            {
                if (cache is not null)
                {
                    req.RetainedCache = null;
                    // Diagnostic guard (hot-session replay divergence investigation): the
                    // RetainedState path below already refuses to admit a rewindable cache whose
                    // LogicalPosition disagrees with the recorded turn boundary -- this in-process
                    // HotSession path reuses the cache object directly and had no equivalent check,
                    // so a drift between HotSession's own cursor bookkeeping and the cache's actual
                    // materialized position could silently write the next turn's tokens at the
                    // wrong position instead of failing loudly.
                    if (cache is IRewindableSequenceKvCache retainedRewindable
                        && retainedRewindable.LogicalPosition != req.TurnStartPosition)
                    {
                        throw new InvalidOperationException(
                            $"Retained cache logical position {retainedRewindable.LogicalPosition} does not " +
                            $"match turn start position {req.TurnStartPosition} (HotSession cursor drift at admission).");
                    }
                }
                else if (req.RetainedState is not null)
                {
                    cache = _fwd.CreateCache();
                    if (req.RetainedState.GetExportedKvBytes() is { Length: > 0 } rawKvBytes && cache is IPersistableSequenceKvCache persistable)
                    {
                        persistable.ImportKvState(rawKvBytes);
                    }
                    if (cache is IRewindableSequenceKvCache rewindable && req.TurnStartPosition > 0)
                    {
                        if (rewindable.CanRewindTo(req.TurnStartPosition))
                            rewindable.RewindTo(req.TurnStartPosition);
                    }
                }
                else if (cacheablePrefixLength > 0 && TryTakePrefix(tokens, cacheablePrefixLength, out var prefix))
                {
                    cache = _prefixFwd!.ForkPrefix(prefix.Cache);
                    reused = prefix.Tokens.Length;
                    Interlocked.Add(ref _prefillTokensReused, reused);
                }
                else
                {
                    cache = _fwd.CreateCache();
                }

                if (req.RetainedState is not null)
                {
                    if (cache is not IRewindableSequenceKvCache rewindable
                        || rewindable.LogicalPosition != req.TurnStartPosition)
                        throw new NotSupportedException(
                            "Retained sequence state requires an exact rewindable cache at its recorded turn boundary.");
                }
            }
            catch (Exception ex)
            {
                req.Output.Writer.TryComplete(ex);
                req.RetainedState?.Fail(req.RetainedCache ?? cache);
                continue;
            }

            _committedTokens += projected;
            // Surface the prompt-token count out-of-band so endpoints can report
            // usage.prompt_tokens / input_tokens without re-tokenizing (issue #150).
            // Emitted at admission, before any text/thinking chunk for this sequence.
            req.Output.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Usage, "", tokens.Length));
            prefilling.Add(new PrefillingSeq
            {
                Req = req,
                Tokens = tokens,
                Cache = cache!,
                ProjectedTokens = projected,
                RetainedState = req.RetainedState,
                TurnStartPosition = req.TurnStartPosition,
                StartPosition = req.TurnStartPosition,
                Consumed = reused,
                InitialConsumed = reused,
                CacheablePrefixLength = cacheablePrefixLength,
                ReservationRenewer = req.ReservationRenewer,
            });
            Interlocked.Increment(ref _activeCount);
        }
    }

    private int GetCacheablePrefixLength(PendingRequest req, int[] tokens)
    {
        if (_prefixCacheBytes <= 0 || _prefixFwd is null) return 0;
        int candidate = tokens.Length;
        if (!string.IsNullOrEmpty(req.CanonicalHistoryPrefix))
        {
            var canonical = _tokenizer.Encode(req.CanonicalHistoryPrefix).ToArray();
            if (canonical.Length == 0 || canonical.Length > tokens.Length
                || !tokens.AsSpan(0, canonical.Length).SequenceEqual(canonical))
                return 0; // never cache a caller hint that does not exactly match the request.
            candidate = canonical.Length;
        }
        // Keep at least one uncached prompt token: prefill's final-token logits seed the first
        // sampled output, and a snapshot intentionally stores KV pages rather than a vocab row.
        candidate = Math.Min(candidate, tokens.Length - 1);
        return candidate / _prefixFwd.PrefixCacheBlockSize * _prefixFwd.PrefixCacheBlockSize;
    }

    private static void ReturnUnstartedRetainedState(PendingRequest req)
    {
        if (req.RetainedState is not { } state) return;
        if (req.RetainedCache is { } cache)
        {
            req.RetainedCache = null;
            state.RollbackAndComplete(cache, req.TurnStartPosition);
        }
        else
        {
            // The state was reserved before admission but no cache exists yet, so releasing the
            // reservation leaves it empty and available for a future first turn.
            state.Fail(null);
        }
    }

    private bool TryTakePrefix(int[] tokens, int maxLength, out PrefixCacheEntry entry)
    {
        for (int i = 0; i < _prefixCache.Count; i++)
        {
            var candidate = _prefixCache[i];
            if (candidate.Tokens.Length <= maxLength
                && tokens.AsSpan(0, candidate.Tokens.Length).SequenceEqual(candidate.Tokens))
            {
                _prefixCache.RemoveAt(i);
                _prefixCache.Insert(0, candidate);
                Interlocked.Increment(ref _prefixCacheHits);
                Interlocked.Add(ref _prefixCacheHitTokens, candidate.Tokens.Length);
                Volatile.Write(ref _prefixCacheLastHitLength, candidate.Tokens.Length);
                entry = candidate;
                return true;
            }
        }
        Interlocked.Increment(ref _prefixCacheMisses);
        // Divergence diagnostic. A miss counter alone cannot distinguish "an unrelated prompt" from
        // "matched 4000 of 4096 tokens and then diverged" — and those call for opposite responses:
        // the first is expected, the second means the retained prefixes are cut at the wrong
        // granularity and nearly all of the reusable work is being thrown away. Recorded only on
        // the miss path, so the hit path keeps its vectorised SequenceEqual untouched.
        int best = 0;
        for (int i = 0; i < _prefixCache.Count; i++)
        {
            // Only ELIGIBLE candidates, matching the population the hit loop above considered.
            // Counting a too-long entry's overlap would report a reuse that was never available.
            var c = _prefixCache[i].Tokens;
            if (c.Length > maxLength) continue;
            int n = Math.Min(c.Length, tokens.Length);
            int k = 0;
            while (k < n && tokens[k] == c[k]) k++;
            if (k > best) best = k;
        }
        Volatile.Write(ref _prefixCacheLastMissLongestMatch, best);
        Interlocked.Add(ref _prefixCacheMissMatchedTokens, best);
        entry = null!;
        return false;
    }

    private void RetainPrefix(PrefillingSeq p)
    {
        if (p.RetainedState is not null || p.CacheablePrefixLength <= 0 || p.InitialConsumed != 0
            || _prefixFwd is null || _prefixCacheBytes <= 0) return;
        long bytes = checked((long)p.CacheablePrefixLength * _fwd.KvBytesPerToken);
        if (bytes > _prefixCacheBytes) return;
        // The prefix might have been populated from an existing snapshot. Keeping another copy
        // would waste a retained handle; the LRU hit above already moved that entry to MRU.
        for (int i = 0; i < _prefixCache.Count; i++)
            if (_prefixCache[i].Tokens.Length == p.CacheablePrefixLength
                && _prefixCache[i].Tokens.AsSpan().SequenceEqual(p.Tokens.AsSpan(0, p.CacheablePrefixLength)))
                return;

        ISequenceKvCache snapshot = _prefixFwd.CapturePrefix(p.Cache, p.CacheablePrefixLength);
        var entry = new PrefixCacheEntry(p.Tokens[..p.CacheablePrefixLength], snapshot, bytes);
        _prefixCache.Insert(0, entry);
        _prefixCacheUsedBytes += bytes;
        Volatile.Write(ref _prefixCacheEntryCount, _prefixCache.Count);
        while (_prefixCache.Count > 0 && _prefixCacheUsedBytes > _prefixCacheBytes)
        {
            int last = _prefixCache.Count - 1;
            var evicted = _prefixCache[last];
            _prefixCache.RemoveAt(last);
            _prefixCacheUsedBytes -= evicted.Bytes;
            evicted.Cache.Dispose();
            Interlocked.Increment(ref _prefixCacheEvictions);
        }
        Volatile.Write(ref _prefixCacheEntryCount, _prefixCache.Count);
    }

    /// <summary>
    /// Advances prefilling prompts by one chunk budget (issue #183 Gap 1). With several
    /// prompts in flight the chunk budget is split across them and run as one packed
    /// forward pass (Gap 2) so weight reads amortize across prompts. Prompts whose final
    /// chunk completes are sampled and promoted to the active decode set.
    /// </summary>
    private void RunPrefillStep(List<PrefillingSeq> prefilling, List<ActiveSeq> active)
    {
        // Cancellation sweep between chunks — a benefit chunking adds for free:
        // a cancelled long-prompt request stops mid-prefill instead of running to the end.
        for (int i = prefilling.Count - 1; i >= 0; i--)
        {
            var p = prefilling[i];
            if (p.Req.Ct.IsCancellationRequested)
            {
                p.Req.Output.Writer.TryComplete();
                DropPrefilling(prefilling, i);
            }
        }
        if (prefilling.Count == 0) return;

        if (!_chunkingEnabled)
        {
            // Unchunked path (prefillChunkTokens == 0, or SnapKV active — its prefill
            // eviction only runs on a whole-prompt startPos==0 prefill). One prompt per
            // iteration, blocking, exactly the pre-#183 behavior.
            var p = prefilling[0];
            float[] logits;
            try
            {
                var remaining = new ArraySegment<int>(p.Tokens, p.Consumed, p.Tokens.Length - p.Consumed);
                logits = _fwd.PrefillWithCache(remaining, p.Cache, startPos: p.StartPosition + p.Consumed).ToArray();
            }
            catch (Exception ex)
            {
                p.Req.Output.Writer.TryComplete(ex);
                DropPrefilling(prefilling, 0);
                return;
            }
            prefilling.RemoveAt(0);
            RetainPrefix(p);
            ActivateSeq(p, logits, active);
            return;
        }

        // Split this iteration's chunk budget across all prefilling prompts (≥1 token
        // each). The budget bounds the decode stall per iteration regardless of how
        // many prompts are prefilling, because the packed pass amortizes weights.
        int sCount = prefilling.Count;
        int perSeq = Math.Max(1, _prefillChunkTokens / sCount);

        var chunks = new ReadOnlyMemory<int>[sCount];
        var startPos = new int[sCount];
        var caches = new ISequenceKvCache[sCount];
        var wantLogits = new bool[sCount];
        var takes = new int[sCount];
        for (int s = 0; s < sCount; s++)
        {
            var p = prefilling[s];
            int take = Math.Min(p.Tokens.Length - p.Consumed, perSeq);
            chunks[s] = p.Tokens.AsMemory(p.Consumed, take);
            startPos[s] = p.StartPosition + p.Consumed;
            caches[s] = p.Cache;
            wantLogits[s] = p.Consumed + take == p.Tokens.Length;
            takes[s] = take;
        }

        float[]?[] logitsPerSeq;
        try
        {
            if (sCount == 1)
            {
                var p = prefilling[0];
                var segment = new ArraySegment<int>(p.Tokens, p.Consumed, takes[0]);
                var logits = _fwd.PrefillWithCache(segment, p.Cache, startPos: p.StartPosition + p.Consumed);
                logitsPerSeq = [wantLogits[0] ? logits.ToArray() : null];
            }
            else
            {
                logitsPerSeq = _fwd.PrefillPackedMulti(chunks, startPos, caches, wantLogits);
            }
        }
        catch (Exception ex)
        {
            // A failed packed pass leaves the involved caches in an indeterminate state;
            // fail every involved request rather than guessing which ones survived.
            for (int i = prefilling.Count - 1; i >= 0; i--)
            {
                prefilling[i].Req.Output.Writer.TryComplete(ex);
                DropPrefilling(prefilling, i);
            }
            return;
        }

        for (int s = sCount - 1; s >= 0; s--)
        {
            var p = prefilling[s];
            p.Consumed += takes[s];
            if (logitsPerSeq[s] is { } logits)
            {
                RetainPrefix(p);
                prefilling.RemoveAt(s);
                ActivateSeq(p, logits, active);
            }
        }
    }

    /// <summary>Removes prefilling[i], disposing its cache and releasing its budget reservation.</summary>
    private void DropPrefilling(List<PrefillingSeq> prefilling, int i)
    {
        var p = prefilling[i];
        if (p.RetainedState is { } state)
            state.RollbackAndComplete(p.Cache, p.TurnStartPosition);
        else
            p.Cache.Dispose();
        _committedTokens -= p.ProjectedTokens;
        prefilling.RemoveAt(i);
        Interlocked.Decrement(ref _activeCount);
    }

    /// <summary>Releases a retired decode sequence's cache and budget reservation.</summary>
    private void RetireSeq(ActiveSeq seq, bool rollback = false)
    {
        if (seq.RetainedState is { } state)
        {
            if (rollback) state.RollbackAndComplete(seq.Cache, seq.TurnStartPosition);
            else state.Complete(seq.Cache, seq.TurnStartPosition, seq.GeneratedTokenIds);
        }
        else
            seq.Cache.Dispose();
        _committedTokens -= seq.ProjectedTokens;
        Interlocked.Decrement(ref _activeCount);
    }

    private static void FlushAndComplete(ActiveSeq seq, bool truncatedByMaxTokens = false,
        bool truncatedByResourceBudget = false)
    {
        var textTail = seq.TextDec.Flush();
        if (textTail.Length > 0)
            seq.Output.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Text, textTail));
        var thinkTail = seq.ThinkDec.Flush();
        if (thinkTail.Length > 0)
            seq.Output.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Thinking, thinkTail));
        seq.Output.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Stop, "",
            TruncatedByMaxTokens: truncatedByMaxTokens,
            TruncatedByResourceBudget: truncatedByResourceBudget));
        seq.Output.Writer.TryComplete();
    }

    /// <summary>
    /// Promotes a fully prefilled prompt to the active decode set: samples the first
    /// token from the prefill logits, seeds the reasoning state machine, and emits the
    /// first chunk. A first token that is already a stop token completes the request
    /// immediately.
    /// </summary>
    private void ActivateSeq(PrefillingSeq p, float[] logits, List<ActiveSeq> active)
    {
        var req = p.Req;

        // Stop on ANY end-of-generation token, not just the configured EOS — matches the
        // single-user InferenceEngine path. A model with an alternate end token (e.g. Gemma's
        // <eos>, distinct from its <turn|> EOS) would otherwise decode it as text and run on.
        // StopTokenIds replaces this set; AdditionalStopTokenIds is unioned on top (issue #304).
        System.Collections.Immutable.ImmutableArray<int> stopIds =
            req.Sp.ResolveStopSet(_tokenizer.EogTokenIds);
        var rng = new Random();

        // Per-request grammar constraint (issue #374/#377). Reset to its watching state so a reused
        // instance can't carry state across requests, then mask the first sampled token if it is
        // already constraining (rare — would require the prompt to end mid-call). Advance it with the
        // first token below, exactly as the decode loop does for every subsequent token.
        var constraint = req.Sp.Constraint;
        constraint?.Reset();

        // Per-request constrained-choice state (SamplingParams.AllowedChoices). Built fresh per
        // request from the trie (cheap: a handful of short strings), masked into firstLogits before
        // the grammar constraint's own filter so a choice-constrained request also respects a
        // grammar if both are set — matches the order the InferenceSession implementation used.
        var choiceState = req.Sp.AllowedChoices is { Count: > 0 } allowedChoices
            ? TokenChoiceTrie.Build(allowedChoices, _tokenizer).CreateState()
            : null;

        var firstLogits = logits;
        if (choiceState is not null)
            choiceState.MaskLogits(firstLogits);
        ReadOnlySpan<float> firstLogitsSpan = constraint is { IsConstraining: true } ctr
            ? ctr.Filter(firstLogits)
            : (ReadOnlySpan<float>)firstLogits;
        // Mirror the decode loop's guard. RecentTokens is seeded from this token, so the history is
        // normally empty here and Greedy is what Sample would do anyway — but a caller that supplied
        // PreviousTokens itself must not have those penalties honoured at temperature > 0 and
        // silently dropped at 0.
        int firstToken = req.Sp.Temperature <= 0f && !req.Sp.HasHistoryPenalty
            ? Sampler.Greedy(firstLogitsSpan)
            : Sampler.Sample(firstLogitsSpan, req.Sp, rng);
        constraint?.Accept(firstToken);
        bool choiceCompletedOnFirstToken = false;
        if (choiceState is not null)
        {
            if (!choiceState.Advance(firstToken))
                throw new InvalidOperationException($"Sampled token {firstToken} is not permitted by choice constraint.");
            choiceCompletedOnFirstToken = choiceState.IsComplete;
        }

        if (stopIds.Contains(firstToken))
        {
            // ORDER IS LOAD-BEARING: publish the turn outcome BEFORE completing the output
            // channel. `TryComplete` releases the consumer's `await foreach`, and HotSession reads
            // `RetainedSequenceState.LastTurn` the instant it resumes — so completing first is a
            // race the consumer loses whenever the scheduler runs it before `state.Complete`.
            // It surfaces as "The retained engine completed without a retained turn outcome" and a
            // Failed operation on a turn that actually succeeded.
            //
            // This is the "first sampled token is a stop token" path — an EOS immediately after
            // prefill — which is why it escaped: it needs a stop token as the very first sample AND
            // the losing interleaving. The ordinary retire path below already gets this right
            // (`state.Complete` then `TryComplete`); this branch was the odd one out.
            if (p.RetainedState is { } state)
                state.Complete(p.Cache, p.TurnStartPosition);
            else
                p.Cache.Dispose();
            req.Output.Writer.TryComplete();
            _committedTokens -= p.ProjectedTokens;
            Interlocked.Decrement(ref _activeCount);
            return;
        }

        // Seed InThinking from the prompt itself (issue #92). Qwen3.6 and other
        // reasoning models append a bare `<think>` token to the generation prompt
        // via their chat template, so the model is already inside a reasoning
        // block before the first sampled token.
        bool promptInThinking = false;
        if (_thinkingEnabled)
        {
            foreach (int tok in p.Tokens)
            {
                if (tok == _thinkTokenId) promptInThinking = true;
                else if (tok == _endThinkTokenId) promptInThinking = false;
            }
        }

        var seq = new ActiveSeq
        {
            CurrentToken = firstToken,
            Position = p.StartPosition + p.Tokens.Length,
            Cache = p.Cache,
            Sp = req.Sp,
            Output = req.Output,
            StopIds = stopIds,
            Rng = rng,
            Ct = req.Ct,
            ProjectedTokens = p.ProjectedTokens,
            RetainedState = p.RetainedState,
            TurnStartPosition = p.TurnStartPosition,
            GeneratedTokenIds = p.RetainedState is null ? null : [firstToken],
            Constraint = constraint,
            ChoiceState = choiceState,
            TokenCount = 1,
            InThinking = promptInThinking,
            ReservationRenewer = p.ReservationRenewer,
            RecentTokens = req.Sp.HasHistoryPenalty ? [firstToken] : null,
        };

        // Route the first sampled token through the same always-consume state machine as the
        // decode loop: a reasoning boundary token is consumed and never emitted, with the state
        // flip gated on the current InThinking. A bare `<channel|>` close (or a double-open) is
        // swallowed rather than leaking its literal marker as text — issue #304.
        if (_thinkingEnabled && firstToken == _thinkTokenId)
        {
            if (!seq.InThinking) seq.InThinking = true;
        }
        else if (_thinkingEnabled && firstToken == _endThinkTokenId)
        {
            if (seq.InThinking) seq.InThinking = false;
        }
        else
        {
            var bytes = _tokenizer.DecodeBytes(firstToken);
            if (seq.InThinking)
            {
                var firstChunk = seq.ThinkDec.Append(bytes);
                if (firstChunk.Length > 0)
                    req.Output.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Thinking, firstChunk));
            }
            else
            {
                var firstChunk = seq.TextDec.Append(bytes);
                if (firstChunk.Length > 0)
                    req.Output.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Text, firstChunk));
            }
        }

        // A choice constraint that is already complete after just the first sampled token (e.g.
        // "YES" is itself an allowed choice): unlike the stop-token branch above, the token IS
        // real content (already emitted above), but its own KV row is not yet materialised — that
        // still needs one more decode step. Clear ChoiceState (nothing left to mask/advance) and
        // let the ordinary StopAfterStep-style discard in the main loop retire it as a natural
        // completion once that step runs.
        if (choiceCompletedOnFirstToken)
        {
            seq.ChoiceState = null;
            seq.ChoiceAlreadyComplete = true;
        }

        active.Add(seq);
    }

    /// <summary>
    /// Whether the batcher loop had actually exited by the time <see cref="Dispose"/> returned.
    ///
    /// <para>This is the signal an owner must consult before releasing anything the loop touches —
    /// the forward pass, the compute backend, the mapped model. Disposal completes either way,
    /// because a shutdown path that can hang is its own kind of bug; but when this is false the
    /// batcher is still running and freeing native, GPU or mmap'd memory underneath it is an
    /// access-violation-class race, not a leak. See <c>OwnedDisposableEngine.Dispose</c>.</para>
    /// </summary>
    public bool DrainedOnDispose { get; private set; }

    /// <summary>How long <see cref="Dispose"/> waits for the batcher to leave its current step.</summary>
    /// <remarks>
    /// The loop tests <c>_disposed</c> at the top of each iteration, so the wait only has to cover
    /// the step already in flight — but that step can be a full chunked prefill on a large model,
    /// which takes far longer than the five seconds this used to allow. The old timeout was short
    /// enough that an ordinary long prefill would routinely fall through it.
    /// </remarks>
    private static readonly TimeSpan BatcherDrainTimeout = TimeSpan.FromSeconds(60);

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _queue.Writer.TryComplete();

        // Wait for the batcher to leave the loop before touching anything it uses. Completing the
        // writer wakes the idle path immediately (WaitToReadAsync returns false); a step already in
        // flight has to finish, since a forward pass cannot be cancelled part-way through.
        if (Task.CurrentId == _batcherTask.Id)
        {
            // Disposing from inside the batcher itself: it cannot wait for its own completion, and
            // it is by definition still on the stack, so nothing it uses may be released.
            DrainedOnDispose = false;
        }
        else
        {
            try { DrainedOnDispose = _batcherTask.Wait(BatcherDrainTimeout); }
            catch { DrainedOnDispose = _batcherTask.IsCompleted; } // faulted/cancelled still means exited
        }

        // The prefix caches are the batcher's own working set, so they are only safe to release once
        // it has exited. If it has not, leak them: process teardown reclaims the memory, whereas
        // freeing it now hands the still-running loop a dangling cache.
        if (!DrainedOnDispose) return;

        foreach (var entry in _prefixCache)
            entry.Cache.Dispose();
        _prefixCache.Clear();
        _prefixCacheUsedBytes = 0;
        Volatile.Write(ref _prefixCacheEntryCount, 0);

        if (_ownsKvCache && _kvCache is IDisposable disposableCache)
        {
            disposableCache.Dispose();
        }
    }
}
