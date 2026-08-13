using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Concrete implementation of <see cref="IInferenceSession"/> owning token history,
/// single-writer concurrency lock, and logical paged KV sequence.
/// </summary>
public sealed class InferenceSession : IInferenceSession
{
    private readonly IKvCache _kvCache;
    private readonly IForwardPass? _forwardPass;
    private readonly SemaphoreSlim _mutex = new(1, 1);
    private readonly List<int> _tokenHistory = new();
    private IKvSequence _kvSequence;
    private SessionState _state;
    private volatile bool _disposed;
    private long _checkpointGeneration = 1;
    private readonly bool _ownsForwardPass;
    public event Action<int, string>? OnTokenGenerated;

    public ISessionMetrics Metrics { get; }
    public ISessionTree Tree { get; }
    public OpenTail.Stingray.Core.Capabilities.ModelCapabilities ModelCapabilities { get; }

    public InferenceSession(IKvCache kvCache, SessionId? id = null, KvSequenceOptions? options = null, IForwardPass? forwardPass = null, bool ownsForwardPass = false, ISessionMetadata? metadata = null, int? maxContextTokens = null, OpenTail.Stingray.Core.Capabilities.ModelCapabilities? modelCapabilities = null)
    {
        _kvCache = kvCache ?? throw new ArgumentNullException(nameof(kvCache));
        _forwardPass = forwardPass;
        _ownsForwardPass = ownsForwardPass;
        Id = id ?? SessionId.New();
        _kvSequence = _kvCache.AllocateSequence(options);
        _state = SessionState.Ready;
        Metadata = metadata ?? new SessionMetadata();
        MaxContextTokens = (options?.MaxContextTokens > 0) ? options.MaxContextTokens : (maxContextTokens > 0 ? maxContextTokens : null);
        Metrics = new SessionMetrics(() => _state == SessionState.Suspended ? 0 : _kvSequence?.PageCount ?? 0);
        Tree = new SessionTree(this, parentTree: null);
        ModelCapabilities = modelCapabilities ?? (forwardPass != null
            ? OpenTail.Stingray.Core.Capabilities.InferenceCapabilityBuilder.Build(hasKvCache: true, forwardPass: forwardPass).Model
            : OpenTail.Stingray.Core.Capabilities.InferenceCapabilityBuilder.CreateDefaultCpu().Model);
    }

    private InferenceSession(IKvCache kvCache, SessionId id, SessionId? parentSessionId, IKvSequence kvSequence, IEnumerable<int> initialTokens, IForwardPass? forwardPass = null, bool ownsForwardPass = false, ISessionMetadata? metadata = null, int? maxContextTokens = null, ISessionMetrics? metrics = null, SessionTree? parentTree = null, OpenTail.Stingray.Core.Capabilities.ModelCapabilities? modelCapabilities = null)
    {
        _kvCache = kvCache;
        _forwardPass = forwardPass;
        _ownsForwardPass = ownsForwardPass;
        Id = id;
        ParentSessionId = parentSessionId;
        _kvSequence = kvSequence;
        _tokenHistory.AddRange(initialTokens);
        _state = SessionState.Ready;
        LastActivityUtc = DateTimeOffset.UtcNow;
        Metadata = metadata ?? new SessionMetadata();
        MaxContextTokens = maxContextTokens > 0 ? maxContextTokens : null;
        Metrics = metrics ?? new SessionMetrics(() => _state == SessionState.Suspended ? 0 : _kvSequence?.PageCount ?? 0);
        Tree = new SessionTree(this, parentTree: parentTree);
        ModelCapabilities = modelCapabilities ?? (forwardPass != null
            ? OpenTail.Stingray.Core.Capabilities.InferenceCapabilityBuilder.Build(hasKvCache: true, forwardPass: forwardPass).Model
            : OpenTail.Stingray.Core.Capabilities.InferenceCapabilityBuilder.CreateDefaultCpu().Model);
    }

    public string ModelId { get; set; } = "default";

    public SessionId Id { get; }
    public SessionId? ParentSessionId { get; }
    public ISessionMetadata Metadata { get; }

    public int? MaxContextTokens { get; }

    private int _coldTokenCount;

    public bool IsContextLimitReached
    {
        get
        {
            lock (_tokenHistory)
            {
                int count = _state == SessionState.Cold ? _coldTokenCount : _tokenHistory.Count;
                return MaxContextTokens is int max && count >= max;
            }
        }
    }

    public int RemainingContextTokens
    {
        get
        {
            lock (_tokenHistory)
            {
                if (MaxContextTokens is not int max) return int.MaxValue;
                int count = _state == SessionState.Cold ? _coldTokenCount : _tokenHistory.Count;
                return Math.Max(0, max - count);
            }
        }
    }
    public DateTimeOffset LastActivityUtc { get; private set; } = DateTimeOffset.UtcNow;
    public OpenTail.Stingray.Core.Tools.IToolProvider? ToolProvider { get; set; }
    public OpenTail.Stingray.Core.Tools.InferenceToolContext? ToolContext { get; set; }
    public OpenTail.Stingray.Core.ITokenizer? Tokenizer { get; set; }
    public Func<IReadOnlyList<int>, IReadOnlyList<OpenTail.Stingray.Core.Tools.ToolCall>?>? ToolCallParser { get; set; }

    private readonly List<OpenTail.Stingray.Core.ISkill> _attachedSkills = new();

    public IReadOnlyList<OpenTail.Stingray.Core.ISkill> AttachedSkills
    {
        get
        {
            lock (_attachedSkills) return _attachedSkills.ToArray();
        }
    }

    public void AttachSkill(OpenTail.Stingray.Core.ISkill skill)
    {
        ArgumentNullException.ThrowIfNull(skill);
        lock (_attachedSkills)
        {
            var newToolNames = new HashSet<string>(skill.Tools.Select(t => t.Name), StringComparer.OrdinalIgnoreCase);
            foreach (var s in _attachedSkills)
            {
                if (string.Equals(s.Name, skill.Name, StringComparison.OrdinalIgnoreCase)) continue;
                foreach (var t in s.Tools)
                {
                    if (newToolNames.Contains(t.Name))
                    {
                        throw new InvalidOperationException($"Tool name collision: tool '{t.Name}' is declared by both skill '{s.Name}' and skill '{skill.Name}'.");
                    }
                }
            }

            _attachedSkills.RemoveAll(s => string.Equals(s.Name, skill.Name, StringComparison.OrdinalIgnoreCase));
            _attachedSkills.Add(skill);
        }
        TouchActivity();
    }

    public bool DetachSkill(string skillName)
    {
        ArgumentNullException.ThrowIfNull(skillName);
        bool removed = false;
        lock (_attachedSkills)
        {
            removed = _attachedSkills.RemoveAll(s => string.Equals(s.Name, skillName, StringComparison.OrdinalIgnoreCase)) > 0;
        }
        if (removed) TouchActivity();
        return removed;
    }

    public bool ValidateToolCall(OpenTail.Stingray.Core.Tools.ToolCall call)
    {
        ArgumentNullException.ThrowIfNull(call);
        lock (_attachedSkills)
        {
            foreach (var skill in _attachedSkills)
            {
                foreach (var tool in skill.Tools)
                {
                    if (string.Equals(tool.Name, call.Name, StringComparison.OrdinalIgnoreCase))
                        return true;
                }
            }
        }

        if (ToolProvider is null) return false;

        var permittedTools = ToolProvider.GetTools(ToolContext);
        foreach (var t in permittedTools)
        {
            if (string.Equals(t.Name, call.Name, StringComparison.OrdinalIgnoreCase))
                return true;
        }

        return false;
    }

    public void TouchActivity()
    {
        LastActivityUtc = DateTimeOffset.UtcNow;
    }

    public SessionState State
    {
        get
        {
            lock (_tokenHistory) return _state;
        }
    }

    public long TokenCount
    {
        get
        {
            lock (_tokenHistory) return _state == SessionState.Cold ? _coldTokenCount : _tokenHistory.Count;
        }
    }

    public IReadOnlyList<int> TokenHistory
    {
        get
        {
            lock (_tokenHistory)
            {
                if (_state == SessionState.Cold)
                {
                    throw new InvalidOperationException($"Cannot inspect token history on cold session {Id}. Call EnsureActiveAsync() first.");
                }
                return _tokenHistory.ToArray();
            }
        }
    }

    public IKvSequence KvSequence => _kvSequence;

    public async ValueTask AppendAsync(ReadOnlyMemory<int> tokens, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        if (_state == SessionState.Suspended)
        {
            await ResumeAsync(cancellationToken).ConfigureAwait(false);
        }
        await _mutex.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (tokens.Length > 0)
            {
                var span = tokens.Span;

                lock (_tokenHistory)
                {
                    if (MaxContextTokens is int max && _tokenHistory.Count + span.Length > max)
                    {
                        throw new ContextLimitExceededException(_tokenHistory.Count, span.Length, max);
                    }
                }

                SetState(SessionState.Generating);
                int historyBefore = _tokenHistory.Count;
                int kvTokensBefore = _kvSequence.TokenCount;
                long startTicks = System.Diagnostics.Stopwatch.GetTimestamp();

                try
                {
                    // 1. Admission-reserve physical KV capacity before attempting to append.
                    // This prevents races where concurrent allocations exhaust free pages
                    // between reservation and physical allocation in CpuKvSequence.Append.
                    OpenTail.Stingray.Engine.IKvReservation? reservation = null;
                    try
                    {
                        // Compute whether this append will require new pages due to token growth,
                        // and whether a copy-on-write of the tail page is required (shared + unaligned).
                        int pageSize = _kvCache.PageSizeTokens;
                        int pagesBefore = _kvSequence.PageCount;
                        int pagesAfter = OpenTail.Stingray.Engine.KvPageMath.GetRequiredPageCount(tokens.Length + _kvSequence.TokenCount, pageSize);
                        int additionalPagesForGrowth = Math.Max(0, pagesAfter - pagesBefore);
                        bool copyOnWriteNeeded = (_kvSequence.PageCount > 0
                            && (_kvSequence.TokenCount % pageSize) != 0
                            && (_kvCache is OpenTail.Stingray.Engine.CpuKvCache cpuCache && cpuCache.IsPageShared(_kvSequence.Pages[^1])));

                        // Reserve capacity for token growth plus, when the tail page is shared and
                        // unaligned, one extra page for the copy-on-write duplication CpuKvSequence.Append
                        // will perform. Reserving the exact page count atomically (rather than reserving
                        // by raw token count, or skipping reservation for the COW-only case) avoids both
                        // an under-reservation on partially-filled pages and a check-then-allocate race
                        // against concurrent appends on other sequences.
                        int requiredPages = additionalPagesForGrowth + (copyOnWriteNeeded ? 1 : 0);
                        if (requiredPages > 0)
                        {
                            reservation = _kvCache.TryReserve(Id.Value.GetHashCode(), requiredPages * pageSize);
                            if (reservation is null)
                            {
                                if (copyOnWriteNeeded && additionalPagesForGrowth == 0)
                                {
                                    throw new InvalidOperationException("Failed to allocate new page for copy-on-write operation.");
                                }
                                throw new InvalidOperationException($"Insufficient KV capacity to append {tokens.Length} tokens. Required pages: {requiredPages}, Free: {_kvCache.FreePages}.");
                            }
                        }

                        // Perform the physical append, which allocates pages from the reserved quota.
                        _kvSequence.Append(tokens.Length);
                    }
                    finally
                    {
                        reservation?.Dispose();
                    }

                    // 2. Commit token history
                    lock (_tokenHistory)
                    {
                        for (int i = 0; i < span.Length; i++)
                        {
                            _tokenHistory.Add(span[i]);
                        }
                    }

                    // 3. Run physical model prefill tensor kernel if forward pass is bound
                    if (_forwardPass is not null)
                    {
                        _forwardPass.Prefill(span.ToArray());
                    }
                }
                catch
                {
                    lock (_tokenHistory)
                    {
                        if (_tokenHistory.Count > historyBefore)
                        {
                            _tokenHistory.RemoveRange(historyBefore, _tokenHistory.Count - historyBefore);
                        }
                    }

                    if (_kvSequence.TokenCount > kvTokensBefore)
                    {
                        _kvSequence.TruncateTo(kvTokensBefore);
                    }

                    if (_forwardPass is not null)
                    {
                        try { _forwardPass.TruncateTo(historyBefore); } catch { }
                    }

                    SetState(SessionState.Ready);
                    throw;
                }

                if (Metrics is SessionMetrics sessionMetrics)
                {
                    sessionMetrics.AddPromptTokens(span.Length, System.Diagnostics.Stopwatch.GetElapsedTime(startTicks));
                }

                lock (_tokenHistory)
                {
                    _checkpointGeneration++;
                }
            }

            SetState(SessionState.Ready);
            TouchActivity();
        }
        catch (ContextLimitExceededException)
        {
            SetState(SessionState.Ready);
            throw;
        }
        finally
        {
            _mutex.Release();
        }
    }

    public async ValueTask AppendToolResultAsync(OpenTail.Stingray.Core.Tools.ToolResult result, CancellationToken cancellationToken = default)
    {
        await AppendToolResultAsync(result, token: null, cancellationToken).ConfigureAwait(false);
    }

    public async ValueTask AppendToolResultAsync(OpenTail.Stingray.Core.Tools.ToolResult result, ResponseContinuationToken? token, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(result);
        ThrowIfDisposed();

        if (token is not null)
        {
            ValidateContinuationToken(token.Value);
        }

        if (Tokenizer is null)
        {
            throw new InvalidOperationException("Cannot append tool result because no ITokenizer is configured on the session.");
        }

        string jsonContent = result.Content.ValueKind != System.Text.Json.JsonValueKind.Undefined
            ? result.Content.GetRawText()
            : "{}";

        string toolResultText = $"<|tool_response|>{result.ToolCallId}:{jsonContent}<|end_of_turn|>";

        var tokens = Tokenizer.Encode(toolResultText);
        int[] tokenArray = tokens is int[] arr ? arr : System.Linq.Enumerable.ToArray(tokens);
        await AppendAsync(tokenArray, cancellationToken).ConfigureAwait(false);
    }

    public ResponseContinuationToken GetContinuationToken()
    {
        ThrowIfDisposed();
        lock (_tokenHistory)
        {
            return new ResponseContinuationToken(Id, _tokenHistory.Count, _checkpointGeneration);
        }
    }

    public void ValidateContinuationToken(ResponseContinuationToken token)
    {
        ThrowIfDisposed();
        if (token.SessionId != Id)
        {
            throw new ArgumentException($"Continuation token SessionId '{token.SessionId}' does not match target session '{Id}'.", nameof(token));
        }

        lock (_tokenHistory)
        {
            if (token.Generation != _checkpointGeneration || token.TokenPosition != _tokenHistory.Count)
            {
                throw new StaleContinuationException(Id, token.TokenPosition, token.Generation, _tokenHistory.Count, _checkpointGeneration);
            }
        }
    }

    public ValueTask ContinueAsync(ResponseContinuationToken token, CancellationToken cancellationToken = default)
    {
        ValidateContinuationToken(token);
        return ValueTask.CompletedTask;
    }

    public async IAsyncEnumerable<GenerateChunk> GenerateAsync(
        SamplingParams sampling,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        if (_state == SessionState.Suspended)
        {
            await ResumeAsync(cancellationToken).ConfigureAwait(false);
        }
        await _mutex.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            SetState(SessionState.Generating);

            if (IsContextLimitReached)
            {
                yield break;
            }

            TokenChoiceTrie.ChoiceState? choiceState = null;
            if (sampling.AllowedChoices is { Count: > 0 } choices && Tokenizer is not null)
            {
                var trie = TokenChoiceTrie.Build(choices, Tokenizer);
                choiceState = trie.CreateState();
            }

            int requestedMax = sampling.MaxNewTokens > 0 ? sampling.MaxNewTokens : 16;
            int remainingTokens = RemainingContextTokens;
            int steps = Math.Min(Math.Min(requestedMax, remainingTokens), 512);

            for (int i = 0; i < steps; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (IsContextLimitReached)
                {
                    break;
                }

                if (choiceState is not null && choiceState.IsComplete)
                {
                    break;
                }

                long stepStartTicks = System.Diagnostics.Stopwatch.GetTimestamp();
                int nextToken;
                if (_forwardPass is not null && _tokenHistory.Count > 0)
                {
                    int currentPos = _tokenHistory.Count;
                    int lastToken = _tokenHistory[^1];
                    var rawLogits = _forwardPass.Forward(lastToken, currentPos);

                    var constraint = sampling.Constraint;
                    ReadOnlySpan<float> logitsForSampling = rawLogits;
                    if (choiceState is not null)
                    {
                        var choiceMaskedLogits = new float[rawLogits.Length];
                        rawLogits.CopyTo(choiceMaskedLogits);
                        choiceState.MaskLogits(choiceMaskedLogits);
                        logitsForSampling = choiceMaskedLogits;
                    }

                    if (constraint is { IsConstraining: true })
                    {
                        logitsForSampling = constraint.Filter(logitsForSampling);
                    }

                    nextToken = Sampler.Sample(logitsForSampling, sampling);
                    constraint?.Accept(nextToken);

                    if (choiceState is not null)
                    {
                        if (!choiceState.Advance(nextToken))
                        {
                            throw new InvalidOperationException($"Sampled token {nextToken} is not permitted by choice constraint.");
                        }
                    }
                }
                else
                {
                    throw new InvalidOperationException($"Cannot generate tokens on session {Id} without a bound {nameof(IForwardPass)} and non-empty token history (history count: {_tokenHistory.Count}).");
                }

                int historyBefore = _tokenHistory.Count;
                int kvTokensBefore = _kvSequence.TokenCount;
                try
                {
                    lock (_tokenHistory)
                    {
                        _tokenHistory.Add(nextToken);
                    }
                    _kvSequence.Append(1);
                }
                catch
                {
                    lock (_tokenHistory)
                    {
                        if (_tokenHistory.Count > historyBefore)
                        {
                            _tokenHistory.RemoveRange(historyBefore, _tokenHistory.Count - historyBefore);
                        }
                    }

                    if (_kvSequence.TokenCount > kvTokensBefore)
                    {
                        _kvSequence.TruncateTo(kvTokensBefore);
                    }

                    if (_forwardPass is not null)
                    {
                        try { _forwardPass.TruncateTo(historyBefore); } catch { }
                    }

                    throw;
                }

                if (Metrics is SessionMetrics sessionMetrics)
                {
                    sessionMetrics.AddGeneratedTokens(1, System.Diagnostics.Stopwatch.GetElapsedTime(stepStartTicks));
                }

                // Structured ToolCall detection and capability validation loop
                if (ToolCallParser is not null)
                {
                    var detectedCalls = ToolCallParser(_tokenHistory);
                    if (detectedCalls is not null && detectedCalls.Count > 0)
                    {
                        var processedCalls = new List<OpenTail.Stingray.Core.Tools.ToolCall>();
                        bool hasUnauthorized = false;

                        foreach (var call in detectedCalls)
                        {
                            bool isAuth = ValidateToolCall(call);
                            if (!isAuth) hasUnauthorized = true;
                            processedCalls.Add(call with { IsAuthorized = isAuth });
                        }

                        yield return new GenerateChunk(
                            Kind: GenerateChunkKind.ToolCall,
                            Text: string.Empty,
                            ToolCalls: processedCalls,
                            HasUnauthorizedToolCall: hasUnauthorized);

                        break; // Stop generation so OpenTail host handles authorized/unauthorized tool calls
                    }
                }

                bool isChoiceDone = choiceState is not null && choiceState.IsComplete;
                bool isLast = i == steps - 1 || isChoiceDone;
                string chunkText = Tokenizer is not null ? Tokenizer.Decode(new[] { nextToken }) : nextToken.ToString();

                var handler = OnTokenGenerated;
                if (handler is not null)
                {
                    try { handler(nextToken, chunkText); } catch { /* Isolate host listener exceptions */ }
                }

                yield return new GenerateChunk(
                    Kind: isLast ? GenerateChunkKind.Stop : GenerateChunkKind.Text,
                    Text: chunkText,
                    TruncatedByMaxTokens: isLast && !isChoiceDone);

                if (isChoiceDone)
                {
                    break;
                }
            }

            SetState(SessionState.Ready);
            TouchActivity();
        }
        finally
        {
            if (_state == SessionState.Generating)
            {
                SetState(SessionState.Ready);
            }
            TouchActivity();
            _mutex.Release();
        }
    }

    public GenerationStream GenerateWithResultAsync(
        SamplingParams sampling,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        return new GenerationStream(this, sampling, cancellationToken);
    }

    private int _rngSeed = 42;
    private int _rngStep;

    public SessionCheckpoint CreateCheckpoint()
    {
        ThrowIfDisposed();
        ThrowIfSuspended();
        lock (_tokenHistory)
        {
            return new SessionCheckpoint(
                Id,
                _tokenHistory.Count,
                _rngSeed,
                _rngStep,
                _tokenHistory.ToArray(),
                DateTimeOffset.UtcNow);
        }
    }

    public void Rollback(SessionCheckpoint checkpoint)
    {
        ThrowIfDisposed();
        ThrowIfSuspended();
        _mutex.WaitAsync().GetAwaiter().GetResult();
        try
        {
            lock (_tokenHistory)
            {
                if (checkpoint.SessionId != Id)
                {
                    throw new InvalidOperationException($"Checkpoint belongs to session {checkpoint.SessionId}, not {Id}.");
                }

                if (checkpoint.TokenPosition < 0 || checkpoint.TokenPosition > _tokenHistory.Count)
                {
                    throw new ArgumentOutOfRangeException(nameof(checkpoint), $"Invalid rollback position {checkpoint.TokenPosition} (current: {_tokenHistory.Count}).");
                }

                int targetPos = (int)checkpoint.TokenPosition;
                _tokenHistory.RemoveRange(targetPos, _tokenHistory.Count - targetPos);
                _rngSeed = checkpoint.RngSeed;
                _rngStep = checkpoint.RngStep;

                // Fork sequence at target position to truncate KV page table to checkpoint
                var truncatedKv = _kvSequence.ForkAt(targetPos);
                _kvSequence.Release();
                _kvSequence = truncatedKv;

                if (_forwardPass is not null)
                {
                    _forwardPass.TruncateTo(targetPos);
                }
            }
            SetState(SessionState.Ready);
        }
        finally
        {
            _mutex.Release();
        }
    }

    public SessionDelta CreateDelta(ResponseContinuationToken baseToken)
    {
        ThrowIfDisposed();
        ThrowIfSuspended();

        if (baseToken.SessionId != Id)
        {
            throw new ArgumentException($"Continuation token SessionId '{baseToken.SessionId}' does not match session '{Id}'.", nameof(baseToken));
        }

        lock (_tokenHistory)
        {
            if (baseToken.Generation > _checkpointGeneration || baseToken.TokenPosition > _tokenHistory.Count)
            {
                throw new ArgumentException($"Continuation token position {baseToken.TokenPosition} (gen: {baseToken.Generation}) is ahead of current committed position {_tokenHistory.Count} (gen: {_checkpointGeneration}).", nameof(baseToken));
            }

            int basePos = (int)baseToken.TokenPosition;
            var appended = _tokenHistory.GetRange(basePos, _tokenHistory.Count - basePos);

            var metaEntries = Metadata.GetEntries();
            var metaDict = new Dictionary<string, string?>(StringComparer.Ordinal);
            foreach (var kv in metaEntries)
            {
                if (kv.Value is not null)
                {
                    metaDict[kv.Key] = kv.Value.ToString();
                }
            }

            var resultToken = new ResponseContinuationToken(Id, _tokenHistory.Count, _checkpointGeneration);

            return new SessionDelta
            {
                SessionId = Id,
                BaseToken = baseToken,
                ResultToken = resultToken,
                AppendedTokens = appended.AsReadOnly(),
                MetadataChanges = System.Collections.Immutable.ImmutableDictionary.ToImmutableDictionary(metaDict),
                CreatedAt = DateTimeOffset.UtcNow
            };
        }
    }

    public async ValueTask ApplyDeltaAsync(SessionDelta delta, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(delta);
        ThrowIfDisposed();

        if (delta.SessionId != Id)
        {
            throw new ArgumentException($"Delta SessionId '{delta.SessionId}' does not match target session '{Id}'.", nameof(delta));
        }

        ResponseContinuationToken currentToken = GetContinuationToken();

        // Idempotency check: if current token matches delta.ResultToken, no-op
        if (currentToken == delta.ResultToken)
        {
            return;
        }

        // Validate base token against current token
        if (currentToken.TokenPosition != delta.BaseToken.TokenPosition || currentToken.Generation != delta.BaseToken.Generation)
        {
            throw new StaleContinuationException(
                Id,
                delta.BaseToken.TokenPosition,
                delta.BaseToken.Generation,
                currentToken.TokenPosition,
                currentToken.Generation);
        }

        // Transactional apply: append tokens first
        if (delta.AppendedTokens is { Count: > 0 } tokens)
        {
            int[] tokenArray = tokens is int[] arr ? arr : System.Linq.Enumerable.ToArray(tokens);
            await AppendAsync(tokenArray, cancellationToken).ConfigureAwait(false);
        }

        // Apply metadata updates
        if (delta.MetadataChanges is not null)
        {
            foreach (var kv in delta.MetadataChanges)
            {
                if (kv.Value is null)
                {
                    Metadata.Remove(kv.Key);
                }
                else
                {
                    Metadata.Set(kv.Key, kv.Value);
                }
            }
        }
    }

    public InferenceSessionSnapshot ToSnapshot(string modelId)
    {
        ThrowIfDisposed();
        ThrowIfSuspended();
        lock (_tokenHistory)
        {
            return new InferenceSessionSnapshot
            {
                Version = 1,
                Id = Id,
                ModelId = modelId ?? "",
                Tokens = _tokenHistory.ToArray(),
                Position = _tokenHistory.Count,
                Generation = _checkpointGeneration,
                SavedAt = DateTimeOffset.UtcNow
            };
        }
    }

    public void RestoreFromSnapshot(InferenceSessionSnapshot snapshot)
    {
        ThrowIfDisposed();
        ThrowIfSuspended();
        ArgumentNullException.ThrowIfNull(snapshot);
        if (snapshot.Id != Id)
        {
            throw new InvalidOperationException($"Snapshot Id {snapshot.Id} does not match Session Id {Id}.");
        }

        _mutex.WaitAsync().GetAwaiter().GetResult();
        try
        {
            lock (_tokenHistory)
            {
                _tokenHistory.Clear();
                _tokenHistory.AddRange(snapshot.Tokens);
                _checkpointGeneration = Math.Max(1, snapshot.Generation);

                var newKv = _kvCache.AllocateSequence();
                newKv.Append(_tokenHistory.Count);
                _kvSequence.Release();
                _kvSequence = newKv;

                if (_forwardPass is not null && _tokenHistory.Count > 0)
                {
                    _forwardPass.Prefill(_tokenHistory.ToArray());
                }
            }
            SetState(SessionState.Ready);
        }
        finally
        {
            _mutex.Release();
        }
    }

    public async ValueTask SuspendAsync(CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        await _mutex.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            lock (_tokenHistory)
            {
                if (_state == SessionState.Suspended) return;

                _kvSequence.Release();
                _state = SessionState.Suspended;
            }
        }
        finally
        {
            _mutex.Release();
        }
    }

    public async ValueTask ResumeAsync(CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        await _mutex.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ResumeUnderMutex();
        }
        finally
        {
            _mutex.Release();
        }
    }

    public async ValueTask EvictToColdAsync(ISessionStore store, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(store);
        ThrowIfDisposed();

        await _mutex.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            lock (_tokenHistory)
            {
                if (_state != SessionState.Suspended)
                {
                    throw new InvalidOperationException($"Cannot evict session in state {_state} to cold storage. Session must be in Suspended state.");
                }
            }

            // Create snapshot while session token history is intact
            InferenceSessionSnapshot snapshot;
            lock (_tokenHistory)
            {
                snapshot = new InferenceSessionSnapshot
                {
                    Version = 1,
                    Id = Id,
                    ModelId = ModelId ?? "default",
                    Tokens = _tokenHistory.ToArray(),
                    Position = _tokenHistory.Count,
                    Generation = _checkpointGeneration,
                    SavedAt = DateTimeOffset.UtcNow
                };
            }

            // PERSISTENCE BEFORE RELEASE: Save checkpoint to store first!
            await store.SaveAsync(snapshot, cancellationToken).ConfigureAwait(false);

            lock (_tokenHistory)
            {
                _coldTokenCount = _tokenHistory.Count;
                _tokenHistory.Clear();

                // If sequence holds physical pages, release them to the free pool
                if (_kvSequence is not null)
                {
                    _kvSequence.Release();
                    _kvSequence = _kvCache.AllocateSequence();
                }

                SetState(SessionState.Cold);
            }
        }
        finally
        {
            _mutex.Release();
        }
    }

    public async ValueTask EnsureActiveAsync(ISessionStore? store = null, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        SessionState currentState;
        lock (_tokenHistory)
        {
            currentState = _state;
        }

        if (currentState == SessionState.Ready || currentState == SessionState.Generating)
        {
            return; // Already active
        }

        await _mutex.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            lock (_tokenHistory)
            {
                currentState = _state;
            }

            if (currentState == SessionState.Ready || currentState == SessionState.Generating)
            {
                return; // Double-check under lock
            }

            if (currentState == SessionState.Suspended)
            {
                ResumeUnderMutex();
                return;
            }

            if (currentState == SessionState.Cold)
            {
                if (store is null)
                {
                    throw new InvalidOperationException($"Cannot rehydrate cold session '{Id}' without an {nameof(ISessionStore)} provider.");
                }

                var snapshot = await store.LoadAsync(Id, cancellationToken).ConfigureAwait(false);
                if (snapshot is null)
                {
                    throw new InvalidOperationException($"Failed to rehydrate cold session '{Id}': snapshot not found in store.");
                }

                // RESERVE BEFORE LOAD: Atomic capacity reservation check held across rehydration
                int requiredTokens = snapshot.Tokens?.Count ?? 0;
                int requiredPages = KvPageMath.GetRequiredPageCount(requiredTokens, _kvCache.PageSizeTokens);

                var res = _kvCache.TryReserve(Id.Value.GetHashCode(), Math.Max(1, requiredTokens));
                try
                {
                    if (res is null && _kvCache.FreePages < requiredPages)
                    {
                        throw new InvalidOperationException($"Insufficient KV capacity to rehydrate cold session '{Id}'. Required pages: {requiredPages}, Free: {_kvCache.FreePages}.");
                    }

                    lock (_tokenHistory)
                    {
                        _tokenHistory.Clear();
                        if (snapshot.Tokens is not null)
                        {
                            _tokenHistory.AddRange(snapshot.Tokens);
                        }
                        _state = SessionState.Suspended;
                    }

                    ResumeUnderMutex();
                }
                finally
                {
                    res?.Dispose();
                }
            }
        }
        finally
        {
            _mutex.Release();
        }
    }

    /// <summary>Rehydrates the suspended KV sequence while the caller owns <see cref="_mutex"/>.</summary>
    private void ResumeUnderMutex()
    {
        IKvSequence? newSequence = null;
        try
        {
            lock (_tokenHistory)
            {
                if (_state != SessionState.Suspended)
                    throw new InvalidOperationException($"Cannot resume session in state {_state}. Only Suspended sessions can be resumed.");

                newSequence = _kvCache.AllocateSequence();
                newSequence.Append(_tokenHistory.Count);
                _kvSequence = newSequence;
            }

            if (_forwardPass is not null)
            {
                int[] tokensSnapshot;
                lock (_tokenHistory) { tokensSnapshot = _tokenHistory.ToArray(); }
                if (tokensSnapshot.Length > 0) _forwardPass.Prefill(tokensSnapshot);
            }

            SetState(SessionState.Ready);
        }
        catch
        {
            if (newSequence is not null)
            {
                lock (_tokenHistory)
                {
                    if (ReferenceEquals(_kvSequence, newSequence))
                    {
                        newSequence.Release();
                        _kvSequence = _kvCache.AllocateSequence();
                    }
                }
            }
            SetState(SessionState.Suspended);
            throw;
        }
    }

    /// <summary>
    /// Creates a zero-copy fork of the session. Shares immutable model weights and physical KV page tables,
    /// while establishing isolated, single-writer session state (tokens, sequence handle, mutex lock, and forward pass execution context).
    /// </summary>
    public IInferenceSession Fork()
    {
        ThrowIfDisposed();
        if (_state != SessionState.Ready)
        {
            throw new InvalidOperationException($"Cannot fork session in state {_state}. Only Ready sessions can be forked.");
        }

        _mutex.WaitAsync().GetAwaiter().GetResult();
        try
        {
            lock (_tokenHistory)
            {
                var childForwardPass = _forwardPass?.CreateContext();
                ThrowIfForwardPassNotIsolated(childForwardPass);
                var childKv = _kvSequence.Fork();
                bool ownsChildForward = childForwardPass is not null && !ReferenceEquals(childForwardPass, _forwardPass);
                TouchActivity();
                var childMetrics = Metrics is SessionMetrics sm ? sm.CloneForFork(() => childKv.PageCount) : null;
                var child = new InferenceSession(_kvCache, SessionId.New(), Id, childKv, _tokenHistory, childForwardPass, ownsForwardPass: ownsChildForward, metadata: Metadata.Clone(), maxContextTokens: MaxContextTokens, metrics: childMetrics, parentTree: Tree as SessionTree, modelCapabilities: ModelCapabilities)
                {
                    Tokenizer = Tokenizer,
                    ToolCallParser = ToolCallParser,
                    ToolProvider = ToolProvider,
                    ToolContext = ToolContext,
                    _rngSeed = SplitSeed(_rngSeed, 0)
                };
                lock (_attachedSkills)
                {
                    foreach (var s in _attachedSkills)
                    {
                        child.AttachSkill(s);
                    }
                }
                return child;
            }
        }
        finally
        {
            _mutex.Release();
        }
    }

    /// <summary>
    /// Spawns N zero-copy child sessions sharing physical KV prefix page tables with atomic failure cleanup.
    /// Each child receives independent derived RNG seeds, sequence handles, and forward pass execution context.
    /// </summary>
    public IReadOnlyList<IInferenceSession> ForkMany(int count)
    {
        if (count < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(count), "Branch count must be at least 1.");
        }

        ThrowIfDisposed();
        if (_state != SessionState.Ready)
        {
            throw new InvalidOperationException($"Cannot fork session in state {_state}. Only Ready sessions can be forked.");
        }

        _mutex.WaitAsync().GetAwaiter().GetResult();
        var branches = new List<IInferenceSession>(count);
        try
        {
            lock (_tokenHistory)
            {
                for (int i = 0; i < count; i++)
                {
                    var childForwardPass = _forwardPass?.CreateContext();
                    ThrowIfForwardPassNotIsolated(childForwardPass);
                    var childKv = _kvSequence.Fork();
                    bool ownsChildForward = childForwardPass is not null && !ReferenceEquals(childForwardPass, _forwardPass);
                    int childSeed = SplitSeed(_rngSeed, i);
                    var childMetrics = Metrics is SessionMetrics sm ? sm.CloneForFork(() => childKv.PageCount) : null;
                    var childSession = new InferenceSession(_kvCache, SessionId.New(), Id, childKv, _tokenHistory, childForwardPass, ownsForwardPass: ownsChildForward, metadata: Metadata.Clone(), maxContextTokens: MaxContextTokens, metrics: childMetrics, parentTree: Tree as SessionTree, modelCapabilities: ModelCapabilities)
                    {
                        Tokenizer = Tokenizer,
                        ToolCallParser = ToolCallParser,
                        ToolProvider = ToolProvider,
                        ToolContext = ToolContext,
                        _rngSeed = childSeed
                    };
                    lock (_attachedSkills)
                    {
                        foreach (var s in _attachedSkills)
                        {
                            childSession.AttachSkill(s);
                        }
                    }
                    branches.Add(childSession);
                }
                TouchActivity();
                return branches;
            }
        }
        catch
        {
            foreach (var branch in branches)
            {
                branch.DisposeAsync().AsTask().GetAwaiter().GetResult();
            }
            throw;
        }
        finally
        {
            _mutex.Release();
        }
    }

    private void SetState(SessionState newState)
    {
        lock (_tokenHistory)
        {
            _state = newState;
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(InferenceSession));
        }
    }

    private void ThrowIfSuspended()
    {
        lock (_tokenHistory)
        {
            if (_state == SessionState.Suspended)
            {
                throw new InvalidOperationException($"Cannot execute operation on suspended session {Id}. Call ResumeAsync() first.");
            }
        }
    }

    /// <summary>
    /// Guards against silent decode-state sharing on fork. <see cref="OpenTail.Stingray.Core.IForwardPass.CreateContext"/>
    /// defaults to returning <c>this</c>, and no production backend currently overrides it to return
    /// a genuinely isolated instance — so without this check, a forked child would share the
    /// parent's KV cache and position counter, silently corrupting both sessions' decode state on
    /// concurrent or interleaved generation. Fails loudly instead.
    /// </summary>
    private void ThrowIfForwardPassNotIsolated(IForwardPass? childForwardPass)
    {
        if (_forwardPass is not null && ReferenceEquals(childForwardPass, _forwardPass))
        {
            throw new NotSupportedException(
                $"Cannot fork session {Id}: bound forward pass '{_forwardPass.GetType().Name}' does not " +
                "implement an isolated CreateContext() and would share decode state (KV cache, position " +
                "counter) with the forked child. Only sessions without a bound forward pass, or backends " +
                "whose forward pass overrides CreateContext() to return an isolated instance, can be forked.");
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;

        await _mutex.WaitAsync().ConfigureAwait(false);
        try
        {
            SetState(SessionState.Disposed);
            (Tree as SessionTree)?.OnDisposed();
            _kvSequence.Release();
            if (_ownsForwardPass && _forwardPass is IDisposable disposableForward)
            {
                disposableForward.Dispose();
            }
        }
        finally
        {
            _mutex.Dispose();
        }
    }

    private static int SplitSeed(int seed, int index)
    {
        uint h = (uint)seed ^ (uint)(index + 1);
        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;
        return (int)h;
    }
}
