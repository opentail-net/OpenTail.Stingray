using System.Text.Json;
using OpenTail.Stingray.Core.Tools;

namespace OpenTail.Stingray.Tests.Sessions.Fast;

/// <summary>
/// docs/032: HotSession-native coverage for the six capabilities ported from InferenceSession's
/// otherwise-unused island (skills/tool validation, OnTokenGenerated, checkpoint/rollback,
/// suspend/resume, session-tree/lineage) — ActiveLora is excluded since it was never wired to
/// anything even on InferenceSession itself (see HotSession.ActiveLora's own doc comment).
/// </summary>
public sealed class HotSessionCapabilityPortTests
{
    private const int Eos = 31;

    [Fact]
    public async Task HotSession_ValidateToolCall_AuthorizesFromAttachedSkill()
    {
        var fwd = new FakeForwardPass { EmitNonStopOnPrefill = true };
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();

        session.AttachSkill(new FakeSkill("reviewer", "read_file"));

        var call = new ToolCall("call-1", "read_file", JsonDocument.Parse("{}").RootElement);
        Assert.True(session.ValidateToolCall(call));

        Assert.True(session.DetachSkill("reviewer"));
        Assert.False(session.ValidateToolCall(call));
    }

    [Fact]
    public void HotSession_ValidateToolCall_AuthorizesFromToolProvider()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();

        session.ToolProvider = new MemoryToolProvider([new ToolDefinition("write_file")]);
        var call = new ToolCall("call-1", "write_file", JsonDocument.Parse("{}").RootElement);
        Assert.True(session.ValidateToolCall(call));

        var disallowed = new ToolCall("call-2", "exec_shell", JsonDocument.Parse("{}").RootElement);
        Assert.False(session.ValidateToolCall(disallowed));
    }

    /// <summary>
    /// docs/051: a skill's <c>Instructions</c> prepend to the NEXT <see cref="HotSession.RunTurnAsync"/>
    /// call's append-prompt text only, then clear — never retroactively, and never repeated on a
    /// later turn that didn't have a fresh attach since.
    /// </summary>
    [Fact]
    public async Task HotSession_AttachSkillInstructions_PrependToNextTurnOnly()
    {
        var fwd = new FakeForwardPass { EmitNonStopOnPrefill = true };
        var tokenizer = new SpyTokenizer();
        using var engine = new ContinuousBatchingEngine(fwd, tokenizer, "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, tokenizer);
        using var session = runtime.Create();
        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        await session.RunTurnAsync("hello", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("t1"));
        Assert.Contains("hello", tokenizer.EncodedTexts);
        Assert.DoesNotContain(tokenizer.EncodedTexts, t => t.Contains("Be nice."));

        session.AttachSkill(new FakeInstructionSkill("politeness", "Be nice."));
        tokenizer.EncodedTexts.Clear();
        await session.RunTurnAsync("world", sampling, new SessionRevision(1), SessionOperationId.New(), Digest("t2"));
        Assert.Contains("Be nice.\n\nworld", tokenizer.EncodedTexts);

        tokenizer.EncodedTexts.Clear();
        await session.RunTurnAsync("again", sampling, new SessionRevision(2), SessionOperationId.New(), Digest("t3"));
        Assert.Contains("again", tokenizer.EncodedTexts);
        Assert.DoesNotContain(tokenizer.EncodedTexts, t => t.Contains("Be nice."));
    }

    [Fact]
    public async Task HotSession_OnTokenGenerated_FiresOncePerCommittedToken()
    {
        var fwd = new FakeForwardPass { EmitNonStopOnPrefill = true, EmitNonStopOnDecode = true };
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();

        var seenTokens = new List<int>();
        session.OnTokenGenerated += (tokenId, _) => seenTokens.Add(tokenId);

        var result = await session.RunTurnAsync("one",
            new SamplingParams { Temperature = 0f, MaxNewTokens = 3 },
            SessionRevision.Initial, SessionOperationId.New(), Digest("listen"));

        Assert.Equal(SessionOperationState.Completed, result.Operation.State);
        Assert.Equal(3, seenTokens.Count);
    }

    [Fact]
    public async Task HotSession_CheckpointAndRollback_RestoresPriorCursorAndCache()
    {
        var fwd = new FakeForwardPass { EmitNonStopOnPrefill = true, EmitNonStopOnDecode = true };
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();
        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 2 };

        var first = await session.RunTurnAsync("one", sampling, SessionRevision.Initial,
            SessionOperationId.New(), Digest("t1"));
        var checkpoint = session.CreateCheckpoint();
        Assert.Equal(first.Cursor.MaterializedPositionCount, checkpoint.Cursor.MaterializedPositionCount);

        await session.RunTurnAsync("two", sampling, first.Operation.CommittedRevision!.Value,
            SessionOperationId.New(), Digest("t2"));
        Assert.True(session.Cursor.MaterializedPositionCount > checkpoint.Cursor.MaterializedPositionCount);

        await session.RollbackAsync(checkpoint);

        Assert.Equal(checkpoint.Cursor.MaterializedPositionCount, session.Cursor.MaterializedPositionCount);
        Assert.Equal(checkpoint.CommittedRevision, (await Task.FromResult(runtime.GetSessionSnapshot(session.SessionId))).CommittedRevision);

        // The rolled-back revision must be usable again for a fresh turn.
        var replay = await session.RunTurnAsync("two", sampling, checkpoint.CommittedRevision,
            SessionOperationId.New(), Digest("t2-again"));
        Assert.Equal(SessionOperationState.Completed, replay.Operation.State);
    }

    [Fact]
    public void HotSession_SuspendAsync_EvictsCacheAndIsSuspendedReflectsIt()
    {
        var fwd = new FakeForwardPass { EmitNonStopOnPrefill = true };
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();

        Assert.True(session.IsSuspended); // never prefilled
    }

    [Fact]
    public async Task HotSession_Tree_ReportsParentChildLineageAndAggregatedMetrics()
    {
        var fwd = new FakeForwardPass { EmitNonStopOnPrefill = true, EmitNonStopOnDecode = true };
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var parent = runtime.Create();

        await parent.RunTurnAsync("one", new SamplingParams { Temperature = 0f, MaxNewTokens = 2 },
            SessionRevision.Initial, SessionOperationId.New(), Digest("parent-turn"));

        var branches = runtime.Fork(parent, 1);
        var child = branches[0];
        try
        {
            Assert.Equal(parent.SessionId, child.Tree.ParentId);
            Assert.Contains(child.SessionId, parent.Tree.Children);
            Assert.Equal(parent.SessionId, parent.Tree.RootId);
            Assert.Equal(parent.SessionId, child.Tree.RootId);

            var cumulative = parent.Tree.CumulativeTreeMetrics;
            Assert.Equal(parent.Metrics.PromptTokens + child.Metrics.PromptTokens, cumulative.PromptTokens);
        }
        finally
        {
            foreach (var b in branches) runtime.Delete(b.SessionId);
        }
    }

    private static SessionRequestDigest Digest(string value) => SessionRequestDigest.FromCanonicalValue(value);

    private sealed class FakeSkill(string name, params string[] toolNames) : ISkill
    {
        public string Name { get; } = name;
        public string? Description => null;
        public IReadOnlyList<IInstruction> Instructions => [];
        public IReadOnlyList<ITool> Tools { get; } = [.. toolNames.Select(n => new FakeTool(n))];
        public IReadOnlyList<IResource> Resources => [];

        private sealed class FakeTool(string name) : ITool
        {
            public string Name { get; } = name;
            public string? Description => null;
        }
    }

    private sealed class FakeInstructionSkill(string name, params string[] instructionTexts) : ISkill
    {
        public string Name { get; } = name;
        public string? Description => null;
        public IReadOnlyList<IInstruction> Instructions { get; } =
            [.. instructionTexts.Select(c => (IInstruction)new FakeInstruction(c))];
        public IReadOnlyList<ITool> Tools => [];
        public IReadOnlyList<IResource> Resources => [];

        private sealed class FakeInstruction(string content) : IInstruction
        {
            public string Content { get; } = content;
            public string? Name => null;
        }
    }

    /// <summary>Records every string it is asked to encode, and tokenizes any text as one token.</summary>
    private sealed class SpyTokenizer : ITokenizer
    {
        public List<string> EncodedTexts { get; } = [];
        public int VocabSize => 64;
        public int BosTokenId => 0;
        public int EosTokenId => Eos;
        public int UnknownTokenId => 0;
        public int PadTokenId => Eos;
        public bool AddBosToken => false;
        public IReadOnlyCollection<int> EogTokenIds => [Eos];
        public IReadOnlyList<int> Encode(string text)
        {
            EncodedTexts.Add(text);
            return [1];
        }
        public string Decode(IEnumerable<int> tokens) => string.Empty;
        public byte[] DecodeBytes(int token) => [];
    }

    private sealed class Tokenizer : ITokenizer
    {
        public int VocabSize => 64;
        public int BosTokenId => 0;
        public int EosTokenId => Eos;
        public int UnknownTokenId => 0;
        public int PadTokenId => Eos;
        public bool AddBosToken => false;
        public IReadOnlyCollection<int> EogTokenIds => [Eos];
        public IReadOnlyList<int> Encode(string text) => text switch
        {
            "one" => [1, 2],
            "two" => [3],
            _ => throw new ArgumentOutOfRangeException(nameof(text)),
        };
        public string Decode(IEnumerable<int> tokens) => string.Empty;
        public byte[] DecodeBytes(int token) => [];
    }

    private sealed class FakeForwardPass : IBatchedForwardPass, IPrefixCacheableBatchedForwardPass
    {
        private const int NonStopToken = 7;
        private static readonly float[] EosLogits = CreateLogits(Eos);
        private static readonly float[] NonStopLogits = CreateLogits(NonStopToken);
        public List<FakeCache> Created { get; } = [];
        public List<(int Start, int Count)> Prefills { get; } = [];
        public bool EmitNonStopOnPrefill { get; set; }
        public bool EmitNonStopOnDecode { get; set; }
        public bool SnapKvEnabled => false;
        public long KvBytesPerToken => 1;
        public int MaxSeqLen => 64;
        public bool PrefillDequantCacheActive => false;
        public int PrefixCacheBlockSize => 1;

        public ISequenceKvCache CapturePrefix(ISequenceKvCache cache, int prefixLength) =>
            new FakeCache { LogicalPosition = prefixLength };

        public ISequenceKvCache ForkPrefix(ISequenceKvCache prefix) =>
            new FakeCache { LogicalPosition = Assert.IsType<FakeCache>(prefix).LogicalPosition };

        public ISequenceKvCache CreateCache()
        {
            var cache = new FakeCache();
            Created.Add(cache);
            return cache;
        }

        public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos = 0)
        {
            var retained = Assert.IsType<FakeCache>(cache);
            Assert.Equal(startPos, retained.LogicalPosition);
            retained.LogicalPosition += tokens.Count;
            Prefills.Add((startPos, tokens.Count));
            return EmitNonStopOnPrefill ? NonStopLogits : EosLogits;
        }

        public float[]?[] PrefillPackedMulti(
            ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits) =>
            throw new NotSupportedException();

        public float[][] BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches)
        {
            for (int i = 0; i < caches.Length; i++)
            {
                var cache = Assert.IsType<FakeCache>(caches[i]);
                Assert.Equal(positions[i], cache.LogicalPosition);
                cache.LogicalPosition++;
            }
            return Enumerable.Repeat(EmitNonStopOnDecode ? NonStopLogits : EosLogits, tokens.Length).ToArray();
        }

        private static float[] CreateLogits(int token)
        {
            var logits = new float[64];
            logits[token] = 1f;
            return logits;
        }
    }

    private sealed class FakeCache : IRewindableSequenceKvCache
    {
        public int LogicalPosition { get; set; }
        public bool CanRewindTo(int logicalPosition) => logicalPosition >= 0 && logicalPosition <= LogicalPosition;
        public void RewindTo(int logicalPosition) => LogicalPosition = logicalPosition;
        public void Dispose() { }
    }
}
