namespace OpenTail.Stingray.Tests.Sessions.Fast;


public sealed class SessionModelBudgetTests
{
    private const int NonStopToken = 7;
    private const int Eos = 31;

    private static SessionRequestDigest Digest(string val) => SessionRequestDigest.FromCanonicalValue(val);

    private sealed class Tokenizer : ITokenizer
    {
        public int VocabSize => 64;
        public int BosTokenId => 0;
        public int EosTokenId => Eos;
        public int UnknownTokenId => 0;
        public int PadTokenId => Eos;
        public bool AddBosToken => false;
        public IReadOnlyCollection<int> EogTokenIds => [Eos];
        public IReadOnlyList<int> Encode(string text) => [1, 2];
        public string Decode(IEnumerable<int> tokens) => "tok";
        public byte[] DecodeBytes(int token) => [(byte)('a' + (token % 26))];
    }

    private sealed class FakeCache : IRewindableSequenceKvCache
    {
        public int LogicalPosition { get; set; }
        public bool CanRewindTo(int logicalPosition) => logicalPosition >= 0 && logicalPosition <= LogicalPosition;
        public void RewindTo(int logicalPosition) => LogicalPosition = logicalPosition;
        public void Dispose() { }
    }

    private sealed class FakeForwardPass : IBatchedForwardPass
    {
        private static readonly float[] NonStopLogits = CreateLogits(NonStopToken);
        public bool SnapKvEnabled => false;
        public long KvBytesPerToken => 10;
        public int MaxSeqLen => 64;
        public bool PrefillDequantCacheActive => false;

        public ISequenceKvCache CreateCache() => new FakeCache();

        public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos = 0)
        {
            var retained = Assert.IsType<FakeCache>(cache);
            retained.LogicalPosition = startPos + tokens.Count;
            return NonStopLogits;
        }

        public float[]?[] PrefillPackedMulti(ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits) =>
            throw new NotSupportedException();

        public float[][] BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches)
        {
            for (int i = 0; i < caches.Length; i++)
            {
                var cache = Assert.IsType<FakeCache>(caches[i]);
                cache.LogicalPosition++;
            }
            return Enumerable.Repeat(NonStopLogits, tokens.Length).ToArray();
        }

        private static float[] CreateLogits(int winner)
        {
            var logits = new float[64];
            logits[winner] = 10f;
            return logits;
        }
    }

    [Fact]
    public async Task SessionModelBudget_EnforcesPerModelPartitionLimits()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);

        var modelBudgets = new Dictionary<string, long>
        {
            ["model-a"] = 1_000,
            ["model-b"] = 10_000
        };

        var options = new HotSessionRuntimeOptions(maxResidentBytes: 20_000, modelBudgets: modelBudgets);
        var runtime = new HotSessionRuntime(engine, new Tokenizer(), options);

        var addrA = new SessionAddress("tenant1", "coder", "t1", "model-a");
        var addrB = new SessionAddress("tenant1", "planner", "t1", "model-b");

        using var sessionA = runtime.Create(addrA);
        using var sessionB = runtime.Create(addrB);

        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        // Turn 1 on Session A under model-a
        await sessionA.RunTurnAsync("hello", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("hello"));
        long modelAResident = runtime.GetModelResidentBytes("model-a");
        Assert.True(modelAResident > 0, "Model A resident bytes should be tracked.");

        // Turn 1 on Session B under model-b
        await sessionB.RunTurnAsync("world", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("world"));
        long modelBResident = runtime.GetModelResidentBytes("model-b");
        Assert.True(modelBResident > 0, "Model B resident bytes should be tracked.");
    }

    [Fact]
    public async Task SessionModelBudget_RejectsOverBudgetTurnForSpecificModel()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);

        // Restrictive cap for model-a (only 1 byte, smaller than 1 token kv bytes)
        var modelBudgets = new Dictionary<string, long>
        {
            ["model-a"] = 1,
            ["model-b"] = 1_000_000
        };

        var options = new HotSessionRuntimeOptions(maxResidentBytes: 2_000_000, modelBudgets: modelBudgets);
        var runtime = new HotSessionRuntime(engine, new Tokenizer(), options);

        var addrA = new SessionAddress("tenant1", "coder", "t1", "model-a");
        var addrB = new SessionAddress("tenant1", "planner", "t1", "model-b");

        using var sessionA = runtime.Create(addrA);
        using var sessionB = runtime.Create(addrB);

        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        // Session A should throw budget exceeded exception because model-a budget is only 1 byte
        await Assert.ThrowsAsync<SessionResourceBudgetExceededException>(() =>
            sessionA.RunTurnAsync("hello", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("hello")));

        // Session B should succeed because model-b has 1,000,000 bytes allocated
        var resB = await sessionB.RunTurnAsync("hello", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("hello"));
        Assert.NotNull(resB);
    }

    [Fact]
    public void SessionModelBudget_ConcurrentInFlightReservations_EnforcesModelCap()
    {
        var options = new HotSessionRuntimeOptions(
            maxResidentBytes: 100_000,
            modelBudgets: new Dictionary<string, long> { ["model-shared"] = 1_000 });

        var budget = new SessionResourceBudget(options);
        var id1 = SessionId.New();
        var id2 = SessionId.New();

        // Turn 1 reserves 600 bytes out of 1000 byte model budget
        using var res1 = budget.Reserve(id1, 600, "model-shared");
        Assert.NotNull(res1);

        // Turn 2 attempting to reserve 500 bytes (600 + 500 = 1100 > 1000) should be rejected in-flight
        Assert.Throws<SessionResourceBudgetExceededException>(() =>
            budget.Reserve(id2, 500, "model-shared"));

        // Turn 2 attempting to reserve 300 bytes (600 + 300 = 900 <= 1000) should succeed
        using var res2 = budget.Reserve(id2, 300, "model-shared");
        Assert.NotNull(res2);
    }

    /// <summary>
    /// Rolling renewal must respect the model cap just as admission does.
    /// </summary>
    /// <remarks>
    /// <c>_modelReservedBytes</c> already contains the renewing session's own reservation, so
    /// crediting it back when computing headroom let a session grow past the cap by the size of its
    /// current reservation. Renewal fires on every decode step, so that was a continuous leak of the
    /// guarantee rather than a corner case — and the in-flight admission test above cannot see it,
    /// because it never renews.
    /// </remarks>
    [Fact]
    public void SessionModelBudget_RenewalCannotExceedModelCapByItsOwnReservation()
    {
        var options = new HotSessionRuntimeOptions(
            maxResidentBytes: 10_000_000,
            modelBudgets: new Dictionary<string, long> { ["model-shared"] = 1_000 });

        var budget = new SessionResourceBudget(options);
        var idA = SessionId.New();
        var idB = SessionId.New();

        using var resA = budget.Reserve(idA, 600, "model-shared");
        using var resB = budget.Reserve(idB, 300, "model-shared");   // model total now 900 of 1000

        // A renewal to 700 would put the model at 1000 exactly — allowed.
        Assert.True(resA.TryRenew(700));

        // A renewal to 800 would put the model at 1100 — must be refused. Before the fix this was
        // accepted, because A's own 700-byte reservation was credited back into the headroom.
        Assert.False(resA.TryRenew(800));

        // Refusal must not have mutated the accounting: the previous renewal still stands.
        Assert.True(resA.TryRenew(700));
    }
}
