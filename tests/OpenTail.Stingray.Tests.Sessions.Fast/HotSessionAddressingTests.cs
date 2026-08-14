using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;

namespace OpenTail.Stingray.Tests.Sessions;

public sealed class HotSessionAddressingTests
{
    private const int Eos = 31;

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
        public byte[] DecodeBytes(int token) => [];
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
        public bool SnapKvEnabled => false;
        public long KvBytesPerToken => 1;
        public int MaxSeqLen => 64;
        public bool PrefillDequantCacheActive => false;
        public ISequenceKvCache CreateCache() => new FakeCache();

        public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos = 0)
        {
            var retained = Assert.IsType<FakeCache>(cache);
            retained.LogicalPosition = startPos + tokens.Count;
            var logits = new float[64];
            logits[Eos] = 1f;
            return logits;
        }

        public float[]?[] PrefillPackedMulti(ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits) =>
            throw new NotSupportedException();

        public float[][] BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches) =>
            throw new NotSupportedException();
    }

    [Fact]
    public void SessionAddress_ProducesDeterministicSessionId()
    {
        var addr1 = new SessionAddress("tenant1", "planner", "thread-42", "model-smollm");
        var addr2 = new SessionAddress("tenant1", "planner", "thread-42", "model-smollm");
        var addr3 = new SessionAddress("tenant1", "coder", "thread-42", "model-smollm");

        Assert.Equal(addr1.ToSessionId(), addr2.ToSessionId());
        Assert.NotEqual(addr1.ToSessionId(), addr3.ToSessionId());
    }

    [Fact]
    public void SessionAddress_ColonFields_DoNotAlias()
    {
        var addr1 = new SessionAddress("a:b", "c", "thread", "model");
        var addr2 = new SessionAddress("a", "b:c", "thread", "model");

        Assert.NotEqual(addr1.ToSessionId(), addr2.ToSessionId());

        // Null must not collide with empty. default(SessionAddress) has null fields; an address of
        // four empty strings is a different address and must map to a different session.
        var allNull = default(SessionAddress);
        var allEmpty = new SessionAddress("", "", "", "");
        Assert.NotEqual(allNull.ToSessionId(), allEmpty.ToSessionId());

        // Determinism: the same address must resolve identically across calls, or routing would
        // mint a fresh session every lookup and every turn would be a cold prefill.
        Assert.Equal(allEmpty.ToSessionId(), new SessionAddress("", "", "", "").ToSessionId());
    }

    [Fact]
    public async Task HotSessionRuntime_CreateAndOpenByAddress_RoutesToCorrectSession()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());

        var addressPlanner = new SessionAddress("tenant-alpha", "planner", "thread-101", "model-v1");
        var addressCoder = new SessionAddress("tenant-alpha", "coder", "thread-101", "model-v1");

        using var sessionPlanner = runtime.Create(addressPlanner);
        using var sessionCoder = runtime.Create(addressCoder);

        Assert.Equal(addressPlanner.ToSessionId(), sessionPlanner.SessionId);
        Assert.Equal(addressCoder.ToSessionId(), sessionCoder.SessionId);

        var retrievedPlanner = runtime.Open(addressPlanner);
        Assert.Same(sessionPlanner, retrievedPlanner);

        var retrievedCoder = runtime.Open(addressCoder);
        Assert.Same(sessionCoder, retrievedCoder);
    }
}
