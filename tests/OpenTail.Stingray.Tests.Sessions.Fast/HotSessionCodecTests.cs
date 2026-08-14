using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using System.Buffers.Binary;

namespace OpenTail.Stingray.Tests.Sessions.Fast;

public sealed class HotSessionCodecTests
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

        private static float[] CreateLogits(int token)
        {
            var logits = new float[64];
            logits[token] = 1f;
            return logits;
        }
    }

    [Fact]
    public async Task HotSession_ExportAndImportState_PreservesPayloadHashAndSessionId()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();

        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 2 };
        await session.RunTurnAsync("hello", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("hello"));

        var stateBytes = session.ExportState("model-test");
        Assert.NotNull(stateBytes);
        Assert.True(stateBytes.Length > 0);

        // Delete active session to allow importing
        runtime.Delete(session.SessionId);

        using var importedSession = runtime.ImportState(stateBytes, "model-test");

        Assert.Equal(session.SessionId, importedSession.SessionId);
        Assert.Equal(session.Cursor.InputIdentity, importedSession.Cursor.InputIdentity);
    }

    [Fact]
    public async Task HotSession_ImportState_RejectsModelFingerprintMismatch()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();

        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };
        await session.RunTurnAsync("hello", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("hello"));

        var stateBytes = session.ExportState("model-original");
        runtime.Delete(session.SessionId);

        Assert.Throws<SessionCursorFormatException>(() =>
            runtime.ImportState(stateBytes, "model-different"));
    }

    [Fact]
    public async Task HotSession_SafetensorsState_PersistsFormatAndRejectsGgufRestore()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());
        using var session = runtime.Create();

        await session.RunTurnAsync("hello", new SamplingParams { Temperature = 0f, MaxNewTokens = 1 },
            SessionRevision.Initial, SessionOperationId.New(), Digest("hello"));

        byte[] stateBytes = session.ExportState("model-package", modelFormat: ModelFormat.SafeTensors);
        var envelope = SessionStateCodec.Decode(stateBytes);
        Assert.Equal(ModelFormat.SafeTensors, envelope.Abi.ModelFormat);

        runtime.Delete(session.SessionId);

        Assert.Throws<SessionCursorFormatException>(() =>
            runtime.ImportState(stateBytes, "model-package"));

        using var restored = runtime.ImportState(stateBytes, "model-package",
            expectedModelFormat: ModelFormat.SafeTensors);
        Assert.Equal(session.SessionId, restored.SessionId);
    }

    [Fact]
    public void HotSession_ImportState_RejectsCorruptHeader()
    {
        var fwd = new FakeForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new Tokenizer());

        byte[] corruptBytes = [0xFF, 0xFF, 0xFF, 0xFF, 0x01, 0x00, 0x01, 0x00];

        Assert.Throws<SessionCursorFormatException>(() =>
            runtime.ImportState(corruptBytes));
    }

    [Fact]
    public void SessionStateCodec_LegacyEnvelope_DefaultsToGguf()
    {
        var cursor = new SessionCursor([], 0, 0, 0, 0, StateCoverage.Full);
        var abi = new SessionStateABI("legacy-model", 10, 64, ModelFormat.SafeTensors);
        var envelope = new SessionStateEnvelope(SessionId.New(), new SessionCursorEnvelope(cursor, []), abi,
            SessionStateCodec.ComputeCompatibilityKey(abi), StatePayloadHash.Compute([]), []);

        byte[] legacyState = DowngradeV2EnvelopeToV1(SessionStateCodec.Encode(envelope));
        var decoded = SessionStateCodec.Decode(legacyState);

        Assert.Equal(ModelFormat.Gguf, decoded.Abi.ModelFormat);
    }

    private static byte[] DowngradeV2EnvelopeToV1(byte[] version2)
    {
        const int headerBytes = 8;
        const int directoryEntryBytes = 11;
        const int v1DirectoryEntries = 5;
        const int v2DirectoryEntries = 6;
        const int modelFormatPayloadBytes = 1;
        int v2PayloadOffset = headerBytes + (v2DirectoryEntries * directoryEntryBytes);
        int v1PayloadOffset = headerBytes + (v1DirectoryEntries * directoryEntryBytes);
        var version1 = new byte[version2.Length - directoryEntryBytes - modelFormatPayloadBytes];

        version2.AsSpan(0, headerBytes).CopyTo(version1);
        BinaryPrimitives.WriteUInt16LittleEndian(version1.AsSpan(4), 1);
        BinaryPrimitives.WriteUInt16LittleEndian(version1.AsSpan(6), v1DirectoryEntries);

        for (int i = 0; i < v1DirectoryEntries; i++)
        {
            var source = version2.AsSpan(headerBytes + (i * directoryEntryBytes), directoryEntryBytes);
            var target = version1.AsSpan(headerBytes + (i * directoryEntryBytes), directoryEntryBytes);
            source[..3].CopyTo(target);
            int offset = BinaryPrimitives.ReadInt32LittleEndian(source[3..7]);
            BinaryPrimitives.WriteInt32LittleEndian(target[3..7], offset - directoryEntryBytes);
            source[7..].CopyTo(target[7..]);
        }

        version2.AsSpan(v2PayloadOffset, version2.Length - v2PayloadOffset - modelFormatPayloadBytes)
            .CopyTo(version1.AsSpan(v1PayloadOffset));
        return version1;
    }

    /// <summary>
    /// The GGUF compatibility key must hash exactly what it hashed before <c>ModelFormat</c> existed.
    /// The expected value is an independently computed SHA-256 of the literal string
    /// <c>"fp:128:512"</c> — deliberately NOT produced by calling the method under test, because a
    /// self-referential expectation would accept any formula change and let every pre-existing session
    /// silently fail to restore.
    /// </summary>
    [Fact]
    public void ComputeCompatibilityKey_Gguf_MatchesPreDiscriminatorHash()
    {
        var abi = new SessionStateABI("fp", 128, 512);

        Assert.Equal(ModelFormat.Gguf, abi.ModelFormat);
        Assert.Equal("995d70e3185b6db02bbbf69fe5607f8e2d3cec467b4525b02d2535cd652ddfe2",
            SessionStateCodec.ComputeCompatibilityKey(abi));
    }

    /// <summary>A SafeTensors session must not be interchangeable with a GGUF one over the same ABI.</summary>
    [Fact]
    public void ComputeCompatibilityKey_SafeTensors_DiffersFromGguf()
    {
        var gguf = new SessionStateABI("fp", 128, 512);
        var safetensors = gguf with { ModelFormat = ModelFormat.SafeTensors };

        Assert.NotEqual(SessionStateCodec.ComputeCompatibilityKey(gguf),
            SessionStateCodec.ComputeCompatibilityKey(safetensors));
    }
}
