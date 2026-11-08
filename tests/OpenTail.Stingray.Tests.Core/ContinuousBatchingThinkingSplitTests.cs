using System.Text;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Phase 6 (Qwen 3.8 plan), streaming: the server's streaming tool-call state machine only sees
/// <see cref="GenerateChunkKind.Text"/> chunks. These tests drive <see cref="ContinuousBatchingEngine"/> with a scripted
/// token sequence, so the reasoning split that the server depends on is checked without a checkpoint.
/// </summary>
public sealed class ContinuousBatchingThinkingSplitTests
{
    [Fact]
    public async Task ToolCallMentionedInsideReasoning_StreamsAsThinking_NotAsText()
    {
        var tok = new ScriptedTokenizer();
        var script = new List<int> { ScriptedTokenizer.ThinkOpen };
        script.AddRange(tok.Bytes("<tool_call>{\"name\":\"bad\"}</tool_call>"));
        script.Add(ScriptedTokenizer.ThinkClose);
        script.AddRange(tok.Bytes("Done."));
        script.Add(ScriptedTokenizer.Eos);

        var chunks = await GenerateAsync(tok, script.ToArray());

        Assert.Equal("<tool_call>{\"name\":\"bad\"}</tool_call>", Join(chunks, GenerateChunkKind.Thinking));
        Assert.Equal("Done.", Join(chunks, GenerateChunkKind.Text));
        Assert.DoesNotContain("<tool_call>", Join(chunks, GenerateChunkKind.Text), StringComparison.Ordinal);
    }

    [Fact]
    public async Task ToolCallAfterReasoning_StreamsAsText_SoTheParserSeesIt()
    {
        var tok = new ScriptedTokenizer();
        var script = new List<int> { ScriptedTokenizer.ThinkOpen };
        script.AddRange(tok.Bytes("I should look it up."));
        script.Add(ScriptedTokenizer.ThinkClose);
        script.AddRange(tok.Bytes("<tool_call>{\"name\":\"get\"}</tool_call>"));
        script.Add(ScriptedTokenizer.Eos);

        var chunks = await GenerateAsync(tok, script.ToArray());

        Assert.Equal("I should look it up.", Join(chunks, GenerateChunkKind.Thinking));
        Assert.Equal("<tool_call>{\"name\":\"get\"}</tool_call>", Join(chunks, GenerateChunkKind.Text));
    }

    private static async Task<List<GenerateChunk>> GenerateAsync(ScriptedTokenizer tok, int[] script)
    {
        var fwd = new ScriptedForwardPass(script, tok.VocabSize);
        using var engine = new ContinuousBatchingEngine(
            fwd, tok, "scripted", maxBatchSize: 1,
            thinkTokenId: ScriptedTokenizer.ThinkOpen, endThinkTokenId: ScriptedTokenizer.ThinkClose);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 128 };
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));

        var chunks = new List<GenerateChunk>();
        await foreach (var c in engine.GenerateChunksAsync("Q", sp, cts.Token))
            chunks.Add(c);
        return chunks;
    }

    private static string Join(List<GenerateChunk> chunks, GenerateChunkKind kind) =>
        string.Concat(chunks.Where(c => c.Kind == kind).Select(c => c.Text));

    /// <summary>Byte-level tokenizer with two single-token reasoning markers.</summary>
    private sealed class ScriptedTokenizer : ITokenizer
    {
        public const int Pad = 0, Bos = 1, Eos = 2, ThinkOpen = 3, ThinkClose = 4, ByteBase = 10;

        public int VocabSize => ByteBase + 256;
        public int BosTokenId => Bos;
        public int EosTokenId => Eos;
        public int UnknownTokenId => Pad;
        public int PadTokenId => Pad;
        public bool AddBosToken => false;
        public (int Open, int Close) ReasoningTokens => (ThinkOpen, ThinkClose);

        public IEnumerable<int> Bytes(string s) => Encoding.UTF8.GetBytes(s).Select(b => ByteBase + b);

        public IReadOnlyList<int> Encode(string text) => Bytes(text).ToArray();

        public string Decode(IEnumerable<int> tokens)
        {
            var sb = new StringBuilder();
            foreach (int t in tokens)
                sb.Append(Encoding.UTF8.GetString(DecodeBytes(t)));
            return sb.ToString();
        }

        public byte[] DecodeBytes(int token) => token switch
        {
            ThinkOpen => Encoding.UTF8.GetBytes("<think>"),
            ThinkClose => Encoding.UTF8.GetBytes("</think>"),
            >= ByteBase and < ByteBase + 256 => [(byte)(token - ByteBase)],
            _ => [],
        };
    }

    /// <summary>Emits a fixed script, one token per call: the prefill's first token, then one per decode step.</summary>
    private sealed class ScriptedForwardPass(int[] script, int vocab) : IBatchedForwardPass
    {
        private sealed class Cache : ISequenceKvCache
        {
            public int Index;
            public void Dispose() { }
        }

        public bool SnapKvEnabled => false;
        public long KvBytesPerToken => 1;
        public int MaxSeqLen => 4096;
        public bool PrefillDequantCacheActive => false;

        public ISequenceKvCache CreateCache() => new Cache();

        private float[] Row(int index)
        {
            int token = index < script.Length ? script[index] : ScriptedTokenizer.Eos;
            var row = new float[vocab];
            row[token] = 1f;
            return row;
        }

        public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos = 0)
        {
            ((Cache)cache).Index = 0;
            return Row(0);
        }

        public float[]?[] PrefillPackedMulti(
            ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits)
        {
            var result = new float[]?[chunks.Length];
            for (int i = 0; i < chunks.Length; i++)
            {
                ((Cache)caches[i]).Index = 0;
                result[i] = wantLogits[i] ? Row(0) : null;
            }
            return result;
        }

        public float[][] BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches)
        {
            var result = new float[tokens.Length][];
            for (int i = 0; i < tokens.Length; i++)
            {
                var c = (Cache)caches[i];
                c.Index++;
                result[i] = Row(c.Index);
            }
            return result;
        }
    }
}
