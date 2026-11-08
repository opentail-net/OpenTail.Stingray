using System.Diagnostics;
using System.Text;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Qwen 3.8 plan, Phase 1 and Phase 3, real checkpoint, skipped when the UD-IQ1_S trunk is absent.
/// <list type="bullet">
/// <item>A prompt prefilled in small engine chunks (crossing many chunk boundaries) gives the same greedy continuation as
/// the unchunked path.</item>
/// <item>The fixed and per-token state bytes of a real qwen4exp sequence, measured from the loaded layer tensors.</item>
/// </list>
/// </summary>
public sealed class Qwen4ExpChunkedPrefillTests
{
    private const string Prompt =
        "The printing press was one of the most important inventions in history. Johannes Gutenberg developed movable type " +
        "in Mainz around 1440, which made books far cheaper to produce. Over the next decades the technology spread across " +
        "Europe, reaching Venice, Paris and London, and it transformed how knowledge was shared. In summary,";

    private static string? FindModel() => ModelLocator.FindDeep("Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf");

    [Fact]
    public async Task ChunkedPrefill_GivesTheSameGreedyContinuation_AsUnchunked()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        using var fixture = Fixture.Open(path!);
        int promptTokens = fixture.Tokenizer.Encode(Prompt).ToArray().Length;
        Assert.True(promptTokens >= 40, $"the prompt is only {promptTokens} tokens, too short to cross several chunks");

        string unchunked = await GenerateAsync(fixture, prefillChunkTokens: 0, steps: 12);
        string chunked = await GenerateAsync(fixture, prefillChunkTokens: 7, steps: 12);

        Console.WriteLine($"CHUNK_PROMPT_TOKENS {promptTokens}");
        Console.WriteLine($"CHUNK_UNCHUNKED {unchunked}");
        Console.WriteLine($"CHUNK_CHUNKED7 {chunked}");
        Assert.False(string.IsNullOrWhiteSpace(unchunked));
        Assert.Equal(unchunked, chunked);
    }

    [Fact]
    public void MeasureFixedAndPerTokenStateBytes_OnTheRealCheckpoint()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        using var fixture = Fixture.Open(path!);
        long fixedBytes = fixture.Batched.FixedBytesPerSequence;
        long perToken = fixture.Batched.KvBytesPerToken;
        Console.WriteLine($"STATE_BYTES fixed_per_sequence={fixedBytes} kv_per_token={perToken}");
        Assert.True(fixedBytes > 0, "a qwen4exp sequence has fixed recurrent and PLE state");
        Assert.True(perToken > 0, "a qwen4exp token adds KV, indexer and PLE bytes");
    }

    [Fact]
    public void MeasureTextThroughput_OnTheRealCheckpoint()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        using var fixture = Fixture.Open(path!);
        int[] ids = fixture.Tokenizer.Encode(Prompt).ToArray();
        var state = fixture.Pass.CreateSequenceState();
        int pos = 0;
        ReadOnlySpan<float> logits = default;

        var prefill = Stopwatch.StartNew();
        foreach (int id in ids) logits = fixture.Pass.Forward(id, pos++, state);
        prefill.Stop();

        const int decodeSteps = 12;
        var decode = Stopwatch.StartNew();
        for (int s = 0; s < decodeSteps; s++)
        {
            int best = 0;
            for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
            logits = fixture.Pass.Forward(best, pos++, state);
        }
        decode.Stop();

        double prefillTps = ids.Length / prefill.Elapsed.TotalSeconds;
        double decodeTps = decodeSteps / decode.Elapsed.TotalSeconds;
        Console.WriteLine($"THROUGHPUT prompt_tokens={ids.Length} prefill_tok_s={prefillTps:F2} decode_steps={decodeSteps} decode_tok_s={decodeTps:F2}");
        Assert.True(prefillTps > 0 && decodeTps > 0);
    }

    private static async Task<string> GenerateAsync(Fixture fixture, int prefillChunkTokens, int steps)
    {
        using var engine = new ContinuousBatchingEngine(fixture.Batched, fixture.Tokenizer, "qwen4exp-chunk",
            maxBatchSize: 2, prefillChunkTokens: prefillChunkTokens);
        var sp = new SamplingParams { Temperature = 0f, RepetitionPenalty = 1f, MaxNewTokens = steps };
        using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(20));
        var sb = new StringBuilder();
        await foreach (string piece in engine.GenerateAsync(Prompt, sp, cts.Token)) sb.Append(piece);
        return sb.ToString();
    }

    /// <summary>The real Qwen pass, loaded once per test, with its batched wrapper.</summary>
    private sealed class Fixture : IDisposable
    {
        private readonly GgufModel _model;
        public Qwen4ExpForwardPass Pass { get; }
        public Qwen4ExpBatchedForwardPass Batched { get; }
        public GgufTokenizer Tokenizer { get; }

        private Fixture(GgufModel model, Qwen4ExpForwardPass pass, Qwen4ExpBatchedForwardPass batched, GgufTokenizer tokenizer)
        {
            _model = model;
            Pass = pass;
            Batched = batched;
            Tokenizer = tokenizer;
        }

        public static Fixture Open(string path)
        {
            var model = GgufModel.Open(path);
            int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
            var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
            var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
            var pass = new Qwen4ExpForwardPass(model, hp, tensors);
            var batched = new Qwen4ExpBatchedForwardPass(pass, hp, totalLayers);
            return new Fixture(model, pass, batched, GgufTokenizer.FromGgufModel(model));
        }

        public void Dispose()
        {
            Pass.Dispose();
            _model.Dispose();
        }
    }
}
