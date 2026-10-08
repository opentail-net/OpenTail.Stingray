using System.Security.Cryptography;
using System.Text;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>The pure parts of golden capture: request, response, tokenizer output, version, hash.</summary>
public sealed class GoldenCaptureParsingTests
{
    [Fact]
    public void Request_IsGreedyRawIds_WithLogProbs_AndInvariantNumbers()
    {
        string json = GoldenCaptureParsing.BuildCompletionRequest([504, 3575, 282], 24);
        Assert.Equal("{\"prompt\":[504,3575,282],\"n_predict\":24,\"temperature\":0,\"top_k\":1,\"seed\":0,\"repeat_penalty\":1.0,\"cache_prompt\":false,\"return_tokens\":true,\"n_probs\":2}", json);
        using var doc = System.Text.Json.JsonDocument.Parse(json);     // and it is valid JSON
        Assert.Equal(3, doc.RootElement.GetProperty("prompt").GetArrayLength());
    }

    [Fact]
    public void Completion_ReadsTokensContentCountsAndReferenceMargins()
    {
        // Shape taken from a real llama-server (build 10306) response: top_logprobs sorted best-first.
        const string json = """
            {"content":" Paris.","tokens":[7042,30],"tokens_predicted":2,"tokens_evaluated":5,
             "completion_probabilities":[
               {"id":7042,"logprob":-0.5,"top_logprobs":[{"id":7042,"logprob":-0.5},{"id":260,"logprob":-1.75}]},
               {"id":30,"logprob":-0.25,"top_logprobs":[{"id":30,"logprob":-0.25},{"id":28,"logprob":-0.30}]}]}
            """;
        var r = GoldenCaptureParsing.ParseCompletion(json);
        Assert.Equal([7042, 30], r.Tokens);
        Assert.Equal(" Paris.", r.Content);
        Assert.Equal(5, r.TokensEvaluated);
        Assert.Equal([1.25, 0.05], r.Margins!);
    }

    [Fact]
    public void Completion_WithoutProbabilities_HasNoMargins_AndWithoutTokensIsRejected()
    {
        Assert.Null(GoldenCaptureParsing.ParseCompletion("""{"content":"x","tokens":[1],"tokens_evaluated":1}""").Margins);
        Assert.Throws<InvalidDataException>(() => GoldenCaptureParsing.ParseCompletion("""{"content":"x"}"""));
    }

    [Theory]
    [InlineData("[504, 3575, 282, 4649, 314]", new[] { 504, 3575, 282, 4649, 314 })]
    [InlineData("some log line\n[1, 2]\ntrailing", new[] { 1, 2 })]
    [InlineData("[]", new int[0])]
    public void TokenizerOutput_ReadsTheFirstIdList(string stdout, int[] expected) =>
        Assert.Equal(expected, GoldenCaptureParsing.ParseTokenizeIds(stdout));

    [Fact]
    public void TokenizerOutput_WithoutAList_IsRejected() =>
        Assert.Throws<InvalidDataException>(() => GoldenCaptureParsing.ParseTokenizeIds("error: model not found"));

    [Fact]
    public void Version_IsReadFromServerOutput() =>
        Assert.Equal("10306 (6b5c2efb4)", GoldenCaptureParsing.ParseVersion("version: 10306 (6b5c2efb4)\nbuilt with Clang 20.1.8 for Windows x86_64"));

    [Fact]
    public void Sha256_MatchesTheFrameworkHash_AndReportsProgress()
    {
        string path = Path.Combine(Path.GetTempPath(), "stingray-sha-" + Guid.NewGuid().ToString("N"));
        try
        {
            var bytes = new byte[3 * (1 << 20) + 123];
            new Random(1).NextBytes(bytes);
            File.WriteAllBytes(path, bytes);
            double last = 0;
            string got = GoldenCaptureParsing.Sha256Hex(path, p => last = p);
            Assert.Equal(Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant(), got);
            Assert.Equal(1.0, last, 3);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void Margins_MakeAMismatchANearTieOnlyWhereTheReferenceWasUndecided()
    {
        // The reference says 20 at position 1 with margin 0.3 (undecided) -> near-tie; with margin 4.0 (confident) -> diverged.
        float[] Counting(IReadOnlyList<int> h) { var l = new float[32]; l[(h[^1] + 1) % 32] = 10f; return l; }
        GoldenFile G(double margin) => new()
        {
            Architecture = "t",
            Cases = [new GoldenCase { Name = "c", PromptTokens = [3, 4], Tokens = [5, 20], NPredict = 2, Mode = "free", Margins = [3.0, margin] }],
        };
        var undecided = GoldenParityRunner.Run(G(0.3), () => new ScriptedStub(Counting));
        Assert.Equal(CaseVerdict.NearTie, undecided.Cases[0].Verdict);
        Assert.Equal(0.3, undecided.Cases[0].FirstMismatch!.ReferenceMargin);
        var confident = GoldenParityRunner.Run(G(4.0), () => new ScriptedStub(Counting));
        Assert.Equal(CaseVerdict.Diverged, confident.Cases[0].Verdict);
        Assert.Contains("ref-margin 4", confident.Format());
    }

    private sealed class ScriptedStub(Func<IReadOnlyList<int>, float[]> script) : OpenTail.Stingray.Core.IForwardPass
    {
        private readonly List<int> _h = [];
        public int VocabSize => 32;
        public int MaxSeqLen => 256;
        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0) { _h.Clear(); _h.AddRange(tokens); return script(_h); }
        public ReadOnlySpan<float> Forward(int token, int position) { _h.Add(token); return script(_h); }
        public void TruncateTo(int length) { }
        public void ResetCache() => _h.Clear();
        public void Dispose() { }
    }
}
