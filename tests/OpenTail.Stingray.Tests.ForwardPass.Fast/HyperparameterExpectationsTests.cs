using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

public sealed class HyperparameterExpectationsTests
{
    private static readonly ModelHyperparams Hp = new() { RopeDim = 16, NumExperts = 8, HasFfnBias = true, UseParallelResidual = true };

    [Fact]
    public void Holds_ForInts_Bools_AndNull_CaseInsensitively()
    {
        var failures = HyperparameterExpectations.Check(Hp, new Dictionary<string, string>
        {
            ["ropedim"] = "16", ["NumExperts"] = "8", ["HASFFNBIAS"] = "TRUE", ["XieluAlphaN"] = "null",
        });
        Assert.Empty(failures);
    }

    [Fact]
    public void Reports_AWrongValue_AndAnUnknownPropertyAsFailures()
    {
        var failures = HyperparameterExpectations.Check(Hp, new Dictionary<string, string> { ["RopeDim"] = "64", ["NoSuchProperty"] = "1" });
        Assert.Equal(2, failures.Count);
        Assert.Contains(failures, f => f.Contains("RopeDim") && f.Contains("expected 64") && f.Contains("resolved 16"));
        Assert.Contains(failures, f => f.Contains("unknown ModelHyperparams property 'NoSuchProperty'"));
    }

    [Fact]
    public void NoExpectations_IsClean() => Assert.Empty(HyperparameterExpectations.Check(Hp, null));

    [Fact]
    public void GoldenFile_RoundTripsExpectationsAndMinConfident()
    {
        var g = new GoldenFile
        {
            Architecture = "t",
            ExpectedHyperparameters = new Dictionary<string, string> { ["RopeDim"] = "16" },
            Cases = [new GoldenCase { Name = "c", PromptTokens = [1], Tokens = [2], MinConfident = 3 }],
        };
        var back = GoldenFile.Parse(g.ToJson());
        Assert.Equal("16", back.ExpectedHyperparameters!["RopeDim"]);
        Assert.Equal(3, back.Cases[0].MinConfident);
    }

    [Fact]
    public void MinConfident_RequiresEnoughMatchesWhereTheReferenceWasConfident()
    {
        float[] Counting(IReadOnlyList<int> h) { var l = new float[32]; l[(h[^1] + 1) % 32] = 10f; return l; }
        GoldenFile G(int minConfident, double[] margins) => new()
        {
            Architecture = "t",
            Cases = [new GoldenCase { Name = "c", PromptTokens = [3, 4], Tokens = [5, 6, 7], NPredict = 3, Mode = "teacherForced", Margins = margins, MinConfident = minConfident }],
        };
        // All three tokens match; margins 2.0, 0.5, 3.0 -> two are confident (>= 1.5).
        var enough = GoldenParityRunner.Run(G(2, [2.0, 0.5, 3.0]), () => new Pass(Counting));
        Assert.Equal(2, enough.Cases[0].ConfidentMatched);
        Assert.True(enough.Passed);

        var tooFew = GoldenParityRunner.Run(G(3, [2.0, 0.5, 3.0]), () => new Pass(Counting));
        Assert.Equal(CaseVerdict.Exact, tooFew.Cases[0].Verdict);          // every token matched, but that is not enough evidence
        Assert.False(tooFew.Passed);
        Assert.Contains("INSUFFICIENT EVIDENCE", tooFew.Format());
    }

    private sealed class Pass(Func<IReadOnlyList<int>, float[]> script) : IForwardPass
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
