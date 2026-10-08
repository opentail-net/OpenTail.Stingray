using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// The golden-parity logic with a scripted forward pass, so classification (exact / near-tie / diverged), teacher forcing, the
/// stepwise-vs-prefill self-check and the golden file format are proven without any checkpoint.
/// </summary>
public sealed class GoldenParityRunnerTests
{
    private const int Vocab = 32;

    /// <summary>Logits are a pure function of the token history, optionally perturbed on the stepwise path.</summary>
    private sealed class ScriptedPass(Func<IReadOnlyList<int>, float[]> script, float stepwiseNoise = 0f) : IForwardPass
    {
        private readonly List<int> _history = [];
        public int VocabSize => Vocab;
        public int MaxSeqLen => 256;
        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
        {
            _history.Clear();
            _history.AddRange(tokens);
            return script(_history);
        }
        public ReadOnlySpan<float> Forward(int token, int position)
        {
            _history.Add(token);
            var l = script(_history);
            if (stepwiseNoise != 0f) l[0] += stepwiseNoise;       // only the decode path is perturbed
            return l;
        }
        public void TruncateTo(int length) { }
        public void ResetCache() => _history.Clear();
        public void Dispose() { }
    }

    /// <summary>"Next token = last token + 1": every position has one clear winner (margin 10).</summary>
    private static float[] Counting(IReadOnlyList<int> history)
    {
        var l = new float[Vocab];
        l[(history[^1] + 1) % Vocab] = 10f;
        return l;
    }

    private static GoldenFile Golden(string mode, int[] tokens, int[]? prompt = null) => new()
    {
        Architecture = "test",
        Cases = [new GoldenCase { Name = "c", PromptTokens = prompt ?? [3, 4], Tokens = tokens, NPredict = tokens.Length, Mode = mode }],
    };

    [Fact]
    public void Exact_WhenEveryTokenMatches()
    {
        var r = GoldenParityRunner.Run(Golden("free", [5, 6, 7, 8]), () => new ScriptedPass(Counting));
        Assert.Equal(CaseVerdict.Exact, r.Verdict);
        Assert.True(r.Passed);
        Assert.Equal(4, r.Cases[0].Matched);
        Assert.Empty(r.Cases[0].Mismatches);
    }

    [Fact]
    public void Diverged_WhenOurChoiceBeatsTheReferenceByALargeGap_AndFreeModeStopsComparing()
    {
        // Counting would give 5,6,7,8; the reference says 5,6,20,8 -> mismatch at index 2 with gap 10 (all of the margin).
        var r = GoldenParityRunner.Run(Golden("free", [5, 6, 20, 8]), () => new ScriptedPass(Counting));
        var c = r.Cases[0];
        Assert.Equal(CaseVerdict.Diverged, c.Verdict);
        Assert.False(r.Passed);
        Assert.Equal(2, c.FirstMismatch!.Index);
        Assert.Equal(20, c.FirstMismatch.Expected);
        Assert.Equal(7, c.FirstMismatch.Actual);
        Assert.Equal(10.0, c.FirstMismatch.Gap, 3);
        Assert.False(c.FirstMismatch.NearTie);
        Assert.Equal(3, c.Compared);                  // stopped after the first mismatch
        Assert.Single(c.Mismatches);
    }

    [Fact]
    public void NearTie_WhenTheReferenceTokenIsWithinTheToleranceOfOurTopChoice()
    {
        // At the position that predicts index 1, the reference token (20) sits 0.01 below our winner (6): a genuine near-tie.
        float[] Script(IReadOnlyList<int> h)
        {
            var l = Counting(h);
            if (h[^1] == 5) l[20] = 10f - 0.01f;
            return l;
        }
        var r = GoldenParityRunner.Run(Golden("free", [5, 20, 7]), () => new ScriptedPass(Script));
        var c = r.Cases[0];
        Assert.Equal(CaseVerdict.NearTie, c.Verdict);
        Assert.True(r.Passed);                         // a near-tie is not a failure
        Assert.True(c.FirstMismatch!.NearTie);
        Assert.Equal(0.01, c.FirstMismatch.Gap, 4);
        // The same file with a stricter tolerance is a divergence.
        var strict = GoldenParityRunner.Run(Golden("free", [5, 20, 7]), () => new ScriptedPass(Script), new ParityOptions { NearTieTolerance = 0.001 });
        Assert.Equal(CaseVerdict.Diverged, strict.Cases[0].Verdict);
    }

    [Fact]
    public void TeacherForced_FeedsTheReferenceTokens_AndComparesEveryPosition()
    {
        // Counting predicts last+1. Reference 5,9,10,30 after prompt 3,4:
        //   idx0: we predict 5            -> match
        //   idx1: after feeding 5 we predict 6, reference says 9 -> mismatch (actual 6)
        //   idx2: after feeding the REFERENCE token 9 we predict 10 -> match. (Free mode would have fed our own 6 and predicted 7.)
        //   idx3: after feeding 10 we predict 11, reference says 30 -> mismatch (actual 11)
        var r = GoldenParityRunner.Run(Golden("teacherForced", [5, 9, 10, 30]), () => new ScriptedPass(Counting));
        var c = r.Cases[0];
        Assert.Equal("teacherForced", c.Mode);
        Assert.Equal(4, c.Compared);
        Assert.Equal(2, c.Matched);
        Assert.Equal([1, 3], c.Mismatches.Select(m => m.Index).ToArray());
        Assert.Equal([6, 11], c.Mismatches.Select(m => m.Actual).ToArray());
        Assert.Equal(CaseVerdict.Diverged, c.Verdict);
    }

    [Fact]
    public void StepwiseCheck_PassesWhenConsistent_AndFailsOnAnInjectedInconsistency()
    {
        var ok = GoldenParityRunner.Run(Golden("free", [5, 6, 7]), () => new ScriptedPass(Counting));
        Assert.True(ok.Stepwise!.ArgmaxAgrees);
        Assert.True(ok.Stepwise.WithinBound);
        Assert.True(ok.Passed);

        // Perturb only the decode path hard enough to flip the argmax and exceed the bound.
        var bad = GoldenParityRunner.Run(Golden("free", [5, 6, 7]), () => new ScriptedPass(Counting, stepwiseNoise: 50f));
        Assert.False(bad.Stepwise!.ArgmaxAgrees);
        Assert.False(bad.Stepwise.WithinBound);
        Assert.False(bad.Passed);
        Assert.Contains("DISAGREES", bad.Format());
    }

    [Fact]
    public void StepwiseCheck_ANearTieArgmaxFlipWithinTheBound_PassesByDefault_AndFailsWhenAgreementIsRequired()
    {
        // Runner-up token 0 sits 0.1 below the winner; the decode path adds 0.5 to it, so the argmax flips while the paths differ by only 0.5.
        float[] Script(IReadOnlyList<int> h)
        {
            var l = Counting(h);
            if ((h[^1] + 1) % Vocab != 0) l[0] = 9.9f;
            return l;
        }
        var lenient = GoldenParityRunner.Run(Golden("free", [5, 6, 7]), () => new ScriptedPass(Script, stepwiseNoise: 0.5f));
        Assert.False(lenient.Stepwise!.ArgmaxAgrees);
        Assert.True(lenient.Stepwise.WithinBound);
        Assert.Equal(0.1f, lenient.Stepwise.ArgmaxGap, 3);
        Assert.True(lenient.Stepwise.ArgmaxGap <= 2 * lenient.Stepwise.MaxAbsDiff);   // a flip is always within 2 x the measured difference
        Assert.True(lenient.Stepwise.Passed);

        var strict = GoldenParityRunner.Run(Golden("free", [5, 6, 7]), () => new ScriptedPass(Script, stepwiseNoise: 0.5f),
            new ParityOptions { RequireStepwiseArgmaxAgreement = true });
        Assert.False(strict.Stepwise!.Passed);
        Assert.Contains("REQUIRED", strict.Format());
    }

    [Fact]
    public void Format_IsReadable_AndNamesTheDivergence()
    {
        var r = GoldenParityRunner.Run(Golden("free", [5, 6, 20]), () => new ScriptedPass(Counting));
        string text = r.Format();
        Assert.Contains("FAIL", text);
        Assert.Contains("expected 20 got 7", text);
        Assert.Contains("DIVERGED", text);
    }

    [Fact]
    public void GoldenFile_RoundTrips_AndRejectsAnUnknownSchema()
    {
        var g = new GoldenFile
        {
            Architecture = "olmo2",
            Notes = "from allenai/OLMo-2-0425-1B-GGUF",
            Model = new GoldenModel { FileName = "m.gguf", SizeBytes = 123, Sha256 = "ab", Source = "owner/repo" },
            Reference = new GoldenReference { Engine = "llama-server", Build = "10306 (6b5c2efb4)", Settings = "-ngl 0 -c 512 -t 4" },
            Cases = [new GoldenCase { Name = "c", PromptTokens = [1, 2], Tokens = [3, 4], NPredict = 2, Mode = "teacherForced" }],
            EngineSettings = new GoldenEngineSettings { Q8Prefill = true },
        };
        var back = GoldenFile.Parse(g.ToJson());
        Assert.Equal("olmo2", back.Architecture);
        Assert.Equal([1, 2], back.Cases[0].PromptTokens);
        Assert.Equal("teacherForced", back.Cases[0].Mode);
        Assert.True(back.EngineSettings!.Q8Prefill);
        Assert.Equal("10306 (6b5c2efb4)", back.Reference.Build);
        Assert.Contains("\"architecture\"", g.ToJson());            // camelCase on disk

        Assert.Throws<InvalidDataException>(() => GoldenFile.Parse(g.ToJson().Replace("\"schema\": 1", "\"schema\": 99")));
    }

    [Theory]
    [InlineData("{\"fileName\":\"C:\\\\Models\\\\x.gguf\"}", true)]            // JSON-escaped drive path
    [InlineData("{\"notes\":\"F:/_models/x.gguf\"}", true)]
    [InlineData("{\"notes\":\"see \\\\\\\\server\\\\share\\\\x\"}", true)]        // UNC
    [InlineData("{\"notes\":\"/Users/someone/x\"}", true)]
    [InlineData("{\"notes\":\"https://huggingface.co/owner/repo\"}", false)]   // a URL is not a drive path
    [InlineData("{\"model\":{\"fileName\":\"OLMo-2-0425-1B-Q8_0.gguf\"}}", false)]
    public void MachineSpecificTextGuard_FlagsPathsButNotUrlsOrFileNames(string json, bool expectFinding) =>
        Assert.Equal(expectFinding, GoldenFile.FindMachineSpecificText(json).Count > 0);

    [Fact]
    public void Save_RefusesAGoldenThatWouldLeakAPath_AndWritesACleanOne()
    {
        string dir = Path.Combine(Path.GetTempPath(), "stingray-golden-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var leaky = new GoldenFile { Architecture = "t", Notes = @"captured from F:\_models\x.gguf" };
            Assert.Throws<InvalidOperationException>(() => leaky.Save(Path.Combine(dir, "leaky.json")));
            Assert.False(File.Exists(Path.Combine(dir, "leaky.json")));

            var clean = new GoldenFile { Architecture = "t", Notes = "ok" };
            clean.Save(Path.Combine(dir, "clean.json"));
            Assert.Equal("t", GoldenFile.Load(Path.Combine(dir, "clean.json")).Architecture);
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void EveryCheckedInGolden_IsParseable_AndFreeOfMachineSpecificText()
    {
        string? root = AppContext.BaseDirectory;
        while (root is not null && !File.Exists(Path.Combine(root, "CLAUDE.md"))) root = Path.GetDirectoryName(root);
        Assert.SkipWhen(root is null, "repo layout not found");
        string dir = Path.Combine(root!, "tests", "OpenTail.Stingray.Tests.ForwardPass", "Goldens");
        if (!Directory.Exists(dir)) return;                       // no goldens yet is fine; the guard activates with the first one
        foreach (var file in Directory.EnumerateFiles(dir, "*.json"))
        {
            string text = File.ReadAllText(file);
            var findings = GoldenFile.FindMachineSpecificText(text);
            Assert.True(findings.Count == 0, $"{Path.GetFileName(file)} contains machine-specific text: {string.Join(", ", findings)}");
            var g = GoldenFile.Parse(text);
            Assert.False(string.IsNullOrWhiteSpace(g.Architecture), $"{Path.GetFileName(file)}: architecture is empty");
            Assert.False(string.IsNullOrWhiteSpace(g.Model.FileName), $"{Path.GetFileName(file)}: model.fileName is empty");
            Assert.True(!g.Model.FileName.Contains('/') && !g.Model.FileName.Contains('\\'),$"{Path.GetFileName(file)}: model.fileName must be a bare file name");
        }
    }
}
