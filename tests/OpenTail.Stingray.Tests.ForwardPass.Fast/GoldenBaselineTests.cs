using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

public sealed class GoldenBaselineTests
{
    [Fact]
    public void BaselineFile_RoundTripSerialization_PreservesAllFields()
    {
        var baseline = new GoldenBaselineFile
        {
            Host = "TestHost",
            TimestampUtc = "2026-10-09T12:00:00Z",
            StingrayVersion = "1.0.7",
            Entries =
            [
                new GoldenBaselineEntry
                {
                    Architecture = "smollm3",
                    GoldenFile = "smollm3.golden.json",
                    ModelFile = "SmolLM3-Q4_K_M.gguf",
                    ModelSha256 = "abcdef123456",
                    PinStatus = "Verified",
                    Verdict = "Exact",
                    Passed = true,
                    ComparedTokens = 24,
                    MatchedTokens = 24,
                    DecodeTokensPerSecond = 85.5,
                    ElapsedSeconds = 1.25,
                    StepwiseMaxAbsDiff = 0.42f,
                    Detail = null
                },
                new GoldenBaselineEntry
                {
                    Architecture = "gptneox",
                    GoldenFile = "gptneox.golden.json",
                    ModelFile = "pythia-160m.Q8_0.gguf",
                    PinStatus = "Verified",
                    Verdict = "NearTie",
                    Passed = true,
                    ComparedTokens = 24,
                    MatchedTokens = 23,
                    DecodeTokensPerSecond = 120.0,
                    ElapsedSeconds = 0.85
                }
            ]
        };

        string json = baseline.ToJson();
        var parsed = GoldenBaselineFile.Parse(json);

        Assert.Equal(1, parsed.Schema);
        Assert.Equal("TestHost", parsed.Host);
        Assert.Equal("2026-10-09T12:00:00Z", parsed.TimestampUtc);
        Assert.Equal("1.0.7", parsed.StingrayVersion);
        Assert.Equal(2, parsed.Entries.Count);

        var first = parsed.Entries[0];
        Assert.Equal("smollm3", first.Architecture);
        Assert.Equal("smollm3.golden.json", first.GoldenFile);
        Assert.Equal("SmolLM3-Q4_K_M.gguf", first.ModelFile);
        Assert.Equal("abcdef123456", first.ModelSha256);
        Assert.Equal("Verified", first.PinStatus);
        Assert.Equal("Exact", first.Verdict);
        Assert.True(first.Passed);
        Assert.Equal(24, first.ComparedTokens);
        Assert.Equal(24, first.MatchedTokens);
        Assert.Equal(85.5, first.DecodeTokensPerSecond);
        Assert.Equal(1.25, first.ElapsedSeconds);
        Assert.Equal(0.42f, first.StepwiseMaxAbsDiff);

        var second = parsed.Entries[1];
        Assert.Equal("gptneox", second.Architecture);
        Assert.Equal("NearTie", second.Verdict);
        Assert.True(second.Passed);
    }

    [Fact]
    public void BaselineFile_UnsupportedSchema_Throws()
    {
        string json = """{ "schema": 999, "host": "test", "entries": [] }""";
        Assert.Throws<InvalidDataException>(() => GoldenBaselineFile.Parse(json));
    }

    [Fact]
    public void BaselineComparator_DetectsRegressionsAndImprovements()
    {
        var prior = new GoldenBaselineFile
        {
            Entries =
            [
                new GoldenBaselineEntry
                {
                    Architecture = "modelA",
                    GoldenFile = "modelA.golden.json",
                    Verdict = "Exact",
                    Passed = true,
                    DecodeTokensPerSecond = 100.0
                },
                new GoldenBaselineEntry
                {
                    Architecture = "modelB",
                    GoldenFile = "modelB.golden.json",
                    Verdict = "Diverged",
                    Passed = false,
                    DecodeTokensPerSecond = 50.0
                },
                new GoldenBaselineEntry
                {
                    Architecture = "modelC",
                    GoldenFile = "modelC.golden.json",
                    Verdict = "Exact",
                    Passed = true,
                    DecodeTokensPerSecond = 100.0
                },
                new GoldenBaselineEntry
                {
                    Architecture = "modelD",
                    GoldenFile = "modelD.golden.json",
                    Verdict = "Exact",
                    Passed = true,
                    DecodeTokensPerSecond = 100.0
                },
                new GoldenBaselineEntry
                {
                    Architecture = "modelGone",
                    GoldenFile = "modelGone.golden.json",
                    Verdict = "Exact",
                    Passed = true,
                    DecodeTokensPerSecond = 80.0
                }
            ]
        };

        var current = new GoldenBaselineFile
        {
            Entries =
            [
                new GoldenBaselineEntry
                {
                    Architecture = "modelA",
                    GoldenFile = "modelA.golden.json",
                    Verdict = "Diverged",
                    Passed = false,
                    DecodeTokensPerSecond = 98.0
                },
                new GoldenBaselineEntry
                {
                    Architecture = "modelB",
                    GoldenFile = "modelB.golden.json",
                    Verdict = "Exact",
                    Passed = true,
                    DecodeTokensPerSecond = 52.0
                },
                new GoldenBaselineEntry
                {
                    Architecture = "modelC",
                    GoldenFile = "modelC.golden.json",
                    Verdict = "Exact",
                    Passed = true,
                    DecodeTokensPerSecond = 75.0 // -25% speed drop
                },
                new GoldenBaselineEntry
                {
                    Architecture = "modelD",
                    GoldenFile = "modelD.golden.json",
                    Verdict = "Exact",
                    Passed = true,
                    DecodeTokensPerSecond = 130.0 // +30% speed gain
                },
                new GoldenBaselineEntry
                {
                    Architecture = "modelNew",
                    GoldenFile = "modelNew.golden.json",
                    Verdict = "Exact",
                    Passed = true,
                    DecodeTokensPerSecond = 90.0
                }
            ]
        };

        var diffs = GoldenBaselineComparator.Compare(prior, current, speedTolerancePct: 0.15);
        var diffMap = diffs.ToDictionary(d => d.GoldenFile);

        Assert.Equal(BaselineDiffStatus.Regression, diffMap["modelA.golden.json"].Status);
        Assert.Equal(BaselineDiffStatus.Improvement, diffMap["modelB.golden.json"].Status);
        Assert.Equal(BaselineDiffStatus.SpeedDrop, diffMap["modelC.golden.json"].Status);
        Assert.Equal(BaselineDiffStatus.SpeedGain, diffMap["modelD.golden.json"].Status);
        Assert.Equal(BaselineDiffStatus.NewEntry, diffMap["modelNew.golden.json"].Status);
        Assert.Equal(BaselineDiffStatus.MissingInCurrent, diffMap["modelGone.golden.json"].Status);
    }
}
