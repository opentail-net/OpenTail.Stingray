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
            OsDescription = "Microsoft Windows 11 Pro",
            ProcessArchitecture = "X64",
            ProcessorCount = 16,
            RuntimeDescription = ".NET 10.0.0",
            TimestampUtc = "2026-10-09T12:00:00Z",
            StingrayVersion = "1.0.7",
            Entries =
            [
                new GoldenBaselineEntry
                {
                    Architecture = "smollm3",
                    GoldenFile = "smollm3.golden.json",
                    GoldenSha256 = "1122334455667788",
                    ModelFile = "SmolLM3-Q4_K_M.gguf",
                    ModelSha256 = "abcdef123456",
                    PinStatus = "Verified",
                    Verdict = "Exact",
                    Passed = true,
                    ComparedTokens = 24,
                    MatchedTokens = 24,
                    DecodeSteps = 23,
                    DecodeTokensPerSecond = 85.5,
                    PrefillTokensPerSecond = 450.0,
                    ElapsedSeconds = 1.25,
                    StepwiseMaxAbsDiff = 0.42f,
                    Detail = null
                },
                new GoldenBaselineEntry
                {
                    Architecture = "gptneox",
                    GoldenFile = "gptneox.golden.json",
                    GoldenSha256 = "9988776655443322",
                    ModelFile = "pythia-160m.Q8_0.gguf",
                    PinStatus = "Verified",
                    Verdict = "NearTie",
                    Passed = true,
                    ComparedTokens = 24,
                    MatchedTokens = 23,
                    DecodeSteps = 23,
                    DecodeTokensPerSecond = 120.0,
                    ElapsedSeconds = 0.85
                }
            ]
        };

        string json = baseline.ToJson();
        var parsed = GoldenBaselineFile.Parse(json);

        Assert.Equal(2, parsed.Schema);
        Assert.Equal("TestHost", parsed.Host);
        Assert.Equal("Microsoft Windows 11 Pro", parsed.OsDescription);
        Assert.Equal("X64", parsed.ProcessArchitecture);
        Assert.Equal(16, parsed.ProcessorCount);
        Assert.Equal(".NET 10.0.0", parsed.RuntimeDescription);
        Assert.Equal("2026-10-09T12:00:00Z", parsed.TimestampUtc);
        Assert.Equal("1.0.7", parsed.StingrayVersion);
        Assert.Equal(2, parsed.Entries.Count);

        var first = parsed.Entries[0];
        Assert.Equal("smollm3", first.Architecture);
        Assert.Equal("smollm3.golden.json", first.GoldenFile);
        Assert.Equal("1122334455667788", first.GoldenSha256);
        Assert.Equal("SmolLM3-Q4_K_M.gguf", first.ModelFile);
        Assert.Equal("abcdef123456", first.ModelSha256);
        Assert.Equal("Verified", first.PinStatus);
        Assert.Equal("Exact", first.Verdict);
        Assert.True(first.Passed);
        Assert.Equal(24, first.ComparedTokens);
        Assert.Equal(24, first.MatchedTokens);
        Assert.Equal(23, first.DecodeSteps);
        Assert.Equal(85.5, first.DecodeTokensPerSecond);
        Assert.Equal(450.0, first.PrefillTokensPerSecond);
        Assert.Equal(1.25, first.ElapsedSeconds);
        Assert.Equal(0.42f, first.StepwiseMaxAbsDiff);

        var second = parsed.Entries[1];
        Assert.Equal("gptneox", second.Architecture);
        Assert.Equal("NearTie", second.Verdict);
        Assert.True(second.Passed);
    }

    [Fact]
    public void BaselineFile_SupportedAndUnsupportedSchemas()
    {
        string schema1 = """{ "schema": 1, "host": "test", "entries": [] }""";
        var parsed1 = GoldenBaselineFile.Parse(schema1);
        Assert.Equal(1, parsed1.Schema);

        string schema2 = """{ "schema": 2, "host": "test", "entries": [] }""";
        var parsed2 = GoldenBaselineFile.Parse(schema2);
        Assert.Equal(2, parsed2.Schema);

        string schema99 = """{ "schema": 999, "host": "test", "entries": [] }""";
        Assert.Throws<InvalidDataException>(() => GoldenBaselineFile.Parse(schema99));
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

    [Fact]
    public void BaselineComparator_ChangedModelHash_ReportsIdentityChanged_NotSpeedRegression()
    {
        var prior = new GoldenBaselineFile
        {
            Entries =
            [
                new GoldenBaselineEntry
                {
                    Architecture = "test-arch",
                    GoldenFile = "test.golden.json",
                    ModelFile = "model.gguf",
                    ModelSha256 = "1111222233334444",
                    GoldenSha256 = "aaaa",
                    Verdict = "Exact",
                    Passed = true,
                    DecodeTokensPerSecond = 100.0
                }
            ]
        };

        var current = new GoldenBaselineFile
        {
            Entries =
            [
                new GoldenBaselineEntry
                {
                    Architecture = "test-arch",
                    GoldenFile = "test.golden.json",
                    ModelFile = "model.gguf",
                    ModelSha256 = "5555666677778888", // Different model hash
                    GoldenSha256 = "aaaa",
                    Verdict = "Exact",
                    Passed = true,
                    DecodeTokensPerSecond = 50.0 // Halved speed, but different model!
                }
            ]
        };

        var diffs = GoldenBaselineComparator.Compare(prior, current);
        Assert.Single(diffs);
        var diff = diffs[0];
        Assert.Equal(BaselineDiffStatus.IdentityChanged, diff.Status);
        Assert.Contains("Model hash changed", diff.Description);
        Assert.Contains("speed not comparable", diff.Description);
    }

    [Fact]
    public void BaselineComparator_ChangedGoldenDefinition_ReportsIdentityChanged_NotSpeedRegression()
    {
        var prior = new GoldenBaselineFile
        {
            Entries =
            [
                new GoldenBaselineEntry
                {
                    Architecture = "test-arch",
                    GoldenFile = "test.golden.json",
                    ModelFile = "model.gguf",
                    ModelSha256 = "1111222233334444",
                    GoldenSha256 = "aaaa1111",
                    Verdict = "Exact",
                    Passed = true,
                    DecodeTokensPerSecond = 100.0
                }
            ]
        };

        var current = new GoldenBaselineFile
        {
            Entries =
            [
                new GoldenBaselineEntry
                {
                    Architecture = "test-arch",
                    GoldenFile = "test.golden.json",
                    ModelFile = "model.gguf",
                    ModelSha256 = "1111222233334444",
                    GoldenSha256 = "bbbb2222", // Different golden hash (different prompt/cases)
                    Verdict = "Exact",
                    Passed = true,
                    DecodeTokensPerSecond = 40.0
                }
            ]
        };

        var diffs = GoldenBaselineComparator.Compare(prior, current);
        Assert.Single(diffs);
        var diff = diffs[0];
        Assert.Equal(BaselineDiffStatus.IdentityChanged, diff.Status);
        Assert.Contains("Golden definition changed", diff.Description);
        Assert.Contains("speed not comparable", diff.Description);
    }

    [Fact]
    public void BaselineComparator_LegacySchema1_DoesNotCompareSpeedSilently()
    {
        var prior = new GoldenBaselineFile
        {
            Schema = 1, // Legacy setup-inclusive throughput
            Entries =
            [
                new GoldenBaselineEntry
                {
                    Architecture = "test-arch",
                    GoldenFile = "test.golden.json",
                    Verdict = "Exact",
                    Passed = true,
                    DecodeTokensPerSecond = 50.0
                }
            ]
        };

        var current = new GoldenBaselineFile
        {
            Schema = 2, // New decode-only throughput
            Entries =
            [
                new GoldenBaselineEntry
                {
                    Architecture = "test-arch",
                    GoldenFile = "test.golden.json",
                    Verdict = "Exact",
                    Passed = true,
                    DecodeTokensPerSecond = 100.0
                }
            ]
        };

        var diffs = GoldenBaselineComparator.Compare(prior, current);
        Assert.Single(diffs);
        Assert.Equal(BaselineDiffStatus.Unchanged, diffs[0].Status);
        Assert.Contains("Historical baseline (schema 1)", diffs[0].Description);
    }

    [Fact]
    public void BaselineComparator_DuplicateEntries_ThrowsDescriptiveError()
    {
        var duplicatePrior = new GoldenBaselineFile
        {
            Entries =
            [
                new GoldenBaselineEntry { GoldenFile = "dup.golden.json", Architecture = "archA" },
                new GoldenBaselineEntry { GoldenFile = "dup.golden.json", Architecture = "archA" }
            ]
        };
        var normalCurrent = new GoldenBaselineFile
        {
            Entries = [new GoldenBaselineEntry { GoldenFile = "other.golden.json", Architecture = "archB" }]
        };

        var ex = Assert.Throws<InvalidDataException>(() => GoldenBaselineComparator.Compare(duplicatePrior, normalCurrent));
        Assert.Contains("duplicate entry for 'dup.golden.json'", ex.Message);
    }

    [Fact]
    public void BaselineComparator_ChangedIdentity_WithVerdictTransition_ReportsIdentityChanged_NotRegressionOrImprovement()
    {
        var prior = new GoldenBaselineFile
        {
            Entries =
            [
                new GoldenBaselineEntry
                {
                    Architecture = "test-arch",
                    GoldenFile = "test.golden.json",
                    GoldenSha256 = "aaaa1111",
                    Verdict = "Exact",
                    Passed = true,
                    DecodeTokensPerSecond = 100.0
                }
            ]
        };

        var current = new GoldenBaselineFile
        {
            Entries =
            [
                new GoldenBaselineEntry
                {
                    Architecture = "test-arch",
                    GoldenFile = "test.golden.json",
                    GoldenSha256 = "bbbb2222", // Golden definition changed
                    Verdict = "Diverged",
                    Passed = false,
                    DecodeTokensPerSecond = 100.0
                }
            ]
        };

        var diffs = GoldenBaselineComparator.Compare(prior, current);
        Assert.Single(diffs);
        var diff = diffs[0];
        // Must be classified as IdentityChanged, NOT Regression
        Assert.Equal(BaselineDiffStatus.IdentityChanged, diff.Status);
        Assert.Contains("Golden definition changed", diff.Description);
        Assert.Contains("not comparable as same-evidence regression", diff.Description);
    }

    [Fact]
    public void BaselineComparator_HostOrRuntimeMismatch_AnnotatesSpeedChanges()
    {
        var prior = new GoldenBaselineFile
        {
            Host = "HOST-ALPHA",
            RuntimeDescription = ".NET 10.0.1",
            OsDescription = "Windows",
            ProcessArchitecture = "X64",
            ProcessorCount = 16,
            Entries =
            [
                new GoldenBaselineEntry
                {
                    Architecture = "test-arch",
                    GoldenFile = "test.golden.json",
                    Verdict = "Exact",
                    Passed = true,
                    DecodeTokensPerSecond = 100.0
                }
            ]
        };

        var current = new GoldenBaselineFile
        {
            Host = "HOST-BETA", // Different machine
            RuntimeDescription = ".NET 10.0.1",
            OsDescription = "Windows",
            ProcessArchitecture = "X64",
            ProcessorCount = 16,
            Entries =
            [
                new GoldenBaselineEntry
                {
                    Architecture = "test-arch",
                    GoldenFile = "test.golden.json",
                    Verdict = "Exact",
                    Passed = true,
                    DecodeTokensPerSecond = 50.0 // 50% speed drop
                }
            ]
        };

        var diffs = GoldenBaselineComparator.Compare(prior, current);
        Assert.Single(diffs);
        var diff = diffs[0];
        Assert.Equal(BaselineDiffStatus.SpeedDrop, diff.Status);
        Assert.Contains("(different environment/host)", diff.Description);
    }
}

