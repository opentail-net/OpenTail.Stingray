namespace OpenTail.Stingray.Tests.Cli;

// Regression coverage for the gap a 2026-11-06 review found: SelectFiles gathered whatever
// shards matched the "-NNNNN-of-NNNNN.gguf" pattern without checking every index was present,
// so a repo missing a shard produced a successful download of an unusable model.
public sealed class PullCommandShardCompletenessTests
{
    private static (string Name, long? Size) F(string name) => (name, (long?)null);

    [Fact]
    public void FindMissingShards_SingleFile_ReportsNothingMissing()
    {
        Assert.Empty(PullCommand.FindMissingShards([F("Model-Q4_K_M.gguf")]));
    }

    [Fact]
    public void FindMissingShards_CompleteSet_ReportsNothingMissing()
    {
        var selected = new List<(string Name, long? Size)>
        {
            F("Model-00001-of-00003.gguf"),
            F("Model-00002-of-00003.gguf"),
            F("Model-00003-of-00003.gguf"),
        };
        Assert.Empty(PullCommand.FindMissingShards(selected));
    }

    [Fact]
    public void FindMissingShards_OnlyOneOfThreeShardsFound_ReportsBothMissingOnes()
    {
        // Regression: the original check short-circuited to "nothing missing" whenever
        // selected.Count < 2, which let a repo missing two of its three shards slip through
        // because SelectFiles only ever found the single shard that was actually present.
        var selected = new List<(string Name, long? Size)> { F("Model-00001-of-00003.gguf") };
        Assert.Equal([2, 3], PullCommand.FindMissingShards(selected));
    }

    [Fact]
    public void FindMissingShards_MiddleShardAbsentFromTheRepo_IsReported()
    {
        var selected = new List<(string Name, long? Size)>
        {
            F("Model-00001-of-00003.gguf"),
            F("Model-00003-of-00003.gguf"),
        };
        Assert.Equal([2], PullCommand.FindMissingShards(selected));
    }

    [Fact]
    public void SelectFiles_OnAnIncompleteShardedRepo_StillSelectsThePresentShards_SoTheCallerMustCheckCompleteness()
    {
        var files = new List<(string Name, long? Size)>
        {
            F("Model-00001-of-00003.gguf"),
            F("Model-00003-of-00003.gguf"),
        };
        var selected = PullCommand.SelectFiles(files, quantHint: null);
        Assert.Equal(2, selected.Count);
        Assert.Equal([2], PullCommand.FindMissingShards(selected));
    }

    [Fact]
    public void SelectFiles_RepoHostsOnlyOneOfThreeShards_StillFlagsTheMissingTwo()
    {
        var files = new List<(string Name, long? Size)> { F("Model-00001-of-00003.gguf") };
        var selected = PullCommand.SelectFiles(files, quantHint: null);
        Assert.Single(selected);
        Assert.Equal([2, 3], PullCommand.FindMissingShards(selected));
    }
}
