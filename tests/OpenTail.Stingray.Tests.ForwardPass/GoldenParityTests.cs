using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// One data-driven test for every file under <c>Goldens/</c> (docs/2-coverage/2026-10-08-golden-parity-and-admission-tooling-plan.md):
/// the recorded prompt token ids go into the model (no tokenizer involved), the greedy continuation is compared with the llama.cpp
/// reference, mismatches are classified as near-ties or divergences by logit gap, and the stepwise-decode vs single-pass-prefill
/// self-check runs. Adding a model's receipt means adding a JSON file, not a test class. A missing checkpoint skips loudly with the
/// file name wanted and every folder searched.
/// </summary>
public sealed class GoldenParityTests : HeavyTestBase
{
    private static string? RepoRoot()
    {
        string? dir = AppContext.BaseDirectory;
        while (dir is not null && !File.Exists(Path.Combine(dir, "CLAUDE.md"))) dir = Path.GetDirectoryName(dir);
        return dir;
    }

    private static string? GoldensDir() =>
        RepoRoot() is { } root ? Path.Combine(root, "tests", "OpenTail.Stingray.Tests.ForwardPass", "Goldens") : null;

    public static IEnumerable<TheoryDataRow<string>> GoldenFiles()
    {
        var dir = GoldensDir();
        var files = dir is not null && Directory.Exists(dir)
            ? Directory.EnumerateFiles(dir, "*.json").OrderBy(f => f, StringComparer.OrdinalIgnoreCase).Select(Path.GetFileName).ToList()
            : [];
        if (files.Count == 0) yield return new TheoryDataRow<string>("(no goldens checked in yet)");
        foreach (var f in files) yield return new TheoryDataRow<string>(f!);
    }

    [Theory]
    [MemberData(nameof(GoldenFiles))]
    public void Golden_MatchesTheRecordedReference(string goldenFile)
    {
        var dir = GoldensDir();
        Assert.SkipWhen(dir is null || goldenFile.StartsWith('(') || !File.Exists(Path.Combine(dir, goldenFile)), "no golden to run");

        var golden = GoldenFile.Load(Path.Combine(dir!, goldenFile));
        var path = ModelLocator.Find(golden.Model.FileName);
        Assert.SkipWhen(path is null,
            ModelLocator.DescribeMiss([golden.Model.FileName], ModelLocator.Roots())
            + (golden.Model.Source is { } src ? $" Obtain it from {src} (e.g. `stingray pull -r {src}`)." : ""));

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        using var source = new GoldenForwardPassSource(modelHandle.Model, maxContextLength: 2048);
        using var scope = new GoldenEngineSettingsScope(golden.EngineSettings);

        var result = GoldenParityRunner.Run(golden, source.Create);
        Assert.True(result.Passed, result.Format());
        Console.WriteLine(result.Format());   // visible with -verbose: near-ties are reported even when they pass
    }
}
