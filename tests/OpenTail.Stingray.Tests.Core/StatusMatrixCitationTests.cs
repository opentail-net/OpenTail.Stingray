using System.Text.RegularExpressions;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Rule 10: every row of the STATUS matrix is sourced. A row that cites a test class or a golden file that no longer exists is an unsourced row
/// that still looks sourced. The first run of this check (2026-11-01) found one: two vision rows named <c>KimiYoutuVisionEmbedderParityTests</c>,
/// a class that was folded into <c>LlamaMtmdVisionParityTests</c> months earlier.
/// </summary>
public sealed class StatusMatrixCitationTests
{
    private static string FindRepoRoot()
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d is not null; d = d.Parent)
            if (File.Exists(Path.Combine(d.FullName, "OpenTail.Stingray.slnx"))) return d.FullName;
        throw new InvalidOperationException("repository root (OpenTail.Stingray.slnx) not found above " + AppContext.BaseDirectory);
    }

    // A backticked identifier ending in Test or Tests, as the matrix writes them.
    private static readonly Regex s_cited = new(@"`([A-Z][A-Za-z0-9_]*Tests?)`", RegexOptions.CultureInvariant);
    private static readonly Regex s_golden = new(@"([A-Za-z0-9_.\-]+\.golden\.json)", RegexOptions.CultureInvariant);

    private static IEnumerable<string> TestSources(string root) =>
        Directory.EnumerateFiles(Path.Combine(root, "tests"), "*.cs", SearchOption.AllDirectories)
            .Where(p => !p.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal)
                     && !p.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}", StringComparison.Ordinal));

    [Fact]
    public void EveryTestClassCitedByTheStatusMatrix_Exists()
    {
        string root = FindRepoRoot();
        string status = File.ReadAllText(Path.Combine(root, "docs", "STATUS.md"));
        var cited = s_cited.Matches(status).Select(m => m.Groups[1].Value).Distinct(StringComparer.Ordinal).ToArray();
        Assert.True(cited.Length >= 20, $"expected the matrix to cite its test classes, found {cited.Length}");   // the scan sees what it is meant to see

        string[] sources = TestSources(root).Select(File.ReadAllText).ToArray();
        var missing = cited
            .Where(name => !sources.Any(s => Regex.IsMatch(s, $@"\bclass {Regex.Escape(name)}\b")))
            .Order(StringComparer.Ordinal).ToArray();

        Assert.True(missing.Length == 0, "STATUS.md cites test classes that do not exist: " + string.Join(", ", missing));
    }

    [Fact]
    public void EveryGoldenFileCitedByTheStatusMatrix_Exists()
    {
        string root = FindRepoRoot();
        string status = File.ReadAllText(Path.Combine(root, "docs", "STATUS.md"));
        var cited = s_golden.Matches(status).Select(m => m.Groups[1].Value).Distinct(StringComparer.Ordinal).ToArray();
        Assert.True(cited.Length >= 5, $"expected the matrix to cite golden files, found {cited.Length}");

        var existing = Directory.EnumerateFiles(Path.Combine(root, "tests"), "*.golden.json", SearchOption.AllDirectories)
            .Select(Path.GetFileName).ToHashSet(StringComparer.Ordinal);
        var missing = cited.Where(g => !existing.Contains(g)).Order(StringComparer.Ordinal).ToArray();

        Assert.True(missing.Length == 0, "STATUS.md cites golden files that do not exist: " + string.Join(", ", missing));
    }
}
