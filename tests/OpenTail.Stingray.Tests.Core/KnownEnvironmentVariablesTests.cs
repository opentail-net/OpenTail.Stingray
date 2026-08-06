using System.Text.RegularExpressions;
using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Guards the <c>STINGRAY_*</c> setting surface.
///
/// <para>These variables are read ad hoc at ~141 call sites, so a misspelling is indistinguishable
/// from "not set" and the engine silently ignores the user's configuration.
/// <see cref="KnownEnvironmentVariables"/> makes that detectable — but only while the list stays in
/// sync with the source, which is what <see cref="ListMatchesSource"/> enforces.</para>
/// </summary>
public sealed partial class KnownEnvironmentVariablesTests
{
    /// <summary>
    /// Matches a C# preprocessor directive that can name a conditional-compilation symbol.
    /// <c>#else</c>/<c>#endif</c> are absent deliberately: they carry no symbol, so they never
    /// produce a false positive and skipping them would be noise.
    /// </summary>
    [GeneratedRegex(@"^\s*#\s*(if|elif|define|undef)\b")]
    private static partial Regex DirectiveLine();

    /// <summary>
    /// Walks up from the test assembly to the repo directory containing <c>src/</c>. Returns null
    /// when the layout is not as expected (e.g. a packaged run), so the source-scanning test can
    /// skip rather than fail for an unrelated reason.
    /// </summary>
    private static string? FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            string src = Path.Combine(dir.FullName, "src");
            if (Directory.Exists(src) && Directory.Exists(Path.Combine(dir.FullName, "tests")))
                return dir.FullName;
            dir = dir.Parent;
        }
        return null;
    }

    /// <summary>
    /// The generated list must equal what the source actually reads. This is the test that makes the
    /// warning trustworthy: adding a new variable without listing it would otherwise make the
    /// warning fire on a legitimate setting, training users to ignore it.
    /// </summary>
    [Fact]
    public void ListMatchesSource()
    {
        string? root = FindRepoRoot();
        Assert.SkipWhen(root is null, "repo layout not found — source scan not applicable here");

        var inSource = new SortedSet<string>(StringComparer.Ordinal);
        foreach (string file in Directory.EnumerateFiles(Path.Combine(root!, "src"), "*.cs", SearchOption.AllDirectories))
        {
            // Skip build artefacts, which contain generated copies of the same literals.
            if (file.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal) ||
                file.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}", StringComparison.Ordinal))
                continue;

            // Skip the DECLARATION itself. It lives under src/ and contains every name by
            // construction, so including it makes the "stale entry" half of this test unable to
            // ever fail — the list would always be its own justification. That bug was real: two
            // names outlived their only usage and this test still passed.
            if (Path.GetFileName(file) == "KnownEnvironmentVariables.cs")
                continue;

            foreach (string line in File.ReadLines(file))
            {
                // Skip preprocessor directives. `STINGRAY_*` is also the naming convention for
                // conditional-compilation symbols (e.g. `#if STINGRAY_TUI`, defined by
                // -p:EnableTuiChat=true), and those are build switches the engine never reads
                // from the environment. Only the directive LINE is skipped, not the block it
                // opens, so a genuine variable read inside `#if` is still counted.
                if (DirectiveLine().IsMatch(line))
                    continue;

                foreach (Match m in Regex.Matches(line, "STINGRAY_[A-Z0-9_]+"))
                    inSource.Add(m.Value);
            }
        }

        var missing = inSource.Except(KnownEnvironmentVariables.All, StringComparer.Ordinal).ToList();
        var stale = KnownEnvironmentVariables.All.Except(inSource, StringComparer.Ordinal).OrderBy(x => x, StringComparer.Ordinal).ToList();

        Assert.True(missing.Count == 0,
            $"{missing.Count} STINGRAY_* variable(s) are read in src/ but missing from " +
            $"KnownEnvironmentVariables.All, so the CLI would warn about them as unknown: " +
            string.Join(", ", missing));
        Assert.True(stale.Count == 0,
            $"{stale.Count} entr(y/ies) in KnownEnvironmentVariables.All are no longer read anywhere " +
            $"in src/ and should be removed: " + string.Join(", ", stale));
    }

    [Fact]
    public void FindUnknown_IgnoresUnrelatedAndKnownVariables()
    {
        string known = KnownEnvironmentVariables.All.First();
        var unknown = KnownEnvironmentVariables.FindUnknown([known, "PATH", "HOME", "OPENTAIL_OTHER"]);
        Assert.Empty(unknown);
    }

    [Fact]
    public void FindUnknown_ReportsPrefixedNamesTheEngineNeverReads()
    {
        var unknown = KnownEnvironmentVariables.FindUnknown(["STINGRAY_DEFINITELY_NOT_REAL", "PATH"]);
        Assert.Equal(["STINGRAY_DEFINITELY_NOT_REAL"], unknown);
    }

    [Fact]
    public void FindUnknown_IsSortedForDeterministicOutput()
    {
        var unknown = KnownEnvironmentVariables.FindUnknown(
            ["STINGRAY_ZZZ_NOT_REAL", "STINGRAY_AAA_NOT_REAL"]);
        Assert.Equal(["STINGRAY_AAA_NOT_REAL", "STINGRAY_ZZZ_NOT_REAL"], unknown);
    }

    /// <summary>A single-character slip is the case this whole mechanism exists for.</summary>
    [Fact]
    public void SuggestClosest_FindsSingleCharacterTypo()
    {
        string known = KnownEnvironmentVariables.All.First(v => v.Length > 12);
        string typo = known[..^1] + (known[^1] == 'X' ? 'Y' : 'X');
        Assert.Equal(known, KnownEnvironmentVariables.SuggestClosest(typo));
    }

    /// <summary>
    /// A name unlike anything real must suggest nothing. A confidently wrong suggestion is worse
    /// than none — it sends the user to edit a variable that was never the problem.
    /// </summary>
    [Fact]
    public void SuggestClosest_ReturnsNullWhenNothingIsClose()
    {
        Assert.Null(KnownEnvironmentVariables.SuggestClosest("STINGRAY_QWERTYUIOPASDFGHJKLZXCVBNM"));
    }

    /// <summary>
    /// Regression: every name shares the 12-character <c>STINGRAY_</c> prefix, so scoring whole
    /// names made unrelated variables look similar. A plausible-but-wrong credential variable was
    /// matched to an unrelated mode setting. Suggestions must score the suffix only.
    /// </summary>
    [Fact]
    public void SuggestClosest_IgnoresTheSharedPrefixWhenScoring()
    {
        Assert.Null(KnownEnvironmentVariables.SuggestClosest("STINGRAY_HF_TOKEN"));
    }

    /// <summary>The motivating real-world slip must still resolve.</summary>
    [Fact]
    public void SuggestClosest_StillResolvesARealisticSlip()
    {
        Assert.Equal("STINGRAY_KV_DTYPE", KnownEnvironmentVariables.SuggestClosest("STINGRAY_KV_TYPE"));
    }
}
