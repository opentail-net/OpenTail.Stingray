using System.Text.RegularExpressions;

namespace OpenTail.Stingray.Tests.Core;

public sealed partial class KnownEnvironmentVariablesTests
{
    /// <summary>
    /// Registry names allowed to lack a row in <c>docs/reference/env-var-inventory.md</c>. Empty since 2026-10-31, when the table became
    /// name-complete (it had fallen to 192 rows for 287 names). Keep it empty: a new variable must arrive with its row, and adding its
    /// name here instead would be a visible step backwards in review.
    /// </summary>
    private static readonly string[] KnownUndocumented =
    [
    ];

    [GeneratedRegex(@"^\|\s*`(STINGRAY_[A-Z0-9_]+)`", RegexOptions.Multiline)]
    private static partial Regex InventoryRow();

    [Fact]
    public void InventoryTable_CoversTheRegistry_ExceptTheListedGap()
    {
        string? root = FindRepoRoot();
        Assert.SkipUnless(root is not null, "repository layout not found (packaged run)");
        string path = Path.Combine(root!, "docs", "reference", "env-var-inventory.md");
        Assert.SkipUnless(File.Exists(path), "env-var-inventory.md not present");

        var rows = InventoryRow().Matches(File.ReadAllText(path)).Select(m => m.Groups[1].Value).ToHashSet(StringComparer.Ordinal);
        var undocumented = KnownEnvironmentVariables.All.Where(n => !rows.Contains(n)).ToHashSet(StringComparer.Ordinal);

        var newlyUndocumented = undocumented.Except(KnownUndocumented, StringComparer.Ordinal).Order(StringComparer.Ordinal).ToList();
        Assert.True(newlyUndocumented.Count == 0,
            "registered but missing from docs/reference/env-var-inventory.md (add a row): " + string.Join(", ", newlyUndocumented));

        var nowDocumented = KnownUndocumented.Where(n => !undocumented.Contains(n)).ToList();
        Assert.True(nowDocumented.Count == 0,
            "now documented (or no longer registered): delete from KnownUndocumented: " + string.Join(", ", nowDocumented));

        var orphanRows = rows.Where(r => !KnownEnvironmentVariables.All.Contains(r)).Order(StringComparer.Ordinal).ToList();
        Assert.True(orphanRows.Count == 0, "inventory rows for names the registry does not contain: " + string.Join(", ", orphanRows));
    }
}
