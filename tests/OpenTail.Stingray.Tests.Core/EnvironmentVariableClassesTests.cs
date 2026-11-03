using System.Text.RegularExpressions;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// The class of each setting lives in code (<see cref="EnvironmentVariableClasses"/>) so the CLI can print it
/// without reading a doc at runtime. The inventory table is still the source, so this test fails whenever the two
/// disagree: a changed class, a missing class, or an extra one.
/// </summary>
public sealed partial class KnownEnvironmentVariablesTests
{
    [GeneratedRegex(@"^\|\s*`(STINGRAY_[A-Z0-9_]+)`\s*\|\s*([^|]+?)\s*\|", RegexOptions.Multiline)]
    private static partial Regex InventoryClassRow();

    [Fact]
    public void ClassMap_MatchesTheInventoryTable_ForEveryName()
    {
        string? root = FindRepoRoot();
        Assert.SkipUnless(root is not null, "repository layout not found (packaged run)");
        string path = Path.Combine(root!, "docs", "reference", "env-var-inventory.md");
        Assert.SkipUnless(File.Exists(path), "env-var-inventory.md not present");

        var fromTable = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (Match m in InventoryClassRow().Matches(File.ReadAllText(path)))
            fromTable[m.Groups[1].Value] = m.Groups[2].Value;

        var mismatched = new List<string>();
        foreach ((string name, string cls) in fromTable.OrderBy(kv => kv.Key, StringComparer.Ordinal))
        {
            string? inCode = EnvironmentVariableClasses.ClassOf(name);
            if (inCode != cls) mismatched.Add($"{name}: table says '{cls}', code says '{inCode ?? "(none)"}'");
        }
        Assert.True(mismatched.Count == 0,
            "class map disagrees with docs/reference/env-var-inventory.md (regenerate from the table): " + string.Join("; ", mismatched));

        var extra = EnvironmentVariableClasses.All.Keys.Where(k => !fromTable.ContainsKey(k)).OrderBy(k => k, StringComparer.Ordinal).ToList();
        Assert.True(extra.Count == 0, "class map names with no row in the inventory table: " + string.Join(", ", extra));
    }

    [Fact]
    public void ClassMap_CoversTheRegistry_Exactly()
    {
        var registered = KnownEnvironmentVariables.All.ToHashSet(StringComparer.Ordinal);
        var classed = EnvironmentVariableClasses.All.Keys.ToHashSet(StringComparer.Ordinal);

        Assert.True(registered.SetEquals(classed),
            "registry and class map differ. Only in registry: "
            + string.Join(", ", registered.Except(classed).Order(StringComparer.Ordinal))
            + ". Only in class map: " + string.Join(", ", classed.Except(registered).Order(StringComparer.Ordinal)));
    }
}
