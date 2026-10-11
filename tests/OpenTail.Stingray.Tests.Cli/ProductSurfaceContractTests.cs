using System.Text.RegularExpressions;
using OpenTail.Stingray.Core.Catalog;
using Xunit;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>
/// The public surface (READMEs, task guides, sample catalogue, CLI) has to agree with the code it describes. Everything here is derived from the
/// repository itself rather than from a second hand-written list: commands from the registrations in <c>Program.cs</c>, tasks and ids from
/// <see cref="ModelCatalog"/>, sample dependencies from the sample <c>.csproj</c> files. What these tests do NOT prove: that a documented command
/// does what its text says (run it), or that a snippet works at run time (see <c>scripts/verify-package-consumers.ps1</c>, which compiles the
/// marked snippets and the package-consumer samples against the packed NuGet package).
/// </summary>
public sealed class ProductSurfaceContractTests
{
    private static string RepoRoot()
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d is not null; d = d.Parent)
            if (File.Exists(Path.Combine(d.FullName, "OpenTail.Stingray.slnx"))) return d.FullName;
        throw new InvalidOperationException("repo root (OpenTail.Stingray.slnx) not found above the test binaries");
    }

    private static string Read(string rel) => File.ReadAllText(Path.Combine(RepoRoot(), rel.Replace('/', Path.DirectorySeparatorChar)));

    /// <summary>The documents a newcomer or a .NET developer reads first.</summary>
    private static readonly string[] UserFacingDocs =
    [
        "README.md",
        "src/OpenTail.Stingray/README.md",
        "src/OpenTail.Stingray.Cli/README.md",
        "src/OpenTail.Stingray.Diffusion/README.md",
        "docs/README.md",
        "docs/WHAT-YOU-CAN-DO.md",
        "docs/MODELS.md",
        "docs/RUNNING.md",
        "docs/STATUS.md",
        "docs/TROUBLESHOOTING.md",
        "docs/guides/README.md",
        "samples/README.md",
    ];

    private static IReadOnlySet<string> RegisteredCommands()
    {
        string program = Read("src/OpenTail.Stingray.Cli/Program.cs");
        var names = Regex.Matches(program, @"AddCommand<[A-Za-z0-9_.]+>\(""([a-z0-9-]+)""\)").Select(m => m.Groups[1].Value).ToHashSet(StringComparer.Ordinal);
        Assert.True(names.Count >= 20, $"expected the CLI to register many commands; found {names.Count}. The AddCommand pattern in Program.cs may have changed.");
        return names;
    }

    [Fact]
    public void DocsCiteOnlyCommandsTheCliRegisters()
    {
        var registered = RegisteredCommands();
        var rx = new Regex(@"(?:^|[`\s])stingray ([a-z][a-z0-9-]+)", RegexOptions.Multiline);
        int cited = 0;
        var problems = new List<string>();
        foreach (string rel in UserFacingDocs)
        {
            foreach (Match m in rx.Matches(Read(rel)))
            {
                cited++;
                string word = m.Groups[1].Value;
                if (!registered.Contains(word)) problems.Add($"{rel}: 'stingray {word}'");
            }
        }
        Assert.True(cited > 20, "the command-citation scan matched almost nothing; the pattern may have rotted");
        Assert.True(problems.Count == 0, "documents cite commands the CLI does not register (or prose that looks like one):\n" + string.Join("\n", problems.Distinct()));
    }

    [Fact]
    public void StarterTasks_AreDocumentedWithTheirSetupCommand()
    {
        foreach (string task in ModelCatalog.Tasks)
        {
            foreach (string rel in new[] { "README.md", "src/OpenTail.Stingray.Cli/README.md" })
                Assert.True(Read(rel).Contains($"stingray setup {task}", StringComparison.Ordinal),
                    $"{rel} does not show 'stingray setup {task}', but ModelCatalog has a '{task}' task.");
        }
    }

    [Fact]
    public void CliReadme_ChatChoices_MatchTheCatalogue()
    {
        var m = Regex.Match(Read("src/OpenTail.Stingray.Cli/README.md"), @"stingray models chat\s+# view chat choices \(([^)]*)\)");
        Assert.True(m.Success, "the CLI README no longer lists the chat choices next to 'stingray models chat'");
        var documented = m.Groups[1].Value.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries).ToHashSet(StringComparer.Ordinal);
        var actual = ModelCatalog.Entries.Where(e => e.Task == "chat").Select(e => e.Id).ToHashSet(StringComparer.Ordinal);
        Assert.True(documented.SetEquals(actual), $"CLI README lists chat choices [{string.Join(", ", documented)}] but the catalogue has [{string.Join(", ", actual)}]");
    }

    [Fact]
    public void RelativeLinks_InUserFacingDocs_Resolve()
    {
        var rx = new Regex(@"\]\(([^)#\s]+)(?:#[^)]*)?\)");
        var broken = new List<string>();
        foreach (string rel in UserFacingDocs)
        {
            string dir = Path.GetDirectoryName(Path.Combine(RepoRoot(), rel.Replace('/', Path.DirectorySeparatorChar)))!;
            foreach (Match m in rx.Matches(Read(rel)))
            {
                string target = m.Groups[1].Value;
                if (target.StartsWith("http", StringComparison.OrdinalIgnoreCase) || target.StartsWith("mailto:", StringComparison.OrdinalIgnoreCase) || target.StartsWith('/')) continue;
                string full = Path.GetFullPath(Path.Combine(dir, target.Replace('/', Path.DirectorySeparatorChar)));
                if (!File.Exists(full) && !Directory.Exists(full)) broken.Add($"{rel} -> {target}");
            }
        }
        Assert.True(broken.Count == 0, "broken relative links:\n" + string.Join("\n", broken.Distinct()));
    }

    [Fact]
    public void PublicDocs_DoNotClaimThereIsNoNativeCodeAnywhere()
    {
        // The package depends on ONNX Runtime (native) for ONNX speech engines, and GPU paths load the vendor driver libraries. An absolute claim is false.
        var banned = new Regex(@"no native binaries|zero native|100% pure managed|100% managed|zero p/invoke|no native dependenc", RegexOptions.IgnoreCase);
        var hits = new List<string>();
        foreach (string rel in UserFacingDocs)
            foreach (Match m in banned.Matches(Read(rel)))
                hits.Add($"{rel}: '{m.Value}'");
        Assert.True(hits.Count == 0, "absolute native-code claims are back; say what Stingray does not require (Python, llama.cpp, a server), not that nothing native is involved:\n" + string.Join("\n", hits));
    }

    [Fact]
    public void ReadmeConsumerMarkers_AreFollowedByACSharpBlock()
    {
        int markers = 0;
        foreach (string rel in new[] { "README.md", "src/OpenTail.Stingray/README.md" })
        {
            var lines = Read(rel).Split('\n').Select(l => l.TrimEnd('\r')).ToArray();
            for (int i = 0; i < lines.Length; i++)
            {
                if (lines[i].Trim() != "<!-- verify:consumer -->") continue;
                markers++;
                int j = i + 1;
                while (j < lines.Length && lines[j].Trim().Length == 0) j++;
                Assert.True(j < lines.Length && lines[j].TrimStart().StartsWith("```csharp", StringComparison.Ordinal), $"{rel} line {i + 1}: a verify:consumer marker is not followed by a ```csharp block");
            }
        }
        Assert.True(markers >= 8, $"expected the READMEs to mark at least the main examples for package verification; found {markers}");
    }

    // ---- samples -------------------------------------------------------------------------------------------------------------

    private sealed record SampleRow(string Dir, string Classification);

    private static List<SampleRow> SampleTable()
    {
        var rows = new List<SampleRow>();
        foreach (string line in Read("samples/README.md").Split('\n'))
        {
            var m = Regex.Match(line, @"^\| \*\*\[[^\]]+\]\(([^)/]+)/\)\*\* \| ([^|]+?) \|");
            if (m.Success) rows.Add(new SampleRow(m.Groups[1].Value, m.Groups[2].Value.Trim()));
        }
        Assert.True(rows.Count >= 5, $"the sample catalogue table was not parsed (found {rows.Count} rows)");
        return rows;
    }

    private static string SampleProject(string dir)
    {
        string full = Path.Combine(RepoRoot(), "samples", dir);
        string[] projs = Directory.GetFiles(full, "*.csproj");
        Assert.True(projs.Length == 1, $"samples/{dir} should contain exactly one .csproj; found {projs.Length}");
        return File.ReadAllText(projs[0]);
    }

    [Fact]
    public void SampleCatalogue_ListsEverySampleDirectory()
    {
        var listed = SampleTable().Select(r => r.Dir).ToHashSet(StringComparer.Ordinal);
        var onDisk = Directory.GetDirectories(Path.Combine(RepoRoot(), "samples")).Where(d => Directory.GetFiles(d, "*.csproj").Length > 0).Select(d => Path.GetFileName(d)!).ToHashSet(StringComparer.Ordinal);
        Assert.True(listed.SetEquals(onDisk), $"samples/README.md lists [{string.Join(", ", listed.Order())}] but the directories with a project are [{string.Join(", ", onDisk.Order())}]");
    }

    [Fact]
    public void PackageConsumerSamples_HaveAPackageBuildMode_AndContributorSamplesDoNot()
    {
        foreach (var row in SampleTable())
        {
            string csproj = SampleProject(row.Dir);
            bool hasPackageMode = csproj.Contains("StingrayPackageVersion", StringComparison.Ordinal)
                && Regex.IsMatch(csproj, @"PackageReference Include=""OpenTail\.Stingray(\.Server)?""");
            if (row.Classification.Equals("Package consumer", StringComparison.Ordinal))
            {
                Assert.True(hasPackageMode, $"samples/{row.Dir} is listed as a Package consumer but its project has no package build mode (StingrayPackageVersion + PackageReference), so nothing proves it builds against the NuGet package.");
                Assert.True(csproj.Contains("ProjectReference", StringComparison.Ordinal), $"samples/{row.Dir}: the default in-repo build should still use the source project.");
            }
            else
            {
                Assert.True(row.Classification.StartsWith("Source-level contributor example", StringComparison.Ordinal), $"samples/{row.Dir}: unknown classification '{row.Classification}'.");
                Assert.False(hasPackageMode, $"samples/{row.Dir} is listed as a contributor example but has a package build mode; classify it as a Package consumer (and verify it) or remove the mode.");
            }
        }
    }

    [Fact]
    public void EverySample_HasAReadmeWithTheRequiredSections()
    {
        foreach (var row in SampleTable())
        {
            string readme = Read($"samples/{row.Dir}/README.md");
            var headings = readme.Split('\n').Where(l => l.TrimStart().StartsWith('#')).ToArray();
            foreach (string needed in new[] { "Prerequisite", "Expected output", "Known limitations", "Verification" })
                Assert.True(headings.Any(h => h.Contains(needed, StringComparison.OrdinalIgnoreCase)), $"samples/{row.Dir}/README.md has no heading containing '{needed}'.");
        }
    }
}
