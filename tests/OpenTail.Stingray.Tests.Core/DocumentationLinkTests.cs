using System.Text.RegularExpressions;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Stale pointers in engineering docs are a recurring cost: a contributor or coding agent following
/// an old path finds nothing. These checks keep relative Markdown links, the repo paths the
/// entry-point docs cite in backticks, and the DEVELOPMENT.md project map in step with the tree.
/// Paths are matched case-exactly so a Windows checkout catches what a Linux CI checkout would.
/// </summary>
public sealed class DocumentationLinkTests
{
    // Scanned for relative Markdown links. docs/done/ is a frozen archive of completed work and is
    // deliberately not policed; tools/ holds git-ignored local reference checkouts.
    private static readonly string[] s_scannedDirectories = [".github", "docs", "samples", "scripts", "src", "tests"];

    // Entry-point docs whose backticked repo paths (`docs/...`, `scripts/...`, ...) must resolve.
    private static readonly string[] s_entryPointDocs =
    [
        "CLAUDE.md",
        "DEVELOPMENT.md",
        "README.md",
        "docs/README.md",
        "docs/00-current-work.md",
    ];

    // Git-ignored locations that exist only on machines with local reference checkouts (CLAUDE.md
    // rule 8) or downloaded weights; links into them can't resolve on a clean clone.
    private static readonly string[] s_localOnlyPrefixes = ["examples/", "tools/", "models/"];

    private static readonly Regex s_markdownLink = new(@"\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+""[^""]*"")?\s*\)");
    private static readonly Regex s_codeFence = new(@"```.*?```", RegexOptions.Singleline);
    private static readonly Regex s_backtickedPath = new(@"`((?:docs|scripts|src|tests|samples)/[^`\s*<>{}]+)`");

    [Fact]
    public void RelativeMarkdownLinks_Resolve()
    {
        string root = FindRepoRoot();
        var broken = new List<string>();

        foreach (string file in EnumerateMarkdown(root))
        {
            string relativeFile = ToRepoPath(root, file);
            string text = s_codeFence.Replace(File.ReadAllText(file), "");
            foreach (Match match in s_markdownLink.Matches(text))
            {
                string target = match.Groups[1].Value;
                if (target.StartsWith('#') || Regex.IsMatch(target, "^[A-Za-z][A-Za-z0-9+.-]*:"))
                    continue;

                string path = Uri.UnescapeDataString(target.Split('#', '?')[0]);
                if (path.Length == 0)
                    continue;

                string resolved = ToRepoPath(root, Path.GetFullPath(Path.Combine(Path.GetDirectoryName(file)!, path)));
                if (IsLocalOnly(resolved))
                    continue;
                if (!ExistsExactCase(root, resolved))
                    broken.Add($"{relativeFile}: {target}");
            }
        }

        Assert.True(broken.Count == 0, "Broken relative links:\n" + string.Join('\n', broken));
    }

    [Fact]
    public void EntryPointDocs_BacktickedRepoPaths_Resolve()
    {
        string root = FindRepoRoot();
        var broken = new List<string>();

        foreach (string doc in s_entryPointDocs)
        {
            string text = File.ReadAllText(Path.Combine(root, doc));
            foreach (Match match in s_backtickedPath.Matches(text))
            {
                string path = match.Groups[1].Value.TrimEnd('.', ',', ':', ';', ')', '/');
                if (!ExistsExactCase(root, path))
                    broken.Add($"{doc}: {path}");
            }
        }

        Assert.True(broken.Count == 0, "Repo paths cited in entry-point docs that don't exist:\n" + string.Join('\n', broken));
    }

    [Fact]
    public void DevelopmentProjectMap_ListsEverySourceProject()
    {
        string root = FindRepoRoot();
        string map = File.ReadAllText(Path.Combine(root, "DEVELOPMENT.md"));

        var missing = Directory.EnumerateDirectories(Path.Combine(root, "src"))
            .Where(dir => Directory.EnumerateFiles(dir, "*.csproj").Any())
            .Select(Path.GetFileName)
            .Where(name => !map.Contains($"{name}/", StringComparison.Ordinal))
            .ToList();

        Assert.True(missing.Count == 0, "DEVELOPMENT.md project map is missing: " + string.Join(", ", missing));
    }

    private static IEnumerable<string> EnumerateMarkdown(string root)
    {
        foreach (string file in Directory.EnumerateFiles(root, "*.md"))
            yield return file;

        foreach (string dir in s_scannedDirectories)
        {
            string full = Path.Combine(root, dir);
            if (!Directory.Exists(full))
                continue;
            foreach (string file in Directory.EnumerateFiles(full, "*.md", SearchOption.AllDirectories))
            {
                string relative = ToRepoPath(root, file);
                if (relative.StartsWith("docs/done/", StringComparison.Ordinal) ||
                    relative.Contains("/bin/", StringComparison.Ordinal) ||
                    relative.Contains("/obj/", StringComparison.Ordinal))
                    continue;
                yield return file;
            }
        }
    }

    private static bool IsLocalOnly(string repoPath) =>
        s_localOnlyPrefixes.Any(prefix => repoPath.StartsWith(prefix, StringComparison.Ordinal));

    private static bool ExistsExactCase(string root, string repoPath)
    {
        if (repoPath.StartsWith("../", StringComparison.Ordinal))
            return false;

        string current = root;
        foreach (string segment in repoPath.Split('/', StringSplitOptions.RemoveEmptyEntries))
        {
            if (!Directory.Exists(current))
                return false;
            string? next = Directory.EnumerateFileSystemEntries(current)
                .FirstOrDefault(entry => string.Equals(Path.GetFileName(entry), segment, StringComparison.Ordinal));
            if (next is null)
                return false;
            current = next;
        }
        return true;
    }

    private static string ToRepoPath(string root, string fullPath) =>
        Path.GetRelativePath(root, fullPath).Replace('\\', '/');

    private static string FindRepoRoot()
    {
        string? root = AppContext.BaseDirectory;
        while (root is not null && !File.Exists(Path.Combine(root, "CLAUDE.md")))
            root = Path.GetDirectoryName(root);
        Assert.NotNull(root);
        return root!;
    }
}
