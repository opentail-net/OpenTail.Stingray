using OpenTail.Stingray.Engine.Scout;
using OpenTail.Stingray.Core.Catalog;
using Xunit;

namespace OpenTail.Stingray.Tests.Cli;

public sealed class CatalogGuardTests
{
    [Fact]
    public void EveryEntry_HasUniqueAndLowerCaseId()
    {
        var seenIds = new HashSet<string>(StringComparer.Ordinal);

        foreach (var entry in ModelCatalog.Entries)
        {
            Assert.False(string.IsNullOrWhiteSpace(entry.Id),
                $"Entry '{entry.Id}': Id must not be null or whitespace.");

            Assert.True(entry.Id == entry.Id.ToLowerInvariant(),
                $"Entry '{entry.Id}': Id must be lower-case.");

            Assert.True(seenIds.Add(entry.Id),
                $"Entry '{entry.Id}': Id is duplicated across catalog entries.");
        }
    }

    [Fact]
    public void EveryEntry_HasValidTask_FromCatalogTasks()
    {
        foreach (var entry in ModelCatalog.Entries)
        {
            Assert.True(
                ModelCatalog.Tasks.Contains(entry.Task, StringComparer.Ordinal),
                $"Entry '{entry.Id}': Task '{entry.Task}' is not one of ModelCatalog.Tasks ({string.Join(", ", ModelCatalog.Tasks)}).");
        }
    }

    [Fact]
    public void Tasks_HaveExactlyOneDefault_AndAtMostThreeAlternatives()
    {
        foreach (var task in ModelCatalog.Tasks)
        {
            var entriesForTask = ModelCatalog.Entries
                .Where(e => e.Task.Equals(task, StringComparison.OrdinalIgnoreCase))
                .ToList();

            Assert.True(entriesForTask.Count >= 1,
                $"Task '{task}' must have at least one entry (the default entry).");

            var defaultEntry = ModelCatalog.DefaultFor(task);
            Assert.NotNull(defaultEntry);

            Assert.True(
                ReferenceEquals(entriesForTask[0], defaultEntry),
                $"Task '{task}': first entry '{entriesForTask[0].Id}' must be the default entry ('{defaultEntry.Id}').");

            // Exactly 1 default + at most 3 alternatives => at most 4 entries total for that task.
            int alternativesCount = entriesForTask.Count - 1;
            Assert.True(alternativesCount <= 3,
                $"Task '{task}' has {alternativesCount} alternatives (maximum allowed is 3). First entry: '{defaultEntry.Id}'.");
        }
    }

    [Fact]
    public void EveryEntry_Files_HaveValidSha256RevisionSizeAndRepoPath()
    {
        foreach (var entry in ModelCatalog.Entries)
        {
            Assert.True(entry.Files.Count > 0,
                $"Entry '{entry.Id}': Files list must not be empty.");

            for (int i = 0; i < entry.Files.Count; i++)
            {
                var file = entry.Files[i];

                Assert.False(string.IsNullOrWhiteSpace(file.RepoPath),
                    $"Entry '{entry.Id}', file index {i}: RepoPath must not be empty.");

                Assert.True(IsValidHex(file.Sha256, 64),
                    $"Entry '{entry.Id}', file '{file.RepoPath}': Sha256 '{file.Sha256}' must be a 64-character lower-case hex string.");

                Assert.True(IsValidHex(file.Revision, 40),
                    $"Entry '{entry.Id}', file '{file.RepoPath}': Revision '{file.Revision}' must be a 40-character lower-case hex string.");

                Assert.True(file.Size > 0,
                    $"Entry '{entry.Id}', file '{file.RepoPath}': Size {file.Size} must be > 0.");
            }
        }
    }

    [Fact]
    public void EveryEntry_HasNonEmptyRunTemplate()
    {
        foreach (var entry in ModelCatalog.Entries)
        {
            Assert.False(string.IsNullOrWhiteSpace(entry.RunTemplate),
                $"Entry '{entry.Id}': RunTemplate must not be empty.");
        }
    }

    [Fact]
    public void EveryEntry_LicenceNeedsConsent_IsTrueWhenNotPermissive()
    {
        foreach (var entry in ModelCatalog.Entries)
        {
            bool startsWithPermissive =
                entry.Licence.StartsWith("MIT", StringComparison.Ordinal) ||
                entry.Licence.StartsWith("Apache-2.0", StringComparison.Ordinal) ||
                entry.Licence.StartsWith("BSD", StringComparison.Ordinal);

            if (!startsWithPermissive)
            {
                Assert.True(entry.LicenceNeedsConsent,
                    $"Entry '{entry.Id}': LicenceNeedsConsent must be true when Licence ('{entry.Licence}') does not start with MIT, Apache-2.0, or BSD.");
            }
        }
    }

    [Fact]
    public void NoTwoEntries_ShareSameRepoPathRevision_WithDifferentSha256()
    {
        // Tracks (Repo, RepoPath, Revision) -> (Sha256, EntryId)
        var seenFiles = new Dictionary<(string Repo, string RepoPath, string Revision), (string Sha256, string EntryId)>();

        foreach (var entry in ModelCatalog.Entries)
        {
            foreach (var file in entry.Files)
            {
                var key = (file.Repo, file.RepoPath, file.Revision);
                if (seenFiles.TryGetValue(key, out var existing))
                {
                    Assert.True(
                        string.Equals(existing.Sha256, file.Sha256, StringComparison.OrdinalIgnoreCase),
                        $"Entry '{entry.Id}' and entry '{existing.EntryId}' share repo '{file.Repo}', path '{file.RepoPath}', revision '{file.Revision}' but have conflicting SHA-256 ('{file.Sha256}' vs '{existing.Sha256}').");
                }
                else
                {
                    seenFiles[key] = (file.Sha256, entry.Id);
                }
            }
        }
    }

    [Fact]
    public void EveryEntry_HasNonEmptyEvidence()
    {
        foreach (var entry in ModelCatalog.Entries)
        {
            Assert.False(string.IsNullOrWhiteSpace(entry.Evidence),
                $"Entry '{entry.Id}': Evidence text must not be empty.");
        }
    }

    [Fact]
    public void ChatTask_DefaultIsQwen05B_AndNewChatEntriesResolveWithCorrectFileCounts()
    {
        var defaultChat = ModelCatalog.DefaultFor("chat");
        Assert.NotNull(defaultChat);
        Assert.Equal("qwen2.5-0.5b", defaultChat.Id);

        var entry15B = ModelCatalog.Find("qwen2.5-1.5b");
        Assert.NotNull(entry15B);
        Assert.Equal("chat", entry15B.Task, ignoreCase: true);
        Assert.Single(entry15B.Files);

        var entry7B = ModelCatalog.Find("qwen2.5-7b");
        Assert.NotNull(entry7B);
        Assert.Equal("chat", entry7B.Task, ignoreCase: true);
        Assert.Equal(2, entry7B.Files.Count);
    }

    [Fact]
    public void ShardedGgufEntries_HaveConsistentShardNaming()
    {
        foreach (var entry in ModelCatalog.Entries)
        {
            if (entry.Files.Count > 1 && entry.Files.All(f => f.FileName.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)))
            {
                for (int i = 0; i < entry.Files.Count; i++)
                {
                    var file = entry.Files[i];
                    string expectedShard = $"-{i + 1:D5}-of-{entry.Files.Count:D5}.gguf";
                    Assert.True(file.FileName.EndsWith(expectedShard, StringComparison.OrdinalIgnoreCase),
                        $"Shard {i} of '{entry.Id}' should end with '{expectedShard}', got '{file.FileName}'.");
                }
            }
        }
    }

    [Fact]
    public void OfferSmallerQualified_PicksLargestFittingSameFamilyEntry_OrNothing()
    {
        var big = ModelCatalog.Find("qwen2.5-7b")!;
        static OpenTail.Stingray.Engine.Scout.PreflightResult Res(OpenTail.Stingray.Engine.Scout.PreflightVerdict v) => new(v, "t", 1L << 30, 8L << 30, 1L << 30);
        var onlySmallestFits = (CatalogEntry e, CancellationToken _) => Res(e.Id == "qwen2.5-0.5b" ? OpenTail.Stingray.Engine.Scout.PreflightVerdict.Allowed : OpenTail.Stingray.Engine.Scout.PreflightVerdict.Blocked);
        Assert.Equal("qwen2.5-0.5b", OpenTail.Stingray.Cli.SetupFlow.OfferSmallerQualified(big, onlySmallestFits, default)?.Id);
        var nothingFits = (CatalogEntry e, CancellationToken _) => Res(OpenTail.Stingray.Engine.Scout.PreflightVerdict.Blocked);
        Assert.Null(OpenTail.Stingray.Cli.SetupFlow.OfferSmallerQualified(big, nothingFits, default));
        var unknown = (CatalogEntry e, CancellationToken _) => Res(OpenTail.Stingray.Engine.Scout.PreflightVerdict.Unknown);
        Assert.Null(OpenTail.Stingray.Cli.SetupFlow.OfferSmallerQualified(big, unknown, default));
    }

    [Fact]
    public void DocsAndReadme_OnlyCiteCatalogueIdsAndTasksThatExist()
    {
        var rx = new System.Text.RegularExpressions.Regex(@"(?:stingray setup|ResolveModelPath\(""|stingray models use [a-z]+) ?([a-z0-9][a-z0-9._-]*)", System.Text.RegularExpressions.RegexOptions.CultureInvariant);
        var known = ModelCatalog.Entries.Select(e => e.Id).Concat(ModelCatalog.Tasks).ToHashSet(StringComparer.OrdinalIgnoreCase);
        int cited = 0;
        foreach (var rel in new[] { "README.md", "docs/STATUS.md", "docs/RUNNING.md", "docs/WHAT-YOU-CAN-DO.md" })
        {
            string path = Path.Combine(RepoRoot(), rel);
            if (!File.Exists(path)) continue;
            foreach (System.Text.RegularExpressions.Match m in rx.Matches(File.ReadAllText(path)))
            {
                cited++;
                string id = m.Groups[1].Value;
                Assert.True(known.Contains(id) || id is "chat" or "speak" or "transcribe", $"{rel} cites catalogue id or task '{id}', which is not in ModelCatalog.");
            }
        }
        Assert.True(cited > 0, "expected at least one catalogue citation in README/docs; the scan pattern may have rotted");
    }

    private static string RepoRoot()
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d is not null; d = d.Parent)
            if (File.Exists(Path.Combine(d.FullName, "OpenTail.Stingray.slnx"))) return d.FullName;
        throw new InvalidOperationException("repo root (OpenTail.Stingray.slnx) not found above the test binaries");
    }

    [Fact]
    public void EveryQualification_CitesAGoldenThatPinsTheSameFile()
    {
        int checkedCount = 0;
        foreach (var entry in ModelCatalog.Entries)
        foreach (var q in entry.Checked)
        {
            string path = Path.Combine(RepoRoot(), q.Golden.Replace('/', Path.DirectorySeparatorChar));
            Assert.True(File.Exists(path), $"Entry '{entry.Id}' cites golden '{q.Golden}', which does not exist.");
            using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(path));
            var model = doc.RootElement.GetProperty("model");
            Assert.Equal(entry.MainFile.FileName, model.GetProperty("fileName").GetString());
            Assert.Equal(entry.MainFile.Sha256, model.GetProperty("sha256").GetString());
            Assert.Equal(entry.MainFile.Size, model.GetProperty("sizeBytes").GetInt64());
            if (entry.Files.Count(f => f.FileName.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)) > 1)
            {
                // A split bundle: the receipt must pin every shard, matching the catalogue name, size and hash one for one.
                var pinned = model.GetProperty("files").EnumerateArray().Select(x => $"{x.GetProperty("fileName").GetString()}|{x.GetProperty("sizeBytes").GetInt64()}|{x.GetProperty("sha256").GetString()}").ToArray();
                var expected = entry.Files.Where(f => f.FileName.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)).Select(f => $"{f.FileName}|{f.Size}|{f.Sha256}").ToArray();
                Assert.Equal(expected, pinned);
            }
            Assert.False(string.IsNullOrWhiteSpace(q.Result) || string.IsNullOrWhiteSpace(q.Engine) || string.IsNullOrWhiteSpace(q.Date));
            checkedCount++;
        }
        Assert.True(checkedCount >= 3, "expected the three Qwen2.5 chat entries to be qualified");
    }

    [Fact]
    public void Family_FallbackStaysInFamily_QualifiedOnly_AndSmallerOnly()
    {
        var big = ModelCatalog.Find("qwen2.5-7b")!;
        Assert.Equal(["qwen2.5-1.5b", "qwen2.5-0.5b"], ModelCatalog.SmallerQualifiedAlternatives(big).Select(e => e.Id));
        Assert.Empty(ModelCatalog.SmallerQualifiedAlternatives(ModelCatalog.Find("qwen2.5-0.5b")!));
        Assert.Empty(ModelCatalog.SmallerQualifiedAlternatives(big with { FamilyId = null }));
        Assert.Empty(ModelCatalog.SmallerQualifiedAlternatives(big with { FamilyId = "another-family" }));
        Assert.Empty(ModelCatalog.SmallerQualifiedAlternatives(big, backend: "vulkan"));
        foreach (var e in ModelCatalog.Entries.Where(e => e.FamilyId is not null))
            Assert.NotEmpty(e.Checked);
    }

    private static bool IsValidHex(string? s, int expectedLength)
    {
        if (s is null || s.Length != expectedLength) return false;
        foreach (char c in s)
        {
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
                return false;
        }
        return true;
    }
}
