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
