using OpenTail.Stingray.Engine;
using System.Text.RegularExpressions;

namespace OpenTail.Stingray.Tests.Core;

public sealed class ArchitectureDocsContractTests
{
    private static readonly string[] s_notAdvertisedDocs =
    [
        "docs/STATUS.md",
        "README.md",
        "docs/WHAT-YOU-CAN-DO.md",
        "docs/RUNNING.md",
    ];

    [Fact]
    public void AdmittedDescriptors_HaveStatusAnchorOrExemption()
    {
        string root = FindRepoRoot();
        string[] statusTableLines = File.ReadLines(Path.Combine(root, "docs/STATUS.md"))
            .Where(line => line.StartsWith('|'))
            .ToArray();

        foreach (var descriptor in ArchitectureRegistry.All.Where(d => d.Status == AdmissionStatus.Admitted))
        {
            if (descriptor.StatusAnchor is { } anchor)
            {
                Assert.Contains(statusTableLines, line =>
                    line.Contains(anchor, StringComparison.OrdinalIgnoreCase));
            }
            else
            {
                Assert.False(string.IsNullOrWhiteSpace(descriptor.StatusExemption),
                    $"{descriptor.Id} has neither a STATUS.md anchor nor an exemption.");
            }
        }
    }

    [Fact]
    public void NotAdmittedDescriptors_AppearOnlyInInternalTakeawaysTable()
    {
        string root = FindRepoRoot();
        string takeaways = File.ReadAllText(Path.Combine(root,
            "docs/2-coverage/2026-10-02-tensorsharp-takeaways-plan.md"));
        int sectionStart = takeaways.IndexOf("### Ported, not verified", StringComparison.Ordinal);
        Assert.True(sectionStart >= 0, "The internal 'Ported, not verified' table is missing.");
        string section = takeaways[sectionStart..];

        foreach (var descriptor in ArchitectureRegistry.All.Where(d => d.Status == AdmissionStatus.NotAdmitted))
        {
            Assert.Contains($"`{descriptor.Id}`", section);
            AssertNotAdvertised(root, descriptor.Id);
        }
    }

    [Fact]
    public void ExperimentalDescriptors_AreNotAdvertised()
    {
        string root = FindRepoRoot();
        foreach (var descriptor in ArchitectureRegistry.All.Where(d => d.Status == AdmissionStatus.Experimental))
            AssertNotAdvertised(root, descriptor.Id);
    }

    [Fact]
    public void EveryDescriptorEvidenceDoc_Exists()
    {
        string root = FindRepoRoot();
        foreach (var descriptor in ArchitectureRegistry.All)
            Assert.True(File.Exists(Path.Combine(root, descriptor.EvidenceDoc)),
                $"{descriptor.Id}: missing {descriptor.EvidenceDoc}");
    }

    private static void AssertNotAdvertised(string root, string architecture)
    {
        string pattern = $@"(?<![A-Za-z0-9_]){Regex.Escape(architecture)}(?![A-Za-z0-9_])";
        foreach (string relativePath in s_notAdvertisedDocs)
        {
            string text = File.ReadAllText(Path.Combine(root, relativePath));
            Assert.False(Regex.IsMatch(text, pattern, RegexOptions.IgnoreCase),
                $"{architecture} must not appear in {relativePath}.");
        }
    }

    private static string FindRepoRoot()
    {
        string? root = AppContext.BaseDirectory;
        while (root is not null && !File.Exists(Path.Combine(root, "CLAUDE.md")))
            root = Path.GetDirectoryName(root);
        Assert.NotNull(root);
        return root!;
    }
}
