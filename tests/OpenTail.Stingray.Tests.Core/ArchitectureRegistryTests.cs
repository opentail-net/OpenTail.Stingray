using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// The descriptor registry: manifest integrity, and that descriptor-backed behaviour matches
/// what the old per-call-site <c>arch == "..."</c> checks did.
/// </summary>
public sealed class ArchitectureRegistryTests
{
    [Fact]
    public void EveryDescriptor_IsValid_AndEvidenceDocExists()
    {
        Assert.NotEmpty(ArchitectureRegistry.All);
        string? root = AppContext.BaseDirectory;
        while (root is not null && !File.Exists(Path.Combine(root, "CLAUDE.md")))
            root = Path.GetDirectoryName(root);
        Assert.NotNull(root);
        foreach (var d in ArchitectureRegistry.All)
            Assert.True(File.Exists(Path.Combine(root!, d.EvidenceDoc)), $"{d.Id}: missing {d.EvidenceDoc}");
    }

    [Theory]
    [InlineData("gemma4", true)]
    [InlineData("qwen3", false)]
    [InlineData("llama", false)]
    [InlineData("granite", false)]
    [InlineData("not-registered", false)]
    public void ThinkingDefaultOff_MatchesLegacyBehaviour(string arch, bool expected) =>
        Assert.Equal(expected, ArchitectureRegistry.ThinkingDefaultOff(arch));

    [Theory]
    [InlineData("llama", FallbackChatFormat.Llama3)]
    [InlineData("llama4", FallbackChatFormat.Llama4)]
    [InlineData("granite", FallbackChatFormat.Granite)]
    [InlineData("qwen2", FallbackChatFormat.ChatMl)]
    [InlineData("gemma4", FallbackChatFormat.ChatMl)]
    public void FallbackChat_MatchesLegacyBehaviour(string arch, FallbackChatFormat expected) =>
        Assert.Equal(expected, ArchitectureRegistry.FallbackChat(arch));

    [Theory]
    [InlineData("gemma4")]
    [InlineData("granite")]
    [InlineData("llama")]
    [InlineData("llama4")]
    [InlineData("qwen3")] // not migrated: still on the legacy allowlist
    public void AdmittedArchitectures_StayAdmitted(string arch) =>
        Assert.True(ModelCompatibility.IsTextGenerationArchitectureSupported(arch));

    [Fact]
    public void NotAdmittedDescriptor_OverridesLegacyAllowlist_AndExperimentalNeedsEnvVar()
    {
        var notAdmitted = new ArchitectureDescriptor
        {
            Id = "x", Status = AdmissionStatus.NotAdmitted, EvidenceDoc = "d", RefusalReason = "r",
        };
        Assert.False(notAdmitted.IsUsable());
        var experimental = new ArchitectureDescriptor
        {
            Id = "x", Status = AdmissionStatus.Experimental, EvidenceDoc = "d", RefusalReason = "r",
            ExperimentalEnvVar = "STINGRAY_TEST_REGISTRY_EXPERIMENTAL",
        };
        Assert.False(experimental.IsUsable());
        Environment.SetEnvironmentVariable("STINGRAY_TEST_REGISTRY_EXPERIMENTAL", "1");
        try { Assert.True(experimental.IsUsable()); }
        finally { Environment.SetEnvironmentVariable("STINGRAY_TEST_REGISTRY_EXPERIMENTAL", null); }
    }

    [Fact]
    public void IncompleteDescriptor_FailsValidation() =>
        Assert.Throws<InvalidOperationException>(() => new ArchitectureDescriptor
        {
            Id = "x", Status = AdmissionStatus.NotAdmitted, EvidenceDoc = "d",
        }.Validate());
}
