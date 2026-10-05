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
            ExperimentalEnvVar = "REGISTRY_TEST_EXPERIMENTAL_FLAG",
        };
        Assert.False(experimental.IsUsable());
        Environment.SetEnvironmentVariable("REGISTRY_TEST_EXPERIMENTAL_FLAG", "1");
        try { Assert.True(experimental.IsUsable()); }
        finally { Environment.SetEnvironmentVariable("REGISTRY_TEST_EXPERIMENTAL_FLAG", null); }
    }

    [Fact]
    public void IncompleteDescriptor_FailsValidation() =>
        Assert.Throws<InvalidOperationException>(() => new ArchitectureDescriptor
        {
            Id = "x", Status = AdmissionStatus.NotAdmitted, EvidenceDoc = "d",
        }.Validate());

    [Theory]
    [InlineData("glm-dsa")]
    [InlineData("glm5next")]
    [InlineData("diffusion-gemma")]
    [InlineData("qwen4exp")]
    [InlineData("deepseek41")]
    [InlineData("deepseek4")]
    [InlineData("deepseek32")]
    public void PortedNotVerifiedFamilies_AreRefused_WithTheirOwnReason(string arch)
    {
        Assert.False(ModelCompatibility.IsTextGenerationArchitectureSupported(arch));
        var d = ArchitectureRegistry.Find(arch);
        Assert.NotNull(d);
        Assert.Equal(AdmissionStatus.NotAdmitted, d!.Status);
    }

    [Fact]
    public void MuseGlimmer_AdmittedUnderBothSpellings()
    {
        Assert.True(ModelCompatibility.IsTextGenerationArchitectureSupported("muse-glimmer"));
        Assert.True(ModelCompatibility.IsTextGenerationArchitectureSupported("muse_glimmer"));
    }

    [Fact]
    public void NoArchitectureIsBothMigratedAndLegacy_ExceptDocumentedOverlap()
    {
        // Migrated descriptors win over the legacy allowlist, so a NotAdmitted descriptor must never be
        // shadowed by a stale allowlist entry.
        foreach (var d in ArchitectureRegistry.All.Where(d => d.Status == AdmissionStatus.NotAdmitted))
            Assert.False(ModelCompatibility.IsTextGenerationArchitectureSupported(d.Id));
    }

    [Fact]
    public void MigratedDescriptorIds_AreNotAlsoInTheLegacyAllowlist()
    {
        // Single source of truth: once a family has a descriptor its allowlist entry must go, or the
        // two could silently disagree. Detected via reflection-free source scan of the allowlist.
        string? root = AppContext.BaseDirectory;
        while (root is not null && !File.Exists(Path.Combine(root, "CLAUDE.md")))
            root = Path.GetDirectoryName(root);
        string src = File.ReadAllText(Path.Combine(root!, "src/OpenTail.Stingray.Engine/ModelCompatibility.cs"));
        int start = src.IndexOf("s_textGenerationArchitectures = new", StringComparison.Ordinal);
        int end = src.IndexOf("};", start, StringComparison.Ordinal);
        string list = string.Join(Environment.NewLine, src[start..end]
            .Split('\n').Where(l => !l.TrimStart().StartsWith("//")));
        foreach (var d in ArchitectureRegistry.All)
            foreach (var name in d.Aliases.Prepend(d.Id))
                Assert.DoesNotContain($"\"{name}\"", list);
    }
}
