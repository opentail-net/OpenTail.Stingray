using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Core;

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
    [InlineData("qwen3")]
    public void AdmittedArchitectures_StayAdmitted(string arch) =>
        Assert.True(ModelCompatibility.IsTextGenerationArchitectureSupported(arch));

    [Fact]
    public void NotAdmittedDescriptor_IsRefused_AndExperimentalNeedsEnvVar()
    {
        var notAdmitted = new ArchitectureDescriptor
        {
            Id = "x", Status = AdmissionStatus.NotAdmitted, EvidenceDoc = "d", RefusalReason = "r",
        };
        Assert.False(notAdmitted.IsUsable());
        Assert.Equal(
            "GGUF architecture 'x' is not admitted by OpenTail.Stingray: r (status NotAdmitted; record: d).",
            notAdmitted.GetRefusalMessage("x"));
        var experimental = new ArchitectureDescriptor
        {
            Id = "x", Status = AdmissionStatus.Experimental, EvidenceDoc = "docs/test.md", RefusalReason = "reason.",
            ExperimentalEnvVar = "REGISTRY_TEST_EXPERIMENTAL_FLAG",
        };
        string? previous = Environment.GetEnvironmentVariable("REGISTRY_TEST_EXPERIMENTAL_FLAG");
        try
        {
            Environment.SetEnvironmentVariable("REGISTRY_TEST_EXPERIMENTAL_FLAG", null);
            Assert.False(experimental.IsUsable());
            Assert.Equal(
                "GGUF architecture 'x' is ported but not verified: reason. " +
                "Set REGISTRY_TEST_EXPERIMENTAL_FLAG=1 to try it; outputs are unverified (record: docs/test.md).",
                experimental.GetRefusalMessage("x"));
            Environment.SetEnvironmentVariable("REGISTRY_TEST_EXPERIMENTAL_FLAG", "1");
            Assert.True(experimental.IsUsable());
        }
        finally
        {
            Environment.SetEnvironmentVariable("REGISTRY_TEST_EXPERIMENTAL_FLAG", previous);
        }
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
    public void BackendLimitedDescriptors_HaveLimitation_AndSelectorFallsBackForUnsupportedBackend()
    {
        var descriptors = new[]
        {
            (Architecture: "rwkv6", Family: ForwardPassFamily.Rwkv, Backends: SupportedBackends.Cpu),
            (Architecture: "rwkv7", Family: ForwardPassFamily.Rwkv, Backends: SupportedBackends.Cpu),
            (Architecture: "gpt-oss", Family: ForwardPassFamily.GptOss, Backends: SupportedBackends.Cpu | SupportedBackends.Vulkan),
            (Architecture: "deepseek2", Family: ForwardPassFamily.DeepSeek2Mla, Backends: SupportedBackends.Cpu | SupportedBackends.Vulkan),
            (Architecture: "muse-glimmer", Family: ForwardPassFamily.Dense, Backends: SupportedBackends.Cpu),
        };

        foreach (var (architecture, family, backends) in descriptors)
        {
            var descriptor = ArchitectureRegistry.Find(architecture);
            Assert.NotNull(descriptor);
            Assert.False(string.IsNullOrWhiteSpace(descriptor!.BackendLimitation), architecture);
            Assert.Equal(family, descriptor.ForwardPassFamily);
            Assert.Equal(backends, descriptor.SupportedBackends);
        }

        foreach (var (architecture, backend) in new[]
        {
            ("deepseek2", ForwardPassBackend.Cuda),
            ("muse-glimmer", ForwardPassBackend.Vulkan),
        })
        {
            var descriptor = ArchitectureRegistry.Find(architecture)!;
            var decision = ForwardPassSelection.Select(new ForwardPassRequest
            {
                
                Architecture = architecture,
                Backend = backend,
                GpuLayers = -1,
                PlannedGpuLayers = 1,
                KvLoraRank = architecture == "deepseek2" ? 1 : 0,
                HasMlaTensors = architecture == "deepseek2",
            });

            Assert.Equal(ForwardPassKind.CpuDense, decision.Kind);
            Assert.Null(decision.Refusal);
        }

        Assert.Equal(ForwardPassKind.RwkvCpu, ForwardPassSelection.Select(new ForwardPassRequest
        {
            Architecture = "rwkv7", Backend = ForwardPassBackend.Cuda, GpuLayers = -1,
        }).Kind);
        Assert.Equal(ForwardPassKind.GptOssCpu, ForwardPassSelection.Select(new ForwardPassRequest
        {
            Architecture = "gpt-oss", Backend = ForwardPassBackend.Cuda, GpuLayers = -1,
        }).Kind);
        Assert.Equal(ForwardPassFamily.Dense, ArchitectureRegistry.Find("deepseek2-ocr")!.ForwardPassFamily);
        Assert.Equal(SupportedBackends.All, ArchitectureRegistry.Find("deepseek2-ocr")!.SupportedBackends);
    }

    [Fact]
    public void NotAdmittedDescriptors_AreNeverSupported()
    {
        foreach (var d in ArchitectureRegistry.All.Where(d => d.Status == AdmissionStatus.NotAdmitted))
            Assert.False(ModelCompatibility.IsTextGenerationArchitectureSupported(d.Id));
    }

    // AdmittedSet_IsExactlyTheSnapshot lives in ArchitectureDescriptorContractTests only
    // (was duplicated verbatim in both files; docs/2-coverage/architecture-admission-followups.md #11).

    [Fact]
    public void PrefillHandoffFamilies_EveryReceiptKey_NamesARegisteredArchitecture()
    {
        // "llama+experts" is a synthetic family key (PrefillHandoffFamilies.FamilyKey): a `llama`
        // GGUF with experts is classified separately from dense llama by design, and has no
        // registered architecture of its own.
        var syntheticFamilyKeys = new HashSet<string>(StringComparer.Ordinal) { "llama+experts" };

        foreach (var (family, path) in PrefillHandoffFamilies.Receipts.Keys)
        {
            if (syntheticFamilyKeys.Contains(family)) continue;
            Assert.True(ArchitectureRegistry.Find(family) is not null,
                $"PrefillHandoffFamilies.Receipts has a key for family '{family}' (path {path}) that names no registered architecture.");
        }
    }

    [Fact]
    public void ExperimentalEnvironmentVariables_AreInKnownInventory()
    {
        var environmentVariables = ArchitectureRegistry.All
            .Where(d => d.Status == AdmissionStatus.Experimental)
            .Select(d => d.ExperimentalEnvVar!)
            .Concat(ExperimentalQuantGateRegistry.All.Select(g => g.EnvironmentVariable));

        foreach (string environmentVariable in environmentVariables)
            Assert.True(KnownEnvironmentVariables.All.Contains(environmentVariable),
                $"{environmentVariable} is missing from KnownEnvironmentVariables.All.");
    }

    // Found 2026-10-08: qwen35/qwen35moe (Qwen3.5/3.6/3.8, Ornith) pointed at CreateDense, so the plan-selected
    // CpuHybridGdn pass threw "Unsupported forward pass kind" at load for every Gated-DeltaNet checkpoint.
    [Theory]
    [InlineData("qwen35")]
    [InlineData("qwen35moe")]
    public void GatedDeltaNetFamilies_UseTheHybridGdnFactory(string arch)
    {
        var d = ArchitectureRegistry.Find(arch);
        Assert.NotNull(d);
        Assert.Equal((Func<ArchitectureLoadContext, IForwardPass>)CommonForwardPassFactory.CreateHybridGdn, d!.CreateForwardPass);
    }
}
