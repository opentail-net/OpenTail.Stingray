namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>The structural-then-evidence gate of the CPU-prefill KV handoff, on synthetic GGUF headers.</summary>
public sealed class PrefillHandoffFamiliesTests
{
    private static Dictionary<string, object> Md(string arch, params (string key, object value)[] extra)
    {
        var d = new Dictionary<string, object> { ["general.architecture"] = arch };
        foreach (var (k, v) in extra) d[k] = v;
        return d;
    }

    [Theory]
    [InlineData("olmoe")]
    [InlineData("qwen3moe")]
    [InlineData("qwen3")]
    public void ReceiptedFamilies_AreAdmitted(string arch)
    {
        var c = PrefillHandoffFamilies.Classify(Md(arch), HandoffPath.VulkanHybrid);
        Assert.Equal(HandoffStatus.Admitted, c.Status);
        Assert.Null(PrefillHandoffFamilies.Refusal(c, null));
    }

    [Theory]
    [InlineData("llama4")]
    [InlineData("glm4moe")]
    public void ConventionalFamiliesWithoutReceipt_AreUnverified_AndOnlyAllLiftsIt(string arch)
    {
        var c = PrefillHandoffFamilies.Classify(Md(arch), HandoffPath.VulkanHybrid);
        Assert.Equal(HandoffStatus.Unverified, c.Status);
        Assert.Contains("no parity receipt for this path", PrefillHandoffFamilies.Refusal(c, null));
        Assert.Contains("no parity receipt for this path", PrefillHandoffFamilies.Refusal(c, "1"));
        Assert.Null(PrefillHandoffFamilies.Refusal(c, "all"));
    }

    [Fact]
    public void MixtralStyleLlama_IsItsOwnFamily_NotDenseLlama()
    {
        var dense = PrefillHandoffFamilies.Classify(Md("llama"), HandoffPath.VulkanHybrid);
        var moe = PrefillHandoffFamilies.Classify(Md("llama", ("llama.expert_count", 8u)), HandoffPath.VulkanHybrid);
        Assert.Equal("llama", dense.Family);
        Assert.Equal("llama+experts", moe.Family);
        Assert.Equal(HandoffStatus.Admitted, moe.Status);   // Mixtral-8x7B receipt on the Vulkan hybrid
        Assert.Equal(HandoffStatus.Unverified, PrefillHandoffFamilies.Classify(Md("llama", ("llama.expert_count", 8u)), HandoffPath.VulkanFullGpu).Status);
        Assert.Equal(HandoffStatus.Unverified, dense.Status);    // dense llama has no receipt of its own
    }

    [Theory]
    [InlineData("deepseek2", "deepseek2.attention.kv_lora_rank", 512u, "latent")]
    [InlineData("lfm2moe", "lfm2moe.shortconv.l_cache", 3u, "recurrent")]
    [InlineData("granitehybrid", "granitehybrid.ssm.conv_kernel", 4u, "recurrent")]
    [InlineData("qwen35moe", "qwen35moe.full_attention_interval", 4u, "recurrent")]
    public void StateThatIsNotKvRows_IsIncompatible_AndNothingLiftsIt(string arch, string key, object value, string reasonPart)
    {
        var c = PrefillHandoffFamilies.Classify(Md(arch, (key, value)), HandoffPath.VulkanHybrid);
        Assert.Equal(HandoffStatus.Incompatible, c.Status);
        Assert.Contains(reasonPart, c.Reason);
        Assert.NotNull(PrefillHandoffFamilies.Refusal(c, "all"));
    }

    [Fact]
    public void KvLoraRankZero_IsNotMla()
    {
        Assert.Equal(HandoffStatus.Unverified, PrefillHandoffFamilies.Classify(Md("glm4moe", ("glm4moe.attention.kv_lora_rank", 0u)), HandoffPath.VulkanHybrid).Status);
    }

    [Fact]
    public void LongRope_AndOwnForwardPass_AndMissingArchitecture_AreIncompatible()
    {
        Assert.Equal(HandoffStatus.Incompatible, PrefillHandoffFamilies.Classify(Md("phimoe"), HandoffPath.CudaHybrid, hasLongRopeTensors: true).Status);
        Assert.Equal(HandoffStatus.Admitted, PrefillHandoffFamilies.Classify(Md("phimoe"), HandoffPath.VulkanHybrid, hasLongRopeTensors: true).Status);   // the Vulkan hybrid implements LongRoPE
        Assert.Equal(HandoffStatus.Admitted, PrefillHandoffFamilies.Classify(Md("phimoe"), HandoffPath.VulkanFullGpu, hasLongRopeTensors: true).Status);
        Assert.Equal(HandoffStatus.Incompatible, PrefillHandoffFamilies.Classify(Md("gpt-oss"), HandoffPath.VulkanHybrid).Status);
        Assert.Equal(HandoffStatus.Incompatible, PrefillHandoffFamilies.Classify(new Dictionary<string, object>(), HandoffPath.VulkanHybrid).Status);
    }

    [Fact]
    public void StructuralRefusalOutranksAReceipt()
    {
        // A receipted name on a header that carries recurrent state must still be refused.
        var c = PrefillHandoffFamilies.Classify(Md("qwen3moe", ("qwen3moe.ssm.conv_kernel", 4u)), HandoffPath.VulkanHybrid);
        Assert.Equal(HandoffStatus.Incompatible, c.Status);
    }

    [Fact]
    public void EveryReceipt_NamesItsEvidenceAndFingerprint()
    {
        foreach (var ((family, path), receipt) in PrefillHandoffFamilies.Receipts)
        {
            Assert.True(receipt.Evidence.Length > 20, $"receipt for '{family}' on {path} must cite a test");
            Assert.StartsWith(family.Split('+')[0], receipt.Fingerprint);
        }
    }

    [Fact]
    public void Receipts_AreForOnePath_AndDoNotAuthorizeAnother()
    {
        // OLMoE has Vulkan hybrid and Vulkan full-GPU receipts; Qwen3-MoE only hybrid; nothing for CUDA (never run).
        Assert.Equal(HandoffStatus.Admitted, PrefillHandoffFamilies.Classify(Md("olmoe"), HandoffPath.VulkanFullGpu).Status);
        Assert.Equal(HandoffStatus.Admitted, PrefillHandoffFamilies.Classify(Md("qwen3moe"), HandoffPath.VulkanHybrid).Status);
        Assert.Equal(HandoffStatus.Unverified, PrefillHandoffFamilies.Classify(Md("qwen3moe"), HandoffPath.VulkanFullGpu).Status);
        Assert.Equal(HandoffStatus.Admitted, PrefillHandoffFamilies.Classify(Md("qwen2moe", ("qwen2moe.attention.head_count_kv", 16u), ("qwen2moe.expert_count", 60u), ("qwen2moe.expert_used_count", 4u)), HandoffPath.VulkanHybrid).Status);
        Assert.Equal(HandoffStatus.Admitted, PrefillHandoffFamilies.Classify(Md("qwen2moe", ("qwen2moe.attention.head_count_kv", 16u), ("qwen2moe.expert_count", 60u), ("qwen2moe.expert_used_count", 4u)), HandoffPath.VulkanFullGpu).Status);
        foreach (var arch in new[] { "olmoe", "qwen3moe", "qwen3" })
            Assert.Equal(HandoffStatus.Unverified, PrefillHandoffFamilies.Classify(Md(arch), HandoffPath.CudaHybrid).Status);
    }

    [Fact]
    public void Fingerprint_ListsWhatTheCheckpointExercises_AndDifferingVariantsAreReported()
    {
        var md = Md("qwen2moe", ("qwen2moe.expert_count", 60u), ("qwen2moe.expert_used_count", 4u), ("qwen2moe.expert_shared_feed_forward_length", 5632u),
            ("qwen2moe.rope.freq_base", 1e6f));
        Assert.Equal("qwen2moe|expert_count=60|expert_used_count=4|expert_shared_feed_forward_length=5632", PrefillHandoffFamilies.Fingerprint(md));

        // a header of a receipted family whose structure differs from the proven checkpoint is still admitted, but says so
        var other = Md("olmoe", ("olmoe.expert_count", 32u));
        var c = PrefillHandoffFamilies.Classify(other, HandoffPath.VulkanHybrid);
        Assert.Equal(HandoffStatus.Admitted, c.Status);
        Assert.Contains("differs from the proven one", c.Reason);
    }

    // Real headers (index only, no weights): one checkpoint for each class the gate has to tell apart. Skips visibly when absent.
    [Theory]
    [InlineData("OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf", HandoffStatus.Admitted)]
    [InlineData("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf", HandoffStatus.Admitted)]
    [InlineData("Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf", HandoffStatus.Admitted)]
    [InlineData("Nous-Hermes-2-Mixtral-8x7B-DPO.i1-Q4_K_S.gguf", HandoffStatus.Admitted)]
    [InlineData("Phi-3.5-MoE-instruct-Q3_K_M.gguf", HandoffStatus.Admitted)]
    [InlineData("DeepSeek-V2-Lite-Chat.Q2_K.gguf", HandoffStatus.Incompatible)]      // MLA
    [InlineData("LFM2-8B-A1B-Q4_K_M.gguf", HandoffStatus.Incompatible)]               // short-conv state
    [InlineData("granite-4.0-h-small-Q2_K.gguf", HandoffStatus.Incompatible)]         // Mamba-2 state
    [InlineData("gpt-oss-20b-MXFP4.gguf", HandoffStatus.Incompatible)]                // own forward pass
    public void RealHeaders_ClassifyAsExpected(string file, HandoffStatus expected)
    {
        string? path = null;
        var external = Path.Combine(@"H:\_models", file);
        if (File.Exists(external)) path = external;
        for (var dir = new DirectoryInfo(Directory.GetCurrentDirectory()); dir is not null && path is null; dir = dir.Parent)
            foreach (var sub in new[] { "models", Path.Combine("models", "_models") })
            {
                var candidate = Path.Combine(dir.FullName, sub, file);
                if (File.Exists(candidate)) { path = candidate; break; }
            }
        Assert.SkipUnless(path is not null, $"{file} is not in models/ or models/_models/");
        using var model = GgufModel.Open(path!);
        var c = PrefillHandoffFamilies.Classify(model.Metadata, HandoffPath.VulkanHybrid, model.FindTensor("rope_factors_short.weight") is not null);
        Assert.True(expected == c.Status, $"{file}: {c.Family} classified {c.Status} ({c.Reason}), expected {expected}");
        if (expected == HandoffStatus.Admitted)
            Assert.DoesNotContain("differs from the proven one", c.Reason); // the receipt was written from this very header
    }
}
