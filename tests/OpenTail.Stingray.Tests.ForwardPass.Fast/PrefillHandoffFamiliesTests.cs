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
        var c = PrefillHandoffFamilies.Classify(Md(arch));
        Assert.Equal(HandoffStatus.Admitted, c.Status);
        Assert.Null(PrefillHandoffFamilies.Refusal(c, null));
    }

    [Theory]
    [InlineData("qwen2moe")]
    [InlineData("llama4")]
    [InlineData("glm4moe")]
    [InlineData("phimoe")]
    public void ConventionalFamiliesWithoutReceipt_AreUnverified_AndOnlyAllLiftsIt(string arch)
    {
        var c = PrefillHandoffFamilies.Classify(Md(arch));
        Assert.Equal(HandoffStatus.Unverified, c.Status);
        Assert.Contains("no hybrid-versus-CPU parity receipt", PrefillHandoffFamilies.Refusal(c, null));
        Assert.Contains("no hybrid-versus-CPU parity receipt", PrefillHandoffFamilies.Refusal(c, "1"));
        Assert.Null(PrefillHandoffFamilies.Refusal(c, "all"));
    }

    [Fact]
    public void MixtralStyleLlama_IsItsOwnFamily_NotDenseLlama()
    {
        var dense = PrefillHandoffFamilies.Classify(Md("llama"));
        var moe = PrefillHandoffFamilies.Classify(Md("llama", ("llama.expert_count", 8u)));
        Assert.Equal("llama", dense.Family);
        Assert.Equal("llama+experts", moe.Family);
        Assert.Equal(HandoffStatus.Unverified, moe.Status);
    }

    [Theory]
    [InlineData("deepseek2", "deepseek2.attention.kv_lora_rank", 512u, "latent")]
    [InlineData("lfm2moe", "lfm2moe.shortconv.l_cache", 3u, "recurrent")]
    [InlineData("granitehybrid", "granitehybrid.ssm.conv_kernel", 4u, "recurrent")]
    [InlineData("qwen35moe", "qwen35moe.full_attention_interval", 4u, "recurrent")]
    public void StateThatIsNotKvRows_IsIncompatible_AndNothingLiftsIt(string arch, string key, object value, string reasonPart)
    {
        var c = PrefillHandoffFamilies.Classify(Md(arch, (key, value)));
        Assert.Equal(HandoffStatus.Incompatible, c.Status);
        Assert.Contains(reasonPart, c.Reason);
        Assert.NotNull(PrefillHandoffFamilies.Refusal(c, "all"));
    }

    [Fact]
    public void KvLoraRankZero_IsNotMla()
    {
        Assert.Equal(HandoffStatus.Unverified, PrefillHandoffFamilies.Classify(Md("glm4moe", ("glm4moe.attention.kv_lora_rank", 0u))).Status);
    }

    [Fact]
    public void LongRope_AndOwnForwardPass_AndMissingArchitecture_AreIncompatible()
    {
        Assert.Equal(HandoffStatus.Incompatible, PrefillHandoffFamilies.Classify(Md("phimoe"), hasLongRopeTensors: true).Status);
        Assert.Equal(HandoffStatus.Incompatible, PrefillHandoffFamilies.Classify(Md("gpt-oss")).Status);
        Assert.Equal(HandoffStatus.Incompatible, PrefillHandoffFamilies.Classify(new Dictionary<string, object>()).Status);
    }

    [Fact]
    public void StructuralRefusalOutranksAReceipt()
    {
        // A receipted name on a header that carries recurrent state must still be refused.
        var c = PrefillHandoffFamilies.Classify(Md("qwen3moe", ("qwen3moe.ssm.conv_kernel", 4u)));
        Assert.Equal(HandoffStatus.Incompatible, c.Status);
    }

    [Fact]
    public void EveryReceipt_NamesItsEvidence()
    {
        foreach (var (family, evidence) in PrefillHandoffFamilies.Receipts)
            Assert.True(evidence.Length > 20, $"receipt for '{family}' must cite a test");
    }

    // Real headers (index only, no weights): one checkpoint for each class the gate has to tell apart. Skips visibly when absent.
    [Theory]
    [InlineData("OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf", HandoffStatus.Admitted)]
    [InlineData("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf", HandoffStatus.Admitted)]
    [InlineData("DeepSeek-V2-Lite-Chat.Q2_K.gguf", HandoffStatus.Incompatible)]      // MLA
    [InlineData("LFM2-8B-A1B-Q4_K_M.gguf", HandoffStatus.Incompatible)]               // short-conv state
    [InlineData("granite-4.0-h-small-Q2_K.gguf", HandoffStatus.Incompatible)]         // Mamba-2 state
    [InlineData("gpt-oss-20b-MXFP4.gguf", HandoffStatus.Incompatible)]                // own forward pass
    public void RealHeaders_ClassifyAsExpected(string file, HandoffStatus expected)
    {
        string? path = null;
        for (var dir = new DirectoryInfo(Directory.GetCurrentDirectory()); dir is not null && path is null; dir = dir.Parent)
            foreach (var sub in new[] { "models", Path.Combine("models", "_models") })
            {
                var candidate = Path.Combine(dir.FullName, sub, file);
                if (File.Exists(candidate)) { path = candidate; break; }
            }
        Assert.SkipUnless(path is not null, $"{file} is not in models/ or models/_models/");
        using var model = GgufModel.Open(path!);
        var c = PrefillHandoffFamilies.Classify(model.Metadata, model.FindTensor("rope_factors_short.weight") is not null);
        Assert.True(expected == c.Status, $"{file}: {c.Family} classified {c.Status} ({c.Reason}), expected {expected}");
    }
}
