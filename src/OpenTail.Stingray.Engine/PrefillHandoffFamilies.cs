namespace OpenTail.Stingray.Engine;

/// <summary>Where a model stands for the CPU-prefill KV handoff (<c>docs/2-coverage/2026-10-03-batched-moe-prefill-plan.md</c>).</summary>
public enum HandoffStatus
{
    /// <summary>Has a hybrid-versus-CPU parity receipt; the handoff runs by default.</summary>
    Admitted,
    /// <summary>Structurally a conventional per-head-KV transformer, but nobody has run the parity test; refused unless the setting is <c>all</c>.</summary>
    Unverified,
    /// <summary>The handoff cannot be correct for it (its state is not plain K/V rows); refused regardless of the setting.</summary>
    Incompatible,
}

/// <summary>The code paths that run the handoff; each has its own KV layout and upload code, so each needs its own receipt.</summary>
public enum HandoffPath
{
    /// <summary><c>HybridForwardPass</c>: Vulkan, some layers on the GPU and the rest on the CPU (<c>-g N</c>).</summary>
    VulkanHybrid,
    /// <summary><c>GpuForwardPass</c>: Vulkan, every layer on the GPU (<c>-g -1</c>).</summary>
    VulkanFullGpu,
    /// <summary><c>CudaHybridForwardPass</c>.</summary>
    CudaHybrid,
}

/// <summary>What one receipt proves: the test that proved it and the structural fingerprint of the checkpoint it ran on.</summary>
public sealed record HandoffReceipt(string Evidence, string Fingerprint);

/// <summary>Result of <see cref="PrefillHandoffFamilies.Classify"/>: the family key, its status and the sentence behind it.</summary>
public sealed record HandoffClassification(string Family, HandoffStatus Status, string Reason);

/// <summary>
/// Which model families may use the CPU-prefill KV handoff, decided in two layers instead of by architecture name alone:
/// <list type="number">
/// <item>a <b>structural</b> check from the GGUF header: the handoff copies per-head K and V rows, so models whose state is
/// something else (MLA latent cache, recurrent / convolution state, a forward pass of their own, LongRoPE the hybrid cannot decode)
/// are <see cref="HandoffStatus.Incompatible"/> and nothing lifts that (LongRoPE only on the CUDA hybrid, the Vulkan paths implement it);</item>
/// <item>an <b>evidence</b> check: a family is <see cref="HandoffStatus.Admitted"/> only with a receipt in
/// <see cref="Receipts"/> (a named hybrid-versus-CPU parity test on a real checkpoint). Structurally fine families without one are
/// <see cref="HandoffStatus.Unverified"/>, and <c>STINGRAY_HYBRID_CPU_PREFILL=all</c> may run them for experiments.</item>
/// </list>
/// Adding a family is therefore one receipt line after its parity test passes, and a new incompatible class is one structural rule.
/// </summary>
public static class PrefillHandoffFamilies
{
    /// <summary>
    /// (family, path) to the receipt that admits it. A receipt for one path says nothing about another: the CPU-layer cache and the
    /// GPU upload are different code. Add a line only together with a passing real-weight run on that path, including a mixed
    /// CPU/GPU layer split where the path has one. The fingerprint records what the checkpoint exercised (experts, top-k,
    /// shared-expert width, KV heads), so a later reader can tell what "admitted" covers.
    /// </summary>
    public static readonly IReadOnlyDictionary<(string Family, HandoffPath Path), HandoffReceipt> Receipts =
        new Dictionary<(string, HandoffPath), HandoffReceipt>
        {
            [("olmoe", HandoffPath.VulkanHybrid)] = new(
                "HybridCpuPrefillHandoffTests (OLMoE-1B-7B Q4_K_M, 4 and 8 GPU layers): byte-exact K/V, logits cosine 0.996-0.9999 after handoff",
                "olmoe|attention.head_count_kv=16|expert_count=64|expert_used_count=8"),
            [("qwen3moe", HandoffPath.VulkanHybrid)] = new(
                "HybridCpuPrefillHandoffTests (Qwen3-Coder-30B-A3B Q4_K_M, 4 GPU layers, 16 slots): same contract",
                "qwen3moe|attention.head_count_kv=4|attention.key_length=128|expert_count=128|expert_used_count=8|expert_feed_forward_length=768|expert_shared_feed_forward_length=0"),
            [("qwen3", HandoffPath.VulkanHybrid)] = new(
                "HybridCpuPrefillHandoffTests (Qwen3-0.6B Q8_0 dense, 8 GPU layers)",
                "qwen3|attention.head_count_kv=8|attention.key_length=128"),
            [("qwen2moe", HandoffPath.VulkanHybrid)] = new(
                "Qwen2MoeGreedyParityTests (CPU vs llama-server, PPL 4.78 vs 4.81) + HybridCpuPrefillHandoffTests (Qwen1.5-MoE-A2.7B-Chat Q4_K_M, 1 and 4 GPU layers, shared expert with sigmoid gate): byte-exact K/V, logits cosine 0.998-0.9998",
                "qwen2moe|attention.head_count_kv=16|expert_count=60|expert_used_count=4"),
            [("llama+experts", HandoffPath.VulkanHybrid)] = new(
                "MixtralGreedyParityTests (CPU vs llama-server, PPL 3.2040 vs 3.1935) + HybridCpuPrefillHandoffTests (Nous-Hermes-2-Mixtral-8x7B-DPO i1-Q4_K_S, 1 and 4 GPU layers): byte-exact K/V, logits cosine 0.9998-1.0000",
                "llama|attention.head_count_kv=8|expert_count=8|expert_used_count=2"),
            [("phimoe", HandoffPath.VulkanHybrid)] = new(
                "PhiMoeGreedyParityTests (CPU vs llama-server, short and long LongRoPE factors) + HybridCpuPrefillHandoffTests (Phi-3.5-MoE-instruct Q3_K_M, 4 and 1 GPU layers at ctx 1024 = short factors and 4 GPU layers at ctx 8192 = long factors; RMSNorm + bias, LM-head bias): byte-exact K/V, logits cosine 0.9995-1.0000",
                "phimoe|attention.head_count_kv=8|expert_count=16|expert_used_count=2"),
            [("olmoe", HandoffPath.VulkanFullGpu)] = new(
                "GpuCpuPrefillHandoffTests (OLMoE-1B-7B Q4_K_M, F32 and packed-fp16 KV): byte-exact K/V, logits equal the CPU pass",
                "olmoe|attention.head_count_kv=16|expert_count=64|expert_used_count=8"),
            [("qwen2moe", HandoffPath.VulkanFullGpu)] = new(
                "GpuCpuPrefillHandoffTests (Qwen1.5-MoE-A2.7B-Chat Q4_K_M, -g -1, shared expert 5632 wide with sigmoid gate): handoff logits equal the CPU pass, 40 decode steps vs sequential GPU prefill: cosine 0.998-1.0000 except one isolated 0.9888 dip at step 34 (near-tie router, neighbours 0.998)",
                "qwen2moe|attention.head_count_kv=16|expert_count=60|expert_used_count=4"),
            [("phimoe", HandoffPath.VulkanFullGpu)] = new(
                "GpuCpuPrefillHandoffTests (Phi-3.5-MoE-instruct Q3_K_M, -g -1, ctx 1024 = short LongRoPE factors; RMSNorm + bias, LM-head bias): handoff logits equal the CPU pass, 40 decode steps vs sequential GPU prefill: cosine >= 0.988 (one dip, rest >= 0.997)",
                "phimoe|attention.head_count_kv=8|expert_count=16|expert_used_count=2"),
            // CudaHybrid: no receipts. The path compiles and has never run (no NVIDIA GPU on the development machine).
        };

    // Header keys (without the architecture prefix) that describe what a checkpoint exercises; recorded in receipts.
    private static readonly string[] s_fingerprintKeys =
    {
        "attention.head_count_kv", "attention.key_length", "expert_count", "expert_used_count", "expert_feed_forward_length",
        "expert_shared_count", "expert_shared_feed_forward_length", "leading_dense_block_count", "expert_gating_func", "expert_weights_norm",
    };

    // The fingerprint is diagnostic, not an admission gate: a header that differs from the proven checkpoint is still admitted and the
    // difference is reported in the evidence text, so checkpoint variants do not each need an allowlist entry.
    /// <summary>Deterministic structural summary of a header, in the form stored in <see cref="HandoffReceipt.Fingerprint"/>.</summary>
    public static string Fingerprint(IReadOnlyDictionary<string, object> metadata)
    {
        string arch = Str(metadata, "general.architecture");
        var sb = new System.Text.StringBuilder(arch);
        foreach (var key in s_fingerprintKeys)
            if (metadata.TryGetValue($"{arch}.{key}", out var v))
                sb.Append('|').Append(key).Append('=').Append(Convert.ToString(v, System.Globalization.CultureInfo.InvariantCulture));
        return sb.ToString();
    }

    // Architectures that run a forward pass of their own, which has no hybrid K/V layout to hand rows to.
    private static readonly HashSet<string> s_ownForwardPass = new(StringComparer.Ordinal) { "gpt-oss" };

    /// <summary>
    /// Family key: the architecture name, except that a <c>llama</c> GGUF with experts is Mixtral-style
    /// (<c>llama+experts</c>), which must not inherit anything from dense Llama.
    /// </summary>
    public static string FamilyKey(IReadOnlyDictionary<string, object> metadata)
    {
        string arch = Str(metadata, "general.architecture");
        return arch == "llama" && Int(metadata, "llama.expert_count") > 0 ? "llama+experts" : arch;
    }

    /// <param name="metadata">GGUF key/values.</param>
    /// <param name="path">Which handoff path is asking; receipts are per path.</param>
    /// <param name="hasLongRopeTensors">The model carries <c>rope_factors_short.weight</c> (Phi-3 / Phi-3.5 LongRoPE).</param>
    public static HandoffClassification Classify(IReadOnlyDictionary<string, object> metadata, HandoffPath path, bool hasLongRopeTensors = false)
    {
        string arch = Str(metadata, "general.architecture");
        string family = FamilyKey(metadata);

        if (arch.Length == 0)
            return new(family, HandoffStatus.Incompatible, "no general.architecture in the header");
        if (s_ownForwardPass.Contains(arch))
            return new(family, HandoffStatus.Incompatible, $"'{arch}' runs its own forward pass, which has no hybrid K/V layout to hand rows to");
        if (Int(metadata, $"{arch}.attention.kv_lora_rank") > 0)
            return new(family, HandoffStatus.Incompatible, "multi-head latent attention caches a compressed latent, not per-head K/V rows");
        if (metadata.Keys.Any(k => k.StartsWith($"{arch}.ssm.", StringComparison.Ordinal) || k.StartsWith($"{arch}.shortconv.", StringComparison.Ordinal))
            || metadata.ContainsKey($"{arch}.full_attention_interval"))
            return new(family, HandoffStatus.Incompatible, "recurrent or convolution layers keep state that is not in the KV cache");
        if (hasLongRopeTensors && path == HandoffPath.CudaHybrid)
            return new(family, HandoffStatus.Incompatible, "LongRoPE: the CUDA hybrid cannot decode it (the Vulkan paths can), so there is nothing to hand the KV to");

        if (Receipts.TryGetValue((family, path), out var receipt))
        {
            string fp = Fingerprint(metadata);
            string variant = fp == receipt.Fingerprint ? "" : $" (this checkpoint differs from the proven one: {fp} vs {receipt.Fingerprint})";
            return new(family, HandoffStatus.Admitted, receipt.Evidence + variant);
        }
        return new(family, HandoffStatus.Unverified, $"conventional K/V layout, but no parity receipt for this path ({path})");
    }

    /// <summary>Null when the handoff may run; otherwise why not. <paramref name="setting"/> <c>all</c> lifts only <see cref="HandoffStatus.Unverified"/>.</summary>
    internal static string? Refusal(HandoffClassification c, string? setting) => c.Status switch
    {
        HandoffStatus.Admitted => null,
        HandoffStatus.Unverified when setting == "all" => null,
        HandoffStatus.Unverified => $"family '{c.Family}' has no parity receipt for this path (the CPU-prefill setting all runs it unverified)",
        _ => $"family '{c.Family}' cannot use the handoff: {c.Reason}",
    };

    private static string Str(IReadOnlyDictionary<string, object> md, string key) =>
        md.TryGetValue(key, out var v) ? Convert.ToString(v, System.Globalization.CultureInfo.InvariantCulture) ?? "" : "";

    private static long Int(IReadOnlyDictionary<string, object> md, string key)
    {
        if (!md.TryGetValue(key, out var v)) return 0;
        try { return Convert.ToInt64(v, System.Globalization.CultureInfo.InvariantCulture); }
        catch (Exception ex) when (ex is FormatException or InvalidCastException or OverflowException) { return 0; }
    }
}
