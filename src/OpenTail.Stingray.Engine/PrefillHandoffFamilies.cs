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

/// <summary>Result of <see cref="PrefillHandoffFamilies.Classify"/>: the family key, its status and the sentence behind it.</summary>
public sealed record HandoffClassification(string Family, HandoffStatus Status, string Reason);

/// <summary>
/// Which model families may use the CPU-prefill KV handoff, decided in two layers instead of by architecture name alone:
/// <list type="number">
/// <item>a <b>structural</b> check from the GGUF header: the handoff copies per-head K and V rows, so models whose state is
/// something else (MLA latent cache, recurrent / convolution state, a forward pass of their own, LongRoPE the hybrid cannot decode)
/// are <see cref="HandoffStatus.Incompatible"/> and nothing lifts that;</item>
/// <item>an <b>evidence</b> check: a family is <see cref="HandoffStatus.Admitted"/> only with a receipt in
/// <see cref="Receipts"/> (a named hybrid-versus-CPU parity test on a real checkpoint). Structurally fine families without one are
/// <see cref="HandoffStatus.Unverified"/>, and <c>STINGRAY_HYBRID_CPU_PREFILL=all</c> may run them for experiments.</item>
/// </list>
/// Adding a family is therefore one receipt line after its parity test passes, and a new incompatible class is one structural rule.
/// </summary>
public static class PrefillHandoffFamilies
{
    /// <summary>Family key to the evidence that admits it. Add a line only together with a passing parity test.</summary>
    public static readonly IReadOnlyDictionary<string, string> Receipts = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["olmoe"] = "VulkanHybridOlmoeParityTests (OLMoE-1B-7B Q4_K_M): byte-exact K/V, logits cosine 0.996-0.9999 after handoff",
        ["qwen3moe"] = "VulkanHybridOlmoeParityTests (Qwen3-Coder-30B-A3B Q4_K_M, 128 experts): same contract",
        ["qwen3"] = "HybridCpuPrefillHandoffTests / GpuCpuPrefillHandoffTests (Qwen3-0.6B, dense)",
    };

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
    /// <param name="hasLongRopeTensors">The model carries <c>rope_factors_short.weight</c> (Phi-3 / Phi-3.5 LongRoPE).</param>
    public static HandoffClassification Classify(IReadOnlyDictionary<string, object> metadata, bool hasLongRopeTensors = false)
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
        if (hasLongRopeTensors)
            return new(family, HandoffStatus.Incompatible, "LongRoPE: the Vulkan/CUDA hybrid cannot decode it, so there is nothing to hand the KV to");

        return Receipts.TryGetValue(family, out var evidence)
            ? new(family, HandoffStatus.Admitted, evidence)
            : new(family, HandoffStatus.Unverified, "conventional K/V layout, but no hybrid-versus-CPU parity receipt");
    }

    /// <summary>Null when the handoff may run; otherwise why not. <paramref name="setting"/> <c>all</c> lifts only <see cref="HandoffStatus.Unverified"/>.</summary>
    internal static string? Refusal(HandoffClassification c, string? setting) => c.Status switch
    {
        HandoffStatus.Admitted => null,
        HandoffStatus.Unverified when setting == "all" => null,
        HandoffStatus.Unverified => $"family '{c.Family}' has no hybrid-versus-CPU parity receipt (STINGRAY_HYBRID_CPU_PREFILL=all runs it unverified)",
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
