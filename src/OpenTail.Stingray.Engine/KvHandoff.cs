namespace OpenTail.Stingray.Engine;

/// <summary>
/// Backend-independent parts of the CPU-prefill KV handoff shared by <see cref="HybridForwardPass"/> (Vulkan) and
/// <see cref="CudaHybridForwardPass"/> (see <c>docs/2-coverage/2026-10-03-batched-moe-prefill-plan.md</c>, Phase 1): which
/// model families are admitted, how big the temporary CPU KV cache is, and moving rows between the CPU pass's
/// <see cref="PagedKvCache"/> and the hybrids' row-major FP32 layouts. Backend code only uploads and downloads.
/// </summary>
internal static unsafe class KvHandoff
{
    /// <summary>
    /// Families admitted by default: those with a hybrid-versus-CPU logit parity test (Vulkan:
    /// <c>VulkanHybridOlmoeParityTests</c> for OLMoE, Qwen3-Coder-30B and Qwen3-0.6B). A setting of <c>all</c> lifts the
    /// restriction for experiments; the default has to stay evidence-based.
    /// </summary>
    private static readonly HashSet<string> s_families = new(StringComparer.Ordinal) { "olmoe", "qwen3moe", "qwen3" };

    /// <summary>Null when the model's family is admitted, otherwise the refusal reason.</summary>
    internal static string? FamilyRefusal(GgufModel model, string? setting)
    {
        if (setting == "all") return null;
        string arch = model.Metadata.TryGetValue("general.architecture", out var a) ? Convert.ToString(a) ?? "" : "";
        return s_families.Contains(arch) ? null : $"architecture '{arch}' has no hybrid-versus-CPU parity test";
    }

    /// <summary>Bytes of the temporary F32 K+V cache the CPU pass allocates for an <paramref name="n"/>-token prompt.</summary>
    internal static long TemporaryKvBytes(int n, int numKvHeads, int headDim, int numLayers) =>
        (long)n * numKvHeads * headDim * 2 * sizeof(float) * numLayers;

    /// <summary>
    /// Extracts K and V rows [0, n) of <paramref name="layer"/> from <paramref name="source"/> into row-major
    /// <c>[n, kvDim]</c> arrays. K rows are contiguous in the paged cache; V is stored transposed per head there, so each
    /// V row is gathered head by head.
    /// </summary>
    internal static void ExtractRows(PagedKvCache source, int layer, int n, int numKvHeads, int headDim, float[] k, float[] v)
    {
        int kvDim = numKvHeads * headDim;
        for (int pos = 0; pos < n; pos++)
        {
            new ReadOnlySpan<float>(source.KeyAt(layer, pos), kvDim).CopyTo(k.AsSpan(pos * kvDim, kvDim));
            for (int h = 0; h < numKvHeads; h++)
                new ReadOnlySpan<float>(source.ValueAtHead(layer, pos, h), headDim)
                    .CopyTo(v.AsSpan(pos * kvDim + h * headDim, headDim));
        }
    }

    /// <summary>Copies row-major <c>[n, kvDim]</c> arrays into one layer of a hybrid's own <see cref="KvCache"/>.</summary>
    internal static void StoreRows(KvCache dst, int cpuLayer, int n, int kvDim, float[] k, float[] v)
    {
        for (int pos = 0; pos < n; pos++)
        {
            new ReadOnlySpan<float>(k, pos * kvDim, kvDim).CopyTo(new Span<float>(dst.KeyAt(cpuLayer, pos), kvDim));
            new ReadOnlySpan<float>(v, pos * kvDim, kvDim).CopyTo(new Span<float>(dst.ValueAt(cpuLayer, pos), kvDim));
        }
    }

    /// <summary>Reads K and V rows [0, n) of one layer of a hybrid's <see cref="KvCache"/> into row-major arrays (diagnostics).</summary>
    internal static void LoadRows(KvCache src, int cpuLayer, int n, int kvDim, float[] k, float[] v)
    {
        for (int pos = 0; pos < n; pos++)
        {
            new ReadOnlySpan<float>(src.KeyAt(cpuLayer, pos), kvDim).CopyTo(k.AsSpan(pos * kvDim, kvDim));
            new ReadOnlySpan<float>(src.ValueAt(cpuLayer, pos), kvDim).CopyTo(v.AsSpan(pos * kvDim, kvDim));
        }
    }
}
