namespace OpenTail.Stingray.Server;

/// <summary>
/// Applies the server host's legacy <c>STINGRAY_*</c> overrides after configuration binding.
/// Keeping this in one tested resolver makes the effective precedence explicit:
/// environment values that parse successfully win over the already-bound options.
/// </summary>
public static class ServerEnvironmentOverrides
{
    public static IReadOnlyList<string> Apply(OpenTailStingrayServerOptions options,
        Func<string, string?> environment)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(environment);
        var applied = new List<string>();

        ApplyString("STINGRAY_MODEL", value => options.ModelPath = value);
        ApplyString("STINGRAY_MMPROJ", value => options.MmprojPath = value);
        ApplyInt("STINGRAY_MAX_BATCH", value => value > 0, value => options.MaxBatchSize = value);
        ApplyInt("STINGRAY_MAX_QUEUE", value => value >= 0, value => options.MaxQueuedRequests = value);
        ApplyInt("STINGRAY_MAX_CONCURRENT", value => value > 0, value => options.MaxConcurrentRequests = value);
        ApplyInt("STINGRAY_PREFILL_CHUNK", value => value >= 0, value => options.PrefillChunkTokens = value);
        ApplyLong("STINGRAY_KV_BUDGET_MB", value => value != 0, value => options.KvBudgetMb = value);
        ApplyLong("STINGRAY_PREFIX_CACHE_MB", _ => true, value => options.PrefixCacheMb = value);
        ApplyLong("STINGRAY_PREFILL_DEQUANT_MB", _ => true, value => options.PrefillDequantCacheMb = value);

        string? backend = environment("STINGRAY_BACKEND");
        if (!string.IsNullOrWhiteSpace(backend)
            && Enum.TryParse(backend, ignoreCase: true, out ServerBackend selectedBackend))
        {
            options.Backend = selectedBackend;
            applied.Add("STINGRAY_BACKEND");
        }

        ApplyInt("STINGRAY_N_GPU_LAYERS", _ => true, value => options.NGpuLayers = value);
        ApplyString("STINGRAY_KV_DTYPE", value => options.KvType = value);
        ApplyEnabled("STINGRAY_TQ", () => options.TurboQuant = true);
        ApplyString("STINGRAY_TQ_MODE", value => options.TqMode = value);
        ApplyEnabled("STINGRAY_NO_THINKING", () => options.DisableThinking = true);
        ApplyEnabled("STINGRAY_PRESERVE_THINKING", () => options.PreserveThinking = true);
        ApplyEnabled("STINGRAY_TOOL_GRAMMAR", () => options.ToolGrammar = true);
        return applied;

        void ApplyString(string name, Action<string> set)
        {
            string? value = environment(name);
            if (!string.IsNullOrWhiteSpace(value)) { set(value); applied.Add(name); }
        }
        void ApplyInt(string name, Func<int, bool> valid, Action<int> set)
        {
            if (int.TryParse(environment(name), out int value) && valid(value)) { set(value); applied.Add(name); }
        }
        void ApplyLong(string name, Func<long, bool> valid, Action<long> set)
        {
            if (long.TryParse(environment(name), out long value) && valid(value)) { set(value); applied.Add(name); }
        }
        void ApplyEnabled(string name, Action set)
        {
            string? value = environment(name);
            if (value == "1" || string.Equals(value, "true", StringComparison.OrdinalIgnoreCase)) { set(); applied.Add(name); }
        }
    }
}

/// <summary>Process-lifetime receipt of valid legacy environment overrides applied by the host.
/// Values are intentionally never retained: they can include filesystem paths or deployment data.</summary>
public sealed class ServerEnvironmentOverrideReceipt
{
    private string[] _names = [];

    public IReadOnlyList<string> Names => Volatile.Read(ref _names);

    public void Record(IEnumerable<string> names) =>
        Volatile.Write(ref _names, [.. names.Order(StringComparer.Ordinal)]);
}
