#nullable enable

using System.Globalization;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// The ONE place inherited <c>STINGRAY_*</c> execution settings are read. Frontends (CLI, server, ModelContext) call
/// <see cref="Planning.ExecutionRequestEnvironment.ApplyTo"/> so the environment becomes REQUEST input; the planner records the
/// result in the plan and the runtime obeys the plan. Runtime/pass code must not read these variables.
/// </summary>
public static class EngineEnvironment
{
    public static string? Raw(string name) => Environment.GetEnvironmentVariable(name);

    private static int? PositiveInt(string name) =>
        int.TryParse(Raw(name), NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) && v > 0 ? v : null;

    private static float? UnitFloat(string name) =>
        float.TryParse(Raw(name), NumberStyles.Float, CultureInfo.InvariantCulture, out float v) && v >= 0f && v <= 1f ? v : null;

    private static bool? Flag(string name, string offValue) => Raw(name) is { } s ? s != offValue : null;

    public static KvStoreMode? ReadKvStore() => Raw("STINGRAY_KV_STORE")?.Trim().ToLowerInvariant() switch
    {
        "bf16" => KvStoreMode.Bf16,
        "auto" => KvStoreMode.Auto,
        null or "" => null,
        _ => KvStoreMode.Fp32,
    };

    public static EngineTuning ReadTuning()
    {
        var kv = new KvSettings(
            string.IsNullOrWhiteSpace(Raw("STINGRAY_KV_DTYPE")) ? null : Raw("STINGRAY_KV_DTYPE")!.Trim(),
            ReadKvStore() ?? KvStoreMode.Fp32,
            PositiveInt("STINGRAY_KV_BF16_MIN_TOKENS") ?? 1024);
        var spec = new SpeculationSettings(
            Raw("STINGRAY_DISABLE_MTP") != "1",
            Raw("STINGRAY_DISABLE_BATCH_VERIFY") != "1",
            Raw("STINGRAY_SPEC_BATCH_VERIFY") != "0",
            PositiveInt("STINGRAY_MTP_DRAFT_N") ?? 0,
            UnitFloat("STINGRAY_MTP_MIN_ACCEPT") ?? 0.55f,
            int.TryParse(Raw("STINGRAY_MTP_BATCH_MAX"), NumberStyles.Integer, CultureInfo.InvariantCulture, out int bm) ? Math.Clamp(bm, 2, 8) : 4,
            Raw("STINGRAY_MTP_BATCHED_MOE_VERIFY") != "0",
            PositiveInt("STINGRAY_DSPARK_VERIFY_LEN") ?? 0,
            UnitFloat("STINGRAY_DSPARK_MIN_CONFIDENCE") ?? 0f);
        var prefill = new PrefillSettings(
            PositiveInt("STINGRAY_PREFILL_CHUNK") ?? 0,
            PositiveInt("STINGRAY_PREFIX_SLOTS") ?? 0,
            PositiveInt("STINGRAY_PREFIX_SCRATCH_TOKENS") ?? 0,
            Raw("STINGRAY_HYBRID_CPU_PREFILL"),
            Raw("STINGRAY_CUDA_HYBRID_CPU_PREFILL"),
            Raw("STINGRAY_GPU_CPU_PREFILL"),
            PositiveInt("STINGRAY_HYBRID_CPU_PREFILL_MIN_TOKENS") ?? 32,
            PositiveInt("STINGRAY_HYBRID_CPU_PREFILL_KV_BUDGET_MB") ?? 4096,
            Raw("STINGRAY_HYBRID_CPU_PREFILL_WARM") != "0");
        return new EngineTuning(kv, spec, prefill, SnapKvConfig.FromEnvironment());
    }

    private static bool? OnOff(string name) => Raw(name)?.Trim().ToLowerInvariant() switch
    {
        "0" or "false" or "off" or "no" or "disabled" => false,
        "1" or "true" or "on" or "yes" or "enabled" => true,
        _ => null,
    };

    public static MoePlan ReadMoe() => new(
        IsMoE: false,
        CpuMoe: OnOff("STINGRAY_CPU_MOE"),
        GpuMoePrefill: OnOff("STINGRAY_MOE_GPU_PREFILL"),
        WarmPin: int.TryParse(Raw("STINGRAY_MOE_WARMPIN"), NumberStyles.Integer, CultureInfo.InvariantCulture, out int wp) && wp >= 0 ? wp : null,
        WarmPinAfter: int.TryParse(Raw("STINGRAY_MOE_WARMPIN_AFTER"), NumberStyles.Integer, CultureInfo.InvariantCulture, out int wa) && wa > 0 ? wa : null,
        PredictPrefetch: OnOff("STINGRAY_MOE_PREDICT_PREFETCH"),
        ExpertStatsPath: string.IsNullOrEmpty(Raw("STINGRAY_EXPERT_STATS")) ? null : Raw("STINGRAY_EXPERT_STATS"));
}
