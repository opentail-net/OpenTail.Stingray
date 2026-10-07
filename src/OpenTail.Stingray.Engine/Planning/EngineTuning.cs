#nullable enable

namespace OpenTail.Stingray.Engine;

/// <summary>KV page store layout (<c>STINGRAY_KV_STORE</c>): F32 pages, real BF16 pages, or narrow-when-long.</summary>
public enum KvStoreMode { Fp32 = 0, Bf16 = 1, Auto = 2 }

/// <summary>
/// KV-cache execution settings recorded in the plan. <see cref="Dtype"/> null means "unspecified": each backend keeps its
/// own default and may fall back, whereas an explicit value must be honoured or fail loudly.
/// </summary>
public sealed record KvSettings(string? Dtype = null, KvStoreMode Store = KvStoreMode.Fp32, int Bf16AutoMinTokens = 1024)
{
    /// <summary>Every KV write is rounded to BF16 precision (<c>STINGRAY_KV_DTYPE=bf16</c>) whatever the store layout.</summary>
    public bool RoundBf16 => string.Equals(Dtype, "bf16", StringComparison.OrdinalIgnoreCase);
}

/// <summary>Speculative-decoding switches that change which decode path executes.</summary>
public sealed record SpeculationSettings(
    bool MtpEnabled = true,
    bool BatchVerify = true,
    bool SpecBatchVerify = true,
    int MtpDraftN = 0,
    float MtpMinAccept = 0.55f,
    int MtpBatchMax = 4,
    bool MtpBatchedMoeVerify = true,
    int DSparkVerifyLen = 0,
    float DSparkMinConfidence = 0f);

/// <summary>Prefill / prefix-cache / hybrid CPU-prefill-handoff settings (0 = the engine's own default).</summary>
public sealed record PrefillSettings(
    int ChunkTokens = 0,
    int PrefixSlots = 0,
    int PrefixScratchTokens = 0,
    string? HybridCpuPrefill = null,
    string? CudaHybridCpuPrefill = null,
    string? GpuCpuPrefill = null,
    int HybridCpuPrefillMinTokens = 32,
    int HybridCpuPrefillKvBudgetMb = 4096,
    bool HybridCpuPrefillWarmExperts = true);

/// <summary>
/// The instance-local execution settings resolved by <see cref="Planning.ExecutionPlanner"/> (KV, speculation, prefill, SnapKV).
/// MoE choices live in <see cref="MoePlan"/>. Nothing here is read from the process environment at run time.
/// </summary>
public sealed record EngineTuning(
    KvSettings Kv,
    SpeculationSettings Speculation,
    PrefillSettings Prefill,
    SnapKvConfig SnapKv)
{
    public static EngineTuning Default { get; } = new(new KvSettings(), new SpeculationSettings(), new PrefillSettings(), default);
}

/// <summary>
/// The bundle a forward pass / engine receives at construction: everything the plan decided that the pass used to read
/// from <c>STINGRAY_*</c>. Owned by exactly one runtime instance, so two instances in one process cannot influence each other.
/// </summary>
public sealed class EngineSettings
{
    public EngineSettings(EngineTuning tuning, MoePlan? moe)
    {
        Tuning = tuning;
        Moe = moe ?? new MoePlan(false);
    }

    public EngineTuning Tuning { get; }
    public MoePlan Moe { get; }
    public KvSettings Kv => Tuning.Kv;
    public SpeculationSettings Speculation => Tuning.Speculation;
    public PrefillSettings Prefill => Tuning.Prefill;
    public SnapKvConfig SnapKv => Tuning.SnapKv;

    public static EngineSettings FromPlan(ExecutionPlan plan) => new(plan.Tuning ?? EngineTuning.Default, plan.Moe);

    /// <summary>
    /// Settings for code that constructs a pass directly (tests, benchmarks, embedding) without a plan: today's inherited
    /// <c>STINGRAY_*</c> values, resolved once into an immutable object. The plan-driven path never uses this.
    /// </summary>
    public static EngineSettings FromEnvironment() => new(EngineEnvironment.ReadTuning(), EngineEnvironment.ReadMoe());
}
