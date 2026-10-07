#nullable enable

namespace OpenTail.Stingray.Engine.Planning;

/// <summary>
/// The single sanctioned bridge from inherited <c>STINGRAY_*</c> environment variables to an <see cref="ExecutionRequest"/>.
/// Explicit request values win; only unspecified (null) fields are filled. Called by frontends before planning, so the
/// environment is input to the request and never an input to the runtime.
/// </summary>
public static class ExecutionRequestEnvironment
{
    public static ExecutionRequest ApplyTo(ExecutionRequest r)
    {
        var tuning = EngineEnvironment.ReadTuning();
        var moe = EngineEnvironment.ReadMoe();
        var snap = tuning.SnapKv;
        return r with
        {
            PinnedKvDtype = r.PinnedKvDtype ?? tuning.Kv.Dtype,
            KvStore = r.KvStore ?? EngineEnvironment.ReadKvStore()?.ToString(),
            KvBf16MinTokens = r.KvBf16MinTokens ?? tuning.Kv.Bf16AutoMinTokens,
            MtpEnabled = r.MtpEnabled ?? tuning.Speculation.MtpEnabled,
            BatchVerify = r.BatchVerify ?? tuning.Speculation.BatchVerify,
            SpecBatchVerify = r.SpecBatchVerify ?? tuning.Speculation.SpecBatchVerify,
            MtpDraftN = r.MtpDraftN ?? tuning.Speculation.MtpDraftN,
            MtpMinAccept = r.MtpMinAccept ?? tuning.Speculation.MtpMinAccept,
            MtpBatchMax = r.MtpBatchMax ?? tuning.Speculation.MtpBatchMax,
            MtpBatchedMoeVerify = r.MtpBatchedMoeVerify ?? tuning.Speculation.MtpBatchedMoeVerify,
            DSparkVerifyLen = r.DSparkVerifyLen ?? tuning.Speculation.DSparkVerifyLen,
            DSparkMinConfidence = r.DSparkMinConfidence ?? tuning.Speculation.DSparkMinConfidence,
            DSparkPlace = r.DSparkPlace ?? EngineEnvironment.Raw("STINGRAY_DSPARK_PLACE"),
            PrefillChunkTokens = r.PrefillChunkTokens ?? tuning.Prefill.ChunkTokens,
            PrefixSlots = r.PrefixSlots ?? tuning.Prefill.PrefixSlots,
            PrefixScratchTokens = r.PrefixScratchTokens ?? tuning.Prefill.PrefixScratchTokens,
            HybridCpuPrefill = r.HybridCpuPrefill ?? tuning.Prefill.HybridCpuPrefill,
            CudaHybridCpuPrefill = r.CudaHybridCpuPrefill ?? tuning.Prefill.CudaHybridCpuPrefill,
            GpuCpuPrefill = r.GpuCpuPrefill ?? tuning.Prefill.GpuCpuPrefill,
            HybridCpuPrefillMinTokens = r.HybridCpuPrefillMinTokens ?? tuning.Prefill.HybridCpuPrefillMinTokens,
            HybridCpuPrefillKvBudgetMb = r.HybridCpuPrefillKvBudgetMb ?? tuning.Prefill.HybridCpuPrefillKvBudgetMb,
            HybridCpuPrefillWarmExperts = r.HybridCpuPrefillWarmExperts ?? tuning.Prefill.HybridCpuPrefillWarmExperts,
            SnapKvEnabled = r.SnapKvEnabled || snap.Enabled,
            SnapKvBudget = r.SnapKvBudget > 0 ? r.SnapKvBudget : snap.Budget,
            SnapKvWindow = r.SnapKvWindow ?? snap.Window,
            SnapKvRecency = r.SnapKvRecency ?? snap.Recency,
            SnapKvBudgetExplicit = r.SnapKvBudgetExplicit || snap.IsBudgetExplicit,
            CpuMoe = r.CpuMoe ?? moe.CpuMoe,
            GpuMoePrefill = r.GpuMoePrefill ?? moe.GpuMoePrefill,
            MoeWarmPin = r.MoeWarmPin ?? moe.WarmPin,
            MoeWarmPinAfter = r.MoeWarmPinAfter ?? moe.WarmPinAfter,
            MoePredictPrefetch = r.MoePredictPrefetch ?? moe.PredictPrefetch,
            ExpertStatsPath = r.ExpertStatsPath ?? moe.ExpertStatsPath,
        };
    }
}
