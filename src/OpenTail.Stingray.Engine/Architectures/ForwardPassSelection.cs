namespace OpenTail.Stingray.Engine;

public enum ForwardPassBackend
{
    Auto,
    Cpu,
    Cuda,
    Vulkan,
}

public enum ForwardPassKind
{
    SafeTensorsCpu,
    CpuDense,
    CpuHybridGdn,
    RwkvCpu,
    GptOssCpu,
    GptOssVulkan,
    DeepSeek2Vulkan,
    CudaDense,
    CudaHybridGdn,
    CudaHybrid,
    VulkanDense,
    VulkanHybridGdn,
    VulkanHybrid,
    VulkanLayerSplit,
}

public sealed record ForwardPassRequest
{
    public string Architecture { get; init; } = "llama";
    public string? UnsupportedBackendName { get; init; }
    public bool IsSafeTensors { get; init; }
    public bool PackageSupported { get; init; } = true;
    public bool IsHybridSsm { get; init; }
    public bool HasHybridGdnLayers { get; init; }
    public bool HasCpuHybridGdnPass { get; init; }
    public bool IsMoE { get; init; }
    public int KvLoraRank { get; init; }
    public bool HasMlaTensors { get; init; }
    public bool ArchitectureSupported { get; init; } = true;
    public bool AllowUnverifiedArchitecture { get; init; }
    public string? ArchitectureRefusal { get; init; }
    public int NumLayers { get; init; } = 1;
    public int GpuLayers { get; init; }
    public int PlannedGpuLayers { get; init; } = -1;
    public ForwardPassBackend Backend { get; init; } = ForwardPassBackend.Auto;
    public bool CudaAvailable { get; init; }
    public bool UnsupportedGpuPath { get; init; }
    public bool UnsupportedPartialCudaPath { get; init; }
    public bool UnsupportedPartialVulkanPath { get; init; }
    public bool LayerHeadDim { get; init; }
    public bool TurboQuant { get; init; }
    public bool ValidateTurboQuantModeOnly { get; init; }
    public bool ValidateKVarNOnly { get; init; }
    public bool ValidateTurboQuantHeadDimOnly { get; init; }
    public bool SkipFamilyRefusals { get; init; }
    public bool SkipTurboQuantShapeValidation { get; init; }
    public bool HasDraftModel { get; init; }
    public bool DraftLookup { get; init; }
    public bool DraftModelExists { get; init; } = true;
    public string? DraftModelPath { get; init; }
    public bool HasDSparkModel { get; init; }
    public bool DsParkRequested { get; init; }
    public bool DSparkModelPathExists { get; init; } = true;
    public bool HasDSparkConfig { get; init; } = true;
    public bool DSparkConfigExists { get; init; } = true;
    public bool DSparkHeadMatchesTarget { get; init; } = true;
    public int DSparkVocabSize { get; init; }
    public int ModelVocabSize { get; init; }
    public int DSparkTargetLayers { get; init; }
    public int ModelLayers { get; init; }
    public int DSparkHiddenSize { get; init; }
    public int ModelHiddenSize { get; init; }
    public bool HasMtpSpecType { get; init; }
    public float DSparkMinConfidence { get; init; }
    public bool DSparkTargetSupported { get; init; } = true;
    public bool SupportsHiddenTaps { get; init; } = true;
    public bool HasSinglePrompt { get; init; } = true;
    public int PromptTokenCount { get; init; }
    public int DSparkBlockSize { get; init; }
    public int TargetContextLength { get; init; }
    public int DSparkContextLength { get; init; }
    public bool DSparkPlacementOff { get; init; }
    public bool IsContinuousBatching { get; init; }
    public bool SupportsContinuousBatching { get; init; }
    public bool SessionsRequested { get; init; }
    public bool SessionsPathSupported { get; init; } = true;
    public bool BatchingSupported { get; init; } = true;
    public bool IsConcreteCpuDensePass { get; init; } = true;
    public bool HasImageInput { get; init; }
    public bool SupportsEmbeddingInput { get; init; } = true;
    public bool HasImageBatching { get; init; }
    public bool IsGemma4 { get; init; }
    public bool MaxBatchSizeGreaterThanOne { get; init; }
    public bool HasTokenConstraint { get; init; }
    public bool HasSamplingPenaltyOrBias { get; init; }
    public bool SampledSpecDisabled { get; init; }
    public bool IsSampled { get; init; }
    public string TurboQuantMode { get; init; } = "auto";
    public int HeadDim { get; init; } = 128;
    public bool HasTools { get; init; }
    public bool KVarNSnapKvBlocked { get; init; }
    public bool KVarNCudaMoeBlocked { get; init; }
    public bool KVarNPartialCudaBlocked { get; init; }
    public int PlannedGpuLayersForKVarN { get; init; } = -1;
    public bool KVarNAutoFallbackBlocked { get; init; }
    public bool LayerSplitOnly { get; init; }
    public bool HasLayerHeadDim { get; init; }
    public bool LayerSplitOnlyLargeMoe { get; init; }
    public bool AutoPlan { get; init; }
    public bool Gemma4KvShareBoundary { get; init; }
    public bool MoEAutoCacheOverflow { get; init; }
    public bool IsSafeTensorsGpuRequested { get; init; }
    public bool IsSafeTensorsDraftRequested { get; init; }
    public bool IsSafeTensorsDSparkRequested { get; init; }
}

public sealed record ForwardPassDecision(ForwardPassKind? Kind, string? Refusal, string? Notice = null)
{
    public bool IsRefused => Refusal is not null;

    public static ForwardPassDecision Select(ForwardPassKind kind, string? notice = null) => new(kind, null, notice);

    public static ForwardPassDecision Refuse(string message) => new(null, message);
}

public static class ForwardPassSelection
{
    public static ForwardPassDecision Select(ForwardPassRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);

        if (request.IsSafeTensors)
        {
            if (!request.PackageSupported)
                return ForwardPassDecision.Refuse("Model package is not supported by its SafeTensors profile.");
                if (request.IsSafeTensorsGpuRequested)
                    return ForwardPassDecision.Refuse("GPU offload is not yet supported for SafeTensors packages.");
                if (request.TurboQuant)
                    return ForwardPassDecision.Refuse("--tq (TurboQuant) is not supported for SafeTensors packages.");
                if (request.IsSafeTensorsDraftRequested)
                    return ForwardPassDecision.Refuse("Speculative decoding is not supported for SafeTensors packages.");
                if (request.IsSafeTensorsDSparkRequested)
                    return ForwardPassDecision.Refuse("DSpark is not supported for SafeTensors packages.");
            return ForwardPassDecision.Select(ForwardPassKind.SafeTensorsCpu);
        }

        if (request.ValidateTurboQuantModeOnly)
        {
            string mode = request.TurboQuantMode.Trim().ToLowerInvariant();
            return mode is "" or "auto" or "lloydmax" or "lloyd-max" or "kvarn"
                ? ForwardPassDecision.Select(ForwardPassKind.CpuDense)
                : ForwardPassDecision.Refuse($"Unknown --tq-mode value '{request.TurboQuantMode}'. Expected one of: auto, lloydmax, kvarn.");
        }

        if (!request.ArchitectureSupported && !request.AllowUnverifiedArchitecture)
            return ForwardPassDecision.Refuse(request.ArchitectureRefusal ??
                $"GGUF architecture '{request.Architecture}' is not admitted by OpenTail.Stingray.");
        if (request.UnsupportedBackendName is { } unsupportedBackend)
            return ForwardPassDecision.Refuse($"Unknown --backend value '{unsupportedBackend}'. Expected one of: auto, vulkan, cuda.");

        ArchitectureDescriptor? descriptor = ArchitectureRegistry.Find(request.Architecture);
        ForwardPassFamily family = descriptor?.ForwardPassFamily ?? ForwardPassFamily.Dense;

        bool draftRequested = request.HasDraftModel || request.DraftLookup;
        string tqMode = request.TurboQuantMode.Trim().ToLowerInvariant();
        bool explicitKvarn = tqMode == "kvarn";
        bool explicitLloydMax = tqMode is "lloydmax" or "lloyd-max";
        bool tqModeIsAuto = tqMode is "auto" or "";
        if (!request.ValidateKVarNOnly && !request.ValidateTurboQuantHeadDimOnly
            && !tqModeIsAuto && !explicitKvarn && !explicitLloydMax)
            return ForwardPassDecision.Refuse($"Unknown --tq-mode value '{request.TurboQuantMode}'. Expected one of: auto, lloydmax, kvarn.");
        if (explicitKvarn && !request.TurboQuant)
            return ForwardPassDecision.Refuse("--tq-mode kvarn requires --tq.");

        if (request.IsHybridSsm && request.TurboQuant
            && !request.ValidateKVarNOnly && !request.ValidateTurboQuantHeadDimOnly)
            return ForwardPassDecision.Refuse("TurboQuant is not supported for hybrid GDN models (no KV cache on GDN layers).");
        if (!request.ValidateTurboQuantModeOnly && !request.ValidateKVarNOnly
            && !request.ValidateTurboQuantHeadDimOnly
            && (request.IsHybridSsm || request.HasHybridGdnLayers)
            && draftRequested)
            return ForwardPassDecision.Refuse("Speculative decoding is not supported for hybrid GDN models (GDN state is destructively updated and cannot be rewound).");
        if (request.ValidateTurboQuantModeOnly)
            return ForwardPassDecision.Select(ForwardPassKind.CpuDense);

        if (family == ForwardPassFamily.Rwkv)
        {
            if (request.SkipFamilyRefusals)
                return ForwardPassDecision.Select(ForwardPassKind.RwkvCpu);
            if (request.TurboQuant || (draftRequested))
            {
                string message = $"{request.Architecture} is recurrent (no KV cache) and supports neither TurboQuant nor speculative decoding.";
                return ForwardPassDecision.Refuse(message);
            }
            return ForwardPassDecision.Select(ForwardPassKind.RwkvCpu);
        }

        bool isGptOss = family == ForwardPassFamily.GptOss;
        if (isGptOss && request.SkipFamilyRefusals)
            return ForwardPassDecision.Select(ForwardPassKind.GptOssCpu);
        if (isGptOss && (request.TurboQuant || (draftRequested)))
        {
            string message = "gpt-oss runs on its own CPU forward pass, which supports neither TurboQuant nor speculative decoding.";
            return ForwardPassDecision.Refuse(message);
        }

        if (request.ValidateKVarNOnly || request.ValidateTurboQuantHeadDimOnly)
        {
            if (request.ValidateKVarNOnly && explicitKvarn && !request.TurboQuant)
                return ForwardPassDecision.Refuse("--tq-mode kvarn requires --tq.");
            if (request.ValidateKVarNOnly)
            {
                if (explicitKvarn && request.KVarNSnapKvBlocked)
                    return ForwardPassDecision.Refuse("--tq-mode kvarn does not compose with SnapKV eviction yet (issue #180 follow-up); unset STINGRAY_SNAPKV_BUDGET.");
                if (explicitKvarn && request.GpuLayers != 0)
                {
                    if (request.Backend == ForwardPassBackend.Vulkan)
                        return ForwardPassDecision.Refuse("--tq-mode kvarn is not supported on the Vulkan backend; use --backend cuda -g -1 (full offload) or -g 0 (CPU).");
                    if (!request.CudaAvailable)
                        return ForwardPassDecision.Refuse("--tq-mode kvarn with GPU offload requires a CUDA device (issue #180 Task 5a); use -g 0 for the CPU path.");
                    if (request.IsMoE)
                        return ForwardPassDecision.Refuse("--tq-mode kvarn on CUDA supports dense models only (issue #180 Task 5a); use -g 0 for MoE.");
                }
            }
            if (request.ValidateTurboQuantHeadDimOnly)
            {
                if (request.TurboQuant && explicitKvarn && !TqSupport.IsKVarNHeadDim(request.HeadDim))
                    return ForwardPassDecision.Refuse($"--tq-mode kvarn requires a power-of-2 head dimension in [8, 1024]; this model has head dim {request.HeadDim}.");
                if (request.TurboQuant && explicitKvarn && request.GpuLayers != 0 && request.HeadDim > TqSupport.KVarNCudaMaxHeadDim)
                    return ForwardPassDecision.Refuse($"--tq-mode kvarn on CUDA requires head dim ≤ 256 (shared-memory WHT cap); this model has head dim {request.HeadDim}. Use -g 0 for the CPU path.");
                if (request.TurboQuant && !explicitKvarn && !TqSupport.IsLloydMaxHeadDim(request.HeadDim))
                    return ForwardPassDecision.Refuse($"TurboQuant requires head dimension 128 or 256; this model has head dim {request.HeadDim}. Remove --tq to run without KV compression.");
            }
            return ForwardPassDecision.Select(family == ForwardPassFamily.GptOss
                ? ForwardPassKind.GptOssCpu
                : ForwardPassKind.CpuDense);
        }

        if (!request.SkipTurboQuantShapeValidation
            && (request.TurboQuant || explicitKvarn || explicitLloydMax))
        {
            if (explicitLloydMax && request.HeadDim is not (128 or 256))
            {
                if (request.TurboQuant)
                    return ForwardPassDecision.Refuse($"TurboQuant requires head dimension 128 or 256; this model has head dim {request.HeadDim}. Remove --tq to run without KV compression.");
            }
            if (!explicitLloydMax && (request.HeadDim < 8 || request.HeadDim > 1024 || (request.HeadDim & (request.HeadDim - 1)) != 0))
            {
                if (request.TurboQuant)
                    return ForwardPassDecision.Refuse($"--tq-mode kvarn requires a power-of-2 head dimension in [8, 1024]; this model has head dim {request.HeadDim}.");
            }
            if (explicitKvarn || tqModeIsAuto && request.TurboQuant)
            {
                string? kvarnBlockedReason = request.KVarNSnapKvBlocked ? "SnapKV" : null;
                ForwardPassBackend requestedBackend = request.Backend;
                if (kvarnBlockedReason is null && request.GpuLayers != 0 && requestedBackend == ForwardPassBackend.Vulkan)
                    kvarnBlockedReason = "Vulkan";
                if (kvarnBlockedReason is null && request.GpuLayers != 0 && !request.CudaAvailable)
                    kvarnBlockedReason = "CUDA is unavailable";
                if (kvarnBlockedReason is null && request.GpuLayers != 0 && (request.IsMoE || request.KVarNCudaMoeBlocked))
                    kvarnBlockedReason = "MoE on CUDA";
                if (kvarnBlockedReason is null && request.KVarNPartialCudaBlocked)
                    kvarnBlockedReason = "partial CUDA offload";
                if (explicitKvarn && kvarnBlockedReason is not null)
                {
                        string message = request.KVarNSnapKvBlocked
                            ? "--tq-mode kvarn does not compose with SnapKV eviction yet (issue #180 follow-up); unset STINGRAY_SNAPKV_BUDGET."
                            : request.Backend == ForwardPassBackend.Vulkan
                                ? "--tq-mode kvarn is not supported on the Vulkan backend; use --backend cuda -g -1 (full offload) or -g 0 (CPU)."
                                : request.GpuLayers != 0 && !request.CudaAvailable
                                    ? "--tq-mode kvarn with GPU offload requires a CUDA device (issue #180 Task 5a); use -g 0 for the CPU path."
                                    : request.IsMoE && request.GpuLayers != 0
                                        ? "--tq-mode kvarn on CUDA supports dense models only (issue #180 Task 5a); use -g 0 for MoE."
                                        : request.KVarNPartialCudaBlocked
                                            ? $"--tq-mode kvarn requires full CUDA offload, but only {request.PlannedGpuLayersForKVarN}/{request.NumLayers} layers fit this GPU. Use -g 0 for the CPU path."
                                            : "--tq-mode kvarn requires full CUDA offload or the CPU path.";
                        return ForwardPassDecision.Refuse(message);
                }
                if (tqModeIsAuto && kvarnBlockedReason is not null)
                {
                    if (request.HeadDim is not (128 or 256))
                        return ForwardPassDecision.Refuse($"TurboQuant requires head dimension 128 or 256; this model has head dim {request.HeadDim}. Remove --tq to run without KV compression.");
                }
            }
            if (explicitKvarn && request.GpuLayers != 0
                && request.GpuLayers > 0 && request.GpuLayers < request.NumLayers)
                return ForwardPassDecision.Refuse($"--tq-mode kvarn requires full CUDA offload, but only {request.PlannedGpuLayersForKVarN}/{request.NumLayers} layers fit this GPU. Use -g 0 for the CPU path.");
            if (explicitKvarn && request.GpuLayers != 0
                && request.Backend != ForwardPassBackend.Vulkan && request.HeadDim > 256)
                return ForwardPassDecision.Refuse($"--tq-mode kvarn on CUDA requires head dim ≤ 256 (shared-memory WHT cap); this model has head dim {request.HeadDim}. Use -g 0 for the CPU path.");
            if (explicitKvarn && request.Backend == ForwardPassBackend.Vulkan
                && request.GpuLayers != 0)
                return ForwardPassDecision.Refuse("--tq-mode kvarn is not supported on the Vulkan backend; use --backend cuda -g -1 (full offload) or -g 0 (CPU).");
        }

        if (request.HasImageInput)
        {
            if (!request.SupportsEmbeddingInput)
                return ForwardPassDecision.Refuse("Image input requires a forward pass that accepts precomputed-embedding input.");
            if (request.HasImageBatching || request.MaxBatchSizeGreaterThanOne && request.BatchingSupported)
                return ForwardPassDecision.Refuse("Image input is not supported with continuous batching (MaxBatchSize > 1). Set MaxBatchSize=1.");
        }

        if (request.SessionsRequested
            && (request.IsSafeTensors || !request.SessionsPathSupported || !request.BatchingSupported || !request.IsConcreteCpuDensePass))
            return ForwardPassDecision.Refuse(request.IsSafeTensors
                ? "EnableSessions currently supports only the proven CPU-dense GGUF lane; SafeTensors session/cache conformance is not available yet."
                : "EnableSessions currently supports only the proven CPU-dense GGUF lane without MoE or TurboQuant. Disable EnableSessions or select CPU dense GGUF.");

        if (request.IsContinuousBatching && request.SupportsContinuousBatching && request.DsParkRequested)
            return ForwardPassDecision.Refuse("DSpark (DSparkModelPath / STINGRAY_DSPARK_MODEL) is not supported with continuous batching (MaxBatchSize > 1) — the tap buffer is single-sequence. Set MaxBatchSize=1.");

        if (request.DsParkRequested)
        {
            if (!request.HasDSparkModel)
                return ForwardPassDecision.Refuse("--spec-type dspark requires --dspark-model <path-to-model.safetensors>.");
            if (!request.DSparkModelPathExists)
                return ForwardPassDecision.Refuse("DSpark model not found: configured DSpark path.");
            if (!request.HasDSparkConfig)
                return ForwardPassDecision.Refuse("DSpark config.json not found next to the safetensors.");
            if (!request.DSparkConfigExists)
                return ForwardPassDecision.Refuse("DSpark config.json not found next to the safetensors.");
            if (!request.DSparkHeadMatchesTarget)
                return ForwardPassDecision.Refuse(
                    $"DSpark head/target mismatch — head expects vocab {request.DSparkVocabSize}, {request.DSparkTargetLayers} target layers, hidden {request.DSparkHiddenSize}; target has vocab {request.ModelVocabSize}, {request.ModelLayers} layers, hidden {request.ModelHiddenSize}. The head must be trained for this target model.");
            if ((request.HasDraftModel || request.DraftLookup))
                return ForwardPassDecision.Refuse("--dspark-model and --draft-model/--draft-lookup are mutually exclusive.");
            if (request.HasMtpSpecType)
                return ForwardPassDecision.Refuse("--spec-type mtp conflicts with --dspark-model; pick one.");
            if (request.DSparkMinConfidence > 1f)
                return ForwardPassDecision.Refuse("--dspark-min-confidence must be in [0, 1].");
            if ((!request.DSparkTargetSupported || !request.SupportsHiddenTaps
                || request.HasTokenConstraint || request.HasTools || request.IsSampled))
                return SelectOrdinary(request, "DSpark disabled; falling back to normal generation.");
            if (request.DSparkPlacementOff)
                return SelectOrdinary(request, "DSpark placement is off; falling back to normal generation.");
            int window = request.DSparkContextLength > 0
                ? Math.Min(request.TargetContextLength, request.DSparkContextLength)
                : request.TargetContextLength;
            if (window > 0 && request.PromptTokenCount + request.DSparkBlockSize + 1 >= window)
                return ForwardPassDecision.Refuse(
                    $"prompt ({request.PromptTokenCount} tokens) + DSpark block ({request.DSparkBlockSize}) does not fit the context window ({window} tokens).");
            if (!request.HasSinglePrompt)
                return SelectOrdinary(request, "DSpark is wired for single-prompt runs only; interactive mode falls back to normal generation.");
        }

        if (draftRequested && request.HasDraftModel && request.DraftLookup)
            return ForwardPassDecision.Refuse("--draft-model and --draft-lookup are mutually exclusive.");
        if (request.HasDraftModel && !request.DraftModelExists)
            return ForwardPassDecision.Refuse($"Draft model not found: {request.DraftModelPath ?? "<path>"}.");
        if (draftRequested && request.HasTokenConstraint)
            return ForwardPassDecision.Select(SelectOrdinary(request).Kind!.Value, "Speculation disabled because a token constraint is active.");
        if (draftRequested && request.IsSampled && request.DraftLookup)
            return ForwardPassDecision.Select(SelectOrdinary(request).Kind!.Value, "--draft-lookup supports greedy (--temp 0) only; falling back to normal generation.");
        if (draftRequested && request.IsSampled && (request.SampledSpecDisabled || request.HasSamplingPenaltyOrBias))
            return ForwardPassDecision.Select(SelectOrdinary(request).Kind!.Value, "Sampled speculative decoding is disabled or incompatible with penalties/logit bias.");

        bool hasDeepSeek2Mla = family == ForwardPassFamily.DeepSeek2Mla
            && request.KvLoraRank > 0 && request.HasMlaTensors;
        bool partialRequest = request.GpuLayers > 0 && request.GpuLayers < request.NumLayers;
        bool cliDraftBlocksMla = draftRequested;
        bool mlaRequestedBackend = request.Backend != ForwardPassBackend.Cuda;
        if (hasDeepSeek2Mla && request.GpuLayers != 0 && !request.TurboQuant
            && mlaRequestedBackend && !partialRequest && !cliDraftBlocksMla)
            return ForwardPassDecision.Select(ForwardPassKind.DeepSeek2Vulkan);

        bool gpuRequested = request.GpuLayers != 0;
        ForwardPassBackend backend = request.Backend;
        if (backend == ForwardPassBackend.Auto && gpuRequested)
            backend = request.CudaAvailable ? ForwardPassBackend.Cuda : ForwardPassBackend.Vulkan;
        if (request.UnsupportedGpuPath && gpuRequested)
        {
            backend = ForwardPassBackend.Cpu;
            gpuRequested = false;
        }

        if (backend == ForwardPassBackend.Cuda && request.UnsupportedPartialCudaPath && gpuRequested)
        {
            backend = ForwardPassBackend.Cpu;
            gpuRequested = false;
        }

        if (backend == ForwardPassBackend.Cuda && request.UnsupportedPartialCudaPath)
            return ForwardPassDecision.Select(ForwardPassKind.CpuDense);

        if (descriptor is not null && gpuRequested
                && family is not (ForwardPassFamily.Rwkv or ForwardPassFamily.GptOss)
                && !SupportsBackend(descriptor.SupportedBackends, backend))
        {
            backend = ForwardPassBackend.Cpu;
            gpuRequested = false;
        }

        if (isGptOss)
        {
            bool cudaOnly = request.Backend == ForwardPassBackend.Cuda;
            bool partial = request.GpuLayers > 0 && request.GpuLayers < request.NumLayers;
            if (gpuRequested && (cudaOnly || partial))
                return ForwardPassDecision.Select(ForwardPassKind.GptOssCpu);
            if (request.Backend == ForwardPassBackend.Auto
                && gpuRequested && !partial)
                return ForwardPassDecision.Select(ForwardPassKind.GptOssVulkan);
            if (backend == ForwardPassBackend.Vulkan && gpuRequested && !partial)
                return ForwardPassDecision.Select(ForwardPassKind.GptOssVulkan);
            return ForwardPassDecision.Select(ForwardPassKind.GptOssCpu);
        }

        if (request.IsHybridSsm || request.HasHybridGdnLayers)
        {
            if (!gpuRequested || backend == ForwardPassBackend.Cpu)
                return ForwardPassDecision.Select(ForwardPassKind.CpuHybridGdn);
            return ForwardPassDecision.Select(backend == ForwardPassBackend.Cuda
                ? ForwardPassKind.CudaHybridGdn
                : ForwardPassKind.VulkanHybridGdn);
        }

        if (backend == ForwardPassBackend.Cuda && request.UnsupportedPartialCudaPath)
            return ForwardPassDecision.Select(ForwardPassKind.CpuDense);

        if (!gpuRequested || backend == ForwardPassBackend.Cpu)
            return ForwardPassDecision.Select(request.HasCpuHybridGdnPass
                ? ForwardPassKind.CpuHybridGdn
                : ForwardPassKind.CpuDense);

        int gpuLayers = request.GpuLayers == -1 ? request.PlannedGpuLayers : request.GpuLayers;
        if (gpuLayers <= 0)
            return ForwardPassDecision.Select(ForwardPassKind.CpuDense);
        if (request.LayerSplitOnlyLargeMoe && request.AutoPlan && request.IsMoE && gpuLayers > 4)
            return ForwardPassDecision.Select(ForwardPassKind.VulkanLayerSplit,
                "Automatic split limited to four GPU layers for large split-only MoE.");
        if (backend == ForwardPassBackend.Cuda)
        {
            if (request.TurboQuant && explicitKvarn && request.KVarNPartialCudaBlocked)
                return ForwardPassDecision.Refuse($"--tq-mode kvarn requires full CUDA offload, but only {request.PlannedGpuLayersForKVarN}/{request.NumLayers} layers fit this GPU. Use -g 0 for the CPU path.");
            if (request.TurboQuant && explicitKvarn && request.KVarNCudaMoeBlocked)
                return ForwardPassDecision.Refuse("--tq-mode kvarn on CUDA supports dense models only (issue #180 Task 5a); use -g 0 for MoE.");
            if (request.MoEAutoCacheOverflow)
                return ForwardPassDecision.Select(ForwardPassKind.CudaHybrid);
            return ForwardPassDecision.Select(gpuLayers >= request.NumLayers
                ? ForwardPassKind.CudaDense
                : ForwardPassKind.CudaHybrid);
        }
        if (gpuLayers >= request.NumLayers)
            return ForwardPassDecision.Select(ForwardPassKind.VulkanDense);
        if (request.HasLayerHeadDim || request.UnsupportedPartialVulkanPath || request.LayerSplitOnly)
        {
            return ForwardPassDecision.Select(ForwardPassKind.VulkanLayerSplit);
        }
        return ForwardPassDecision.Select(ForwardPassKind.VulkanHybrid);
    }

    private static ForwardPassDecision SelectOrdinary(ForwardPassRequest request, string? notice = null)
    {
        var copy = request with
        {
            DsParkRequested = false,
            HasDraftModel = false,
            DraftLookup = false,
            HasTokenConstraint = false,
            HasTools = false,
            IsSampled = false,
        };
        var selected = Select(copy);
        return selected with { Notice = notice ?? selected.Notice };
    }

    private static bool SupportsBackend(SupportedBackends supportedBackends, ForwardPassBackend backend) => backend switch
    {
        ForwardPassBackend.Cpu => (supportedBackends & SupportedBackends.Cpu) != 0,
        ForwardPassBackend.Cuda => (supportedBackends & SupportedBackends.Cuda) != 0,
        ForwardPassBackend.Vulkan => (supportedBackends & SupportedBackends.Vulkan) != 0,
        _ => false,
    };
}
