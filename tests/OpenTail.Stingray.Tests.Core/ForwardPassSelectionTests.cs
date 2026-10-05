using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

public sealed class ForwardPassSelectionTests
{
    public static TheoryData<string, ForwardPassRequest, ForwardPassKind?, string?> MatrixCases => new()
    {
        { "Safetensors-supported", new() { IsSafeTensors = true }, ForwardPassKind.SafeTensorsCpu, null },
        { "Safetensors-profile-refusal", new() { IsSafeTensors = true, PackageSupported = false }, null, "Model package is not supported by its SafeTensors profile." },
        { "Safetensors-gpu-refusal", new() { IsSafeTensors = true, IsSafeTensorsGpuRequested = true }, null, "GPU offload is not yet supported for SafeTensors packages." },
        { "Safetensors-tq-refusal", new() { IsSafeTensors = true, TurboQuant = true }, null, "--tq (TurboQuant) is not supported for SafeTensors packages." },
        { "Safetensors-draft-refusal", new() { IsSafeTensors = true, IsSafeTensorsDraftRequested = true }, null, "Speculative decoding is not supported for SafeTensors packages." },
        { "Safetensors-dspark-refusal", new() { IsSafeTensors = true, IsSafeTensorsDSparkRequested = true }, null, "DSpark is not supported for SafeTensors packages." },
        { "unsupported-architecture-refusal", new() { Architecture = "unverified", ArchitectureSupported = false, ArchitectureRefusal = "architecture denied" }, null, "architecture denied" },
        { "unsupported-architecture-override", new() { Architecture = "unverified", ArchitectureSupported = false, AllowUnverifiedArchitecture = true }, ForwardPassKind.CpuDense, null },
        { "cpu-dense", new(), ForwardPassKind.CpuDense, null },
        { "cpu-hybrid-gdn", new() { IsHybridSsm = true }, ForwardPassKind.CpuHybridGdn, null },
        { "cuda-hybrid-gdn", new() { IsHybridSsm = true, GpuLayers = -1, Backend = ForwardPassBackend.Cuda }, ForwardPassKind.CudaHybridGdn, null },
        { "vulkan-hybrid-gdn", new() { IsHybridSsm = true, GpuLayers = -1, Backend = ForwardPassBackend.Vulkan }, ForwardPassKind.VulkanHybridGdn, null },
        { "rwkv-cpu", new() { Architecture = "rwkv7", GpuLayers = -1, Backend = ForwardPassBackend.Vulkan }, ForwardPassKind.RwkvCpu, null },
        { "rwkv-tq-refusal-cli", new() { Architecture = "rwkv6", TurboQuant = true }, null, "rwkv6 is recurrent (no KV cache) and supports neither TurboQuant nor speculative decoding." },
        { "rwkv-tq-refusal-server", new() { Frontend = ForwardPassFrontend.Server, Architecture = "rwkv6", TurboQuant = true }, null, "TurboQuant is not supported for RWKV (no KV cache)." },
        { "rwkv-spec-refusal-cli", new() { Architecture = "rwkv7", HasDraftModel = true }, null, "rwkv7 is recurrent (no KV cache) and supports neither TurboQuant nor speculative decoding." },
        { "rwkv-spec-server-cpu", new() { Frontend = ForwardPassFrontend.Server, Architecture = "rwkv7", HasDraftModel = true }, ForwardPassKind.RwkvCpu, null },
        { "gpt-oss-cpu", new() { Architecture = "gpt-oss" }, ForwardPassKind.GptOssCpu, null },
        { "gpt-oss-vulkan", new() { Architecture = "gpt-oss", GpuLayers = -1, Backend = ForwardPassBackend.Vulkan }, ForwardPassKind.GptOssVulkan, null },
        { "gpt-oss-cuda-cli-fallback", new() { Architecture = "gpt-oss", GpuLayers = -1, Backend = ForwardPassBackend.Cuda }, ForwardPassKind.GptOssCpu, null },
        { "gpt-oss-cuda-server-fallback", new() { Frontend = ForwardPassFrontend.Server, Architecture = "gpt-oss", GpuLayers = -1, Backend = ForwardPassBackend.Cuda }, ForwardPassKind.GptOssCpu, null },
        { "gpt-oss-spec-refusal", new() { Architecture = "gpt-oss", HasDraftModel = true }, null, "gpt-oss runs on its own CPU forward pass, which supports neither TurboQuant nor speculative decoding." },
        { "gpt-oss-server-spec-available", new() { Frontend = ForwardPassFrontend.Server, Architecture = "gpt-oss", HasDraftModel = true }, ForwardPassKind.GptOssCpu, null },
        { "deepseek2-mla-vulkan", new() { Architecture = "deepseek2", KvLoraRank = 1, HasMlaTensors = true, GpuLayers = -1, Backend = ForwardPassBackend.Vulkan }, ForwardPassKind.DeepSeek2Vulkan, null },
        { "deepseek2-mla-cuda-limit", new() { Architecture = "deepseek2", KvLoraRank = 1, HasMlaTensors = true, GpuLayers = -1, Backend = ForwardPassBackend.Cuda }, null, "DeepSeek2 MLA supports CPU or full Vulkan offload; CUDA and partial GPU offload use CPU." },
        { "deepseek2-mla-server-auto", new() { Frontend = ForwardPassFrontend.Server, Architecture = "deepseek2", KvLoraRank = 1, HasMlaTensors = true, GpuLayers = -1, Backend = ForwardPassBackend.Auto }, ForwardPassKind.DeepSeek2Vulkan, null },
        { "deepseek2-mla-cli-spec-divergence", new() { Architecture = "deepseek2", KvLoraRank = 1, HasMlaTensors = true, GpuLayers = -1, PlannedGpuLayers = 2, NumLayers = 4, Backend = ForwardPassBackend.Vulkan, HasDraftModel = true }, ForwardPassKind.VulkanHybrid, null },
        { "deepseek2-mla-server-no-spec-flag", new() { Frontend = ForwardPassFrontend.Server, Architecture = "deepseek2", KvLoraRank = 1, HasMlaTensors = true, GpuLayers = -1, Backend = ForwardPassBackend.Vulkan, HasDraftModel = true }, ForwardPassKind.DeepSeek2Vulkan, null },
        { "unsupported-gpu-to-cpu", new() { GpuLayers = -1, Backend = ForwardPassBackend.Vulkan, UnsupportedGpuPath = true, HasCpuHybridGdnPass = true }, ForwardPassKind.CpuHybridGdn, null },
        { "cuda-partial-unsupported-to-cpu", new() { GpuLayers = 2, NumLayers = 4, Backend = ForwardPassBackend.Cuda, UnsupportedPartialCudaPath = true }, ForwardPassKind.CpuDense, null },
        { "cuda-dense", new() { GpuLayers = -1, PlannedGpuLayers = 4, NumLayers = 4, Backend = ForwardPassBackend.Cuda }, ForwardPassKind.CudaDense, null },
        { "cuda-hybrid", new() { GpuLayers = 2, NumLayers = 4, Backend = ForwardPassBackend.Cuda }, ForwardPassKind.CudaHybrid, null },
        { "vulkan-dense", new() { GpuLayers = -1, PlannedGpuLayers = 4, NumLayers = 4, Backend = ForwardPassBackend.Vulkan }, ForwardPassKind.VulkanDense, null },
        { "vulkan-hybrid", new() { GpuLayers = 2, NumLayers = 4, Backend = ForwardPassBackend.Vulkan }, ForwardPassKind.VulkanHybrid, null },
        { "vulkan-layer-split", new() { GpuLayers = 2, NumLayers = 4, Backend = ForwardPassBackend.Vulkan, UnsupportedPartialVulkanPath = true }, ForwardPassKind.VulkanLayerSplit, null },
        { "vulkan-large-moe-split-cap", new() { GpuLayers = -1, PlannedGpuLayers = 12, NumLayers = 16, Backend = ForwardPassBackend.Vulkan, IsMoE = true, LayerSplitOnly = true, AutoPlan = true, LayerSplitOnlyLargeMoe = true }, ForwardPassKind.VulkanLayerSplit, null },
        { "muse-glimmer-gpu-limit", new() { Architecture = "muse-glimmer", GpuLayers = -1, PlannedGpuLayers = 1, Backend = ForwardPassBackend.Cuda }, null, "Muse-Glimmer's attention output gate and embedding norm are supported by the CPU pass only." },
        { "cli-hybrid-tq-refusal", new() { IsHybridSsm = true, TurboQuant = true }, null, "TurboQuant is not supported for hybrid GDN models (no KV cache on GDN layers)." },
        { "server-hybrid-tq-refusal", new() { Frontend = ForwardPassFrontend.Server, IsHybridSsm = true, TurboQuant = true }, null, "TurboQuant is not supported for hybrid GDN models (no KV cache on GDN layers)." },
        { "cli-hybrid-spec-refusal", new() { IsHybridSsm = true, HasDraftModel = true }, null, "Speculative decoding is not supported for hybrid GDN models (GDN state is destructively updated and cannot be rewound)." },
        { "server-hybrid-spec-target", new() { Frontend = ForwardPassFrontend.Server, IsHybridSsm = true, HasDraftModel = true }, ForwardPassKind.CpuHybridGdn, null },
        { "tq-unknown-mode-cli", new() { TurboQuantMode = "other" }, null, "Unknown --tq-mode value 'other'. Expected one of: auto, lloydmax, kvarn." },
        { "tq-unknown-mode-server", new() { Frontend = ForwardPassFrontend.Server, TurboQuantMode = "other" }, null, "Unknown TqMode 'other'. Expected one of: auto, lloydmax, kvarn." },
        { "tq-explicit-kvarn-without-enable", new() { TurboQuantMode = "kvarn" }, null, "--tq-mode kvarn requires --tq." },
        { "tq-lloydmax-bad-head-cli", new() { TurboQuant = true, TurboQuantMode = "lloydmax", HeadDim = 64 }, null, "TurboQuant requires head dimension 128 or 256; this model has head dim 64. Remove --tq to run without KV compression." },
        { "tq-lloydmax-bad-head-server", new() { Frontend = ForwardPassFrontend.Server, TurboQuant = true, TurboQuantMode = "lloydmax", HeadDim = 64 }, null, "TurboQuant Lloyd-Max requires head dimension 128 or 256; this model has head dim 64." },
        { "tq-kvarn-bad-head", new() { TurboQuant = true, TurboQuantMode = "kvarn", HeadDim = 6 }, null, "--tq-mode kvarn requires a power-of-2 head dimension in [8, 1024]; this model has head dim 6." },
        { "tq-kvarn-snapkv-cli", new() { TurboQuant = true, TurboQuantMode = "kvarn", KVarNSnapKvBlocked = true }, null, "--tq-mode kvarn does not compose with SnapKV eviction yet (issue #180 follow-up); unset STINGRAY_SNAPKV_BUDGET." },
        { "tq-kvarn-vulkan-cli", new() { TurboQuant = true, TurboQuantMode = "kvarn", Backend = ForwardPassBackend.Vulkan, GpuLayers = -1 }, null, "--tq-mode kvarn is not supported on the Vulkan backend; use --backend cuda -g -1 (full offload) or -g 0 (CPU)." },
        { "tq-kvarn-cuda-unavailable", new() { TurboQuant = true, TurboQuantMode = "kvarn", GpuLayers = -1, Backend = ForwardPassBackend.Cuda, CudaAvailable = false }, null, "--tq-mode kvarn with GPU offload requires a CUDA device (issue #180 Task 5a); use -g 0 for the CPU path." },
        { "tq-kvarn-moe-cuda", new() { TurboQuant = true, TurboQuantMode = "kvarn", GpuLayers = -1, Backend = ForwardPassBackend.Cuda, CudaAvailable = true, IsMoE = true, KVarNCudaMoeBlocked = true }, null, "--tq-mode kvarn on CUDA supports dense models only (issue #180 Task 5a); use -g 0 for MoE." },
        { "tq-kvarn-partial-cuda", new() { TurboQuant = true, TurboQuantMode = "kvarn", GpuLayers = 2, NumLayers = 4, Backend = ForwardPassBackend.Cuda, CudaAvailable = true, KVarNPartialCudaBlocked = true, PlannedGpuLayersForKVarN = 2 }, null, "--tq-mode kvarn requires full CUDA offload, but only 2/4 layers fit this GPU. Use -g 0 for the CPU path." },
        { "tq-kvarn-cuda-dim-cap", new() { TurboQuant = true, TurboQuantMode = "kvarn", HeadDim = 512, GpuLayers = -1, Backend = ForwardPassBackend.Cuda, CudaAvailable = true }, null, "--tq-mode kvarn on CUDA requires head dim ≤ 256 (shared-memory WHT cap); this model has head dim 512. Use -g 0 for the CPU path." },
        { "cli-draft-model-and-lookup", new() { HasDraftModel = true, DraftLookup = true }, null, "--draft-model and --draft-lookup are mutually exclusive." },
        { "cli-draft-file-missing", new() { HasDraftModel = true, DraftModelExists = false, DraftModelPath = "draft.gguf" }, null, "Draft model not found: draft.gguf." },
        { "cli-draft-token-constraint-fallback", new() { HasDraftModel = true, HasTokenConstraint = true }, ForwardPassKind.CpuDense, null },
        { "cli-sampled-draftlookup-fallback", new() { DraftLookup = true, IsSampled = true }, ForwardPassKind.CpuDense, null },
        { "server-standard-spec-option-absent", new() { Frontend = ForwardPassFrontend.Server, HasDraftModel = true }, ForwardPassKind.CpuDense, null },
        { "cli-dspark-model-missing", new() { DsParkRequested = true }, null, "--spec-type dspark requires --dspark-model <path-to-model.safetensors>." },
        { "cli-dspark-mutually-exclusive", new() { DsParkRequested = true, HasDSparkModel = true, HasDraftModel = true }, null, "--dspark-model and --draft-model/--draft-lookup are mutually exclusive." },
        { "cli-dspark-mtp-conflict", new() { DsParkRequested = true, HasDSparkModel = true, HasMtpSpecType = true }, null, "--spec-type mtp conflicts with --dspark-model; pick one." },
        { "cli-dspark-confidence", new() { DsParkRequested = true, HasDSparkModel = true, DSparkMinConfidence = 1.1f }, null, "--dspark-min-confidence must be in [0, 1]." },
        { "cli-dspark-target-fallback", new() { DsParkRequested = true, HasDSparkModel = true, DSparkTargetSupported = false }, ForwardPassKind.CpuDense, null },
        { "cli-dspark-no-single-prompt-fallback", new() { DsParkRequested = true, HasDSparkModel = true, HasSinglePrompt = false }, ForwardPassKind.CpuDense, null },
        { "cli-dspark-placement-off", new() { DsParkRequested = true, HasDSparkModel = true, DSparkPlacementOff = true }, ForwardPassKind.CpuDense, null },
        { "server-dspark-placement-off", new() { Frontend = ForwardPassFrontend.Server, DsParkRequested = true, HasDSparkModel = true, DSparkPlacementOff = true }, null, "DSpark was configured but placement resolved to Off." },
        { "server-dspark-no-hidden-taps", new() { Frontend = ForwardPassFrontend.Server, DsParkRequested = true, HasDSparkModel = true, SupportsHiddenTaps = false }, null, "DSpark requires a tap-capable dense forward pass (CPU, NGpuLayers=0, or full CUDA offload, NGpuLayers=-1; no MoE / Gemma-4 / TurboQuant / SnapKV)." },
        { "server-dspark-model-path-missing", new() { Frontend = ForwardPassFrontend.Server, DsParkRequested = true, HasDSparkModel = true, DSparkModelPathExists = false }, null, "DSpark model not found: configured DSpark path." },
        { "server-dspark-target-mismatch", new() { Frontend = ForwardPassFrontend.Server, DsParkRequested = true, HasDSparkModel = true, DSparkHeadMatchesTarget = false, DSparkVocabSize = 10, ModelVocabSize = 20, DSparkTargetLayers = 2, ModelLayers = 4, DSparkHiddenSize = 8, ModelHiddenSize = 16 }, null, "DSpark head/target mismatch — head expects vocab 10, 2 target layers, hidden 8; target has vocab 20, 4 layers, hidden 16." },
        { "server-dspark-batching", new() { Frontend = ForwardPassFrontend.Server, DsParkRequested = true, HasDSparkModel = true, IsContinuousBatching = true, SupportsContinuousBatching = true }, null, "DSpark (DSparkModelPath / STINGRAY_DSPARK_MODEL) is not supported with continuous batching (MaxBatchSize > 1) — the tap buffer is single-sequence. Set MaxBatchSize=1." },
        { "server-image-non-gemma", new() { Frontend = ForwardPassFrontend.Server, Architecture = "qwen3", HasImageInput = true, IsGemma4 = false }, null, "Image input (MmprojPath / STINGRAY_MMPROJ) is only supported for Gemma 4 (gemma4uv) text models; this model's architecture is 'qwen3'." },
        { "server-image-no-embedding", new() { Frontend = ForwardPassFrontend.Server, Architecture = "gemma4", IsGemma4 = true, HasImageInput = true, SupportsEmbeddingInput = false }, null, "MmprojPath / STINGRAY_MMPROJ is set but image input requires a forward pass that accepts precomputed-embedding input: CPU (NGpuLayers=0) or full CUDA offload (NGpuLayers=-1) of a Gemma 4 model that fits VRAM." },
        { "server-image-batching", new() { Frontend = ForwardPassFrontend.Server, Architecture = "gemma4", IsGemma4 = true, HasImageInput = true, HasImageBatching = true }, null, "Image input is not supported with continuous batching (MaxBatchSize > 1). Set MaxBatchSize=1." },
    };

    [Theory]
    [MemberData(nameof(MatrixCases))]
    public void Select_MatchesMatrixCase(string caseId, ForwardPassRequest request, ForwardPassKind? expectedKind, string? expectedRefusal)
    {
        var decision = ForwardPassSelection.Select(request);

        Assert.Equal(expectedKind, decision.Kind);
        Assert.True(expectedRefusal == decision.Refusal, caseId);
        Assert.Equal(expectedRefusal is not null, decision.IsRefused);
    }
}
