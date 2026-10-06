#nullable enable

using System.Collections.Immutable;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine.Packaging;

namespace OpenTail.Stingray.Engine.Planning;

/// <summary>
/// Intrinsic, immutable description of a model package and its planning facts.
/// Serves as the authoritative source of model truth for <see cref="ExecutionPlanner"/>
/// without requiring disk or tensor re-reads.
/// </summary>
public sealed record ModelDescription(
    ModelPackageIdentity Identity,
    ModelSemanticDescription Semantics,
    ModelCapabilitySummary Capabilities,
    ModelResourceSummary Resources,
    ModelPlanningFacts PlanningFacts,
    string? PrimaryPath = null)
{
    /// <summary>
    /// Performs a one-time package inspection to construct an immutable <see cref="ModelDescription"/>.
    /// Once constructed, the package files are closed and callers must rely solely on this snapshot.
    /// </summary>
    public static ModelDescription FromPackage(IModelPackage package)
    {
        ArgumentNullException.ThrowIfNull(package);

        if (package.Format == ModelFormat.SafeTensors)
        {
            using var tensorSource = SafetensorsTensorSource.Open(package.PrimaryPath);
            return FromTensorSource(package.Identity, package.PrimaryPath, tensorSource, ModelFormat.SafeTensors, package);
        }

        // GGUF or Ollama (whose primary blob is GGUF format)
        using var gguf = GgufModel.Open(package.PrimaryPath);
        return FromGgufModel(package.Identity, package.PrimaryPath, gguf, package.Format, package);
    }

    /// <summary>
    /// Constructs a <see cref="ModelDescription"/> from an already-loaded <see cref="Model"/>.
    /// </summary>
    public static ModelDescription FromModel(Model model, IModelPackage? package = null)
    {
        ArgumentNullException.ThrowIfNull(model);

        var identity = package?.Identity ?? ModelPackageIdentity.CreateProvisional(
            model.IsGguf ? ModelFormat.Gguf : ModelFormat.SafeTensors,
            [new ModelPackageComponentIdentity(ModelPackageRoles.PrimaryWeights, null, model.ModelPath)]);

        if (model.IsGguf)
        {
            return FromGgufModel(identity, model.ModelPath, model.Gguf, ModelFormat.Gguf, package);
        }

        return FromTensorSource(identity, model.ModelPath, model.TensorSource, ModelFormat.SafeTensors, package);
    }

    private static ModelDescription FromGgufModel(
        ModelPackageIdentity identity,
        string primaryPath,
        GgufModel gguf,
        ModelFormat format,
        IModelPackage? package)
    {
        var hp = ModelHyperparams.FromGgufMetadata(gguf.Metadata, gguf);
        string? rawArch = gguf.Metadata.TryGetValue("general.architecture", out var a) ? Convert.ToString(a) : null;

        var probe = new ArchitectureProbe
        {
            Path = primaryPath,
            Architecture = rawArch,
            TensorSource = gguf,
            Hyperparams = hp,
            IsGguf = true,
            Gguf = gguf,
        };

        bool resolved = ArchitectureRegistry.TryResolve(probe, out var descriptor);
        string arch = resolved ? descriptor.Id : (rawArch ?? "unknown");
        ForwardPassFamily family = descriptor?.ForwardPassFamily ?? (hp.IsHybridSsm ? ForwardPassFamily.HybridGdn : ForwardPassFamily.Dense);

        var facts = BuildPlanningFacts(gguf, hp, rawArch);
        var capabilities = BuildCapabilitySummary(family, hp, package, gguf);
        var resources = BuildResourceSummary(gguf, hp, facts);
        var semantics = BuildSemanticDescription(identity, primaryPath, arch, family, format, probe, descriptor, package, facts);

        return new ModelDescription(identity, semantics, capabilities, resources, facts, primaryPath);
    }

    private static ModelDescription FromTensorSource(
        ModelPackageIdentity identity,
        string primaryPath,
        IModelTensorSource tensorSource,
        ModelFormat format,
        IModelPackage? package)
    {
        var hp = ModelHyperparams.FromGgufMetadata(tensorSource.Metadata, tensorSource);
        string? rawArch = tensorSource.Metadata.TryGetValue("general.architecture", out var a) ? Convert.ToString(a) : null;

        var probe = new ArchitectureProbe
        {
            Path = primaryPath,
            Architecture = rawArch,
            TensorSource = tensorSource,
            Hyperparams = hp,
            IsGguf = false,
            Gguf = null,
        };

        bool resolved = ArchitectureRegistry.TryResolve(probe, out var descriptor);
        string arch = resolved ? descriptor.Id : (rawArch ?? "unknown");
        ForwardPassFamily family = descriptor?.ForwardPassFamily ?? ForwardPassFamily.Dense;

        var facts = BuildPlanningFactsFromSource(tensorSource, hp, rawArch);
        var capabilities = BuildCapabilitySummary(family, hp, package, null);
        var resources = BuildResourceSummaryFromSource(tensorSource, hp, facts);
        var semantics = BuildSemanticDescription(identity, primaryPath, arch, family, format, probe, descriptor, package, facts);

        return new ModelDescription(identity, semantics, capabilities, resources, facts, primaryPath);
    }

    private static ModelPlanningFacts BuildPlanningFacts(GgufModel gguf, ModelHyperparams hp, string? rawArch)
    {
        var tokenEmbd = gguf.FindTensor("token_embd.weight");
        var output = gguf.FindTensor("output.weight");

        bool cpuFixedWeights = ShouldKeepFixedWeightsOnCpu(tokenEmbd, output);
        long embGpuBytes = cpuFixedWeights ? 0 : MeasureGpuEmbeddingBytes(tokenEmbd);
        long outputGpuBytes = cpuFixedWeights
            ? 0
            : (output is not null ? EstimateGpuTensorBytes(output.Value) : embGpuBytes);

        var perLayerGpuBuilder = ImmutableArray.CreateBuilder<long>(hp.NumLayers);
        var perLayerCpuBuilder = ImmutableArray.CreateBuilder<long>(hp.NumLayers);
        long totalGpuWeightBytes = embGpuBytes + outputGpuBytes;
        long totalCpuWeightBytes = (tokenEmbd?.ByteSize ?? 0) + (output?.ByteSize ?? 0);

        for (int i = 0; i < hp.NumLayers; i++)
        {
            long gpuBytes = MeasureLayerGpuBytes(gguf, hp, i);
            long cpuBytes = MeasureLayerCpuBytes(gguf, hp, i);
            perLayerGpuBuilder.Add(gpuBytes);
            perLayerCpuBuilder.Add(cpuBytes);
            totalGpuWeightBytes += gpuBytes;
            totalCpuWeightBytes += cpuBytes;
        }

        long moeRoutedExpertGpuBytes = hp.IsMoE ? MeasureMoeRoutedExpertsGpuBytes(gguf, hp) : 0;

        int headDim = hp.HeadDim;
        long scratchBytes = (long)(hp.EmbeddingDim * 3 + hp.NumHeads * headDim
            + hp.NumKvHeads * headDim * 2 + hp.NumHeads * headDim
            + Math.Max(hp.IntermediateDim, hp.IsMoE ? hp.ExpertIntermediateDim : hp.IntermediateDim) * 2
            + hp.VocabSize + (hp.IsMoE ? hp.NumExperts + hp.EmbeddingDim * 2 : 0)) * sizeof(float);

        bool hasMlaTensors = gguf.FindTensor("blk.0.attn_kv_b.weight") is not null ||
                             gguf.FindTensor("blk.0.attn_k_norm.weight") is not null;

        var tensorNames = ImmutableArray.CreateRange(gguf.Tensors.Select(t => t.Name));
        DType primaryDType = DeterminePrimaryDType(gguf);

        return new ModelPlanningFacts
        {
            NumLayers = hp.NumLayers,
            ContextLength = hp.ContextLength,
            EmbeddingDim = hp.EmbeddingDim,
            HeadDim = hp.HeadDim,
            NumHeads = hp.NumHeads,
            NumKvHeads = hp.NumKvHeads,
            IntermediateDim = hp.IntermediateDim,
            VocabSize = hp.VocabSize,
            RmsNormEps = hp.RmsNormEps,
            RopeTheta = hp.RopeTheta,
            RopeScale = hp.RopeYarnFactor,
            RopeDim = hp.RopeDim,
            IsMoE = hp.IsMoE,
            NumExperts = hp.NumExperts,
            NumActiveExperts = hp.NumActiveExperts,
            ExpertIntermediateDim = hp.ExpertIntermediateDim,
            HasSharedExpert = hp.HasSharedExpert,
            KvLoraRank = hp.KvLoraRank,
            HasMlaTensors = hasMlaTensors,
            IsHybridSsm = hp.IsHybridSsm,
            HasHybridGdnLayers = hp.IsHybridSsm,
            HasCpuHybridGdnPass = hp.IsHybridSsm,
            IsGemma4 = hp.LayerHeadDim is not null || string.Equals(rawArch, "gemma4", StringComparison.OrdinalIgnoreCase),
            HasLayerHeadDim = hp.LayerHeadDim is not null,
            HasAttnBias = hp.HasAttnBias,
            HasQkNorm = hp.HasQkNorm,
            UseL2QkNorm = hp.UseL2QkNorm,
            PrimaryWeightDType = primaryDType,
            EmbeddingDType = tokenEmbd?.DType ?? DType.Float32,
            OutputDType = output?.DType,
            EmbeddingGpuBytes = embGpuBytes,
            OutputGpuBytes = outputGpuBytes,
            ShouldKeepFixedWeightsOnCpu = cpuFixedWeights,
            PerLayerGpuWeightBytes = perLayerGpuBuilder.MoveToImmutable(),
            PerLayerCpuWeightBytes = perLayerCpuBuilder.MoveToImmutable(),
            TotalGpuWeightBytes = totalGpuWeightBytes,
            TotalCpuWeightBytes = totalCpuWeightBytes,
            ScratchBytes = scratchBytes,
            MoeRoutedExpertGpuBytes = moeRoutedExpertGpuBytes,
            TensorNames = tensorNames,
            SwaWindowSizes = hp.IsSwaLayer != null ? ImmutableArray.CreateRange(hp.IsSwaLayer.Select(b => b ? 1 : 0)) : null,
            KvSourceLayers = hp.KvSourceLayer != null ? ImmutableArray.CreateRange(hp.KvSourceLayer) : null,
            LayerHeadDims = hp.LayerHeadDim != null ? ImmutableArray.CreateRange(hp.LayerHeadDim) : null,
        };
    }

    private static ModelPlanningFacts BuildPlanningFactsFromSource(IModelTensorSource source, ModelHyperparams hp, string? rawArch)
    {
        var tokenEmbd = source.FindTensor("token_embd.weight") ?? source.FindTensor("model.embed_tokens.weight");
        var output = source.FindTensor("output.weight") ?? source.FindTensor("lm_head.weight");

        long embBytes = tokenEmbd?.ByteSize ?? 0;
        long outputBytes = output?.ByteSize ?? embBytes;

        var perLayerCpuBuilder = ImmutableArray.CreateBuilder<long>(hp.NumLayers);
        for (int i = 0; i < hp.NumLayers; i++)
        {
            perLayerCpuBuilder.Add(0);
        }

        int headDim = hp.HeadDim;
        long scratchBytes = (long)(hp.EmbeddingDim * 3 + hp.NumHeads * headDim
            + hp.NumKvHeads * headDim * 2 + hp.NumHeads * headDim
            + Math.Max(hp.IntermediateDim, hp.IsMoE ? hp.ExpertIntermediateDim : hp.IntermediateDim) * 2
            + hp.VocabSize) * sizeof(float);

        var tensorNames = ImmutableArray.CreateRange(source.Tensors.Select(t => t.Name));
        DType primaryDType = tokenEmbd?.DType ?? DType.Float32;

        return new ModelPlanningFacts
        {
            NumLayers = hp.NumLayers,
            ContextLength = hp.ContextLength,
            EmbeddingDim = hp.EmbeddingDim,
            HeadDim = hp.HeadDim,
            NumHeads = hp.NumHeads,
            NumKvHeads = hp.NumKvHeads,
            IntermediateDim = hp.IntermediateDim,
            VocabSize = hp.VocabSize,
            RmsNormEps = hp.RmsNormEps,
            RopeTheta = hp.RopeTheta,
            RopeScale = hp.RopeYarnFactor,
            RopeDim = hp.RopeDim,
            IsMoE = hp.IsMoE,
            NumExperts = hp.NumExperts,
            NumActiveExperts = hp.NumActiveExperts,
            ExpertIntermediateDim = hp.ExpertIntermediateDim,
            HasSharedExpert = hp.HasSharedExpert,
            KvLoraRank = hp.KvLoraRank,
            HasMlaTensors = false,
            IsHybridSsm = hp.IsHybridSsm,
            HasHybridGdnLayers = hp.IsHybridSsm,
            HasCpuHybridGdnPass = hp.IsHybridSsm,
            IsGemma4 = hp.LayerHeadDim is not null || string.Equals(rawArch, "gemma4", StringComparison.OrdinalIgnoreCase),
            HasLayerHeadDim = hp.LayerHeadDim is not null,
            HasAttnBias = hp.HasAttnBias,
            HasQkNorm = hp.HasQkNorm,
            UseL2QkNorm = hp.UseL2QkNorm,
            PrimaryWeightDType = primaryDType,
            EmbeddingDType = tokenEmbd?.DType ?? DType.Float32,
            OutputDType = output?.DType,
            EmbeddingGpuBytes = embBytes,
            OutputGpuBytes = outputBytes,
            ShouldKeepFixedWeightsOnCpu = true, // SafeTensors runs on CPU
            PerLayerGpuWeightBytes = perLayerCpuBuilder.ToImmutable(),
            PerLayerCpuWeightBytes = perLayerCpuBuilder.MoveToImmutable(),
            TotalGpuWeightBytes = 0,
            TotalCpuWeightBytes = embBytes + outputBytes,
            ScratchBytes = scratchBytes,
            MoeRoutedExpertGpuBytes = 0,
            TensorNames = tensorNames,
            SwaWindowSizes = null,
            KvSourceLayers = null,
            LayerHeadDims = null,
        };
    }

    private static ModelCapabilitySummary BuildCapabilitySummary(
        ForwardPassFamily family,
        ModelHyperparams hp,
        IModelPackage? package,
        GgufModel? gguf)
    {
        bool isRwkv = family == ForwardPassFamily.Rwkv;
        bool isSafeTensors = package?.Format == ModelFormat.SafeTensors;
        bool isGptOss = family == ForwardPassFamily.GptOss;

        bool supportsContinuousBatching = !isRwkv && !isSafeTensors;
        bool supportsSpeculation = !isRwkv && !hp.IsHybridSsm && !isGptOss && !isSafeTensors;

        bool supportsVision = package?.Components.Any(c => c.Role == ModelPackageRoles.VisionProjector) == true
            || (gguf != null && (gguf.FindTensor("mm.0.weight") is not null || gguf.FindTensor("v.patch_embd.weight") is not null));

        string stateModel = isRwkv
            ? "recurrent_rwkv"
            : (hp.IsHybridSsm ? "recurrent_gdn" : "kv_cache");

        return new ModelCapabilitySummary(
            SupportsContinuousBatching: supportsContinuousBatching,
            SupportsSpeculation: supportsSpeculation,
            SupportsVision: supportsVision,
            SupportsEmbeddingInput: true,
            SupportsTools: true,
            StateModel: stateModel,
            SupportsHiddenTaps: true,
            SupportsVulkan: !isSafeTensors,
            SupportsCuda: !isSafeTensors,
            SupportsCpu: true);
    }

    private static ModelResourceSummary BuildResourceSummary(GgufModel gguf, ModelHyperparams hp, ModelPlanningFacts facts)
    {
        long totalWeightBytes = gguf.Tensors.Sum(t => t.ByteSize);
        long estimatedContextBytes = (long)hp.NumLayers * hp.NumKvHeads * hp.HeadDim * 2 * hp.ContextLength * sizeof(float);

        return new ModelResourceSummary(
            TotalWeightBytes: totalWeightBytes,
            EstimatedContextBytes: estimatedContextBytes,
            PrimaryDType: facts.PrimaryWeightDType,
            LayerCount: hp.NumLayers,
            ContextLimit: hp.ContextLength,
            EmbeddingBytes: facts.EmbeddingGpuBytes,
            OutputBytes: facts.OutputGpuBytes);
    }

    private static ModelResourceSummary BuildResourceSummaryFromSource(IModelTensorSource source, ModelHyperparams hp, ModelPlanningFacts facts)
    {
        long totalWeightBytes = source.Tensors.Sum(t => t.ByteSize);
        long estimatedContextBytes = (long)hp.NumLayers * hp.NumKvHeads * hp.HeadDim * 2 * hp.ContextLength * sizeof(float);

        return new ModelResourceSummary(
            TotalWeightBytes: totalWeightBytes,
            EstimatedContextBytes: estimatedContextBytes,
            PrimaryDType: facts.PrimaryWeightDType,
            LayerCount: hp.NumLayers,
            ContextLimit: hp.ContextLength,
            EmbeddingBytes: facts.EmbeddingGpuBytes,
            OutputBytes: facts.OutputGpuBytes);
    }

    private static ModelSemanticDescription BuildSemanticDescription(
        ModelPackageIdentity identity,
        string primaryPath,
        string arch,
        ForwardPassFamily family,
        ModelFormat format,
        ArchitectureProbe probe,
        ArchitectureDescriptor? descriptor,
        IModelPackage? package,
        ModelPlanningFacts facts)
    {
        string modelName = probe.GetMetadataString("general.name")
            ?? Path.GetFileNameWithoutExtension(primaryPath);

        long paramCount = probe.GetMetadataInt64("general.parameter_count") ?? 0;
        if (paramCount == 0 && probe.Gguf is not null)
        {
            paramCount = probe.Gguf.Tensors.Sum(t => (long)t.ElementCount);
        }

        string? sidecarSemanticFamily = package?.SidecarMetadata?.SemanticFamily;

        return new ModelSemanticDescription(
            Architecture: arch,
            Family: family,
            ModelName: modelName,
            ParameterCount: paramCount,
            Format: format,
            QuantizationDescription: facts.PrimaryWeightDType.ToString(),
            SemanticFamily: descriptor?.Id ?? sidecarSemanticFamily,
            StateModel: facts.IsHybridSsm ? "recurrent_gdn" : (family == ForwardPassFamily.Rwkv ? "recurrent_rwkv" : "kv_cache"));
    }

    private static DType DeterminePrimaryDType(GgufModel gguf)
    {
        var blk0 = gguf.FindTensor("blk.0.attn_q.weight") ?? gguf.FindTensor("token_embd.weight");
        if (blk0 is not null)
            return blk0.Value.DType;

        return gguf.Tensors.FirstOrDefault().DType;
    }

    private static bool ShouldKeepFixedWeightsOnCpu(GgufTensorInfo? embedding, GgufTensorInfo? output)
    {
        if (embedding is null) return false;
        const long maxStorageBufferBytes = 2L * 1024 * 1024 * 1024 - 1;
        long embBytes = embedding.Value.DType == DType.Q4_K
            ? (embedding.Value.ByteSize + 3) & ~3L
            : embedding.Value.ElementCount * sizeof(float);
        if (embBytes > maxStorageBufferBytes)
            return true;
        if (output is not null && EstimateGpuTensorBytes(output.Value) > maxStorageBufferBytes)
            return true;
        return false;
    }

    private static long MeasureGpuEmbeddingBytes(GgufTensorInfo? info)
    {
        if (info is null) return 0;
        var dt = info.Value.DType;
        if (dt == DType.Q4_K || dt == DType.Q8_0 || dt == DType.Q6_K)
            return (info.Value.ByteSize + 3) & ~3L;
        return info.Value.ElementCount * sizeof(float);
    }

    private static long EstimateGpuTensorBytes(GgufTensorInfo tensor)
    {
        switch (tensor.DType)
        {
            case DType.Float32:
            case DType.Q4_0:
            case DType.Q4_K:
            case DType.Q5_K:
            case DType.Q6_K:
            case DType.Q8_0:
                return (tensor.ByteSize + 3) & ~3L;
            default:
                return tensor.ElementCount * sizeof(float);
        }
    }

    private static long MeasureLayerGpuBytes(GgufModel model, ModelHyperparams hp, int layer)
    {
        long total = 0;
        string[] suffixes = hp.IsMoE
            ?
            [
                "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
                "attn_output.weight", "ffn_norm.weight", "ffn_gate_inp.weight"
            ]
            :
            [
                "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
                "attn_output.weight", "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight",
                "ffn_down.weight"
            ];

        foreach (var suffix in suffixes)
        {
            var info = model.FindTensor($"blk.{layer}.{suffix}");
            if (info is not null)
                total += EstimateGpuTensorBytes(info.Value);
        }

        if (hp.IsMoE && hp.HasSharedExpert)
        {
            foreach (var name in new[] { "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight" })
            {
                var info = model.FindTensor($"blk.{layer}.{name}");
                if (info is not null)
                    total += EstimateGpuTensorBytes(info.Value);
            }
        }

        if (hp.HasAttnBias)
        {
            foreach (var name in new[] { "attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_output.bias" })
            {
                var info = model.FindTensor($"blk.{layer}.{name}");
                if (info is not null)
                    total += EstimateGpuTensorBytes(info.Value);
            }
        }

        if (hp.HasQkNorm && !hp.UseL2QkNorm)
        {
            foreach (var name in new[] { "attn_q_norm.weight", "attn_k_norm.weight" })
            {
                var info = model.FindTensor($"blk.{layer}.{name}");
                if (info is not null)
                    total += EstimateGpuTensorBytes(info.Value);
            }
        }

        return total;
    }

    private static long MeasureLayerCpuBytes(GgufModel model, ModelHyperparams hp, int layer)
    {
        long total = 0;
        string prefix = $"blk.{layer}.";
        foreach (var tensor in model.Tensors)
        {
            if (tensor.Name.StartsWith(prefix, StringComparison.Ordinal))
            {
                total += tensor.ByteSize;
            }
        }
        return total;
    }

    private static long MeasureMoeRoutedExpertsGpuBytes(GgufModel model, ModelHyperparams hp)
    {
        long total = 0;
        for (int i = 0; i < hp.NumLayers; i++)
        {
            foreach (var name in new[] { "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight" })
            {
                var info = model.FindTensor($"blk.{i}.{name}");
                if (info is not null)
                    total += EstimateGpuTensorBytes(info.Value);
            }
        }
        return total;
    }

    /// <summary>
    /// Constructs a <see cref="ForwardPassRequest"/> purely from this <see cref="ModelDescription"/>
    /// and user execution options without reopening disk files.
    /// </summary>
    public ForwardPassRequest CreateForwardPassRequest(
        ForwardPassFrontend frontend = ForwardPassFrontend.Cli,
        ForwardPassBackend backend = ForwardPassBackend.Auto,
        int gpuLayers = 0,
        int plannedGpuLayers = -1,
        bool cudaAvailable = false,
        bool turboQuant = false,
        string turboQuantMode = "auto",
        bool hasDraftModel = false,
        bool draftLookup = false,
        bool isContinuousBatching = false,
        int promptTokenCount = 0,
        int targetContextLength = 0,
        bool allowUnverifiedArchitecture = false,
        bool unsupportedGpuPath = false,
        bool unsupportedPartialCudaPath = false,
        bool unsupportedPartialVulkanPath = false,
        string? unsupportedBackendName = null)
    {
        return new ForwardPassRequest
        {
            Frontend = frontend,
            Architecture = Semantics.Architecture,
            UnsupportedBackendName = unsupportedBackendName,
            IsSafeTensors = Semantics.Format == ModelFormat.SafeTensors,
            PackageSupported = true,
            IsHybridSsm = PlanningFacts.IsHybridSsm,
            HasHybridGdnLayers = PlanningFacts.HasHybridGdnLayers,
            HasCpuHybridGdnPass = PlanningFacts.HasCpuHybridGdnPass,
            IsMoE = PlanningFacts.IsMoE,
            KvLoraRank = PlanningFacts.KvLoraRank,
            HasMlaTensors = PlanningFacts.HasMlaTensors,
            ArchitectureSupported = ArchitectureRegistry.Find(Semantics.Architecture) is not null,
            AllowUnverifiedArchitecture = allowUnverifiedArchitecture,
            NumLayers = PlanningFacts.NumLayers,
            GpuLayers = gpuLayers,
            PlannedGpuLayers = plannedGpuLayers,
            Backend = backend,
            CudaAvailable = cudaAvailable,
            UnsupportedGpuPath = unsupportedGpuPath,
            UnsupportedPartialCudaPath = unsupportedPartialCudaPath,
            UnsupportedPartialVulkanPath = unsupportedPartialVulkanPath,
            LayerHeadDim = PlanningFacts.HasLayerHeadDim,
            TurboQuant = turboQuant,
            TurboQuantMode = turboQuantMode,
            HasDraftModel = hasDraftModel,
            DraftLookup = draftLookup,
            IsContinuousBatching = isContinuousBatching,
            SupportsContinuousBatching = Capabilities.SupportsContinuousBatching,
            SupportsEmbeddingInput = Capabilities.SupportsEmbeddingInput,
            SupportsHiddenTaps = Capabilities.SupportsHiddenTaps,
            PromptTokenCount = promptTokenCount,
            TargetContextLength = targetContextLength > 0 ? targetContextLength : PlanningFacts.ContextLength,
            HeadDim = PlanningFacts.HeadDim,
            IsGemma4 = PlanningFacts.IsGemma4,
            ModelLayers = PlanningFacts.NumLayers,
            ModelHiddenSize = PlanningFacts.EmbeddingDim,
            ModelVocabSize = PlanningFacts.VocabSize,
        };
    }
}

/// <summary>Semantic model attributes (architecture, family, naming, format).</summary>
public sealed record ModelSemanticDescription(
    string Architecture,
    ForwardPassFamily Family,
    string? ModelName,
    long ParameterCount,
    ModelFormat Format,
    string? QuantizationDescription = null,
    string? SemanticFamily = null,
    string? StateModel = null);

/// <summary>Declared and derived model execution capabilities.</summary>
public sealed record ModelCapabilitySummary(
    bool SupportsContinuousBatching,
    bool SupportsSpeculation,
    bool SupportsVision,
    bool SupportsEmbeddingInput,
    bool SupportsTools,
    string StateModel,
    bool SupportsHiddenTaps = true,
    bool SupportsVulkan = true,
    bool SupportsCuda = true,
    bool SupportsCpu = true);

/// <summary>Summary of model storage and memory requirements.</summary>
public sealed record ModelResourceSummary(
    long TotalWeightBytes,
    long EstimatedContextBytes,
    DType PrimaryDType,
    int LayerCount,
    int ContextLimit,
    long EmbeddingBytes,
    long OutputBytes);

/// <summary>
/// Intrinsic immutable planning facts required by TierPlanner and ForwardPassSelection
/// without reopening model files.
/// </summary>
public sealed record ModelPlanningFacts
{
    public required int NumLayers { get; init; }
    public required int ContextLength { get; init; }
    public required int EmbeddingDim { get; init; }
    public required int HeadDim { get; init; }
    public required int NumHeads { get; init; }
    public required int NumKvHeads { get; init; }
    public required int IntermediateDim { get; init; }
    public required int VocabSize { get; init; }
    public float RmsNormEps { get; init; } = 1e-5f;
    public float RopeTheta { get; init; } = 10_000f;
    public float RopeScale { get; init; } = 1.0f;
    public int RopeDim { get; init; }
    public bool IsMoE { get; init; }
    public int NumExperts { get; init; }
    public int NumActiveExperts { get; init; }
    public int ExpertIntermediateDim { get; init; }
    public bool HasSharedExpert { get; init; }
    public int KvLoraRank { get; init; }
    public bool HasMlaTensors { get; init; }
    public bool IsHybridSsm { get; init; }
    public bool HasHybridGdnLayers { get; init; }
    public bool HasCpuHybridGdnPass { get; init; }
    public bool IsGemma4 { get; init; }
    public bool HasLayerHeadDim { get; init; }
    public bool HasAttnBias { get; init; }
    public bool HasQkNorm { get; init; }
    public bool UseL2QkNorm { get; init; }
    public DType PrimaryWeightDType { get; init; } = DType.Float32;
    public DType EmbeddingDType { get; init; } = DType.Float32;
    public DType? OutputDType { get; init; }
    public long EmbeddingGpuBytes { get; init; }
    public long OutputGpuBytes { get; init; }
    public bool ShouldKeepFixedWeightsOnCpu { get; init; }
    public ImmutableArray<long> PerLayerGpuWeightBytes { get; init; } = [];
    public ImmutableArray<long> PerLayerCpuWeightBytes { get; init; } = [];
    public long TotalGpuWeightBytes { get; init; }
    public long TotalCpuWeightBytes { get; init; }
    public long ScratchBytes { get; init; }
    public long MoeRoutedExpertGpuBytes { get; init; }
    public ImmutableArray<string> TensorNames { get; init; } = [];
    public ImmutableArray<int>? SwaWindowSizes { get; init; }
    public ImmutableArray<int>? KvSourceLayers { get; init; }
    public ImmutableArray<int>? LayerHeadDims { get; init; }
}
