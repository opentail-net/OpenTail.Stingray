#nullable enable

using OpenTail.Stingray.Core;
using OpenTail.Stingray.Core.Grammar;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Cuda;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Packaging;
using OpenTail.Stingray.Engine.Planning;
using OpenTail.Stingray.Engine.Runtime;
using OpenTail.Stingray.Sessions;
using OpenTail.Stingray.Vision;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Server;

public static class InferenceEngineLoader
{
    /// <summary>
    /// Constructs an inference engine directly from an immutable <see cref="ExecutionPlan"/> (§5.3 of plan).
    /// Guarantees that runtime execution consumes the plan rather than rediscovering execution policy.
    /// </summary>
    public static LoadedEngine LoadFromPlan(ExecutionPlan plan, OpenTailStingrayServerOptions? baseOptions = null)
    {
        ArgumentNullException.ThrowIfNull(plan);
        var detectedFormat = DetectModelFormat(plan.ModelPath);
        if (plan.ModelFormat != detectedFormat)
            throw new InvalidOperationException(
                $"Execution plan declares model format '{plan.ModelFormat}', but '{plan.ModelPath}' resolves as '{detectedFormat}'.");

        var model = Model.Load(new ModelParams(plan.ModelPath)
        {
            Backend = plan.Backend,
            GpuLayerCount = plan.GpuLayers
        });

        var instance = RuntimeInstance.Create(plan, model);

        HotSessionRuntime? sessionRuntime = null;
        ColdSessionRuntime? coldSessionRuntime = null;
        if (baseOptions?.EnableSessions == true && instance.Engine is ContinuousBatchingEngine batchingEngine)
        {
            sessionRuntime = new HotSessionRuntime(batchingEngine, instance.Tokenizer);
            if (!string.IsNullOrWhiteSpace(baseOptions.SessionStorageDirectory))
            {
                coldSessionRuntime = new ColdSessionRuntime(sessionRuntime, batchingEngine,
                    baseOptions.SessionStorageDirectory, plan.ModelFormat);
            }
        }

        string targetArch = plan.Provenance?.TargetArchitecture ?? model.Architecture;

        if (!string.IsNullOrWhiteSpace(baseOptions?.MmprojPath) && instance.Engine is InferenceEngine ieVision)
        {
            if (!string.Equals(targetArch, "gemma4", StringComparison.Ordinal))
                throw new InvalidOperationException(
                    "Image input (MmprojPath / STINGRAY_MMPROJ) is only supported for Gemma 4 (gemma4uv) " +
                    $"text models; this model's architecture is '{targetArch}'.");
            if (!instance.ForwardPass.SupportsEmbeddingInput)
                throw new InvalidOperationException(
                    "MmprojPath / STINGRAY_MMPROJ is set but image input requires a forward pass that accepts " +
                    "precomputed-embedding input: CPU (NGpuLayers=0) or full CUDA offload (NGpuLayers=-1) of a " +
                    $"Gemma 4 model that fits VRAM. The configured pass ({instance.ForwardPass.GetType().Name}) does not support it.");
            if (baseOptions.MaxBatchSize > 1 && instance.Engine is ContinuousBatchingEngine)
                throw new InvalidOperationException(
                    "Image input is not supported with continuous batching (MaxBatchSize > 1). Set MaxBatchSize=1.");

            var mmprojPath = ResolvePath(baseOptions.MmprojPath, "mmproj projector", "STINGRAY_MMPROJ", "MmprojPath");
            var visionModel = VisionModel.Open(mmprojPath);
            var visionEmbedder = new GemmaUvVisionEmbedder(visionModel);
            int imgOpen = instance.Tokenizer.SpecialTokens.TryGetValue("<|image>", out var o) ? o : 255999;
            int imgClose = instance.Tokenizer.SpecialTokens.TryGetValue("<image|>", out var c) ? c : 258882;
            int imgPlaceholder = instance.Tokenizer.SpecialTokens.TryGetValue("<|image|>", out var p) ? p : 258880;
            ieVision.EnableImageInput(visionEmbedder, visionModel, imgOpen, imgClose, imgPlaceholder);
        }

        if (!string.IsNullOrWhiteSpace(baseOptions?.DSparkModelPath) && instance.Engine is InferenceEngine ieDspark && model.IsGguf)
        {
            AttachDSpark(ieDspark, instance.ForwardPass, model.Gguf, model.Hyperparams, instance.OwnedDisposables.ToList(), baseOptions, baseOptions.DSparkModelPath, plan.ContextSize);
        }

        var chatTemplate = (instance.Tokenizer as GgufTokenizer)?.ChatTemplate;
        var grammarVocab = new GrammarVocabulary(instance.Tokenizer);
        var toolBoundaryMarkers = ToolCallAdapterRegistry.Get(targetArch).ToolBoundaryStopMarkers;
        var toolBoundaryStopTokenIds = toolBoundaryMarkers
            .Select(m => instance.Tokenizer.SpecialTokens.TryGetValue(m, out int id) ? id : -1)
            .Where(id => id > 0)
            .Distinct()
            .ToArray();

        var runtimeRes = DescribeRuntime(instance.ForwardPass, plan.ModelFormat);
        long? gpuWeightBytes = plan.Placement?.GpuWeightBytes;
        var engine = instance.Engine is ContinuousBatchingEngine cbe
            ? new OwnedDisposableEngine(cbe, instance.OwnedDisposables.ToList())
            : instance.Engine;

        return new LoadedEngine(
            engine,
            targetArch,
            chatTemplate,
            toolBoundaryStopTokenIds,
            grammarVocab,
            instance.Tokenizer,
            sessionRuntime,
            coldSessionRuntime,
            instance.ForwardPass is ForwardPass cpuFwd ? cpuFwd.GetBatchedPrefillCapability() : null,
            runtimeRes,
            gpuWeightBytes,
            instance);
    }

    private static ModelFormat DetectModelFormat(string modelPath) =>
        Directory.Exists(modelPath)
        || modelPath.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)
        || modelPath.EndsWith(".safetensors.index.json", StringComparison.OrdinalIgnoreCase)
            ? ModelFormat.SafeTensors
            : ModelFormat.Gguf;

    /// <summary>
    /// Opens the model referenced by <paramref name="opts"/>, resolves execution policy via
    /// <see cref="ExecutionPlanner"/> into an immutable <see cref="ExecutionPlan"/>, and constructs
    /// the runtime via <see cref="LoadFromPlan(ExecutionPlan, OpenTailStingrayServerOptions)"/>.
    /// </summary>
    public static LoadedEngine Load(OpenTailStingrayServerOptions opts)
    {
        ApplyMoeEnvironment(opts);

        if (opts.MinBatchBlas > 0)
            SimdKernels.MinBatchForBlas = opts.MinBatchBlas;
        if (opts.CpuThreads > 0)
            SimdKernels.CpuThreads = opts.CpuThreads;

        if (!string.IsNullOrWhiteSpace(opts.KvType))
            Environment.SetEnvironmentVariable("STINGRAY_KV_DTYPE", opts.KvType);

        var modelPath = ResolvePath(opts.ModelPath, "model", "STINGRAY_MODEL", "ModelPath");

        bool isPackage = Directory.Exists(modelPath)
            || modelPath.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)
            || modelPath.EndsWith(".safetensors.index.json", StringComparison.OrdinalIgnoreCase);

        if (opts.EnableSessions && isPackage)
            throw new InvalidOperationException(
                "EnableSessions currently supports only the proven CPU-dense GGUF lane; " +
                "SafeTensors session/cache conformance is not available yet.");

        IModelPackage package = isPackage
            ? SafeTensorsModelPackage.Open(modelPath)
            : LooseGgufModelPackage.Open(modelPath);

        var modelDesc = ModelDescription.FromPackage(package);
        var capabilities = BackendCapabilities.Detect();

        var request = new ExecutionRequest
        {
            ModelPath = modelPath,
            Goal = "balanced",
            PinnedBackend = opts.Backend switch
            {
                ServerBackend.Cuda => "cuda",
                ServerBackend.Vulkan => "vulkan",
                ServerBackend.Cpu => "cpu",
                _ => "auto"
            },
            PinnedGpuLayers = opts.NGpuLayers != -1 ? opts.NGpuLayers : (int?)null,
            PinnedContextSize = opts.ContextSize > 0 ? opts.ContextSize : null,
            PinnedKvDtype = opts.KvType,
            TurboQuant = opts.TurboQuant,
            TurboQuantMode = opts.TqMode ?? "auto",
            FlashAttention = true,
            ThreadCount = opts.CpuThreads,
            BatchingMode = (opts.MaxBatchSize > 1 || opts.EnableSessions) ? BatchingMode.Continuous : BatchingMode.Sequential,
            MaxBatchSize = opts.MaxBatchSize > 0 ? opts.MaxBatchSize : 1,
            DraftModelPath = null,
            DSparkModelPath = opts.DSparkModelPath,
            SnapKvEnabled = SnapKvConfig.FromEnvironment().Enabled,
            SnapKvBudget = SnapKvConfig.FromEnvironment().Budget,
        };

        var plan = ExecutionPlanner.Plan(modelDesc, request, capabilities);
        return LoadFromPlan(plan, opts);
    }

    // ── DSpark draft head (docs/dspark-plan.md Phase 6, PR #413) ─────────────

    private static void AttachDSpark(InferenceEngine ie, IForwardPass fwd, GgufModel model,
        ModelHyperparams hp, List<IDisposable> owned, OpenTailStingrayServerOptions opts,
        string configuredPath, int ctxSize)
    {
        string stPath = configuredPath;
        if (Directory.Exists(stPath)) stPath = Path.Combine(stPath, "model.safetensors");
        if (!File.Exists(stPath))
            throw new FileNotFoundException($"DSpark model not found: {stPath}");
        string cfgPath = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(stPath))!, "config.json");
        if (!File.Exists(cfgPath))
            throw new FileNotFoundException($"DSpark config.json not found next to the safetensors: {cfgPath}");

        var cfg = DSparkConfig.FromJsonFile(cfgPath);
        if (cfg.VocabSize != hp.VocabSize || cfg.NumTargetLayers != hp.NumLayers
            || cfg.HiddenSize != hp.EmbeddingDim)
            throw new InvalidOperationException(
                $"DSpark head/target mismatch — head expects vocab {cfg.VocabSize}, " +
                $"{cfg.NumTargetLayers} target layers, hidden {cfg.HiddenSize}; target has " +
                $"vocab {hp.VocabSize}, {hp.NumLayers} layers, hidden {hp.EmbeddingDim}.");
        if (!fwd.SupportsHiddenTaps)
            throw new InvalidOperationException(
                "DSpark requires a tap-capable dense forward pass (CPU, NGpuLayers=0, or full " +
                "CUDA offload, NGpuLayers=-1; no MoE / Gemma-4 / TurboQuant / SnapKV). " +
                $"The configured pass ({fwd.GetType().Name}) can't capture hidden taps.");

        CudaBackend? cuda = null;
        if (fwd is CudaForwardPass)
            foreach (var d in owned)
                if (d is CudaBackend cb) { cuda = cb; break; }

        var userPlace = !string.IsNullOrWhiteSpace(opts.DSparkPlace)
            ? DSparkPlacementPlanner.ParsePlacement(opts.DSparkPlace)
            : DSparkPlacementPlanner.ResolvePlacement(null);
        var hwProfile = cuda is not null ? HardwareProfile.Detect(cuda) : HardwareProfile.Detect();
        var targetPlacement = TierPlanner.Plan(model, hp, hwProfile, requestedCtxSize: ctxSize);
        long headBytesGpu = CudaDSparkDraftModel.EstimateGpuResidentBytes(cfg);
        long headBytesCpu = DSparkDraftModel.EstimateResidentBytes(cfg);
        long tapBytes = (long)targetPlacement.RecommendedCtxSize * cfg.TapDim * sizeof(float);
        var decision = DSparkPlacementPlanner.Plan(
            hwProfile, targetPlacement, headBytesGpu, headBytesCpu, userPlace,
            hostTapBytes: tapBytes);

        if (decision.Placement == DSparkPlacement.Gpu && cuda is null)
        {
            // Gpu → Cpu → Off graceful fallback: re-plan in Auto over a GPU-less
            // profile so the RAM budget is actually checked.
            decision = DSparkPlacementPlanner.Plan(
                hwProfile with { VramBytes = 0 }, targetPlacement,
                headBytesGpu, headBytesCpu, DSparkPlacement.Auto,
                hostTapBytes: tapBytes);
        }
        Console.Error.WriteLine($"[InferenceEngine] DSpark placement: {decision.Placement} — {decision.Reason}");
        if (decision.Placement == DSparkPlacement.Off)
            throw new InvalidOperationException(
                $"DSpark was configured (STINGRAY_DSPARK_MODEL / DSparkModelPath) but placement " +
                $"resolved to Off — {decision.Reason}. Free resources, pass DSparkPlace=cpu/gpu " +
                "explicitly, or unset the head.");

        fwd.EnableHiddenTaps(cfg.TargetLayerIds);
        using var st = SafetensorsLoader.Open(stPath);
        IDSparkDraft draft = decision.Placement == DSparkPlacement.Gpu
            ? new CudaDSparkDraftModel(cfg, st, cuda!, fwd.MaxSeqLen)
            : new DSparkDraftModel(cfg, st, fwd.MaxSeqLen);
        try
        {
            ie.AttachDSparkDraft(draft);
        }
        catch
        {
            draft.Dispose();
            throw;
        }
        Console.Error.WriteLine(
            $"[InferenceEngine] DSpark draft attached: {cfg.NumLayers}L block-{cfg.BlockSize} " +
            $"({decision.Placement}) from {stPath}");
    }

    private static ServerRuntimeResolution DescribeRuntime(IForwardPass forwardPass, ModelFormat format)
    {
        (string backend, string route) = forwardPass switch
        {
            ForwardPass => ("cpu", "cpu-dense"),
            HybridGdnForwardPass => ("cpu", "cpu-hybrid-gdn"),
            CudaForwardPass => ("cuda", "cuda-full"),
            CudaHybridForwardPass => ("cuda", "cuda-hybrid"),
            CudaHybridGdnForwardPass => ("cuda", "cuda-hybrid-gdn"),
            GpuForwardPass => ("vulkan", "vulkan-full"),
            HybridForwardPass => ("vulkan", "vulkan-hybrid"),
            VulkanHybridGdnForwardPass => ("vulkan", "vulkan-hybrid-gdn"),
            _ => ("unknown", forwardPass.GetType().Name),
        };
        return new ServerRuntimeResolution(backend, route, format.ToString().ToLowerInvariant(), forwardPass.MaxSeqLen);
    }

    private static void ApplyMoeEnvironment(OpenTailStingrayServerOptions opts)
    {
        if (opts.MoeWarmPin is int wp)
            Environment.SetEnvironmentVariable("STINGRAY_MOE_WARMPIN", wp.ToString(System.Globalization.CultureInfo.InvariantCulture));
        if (opts.MoeWarmPinAfter > 0)
            Environment.SetEnvironmentVariable("STINGRAY_MOE_WARMPIN_AFTER", opts.MoeWarmPinAfter.ToString(System.Globalization.CultureInfo.InvariantCulture));
        if (!opts.MoePredictPrefetch)
            Environment.SetEnvironmentVariable("STINGRAY_MOE_PREDICT_PREFETCH", "0");
        if (!string.IsNullOrEmpty(opts.ExpertStatsPath))
            Environment.SetEnvironmentVariable("STINGRAY_EXPERT_STATS", opts.ExpertStatsPath);

        if (opts.CpuMoe is bool cpuMoe)
            Environment.SetEnvironmentVariable("STINGRAY_CPU_MOE", cpuMoe ? "1" : "0");

        if (opts.GpuMoePrefill is bool gpuMoePrefill)
            Environment.SetEnvironmentVariable("STINGRAY_MOE_GPU_PREFILL", gpuMoePrefill ? "1" : "0");
    }

    private static bool PathExists(string p) => File.Exists(p) || Directory.Exists(p);

    private static string ResolvePath(string? path, string what, string envVar, string configKey)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new InvalidOperationException(
                $"OpenTailStingrayServerOptions.{configKey} ({what} path) is required. " +
                $"Set it via Configure(o => o.{configKey} = ...) or the {envVar} environment variable.");

        if (Path.IsPathRooted(path) && PathExists(path))
            return path;

        if (PathExists(path))
            return Path.GetFullPath(path);

        var candidates = new List<string>
        {
            Path.Combine(Directory.GetCurrentDirectory(), path),
            Path.Combine(AppContext.BaseDirectory, path),
        };
        var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        for (int i = 0; i < 5 && dir is not null; i++, dir = dir.Parent)
            candidates.Add(Path.Combine(dir.FullName, path));

        var resolved = candidates.FirstOrDefault(PathExists);
        if (resolved is not null) return resolved;

        throw new InvalidOperationException(
            $"{char.ToUpperInvariant(what[0])}{what[1..]} file not found: '{path}'. " +
            $"Set OpenTailStingrayServerOptions.{configKey}, the {envVar} environment variable, " +
            $"or the OpenTail.Stingray:{configKey} configuration key.");
    }
}

internal sealed class OwnedDisposableEngine(IInferenceEngine inner, IList<IDisposable> owned)
    : IInferenceEngine, IContinuousBatchingObservability, IDisposable
{
    private IContinuousBatchingObservability? Batching => inner as IContinuousBatchingObservability;
    public string ModelId             => inner.ModelId;
    public int QueueDepth             => inner.QueueDepth;
    public int ActiveRequests         => inner.ActiveRequests;
    public bool PrefixCacheEnabled    => inner.PrefixCacheEnabled;
    public long PrefillTokensReused   => inner.PrefillTokensReused;
    public bool IsContinuousBatching => Batching?.IsContinuousBatching ?? false;
    public int PrefillChunkTokens => Batching?.PrefillChunkTokens ?? 0;
    public long KvTokenBudget => Batching?.KvTokenBudget ?? 0;
    public long CommittedKvTokens => Batching?.CommittedKvTokens ?? 0;
    public long PrefixCacheBudgetBytes => Batching?.PrefixCacheBudgetBytes ?? 0;
    public long PrefixCacheUsedBytes => Batching?.PrefixCacheUsedBytes ?? 0;
    public int PrefixCacheEntries => Batching?.PrefixCacheEntries ?? 0;
    public long PrefixCacheHits => Batching?.PrefixCacheHits ?? 0;
    public long PrefixCacheMisses => Batching?.PrefixCacheMisses ?? 0;
    public long PrefixCacheEvictions => Batching?.PrefixCacheEvictions ?? 0;
    public long BatchedArgmaxSteps => Batching?.BatchedArgmaxSteps ?? 0;
    public long BatchedFullLogitsSteps => Batching?.BatchedFullLogitsSteps ?? 0;
    public long BatchedArgmaxSequences => Batching?.BatchedArgmaxSequences ?? 0;
    public long BatchedFullLogitsSequences => Batching?.BatchedFullLogitsSequences ?? 0;

    public IAsyncEnumerable<GenerateChunk> GenerateChunksAsync(
        string prompt, SamplingParams sp, CancellationToken ct = default, string? canonicalHistoryPrefix = null)
        => inner.GenerateChunksAsync(prompt, sp, ct, canonicalHistoryPrefix);

    public void Dispose()
    {
        (inner as IDisposable)?.Dispose();

        if (inner is ContinuousBatchingEngine { DrainedOnDispose: false })
            return;

        for (int i = owned.Count - 1; i >= 0; i--)
        {
            try { owned[i].Dispose(); } catch { /* best-effort teardown */ }
        }
    }
}
