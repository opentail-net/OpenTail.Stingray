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
    public static LoadedEngine LoadFromPlan(ExecutionPlan plan, string? sessionStorageDirectory = null)
    {
        ArgumentNullException.ThrowIfNull(plan);
        var detectedFormat = DetectModelFormat(plan.ModelPath);
        if (plan.ModelFormat != detectedFormat)
            throw new InvalidOperationException(
                $"Execution plan declares model format '{plan.ModelFormat}', but '{plan.ModelPath}' resolves as '{detectedFormat}'.");

        // Memory check before the weights are touched, same evaluator as scout/chat/run. CPU-only GGUF plans only (GPU placement is not
        // estimated). Only a KNOWN over-budget estimate refuses; STINGRAY_IGNORE_PREFLIGHT=1 loads anyway.
        if (plan.GpuLayers == 0 && detectedFormat == ModelFormat.Gguf)
        {
            var preflight = OpenTail.Stingray.Engine.Scout.LoadPreflight.EvaluateFile(plan.ModelPath, plan.ContextSize);
            bool ignore = Environment.GetEnvironmentVariable("STINGRAY_IGNORE_PREFLIGHT") == "1";
            if (!OpenTail.Stingray.Engine.Scout.LoadPreflight.ShouldProceed(preflight, ignore, out string? note))
                throw new InvalidOperationException("Not loading: " + note + " (set STINGRAY_IGNORE_PREFLIGHT=1 to override)");
        }

        var model = Model.Load(new ModelParams(plan.ModelPath)
        {
            Backend = plan.Backend,
            GpuLayerCount = plan.GpuLayers
        });

        var instance = RuntimeInstance.Create(plan, model);

        HotSessionRuntime? sessionRuntime = null;
        ColdSessionRuntime? coldSessionRuntime = null;
        if (plan.Batching?.EnableSessions == true && instance.Engine is ContinuousBatchingEngine batchingEngine)
        {
            sessionRuntime = new HotSessionRuntime(batchingEngine, instance.Tokenizer);
            if (!string.IsNullOrWhiteSpace(sessionStorageDirectory))
            {
                coldSessionRuntime = new ColdSessionRuntime(sessionRuntime, batchingEngine,
                    sessionStorageDirectory, plan.ModelFormat);
            }
        }

        string targetArch = plan.Provenance?.TargetArchitecture ?? model.Architecture;

        if (!string.IsNullOrWhiteSpace(plan.Modality?.MmprojPath) && instance.Engine is InferenceEngine ieVision)
        {
            if (ArchitectureRegistry.Find(targetArch)?.SupportsImageInput != true)
                throw new InvalidOperationException(
                    "Image input (MmprojPath / STINGRAY_MMPROJ) is only supported for architectures that declare image input (currently Gemma 4, gemma4uv) " +
                    $"text models; this model's architecture is '{targetArch}'.");
            if (!instance.ForwardPass.SupportsEmbeddingInput)
                throw new InvalidOperationException(
                    "MmprojPath / STINGRAY_MMPROJ is set but image input requires a forward pass that accepts " +
                    "precomputed-embedding input: CPU (NGpuLayers=0) or full CUDA offload (NGpuLayers=-1) of a " +
                    $"Gemma 4 model that fits VRAM. The configured pass ({instance.ForwardPass.GetType().Name}) does not support it.");
            if (plan.Batching?.MaxBatchSize > 1 && instance.Engine is ContinuousBatchingEngine)
                throw new InvalidOperationException(
                    "Image input is not supported with continuous batching (MaxBatchSize > 1). Set MaxBatchSize=1.");

            var mmprojPath = ResolvePath(plan.Modality.MmprojPath, "mmproj projector", "STINGRAY_MMPROJ", "MmprojPath");
            var visionModel = VisionModel.Open(mmprojPath);
            var visionEmbedder = new GemmaUvVisionEmbedder(visionModel);
            int imgOpen = instance.Tokenizer.SpecialTokens.TryGetValue("<|image>", out var o) ? o : 255999;
            int imgClose = instance.Tokenizer.SpecialTokens.TryGetValue("<image|>", out var c) ? c : 258882;
            int imgPlaceholder = instance.Tokenizer.SpecialTokens.TryGetValue("<|image|>", out var p) ? p : 258880;
            ieVision.EnableImageInput(visionEmbedder, visionModel, imgOpen, imgClose, imgPlaceholder);
        }

        if (plan.Speculation is { DSparkEnabled: true } && !string.IsNullOrWhiteSpace(plan.Speculation.DSparkModelPath) && instance.Engine is InferenceEngine ieDspark && model.IsGguf)
        {
            AttachDSpark(ieDspark, instance.ForwardPass, plan.Speculation, instance.OwnedDisposables.ToList());
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
            ? new OwnedDisposableEngine(cbe, instance)
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
    /// the runtime via <see cref="LoadFromPlan(ExecutionPlan, string)"/>.
    /// </summary>
    public static LoadedEngine Load(OpenTailStingrayServerOptions opts)
    {
        if (opts.MinBatchBlas > 0)
            SimdKernels.MinBatchForBlas = opts.MinBatchBlas;
        if (opts.CpuThreads > 0)
            SimdKernels.CpuThreads = opts.CpuThreads;

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
            EnableSessions = opts.EnableSessions,
            DSparkModelPath = !string.IsNullOrWhiteSpace(opts.DSparkModelPath)
                ? opts.DSparkModelPath
                : Environment.GetEnvironmentVariable("STINGRAY_DSPARK_MODEL"),
            MmprojPath = opts.MmprojPath,
            DSparkPlace = opts.DSparkPlace,
            DSparkRequired = true,   // the server never silently drops a configured head
            CpuMoe = opts.CpuMoe,
            GpuMoePrefill = opts.GpuMoePrefill,
            MoeWarmPin = opts.MoeWarmPin,
            MoeWarmPinAfter = opts.MoeWarmPinAfter > 0 ? (int)opts.MoeWarmPinAfter : null,
            MoePredictPrefetch = opts.MoePredictPrefetch ? null : false,
            ExpertStatsPath = string.IsNullOrEmpty(opts.ExpertStatsPath) ? null : opts.ExpertStatsPath,
        };

        var plan = ExecutionPlanner.Plan(modelDesc, ExecutionRequestEnvironment.ApplyTo(request), capabilities);
        return LoadFromPlan(plan, opts.SessionStorageDirectory);
    }

    // ── DSpark draft head (docs/dspark-plan.md Phase 6, PR #413) ─────────────
    /// <summary>
    /// Execution-only: loads the draft head and attaches it exactly as the plan recorded. Placement (GPU/CPU/off)
    /// was decided by <see cref="ExecutionPlanner"/>; nothing here consults TierPlanner or DSparkPlacementPlanner,
    /// and a plan that cannot be honoured fails loudly instead of being re-planned.
    /// </summary>
    private static void AttachDSpark(InferenceEngine ie, IForwardPass fwd, SpeculationPlan spec, List<IDisposable> owned)
    {
        if (!spec.DSparkEnabled || spec.DSparkPlacement == DSparkPlacement.Off)
            throw new InvalidOperationException(
                "DSpark was configured (STINGRAY_DSPARK_MODEL / DSparkModelPath) but placement " +
                $"resolved to Off — {spec.DSparkPlacementReason}. Free resources, pass DSparkPlace=cpu/gpu " +
                "explicitly, or unset the head.");
        if (!fwd.SupportsHiddenTaps)
            throw new InvalidOperationException(
                "DSpark requires a tap-capable dense forward pass (CPU, NGpuLayers=0, or full " +
                "CUDA offload, NGpuLayers=-1; no MoE / Gemma-4 / TurboQuant / SnapKV). " +
                $"The configured pass ({fwd.GetType().Name}) can't capture hidden taps.");

        string stPath = spec.DSparkModelPath!;
        if (Directory.Exists(stPath)) stPath = Path.Combine(stPath, "model.safetensors");
        string cfgPath = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(stPath))!, "config.json");
        var cfg = DSparkConfig.FromJsonFile(cfgPath);

        CudaBackend? cuda = null;
        if (spec.DSparkPlacement == DSparkPlacement.Gpu)
        {
            foreach (var d in owned)
                if (d is CudaBackend cb) { cuda = cb; break; }
            if (fwd is not CudaForwardPass || cuda is null)
                throw new InvalidOperationException(
                    "The plan places the DSpark draft on GPU but the runtime has no full-CUDA target backend.");
        }

        Console.Error.WriteLine($"[InferenceEngine] DSpark placement: {spec.DSparkPlacement} — {spec.DSparkPlacementReason}");
        fwd.EnableHiddenTaps(cfg.TargetLayerIds);
        using var st = SafetensorsLoader.Open(stPath);
        IDSparkDraft draft = spec.DSparkPlacement == DSparkPlacement.Gpu
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
            $"({spec.DSparkPlacement}) from {stPath}");
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

internal sealed class OwnedDisposableEngine(IInferenceEngine inner, RuntimeInstance owner)
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

    // RuntimeInstance is the single owner: it disposes the engine, applies the drain guard, then frees backends once.
    public void Dispose() => owner.Dispose();
}
