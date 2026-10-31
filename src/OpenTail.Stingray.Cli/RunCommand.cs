using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine.Packaging;
using OpenTail.Stingray.Engine.Planning;
using OpenTail.Stingray.Engine.Runtime;

namespace OpenTail.Stingray.Cli;

/// <summary>
/// Main inference command. Parameter names match llama-cli where applicable.
/// Usage: stingray -m model.gguf -p "Hello" -n 128 --temp 0.7
/// </summary>
public sealed partial class RunCommand : Command<RunCommand.Settings>
{

    /// <summary>What the argument-handling prologue decided, handed to the model-loading phase.</summary>
    /// <param name="ResolvedPlan">The <c>--auto</c> execution plan, if one was built.</param>
    /// <param name="GpuDeviceIndex">Resolved <c>--device</c> index (-1 = auto).</param>
    /// <param name="DeviceNone">True when <c>--device none/cpu</c> forces the CPU path.</param>
    /// <param name="EffNGpuLayers">GPU layers after applying device, explicit flag and plan precedence.</param>
    /// <param name="ModelPath">The model file or package directory to load (explicit, or the default search).</param>
    private sealed record RunPrologue(ExecutionPlan? ResolvedPlan, int GpuDeviceIndex, bool DeviceNone, int EffNGpuLayers, string ModelPath);

    /// <summary>
    /// Everything <c>run</c> does with its arguments before a model is touched: <c>--auto</c> planning, <c>--file</c>
    /// prompt loading, thread/BLAS/device settings, the MoE and KV-cache environment overrides, and locating the model.
    /// Extracted verbatim from the head of <see cref="Execute"/> so that method is the model-loading and generation
    /// phases only; <c>RunCommandPrologueTests</c> pins the early-exit behaviour. Returns false with an exit code when
    /// the arguments cannot be honoured.
    /// </summary>
    private static bool TryPrepare(Settings settings, out RunPrologue prologue, out int exitCode)
    {
        ExecutionPlan? resolvedPlan = null;
        if (settings.Explain && !settings.Auto)
        {
            AnsiConsole.ErrorLine("[red]Error:[/] --explain requires --auto so the displayed plan is the plan that will execute.");
            { prologue = null!; exitCode = ExitCodes.Failure; return false; }
        }

        // --file/-f (llama.cpp): load the prompt from a file. Overrides -p; lets prompts exceed
        // the shell command-line length limit. Read as-is (no trailing-newline stripping).
        if (settings.PromptFile is { Length: > 0 } promptFile)
        {
            if (!File.Exists(promptFile))
            {
                AnsiConsole.ErrorLine($"[red]Prompt file not found:[/] {Markup.Escape(promptFile)}");
                { prologue = null!; exitCode = ExitCodes.Failure; return false; }
            }
            // Read failures (locked file, permissions, bad path) should fail loud + clean, not
            // throw a stack trace; Escape the message since paths can carry Spectre markup chars.
            try
            {
                settings.Prompt = File.ReadAllText(promptFile);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                          or System.Security.SecurityException or NotSupportedException)
            {
                AnsiConsole.ErrorLine($"[red]Error reading prompt file:[/] {Markup.Escape(ex.Message)}");
                { prologue = null!; exitCode = ExitCodes.Failure; return false; }
            }
        }

        if (settings.MinBatchBlas > 0)
            SimdKernels.MinBatchForBlas = settings.MinBatchBlas;

        // -t/--threads must be applied before any forward pass is constructed, since the SIMD
        // kernels capture the parallel options. An explicit flag beats STINGRAY_CPU_THREADS.
        if (settings.Threads > 0)
            SimdKernels.CpuThreads = settings.Threads;

        // Inert llama.cpp flags: accepted so common command lines run, but they do nothing here.
        // Warned rather than refused — a warning is not silent, so this does not breach the
        // "never accept and ignore" rule in docs/llamacpp-onramp-plan.md.
        if (settings.FlashAttn)
            AnsiConsole.MarkupLine("[yellow]Note:[/] -fa/--flash-attn has no effect: attention is already fused in the OpenTail backends.");
        if (settings.NoWarmup)
            AnsiConsole.MarkupLine("[yellow]Note:[/] --no-warmup has no effect: OpenTail has no separate warmup step.");

        // Resolve --device before any GPU call (it may set CUDA_VISIBLE_DEVICES, which the CUDA
        // driver only reads at first init; Vulkan takes the index explicitly below). `--device none`
        // forces the CPU path, overriding --n-gpu-layers.
        int gpuDeviceIndex;
        bool deviceNone;
        try
        {
            gpuDeviceIndex = GpuDevice.Resolve(settings.Device, out deviceNone);
        }
        catch (InvalidOperationException ex)
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] {Markup.Escape(ex.Message)}");
            { prologue = null!; exitCode = ExitCodes.Failure; return false; }
        }
        // `is > 0`, not `!= 0`. NGpuLayers became int? so the planner could tell "unset" from an
        // explicit "-g 0"; under the old int/default-0 shape `!= 0` meant "user asked for GPU
        // layers", but with a nullable, unset is null and `null != 0` is TRUE — which made this
        // note fire on every `--device none` run, claiming to have overridden a flag the user
        // never passed.
        if (deviceNone && settings.NGpuLayers is > 0 or -1)
            AnsiConsole.MarkupLine("[yellow]Note:[/] --device none overrides --ngl/-g; running on CPU.");
        // Precedence: --device none > explicit --ngl/-g > ExecutionPlan (only under --auto) > CPU.
        //
        // The null test is load-bearing. NGpuLayers is int?, so "unset" is null and "-g 0" is an
        // EXPLICIT request for CPU-only. Testing `== 0` (as the first wiring did) inverted this:
        // null == 0 is false, so the plan was ignored in the default case and applied only when the
        // user had explicitly asked for CPU — overriding the one value they were most deliberate
        // about, which Phase 1's "explicit pins are never silently overridden" criterion forbids.
        //
        // --auto alone gates planning. --goal is a parameter OF the plan, not a second trigger for
        // it; letting `--goal quality` silently enable placement changes would make an opt-in
        // feature reachable without opting in.
        int effNGpuLayers = deviceNone
            ? 0
            : settings.NGpuLayers ?? (settings.Auto && resolvedPlan is not null ? resolvedPlan.GpuLayers : 0);

        // MoE expert-execution knobs, KV dtype and the device pin are NOT applied here: they travel in the
        // ExecutionRequest -> ExecutionPlan, and RuntimeInstance applies them. Only argument validation stays.
        if (!TryValidateCpuMoeFlags(settings.NCpuMoe, out string? cpuMoeError))
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] {Markup.Escape(cpuMoeError!)}");
            { prologue = null!; exitCode = ExitCodes.Failure; return false; }
        }

        var modelPath = settings.ModelPath;
        if (modelPath is null)
        {
            foreach (var candidate in new[] { "models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf", "model.gguf" })
                if (File.Exists(candidate) || Directory.Exists(candidate)) { modelPath = candidate; break; }
        }
        if (modelPath is null || (!File.Exists(modelPath) && !Directory.Exists(modelPath)))
        {
            AnsiConsole.ErrorLine("[red]Error:[/] No model file or package directory found. Use [yellow]-m <path>[/]");
            { prologue = null!; exitCode = ExitCodes.Failure; return false; }
        }

        bool isPackage = Directory.Exists(modelPath)
            || modelPath.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)
            || modelPath.EndsWith(".safetensors.index.json", StringComparison.OrdinalIgnoreCase);

        if (isPackage)
        {
            var pkgReport = ModelPackageInspector.Inspect(modelPath);
            if (!pkgReport.IsSupported)
            {
                AnsiConsole.ErrorLine("[red]Error:[/] SafeTensors package not supported:");
                foreach (var r in pkgReport.Rejections)
                    AnsiConsole.ErrorLine($"  [red]·[/] {Markup.Escape(r.Detail)}");
                AnsiConsole.ErrorLine("[dim]GGUF is the recommended deployment format for quantized models.[/]");
                { prologue = null!; exitCode = ExitCodes.Failure; return false; }
            }
        }

        if (!modelPath.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
        {
            try
            {
                int? pinGpuLayers = deviceNone ? 0 : (settings.NGpuLayers ?? (settings.Auto ? null : 0));
                string? pinBackend = deviceNone ? "cpu" : settings.Backend;
                int? pinCtxSize = settings.CtxSize > 0 ? settings.CtxSize : null;

                IModelPackage package = isPackage
                    ? SafeTensorsModelPackage.Open(modelPath)
                    : (modelPath.EndsWith(".json", StringComparison.OrdinalIgnoreCase)
                        ? OllamaModelPackage.Open(modelPath)
                        : LooseGgufModelPackage.Open(modelPath));

                var modelDesc = ModelDescription.FromPackage(package);
                var capabilities = BackendCapabilities.Detect(noGpuProbe: false, deviceIndex: gpuDeviceIndex);
                var request = new ExecutionRequest
                {
                    ModelPath = modelPath,
                    Goal = settings.Goal ?? "balanced",
                    PinnedBackend = pinBackend,
                    PinnedGpuLayers = pinGpuLayers,
                    PinnedContextSize = pinCtxSize,
                    PinnedKvDtype = settings.KvType,
                    TurboQuant = settings.TurboQuant,
                    TurboQuantMode = settings.TqModeStr ?? "auto",
                    DraftModelPath = settings.DraftModelPath,
                    DSparkModelPath = settings.DSparkModelPath,
                    MmprojPath = settings.MmprojPath,
                    ThreadCount = settings.Threads,
                    AllowUnverifiedArchitecture = settings.AllowUnverifiedArch,
                    DeviceIndex = gpuDeviceIndex,
                    DSparkPlace = settings.DSparkPlaceStr,
                    CpuMoe = settings.CpuMoe ? true : null,
                    GpuMoePrefill = settings.GpuMoePrefill,
                    MoeWarmPin = settings.MoeWarmPin,
                    MoeWarmPinAfter = settings.MoeWarmPinAfter > 0 ? (int)settings.MoeWarmPinAfter : null,
                    MoePredictPrefetch = settings.NoMoePredictPrefetch ? false : null,
                    ExpertStatsPath = string.IsNullOrEmpty(settings.ExpertStatsPath) ? null : settings.ExpertStatsPath,
                };

                request = ExecutionRequestEnvironment.ApplyTo(request);
                resolvedPlan = ExecutionPlanner.Plan(modelDesc, request, capabilities);

                if (settings.Auto)
                {
                    Console.WriteLine(resolvedPlan.CompactSummary());

                    if (settings.Explain)
                    {
                        Console.WriteLine("\n[ExecutionPlan Decision Trace]");
                        foreach (var d in resolvedPlan.Decisions)
                        {
                            Console.WriteLine($"  - [{d.Code}] {d.SelectedValue} ({d.Reason} - {d.Source})");
                        }
                        Console.WriteLine();
                    }
                }
            }
            catch (Exception ex)
            {
                if (ex is NotSupportedException or PlanNotExecutableException or FileNotFoundException or InvalidOperationException)
                {
                    AnsiConsole.ErrorLine($"[red]Error:[/] {Markup.Escape(ex.Message)}");
                    { prologue = null!; exitCode = ExitCodes.Failure; return false; }
                }
                throw;
            }
        }

        effNGpuLayers = resolvedPlan?.GpuLayers ?? effNGpuLayers;

        // Memory check before anything is loaded, same evaluator as scout/chat/setup. CPU runs only: GPU placement is not estimated.
        // Only a KNOWN over-budget estimate stops the run; an unknown estimate warns and proceeds.
        if (effNGpuLayers == 0)
        {
            int preflightCtx = settings.CtxSize > 0 ? settings.CtxSize : (resolvedPlan?.ContextSize ?? 2048);
            var preflight = OpenTail.Stingray.Engine.Scout.LoadPreflight.EvaluateFile(modelPath, preflightCtx);
            if (!OpenTail.Stingray.Engine.Scout.LoadPreflight.ShouldProceed(preflight, settings.IgnorePreflight, out string? preflightNote))
            {
                AnsiConsole.ErrorLine("[red]Not loading:[/] " + Markup.Escape(preflightNote!));
                { prologue = null!; exitCode = ExitCodes.Failure; return false; }
            }
            if (preflightNote is not null) AnsiConsole.MarkupLine("[dim]" + Markup.Escape(preflightNote) + "[/]");
        }

        prologue = new RunPrologue(resolvedPlan, gpuDeviceIndex, deviceNone, effNGpuLayers, modelPath);
        exitCode = ExitCodes.Success;
        return true;
    }

    /// <summary>
    /// <c>run -m model.onnx</c>: a graph smoke test, not text generation. Loads the ONNX graph, feeds it the prompt's characters as
    /// dummy token ids (plus zero embeddings, a mask and positions, and empty past-KV tensors where the graph asks for them),
    /// runs it once and reports the output shapes. Self-contained: shares no state with the generation modes.
    /// </summary>
    private static int RunOnnxGraph(Settings settings, string modelPath)
    {
        using var onnxSession = OnnxModelSession.TryLoad(modelPath);
        if (onnxSession == null)
        {
            AnsiConsole.ErrorLine("[red]Error:[/] Could not load ONNX model file. Ensure onnxruntime.dll is available.");
            return 1;
        }

        AnsiConsole.MarkupLine($"[bold]{Markup.Escape(Path.GetFileName(modelPath))}[/] (ONNX Graph)");
        AnsiConsole.MarkupLine($"  [cyan]Inputs ({onnxSession.InputNames.Count}):[/] {string.Join(", ", onnxSession.InputNames)}");
        AnsiConsole.MarkupLine($"  [cyan]Outputs ({onnxSession.OutputNames.Count}):[/] {string.Join(", ", onnxSession.OutputNames)}");

        string prompt = settings.Prompt ?? "Hello world";
        AnsiConsole.MarkupLine($"\n[dim]Executing ONNX Graph with prompt:[/] {Markup.Escape(prompt)}");

        long[] inputIds = prompt.Select(c => (long)c).ToArray();
        long[] mask = new long[inputIds.Length];
        Array.Fill(mask, 1L);
        long[] posIds = Enumerable.Range(0, inputIds.Length).Select(i => (long)i).ToArray();

        var inputList = new List<(string Name, Array Data, int[] Shape)>
        {
            ("input_ids", inputIds, [1, inputIds.Length]),
            ("inputs_embeds", new float[inputIds.Length * 1024], [1, inputIds.Length, 1024]),
            ("attention_mask", mask, [1, inputIds.Length]),
            ("position_ids", posIds, [1, inputIds.Length])
        };

        foreach (var inName in onnxSession.InputNames)
        {
            if (inName.StartsWith("past_key_values"))
            {
                inputList.Add((inName, Array.Empty<float>(), [1, 16, 0, 64]));
            }
        }

        var swOnnx = Stopwatch.StartNew();
        var outputs = onnxSession.Run(inputList.ToArray());
        swOnnx.Stop();

        AnsiConsole.MarkupLine($"\n[green]ONNX Graph Executed in {swOnnx.ElapsedMilliseconds}ms[/]");
        foreach (var kv in outputs)
        {
            AnsiConsole.MarkupLine($"  Output [bold]{kv.Key}[/]: {kv.Value.Length} elements");
        }
        return 0;
    }
    /// <summary>
    /// The loaded engine state the generation modes read, bundled so a mode can live in its own method instead of in the middle
    /// of <see cref="Execute"/>. Built once the backend is configured; modes treat it as read-only.
    /// </summary>
    private sealed record RunEngine(
        GgufModel? Model, ModelHyperparams Hp, GgufTokenizer Tokenizer, int CtxSize, int NGpuLayers, int GpuDeviceIndex,
        ForwardPass? Fwd, HybridGdnForwardPass? HybridFwd, IForwardPass? StandaloneFwd, IForwardPass? MtpFwd,
        IDisposable? GpuBackend, IDisposable? GpuFwd,
        Func<int, int, ReadOnlySpan<float>> Forward, Func<IReadOnlyList<int>, ReadOnlySpan<float>> Prefill, Action ResetCache,
        ExecutionPlan Plan);

    protected override int Execute(Settings settings, CancellationToken cancellation)
    {
        if (!TryPrepare(settings, out var prologue, out int prologueExit))
            return prologueExit;
        ExecutionPlan? resolvedPlan = prologue.ResolvedPlan;
        int gpuDeviceIndex = prologue.GpuDeviceIndex;
        bool deviceNone = prologue.DeviceNone;
        int effNGpuLayers = prologue.EffNGpuLayers;
        string modelPath = prologue.ModelPath;

        AnsiConsole.MarkupLine($"[dim]Loading model:[/] {modelPath}");
        var sw = Stopwatch.StartNew();

        // Pre-declare shared variables so the goto from the SafeTensors branch is valid.
        // In the GGUF path these are assigned in their original locations below, unchanged.
        ModelHyperparams hp;
        GgufTokenizer tokenizer;
        int ctxSize;
        int nGpuLayers;
        // Not a `using var`: the SafeTensors branch jumps past this declaration, and C# forbids a
        // goto that skips a using declaration (CS8648). Disposed explicitly in the finally below.
        GgufModel? model = null;
        ForwardPass? fwd = null;
        HybridGdnForwardPass? hybridFwd = null;
        // Architecture-specific CPU forward pass (gpt-oss: sinks, SWA, biased MoE, OAI SwiGLU, YaRN;
        // rwkv6/rwkv7: recurrent, no KV cache) — the generic ForwardPass models none of these.
        IForwardPass? standaloneFwd = null;
        IForwardPass? mtpFwd = null;
        IDisposable? gpuBackend = null;
        IDisposable? gpuFwd = null;
        // Tracks SafetensorsTensorSource lifetime in the package path; null in the GGUF path.
        // Disposed alongside fwd at every cleanup site below.
        SafetensorsTensorSource? stTensorSource = null;
        // Shared by both paths, and not a `using var` for the same CS8648 reason as `model`.
        CpuBackend? cpuBackend = null;
        RuntimeInstance? runtimeInstance = null;
        Func<int, int, ReadOnlySpan<float>> forward;
        Func<IReadOnlyList<int>, ReadOnlySpan<float>> prefill;
        Action resetCache;

        // ── ONNX model branch ────────────────────────────────────────────────
        if (modelPath.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
            return RunOnnxGraph(settings, modelPath);

        // ── Plan-driven runtime execution (§5 of plan) ───────────────────────
        // TryPrepare always resolves a plan for every non-ONNX model; there is no second selection path.
        if (resolvedPlan is null)
            throw new InvalidOperationException("TryPrepare did not produce an ExecutionPlan for a non-ONNX model.");
        {
            var modelParams = new ModelParams(modelPath)
            {
                Backend = resolvedPlan.Backend,
                GpuLayerCount = resolvedPlan.GpuLayers
            };
            var modelObj = Model.Load(modelParams);
            runtimeInstance = RuntimeInstance.Create(resolvedPlan, modelObj);

            hp = modelObj.Hyperparams;
            s_arch = resolvedPlan.Provenance?.TargetArchitecture ?? modelObj.Architecture;
            s_speculation = resolvedPlan.Tuning?.Speculation ?? new SpeculationSettings();
            ctxSize = resolvedPlan.ContextSize;
            tokenizer = (runtimeInstance.Tokenizer as GgufTokenizer)
                ?? (modelObj.IsGguf ? GgufTokenizer.FromGgufModel(modelObj.Gguf) : null)
                ?? throw new InvalidOperationException($"Model '{modelPath}' does not have a valid tokenizer.");
            s_jinja = tokenizer.ChatTemplate;
            s_hasLlama3Headers = tokenizer.SpecialTokens.ContainsKey("<|start_header_id|>");
            (s_thinkTokenId, s_endThinkTokenId) = tokenizer.ReasoningTokens;
            s_noThinking = ResolveThinkingOff(s_arch, settings.Thinking, settings.NoThinking);

            nGpuLayers = resolvedPlan.GpuLayers;
            gpuDeviceIndex = prologue.GpuDeviceIndex;
            model = modelObj.IsGguf ? modelObj.Gguf : null;
            stTensorSource = !modelObj.IsGguf ? modelObj.TensorSource as SafetensorsTensorSource : null;
            fwd = runtimeInstance.ForwardPass as ForwardPass;
            hybridFwd = runtimeInstance.ForwardPass as HybridGdnForwardPass;
            standaloneFwd = runtimeInstance.ForwardPass;
            // Found 2026-10-08: the plan-driven migration (dd198b80) dropped the old per-backend "if (HasMtpHead) mtpFwd = ..." assignments,
            // so --spec-type mtp was always rejected. Any pass that reports an MTP head (CPU, CUDA or Vulkan hybrid GDN) drives MTP.
            if (runtimeInstance.ForwardPass is { HasMtpHead: true } mtpCapable) mtpFwd = mtpCapable;
            gpuBackend = (IDisposable?)runtimeInstance.VulkanBackend ?? runtimeInstance.CudaBackend;
            gpuFwd = runtimeInstance.ForwardPass as IDisposable;
            cpuBackend = runtimeInstance.CpuBackend;

            forward = runtimeInstance.ForwardPass.Forward;
            prefill = tokens => runtimeInstance.ForwardPass.Prefill(tokens);
            resetCache = runtimeInstance.ForwardPass.ResetCache;
        }

        int activeContextLength = (gpuFwd as IForwardPass)?.MaxSeqLen
            ?? (fwd as IForwardPass)?.MaxSeqLen
            ?? (hybridFwd as IForwardPass)?.MaxSeqLen
            ?? hp.ContextLength;
        AnsiConsole.MarkupLine($"[dim]Model loaded in {sw.Elapsed.TotalSeconds:F1}s — " +
            $"{hp.NumLayers}L, {hp.EmbeddingDim}d, headDim={hp.HeadDim}, {hp.VocabSize} vocab, ctx={activeContextLength}[/]");

        // Tools, JSON schema, prompt/template overrides, logit bias and the sampling parameters: see TryBuildSampling.
        if (!TryBuildSampling(settings, tokenizer, out var sp, out var rng, out int samplingExit))
            return samplingExit;

        // Speculative decoding (--draft-model / --draft-lookup): see TryRunSpeculative.
        var engine = new RunEngine(model, hp, tokenizer, ctxSize, nGpuLayers, gpuDeviceIndex, fwd, hybridFwd, standaloneFwd, mtpFwd,
                                   gpuBackend, gpuFwd, forward, prefill, resetCache, resolvedPlan);
        if (TryRunSpeculative(settings, engine, sp, rng) is int specExit)
            return specExit;

        // DSpark block-speculative decoding (--dspark-model): see TryRunDSpark.
        if (TryRunDSpark(settings, engine, sp) is int dsparkExit)
            return dsparkExit;

        try
        {
            IForwardPass activeForwardPass = (gpuFwd as IForwardPass) ?? (fwd as IForwardPass) ?? (hybridFwd as IForwardPass) ?? standaloneFwd
                ?? throw new InvalidOperationException("No forward pass was configured.");
            if (settings.ImagePaths is { Length: > 0 })
                return RunImagePrompt(settings, activeForwardPass, tokenizer, hp, sp, rng,
                    activeForwardPass.MaxSeqLen);
            if (settings.Prompt is not null)
                return RunSinglePrompt(settings, forward, prefill, tokenizer, sp, rng, mtpFwd,
                    activeForwardPass.MaxSeqLen);
            return RunInteractive(settings, forward, prefill, resetCache, tokenizer, sp, rng, mtpFwd,
                activeForwardPass.MaxSeqLen);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex);
            return 1;
        }
        finally
        {
            runtimeInstance.Model?.Dispose();
            runtimeInstance.Dispose();
        }
    }

    /// <summary>
    /// Converts CLI pins into the planner's narrower input surface. Kept separate so the
    /// precedence rules are testable without loading a model or initializing a backend.
    /// </summary>
    internal static AutoPlanInputs ResolveAutoPlanInputs(Settings settings)
    {
        bool deviceNone = string.Equals(settings.Device, "none", StringComparison.OrdinalIgnoreCase);
        return new(
            deviceNone ? "cpu" : settings.Backend,
            deviceNone ? 0 : settings.NGpuLayers,
            settings.CtxSize > 0 ? settings.CtxSize : null);
    }

    internal sealed record AutoPlanInputs(string? Backend, int? GpuLayers, int? ContextSize);

    internal static string s_arch = "qwen2"; // set during model load
    // Speculation switches from the resolved ExecutionPlan (CLI hosts exactly one runtime per process).
    private static SpeculationSettings s_speculation = new();
    internal static bool s_hasLlama3Headers;
    // Effective "thinking off" state: --no-thinking OR a model whose recommended config
    // disables reasoning (Gemma 4 E4B-it is not a reasoning model). Set during model load.
    private static bool s_noThinking;
    private static int s_thinkTokenId = -1;    // <think> token for any model using the <think>/</think> special-token convention
    private static int s_endThinkTokenId = -1; // </think> token for any model using the <think>/</think> special-token convention
    internal static JinjaChatTemplate? s_jinja;  // parsed from GGUF tokenizer.chat_template
    // Tool definitions loaded from --tools (template-facing object graph: a list of
    // {type, function:{…}} dicts), rendered into the chat template's `tools` variable. Null
    // unless --tools was given, in which case the prompt advertises no tools (legacy behaviour).
    private static IReadOnlyList<object?>? s_tools;
}
