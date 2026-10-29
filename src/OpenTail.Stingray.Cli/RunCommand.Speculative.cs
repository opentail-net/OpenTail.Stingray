using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Cli;

public sealed partial class RunCommand
{
    /// <summary>
    /// Speculative decoding (<c>--draft-model</c> / <c>--draft-lookup</c>). Returns null when speculation was not requested (or was
    /// dropped because a grammar constraint is active), so the caller continues to normal generation; otherwise the run's exit code.
    /// Speculative decoding path (requires --draft-model and --temp 0). Supported
    /// targets: pure CPU (-g 0) and full CUDA offload of a dense model (issue #207 —
    /// packed k-token verify via CudaForwardPass.BatchVerify). Vulkan and the partial-
    /// offload hybrids fall back to normal generation: without a batched verify,
    /// speculation costs k sequential target forwards per step and is never a win.
    /// </summary>
    private static int? TryRunSpeculative(Settings settings, RunEngine e, SamplingParams sp, Random rng)
    {
        var ctxSize = e.CtxSize; var forward = e.Forward; var fwd = e.Fwd; var standaloneFwd = e.StandaloneFwd;
        var gpuBackend = e.GpuBackend; var gpuDeviceIndex = e.GpuDeviceIndex; var gpuFwd = e.GpuFwd;
        var hybridFwd = e.HybridFwd; var model = e.Model; var nGpuLayers = e.NGpuLayers; var tokenizer = e.Tokenizer;

        bool specRequested = settings.DraftModelPath is not null || settings.DraftLookup;
        if (specRequested && sp.Constraint is not null)
        {
            // Any active constraint (--tool-grammar and/or --json-schema/--json-schema-file, issue
            // #423 follow-up) masks one token at a time against running grammar state; a multi-token
            // speculative verify can't honor it. Drop speculation so the constraint actually applies
            // (the standard decode path below reads sp.Constraint).
            AnsiConsole.MarkupLine("[yellow]Warning:[/] --tool-grammar/--json-schema is not applied with speculative decoding (--draft-model/--draft-lookup); generating without speculation so the constraint takes effect.");
            specRequested = false;
        }
        if (specRequested)
        {
            bool cudaSpecTarget = gpuFwd is CudaForwardPass { SupportsBatchVerify: true };
            // Vulkan full-offload of a dense Q4_K/Q6_K model exposes the same weight-amortized
            // BatchVerify (issue #308). gemma4/TurboQuant report SupportsBatchVerify=false (no spec);
            // MoE/bias models report true but stay on the bit-exact K-loop fallback (still lossless,
            // just not weight-amortized). --draft-LOOKUP only on Vulkan; --draft-model (needs a 2nd
            // GpuForwardPass + VRAM mgmt) is a CUDA-only follow-up.
            bool vulkanSpecTarget = gpuFwd is GpuForwardPass { SupportsBatchVerify: true };
            bool gpuSpecTarget = cudaSpecTarget || vulkanSpecTarget;
            // Sampled speculative decoding (issue #178): temp>0 now drives distribution-preserving
            // spec sampling on the model-draft path (greedy at temp 0 stays byte-stable). Gated to
            // model drafts (lookup proposals expose no q), to non-penalized/-biased sampling (draft
            // and target must agree on the distribution), and bypassable via STINGRAY_SPEC_SAMPLE=0.
            bool sampledSpec = settings.Temperature > 0f;
            bool specSampleDisabled = Environment.GetEnvironmentVariable("STINGRAY_SPEC_SAMPLE") == "0";
            bool hasPenalty = sp.RepetitionPenalty != 1f && sp.PreviousTokens is { Count: > 0 };
            bool hasBias = sp.LogitBias is { Count: > 0 };
            if (settings.DraftModelPath is not null && settings.DraftLookup)
            {
                AnsiConsole.ErrorLine("[red]Error:[/] --draft-model and --draft-lookup are mutually exclusive.");
                return 1;
            }
            if (nGpuLayers != 0 && !gpuSpecTarget)
            {
                AnsiConsole.MarkupLine("[yellow]Warning:[/] Speculative decoding requires pure CPU (-g 0), full CUDA offload of a dense or Gemma-4 model, or full Vulkan offload of a dense Q4_K/Q6_K model (--draft-lookup). Falling back to normal generation.");
            }
            else if (sampledSpec && settings.DraftLookup)
            {
                AnsiConsole.MarkupLine("[yellow]Warning:[/] --draft-lookup supports greedy (--temp 0) only; sampled speculative decoding needs --draft-model. Falling back to normal generation.");
            }
            else if (sampledSpec && specSampleDisabled)
            {
                AnsiConsole.MarkupLine("[yellow]Note:[/] STINGRAY_SPEC_SAMPLE=0 — sampled speculative decoding disabled; using normal sampled generation.");
            }
            else if (sampledSpec && (hasPenalty || hasBias))
            {
                AnsiConsole.MarkupLine("[yellow]Warning:[/] sampled speculative decoding does not yet support --repeat-penalty / logit bias (draft and target must share the same distribution); falling back to normal generation.");
            }
            else if (settings.DraftLookup)
            {
                // Prompt-lookup drafting (issue #207): no draft model — proposals come from
                // n-gram matches against prompt + generated history, verified by the same
                // batched-verify step. Floor is ~baseline (no match → plain decode step).
                try
                {
                    // gpuFwd is the CudaForwardPass or GpuForwardPass (both IForwardPass) on a GPU
                    // spec target; fall back to the CPU pass otherwise.
                    IForwardPass lookupTarget = gpuSpecTarget ? (IForwardPass)gpuFwd! : fwd!;
                    AnsiConsole.MarkupLine($"[dim]Speculative decoding: prompt-lookup (n-gram) drafting | Lookahead k={settings.SpecLookahead}[/]");
                    if (settings.Prompt is not null)
                        return RunSpeculativeSinglePrompt(settings, lookupTarget, null, tokenizer, sp, rng);
                    return RunSpeculativeInteractive(settings, lookupTarget, null, tokenizer, sp, rng);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine(ex);
                    return 1;
                }
                finally
                {
                    gpuFwd?.Dispose();
                    gpuBackend?.Dispose();
                    fwd?.Dispose();
                    hybridFwd?.Dispose();
                    standaloneFwd?.Dispose();
                }
            }
            else if (!File.Exists(settings.DraftModelPath))
            {
                AnsiConsole.ErrorLine($"[red]Error:[/] Draft model not found: {settings.DraftModelPath}");
                return 1;
            }
            else
            {
                try
                {
                    AnsiConsole.MarkupLine($"[dim]Loading draft model:[/] {settings.DraftModelPath}");
                    using var draftModel = GgufModel.Open(settings.DraftModelPath);
                    var draftHp = ArchitectureModelResolver.ResolveHyperparams(draftModel);
                    if (cudaSpecTarget)
                    {
                        var target = (CudaForwardPass)gpuFwd!;
                        // The draft gets its OWN CudaBackend: graph capture state is one
                        // exec graph per backend instance, so sharing the target's backend
                        // would have the draft's decode graph clobber the target's.
                        //
                        // Clamp the draft's context: the decoder advances both passes in
                        // lockstep, so the draft never sees a position past the target's
                        // window — and unless the user pinned -c explicitly, cap it at 4096
                        // (the decode runners bound generation by BOTH windows, so a smaller
                        // draft ring only caps session length, never indexes out of range).
                        // Passing 0 would size the draft's KV from the VRAM left AFTER the
                        // target loaded — measured on the 12 GB 4070 Ti: the 0.6B draft
                        // grabbed a 34K-ctx / ~7 GB ring next to the 8B target (decode
                        // 75 → 13 t/s, WDDM paging); even a target-matched 12K fp32 ring
                        // (~2.8 GB) left so little headroom that the draft's weights paged
                        // in and out every step (draft forward 2.9 → ~15 ms, decode 34 t/s).
                        int draftCtx = ctxSize > 0 ? target.MaxSeqLen : Math.Min(target.MaxSeqLen, 4096);
                        using var draftCuda = CudaBackend.Create();
                        using var draftFwd = new CudaForwardPass(draftModel, draftCuda, draftHp, draftCtx);
                        AnsiConsole.MarkupLine($"[dim]Draft model: {draftHp.NumLayers}L, {draftHp.EmbeddingDim}d ([green]CUDA[/]) | Lookahead k={settings.SpecLookahead}[/]");
                        if (settings.Prompt is not null)
                            return RunSpeculativeSinglePrompt(settings, target, draftFwd, tokenizer, sp, rng);
                        return RunSpeculativeInteractive(settings, target, draftFwd, tokenizer, sp, rng);
                    }
                    else if (vulkanSpecTarget)
                    {
                        var target = (GpuForwardPass)gpuFwd!;
                        // The draft gets its own VulkanBackend (its own command buffer and
                        // pipelines) and the same context clamp as the CUDA branch above.
                        int draftCtx = ctxSize > 0 ? target.MaxSeqLen : Math.Min(target.MaxSeqLen, 4096);
                        using var draftVk = new VulkanBackend(gpuDeviceIndex);
                        using var draftFwd = new GpuForwardPass(draftModel, draftVk, draftHp, draftCtx);
                        AnsiConsole.MarkupLine($"[dim]Draft model: {draftHp.NumLayers}L, {draftHp.EmbeddingDim}d ([green]Vulkan[/]) | Lookahead k={settings.SpecLookahead}[/]");
                        if (settings.Prompt is not null)
                            return RunSpeculativeSinglePrompt(settings, target, draftFwd, tokenizer, sp, rng);
                        return RunSpeculativeInteractive(settings, target, draftFwd, tokenizer, sp, rng);
                    }
                    else
                    {
                        using var draftCpuBackend = new CpuBackend();
                        using var draftFwd = new ForwardPass(draftModel, draftCpuBackend, draftHp);
                        AnsiConsole.MarkupLine($"[dim]Draft model: {draftHp.NumLayers}L, {draftHp.EmbeddingDim}d ([blue]CPU[/]) | Lookahead k={settings.SpecLookahead}[/]");
                        if (settings.Prompt is not null)
                            return RunSpeculativeSinglePrompt(settings, fwd!, draftFwd, tokenizer, sp, rng);
                        return RunSpeculativeInteractive(settings, fwd!, draftFwd, tokenizer, sp, rng);
                    }
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine(ex);
                    return 1;
                }
                finally
                {
                    gpuFwd?.Dispose();
                    gpuBackend?.Dispose();
                    fwd?.Dispose();
                    hybridFwd?.Dispose();
                    standaloneFwd?.Dispose();
                }
            }
        }

        return null;
    }

    /// <summary>
    /// DSpark block-speculative decoding (<c>--dspark-model</c>). Returns null when DSpark was not requested or its placement is
    /// off, so the caller continues to normal generation; otherwise the run's exit code. Original notes:
    /// DSpark block-speculative decoding (docs/dspark-plan.md, PR #413): a DeepSpec
    /// draft head conditioned on target hidden-state taps. Greedy-only and CPU-target
    /// for now (spec Phases 1–3; the CUDA draft path is Phase 4). Placement Off (auto
    /// or explicit) falls through to normal generation rather than erroring.
    /// </summary>
    private static int? TryRunDSpark(Settings settings, RunEngine e, SamplingParams sp)
    {
        var ctxSize = e.CtxSize; var fwd = e.Fwd; var standaloneFwd = e.StandaloneFwd; var gpuBackend = e.GpuBackend; var gpuFwd = e.GpuFwd;
        var hp = e.Hp; var hybridFwd = e.HybridFwd; var model = e.Model; var nGpuLayers = e.NGpuLayers; var tokenizer = e.Tokenizer;

        bool dsparkRequested = settings.DSparkModelPath is not null || sp.SpecType == SpecType.DSpark;
        if (dsparkRequested)
        {
            if (settings.DSparkModelPath is null)
            {
                AnsiConsole.ErrorLine("[red]Error:[/] --spec-type dspark requires --dspark-model <path-to-model.safetensors>.");
                return 1;
            }
            if (settings.DraftModelPath is not null || settings.DraftLookup)
            {
                AnsiConsole.ErrorLine("[red]Error:[/] --dspark-model and --draft-model/--draft-lookup are mutually exclusive.");
                return 1;
            }
            if (sp.SpecType == SpecType.Mtp)
            {
                // An explicit conflicting --spec-type must not be silently outranked
                // by the presence of --dspark-model.
                AnsiConsole.ErrorLine("[red]Error:[/] --spec-type mtp conflicts with --dspark-model; pick one.");
                return 1;
            }
            if (settings.DSparkMinConfidence > 1f)
            {
                // Same [0,1] contract the --spec-draft-p-min validation enforces;
                // a threshold above any sigmoid output would silently disable all
                // drafting instead of doing what the user meant.
                AnsiConsole.ErrorLine($"[red]Error:[/] --dspark-min-confidence={settings.DSparkMinConfidence} must be in [0, 1].");
                return 1;
            }

            // Supported targets: pure CPU (-g 0) and dense full CUDA offload (-g -1,
            // Phase 4). Vulkan and the partial-offload hybrids fall back — no tap
            // capture there yet.
            IForwardPass? dsparkTarget = null;
            CudaBackend? dsparkCuda = null;
            if (nGpuLayers == 0 && fwd is not null)
            {
                dsparkTarget = fwd;
            }
            else if (gpuFwd is CudaForwardPass cudaTarget && gpuBackend is CudaBackend cudaBk)
            {
                dsparkTarget = cudaTarget;
                dsparkCuda = cudaBk;
            }

            string? dsparkReject = null;
            if (sp.SpecType == SpecType.None)
                dsparkReject = "--spec-type none explicitly disables speculation";
            else if (sp.Constraint is not null)
                dsparkReject = "--tool-grammar/--json-schema is active (a multi-token verify can't honor a token-level constraint)";
            else if (settings.ToolsPath is not null)
                dsparkReject = "--tools capture is not wired on the DSpark path (same restriction as MTP)";
            else if (settings.Temperature > 0f)
                dsparkReject = "DSpark is greedy-only for now; pass --temp 0";
            else if (dsparkTarget is null)
                dsparkReject = "DSpark requires a pure CPU target (-g 0) or dense full CUDA offload (-g -1); Vulkan and partial-offload hybrids have no tap capture";
            else if (!dsparkTarget.SupportsHiddenTaps)
                dsparkReject = "the target pass can't capture hidden taps (SnapKV eviction, TurboQuant KV, MoE, or Gemma-4 transforms active)";

            if (dsparkReject is not null)
            {
                AnsiConsole.MarkupLine($"[yellow]Warning:[/] DSpark disabled — {dsparkReject}. Falling back to normal generation.");
            }
            else if (settings.Prompt is null)
            {
                AnsiConsole.MarkupLine("[yellow]Warning:[/] DSpark is wired for single-prompt runs only (like MTP); interactive mode falls back to normal generation.");
            }
            else
            {
                int rc;
                try
                {
                    // DSpark is refused for SafeTensors packages above, so this is only ever
                    // reached on the GGUF path where `model` is assigned.
                    rc = TryRunDSparkSinglePrompt(settings, model!, hp, dsparkTarget!, dsparkCuda,
                        tokenizer, sp, ctxSize, e.Plan);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine(ex);
                    rc = 1;
                }
                if (rc >= 0)
                {
                    gpuFwd?.Dispose();
                    gpuBackend?.Dispose();
                    fwd?.Dispose();
                    hybridFwd?.Dispose();
                    standaloneFwd?.Dispose();
                    return rc;
                }
                // rc < 0: placement said Off — fall through to normal generation.
            }
        }

        return null;
    }

    /// <summary>
    /// True when a prompt of <paramref name="promptTokens"/> tokens leaves no room to
    /// speculate inside BOTH context windows (prompt + lookahead + 1 correction token).
    /// Prints an actionable error: the typical trigger is the CUDA draft's 4096-token
    /// KV ring cap when <c>-c</c> isn't pinned, where prefilling past the ring would
    /// write K/V out of range and a tail prompt would silently emit zero tokens.
    /// </summary>
    private static bool SpecWindowExhausted(int promptTokens,
        IForwardPass target, IForwardPass? draft, int lookahead)
    {
        int window = Math.Min(target.MaxSeqLen, draft?.MaxSeqLen ?? int.MaxValue);
        if (promptTokens + lookahead + 1 < window) return false;
        AnsiConsole.ErrorLine(
            $"[red]Error:[/] prompt ({promptTokens} tokens) + lookahead ({lookahead}) does not fit the " +
            $"speculative context window ({window} tokens" +
            (draft is not null && draft.MaxSeqLen < target.MaxSeqLen
                ? $", limited by the draft model's KV ring — pass -c to size it explicitly"
                : "") +
            "). Shorten the prompt, raise -c, or drop --draft-model/--draft-lookup.");
        return true;
    }

    private static int RunSpeculativeSinglePrompt(Settings s,
        IForwardPass target, IForwardPass? draft,
        GgufTokenizer tok, SamplingParams sp, Random rng)
    {
        var prompt = FormatPrompt(s.Prompt!, s.SystemPrompt, enableThinking: !s_noThinking);
        var tokens = tok.Encode(prompt);

        // The prompt must fit BOTH context windows BEFORE any prefill runs — the
        // draft's ring may be much smaller than the target's (the CUDA spec path
        // caps it at 4096 when -c isn't pinned), and a too-long prompt would write
        // K/V past the ring's end during draft.Prefill, not merely cap generation.
        if (SpecWindowExhausted(tokens.Count, target, draft, s.SpecLookahead))
            return 1;

        if (!s.NoDisplayPrompt)
            Console.Write(s.Prompt);

        var sw = Stopwatch.StartNew();
        // Prefill (batched-trunk path — the per-token Forward loop this replaces was
        // ~30× slower on the CUDA target). A null draft means prompt-lookup mode.
        ReadOnlySpan<float> targetLogits = target.Prefill(tokens);
        ReadOnlySpan<float> draftLogits = draft is not null ? draft.Prefill(tokens) : default;
        var prefillMs = sw.Elapsed.TotalMilliseconds;

        SpeculativeDecoder spec;
        if (draft is not null)
        {
            // temp>0 → sampled (distribution-preserving) accept; temp 0 → greedy (byte-stable).
            spec = sp.Temperature > 0f
                ? new SpeculativeDecoder(target, draft, sp, rng, s.SpecLookahead) { BatchVerify = s_speculation.SpecBatchVerify }
                : new SpeculativeDecoder(target, draft, s.SpecLookahead) { BatchVerify = s_speculation.SpecBatchVerify };
            spec.Initialize(tokens.Count, targetLogits, draftLogits);
        }
        else
        {
            spec = new SpeculativeDecoder(target, new PromptLookupDraft(), s.SpecLookahead) { BatchVerify = s_speculation.SpecBatchVerify };
            spec.Initialize(tokens, targetLogits);
        }

        // Bound generation by BOTH context windows (the draft's may be smaller — the CUDA
        // spec path caps its KV ring), leaving lookahead headroom for the last spec step.
        // The guard above ensures maxNew >= 1 here.
        int maxNew = Math.Min(sp.MaxNewTokens,
            Math.Min(target.MaxSeqLen, draft?.MaxSeqLen ?? int.MaxValue) - tokens.Count - s.SpecLookahead - 1);
        if (maxNew < sp.MaxNewTokens)
            AnsiConsole.MarkupLine($"[yellow]Note:[/] generation capped at {maxNew} tokens by the context window.");

        sw.Restart();
        int generated = 0;
        int totalDecoded = 0;
        bool inThinking = false;
        var streamDec = new Utf8StreamDecoder();
        bool hideThinking = s.HideThinking;
        spec.Decode(maxNew, sp.StopTokenIds ?? [], token =>
        {
            if (EmitToken(token, tok, streamDec, ref inThinking, hideThinking)) generated++;
            totalDecoded++;
        });
        var tail = streamDec.Flush();
        if (!(hideThinking && inThinking)) Console.Write(tail);
        if (inThinking) Console.Write("\x1b[0m");
        var decodeMs = sw.Elapsed.TotalMilliseconds;

        Console.WriteLine();
        EngineEvents.Emit(new(EngineEventKind.PrefillCompleted, A: tokens.Count, Ms: prefillMs));
        EngineEvents.Emit(new(EngineEventKind.DecodeCompleted, A: totalDecoded, B: generated, Ms: decodeMs));
        AnsiConsole.MarkupLine($"\n[dim]Prefill: {tokens.Count} tokens, {tokens.Count / (prefillMs / 1000):F1} t/s | " +
            $"Decode: {totalDecoded} tokens, {totalDecoded / (decodeMs / 1000):F1} t/s" +
            (totalDecoded > generated ? $" ({generated} visible, {totalDecoded - generated} thinking)" : "") +
            $" | Acceptance rate: {spec.AcceptanceRate:P0} | " +
            $"draft {spec.DraftMs:F0}ms / verify {spec.VerifyMs:F0}ms / commit {spec.CommitMs:F0}ms[/]");
        return 0;
    }

    private static int RunSpeculativeInteractive(Settings s,
        IForwardPass target, IForwardPass? draft,
        GgufTokenizer tok, SamplingParams sp, Random rng)
    {
        AnsiConsole.MarkupLine("[green]Interactive chat (speculative decoding).[/] Type a message, or [yellow]/exit[/] to quit.\n");
        var spec = draft is not null
            ? (sp.Temperature > 0f
                ? new SpeculativeDecoder(target, draft, sp, rng, s.SpecLookahead) { BatchVerify = s_speculation.SpecBatchVerify }
                : new SpeculativeDecoder(target, draft, s.SpecLookahead) { BatchVerify = s_speculation.SpecBatchVerify })
            : new SpeculativeDecoder(target, new PromptLookupDraft(), s.SpecLookahead) { BatchVerify = s_speculation.SpecBatchVerify };

        while (true)
        {
            AnsiConsole.Markup("[bold]> [/]");
            var input = Console.ReadLine();
            if (input is null or "/exit" or "/quit") break;
            if (string.IsNullOrWhiteSpace(input)) continue;

            var prompt = FormatPrompt(input, s.SystemPrompt, enableThinking: !s_noThinking);
            var tokens = tok.Encode(prompt);

            // Same pre-prefill window guard as the single-prompt runner: the draft
            // ring may be smaller than the target's window, and prefilling past it
            // writes K/V out of range rather than just capping generation.
            if (SpecWindowExhausted(tokens.Count, target, draft, s.SpecLookahead))
                continue;

            target.ResetCache();
            draft?.ResetCache();

            var sw = Stopwatch.StartNew();
            ReadOnlySpan<float> targetLogits = target.Prefill(tokens);

            if (draft is not null)
                spec.Initialize(tokens.Count, targetLogits, draft.Prefill(tokens));
            else
                spec.Initialize(tokens, targetLogits);

            int maxNew = Math.Min(sp.MaxNewTokens,
                Math.Min(target.MaxSeqLen, draft?.MaxSeqLen ?? int.MaxValue) - tokens.Count - s.SpecLookahead - 1);

            sw.Restart();
            int generated = 0;
            int totalDecoded = 0;
            bool inThinking = false;
            var streamDec = new Utf8StreamDecoder();
            bool hideThinking = s.HideThinking;
            spec.Decode(maxNew, sp.StopTokenIds ?? [], token =>
            {
                if (EmitToken(token, tok, streamDec, ref inThinking, hideThinking)) generated++;
                totalDecoded++;
            });
            var tail = streamDec.Flush();
            if (!(hideThinking && inThinking)) Console.Write(tail);
            if (inThinking) Console.Write("\x1b[0m");
            var decodeMs = sw.Elapsed.TotalMilliseconds;

            Console.WriteLine();
            AnsiConsole.MarkupLine($"[dim]{totalDecoded} tokens, {totalDecoded / (decodeMs / 1000):F1} t/s" +
                (totalDecoded > generated ? $" ({generated} visible, {totalDecoded - generated} thinking)" : "") +
                $" | Accept: {spec.AcceptanceRate:P0}[/]\n");

            if (OpenTail.Stingray.Engine.DecodeProfileTimers.Enabled)
                OpenTail.Stingray.Engine.DecodeProfileTimers.Report(Console.Out);

            if (s.SingleTurn) break;
        }
        return 0;
    }

    /// <summary>
    /// DSpark single-prompt runner (docs/dspark-plan.md, PR #413): resolves the draft-head
    /// files (model.safetensors + sibling config.json), validates head↔target compatibility,
    /// runs the placement planner, loads the head (GPU or CPU per the decision — Phase 4),
    /// enables hidden taps on the target, and drives <see cref="DSparkDecoder"/>. Returns
    /// 0/1 like the other runners, or -1 when the placement decision is Off — the caller
    /// falls back to normal generation.
    /// </summary>
    private static int TryRunDSparkSinglePrompt(Settings s, GgufModel model, ModelHyperparams hp,
        IForwardPass target, CudaBackend? cuda, GgufTokenizer tok, SamplingParams sp, int ctxSize, ExecutionPlan plan)
    {
        string stPath = s.DSparkModelPath!;
        if (Directory.Exists(stPath)) stPath = Path.Combine(stPath, "model.safetensors");
        if (!File.Exists(stPath))
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] DSpark model not found: {stPath}");
            return 1;
        }
        string cfgPath = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(stPath))!, "config.json");
        if (!File.Exists(cfgPath))
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] DSpark config.json not found next to the safetensors: {cfgPath}");
            return 1;
        }

        var cfg = DSparkConfig.FromJsonFile(cfgPath);
        if (cfg.VocabSize != hp.VocabSize || cfg.NumTargetLayers != hp.NumLayers
            || cfg.HiddenSize != hp.EmbeddingDim)
        {
            AnsiConsole.ErrorLine(
                $"[red]Error:[/] DSpark head/target mismatch — head expects vocab {cfg.VocabSize}, " +
                $"{cfg.NumTargetLayers} target layers, hidden {cfg.HiddenSize}; target has " +
                $"vocab {hp.VocabSize}, {hp.NumLayers} layers, hidden {hp.EmbeddingDim}. " +
                "The head must be trained for this target model.");
            return 1;
        }

        // Placement was decided by ExecutionPlanner and recorded in the plan; the CLI only obeys it.
        var spec = plan.Speculation;
        AnsiConsole.MarkupLine($"[dim]DSpark placement: {spec?.DSparkPlacement} — {(spec?.DSparkPlacementReason ?? "").EscapeMarkup()}[/]");
        if (spec is null || !spec.DSparkEnabled || spec.DSparkPlacement == DSparkPlacement.Off)
        {
            AnsiConsole.MarkupLine("[yellow]Note:[/] DSpark placement is off; falling back to normal generation.");
            return -1;
        }
        var decision = (Placement: spec.DSparkPlacement, Reason: spec.DSparkPlacementReason);
        if (decision.Placement == DSparkPlacement.Gpu && cuda is null)
        {
            AnsiConsole.ErrorLine("[red]Error:[/] the plan places the DSpark draft on GPU but no CUDA target backend is active.");
            return 1;
        }

        AnsiConsole.MarkupLine($"[dim]Loading DSpark draft head:[/] {stPath}");
        using var st = SafetensorsLoader.Open(stPath);
        using IDSparkDraft draft = decision.Placement == DSparkPlacement.Gpu
            ? new CudaDSparkDraftModel(cfg, st, cuda!, target.MaxSeqLen)
            : new DSparkDraftModel(cfg, st, target.MaxSeqLen);
        AnsiConsole.MarkupLine(
            $"[dim]DSpark draft: {cfg.NumLayers}L block-{cfg.BlockSize} " +
            $"([{(decision.Placement == DSparkPlacement.Gpu ? "green]GPU" : "blue]CPU")}[/])[/]");
        target.EnableHiddenTaps(cfg.TargetLayerIds);

        var prompt = FormatPrompt(s.Prompt!, s.SystemPrompt, enableThinking: !s_noThinking);
        var tokens = tok.Encode(prompt);
        // The head's RoPE window (max_position_embeddings) can be smaller than the
        // target's — bound the whole session by BOTH, like the spec-decode runner does.
        int window = Math.Min(target.MaxSeqLen, draft.MaxContext);
        if (tokens.Count + cfg.BlockSize + 1 >= window)
        {
            AnsiConsole.ErrorLine(
                $"[red]Error:[/] prompt ({tokens.Count} tokens) + DSpark block ({cfg.BlockSize}) " +
                $"does not fit the context window ({window} tokens" +
                (draft.MaxContext < target.MaxSeqLen ? ", limited by the draft head's RoPE window" : "") +
                ").");
            return 1;
        }

        if (!s.NoDisplayPrompt)
            Console.Write(s.Prompt);

        var sw = Stopwatch.StartNew();
        ReadOnlySpan<float> logits = target.Prefill(tokens);
        var prefillMs = sw.Elapsed.TotalMilliseconds;

        var decoder = new DSparkDecoder(target, draft);
        decoder.Initialize(tokens.Count, logits);

        int maxNew = Math.Min(sp.MaxNewTokens,
            window - tokens.Count - cfg.BlockSize - 1);
        if (maxNew < sp.MaxNewTokens)
            AnsiConsole.MarkupLine($"[yellow]Note:[/] generation capped at {maxNew} tokens by the context window.");

        float minConfidence = DSparkDecoder.ResolveMinConfidence(s.DSparkMinConfidence, s_speculation);
        int verifyLenCap = DSparkDecoder.ResolveVerifyLen(s.DSparkVerifyLen, s_speculation);

        sw.Restart();
        int generated = 0;
        int totalDecoded = 0;
        bool inThinking = false;
        var streamDec = new Utf8StreamDecoder();
        bool hideThinking = s.HideThinking;
        decoder.Decode(maxNew, sp.StopTokenIds ?? [], token =>
        {
            if (EmitToken(token, tok, streamDec, ref inThinking, hideThinking)) generated++;
            totalDecoded++;
        }, minConfidence, verifyLenCap);
        var tail = streamDec.Flush();
        if (!(hideThinking && inThinking)) Console.Write(tail);
        if (inThinking) Console.Write("\x1b[0m");
        var decodeMs = sw.Elapsed.TotalMilliseconds;

        Console.WriteLine();
        EngineEvents.Emit(new(EngineEventKind.PrefillCompleted, A: tokens.Count, Ms: prefillMs));
        EngineEvents.Emit(new(EngineEventKind.DecodeCompleted, A: totalDecoded, B: generated, Ms: decodeMs));
        AnsiConsole.MarkupLine($"\n[dim]Prefill: {tokens.Count} tokens, {tokens.Count / (prefillMs / 1000):F1} t/s | " +
            $"Decode: {totalDecoded} tokens, {totalDecoded / (decodeMs / 1000):F1} t/s" +
            (totalDecoded > generated ? $" ({generated} visible, {totalDecoded - generated} thinking)" : "") +
            $" | DSpark accept: {decoder.AcceptanceRate:P0} ({decoder.TotalDraftsAccepted}/{decoder.TotalDraftsEmitted}) | " +
            $"draft {decoder.DraftMs:F0}ms / verify {decoder.VerifyMs:F0}ms / commit {decoder.CommitMs:F0}ms[/]");
        // Draft-internal breakdown for the #428 perf work: enqueue = launch-issue CPU
        // time, gpu-wait = GPU execution + D2H collapsed into the per-block download
        // sync, heads = host Markov/confidence chain.
        if (Environment.GetEnvironmentVariable("STINGRAY_DSPARK_TIMING") == "1"
            && draft is CudaDSparkDraftModel cudaDraft)
            AnsiConsole.MarkupLine(
                $"[dim]DSpark draft breakdown: enqueue {decoder.DraftMs - cudaDraft.GpuWaitMs - cudaDraft.HostHeadsMs:F0}ms / " +
                $"gpu-wait {cudaDraft.GpuWaitMs:F0}ms / heads {cudaDraft.HostHeadsMs:F0}ms[/]");
        return 0;
    }

    // Decides whether to engage the MTP self-speculative path on the CLI side.
    // Mirrors the InferenceEngine gate but reads `--no-thinking` from CLI settings
    // (vs. the engine which inspects whether the model has think tokens registered).
    // The CLI path can engage MTP even on models that registered <think>/</think>,
    // as long as the user passed --no-thinking — in that case the template renders
    // with enable_thinking=false and no think tokens appear in the prompt or output.
    private static bool ResolveCliMtp(IForwardPass? mtpFwd, SamplingParams sp, bool noThinking, out string? rejectReason)
    {
        rejectReason = null;
        bool envDisabled = !s_speculation.MtpEnabled;
        bool eligible = mtpFwd is not null
                        && mtpFwd.HasMtpHead
                        && sp.Temperature <= 0f
                        && !sp.HasHistoryPenalty
                        && noThinking
                        && !envDisabled;

        // Spec-decode CLI flag validation:
        //   --spec-draft-p-min: clamped to [0, 1] at the MtpDecoder boundary. Accepted
        //     on any spec path. 1.0 (or 0 / unset) = pure argmax-match; p ∈ (0,1) is
        //     llama.cpp's probabilistic-accept rule from #38. Reject obviously bad input.
        //   --spec-draft-n-min: accepted as a no-op under N=2 batched verify.
        //   --spec-draft-n-max == 2 enables #30 batched verify; > 2 still rejected.
        if (sp.SpecType != SpecType.None)
        {
            if (sp.SpecDraftPMin > 1f)
            {
                rejectReason = $"--spec-draft-p-min={sp.SpecDraftPMin} must be in [0, 1].";
                return false;
            }
        }

        // Max drafts per step = batch capacity − 1 (the certain token rides in the
        // batch). The pass's snapshot-ring capacity bounds the batch (issue #30);
        // without batched verify the sequential path drafts exactly 1 per step.
        int maxDraftN = (mtpFwd is not null && mtpFwd.SupportsBatchVerify)
            ? Math.Max(1, mtpFwd.MaxBatchVerifyTokens - 1)
            : 1;

        switch (sp.SpecType)
        {
            case SpecType.None:
                return false;
            case SpecType.DSpark:
                // The user asked for DSpark; if that path already rejected/fell back
                // upstream, silently swapping in MTP would contradict the printed
                // "falling back to normal generation" warning.
                return false;
            case SpecType.Mtp:
                if (envDisabled) { rejectReason = "--spec-type mtp conflicts with STINGRAY_DISABLE_MTP=1."; return false; }
                if (mtpFwd is null || !mtpFwd.HasMtpHead) { rejectReason = "--spec-type mtp requires a model with an MTP head (nextn tensors)."; return false; }
                if (sp.Temperature > 0f) { rejectReason = "--spec-type mtp requires greedy sampling (--temp 0)."; return false; }
                if (!noThinking) { rejectReason = "--spec-type mtp requires --no-thinking (chat template must render with enable_thinking=false)."; return false; }
                if (sp.HasHistoryPenalty)
                    AnsiConsole.MarkupLine("[yellow]Note:[/] --spec-type mtp decodes by pure argmax and ignores the repetition/presence penalties; pass --repeat-penalty 1.0 to make plain decoding match it exactly.");
                WarnIfDraftNClamped(sp.SpecDraftNMax, maxDraftN);
                return true;
            default: // Auto
                if (eligible)
                    WarnIfDraftNClamped(sp.SpecDraftNMax, maxDraftN);
                return eligible;
        }
    }

    /// <summary>
    /// A draft chain deeper than the snapshot ring's capacity is CLAMPED, not rejected
    /// (rejecting would disable MTP entirely and run SLOWER — the silent-baseline trap
    /// the old SpecDraftNMax&gt;1 throw existed to prevent). Warn so the user knows the
    /// effective depth and the knob that raises it; MtpDecoder clamps per step.
    /// </summary>
    private static void WarnIfDraftNClamped(int requested, int maxDraftN)
    {
        if (requested > maxDraftN)
            AnsiConsole.MarkupLine(
                $"[yellow]Note:[/] --spec-draft-n-max={requested} exceeds the snapshot-ring capacity; " +
                $"running {maxDraftN} draft(s)/step. Set STINGRAY_MTP_BATCH_MAX={requested + 1} to go deeper " +
                "(each ring slot costs ~150 MiB VRAM on 27B-class models).");
    }

    // MTP self-speculative decode path. Reuses the same UTF-8 streaming + EmitToken
    // logic as the baseline DecodeLoop but drives token emission via MtpDecoder.
    // Requires --no-thinking, so no thinking-mode bookkeeping here.
    private static (int generated, int totalDecoded, float acceptanceRate, long accepted, long emitted) DecodeLoopMtp(
        IForwardPass mtpFwd, IReadOnlyList<int> promptTokens, ReadOnlySpan<float> initialLogits,
        GgufTokenizer tok, SamplingParams sp, bool hideThinking, bool verbosePromptLogging = false)
    {
        var mtpDec = new MtpDecoder(mtpFwd, s_speculation);
        mtpDec.Initialize(promptTokens.Count, initialLogits);
        // Populate the MTP KV cache for the full prompt. Cost: ~1.6%/token; only paid
        // on the MTP-enabled run.
        mtpFwd.PrefillMtp(promptTokens, 0);

        var streamDec = new Utf8StreamDecoder();
        bool inThinking = false;
        int generated = 0;
        int totalDecoded = 0;

        // Materialize stop ids once (MtpDecoder takes ReadOnlySpan<int>).
        int[] stopIds = sp.StopTokenIds ?? [];

        mtpDec.Decode(sp.MaxNewTokens, stopIds.AsSpan(), next =>
        {
            if (verbosePromptLogging)
                Console.Error.WriteLine($"[DBG] tok={totalDecoded} next={next}('{tok.Decode([next])}')");
            totalDecoded++;
            if (EmitToken(next, tok, streamDec, ref inThinking, hideThinking)) generated++;
        }, pMin: sp.SpecDraftPMin, draftN: MtpDecoder.ResolveDraftN(sp.SpecDraftNMax, s_speculation, mtpFwd),
           ct: CancellationToken.None);

        if (Environment.GetEnvironmentVariable("STINGRAY_TRACE_MTP") == "1" && mtpDec.TotalDraftsEmitted > 0)
            Console.Error.WriteLine(
                $"[mtp] phase ms: draft={mtpDec.DraftMs:F0} verify={mtpDec.VerifyMs:F0} commit={mtpDec.CommitMs:F0}");

        // Flush the UTF-8 decoder tail, applying the same hide-thinking gate as DecodeLoop.
        var tail = streamDec.Flush();
        if (!(hideThinking && inThinking))
            Console.Write(tail);
        if (inThinking) Console.Write("\x1b[0m");

        return (generated, totalDecoded, mtpDec.AcceptanceRate, mtpDec.TotalDraftsAccepted, mtpDec.TotalDraftsEmitted);
    }

    // Accept llama.cpp's "draft-mtp" alongside the shorter "mtp" so existing command
    // lines copy-paste over. Unknown values fall back to auto with a console warning.
    private static SpecType ParseSpecType(string? value)
    {
        if (string.IsNullOrWhiteSpace(value)) return SpecType.Auto;
        return value.Trim().ToLowerInvariant() switch
        {
            "auto" or "" => SpecType.Auto,
            "none" or "off" or "disabled" => SpecType.None,
            "mtp" or "draft-mtp" => SpecType.Mtp,
            "dspark" => SpecType.DSpark,
            _ => WarnUnknownSpecType(value),
        };

        static SpecType WarnUnknownSpecType(string v)
        {
            AnsiConsole.MarkupLine($"[yellow]Warning:[/] Unknown --spec-type [yellow]{Markup.Escape(v)}[/]; expected auto|none|mtp|dspark. Falling back to auto.");
            return SpecType.Auto;
        }
    }
}
