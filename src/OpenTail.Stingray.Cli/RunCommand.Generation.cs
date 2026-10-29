namespace OpenTail.Stingray.Cli;

public sealed partial class RunCommand
{
    private static int RunSinglePrompt(Settings s,
        Func<int, int, ReadOnlySpan<float>> forward,
        Func<IReadOnlyList<int>, ReadOnlySpan<float>> prefill,
        GgufTokenizer tok, SamplingParams sp, Random rng,
        IForwardPass? mtpFwd, int maxContextLength)
    {
        var prompt = FormatPrompt(s.Prompt!, s.SystemPrompt, enableThinking: !s_noThinking);
        var tokens = tok.Encode(prompt);

        tokens = EnsureBos(tokens, tok);

        // The final prompt token occupies one context slot and ordinary generation needs at
        // least one more. Check after every tokenizer/BOS transformation and before Prefill:
        // ForwardPass uses this same bound for its scratch allocation, so admitting an oversized
        // prompt after wiring --ctx-size would be an unsafe write, not a harmless truncation.
        if (tokens.Count >= maxContextLength)
        {
            AnsiConsole.ErrorLine(
                $"[red]Error:[/] prompt has {tokens.Count} tokens but the active context is " +
                $"{maxContextLength}; shorten the prompt or raise [yellow]--ctx-size[/] so at least one token can be generated.");
            return 1;
        }

        // The prompt check above is necessary but not sufficient: generation continues from the
        // prompt's last position, so the context bounds prompt + output together. With
        // --ctx-size 512 and the default --n-predict 512, any prompt at all runs off the end.
        // Stop at context-full (llama.cpp's -2 semantics) rather than decoding past MaxSeqLen.
        sp = ClampToRemainingContext(sp, tokens.Count, maxContextLength);

        if (s.VerbosePrompt)
        {
            var escaped = prompt.Replace("\n", "\\n").Replace("\r", "\\r");
            AnsiConsole.MarkupLine($"[dim]Prompt (escaped): {Markup.Escape(escaped)}[/]");
            AnsiConsole.MarkupLine($"[dim]Prompt tokens ({tokens.Count}): {string.Join(", ", tokens)}[/]");
        }

        var sw = Stopwatch.StartNew();
        var logits = prefill(tokens);
        var prefillMs = sw.Elapsed.TotalMilliseconds;

        if (OpenTail.Stingray.Engine.PrefillProfileTimers.Enabled)
            OpenTail.Stingray.Engine.PrefillProfileTimers.Report(Console.Out);
        // STINGRAY_PROFILE_GPU_SPLIT=1: Vulkan queue submits (each one a fence wait) per phase.
        OpenTail.Stingray.Vulkan.VulkanBackend.PrintGpuProfile($"prefill {tokens.Count} tokens");
        OpenTail.Stingray.Vulkan.VulkanBackend.ResetGpuProfile();

        if (!s.NoDisplayPrompt)
            Console.Write(s.Prompt);

        // MTP self-speculative decode (issue #32). Activates when the model ships an
        // MTP head AND sampling is greedy AND the user disabled thinking on the chat
        // template (--no-thinking) AND sp.SpecType permits. STINGRAY_DISABLE_MTP=1 is
        // a back-compat off-switch that wins.
        bool useMtp = ResolveCliMtp(mtpFwd, sp, s_noThinking, out string? mtpReject);
        if (mtpReject != null)
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] {Markup.Escape(mtpReject)}");
            return 1;
        }

        // When --tools is active, capture the raw token stream so the calls can be parsed and shown
        // afterward, and route through the standard decode loop: the MTP fast path can't honor the
        // argument-grammar constraint (sp.Constraint) and has no capture hook.
        List<int>? toolCapture = s.ToolsPath is { Length: > 0 } ? new List<int>() : null;
        if (toolCapture is not null) useMtp = false;

        sw.Restart();
        int generated, totalDecoded;
        float? acceptanceRate = null;
        long mtpAccepted = 0, mtpEmitted = 0;
        if (useMtp)
        {
            (generated, totalDecoded, acceptanceRate, mtpAccepted, mtpEmitted) =
                DecodeLoopMtp(mtpFwd!, tokens, logits, tok, sp, s.HideThinking, s.VerbosePrompt);
        }
        else
        {
            (generated, totalDecoded) =
                DecodeLoop(forward, logits, tokens.Count, tok, sp, rng, s.VerbosePrompt, s.HideThinking, s.MaxThinkingTokens, toolCapture, repeatLastN: s.RepeatLastN);
        }
        var decodeMs = sw.Elapsed.TotalMilliseconds;

        Console.WriteLine();
        EngineEvents.Emit(new(EngineEventKind.PrefillCompleted, A: tokens.Count, Ms: prefillMs));
        EngineEvents.Emit(new(EngineEventKind.DecodeCompleted, A: totalDecoded, B: generated, Ms: decodeMs));
        AnsiConsole.MarkupLine($"\n[dim]Prefill: {tokens.Count} tokens, {tokens.Count / (prefillMs / 1000):F1} t/s | " +
            $"Decode: {totalDecoded} tokens, {totalDecoded / (decodeMs / 1000):F1} t/s" +
            (totalDecoded > generated ? $" ({generated} visible, {totalDecoded - generated} thinking)" : "") +
            (acceptanceRate is float ar ? $" | MTP accept: {ar:P0} ({mtpAccepted}/{mtpEmitted})" : "") +
            "[/]");
        OpenTail.Stingray.Vulkan.VulkanBackend.PrintGpuProfile($"decode {totalDecoded} tokens");

        if (OpenTail.Stingray.Engine.DecodeProfileTimers.Enabled)
            OpenTail.Stingray.Engine.DecodeProfileTimers.Report(Console.Out);
        OpenTail.Stingray.Engine.HybridGdnForwardPass.ReportGdnProfile(Console.Out);

        if (toolCapture is not null)
            PrintToolCalls(tok, toolCapture);
        return 0;
    }

    private static int RunInteractive(Settings s,
        Func<int, int, ReadOnlySpan<float>> forward,
        Func<IReadOnlyList<int>, ReadOnlySpan<float>> prefill,
        Action resetCache,
        GgufTokenizer tok, SamplingParams sp, Random rng,
        IForwardPass? mtpFwd, int maxContextLength)
    {
        // mtpFwd reserved for interactive MTP wiring (follow-up to #32). Today the
        // interactive loop stays on the baseline decode path; the bench surface and
        // single-prompt runs (RunSinglePrompt above) are what exercise MTP.
        _ = mtpFwd;

        // Full-screen chat UI when attached to a real terminal. It takes over the alternate
        // screen buffer, so a redirected stdin/stdout (scripted use) keeps the plain
        // line-oriented loop below, which stays pipe-friendly.
        //
        // Compiled only when -p:EnableTuiChat=true. The UI depends on OpenTail.TUI, which is not
        // yet published with the API this uses (IModel/IMsg/Cmd, TranscriptEntry, TextInputState,
        // ViewportState, CellBuffer), so the default build has no such dependency and the plain
        // loop below is the interactive path. Nothing else in the CLI touches Tui/.
#if STINGRAY_TUI
        if (Tui.ChatTui.IsSupported && !s.SingleTurn)
        {
            var engine = new Tui.ChatEngine((message, onText, ct) =>
            {
                var prompt = FormatPrompt(message, s.SystemPrompt, enableThinking: !s_noThinking);
                var tokens = tok.Encode(prompt);

                // Same context bound as the single-prompt path: the scratch buffers are sized
                // from it and the KV cache is not, so overrunning is an out-of-bounds write.
                // Decline the turn rather than prefill it; the transcript keeps its history.
                if (tokens.Count >= maxContextLength) return 0;
                var turnSp = ClampToRemainingContext(sp, tokens.Count, maxContextLength);

                resetCache();
                var logits = prefill(tokens);

                var (_, totalDecoded) = DecodeLoop(
                    forward, logits, tokens.Count, tok, turnSp, rng,
                    hideThinking: s.HideThinking, maxThinkingTokens: s.MaxThinkingTokens,
                    sink: onText, cancellation: ct, repeatLastN: s.RepeatLastN);

                return totalDecoded;
            });

            return Tui.ChatTui.Run(engine, Path.GetFileName(s.ModelPath) ?? "model");
        }
#endif

        AnsiConsole.MarkupLine("[green]Interactive chat.[/] Type a message, or [yellow]/exit[/] to quit.\n");

        while (true)
        {
            AnsiConsole.Markup("[bold]> [/]");
            var input = Console.ReadLine();
            if (input is null or "/exit" or "/quit") break;
            if (string.IsNullOrWhiteSpace(input)) continue;

            var prompt = FormatPrompt(input, s.SystemPrompt, enableThinking: !s_noThinking);
            var tokens = tok.Encode(prompt);

            // Reject rather than truncate, and stay in the loop so the session survives a single
            // oversized paste. ForwardPass sizes its scratch from this bound but its KV cache is
            // far larger, so prefilling past it would be an out-of-bounds write.
            if (tokens.Count >= maxContextLength)
            {
                AnsiConsole.ErrorLine(
                    $"[red]Error:[/] message is {tokens.Count} tokens but the active context is " +
                    $"{maxContextLength}; shorten it or restart with a larger [yellow]--ctx-size[/].");
                continue;
            }
            var turnSp = ClampToRemainingContext(sp, tokens.Count, maxContextLength);

            resetCache();
            var sw = Stopwatch.StartNew();
            var logits = prefill(tokens);

            sw.Restart();
            var (generated, totalDecoded) = DecodeLoop(forward, logits, tokens.Count, tok, turnSp, rng, hideThinking: s.HideThinking, maxThinkingTokens: s.MaxThinkingTokens, repeatLastN: s.RepeatLastN);
            var decodeMs = sw.Elapsed.TotalMilliseconds;

            Console.WriteLine();
            AnsiConsole.MarkupLine($"[dim]{totalDecoded} tokens, {totalDecoded / (decodeMs / 1000):F1} t/s" +
                (totalDecoded > generated ? $" ({generated} visible, {totalDecoded - generated} thinking)" : "") +
                "[/]\n");

            if (s.SingleTurn) break;
        }
        return 0;
    }

    private static (int generated, int totalDecoded) DecodeLoop(
        Func<int, int, ReadOnlySpan<float>> forward,
        ReadOnlySpan<float> initialLogits,
        int startPos,
        GgufTokenizer tok,
        SamplingParams sp,
        Random rng,
        bool verbosePromptLogging = false,
        bool hideThinking = false,
        int maxThinkingTokens = 0,
        List<int>? captureTokens = null,
        Action<string>? sink = null,
        CancellationToken cancellation = default,
        int repeatLastN = 64,
        bool graniteDiagnostic = false)
    {
        var logits = initialLogits;
        int generated = 0;
        int totalDecoded = 0;
        bool inThinking = false;
        int thinkingTokenCount = 0;
        // Presence/frequency count the WHOLE completion (OpenAI), so history may only be trimmed to
        // the repetition window when neither is active — which is the common case, so the usual
        // memory profile is unchanged. The repetition window itself is applied by the sampler now,
        // not by evicting this list, so the two families no longer share one window.
        int historyCap = (sp.PresencePenalty != 0f || sp.FrequencyPenalty != 0f)
            ? int.MaxValue
            : ResolvePenaltyHistoryCap(repeatLastN);

        // --repeat-last-n and SamplingParams.RepeatLastN encode "no limit" differently. Translate
        // here so the flag's 0/-1 convention stays a CLI concern:
        //   0  → repetition penalty off entirely (neutralise the multiplier)
        //   <0 → whole history        (sampler: 0)
        //   >0 → that many trailing tokens
        float effectiveRepetition = repeatLastN == 0 ? 1.0f : sp.RepetitionPenalty;
        int samplerRepeatLastN = repeatLastN < 0 ? 0 : repeatLastN;
        var recentTokens = new List<int>(Math.Min(historyCap, 256));
        var streamDec = new Utf8StreamDecoder();
        // Tool-call grammar constraint (issue #374): start each response from the watching state so
        // a reused instance serves this generation fresh. No-op when no constraint is attached.
        sp.Constraint?.Reset();
        bool profDecode = OpenTail.Stingray.Engine.DecodeProfileTimers.Enabled;
        for (int i = 0; i < sp.MaxNewTokens; i++)
        {
            // Stop between tokens when the caller cancels (the TUI quitting mid-generation).
            // Breaking rather than throwing keeps whatever has already been decoded, and leaves
            // the KV cache consistent with the tokens actually emitted.
            if (cancellation.IsCancellationRequested) break;

            long nonTrunkStart = profDecode ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;

            var spWithHistory = (effectiveRepetition != 1.0f || sp.PresencePenalty != 0f || sp.FrequencyPenalty != 0f)
                && recentTokens.Count > 0
                ? sp with
                {
                    PreviousTokens = recentTokens,
                    RepetitionPenalty = effectiveRepetition,
                    RepeatLastN = samplerRepeatLastN,
                }
                : sp;
            int next;
            if (inThinking && maxThinkingTokens > 0 && thinkingTokenCount >= maxThinkingTokens && s_endThinkTokenId > 0)
            {
                // Force </think> to exit a runaway reasoning block; the close tag still
                // goes through forward() below so the model continues from the post-think state.
                next = s_endThinkTokenId;
            }
            else
            {
                // While the grammar is restricting the vocabulary, sample from the masked logits so
                // only a grammar-legal token can be chosen; otherwise sample exactly as before.
                var sampleLogits = sp.Constraint is { IsConstraining: true } ctr ? ctr.Filter(logits) : logits;
                next = sp.Temperature <= 0 && !spWithHistory.HasHistoryPenalty
                    ? Sampler.Greedy(sampleLogits)
                    : Sampler.Sample(sampleLogits, spWithHistory, rng);
            }
            if (verbosePromptLogging && i == 0) PrintTopLogProbs(logits, tok, 10);
            if (verbosePromptLogging || graniteDiagnostic && i < 5)
            {
                Console.Error.WriteLine($"[{(graniteDiagnostic ? "GRANITE4-DIAG" : "DBG")}] tok={i} next={next}('{tok.Decode([next])}') stop={sp.StopTokenIds.Contains(next)} top5:{FormatTopLogits(logits, 5)}");
            }
            if (i == 0 && Environment.GetEnvironmentVariable("STINGRAY_DBG_TOKEN_RANK") is { } watchTokStr
                && int.TryParse(watchTokStr, out int watchTok) && watchTok < logits.Length)
            {
                float watchLogit = logits[watchTok];
                int rank = 1;
                for (int j = 0; j < logits.Length; j++)
                    if (logits[j] > watchLogit) rank++;
                Console.Error.WriteLine($"[DBG-RANK] token={watchTok} logit={watchLogit:F4} rank={rank}/{logits.Length}");
            }
            if (sp.StopTokenIds.Contains(next)) break;
            // Advance the constraint by every emitted token (so it can detect a tool call beginning
            // and ending) and capture the raw stream for post-generation tool-call parsing.
            sp.Constraint?.Accept(next);
            captureTokens?.Add(next);
            // Counter resets on each <think> open (in case the model opens multiple blocks)
            // and counts every token emitted while inThinking is true on entry, including
            // the boundary tokens themselves — that keeps the budget predictable: N tokens
            // of reasoning content trip the force-close on iteration N+1.
            if (next == s_thinkTokenId) thinkingTokenCount = 0;
            else if (inThinking) thinkingTokenCount++;
            if (EmitToken(next, tok, streamDec, ref inThinking, hideThinking, sink)) generated++;
            totalDecoded++;
            recentTokens.Add(next);
            if (recentTokens.Count > historyCap) recentTokens.RemoveAt(0);
            if (profDecode) OpenTail.Stingray.Engine.DecodeProfileTimers.AddNonTrunk(System.Diagnostics.Stopwatch.GetTimestamp() - nonTrunkStart);
            logits = forward(next, startPos + i);
        }
        // When hiding reasoning, the decoder may still hold an in-thinking tail —
        // flush it through the same gate so nothing leaks to stdout.
        var tail = streamDec.Flush();
        if (!(hideThinking && inThinking))
        {
            if (sink is null) Console.Write(tail);
            else if (tail.Length > 0) sink(tail);
        }
        // Style reset is stdout-only; a sink receives plain text.
        if (inThinking && sink is null) Console.Write(AnsiCodes.Esc + "[0m");
        return (generated, totalDecoded);
    }

    /// <summary>
    /// Writes <paramref name="next"/> to <paramref name="sink"/> (stdout when null), handling the
    /// &lt;think&gt;/&lt;/think&gt; boundary tokens and dim-styling everything inside. Returns true when
    /// the emitted token counts toward the visible decode total (i.e. not a thinking-mode token
    /// and not a boundary marker).
    /// </summary>
    /// <param name="sink">
    /// Receives decoded text instead of stdout. The interactive TUI passes its own sink so tokens
    /// stream into the transcript; ANSI styling is suppressed for sinks since the TUI applies its
    /// own styling per transcript role.
    /// </param>
    private static bool EmitToken(int next, GgufTokenizer tok, Utf8StreamDecoder streamDec, ref bool inThinking, bool hideThinking = false, Action<string>? sink = null)
    {
        void Emit(string text)
        {
            if (sink is null) Console.Write(text);
            else if (text.Length > 0) sink(text);
        }

        // Styling escapes are stdout-only; a sink receives plain text.
        void EmitStyle(string ansi)
        {
            if (sink is null) Console.Write(ansi);
        }

        // Reasoning boundary tokens are always consumed (never printed), with the state flip
        // gated on the current mode so a malformed double-open or a bare close — e.g. Gemma 4's
        // post-tool prompt primes <|channel>, so the answer pass emits a lone <channel|> with no
        // open — is swallowed rather than rendered as a literal marker (issue #304).
        if (next == s_thinkTokenId)
        {
            if (!inThinking)
            {
                inThinking = true;
                // No trailing \n: the model often emits its own leading newline inside the block,
                // and a double break before the reasoning starts looks noisy.
                EmitStyle(AnsiCodes.Esc + "[2m");
                Emit("[Thinking...] ");
            }
            return false;
        }
        if (next == s_endThinkTokenId)
        {
            if (inThinking)
            {
                inThinking = false;
                EmitStyle(AnsiCodes.Esc + "[0m");
                Emit(((char)10).ToString());
            }
            return false;
        }
        // Stream through the same UTF-8 decoder regardless of mode so multibyte
        // sequences split across thinking/visible boundaries stay intact. When
        // hideThinking is set we still consume the bytes (so the decoder stays in
        // sync across the boundary) but discard the rendered output.
        var rendered = streamDec.Append(tok.DecodeBytes(next));
        if (!(hideThinking && inThinking))
            Emit(rendered);
        return !inThinking;
    }
}
