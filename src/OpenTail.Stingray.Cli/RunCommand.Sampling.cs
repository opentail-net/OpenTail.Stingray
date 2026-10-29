using OpenTail.Stingray.Core.Grammar;

namespace OpenTail.Stingray.Cli;

public sealed partial class RunCommand
{
    /// <summary>
    /// Builds a whole-turn <see cref="JsonSchemaOutputConstraint"/> (issue #423 follow-up) from
    /// <c>--json-schema</c> (inline) or <c>--json-schema-file</c>/<c>--jf</c> (file), or leaves
    /// <paramref name="constraint"/> <c>null</c> when neither flag is given.
    /// <paramref name="ordered"/> maps <c>--json-schema-ordered</c> (issue #425: properties in
    /// declaration order). Returns <c>false</c> (with <paramref name="error"/> set) on mutual
    /// exclusivity, a missing/unreadable file, malformed JSON, or a schema that can't be compiled
    /// to a constraint -- an explicit schema request is a hard requirement, so failures are
    /// reported rather than silently ignored.
    /// </summary>
    internal static bool TryLoadJsonSchemaConstraint(
        string? inlineSchema, string? schemaFilePath, GrammarVocabulary vocab,
        out ITokenConstraint? constraint, out string? error, bool ordered = false)
    {
        constraint = null;
        error = null;

        if (inlineSchema is { Length: > 0 } && schemaFilePath is { Length: > 0 })
        {
            error = "--json-schema and --json-schema-file/--jf are mutually exclusive; pass only one.";
            return false;
        }

        string schemaText;
        if (schemaFilePath is { Length: > 0 })
        {
            if (!File.Exists(schemaFilePath))
            {
                error = $"schema file not found: {schemaFilePath}";
                return false;
            }
            try
            {
                schemaText = File.ReadAllText(schemaFilePath);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                          or System.Security.SecurityException or NotSupportedException)
            {
                error = $"could not read --json-schema-file: {ex.Message}";
                return false;
            }
        }
        else if (inlineSchema is { Length: > 0 })
        {
            schemaText = inlineSchema;
        }
        else
        {
            // An explicit ordered request without a schema is user error, not a no-op -- silently
            // generating unconstrained output would violate the fail-loudly contract above.
            if (ordered)
            {
                error = "--json-schema-ordered requires --json-schema or --json-schema-file/--jf.";
                return false;
            }
            return true;   // neither flag given -- no-op
        }

        JsonElement root;
        try
        {
            using var doc = JsonDocument.Parse(schemaText);
            root = doc.RootElement.Clone();
        }
        catch (JsonException ex)
        {
            error = $"could not parse JSON schema: {ex.Message}";
            return false;
        }

        try
        {
            var schemaObject = ToolSchema.FromOpenAiFunction("_", root).Arguments;
            constraint = new JsonSchemaOutputConstraint(vocab, schemaObject, ordered);
            return true;
        }
        catch (ArgumentException ex)
        {
            error = $"--json-schema could not be compiled: {ex.Message}";
            return false;
        }
    }

    /// <summary>
    /// Everything between "the model is loaded" and "generation starts" that depends only on the arguments and the tokenizer:
    /// <c>--tools</c> / <c>--tool-grammar</c>, <c>--json-schema</c>, <c>-e</c> prompt escapes, the <c>--chat-template</c> override,
    /// <c>--logit-bias</c>, then the sampling parameters and the RNG. Sets the tool / template statics the decode loop reads.
    /// Extracted verbatim from <see cref="Execute"/>; false means a clean failure already reported on stderr.
    /// </summary>
    private static bool TryBuildSampling(Settings settings, GgufTokenizer tokenizer,
                                         out SamplingParams samplingParams, out Random random, out int exitCode)
    {
        // ── Tool calling (optional) ───────────────────────────────────────────────
        // --tools advertises OpenAI-format tool definitions to the model via its chat template;
        // --tool-grammar additionally constrains the argument bytes to the supplied JSON Schemas
        // (issue #374) for families with constraint support. Both are single-prompt features.
        // Shared by --json-schema/--json-schema-file below (issue #423 follow-up) -- constructing
        // this is free even when unused (the expensive per-token byte table builds lazily).
        var grammarVocab = new GrammarVocabulary(tokenizer);
        List<ToolSchema>? toolSchemas = null;
        ITokenConstraint? toolConstraint = null;
        int[] toolBoundaryStops = [];
        s_tools = null;   // reset: this run advertises tools only if --tools is given (no leak across in-process runs)
        if (settings.ToolsPath is { Length: > 0 } toolsPath)
        {
            if (!File.Exists(toolsPath))
            {
                AnsiConsole.ErrorLine($"[red]Error:[/] tools file not found: {Markup.Escape(toolsPath)}");
                { samplingParams = null!; random = null!; exitCode = ExitCodes.Failure; return false; }
            }
            try
            {
                (s_tools, toolSchemas) = LoadTools(toolsPath);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                          or System.Security.SecurityException or NotSupportedException
                                          or JsonException or FormatException)
            {
                AnsiConsole.ErrorLine($"[red]Error:[/] could not parse --tools file: {Markup.Escape(ex.Message)}");
                { samplingParams = null!; random = null!; exitCode = ExitCodes.Failure; return false; }
            }
            AnsiConsole.MarkupLine($"[dim]Loaded {toolSchemas.Count} tool(s) from {Markup.Escape(Path.GetFileName(toolsPath))}.[/]");

            var adapter = ToolCallAdapterRegistry.Get(s_arch);

            // Halt right after the tool call(s) instead of running into a hallucinated trailing
            // turn (issue #304): add the adapter's tool-boundary markers (Gemma 4: <|tool_response>)
            // to the stop set, resolved against the vocab.
            toolBoundaryStops = adapter.ToolBoundaryStopMarkers
                .Select(m => tokenizer.SpecialTokens.TryGetValue(m, out int id) ? id : -1)
                .Where(id => id > 0)
                .Distinct()
                .ToArray();

            if (settings.ToolGrammar)
            {
                toolConstraint = adapter.BuildArgumentConstraint(toolSchemas, grammarVocab);
                AnsiConsole.MarkupLine(toolConstraint is not null
                    ? "[dim]Tool-call arguments are grammar-constrained (issue #374).[/]"
                    : $"[yellow]Warning:[/] --tool-grammar has no effect for arch '{s_arch}' (no constraint support, or no supplied tool is constrainable); arguments generate unconstrained.");
            }
        }
        else if (settings.ToolGrammar)
        {
            AnsiConsole.MarkupLine("[yellow]Warning:[/] --tool-grammar requires --tools (a schema to constrain against); ignoring.");
        }

        // ── JSON-Schema-constrained output (issue #423 follow-up) ─────────────────
        if (!TryLoadJsonSchemaConstraint(settings.JsonSchema, settings.JsonSchemaFile, grammarVocab,
                out ITokenConstraint? jsonSchemaConstraint, out string? jsonSchemaError,
                ordered: settings.JsonSchemaOrdered))
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] {Markup.Escape(jsonSchemaError!)}");
            { samplingParams = null!; random = null!; exitCode = ExitCodes.Failure; return false; }
        }
        if (jsonSchemaConstraint is not null)
        {
            AnsiConsole.MarkupLine("[dim]Response is constrained to the supplied JSON schema.[/]");
            if (toolConstraint is not null)
                AnsiConsole.MarkupLine(
                    "[yellow]Warning:[/] --json-schema/--json-schema-file combined with --tool-grammar likely " +
                    "makes tool calls unreachable (the schema constrains the whole response from the first " +
                    "token) — use one or the other.");
        }

        // -e/--escape: expand escape sequences in the prompt before it is templated or tokenized.
        if (settings.Escape && settings.Prompt is not null)
            settings.Prompt = ProcessEscapeSequences(settings.Prompt);

        // --chat-template: raw Jinja only. Named shortcuts are refused rather than approximated —
        // a hand-written chatml/llama3 template loads, runs, and degrades output with no error,
        // which is the worst failure mode available here.
        if (settings.ChatTemplateOverride is { Length: > 0 } tmplOverride)
        {
            string trimmed = tmplOverride.Trim();
            if (!trimmed.Contains("{{", StringComparison.Ordinal) && !trimmed.Contains("{%", StringComparison.Ordinal))
            {
                AnsiConsole.ErrorLine($"[red]Error:[/] --chat-template '{Markup.Escape(trimmed)}' is not Jinja source. " +
                    "Named shortcuts are not supported: pass the model's raw Jinja2 template, or omit the flag to use " +
                    "the one embedded in the model.");
                { samplingParams = null!; random = null!; exitCode = ExitCodes.Failure; return false; }
            }
            s_jinja = new JinjaChatTemplate(trimmed);
            AnsiConsole.MarkupLine("[dim]Chat template overridden via --chat-template.[/]");
        }

        if (UsesUnsupportedChatMlFallback(s_arch, s_jinja is not null,
                Environment.GetEnvironmentVariable("STINGRAY_RAW_PROMPT") == "1", tokenizer.SpecialTokens))
            AnsiConsole.MarkupLine("[yellow]Warning:[/] this model has no chat template and its vocabulary has no " +
                "<|im_start|> token, so the ChatML fallback wrapper will be read as plain text. For a base model, set " +
                "STINGRAY_RAW_PROMPT=1 for raw text completion.");

        // --logit-bias: TOKEN_ID+BIAS / TOKEN_ID-BIAS entries into a bias map.
        IReadOnlyDictionary<int, float>? logitBiasMap = null;
        if (settings.LogitBias is { Length: > 0 } biasEntries
            && !TryParseLogitBias(biasEntries, out logitBiasMap, out string? biasError))
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] --logit-bias: {Markup.Escape(biasError!)}");
            { samplingParams = null!; random = null!; exitCode = ExitCodes.Failure; return false; }
        }

        var sp = new SamplingParams
        {
            Temperature = settings.Temperature,
            TopK = settings.TopK,
            TopP = settings.TopP,
            MinP = settings.MinP,
            MaxNewTokens = settings.NPredict,
            StopTokenIds = toolBoundaryStops.Length > 0
                ? [.. BuildStopTokenIds(tokenizer), .. toolBoundaryStops]
                : [.. BuildStopTokenIds(tokenizer)],
            RepetitionPenalty = settings.RepPenalty,
            PresencePenalty = settings.PresencePenalty ?? 0f,
            FrequencyPenalty = settings.FrequencyPenalty ?? 0f,
            LogitBias = logitBiasMap,
            SpecType = ParseSpecType(settings.SpecTypeStr),
            SpecDraftNMax = settings.SpecDraftNMax,
            SpecDraftNMin = settings.SpecDraftNMin,
            SpecDraftPMin = settings.SpecDraftPMin,
            Constraint = TokenConstraints.Combine(jsonSchemaConstraint, toolConstraint),
        };
        var rng = settings.Seed >= 0 ? new Random(settings.Seed) : new Random();

        samplingParams = sp;
        random = rng;
        exitCode = ExitCodes.Success;
        return true;
    }

    /// <summary>
    /// Bounds generation so prompt + output fit the active context. ForwardPass sizes its
    /// attention-score and RoPE scratch from the same ceiling but its KV cache is far larger, so
    /// decoding past it is an out-of-bounds native access; the engine now throws, and this is what
    /// keeps ordinary "context full" from reaching that throw. Returns <paramref name="sp"/>
    /// unchanged when the request already fits.
    /// </summary>
    internal static SamplingParams ClampToRemainingContext(SamplingParams sp, int promptTokens, int maxContextLength)
    {
        int room = maxContextLength - promptTokens;
        if (room >= sp.MaxNewTokens) return sp;

        AnsiConsole.MarkupLine(
            $"[yellow]Note:[/] --n-predict {sp.MaxNewTokens} exceeds the {maxContextLength}-token context " +
            $"with a {promptTokens}-token prompt; generating at most {room} token{(room == 1 ? "" : "s")}. " +
            $"Raise [yellow]--ctx-size[/] for a longer response.");
        return sp with { MaxNewTokens = room };
    }

    /// <summary>
    /// Builds the stop token ID list. Delegates to <see cref="GgufTokenizer.EogTokenIds"/> —
    /// the single source of truth for end-of-generation tokens (EOS plus the end-of-turn
    /// variants used by Llama 3/4, Mistral, Phi, Gemma, etc.) — so the CLI and server stop on
    /// exactly the same set. Notably this is what lets the CLI halt on Gemma 4's <c>&lt;eos&gt;</c>
    /// (id 1, distinct from its configured EOS <c>&lt;turn|&gt;</c> at id 106) instead of decoding
    /// it as literal text.
    /// </summary>
    private static IReadOnlyList<int> BuildStopTokenIds(GgufTokenizer tokenizer) => tokenizer.EogTokenIds;

    /// <summary>
    /// Resolves <c>--repeat-last-n</c> into the size the decode loop's history buffer must hold.
    /// </summary>
    /// <remarks>
    /// This exists as a named, testable function because the previous version hard-coded the
    /// buffer at 64 while accepting any value: <c>--repeat-last-n 256</c> and <c>-1</c> both
    /// silently behaved as 64, and the tests could not see it because they only asserted that the
    /// flag BOUND. The buffer size is the behaviour; assert this, not the parsed integer.
    /// </remarks>
    /// <param name="repeatLastN">0 = disabled, -1 (or any negative) = full context, otherwise the window.</param>
    internal static int ResolvePenaltyHistoryCap(int repeatLastN) => repeatLastN switch
    {
        0 => 0,                 // no history retained; the penalty never sees a token
        < 0 => int.MaxValue,    // full context: never evict
        _ => repeatLastN,
    };

    /// <summary>
    /// Parse llama.cpp-style logit-bias entries (<c>TOKEN_ID+BIAS</c> / <c>TOKEN_ID-BIAS</c>,
    /// e.g. <c>1234+1.5</c>, <c>5678-100</c>) into a token-id → bias map. Invariant culture, so a
    /// machine's locale cannot change how a command line is interpreted.
    /// </summary>
    internal static bool TryParseLogitBias(
        string[] entries,
        out IReadOnlyDictionary<int, float>? result,
        out string? error)
    {
        var map = new Dictionary<int, float>(entries.Length);
        foreach (string entry in entries)
        {
            // Scan back for the sign that separates id from bias. Start at 1, not 0, so a leading
            // '-' is treated as part of a (rejected) negative token id rather than as the separator.
            int sep = -1;
            for (int i = entry.Length - 1; i > 0; i--)
                if (entry[i] is '+' or '-') { sep = i; break; }

            if (sep < 0)
            {
                result = null;
                error = $"invalid entry '{entry}': expected TOKEN_ID+BIAS or TOKEN_ID-BIAS (e.g. '1234+1.5').";
                return false;
            }

            if (!int.TryParse(entry[..sep], System.Globalization.NumberStyles.Integer,
                    System.Globalization.CultureInfo.InvariantCulture, out int tokenId) || tokenId < 0)
            {
                result = null;
                error = $"invalid token id '{entry[..sep]}' in '{entry}': must be a non-negative integer.";
                return false;
            }
            // entry[sep..] keeps the sign, so "+1.5" and "-100" both parse directly.
            if (!float.TryParse(entry[sep..], System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out float bias))
            {
                result = null;
                error = $"invalid bias value '{entry[sep..]}' in '{entry}': must be a number (e.g. +1.5 or -100).";
                return false;
            }
            map[tokenId] = bias;
        }
        result = map;
        error = null;
        return true;
    }
}
