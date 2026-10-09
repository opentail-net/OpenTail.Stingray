using OpenTail.Stingray.Engine.Verification;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Cli;

/// <summary>
/// Admission-decision support for a GGUF whose <c>general.architecture</c> string is not yet in
/// <see cref="ModelCompatibility"/>'s allowlist.
///
/// <para>This session's own backlog (see <c>docs/00-current-work.md</c>) repeatedly found the same
/// shape of result over a dozen "new" architectures: most needed zero new forward-pass code because
/// generic tensor-presence dispatch already covered them, and the actual blocker was almost always
/// the tokenizer axis (missing <c>tokenizer.ggml.merges</c>, scores-only SPM, etc.) — discovered each
/// time by manually downloading a checkpoint, running it under a diagnostic bypass, and comparing
/// against an external llama.cpp reference by hand. This command automates the mechanical half of
/// that process (tensor-inventory triage + a formatted greedy-token comparison against a supplied
/// reference), so the remaining manual work is only capturing the reference itself.</para>
///
/// <para>It does NOT decide admission on its own — a real reference token sequence (from llama.cpp
/// or another independent implementation) is still required for a trustworthy verdict. Without one,
/// this only reports structural triage: tokenizer shape and whether the run produced finite,
/// non-degenerate logits at all.</para>
/// </summary>
public sealed class AdmitArchCommand : Command<AdmitArchCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-m|--model <PATH>")]
        [Description("GGUF to evaluate")]
        public string ModelPath { get; init; } = "";

        [CommandOption("-p|--prompt <TEXT>")]
        [Description("Raw prompt to tokenize and greedy-decode (no chat template applied)")]
        public string Prompt { get; init; } = "The capital of France is";

        [CommandOption("-n|--tokens <N>")]
        [Description("Number of greedy tokens to generate")]
        public int MaxTokens { get; init; } = 8;

        [CommandOption("--reference-tokens <IDS>")]
        [Description("Comma-separated reference token ids (from llama.cpp or another oracle) to compare against, e.g. from `llama-server .../completion` with return_tokens:true")]
        public string? ReferenceTokens { get; init; }

        [CommandOption("--golden <PATH>")]
        [Description("Golden reference file (written by capture-golden): recorded prompt token ids and llama.cpp continuation. Runs BEFORE the allowlist gate, so an already-admitted architecture can be re-verified, and no tokenizer is involved.")]
        public string? Golden { get; init; }

        [CommandOption("--ctx-size <N>")]
        public int CtxSize { get; init; } = 512;
    }

    protected override int Execute(Settings settings, CancellationToken cancellation)
    {
        if (!ModelPathResolver.TryRequireModelFile(settings.ModelPath, out int modelFileExit))
        {
            return modelFileExit;
        }

        using var model = GgufModel.Open(settings.ModelPath);
        string arch = model.Metadata.TryGetValue("general.architecture", out var a) ? Convert.ToString(a) ?? "" : "";

        AnsiConsole.MarkupLine($"[bold]Architecture:[/] {Markup.Escape(arch)}");

        if (settings.Golden is { Length: > 0 })
            return RunGolden(model, arch, settings);

        bool alreadySupported = ModelCompatibility.IsTextGenerationArchitectureSupported(arch);
        if (alreadySupported)
        {
            AnsiConsole.MarkupLine("[green]Already allowlisted[/] in ModelCompatibility — nothing to admit.");
            return 0;
        }
        AnsiConsole.MarkupLine("[yellow]Not allowlisted.[/] Running structural triage before attempting a real forward pass.");
        AnsiConsole.WriteLine();

        // ── Tokenizer-axis triage. Historically the single most common blocker (see the
        // xverse/minicpm/internlm2/baichuan/ernie4_5/orion notes in ModelCompatibility.cs): a GGUF
        // declaring tokenizer.ggml.model=llama with tokenizer.ggml.scores but no
        // tokenizer.ggml.merges tokenizes to near-character fragments unless SpmMergePiecesByScore
        // (already implemented) is reached — but only for that specific shape. Surface it directly
        // rather than making every future admission re-discover this by hand.
        string tokModel = model.Metadata.TryGetValue("tokenizer.ggml.model", out var tm) ? Convert.ToString(tm) ?? "" : "(absent)";
        bool hasMerges = model.Metadata.ContainsKey("tokenizer.ggml.merges");
        bool hasScores = model.Metadata.ContainsKey("tokenizer.ggml.scores");
        AnsiConsole.MarkupLine($"[bold]Tokenizer:[/] tokenizer.ggml.model={Markup.Escape(tokModel)}, merges={hasMerges}, scores={hasScores}");
        if (tokModel == "llama" && !hasMerges && hasScores)
            AnsiConsole.MarkupLine("  [dim]-> scores-only SPM shape (the minicpm/xverse/orion class). Already handled by GgufTokenizer.SpmMergePiecesByScore.[/]");
        else if (tokModel == "llama" && !hasMerges && !hasScores)
            AnsiConsole.MarkupLine("  [yellow]-> neither merges nor scores present — likely fragments to near-character level; check tokenizer output below carefully.[/]");
        else if (tokModel == "t5")
            AnsiConsole.MarkupLine("  [dim]-> Unigram-LM (real llama.cpp LLAMA_VOCAB_TYPE_UGM) — routed through UnigramTokenizer.FromGgufVocab.[/]");

        // ── Tensor-shape triage: report the per-layer tensor suffix inventory (same grouping
        // ListTensorsCommand --summary uses) so a reviewer can eyeball whether every tensor this
        // checkpoint carries is one this engine's generic dispatch already knows how to read
        // (fused qkv, separate norm-with-bias, gated vs. non-gated FFN, etc.) before spending time
        // on a real run.
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[bold]Layer-0 tensor inventory[/] (compare shapes/names against a known-working architecture's):");
        foreach (var t in model.Tensors.Where(t => t.Name.StartsWith("blk.0.", StringComparison.Ordinal))
                     .OrderBy(t => t.Name, StringComparer.Ordinal))
            AnsiConsole.MarkupLine($"  {Markup.Escape(t.Name),-32} {t.DType,-10} [{string.Join(",", t.Dimensions.Take(t.NDimensions))}]");

        // ── Attempt a real run under the existing diagnostic bypass. This is the same mechanism
        // --allow-unverified-arch already exercises in RunCommand — reused here directly rather than
        // re-implemented, so this command's verdict reflects the identical code path a real
        // admission would run under.
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[bold]Attempting a real forward pass[/] (bypassing the architecture gate)...");
        List<int> generated;
        IReadOnlyList<int> promptTokens;
        try
        {
            var hp = ArchitectureModelResolver.ResolveHyperparams(model);
            var tokenizer = GgufTokenizer.FromGgufModel(model);
            // Match llama-server's raw completion (the reference this is compared against): BOS first when the model's
            // metadata asks for it. Without it, BOS-dependent models (LFM2) degenerate and every verdict is wrong.
            var ids = tokenizer.Encode(settings.Prompt).ToList();
            if (tokenizer.AddBosToken && tokenizer.BosTokenId >= 0 && (ids.Count == 0 || ids[0] != tokenizer.BosTokenId))
                ids.Insert(0, tokenizer.BosTokenId);
            promptTokens = ids;
            using var backend = new CpuBackend();
            using var fwd = new ForwardPass(model, backend, hp, maxContextLength: settings.CtxSize);

            var logits = fwd.Prefill(promptTokens);
            if (logits.IsEmpty || logits.ToArray().Any(float.IsNaN))
            {
                AnsiConsole.MarkupLine("[red]REJECT:[/] prefill produced empty or NaN logits — the forward pass does not even run structurally sound on this architecture.");
                return 1;
            }

            generated = [];
            int nextTok = Argmax(logits);
            generated.Add(nextTok);
            int pos = promptTokens.Count;
            for (int i = 1; i < settings.MaxTokens; i++)
            {
                var stepLogits = fwd.Forward(nextTok, pos);
                if (stepLogits.ToArray().Any(float.IsNaN))
                {
                    AnsiConsole.MarkupLine($"[red]REJECT:[/] logits went NaN at generated position {i}.");
                    return 1;
                }
                nextTok = Argmax(stepLogits);
                generated.Add(nextTok);
                pos++;
            }
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            AnsiConsole.MarkupLine($"[red]REJECT:[/] forward pass threw {ex.GetType().Name}: {Markup.Escape(ex.Message)}");
            return 1;
        }

        AnsiConsole.MarkupLine($"[green]Ran cleanly.[/] Prompt tokens: [{string.Join(", ", promptTokens)}]");
        AnsiConsole.MarkupLine($"Greedy continuation ({settings.MaxTokens} tokens): [{string.Join(", ", generated)}]");

        if (settings.ReferenceTokens is not { Length: > 0 })
        {
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine("[yellow]No --reference-tokens supplied[/] — cannot render an ADMIT/REJECT verdict.");
            AnsiConsole.MarkupLine("Capture a real reference (e.g. `llama-server .../completion` with `return_tokens:true`,");
            AnsiConsole.MarkupLine("or `llama-tokenize` + `llama-cli --temp 0 --top-k 1 --seed 0`), then rerun with");
            AnsiConsole.MarkupLine("[yellow]--reference-tokens id1,id2,...[/] to get a pasteable ModelCompatibility.cs receipt.");
            return 0;
        }

        int[] reference;
        try
        {
            reference = settings.ReferenceTokens.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Select(int.Parse).ToArray();
        }
        catch (FormatException)
        {
            AnsiConsole.ErrorLine("[red]Error:[/] --reference-tokens must be a comma-separated list of integers.");
            return 1;
        }

        int matched = 0;
        while (matched < reference.Length && matched < generated.Count && reference[matched] == generated[matched])
            matched++;

        AnsiConsole.WriteLine();
        bool fullMatch = matched == reference.Length && reference.Length == generated.Count;
        if (fullMatch)
        {
            AnsiConsole.MarkupLine($"[green bold]ADMIT[/] — full {matched}-of-{matched}-token exact greedy match, zero divergence.");
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine("[dim]Paste-ready allowlist comment:[/]");
            Console.WriteLine($"// {arch} — ADMITTED {DateTime.UtcNow:yyyy-MM-dd}, full {matched}-of-{matched}-token exact");
            Console.WriteLine($"// greedy match (automated via `stingray admit-arch`), tokenizer.ggml.model={tokModel},");
            Console.WriteLine($"// merges={hasMerges}, scores={hasScores}. Verify the checkpoint's license bucket");
            Console.WriteLine("// before deciding whether a permanent parity test can be committed.");
            Console.WriteLine($"\"{arch}\",");
        }
        else
        {
            AnsiConsole.MarkupLine($"[red bold]NOT YET ADMISSIBLE[/] — matched {matched} of {reference.Length} reference tokens before diverging.");
            if (matched < generated.Count && matched < reference.Length)
                AnsiConsole.MarkupLine($"  First divergence at position {matched}: engine={generated[matched]}, reference={reference[matched]}.");
            AnsiConsole.MarkupLine("  This is either a real architecture/tokenizer gap, or ordinary quantization-sensitivity");
            AnsiConsole.MarkupLine("  near a close logit tie (dump top-5 logits at the divergence position before concluding either way).");
        }

        return fullMatch ? 0 : 1;
    }

    /// <summary>
    /// Runs a golden reference against this model: the recorded prompt ids go in (no tokenizer), the greedy continuation is compared with the
    /// recorded one with near-tie classification, and the model-independent stepwise-vs-prefill check runs too. Exit 0 = pass (exact or
    /// near-tie only), 1 = diverged or inconsistent.
    /// </summary>
    private static int RunGolden(GgufModel model, string arch, Settings settings)
    {
        GoldenFile golden;
        try { golden = GoldenFile.Load(settings.Golden!); }
        catch (Exception ex) when (ex is IOException or InvalidDataException or System.Text.Json.JsonException or UnauthorizedAccessException)
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] cannot read golden '{Markup.Escape(settings.Golden!)}': {Markup.Escape(ex.Message)}");
            return 1;
        }

        if (!string.Equals(golden.Architecture, arch, StringComparison.OrdinalIgnoreCase))
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] the golden is for architecture '{Markup.Escape(golden.Architecture)}' but this file declares '{Markup.Escape(arch)}'.");
            return 1;
        }

        string fileName = Path.GetFileName(settings.ModelPath);
        AnsiConsole.MarkupLine($"[bold]Golden:[/] {Markup.Escape(Path.GetFileName(settings.Golden!))} ({golden.Cases.Count} case(s)), recorded on " +
            $"{Markup.Escape(golden.Model.FileName)} by {Markup.Escape(golden.Reference.Engine)} {Markup.Escape(golden.Reference.Build ?? "")}");
        if (!string.Equals(fileName, golden.Model.FileName, StringComparison.OrdinalIgnoreCase))
            AnsiConsole.MarkupLine($"[yellow]Note:[/] this file is named '{Markup.Escape(fileName)}', the golden was recorded on '{Markup.Escape(golden.Model.FileName)}'. " +
                "A golden is evidence for the file it was recorded on; a different conversion can legitimately differ.");
        AnsiConsole.MarkupLine("[dim]Checking the file against the golden's SHA-256 (cached beside the model after the first time)...[/]");
        var pin = ModelFingerprinter.CheckPin(golden, settings.ModelPath);
        string pinColor = pin.Status switch { PinStatus.Verified => "green", PinStatus.Mismatch => "yellow", _ => "dim" };
        AnsiConsole.MarkupLine($"[{pinColor}]{Markup.Escape(pin.Describe(fileName))}[/]");

        using var source = new GoldenForwardPassSource(model, Math.Max(settings.CtxSize, 1024));
        using var scope = new GoldenEngineSettingsScope(golden.EngineSettings);
        var hpExpect = HyperparameterExpectations.Check(ArchitectureModelResolver.ResolveHyperparams(model), golden.ExpectedHyperparameters);
        foreach (var f in hpExpect) AnsiConsole.ErrorLine($"[red]Hyperparameter guard failed:[/] {Markup.Escape(f)}");
        var result = GoldenParityRunner.Run(golden, source.Create, new ParityOptions { RequireStepwiseArgmaxAgreement = false });

        AnsiConsole.WriteLine();
        Console.WriteLine(result.Format());
        switch (result.Verdict)
        {
            case CaseVerdict.Exact when result.Passed:
                AnsiConsole.MarkupLine("[green bold]GOLDEN MATCH[/] — every recorded token reproduced exactly, and decode vs prefill within the bound.");
                break;
            case CaseVerdict.NearTie when result.Passed:
                AnsiConsole.MarkupLine("[yellow bold]GOLDEN MATCH (near-tie)[/] — differs only where the reference itself was undecided (margin below the confident threshold) or the logits were within the near-tie tolerance.");
                break;
            default:
                AnsiConsole.MarkupLine("[red bold]NOT YET ADMISSIBLE[/] — see the divergence above (check CLAUDE.md rule 8 and docs/reference/numerics-investigation-method.md before concluding).");
                break;
        }

        if (result.Passed && hpExpect.Count == 0)
        {
            if (pin.Status == PinStatus.Verified)
            {
                AnsiConsole.MarkupLine("[dim]Paste-ready evidence line for the descriptor / STATUS row:[/]");
                Console.WriteLine($"// {arch} — verified {DateTime.UtcNow:yyyy-MM-dd} against golden {Path.GetFileName(settings.Golden!)}: " +
                    $"{golden.Reference.Engine} {golden.Reference.Build}, sha256 {golden.Model.Sha256}, {result.Verdict}.");
                return 0;
            }
            if (pin.Status == PinStatus.Mismatch)
            {
                AnsiConsole.MarkupLine("[yellow bold]UNPINNED PARITY RESULT[/] — forward pass matches, but file SHA-256 does NOT match the golden. Cannot admit architecture on an unpinned file.");
                return 1;
            }
            // NotRecorded
            AnsiConsole.MarkupLine("[yellow]LEGACY UNPINNED RECEIPT[/] — forward pass matches, but the golden recorded no SHA-256. Re-capture with capture-golden to produce pinned admission evidence.");
            return 1;
        }
        return 1;
    }

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }
}
