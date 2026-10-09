using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Cli;

/// <summary>
/// Runs recorded golden parity references against available models and reports a host baseline
/// (docs/2-coverage/2026-10-08-golden-parity-and-admission-tooling-plan.md Phase 7).
/// </summary>
public sealed class VerifyGoldensCommand : Command<VerifyGoldensCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-d|--dir <PATH>")]
        [Description("Directory containing .golden.json files (defaults to repo tests/.../Goldens)")]
        public string? Dir { get; init; }

        [CommandOption("-g|--golden <NAME>")]
        [Description("Filter to run only matching golden file(s) or architecture(s)")]
        public string? Golden { get; init; }

        [CommandOption("--baseline <PATH>")]
        [Description("Write per-host execution and performance baseline JSON record")]
        public string? Baseline { get; init; }

        [CommandOption("--diff <PATH>")]
        [Description("Compare current verification run against a previously recorded baseline JSON")]
        public string? Diff { get; init; }

        [CommandOption("--strict")]
        [Description("Fail with non-zero exit code if any model is unpinned, skipped, divergent, near-tie, or if requested baseline diff fails")]
        public bool Strict { get; init; }

        [CommandOption("--ctx-size <N>")]
        [Description("Context size for execution (default: 2048)")]
        public int CtxSize { get; init; } = 2048;

        [CommandOption("-v|--verbose")]
        [Description("Print detailed per-token and case breakdown")]
        public bool Verbose { get; init; }
    }

    protected override int Execute(Settings settings, CancellationToken cancellation)
    {
        string? goldensDir = ResolveGoldensDirectory(settings.Dir);
        if (goldensDir is null || !Directory.Exists(goldensDir))
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] Goldens directory not found. Specify via --dir <PATH>.");
            return 1;
        }

        var files = Directory.EnumerateFiles(goldensDir, "*.json")
            .Where(f => !Path.GetFileName(f).StartsWith('.'))
            .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
            .ToList();

        if (settings.Golden is { Length: > 0 } filter)
        {
            files = files.Where(f =>
                Path.GetFileName(f).Contains(filter, StringComparison.OrdinalIgnoreCase) ||
                Path.GetFileNameWithoutExtension(f).Contains(filter, StringComparison.OrdinalIgnoreCase)).ToList();
        }

        if (files.Count == 0)
        {
            if (settings.Strict)
            {
                AnsiConsole.ErrorLine($"[red]Error:[/] No golden files found matching criteria in: {Markup.Escape(goldensDir)}");
                return 1;
            }
            AnsiConsole.MarkupLine($"[yellow]No golden files found matching criteria in:[/] {Markup.Escape(goldensDir)}");
            return 0;
        }

        AnsiConsole.MarkupLine($"[bold]Verifying {files.Count} golden reference(s)[/] from {Markup.Escape(goldensDir)}...");
        AnsiConsole.WriteLine();

        var table = new Table()
            .AddColumn("Architecture")
            .AddColumn("Golden File")
            .AddColumn("Model File")
            .AddColumn("Pin")
            .AddColumn("Verdict")
            .AddColumn("Tokens")
            .AddColumn("Speed")
            .AddColumn("Time");

        var baselineEntries = new List<GoldenBaselineEntry>();
        int totalRun = 0;
        int passedCount = 0;
        int exactCount = 0;
        int nearTieCount = 0;
        int divergedCount = 0;
        int skippedCount = 0;
        int guardFailCount = 0;
        int crashCount = 0;
        int archMismatchCount = 0;
        int unpinnedCount = 0;
        bool commandError = false;
        bool wasCancelled = false;

        foreach (var file in files)
        {
            if (cancellation.IsCancellationRequested)
            {
                wasCancelled = true;
                break;
            }

            string goldenName = Path.GetFileName(file);
            string goldenSha256 = GoldenCaptureParsing.Sha256Hex(file);
            GoldenFile golden;
            try
            {
                golden = GoldenFile.Load(file);
            }
            catch (Exception ex)
            {
                // If it's explicitly named .golden.json, report the error. Otherwise, it might be an unrelated JSON (like a baseline).
                if (goldenName.EndsWith(".golden.json", StringComparison.OrdinalIgnoreCase))
                {
                    AnsiConsole.ErrorLine($"[red]Error loading {Markup.Escape(goldenName)}:[/] {Markup.Escape(ex.Message)}");
                    divergedCount++;
                }
                continue;
            }

            if (golden.Model is null || string.IsNullOrWhiteSpace(golden.Model.FileName) || string.IsNullOrWhiteSpace(golden.Architecture))
            {
                // Non-golden JSON file (e.g. a baseline or config file) in the same directory
                continue;
            }

            string? modelPath = ModelLocator.Find(golden.Model.FileName);
            if (modelPath is null)
            {
                skippedCount++;
                table.AddRow(
                    Markup.Escape(golden.Architecture),
                    Markup.Escape(goldenName),
                    Markup.Escape(golden.Model.FileName),
                    "[dim]Missing[/]",
                    "[dim]SKIPPED[/]",
                    "-",
                    "-",
                    "-");

                baselineEntries.Add(new GoldenBaselineEntry
                {
                    Architecture = golden.Architecture,
                    GoldenFile = goldenName,
                    GoldenSha256 = goldenSha256,
                    ModelFile = golden.Model.FileName,
                    ModelSha256 = golden.Model.Sha256,
                    PinStatus = "Missing",
                    Verdict = "Skipped",
                    Passed = false,
                    Detail = "Checkpoint missing on local drives"
                });
                continue;
            }

            totalRun++;
            var pin = ModelFingerprinter.CheckPin(golden, modelPath);
            if (pin.Status != PinStatus.Verified) unpinnedCount++;

            string pinText = pin.Status switch
            {
                PinStatus.Verified => "[green]Verified[/]",
                PinStatus.Mismatch => "[yellow]Mismatch[/]",
                _ => "[dim]NotRecorded[/]"
            };

            var sw = Stopwatch.StartNew();
            GoldenRunResult result;
            IReadOnlyList<string> guardFailures = [];
            int totalCompared = 0;
            int totalMatched = 0;

            try
            {
                using var model = GgufModel.Open(modelPath);
                string arch = Convert.ToString(model.Metadata.GetValueOrDefault("general.architecture")) ?? "";
                if (!string.Equals(golden.Architecture, arch, StringComparison.OrdinalIgnoreCase))
                {
                    sw.Stop();
                    string msg = $"Architecture mismatch: golden expects '{golden.Architecture}', checkpoint declares '{arch}'";
                    AnsiConsole.ErrorLine($"[red]Error in {Markup.Escape(goldenName)}:[/] {Markup.Escape(msg)}");
                    archMismatchCount++;

                    table.AddRow(
                        Markup.Escape(golden.Architecture),
                        Markup.Escape(goldenName),
                        Markup.Escape(golden.Model.FileName),
                        pinText,
                        "[red bold]ArchMismatch[/]",
                        "-",
                        "-",
                        $"{sw.Elapsed.TotalSeconds:F2}s");

                    baselineEntries.Add(new GoldenBaselineEntry
                    {
                        Architecture = golden.Architecture,
                        GoldenFile = goldenName,
                        GoldenSha256 = goldenSha256,
                        ModelFile = golden.Model.FileName,
                        ModelSha256 = pin.Actual ?? golden.Model.Sha256,
                        PinStatus = pin.Status.ToString(),
                        Verdict = "ArchMismatch",
                        Passed = false,
                        ElapsedSeconds = sw.Elapsed.TotalSeconds,
                        Detail = msg
                    });
                    continue;
                }

                var resolvedHp = ArchitectureModelResolver.ResolveHyperparams(model);
                guardFailures = HyperparameterExpectations.Check(resolvedHp, golden.ExpectedHyperparameters);

                using var source = new GoldenForwardPassSource(model, Math.Max(settings.CtxSize, 1024));
                using var scope = new GoldenEngineSettingsScope(golden.EngineSettings);

                result = GoldenParityRunner.Run(golden, source.Create, new ParityOptions { RequireStepwiseArgmaxAgreement = false });
                sw.Stop();

                foreach (var c in result.Cases)
                {
                    totalCompared += c.Compared;
                    totalMatched += c.Matched;
                }
            }
            catch (Exception ex)
            {
                sw.Stop();
                table.AddRow(
                    Markup.Escape(golden.Architecture),
                    Markup.Escape(goldenName),
                    Markup.Escape(golden.Model.FileName),
                    pinText,
                    "[red bold]CRASH[/]",
                    "-",
                    "-",
                    $"{sw.Elapsed.TotalSeconds:F2}s");

                crashCount++;
                baselineEntries.Add(new GoldenBaselineEntry
                {
                    Architecture = golden.Architecture,
                    GoldenFile = goldenName,
                    GoldenSha256 = goldenSha256,
                    ModelFile = golden.Model.FileName,
                    ModelSha256 = pin.Actual ?? golden.Model.Sha256,
                    PinStatus = pin.Status.ToString(),
                    Verdict = "Crash",
                    Passed = false,
                    ElapsedSeconds = sw.Elapsed.TotalSeconds,
                    Detail = ex.Message
                });
                continue;
            }

            double elapsedSec = Math.Max(sw.Elapsed.TotalSeconds, 0.001);
            double? tokPerSec = result.Timing?.DecodeTokensPerSecond;
            string speedText = tokPerSec is { } spd ? $"{spd.ToString("F1", CultureInfo.InvariantCulture)} t/s" : "-";
            string timeText = $"{elapsedSec.ToString("F2", CultureInfo.InvariantCulture)}s";
            string tokenText = $"{totalMatched}/{totalCompared}";

            bool hasGuardFailure = guardFailures.Count > 0;
            if (hasGuardFailure) guardFailCount++;

            bool hasEvidenceFailure = !result.Cases.All(c => c.EvidenceOk);

            string verdictText;
            string baselineVerdict;
            if (hasGuardFailure)
            {
                verdictText = "[red bold]GuardFail[/]";
                baselineVerdict = "GuardFail";
            }
            else if (hasEvidenceFailure)
            {
                verdictText = "[red bold]InsufficientEvidence[/]";
                baselineVerdict = "InsufficientEvidence";
                divergedCount++;
            }
            else if (!result.Passed)
            {
                verdictText = "[red bold]Diverged[/]";
                baselineVerdict = "Diverged";
                divergedCount++;
            }
            else if (result.Verdict == CaseVerdict.Exact)
            {
                verdictText = pin.Status == PinStatus.Verified ? "[green bold]Exact[/]" : "[yellow bold]Exact (Unpinned)[/]";
                baselineVerdict = "Exact";
                passedCount++;
                exactCount++;
            }
            else
            {
                verdictText = pin.Status == PinStatus.Verified ? "[yellow bold]NearTie[/]" : "[yellow]NearTie (Unpinned)[/]";
                baselineVerdict = "NearTie";
                passedCount++;
                nearTieCount++;
            }

            table.AddRow(
                Markup.Escape(golden.Architecture),
                Markup.Escape(goldenName),
                Markup.Escape(golden.Model.FileName),
                pinText,
                verdictText,
                tokenText,
                speedText,
                timeText);

            if (settings.Verbose)
            {
                AnsiConsole.WriteLine();
                AnsiConsole.MarkupLine($"[bold]Result for {Markup.Escape(goldenName)}:[/]");
                Console.WriteLine(result.Format());
                foreach (var g in guardFailures)
                    AnsiConsole.ErrorLine($"  [red]Guard failure:[/] {Markup.Escape(g)}");
            }

            baselineEntries.Add(new GoldenBaselineEntry
            {
                Architecture = golden.Architecture,
                GoldenFile = goldenName,
                GoldenSha256 = goldenSha256,
                ModelFile = golden.Model.FileName,
                ModelSha256 = pin.Actual ?? golden.Model.Sha256,
                PinStatus = pin.Status.ToString(),
                Verdict = baselineVerdict,
                Passed = result.Passed && !hasGuardFailure,
                ComparedTokens = totalCompared,
                MatchedTokens = totalMatched,
                DecodeSteps = result.Timing?.DecodeSteps ?? 0,
                DecodeTokensPerSecond = tokPerSec,
                PrefillTokensPerSecond = result.Timing?.PrefillTokensPerSecond,
                ElapsedSeconds = elapsedSec,
                StepwiseMaxAbsDiff = result.Stepwise?.MaxAbsDiff,
                Detail = hasGuardFailure ? string.Join("; ", guardFailures) : null
            });
        }

        if (wasCancelled)
        {
            AnsiConsole.ErrorLine("[red]Run was cancelled. Incomplete baseline was not saved.[/]");
            return 1;
        }

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
        string divColor = (divergedCount > 0 || archMismatchCount > 0 || crashCount > 0) ? "red" : "dim";
        string guardColor = guardFailCount > 0 ? "red" : "dim";
        string unpinnedColor = unpinnedCount > 0 ? "yellow" : "dim";
        AnsiConsole.MarkupLine(
            $"[bold]Summary:[/] [green]{passedCount} passed[/] ({exactCount} exact, {nearTieCount} near-tie), " +
            $"[{divColor}]{divergedCount + crashCount + archMismatchCount} diverged/failed[/], [{guardColor}]{guardFailCount} guard failed[/], [{unpinnedColor}]{unpinnedCount} unpinned[/], [dim]{skippedCount} skipped[/] " +
            $"(out of {files.Count} goldens).");

        var currentBaseline = new GoldenBaselineFile
        {
            StingrayVersion = StingrayBuildVersion.Value,
            Entries = baselineEntries
        };

        if (settings.Baseline is { Length: > 0 } baselineOut)
        {
            try
            {
                currentBaseline.Save(baselineOut);
                AnsiConsole.MarkupLine($"[green]Baseline recorded to:[/] {Markup.Escape(baselineOut)}");
            }
            catch (Exception ex)
            {
                AnsiConsole.ErrorLine($"[red]Error writing baseline:[/] {Markup.Escape(ex.Message)}");
                commandError = true;
            }
        }

        if (settings.Diff is { Length: > 0 } diffPath)
        {
            if (File.Exists(diffPath))
            {
                try
                {
                    var priorBaseline = GoldenBaselineFile.Load(diffPath);
                    var diffs = GoldenBaselineComparator.Compare(priorBaseline, currentBaseline);

                    AnsiConsole.WriteLine();
                    AnsiConsole.MarkupLine($"[bold]Baseline Diff Comparison[/] against {Markup.Escape(diffPath)} (recorded {Markup.Escape(priorBaseline.TimestampUtc)}):");

                    var diffTable = new Table()
                        .AddColumn("Architecture")
                        .AddColumn("Golden File")
                        .AddColumn("Prior Verdict")
                        .AddColumn("Current Verdict")
                        .AddColumn("Prior Speed")
                        .AddColumn("Current Speed")
                        .AddColumn("Status")
                        .AddColumn("Details");

                    foreach (var d in diffs)
                    {
                        string statusMarkup = d.Status switch
                        {
                            BaselineDiffStatus.Regression => "[red bold]Regression[/]",
                            BaselineDiffStatus.Improvement => "[green bold]Improvement[/]",
                            BaselineDiffStatus.SpeedDrop => "[yellow]SpeedDrop[/]",
                            BaselineDiffStatus.SpeedGain => "[green]SpeedGain[/]",
                            BaselineDiffStatus.NewEntry => "[cyan]New[/]",
                            BaselineDiffStatus.MissingInCurrent => "[dim]Missing[/]",
                            BaselineDiffStatus.IdentityChanged => "[yellow bold]IdentityChanged[/]",
                            _ => "[dim]Unchanged[/]"
                        };

                        diffTable.AddRow(
                            Markup.Escape(d.Architecture),
                            Markup.Escape(d.GoldenFile),
                            Markup.Escape(d.PriorVerdict ?? "-"),
                            Markup.Escape(d.CurrentVerdict),
                            d.PriorSpeed is { } ps ? $"{ps.ToString("F1", CultureInfo.InvariantCulture)} t/s" : "-",
                            d.CurrentSpeed is { } cs ? $"{cs.ToString("F1", CultureInfo.InvariantCulture)} t/s" : "-",
                            statusMarkup,
                            Markup.Escape(d.Description));
                    }

                    AnsiConsole.Write(diffTable);
                }
                catch (Exception ex)
                {
                    AnsiConsole.ErrorLine($"[red]Failed to compute baseline diff:[/] {Markup.Escape(ex.Message)}");
                    commandError = true;
                }
            }
            else
            {
                AnsiConsole.ErrorLine($"[red]Diff baseline file not found:[/] {Markup.Escape(diffPath)}");
                commandError = true;
            }
        }

        if (commandError) return 1;
        if (divergedCount > 0 || guardFailCount > 0 || crashCount > 0 || archMismatchCount > 0) return 1;
        if (settings.Strict && (skippedCount > 0 || unpinnedCount > 0 || nearTieCount > 0)) return 1;
        return 0;
    }

    private static string? ResolveGoldensDirectory(string? overrideDir)
    {
        if (overrideDir is { Length: > 0 }) return Path.GetFullPath(overrideDir);

        string? dir = AppContext.BaseDirectory;
        while (dir is not null && !File.Exists(Path.Combine(dir, "CLAUDE.md")) && !File.Exists(Path.Combine(dir, "OpenTail.Stingray.slnx")))
        {
            dir = Path.GetDirectoryName(dir);
        }

        if (dir is not null)
        {
            string candidate = Path.Combine(dir, "tests", "OpenTail.Stingray.Tests.ForwardPass", "Goldens");
            if (Directory.Exists(candidate)) return candidate;
        }

        string? cur = Directory.GetCurrentDirectory();
        while (cur is not null)
        {
            string candidate = Path.Combine(cur, "tests", "OpenTail.Stingray.Tests.ForwardPass", "Goldens");
            if (Directory.Exists(candidate)) return candidate;
            cur = Path.GetDirectoryName(cur);
        }

        return null;
    }
}

