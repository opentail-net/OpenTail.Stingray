using OpenTail.Stingray.Core.Catalog;
using OpenTail.Stingray.Cli.Scout;
using OpenTail.Stingray.Core.Net;

namespace OpenTail.Stingray.Cli;

/// <summary>Answers yes/no questions. The console implementation asks; tests answer from a script.</summary>
public interface ISetupPrompt
{
    bool Confirm(string question, bool defaultYes);
}

public sealed class ConsolePrompt : ISetupPrompt
{
    public bool Confirm(string question, bool defaultYes)
    {
        AnsiConsole.Markup($"{Markup.Escape(question)} [dim]({(defaultYes ? "Y/n" : "y/N")})[/] ");
        string? answer = Console.ReadLine()?.Trim();
        if (string.IsNullOrEmpty(answer)) return defaultYes;
        return answer.StartsWith('y') || answer.StartsWith('Y');
    }
}

public sealed record SetupOutcome(bool Installed, string? Message);

/// <summary>
/// The install path shared by <c>stingray setup</c> and by task commands (<c>chat</c>, <c>speak</c>, <c>transcribe</c>) that find their model missing:
/// show what it costs, get the licence consent and the download confirmation, then install through the verifying installer.
/// Nothing is downloaded without a yes (or an explicit <paramref name="yes"/>); a non-interactive session never prompts and never downloads.
/// </summary>
public static class SetupFlow
{
    public static SetupOutcome Run(CatalogEntry entry, ModelHome home, HttpClient http, bool yes, bool acceptLicence, bool interactive, ISetupPrompt prompt, CancellationToken ct, Func<string, string?>? env = null,
        Func<CatalogEntry, CancellationToken, PreflightResult?>? feasibility = null)
    {
        long remaining = home.RemainingBytes(entry);
        SetupCommand.PrintSummary(entry, home, remaining);

        // The external-access policy is checked up front so a refusal is a clear message and not a failure halfway through a prompt.
        if (remaining > 0)
        {
            var access = ExternalAccess.Evaluate(env);
            if (!access.Allowed)
            {
                AnsiConsole.ErrorLine($"[red]Not downloading:[/] {Markup.Escape(access.Reason)}. To allow it: {Markup.Escape(access.HowToEnable ?? "see STINGRAY_ALLOW_EXTERNAL")}.");
                return new(false, access.Reason);
            }
        }

        // Does it fit this machine? Asked of the pinned remote file BEFORE any download, so a model that cannot run is not fetched first.
        if (remaining > 0 && feasibility?.Invoke(entry, ct) is { } fit)
        {
            if (fit.Verdict == PreflightVerdict.Allowed)
                AnsiConsole.MarkupLine($"[green]Fits this machine:[/] about {Markup.Escape(LoadPreflight.Gib(fit.EstimatedPeakBytes))} for a CPU run (memory {Markup.Escape(LoadPreflight.Gib(fit.BudgetBytes))}).");
            else if (!LoadPreflight.ShouldProceed(fit, ignore: false, out string? why))
            {
                AnsiConsole.MarkupLine($"[yellow]{Markup.Escape(why!)}[/]");
                if (!interactive || yes) return new(false, "does not fit this machine");
                if (!prompt.Confirm("Download anyway?", defaultYes: false)) return new(false, "does not fit this machine");
            }
            else if (why is not null)
                AnsiConsole.MarkupLine($"[dim]{Markup.Escape(why)}[/]");
        }

        if (entry.LicenceNeedsConsent && !acceptLicence)
        {
            if (!interactive)
            {
                AnsiConsole.ErrorLine("[red]This licence needs an explicit yes.[/] Read it, then rerun with [yellow]--accept-licence[/].");
                return new(false, "licence needs consent");
            }
            if (!prompt.Confirm("Do you accept this licence?", defaultYes: false))
                return new(false, "licence not accepted");
        }
        if (remaining > 0 && !yes)
        {
            if (!interactive)
            {
                AnsiConsole.MarkupLine("Not an interactive terminal: rerun with [yellow]--yes[/] to download.");
                return new(false, "not interactive");
            }
            if (!prompt.Confirm($"Download {ConsoleDownloadProgress.FormatBytes(remaining)} to {home.Root}?", defaultYes: true))
                return new(false, "download declined");
        }

        ConsoleDownloadProgress? printer = null;
        try
        {
            ModelInstaller.EnsureAsync(entry, home, http,
                onFileStart: (file, present) =>
                {
                    printer?.Finish();
                    printer = new ConsoleDownloadProgress();
                    AnsiConsole.MarkupLine(present
                        ? $"[dim]Checking[/] {Markup.Escape(file.FileName)}"
                        : $"[bold]Downloading[/] {Markup.Escape(file.FileName)} ({ConsoleDownloadProgress.FormatBytes(file.Size)})");
                },
                progress: (_, bytes, total) => printer?.Report(bytes, total),
                ct).GetAwaiter().GetResult();
            printer?.Finish();
        }
        catch (Exception ex) when (ex is IOException or HttpRequestException or TaskCanceledException)
        {
            printer?.Finish();
            AnsiConsole.ErrorLine($"[red]Error:[/] {Markup.Escape(ex.Message)}");
            if (ex is not ModelHashMismatchException)
                AnsiConsole.MarkupLine("Partial downloads are kept; rerun the same command to resume.");
            return new(false, ex.Message);
        }

        AnsiConsole.MarkupLine($"[green]Ready:[/] {Markup.Escape(entry.Id)} ({Markup.Escape(entry.Task)})");
        return new(true, null);
    }

    /// <summary>The pre-download check the CLI uses: reads the pinned remote index (about 2 MiB) and returns null when the network is off or unreachable.</summary>
    public static Func<CatalogEntry, CancellationToken, PreflightResult?> DefaultFeasibility(int contextTokens = 2048) => (entry, ct) =>
    {
        using var http = new ExternalHttpClient(userAgent: "OpenTail.Stingray/setup");
        return LoadPreflight.EvaluateCatalogEntryAsync(entry, contextTokens, http, ct).GetAwaiter().GetResult();
    };
}
