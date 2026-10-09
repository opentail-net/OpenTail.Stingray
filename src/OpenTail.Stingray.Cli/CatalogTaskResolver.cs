using OpenTail.Stingray.Core.Catalog;

namespace OpenTail.Stingray.Cli;

public sealed record ResolvedModelTask(
    CatalogEntry? Entry,
    string ModelPath);

public static class CatalogTaskResolver
{
    public static bool TryResolve(
        string taskName,
        string? modelId,
        string? modelFile,
        out ResolvedModelTask? resolved,
        out string? errorMessage,
        ModelHome? customHome = null,
        CatalogEntry? entryOverride = null)
    {
        // 1. Explicit model-file override
        if (!string.IsNullOrWhiteSpace(modelFile))
        {
            if (!File.Exists(modelFile) && !Directory.Exists(modelFile))
            {
                resolved = null;
                errorMessage = $"Error: specified model file not found: {modelFile}";
                return false;
            }

            resolved = new ResolvedModelTask(null, Path.GetFullPath(modelFile));
            errorMessage = null;
            return true;
        }

        // 2. Resolve catalog entry (entryOverride exists so tests can use a tiny entry instead of a real multi-hundred-MB model)
        CatalogEntry? entry;
        if (entryOverride is not null)
        {
            entry = entryOverride;
        }
        else if (!string.IsNullOrWhiteSpace(modelId))
        {
            entry = ModelCatalog.Find(modelId);
            if (entry is null)
            {
                var validOptions = string.Join(", ", ModelCatalog.Entries.Select(e => e.Id));
                resolved = null;
                errorMessage = $"Error: unknown model '{modelId}'. Available catalog models: {validOptions}";
                return false;
            }
        }
        else
        {
            entry = ModelCatalog.DefaultFor(taskName);
            if (entry is null)
            {
                resolved = null;
                errorMessage = $"Error: no default catalog model defined for task '{taskName}'.";
                return false;
            }
        }

        // 3. Check installation state in model home
        var home = customHome ?? ModelHome.Default();
        var state = home.StateOf(entry);

        if (state == InstallState.Missing)
        {
            resolved = null;
            errorMessage = $"Model for {taskName} ('{entry.Id}') is not installed.\nRun: stingray setup {taskName}";
            return false;
        }

        if (state == InstallState.Partial)
        {
            resolved = null;
            errorMessage = $"Model for {taskName} ('{entry.Id}') is incomplete or damaged.\nRun: stingray setup {taskName}";
            return false;
        }

        resolved = new ResolvedModelTask(entry, home.PathOf(entry.MainFile));
        errorMessage = null;
        return true;
    }

    /// <summary>
    /// Like <see cref="TryResolve"/>, but when the model is simply not installed (or an install was interrupted) and the terminal is interactive,
    /// offers to install it right here with the same summary, licence consent and confirmation as <c>stingray setup</c>, then continues.
    /// Non-interactive sessions get the one-line fix and nothing is downloaded. A declined or failed install returns false with a message.
    /// </summary>
    public static bool TryResolveOrOffer(
        string taskName, string? modelId, string? modelFile,
        out ResolvedModelTask? resolved, out string? errorMessage, CancellationToken ct,
        bool? interactive = null, ISetupPrompt? prompt = null, ModelHome? customHome = null, Func<HttpClient>? httpFactory = null,
        CatalogEntry? entryOverride = null, Func<string, string?>? env = null, Func<CatalogEntry, CancellationToken, OpenTail.Stingray.Cli.Scout.PreflightResult?>? feasibility = null)
    {
        if (TryResolve(taskName, modelId, modelFile, out resolved, out errorMessage, customHome, entryOverride))
            return true;

        // Only "not installed / damaged" is offered a fix; an unknown id or a bad file path is the user's mistake and stays an error.
        if (!string.IsNullOrWhiteSpace(modelFile)) return false;
        var entry = entryOverride ?? (!string.IsNullOrWhiteSpace(modelId) ? ModelCatalog.Find(modelId) : ModelCatalog.DefaultFor(taskName));
        var home = customHome ?? ModelHome.Default();
        if (entry is null || home.StateOf(entry) == InstallState.Installed) return false;
        if (!(interactive ?? !Console.IsInputRedirected)) return false;

        AnsiConsole.MarkupLine($"The {Markup.Escape(taskName)} model [yellow]{Markup.Escape(entry.Id)}[/] is not installed.");
        using var http = httpFactory?.Invoke() ?? ModelDownloader.CreateClient("OpenTail.Stingray/" + taskName);
        var outcome = SetupFlow.Run(entry, home, http, yes: false, acceptLicence: false, interactive: true, prompt ?? new ConsolePrompt(), ct, env, feasibility ?? (customHome is null && httpFactory is null ? SetupFlow.DefaultFeasibility() : null));
        if (!outcome.Installed)
        {
            errorMessage = $"Model for {taskName} ('{entry.Id}') was not installed ({outcome.Message}).\nRun: stingray setup {taskName}";
            resolved = null;
            return false;
        }
        return TryResolve(taskName, modelId, modelFile, out resolved, out errorMessage, customHome, entryOverride);
    }
}