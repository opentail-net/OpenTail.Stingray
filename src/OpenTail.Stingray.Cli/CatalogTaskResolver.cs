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
        ModelHome? customHome = null)
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

        // 2. Resolve catalog entry
        CatalogEntry? entry;
        if (!string.IsNullOrWhiteSpace(modelId))
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
}
