using System.Text.Json;
using System.Text.Json.Serialization;

namespace OpenTail.Stingray.Cli;

[JsonSourceGenerationOptions(WriteIndented = true)]
[JsonSerializable(typeof(Dictionary<string, string>))]
internal partial class FavouritesJsonContext : JsonSerializerContext;

/// <summary>
/// Manages per-user default model preferences (favourites.json) for tasks (Plan P3).
/// </summary>
public static class Favourites
{
    public const string EnvironmentVariable = "STINGRAY_CONFIG_DIR";
    public const string FileName = "favourites.json";

    /// <summary>
    /// Resolves the directory where user preferences live.
    /// Priority: STINGRAY_CONFIG_DIR > %APPDATA%\stingray (Windows) or $XDG_CONFIG_HOME/stingray (~/.config/stingray).
    /// </summary>
    public static string ConfigDirectory(Func<string, string?>? env = null)
    {
        string? overrideDir = env?.Invoke(EnvironmentVariable) ?? Environment.GetEnvironmentVariable(EnvironmentVariable);
        if (!string.IsNullOrWhiteSpace(overrideDir))
            return Path.GetFullPath(overrideDir);

        if (OperatingSystem.IsWindows())
        {
            string appData = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            return Path.Combine(string.IsNullOrWhiteSpace(appData) ? Path.GetTempPath() : appData, "stingray");
        }

        string? xdg = env?.Invoke("XDG_CONFIG_HOME") ?? Environment.GetEnvironmentVariable("XDG_CONFIG_HOME");
        string configRoot = !string.IsNullOrWhiteSpace(xdg)
            ? xdg
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".config");
        return Path.Combine(configRoot, "stingray");
    }

    /// <summary>
    /// Absolute path to the favourites.json file.
    /// </summary>
    public static string FilePath(string? configDir = null, Func<string, string?>? env = null)
        => Path.Combine(configDir ?? ConfigDirectory(env), FileName);

    /// <summary>
    /// Loads all configured task favourites. Returns false with an explicit error when the file exists but is corrupt.
    /// </summary>
    public static bool TryGetAll(
        out Dictionary<string, string> favourites,
        out string? error,
        string? configDir = null,
        Func<string, string?>? env = null)
    {
        string path = FilePath(configDir, env);
        favourites = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        error = null;

        if (!File.Exists(path))
            return true;

        try
        {
            string json = File.ReadAllText(path);
            if (string.IsNullOrWhiteSpace(json))
            {
                error = $"Error reading favourites file '{path}': file is empty.";
                return false;
            }

            var parsed = JsonSerializer.Deserialize(json, FavouritesJsonContext.Default.DictionaryStringString);
            if (parsed is null)
            {
                error = $"Error reading favourites file '{path}': invalid JSON content.";
                return false;
            }

            foreach (var (k, v) in parsed)
            {
                if (!string.IsNullOrWhiteSpace(k) && !string.IsNullOrWhiteSpace(v))
                    favourites[k] = v;
            }

            return true;
        }
        catch (Exception ex) when (ex is JsonException or IOException or UnauthorizedAccessException)
        {
            error = $"Error reading favourites file '{path}': {ex.Message}";
            return false;
        }
    }

    /// <summary>
    /// Attempts to read the favourite catalogue id for <paramref name="task"/>.
    /// </summary>
    public static bool TryGet(
        string task,
        out string? modelId,
        out string? error,
        string? configDir = null,
        Func<string, string?>? env = null)
    {
        modelId = null;
        if (!TryGetAll(out var all, out error, configDir, env))
            return false;

        all.TryGetValue(task, out modelId);
        return true;
    }

    /// <summary>
    /// Sets the favourite model id for a task. Writes atomically and refuses to overwrite a corrupt file.
    /// </summary>
    public static void Set(
        string task,
        string modelId,
        string? configDir = null,
        Func<string, string?>? env = null)
    {
        string path = FilePath(configDir, env);
        if (File.Exists(path) && !TryGetAll(out _, out string? readError, configDir, env))
            throw new InvalidOperationException($"Cannot update favourites: {readError}");

        TryGetAll(out var current, out _, configDir, env);
        current[task] = modelId;

        WriteAtomic(path, current);
    }

    /// <summary>
    /// Removes the favourite model for a task. Writes atomically and refuses to overwrite a corrupt file.
    /// </summary>
    public static void Clear(
        string task,
        string? configDir = null,
        Func<string, string?>? env = null)
    {
        string path = FilePath(configDir, env);
        if (File.Exists(path) && !TryGetAll(out _, out string? readError, configDir, env))
            throw new InvalidOperationException($"Cannot update favourites: {readError}");

        TryGetAll(out var current, out _, configDir, env);
        if (current.Remove(task))
        {
            WriteAtomic(path, current);
        }
    }

    private static void WriteAtomic(string path, Dictionary<string, string> favourites)
    {
        string dir = Path.GetDirectoryName(path)!;
        Directory.CreateDirectory(dir);

        string json = JsonSerializer.Serialize(favourites, FavouritesJsonContext.Default.DictionaryStringString);
        string tmpPath = Path.Combine(dir, $"{Path.GetFileName(path)}.{Guid.NewGuid():N}.tmp");

        File.WriteAllText(tmpPath, json);
        File.Move(tmpPath, path, overwrite: true);
    }
}
