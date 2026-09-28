namespace OpenTail.Stingray.Core.Catalog;

/// <summary>Install state of a catalog entry in a model home.</summary>
public enum InstallState
{
    /// <summary>No file of the bundle is present.</summary>
    Missing,
    /// <summary>Some files are present, or a download was interrupted.</summary>
    Partial,
    /// <summary>Every file is present at its final name (downloads are only renamed there after the SHA-256 check).</summary>
    Installed,
}

/// <summary>
/// The directory catalog downloads go to, outside any repository: <c>STINGRAY_MODEL_HOME</c> if set,
/// else <c>%LOCALAPPDATA%\stingray\models</c> on Windows and <c>$XDG_CACHE_HOME/stingray/models</c>
/// (default <c>~/.cache/stingray/models</c>) elsewhere.
///
/// <para>Files are stored flat under their Hugging Face file names, so a Piper voice's
/// <c>.onnx.json</c> sits beside its <c>.onnx</c> as the pipeline expects, and pointing
/// <c>STINGRAY_MODEL_HOME</c> at an existing <c>models/</c> folder reuses files already there.</para>
/// </summary>
public sealed class ModelHome
{
    /// <summary>Environment variable that overrides the default location.</summary>
    public const string EnvironmentVariable = "STINGRAY_MODEL_HOME";

    /// <summary>Suffix of an in-progress (unverified) download.</summary>
    public const string PartialSuffix = ".part";

    /// <summary>Absolute root directory.</summary>
    public string Root { get; }

    /// <summary>Uses <paramref name="root"/> as the model home.</summary>
    public ModelHome(string root) => Root = Path.GetFullPath(root);

    /// <summary>The model home for this user and process.</summary>
    public static ModelHome Default() => new(DefaultRoot());

    /// <summary>Resolves the default root without creating it.</summary>
    public static string DefaultRoot()
    {
        string? overrideRoot = Environment.GetEnvironmentVariable(EnvironmentVariable);
        if (!string.IsNullOrWhiteSpace(overrideRoot))
            return overrideRoot;

        if (OperatingSystem.IsWindows())
            return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "stingray", "models");

        string? xdg = Environment.GetEnvironmentVariable("XDG_CACHE_HOME");
        string cache = !string.IsNullOrWhiteSpace(xdg)
            ? xdg
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache");
        return Path.Combine(cache, "stingray", "models");
    }

    /// <summary>Local path of a catalog file (whether or not it exists).</summary>
    public string PathOf(CatalogFile file) => Path.Combine(Root, file.FileName);

    /// <summary>Install state of every file of <paramref name="entry"/>.</summary>
    public InstallState StateOf(CatalogEntry entry)
    {
        int present = 0;
        bool anyPartial = false;
        foreach (var f in entry.Files)
        {
            string path = PathOf(f);
            if (File.Exists(path) && new FileInfo(path).Length == f.Size) present++;
            else if (File.Exists(path) || File.Exists(path + PartialSuffix)) anyPartial = true;
        }
        if (present == entry.Files.Count) return InstallState.Installed;
        return present > 0 || anyPartial ? InstallState.Partial : InstallState.Missing;
    }

    /// <summary>Bytes still to download for <paramref name="entry"/> (counts resumable partial files).</summary>
    public long RemainingBytes(CatalogEntry entry)
    {
        long remaining = 0;
        foreach (var f in entry.Files)
        {
            string path = PathOf(f);
            if (File.Exists(path) && new FileInfo(path).Length == f.Size) continue;
            string part = path + PartialSuffix;
            long have = File.Exists(part) ? Math.Min(new FileInfo(part).Length, f.Size) : 0;
            remaining += f.Size - have;
        }
        return remaining;
    }
}
