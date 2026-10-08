namespace OpenTail.Stingray.Engine.Verification;

/// <summary>
/// Finds a model checkpoint by file name. One implementation for every test and tool that used to carry its own hard-coded folder list
/// (docs/2-coverage/2026-10-08-golden-parity-and-admission-tooling-plan.md, Phase 1). No drive letter or user path lives in code:
/// per-machine locations come from the <c>STINGRAY_MODEL_DIRS</c> environment variable, and fixed/removable drives are enumerated at
/// run time looking only for the conventional folder names below.
/// </summary>
public static class ModelLocator
{
    /// <summary>Folder names looked for at the root of every drive, and under every ancestor of the working directory.</summary>
    private static readonly string[] s_driveRootFolders = ["_models", "_other_models", "models"];

    /// <summary>
    /// Existing directories to search, in priority order: <c>STINGRAY_MODEL_DIRS</c> entries, then <c>models</c> and <c>models/_models</c>
    /// under the working directory and the test binary directory and their ancestors (up to 8 levels), then the conventional folders at
    /// the root of each fixed or removable drive. Only directories that exist are returned.
    /// </summary>
    public static IReadOnlyList<string> Roots(string? startDirectory = null)
    {
        var roots = new List<string>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        void Add(string path)
        {
            try
            {
                string full = Path.GetFullPath(path);
                if (Directory.Exists(full) && seen.Add(full)) roots.Add(full);
            }
            catch { /* unreadable or malformed entry: skip it */ }
        }

        string? env = Environment.GetEnvironmentVariable("STINGRAY_MODEL_DIRS");
        if (!string.IsNullOrWhiteSpace(env))
            foreach (var entry in env.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                Add(entry);

        foreach (var start in new[] { startDirectory ?? Directory.GetCurrentDirectory(), AppContext.BaseDirectory })
        {
            string? dir = start;
            for (int i = 0; i < 8 && dir is not null; i++)
            {
                Add(Path.Combine(dir, "models"));
                Add(Path.Combine(dir, "models", "_models"));
                dir = Path.GetDirectoryName(dir.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
            }
        }

        try
        {
            foreach (var drive in DriveInfo.GetDrives())
            {
                if (drive.DriveType is not (DriveType.Fixed or DriveType.Removable)) continue;
                bool ready;
                try { ready = drive.IsReady; } catch { ready = false; }
                if (!ready) continue;
                foreach (var folder in s_driveRootFolders)
                    Add(Path.Combine(drive.RootDirectory.FullName, folder));
            }
        }
        catch { /* drive enumeration unavailable on this platform: the other roots still apply */ }

        return roots;
    }

    /// <summary>Full path of the first file found, trying the names in order across all roots; null when none exists.</summary>
    public static string? Find(params string[] fileNames) => Find(Roots(), fileNames);

    /// <summary>As <see cref="Find(string[])"/> over an explicit root list (used by tests and by callers with their own roots).</summary>
    public static string? Find(IEnumerable<string> roots, IEnumerable<string> fileNames)
    {
        var rootList = roots as IReadOnlyList<string> ?? roots.ToList();
        foreach (var name in fileNames)
            foreach (var root in rootList)
            {
                string candidate = Path.Combine(root, name);
                if (File.Exists(candidate)) return candidate;
            }
        return null;
    }

    /// <summary>
    /// <see cref="Find(string[])"/>, but a miss is announced on stderr with the names wanted and the folders searched, so a skipped test
    /// says why instead of silently reporting a pass (CLAUDE.md rule 12).
    /// </summary>
    public static string? FindOrReport(params string[] fileNames)
    {
        var roots = Roots();
        string? path = Find(roots, fileNames);
        if (path is null) Console.Error.WriteLine(DescribeMiss(fileNames, roots));
        return path;
    }

    /// <summary>Human-readable explanation of a miss.</summary>
    public static string DescribeMiss(IEnumerable<string> fileNames, IEnumerable<string> roots) =>
        $"[ModelLocator] NOT FOUND: {string.Join(" | ", fileNames)}. Searched {roots.Count()} folder(s): " +
        $"{string.Join("; ", roots)}. Set STINGRAY_MODEL_DIRS (a path list) to add folders, or put the file in a folder named " +
        $"{string.Join("/", s_driveRootFolders)} at the root of a drive.";
}
