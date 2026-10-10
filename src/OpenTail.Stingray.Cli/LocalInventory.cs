using OpenTail.Stingray.Engine.Scout;
using System.Text.RegularExpressions;
using OpenTail.Stingray.Cli.Scout;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Core.Catalog;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Cli;

public enum IntegrityState
{
    NotChecked,
    Intact,
    Damaged,
}

public sealed record CatalogInventoryItem(
    CatalogEntry Entry,
    InstallState State,
    IntegrityState Integrity);

public sealed record LocalGgufItem(
    string Path,
    string FileName,
    long SizeBytes,
    string Architecture,
    string AdmissionStatus,
    string Quantization,
    string FitStatus,
    bool IsReadable,
    string? ErrorMessage);

public sealed record InventoryReport(
    ModelHome Home,
    IReadOnlyList<string> ScannedDirectories,
    IReadOnlyList<CatalogInventoryItem> CatalogItems,
    IReadOnlyList<LocalGgufItem> LocalGgufs);

/// <summary>
/// Scans the catalogue state and local directories for GGUF models on disk.
/// Implements Plan P4 (local inventory and integrity verification).
/// </summary>
public static partial class LocalInventory
{
    public const string ModelDirsEnvVar = "STINGRAY_MODEL_DIRS";

    [GeneratedRegex(@"^(.+)-(\d+)-of-(\d+)(\.gguf)$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex ShardPattern();

    /// <summary>
    /// Scans catalogue state and GGUF files found in <paramref name="home"/> and in <paramref name="extraDirs"/> (or STINGRAY_MODEL_DIRS).
    /// Reads each GGUF index without loading weights; never modifies or deletes any model file.
    /// </summary>
    public static InventoryReport Scan(
        ModelHome? home = null,
        IEnumerable<string>? extraDirs = null,
        bool verify = false,
        long? ramBytes = null,
        int contextTokens = 4096,
        CancellationToken ct = default)
    {
        home ??= ModelHome.Default();
        var catalogItems = new List<CatalogInventoryItem>();

        foreach (var entry in ModelCatalog.Entries)
        {
            ct.ThrowIfCancellationRequested();
            var state = home.StateOf(entry);
            var integrity = IntegrityState.NotChecked;

            if (verify && state == InstallState.Installed)
            {
                bool allIntact = true;
                foreach (var file in entry.Files)
                {
                    string path = home.PathOf(file);
                    if (!File.Exists(path))
                    {
                        allIntact = false;
                        break;
                    }
                    var fp = ModelFingerprinter.Compute(path);
                    if (!string.Equals(fp.Sha256, file.Sha256, StringComparison.OrdinalIgnoreCase))
                    {
                        allIntact = false;
                        break;
                    }
                }
                integrity = allIntact ? IntegrityState.Intact : IntegrityState.Damaged;
            }

            catalogItems.Add(new CatalogInventoryItem(entry, state, integrity));
        }

        var dirsToScan = new List<string>();
        if (Directory.Exists(home.Root))
            dirsToScan.Add(home.Root);

        var configuredDirs = extraDirs ?? GetEnvironmentModelDirs();
        foreach (var dir in configuredDirs)
        {
            if (Directory.Exists(dir) && !dirsToScan.Any(d => string.Equals(System.IO.Path.GetFullPath(d), System.IO.Path.GetFullPath(dir), StringComparison.OrdinalIgnoreCase)))
            {
                dirsToScan.Add(dir);
            }
        }

        var localGgufs = new List<LocalGgufItem>();
        var seenFiles = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var dir in dirsToScan)
        {
            ct.ThrowIfCancellationRequested();
            IEnumerable<string> files;
            try
            {
                files = Directory.EnumerateFiles(dir, "*.gguf", SearchOption.TopDirectoryOnly)
                    .OrderBy(f => System.IO.Path.GetFileName(f), StringComparer.OrdinalIgnoreCase);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                continue;
            }

            foreach (var file in files)
            {
                ct.ThrowIfCancellationRequested();
                string fullPath = System.IO.Path.GetFullPath(file);
                if (!seenFiles.Add(fullPath)) continue;

                string fileName = System.IO.Path.GetFileName(fullPath);
                var m = ShardPattern().Match(fileName);
                if (m.Success &&
                    int.TryParse(m.Groups[2].Value, out int shardNum) &&
                    int.TryParse(m.Groups[3].Value, out int total) &&
                    total > 1 &&
                    shardNum >= 1 &&
                    shardNum <= total)
                {
                    string prefix = m.Groups[1].Value;
                    string ext = m.Groups[4].Value;
                    string fileDir = System.IO.Path.GetDirectoryName(fullPath) ?? ".";

                    var shardPaths = new string[total];
                    for (int i = 0; i < total; i++)
                    {
                        shardPaths[i] = System.IO.Path.Combine(fileDir, $"{prefix}-{i + 1:D5}-of-{total:D5}{ext}");
                    }

                    string canonicalFirstShardPath = System.IO.Path.GetFullPath(shardPaths[0]);
                    string canonicalFirstShardName = System.IO.Path.GetFileName(canonicalFirstShardPath);

                    // Mark all required shard paths as seen so sibling files are not processed as duplicates
                    foreach (var sp in shardPaths)
                    {
                        seenFiles.Add(System.IO.Path.GetFullPath(sp));
                    }

                    long combinedSize = 0;
                    var missingShards = new List<string>();
                    foreach (var sp in shardPaths)
                    {
                        if (File.Exists(sp))
                        {
                            combinedSize += GetFileSize(sp);
                        }
                        else
                        {
                            missingShards.Add(sp);
                        }
                    }

                    if (missingShards.Count > 0)
                    {
                        string missingList = string.Join(", ", missingShards.Select(System.IO.Path.GetFileName));
                        localGgufs.Add(new LocalGgufItem(
                            canonicalFirstShardPath,
                            canonicalFirstShardName,
                            combinedSize,
                            "unreadable",
                            "not supported",
                            "-",
                            "n/a",
                            false,
                            $"Missing required shard file(s): {missingList}"));
                        continue;
                    }

                    localGgufs.Add(InspectModel(
                        canonicalFirstShardPath,
                        canonicalFirstShardPath,
                        canonicalFirstShardName,
                        combinedSize,
                        contextTokens,
                        ramBytes));
                    continue;
                }

                long singleSize = GetFileSize(fullPath);
                localGgufs.Add(InspectModel(
                    fullPath,
                    fullPath,
                    fileName,
                    singleSize,
                    contextTokens,
                    ramBytes));
            }
        }

        return new InventoryReport(home, dirsToScan, catalogItems, localGgufs);
    }

    private static LocalGgufItem InspectModel(
        string openPath,
        string itemPath,
        string itemName,
        long itemSize,
        int contextTokens,
        long? ramBytes)
    {
        try
        {
            using var model = GgufModel.Open(openPath);
            string arch = model.Metadata.TryGetValue("general.architecture", out object? a)
                ? Convert.ToString(a) ?? "unknown" : "unknown";

            var desc = ArchitectureRegistry.Find(arch);
            // CLAUDE.md rule 14: show a ported-not-verified family as plain "not supported"
            string admissionStatus = (desc is not null && desc.Status == AdmissionStatus.Admitted)
                ? "admitted"
                : "not supported";

            string quant = model.Tensors.Count > 0
                ? model.Tensors.GroupBy(t => t.DType.ToString())
                    .Select(g => (
                        DType: g.Key,
                        Bytes: g.Sum(t => { try { return t.ByteSize; } catch { return 0L; } }),
                        Count: g.Count()))
                    .OrderByDescending(x => x.Bytes)
                    .ThenByDescending(x => x.Count)
                    .FirstOrDefault().DType ?? "unknown"
                : "unknown";

            var preflight = LoadPreflight.EvaluateFile(openPath, contextTokens, ramBytes);
            string fitStatus = preflight.Verdict switch
            {
                PreflightVerdict.Allowed => "fits",
                PreflightVerdict.Blocked => "does not fit",
                PreflightVerdict.Unknown => "unknown",
                _ => "n/a",
            };

            return new LocalGgufItem(itemPath, itemName, itemSize, arch, admissionStatus, quant, fitStatus, true, null);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            return new LocalGgufItem(itemPath, itemName, itemSize, "unreadable", "not supported", "-", "n/a", false, ex.GetType().Name);
        }
    }

    private static long GetFileSize(string path)
    {
        try
        {
            using var probe = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            return RandomAccess.GetLength(probe);
        }
        catch
        {
            try { return new FileInfo(path).Length; }
            catch { return 0L; }
        }
    }

    private static IReadOnlyList<string> GetEnvironmentModelDirs()
    {
        string? env = Environment.GetEnvironmentVariable(ModelDirsEnvVar);
        if (string.IsNullOrWhiteSpace(env)) return [];
        return env.Split(System.IO.Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
    }
}
