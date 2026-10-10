using System.Globalization;

namespace OpenTail.Stingray.Engine.Verification;

/// <summary>SHA-256 and size of a model file; <see cref="FromCache"/> says whether the hash came from the sidecar instead of reading the file.</summary>
public sealed record ModelFingerprint(string Sha256, long SizeBytes, bool FromCache);

public enum PinStatus
{
    /// <summary>The file's SHA-256 equals the one the golden was recorded on.</summary>
    Verified,
    /// <summary>The file differs from the one the golden was recorded on (a different conversion or quantization): the result is valid for the file that was run, not for the recorded one.</summary>
    Mismatch,
    /// <summary>The golden recorded no hash (migrated receipt, or captured with --no-hash).</summary>
    NotRecorded,
}

public sealed record PinResult(PinStatus Status, string? Expected, string? Actual)
{
    /// <summary>One line for reports; empty when there is nothing to say.</summary>
    public string Describe(string fileName) => Status switch
    {
        PinStatus.Verified => $"pinned: {fileName} matches the verified file (sha256 {Expected![..12]}...)",
        PinStatus.Mismatch => $"UNPINNED: {fileName} is NOT the file this golden was recorded on (expected sha256 {Expected![..12]}..., got {Actual![..12]}...); a different conversion can legitimately differ",
        _ => "pin: the golden recorded no hash, so the file cannot be confirmed",
    };
}

/// <summary>
/// Hashes model files once. A 12 GB checkpoint takes tens of seconds to hash, so the result is cached in a small sidecar next to the model
/// (<c>&lt;file&gt;.sha256</c>) keyed by size and last-write time: any change to either invalidates it. An unwritable folder just means no cache.
/// </summary>
public static class ModelFingerprinter
{
    public static string SidecarPath(string modelPath) => modelPath + ".sha256";

    public static ModelFingerprint Compute(string path, Action<double>? progress = null)
    {
        var info = new FileInfo(path);
        long size = info.Length;
        long mtime = info.LastWriteTimeUtc.Ticks;
        string sidecar = SidecarPath(path);

        if (TryReadSidecar(sidecar, size, mtime, out string? cached))
            return new ModelFingerprint(cached!, size, FromCache: true);

        string sha = GoldenCaptureParsing.Sha256Hex(path, progress);
        TryWriteSidecar(sidecar, sha, size, mtime);
        return new ModelFingerprint(sha, size, FromCache: false);
    }

    /// <summary>Compares a golden's recorded hash with the file's.</summary>
    public static PinResult CheckPin(GoldenFile golden, string modelPath, Action<double>? progress = null)
    {
        if (golden.Model.Files is { Count: > 1 } files)
        {
            // A split model: every shard must be present next to the first and match its recorded hash.
            string dir = Path.GetDirectoryName(Path.GetFullPath(modelPath)) ?? ".";
            foreach (var f in files)
            {
                string p = Path.Combine(dir, f.FileName);
                if (!File.Exists(p)) return new PinResult(PinStatus.Mismatch, f.Sha256, new string('0', 64));
                string got = Compute(p, progress).Sha256;
                if (!string.Equals(f.Sha256, got, StringComparison.OrdinalIgnoreCase)) return new PinResult(PinStatus.Mismatch, f.Sha256, got);
            }
            return new PinResult(PinStatus.Verified, files[0].Sha256, files[0].Sha256);
        }
        string? expected = golden.Model.Sha256;
        if (string.IsNullOrWhiteSpace(expected)) return new PinResult(PinStatus.NotRecorded, null, null);
        string actual = Compute(modelPath, progress).Sha256;
        return string.Equals(expected, actual, StringComparison.OrdinalIgnoreCase)
            ? new PinResult(PinStatus.Verified, expected, actual)
            : new PinResult(PinStatus.Mismatch, expected, actual);
    }

    private static bool TryReadSidecar(string sidecar, long size, long mtime, out string? sha)
    {
        sha = null;
        try
        {
            if (!File.Exists(sidecar)) return false;
            var parts = File.ReadAllText(sidecar).Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 3 || parts[0].Length != 64) return false;
            if (!long.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out long s) || s != size) return false;
            if (!long.TryParse(parts[2], NumberStyles.Integer, CultureInfo.InvariantCulture, out long m) || m != mtime) return false;
            sha = parts[0].ToLowerInvariant();
            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { return false; }
    }

    private static void TryWriteSidecar(string sidecar, string sha, long size, long mtime)
    {
        try { File.WriteAllText(sidecar, string.Create(CultureInfo.InvariantCulture, $"{sha} {size} {mtime}{Environment.NewLine}")); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* read-only folder: no cache, still correct */ }
    }
}
