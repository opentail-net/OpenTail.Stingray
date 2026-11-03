using System.Security.Cryptography;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// SHA-256 of a checkpoint, cached per file (size, modified time, name) in the temp folder so a 20 GB file is hashed once. A parity receipt (expected tokens captured from an
/// independent reference) holds for one exact file: two same-named, same-sized copies of a checkpoint can differ, and a lookup by name alone then silently turns the receipt
/// into a failing or, worse, a meaningless test.
/// </summary>
internal static class ModelFileIdentity
{
    public static string Sha256(string path)
    {
        var info = new FileInfo(path);
        string cacheDir = Path.Combine(Path.GetTempPath(), "stingray-test-sha256");
        string cacheFile = Path.Combine(cacheDir, $"{info.Length}-{info.LastWriteTimeUtc.Ticks}-{info.Name}.sha256");
        try
        {
            if (File.Exists(cacheFile) && File.ReadAllText(cacheFile).Trim() is { Length: 64 } cached) return cached;
        }
        catch (IOException) { /* unreadable cache: hash again */ }

        using var sha = SHA256.Create();
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.SequentialScan);
        string hex = Convert.ToHexStringLower(sha.ComputeHash(stream));
        try
        {
            Directory.CreateDirectory(cacheDir);
            File.WriteAllText(cacheFile, hex);
        }
        catch (IOException) { /* cache is best effort */ }
        return hex;
    }
}
