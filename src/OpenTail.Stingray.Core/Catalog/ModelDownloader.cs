using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;

namespace OpenTail.Stingray.Core.Catalog;

/// <summary>What <see cref="ModelDownloader.DownloadAsync"/> did.</summary>
public enum DownloadOutcome
{
    /// <summary>Bytes were fetched (from scratch or resumed).</summary>
    Downloaded,
    /// <summary>The file was already complete; nothing was fetched.</summary>
    AlreadyComplete,
}

/// <summary>
/// Resumable Hugging Face downloads, shared by <c>stingray pull</c> and catalog installs.
/// Sends <c>HF_TOKEN</c> as a bearer token when set (gated repos).
/// </summary>
public static class ModelDownloader
{
    /// <summary>Creates an HttpClient suitable for large model downloads.</summary>
    public static HttpClient CreateClient(string userAgent)
    {
        var http = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        http.DefaultRequestHeaders.UserAgent.ParseAdd(userAgent);
        return http;
    }

    /// <summary>
    /// Streams <paramref name="url"/> to <paramref name="destPath"/>. An existing file at least
    /// <paramref name="expectedSize"/> long counts as complete (a size check, not a hash check);
    /// a shorter one is resumed with a Range request, or restarted if the server ignores it.
    /// </summary>
    /// <param name="progress">Called with (bytes on disk, total bytes if known) as data arrives.</param>
    public static async Task<DownloadOutcome> DownloadAsync(
        HttpClient http, string url, string destPath, long? expectedSize,
        Action<long, long?>? progress, CancellationToken ct)
    {
        long existing = File.Exists(destPath) ? new FileInfo(destPath).Length : 0;
        if (existing > 0 && expectedSize is { } exp && existing >= exp)
            return DownloadOutcome.AlreadyComplete;

        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        string? token = Environment.GetEnvironmentVariable("HF_TOKEN");
        if (!string.IsNullOrEmpty(token))
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
        if (existing > 0)
            request.Headers.Range = new RangeHeaderValue(existing, null);

        using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);
        // 416 on a Range request from the end of an existing file: nothing left to fetch. The listing gives no sizes
        // for some repos, so the size check above cannot catch a complete file.
        if (existing > 0 && response.StatusCode == HttpStatusCode.RequestedRangeNotSatisfiable)
            return DownloadOutcome.AlreadyComplete;
        bool resumed = existing > 0 && response.StatusCode == HttpStatusCode.PartialContent;
        if (existing > 0 && !resumed)
            existing = 0; // Server ignored the Range request; restart from scratch.
        response.EnsureSuccessStatusCode();

        long? total = response.Content.Headers.ContentLength is { } cl ? cl + existing : expectedSize;
        await using var contentStream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        await using var fileStream = new FileStream(destPath, resumed ? FileMode.Append : FileMode.Create, FileAccess.Write);

        byte[] buffer = new byte[1024 * 1024];
        long downloaded = existing;
        int read;
        while ((read = await contentStream.ReadAsync(buffer, ct).ConfigureAwait(false)) > 0)
        {
            await fileStream.WriteAsync(buffer.AsMemory(0, read), ct).ConfigureAwait(false);
            downloaded += read;
            progress?.Invoke(downloaded, total);
        }
        return DownloadOutcome.Downloaded;
    }

    /// <summary>Lower-case hex SHA-256 of a file.</summary>
    public static async Task<string> Sha256Async(string path, CancellationToken ct)
    {
        await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true);
        byte[] hash = await SHA256.HashDataAsync(stream, ct).ConfigureAwait(false);
        return Convert.ToHexStringLower(hash);
    }
}

/// <summary>Thrown when a downloaded or existing catalog file does not have the catalog's SHA-256.</summary>
public sealed class ModelHashMismatchException(string path, string expected, string actual)
    : IOException($"{path}: SHA-256 {actual} does not match the catalog's {expected}. Delete it and run setup again.")
{
    /// <summary>File that failed the check.</summary>
    public string FilePath { get; } = path;
}

/// <summary>Thrown when a catalog file has to be downloaded but offline mode is on.</summary>
public sealed class ModelOfflineException(string fileName)
    : IOException($"{fileName} is not in the model home and external access is off (STINGRAY_OFFLINE / HF_HUB_OFFLINE / STINGRAY_ALLOW_EXTERNAL=0). Run setup without it, or put the file there yourself.")
{
}

/// <summary>Installs catalog entries into a <see cref="ModelHome"/>.</summary>
public static class ModelInstaller
{
    /// <summary>True when the shared external-access policy denies the network (<c>STINGRAY_OFFLINE</c>, <c>HF_HUB_OFFLINE</c>, or <c>STINGRAY_ALLOW_EXTERNAL</c> switched off): never touch the network.</summary>
    public static bool OfflineFromEnvironment() => !OpenTail.Stingray.Core.Net.ExternalAccess.Evaluate().Allowed;

    /// <summary>Progress of one file: which file, bytes on disk, total size.</summary>
    public delegate void FileProgress(CatalogFile file, long bytes, long total);

    /// <summary>
    /// Makes every file of <paramref name="entry"/> present and verified under <paramref name="home"/>,
    /// and returns the main file's path. Each file downloads to <c>name.part</c> (resumable across
    /// runs), is checked against the catalog SHA-256, and only then renamed to its final name, so a
    /// file at its final name is always a verified one. A file already at its final name is re-hashed
    /// once here (cheap next to a download) in case the home points at a hand-filled folder.
    /// </summary>
    /// <param name="onFileStart">Called before each file with whether it is already present.</param>
    /// <param name="offline">Null reads <see cref="OfflineFromEnvironment"/>. When on, files already present are still verified, and a missing one throws <see cref="ModelOfflineException"/> instead of downloading.</param>
    public static async Task<string> EnsureAsync(
        CatalogEntry entry, ModelHome home, HttpClient http,
        Action<CatalogFile, bool>? onFileStart, FileProgress? progress, CancellationToken ct,
        bool? offline = null)
    {
        bool isOffline = offline ?? OfflineFromEnvironment();
        Directory.CreateDirectory(home.Root);
        foreach (var file in entry.Files)
        {
            string dest = home.PathOf(file);
            if (File.Exists(dest) && new FileInfo(dest).Length == file.Size)
            {
                onFileStart?.Invoke(file, true);
                await VerifyAsync(dest, file, ct).ConfigureAwait(false);
                continue;
            }

            if (isOffline) throw new ModelOfflineException(file.FileName);
            onFileStart?.Invoke(file, false);
            string part = dest + ModelHome.PartialSuffix;
            await ModelDownloader.DownloadAsync(http, file.Url, part, file.Size,
                progress is null ? null : (b, _) => progress(file, b, file.Size), ct).ConfigureAwait(false);

            try
            {
                await VerifyAsync(part, file, ct).ConfigureAwait(false);
            }
            catch (ModelHashMismatchException)
            {
                File.Delete(part); // A corrupt partial file would otherwise be "resumed" forever.
                throw;
            }
            File.Move(part, dest, overwrite: true);
        }
        return home.PathOf(entry.MainFile);
    }

    private static async Task VerifyAsync(string path, CatalogFile file, CancellationToken ct)
    {
        string actual = await ModelDownloader.Sha256Async(path, ct).ConfigureAwait(false);
        if (!actual.Equals(file.Sha256, StringComparison.OrdinalIgnoreCase))
            throw new ModelHashMismatchException(path, file.Sha256, actual);
    }
}
