using System.Net;
using System.Net.Http.Headers;

namespace OpenTail.Stingray.Core.Net;

public enum RemoteIndexOutcome
{
    /// <summary>The whole index of every shard was read and parsed.</summary>
    Complete,
    /// <summary>The index is larger than the byte cap. Nothing more was read; the model was NOT downloaded to finish the job.</summary>
    IncompleteOverCap,
    /// <summary>The Hub refused (private or gated repo, or a missing/invalid token).</summary>
    AccessRestricted,
    NotFound,
    /// <summary>Network error, a server that ignored the Range request, an inconsistent response, or a malformed GGUF. <see cref="RemoteIndexResult.Detail"/> says which.</summary>
    Failed,
}

public sealed record RemoteIndexResult(RemoteIndexOutcome Outcome, GgufIndex? Index, IReadOnlyList<long> ShardSizes, long BytesRead, string Detail);

/// <summary>
/// Reads the index of a hosted GGUF with HTTP Range requests: a first chunk, then more only while the parser says the index is not finished.
/// The guardrails are the point of this class:
///   * every request is pinned to a commit SHA, so the pieces of one index cannot come from different revisions;
///   * a ranged answer must be <c>206</c> with a <c>Content-Range</c> that starts where asked and agrees on the file size every time;
///   * a server that ignores Range and sends <c>200</c> with a whole large file is refused without reading the body;
///   * never more bytes than were asked for are read, and never more than <c>maxIndexBytes</c> per shard in total;
///   * past the cap the answer is <see cref="RemoteIndexOutcome.IncompleteOverCap"/>, never a fallback download.
/// </summary>
public static class RemoteGgufReader
{
    public const long DefaultMaxIndexBytes = 128L << 20;
    public const int InitialChunkBytes = 2 << 20;

    public static async Task<RemoteIndexResult> ReadAsync(
        ExternalHttpClient http, string repo, string revision, IReadOnlyList<string> shardPaths,
        long maxIndexBytes, CancellationToken ct, int initialChunkBytes = InitialChunkBytes)
    {
        ArgumentNullException.ThrowIfNull(http);
        if (shardPaths.Count == 0) throw new ArgumentException("At least one shard path is required.", nameof(shardPaths));
        if (maxIndexBytes < 1024) throw new ArgumentOutOfRangeException(nameof(maxIndexBytes));

        var urls = shardPaths.Select(p => HubClient.ResolveUrl(repo, revision, p)).ToArray();
        var data = new byte[shardPaths.Count][];
        var sizes = new long[shardPaths.Count];
        long read = 0;

        try
        {
            for (int s = 0; s < urls.Length; s++)
            {
                var first = await FetchAsync(http, urls[s], 0, Math.Min(initialChunkBytes, maxIndexBytes) - 1, expectedTotal: null, ct).ConfigureAwait(false);
                if (first.Failure is { } f) return Fail(f, sizes, read);
                data[s] = first.Bytes!; sizes[s] = first.Total; read += first.Bytes!.Length;
            }

            while (true)
            {
                try
                {
                    var index = GgufModel.ParseIndex(data.Select((d, i) => ((ReadOnlyMemory<byte>)d, sizes[i])).ToArray());
                    return new RemoteIndexResult(RemoteIndexOutcome.Complete, index, sizes, read, $"{read} bytes read for the index of {shardPaths.Count} file(s).");
                }
                catch (InvalidDataException ex) when (GgufTruncation.IsTruncation(ex, out int shard))
                {
                    long have = data[shard].Length;
                    if (have >= sizes[shard])
                        return new(RemoteIndexOutcome.Failed, null, sizes, read, $"{shardPaths[shard]} ended inside its own index (malformed file): {ex.Message}");
                    if (have >= maxIndexBytes)
                        return new(RemoteIndexOutcome.IncompleteOverCap, null, sizes, read,
                            $"The index of {shardPaths[shard]} is larger than the {maxIndexBytes} byte limit. Raise the limit to inspect it; the model was not downloaded.");

                    long next = Math.Min(Math.Min(have * 2, maxIndexBytes), sizes[shard]);
                    var more = await FetchAsync(http, urls[shard], have, next - 1, expectedTotal: sizes[shard], ct).ConfigureAwait(false);
                    if (more.Failure is { } f) return Fail(f, sizes, read);
                    var joined = new byte[have + more.Bytes!.Length];
                    data[shard].CopyTo(joined, 0); more.Bytes.CopyTo(joined, have);
                    data[shard] = joined; read += more.Bytes.Length;
                }
                catch (InvalidDataException ex)
                {
                    return new(RemoteIndexOutcome.Failed, null, sizes, read, $"Not a readable GGUF: {ex.Message}");
                }
            }
        }
        catch (HttpRequestException ex)
        {
            return new(RemoteIndexOutcome.Failed, null, sizes, read, $"Network error: {ex.Message}");
        }
        catch (TaskCanceledException) when (!ct.IsCancellationRequested)
        {
            return new(RemoteIndexOutcome.Failed, null, sizes, read, "The request timed out.");
        }
    }

    private static RemoteIndexResult Fail((RemoteIndexOutcome Outcome, string Detail) f, long[] sizes, long read) => new(f.Outcome, null, sizes, read, f.Detail);

    private readonly record struct Chunk(byte[]? Bytes, long Total, (RemoteIndexOutcome Outcome, string Detail)? Failure);

    /// <summary>One ranged GET with every check described on the class. Returns exactly the bytes asked for, or a failure; never more.</summary>
    private static async Task<Chunk> FetchAsync(ExternalHttpClient http, Uri url, long from, long toInclusive, long? expectedTotal, CancellationToken ct)
    {
        using var resp = await http.GetAsync(url, ct, from, toInclusive).ConfigureAwait(false);
        long want = toInclusive - from + 1;

        switch (resp.StatusCode)
        {
            case HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden:
                return Bad(RemoteIndexOutcome.AccessRestricted, $"The Hub refused access ({(int)resp.StatusCode}). The repo may be private or gated; accept its terms on huggingface.co and set HF_TOKEN.");
            case HttpStatusCode.NotFound:
                return Bad(RemoteIndexOutcome.NotFound, "The file or revision was not found.");
            case HttpStatusCode.PartialContent:
                break;
            case HttpStatusCode.OK:
                // The server ignored Range. Acceptable only when the whole file is no bigger than what was asked for anyway.
                if (from == 0 && resp.Content.Headers.ContentLength is { } whole && whole <= want)
                    return await ReadExactAsync(resp, (int)whole, whole, expectedTotal, ct).ConfigureAwait(false);
                return Bad(RemoteIndexOutcome.Failed, "The server ignored the Range request and offered the whole file; refusing to read it.");
            default:
                return Bad(RemoteIndexOutcome.Failed, $"Unexpected HTTP status {(int)resp.StatusCode}.");
        }

        ContentRangeHeaderValue? cr = resp.Content.Headers.ContentRange;
        if (cr is null || cr.From is null || cr.To is null || cr.Length is null)
            return Bad(RemoteIndexOutcome.Failed, "A ranged response arrived without a usable Content-Range.");
        if (cr.From != from || cr.To < cr.From || cr.To > toInclusive)
            return Bad(RemoteIndexOutcome.Failed, $"Content-Range {cr.From}-{cr.To} does not match the requested {from}-{toInclusive}.");
        long len = cr.To.Value - cr.From.Value + 1;
        if (resp.Content.Headers.ContentLength is { } cl && cl != len)
            return Bad(RemoteIndexOutcome.Failed, $"Content-Length {cl} disagrees with Content-Range ({len} bytes).");
        return await ReadExactAsync(resp, (int)len, cr.Length.Value, expectedTotal, ct).ConfigureAwait(false);

        static Chunk Bad(RemoteIndexOutcome o, string d) => new(null, 0, (o, d));
    }

    private static async Task<Chunk> ReadExactAsync(HttpResponseMessage resp, int count, long total, long? expectedTotal, CancellationToken ct)
    {
        if (expectedTotal is { } e && e != total)
            return new(null, 0, (RemoteIndexOutcome.Failed, $"The file size changed between requests ({e} then {total})."));
        var buf = new byte[count];
        await using var s = await resp.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        int got = 0;
        while (got < count)
        {
            int n = await s.ReadAsync(buf.AsMemory(got, count - got), ct).ConfigureAwait(false);
            if (n == 0) return new(null, 0, (RemoteIndexOutcome.Failed, $"The response ended after {got} of {count} bytes."));
            got += n;
        }
        // Whatever the server may still send is deliberately not read: the response is disposed by the caller.
        return new(buf, total, null);
    }
}
