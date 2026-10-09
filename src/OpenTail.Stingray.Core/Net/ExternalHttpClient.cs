using System.Net;
using System.Net.Http.Headers;

namespace OpenTail.Stingray.Core.Net;

/// <summary>What leaving the machine actually cost, for the report: no URLs with queries, no tokens, nothing that identifies the user.</summary>
public sealed class ExternalAccessLog
{
    private readonly object _gate = new();
    private readonly SortedSet<string> _hosts = new(StringComparer.Ordinal);
    private int _requests;
    private long _bytes;

    public int Requests { get { lock (_gate) return _requests; } }
    public long BytesReceived { get { lock (_gate) return _bytes; } }
    public IReadOnlyList<string> Hosts { get { lock (_gate) return _hosts.ToArray(); } }

    internal void Request(string host) { lock (_gate) { _requests++; _hosts.Add(host); } }
    internal void Received(long bytes) { lock (_gate) _bytes += bytes; }
}

/// <summary>
/// The only way new code reaches the network. It enforces <see cref="ExternalAccess"/>, contacts only Hugging Face hosts, follows redirects by hand so that
/// every hop is host-checked, and sends <c>HF_TOKEN</c> only to the Hub itself (never to the CDN a download redirects to). Nothing is cached or written to disk.
/// </summary>
public sealed class ExternalHttpClient : IDisposable
{
    private const int MaxRedirects = 5;
    private readonly HttpClient _http;
    private readonly Func<string, string?> _env;

    public ExternalAccessLog Log { get; } = new();

    /// <param name="inner">Test hook: replaces the real network handler. Redirect-following must be off in it.</param>
    public ExternalHttpClient(string userAgent = "OpenTail.Stingray", HttpMessageHandler? inner = null,
        Func<string, string?>? env = null, TimeSpan? timeout = null)
    {
        _env = env ?? (n => Environment.GetEnvironmentVariable(n));
        _http = new HttpClient(inner ?? new SocketsHttpHandler { AllowAutoRedirect = false, AutomaticDecompression = DecompressionMethods.None })
        {
            Timeout = timeout ?? TimeSpan.FromSeconds(60),
        };
        _http.DefaultRequestHeaders.UserAgent.ParseAdd(userAgent);
    }

    /// <summary>Throws <see cref="ExternalAccessDeniedException"/> when the policy says no. Called before every request, not once at startup.</summary>
    public void EnsureAllowed()
    {
        var d = ExternalAccess.Evaluate(_env);
        if (!d.Allowed) throw new ExternalAccessDeniedException(d.Reason, d.HowToEnable);
    }

    /// <summary>GET with optional byte range. The caller disposes the response; its stream is counted into <see cref="Log"/>.</summary>
    public async Task<HttpResponseMessage> GetAsync(Uri url, CancellationToken ct, long? rangeFrom = null, long? rangeTo = null)
    {
        EnsureAllowed();
        string? token = _env("HF_TOKEN");
        var current = url;
        for (int hop = 0; ; hop++)
        {
            if (!ExternalAccess.IsAllowedHost(current.Host) || current.Scheme != Uri.UriSchemeHttps)
                throw new ExternalAccessDeniedException($"{current.Scheme}://{current.Host} is not a Hugging Face host", null);

            using var request = new HttpRequestMessage(HttpMethod.Get, current);
            if (rangeFrom is not null) request.Headers.Range = new RangeHeaderValue(rangeFrom, rangeTo);
            if (!string.IsNullOrEmpty(token) && ExternalAccess.MayReceiveToken(current.Host))
                request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);

            Log.Request(current.Host.ToLowerInvariant());
            var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);

            if ((int)response.StatusCode is >= 300 and < 400 && response.Headers.Location is { } loc)
            {
                response.Dispose();
                if (hop >= MaxRedirects) throw new HttpRequestException($"more than {MaxRedirects} redirects from {url.Host}");
                current = loc.IsAbsoluteUri ? loc : new Uri(current, loc);
                continue;
            }
            return new CountingResponse(response, Log);
        }
    }

    /// <summary>Reads a whole (small) JSON-sized body. Use <see cref="GetAsync"/> for anything large.</summary>
    public async Task<string> GetStringAsync(Uri url, CancellationToken ct, long maxBytes = 32L << 20)
    {
        using var response = await GetAsync(url, ct).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        await using var s = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        using var ms = new MemoryStream();
        var buf = new byte[81920];
        int n;
        while ((n = await s.ReadAsync(buf, ct).ConfigureAwait(false)) > 0)
        {
            ms.Write(buf, 0, n);
            if (ms.Length > maxBytes) throw new InvalidDataException($"response from {url.Host} is larger than {maxBytes} bytes");
        }
        return System.Text.Encoding.UTF8.GetString(ms.GetBuffer(), 0, (int)ms.Length);
    }

    public void Dispose() => _http.Dispose();

    /// <summary>Wraps a response so bytes actually read are tallied into the log.</summary>
    private sealed class CountingResponse : HttpResponseMessage
    {
        public CountingResponse(HttpResponseMessage inner, ExternalAccessLog log) : base(inner.StatusCode)
        {
            ReasonPhrase = inner.ReasonPhrase;
            Version = inner.Version;
            RequestMessage = inner.RequestMessage;
            foreach (var h in inner.Headers) Headers.TryAddWithoutValidation(h.Key, h.Value);
            Content = new CountingContent(inner, log);
        }
    }

    private sealed class CountingContent : HttpContent
    {
        private readonly HttpResponseMessage _inner;
        private readonly ExternalAccessLog _log;
        public CountingContent(HttpResponseMessage inner, ExternalAccessLog log)
        {
            _inner = inner; _log = log;
            foreach (var h in inner.Content.Headers) Headers.TryAddWithoutValidation(h.Key, h.Value);
        }
        protected override async Task<Stream> CreateContentReadStreamAsync()
            => new CountingStream(await _inner.Content.ReadAsStreamAsync().ConfigureAwait(false), _log);
        protected override async Task SerializeToStreamAsync(Stream stream, TransportContext? context)
        {
            await using var s = await CreateContentReadStreamAsync().ConfigureAwait(false);
            await s.CopyToAsync(stream).ConfigureAwait(false);
        }
        protected override bool TryComputeLength(out long length)
        {
            length = _inner.Content.Headers.ContentLength ?? -1;
            return _inner.Content.Headers.ContentLength.HasValue;
        }
        protected override void Dispose(bool disposing)
        {
            if (disposing) _inner.Dispose();
            base.Dispose(disposing);
        }
    }

    private sealed class CountingStream(Stream inner, ExternalAccessLog log) : Stream
    {
        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override int Read(byte[] buffer, int offset, int count) { int n = inner.Read(buffer, offset, count); log.Received(n); return n; }
        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken ct = default) { int n = await inner.ReadAsync(buffer, ct).ConfigureAwait(false); log.Received(n); return n; }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        protected override void Dispose(bool disposing) { if (disposing) inner.Dispose(); base.Dispose(disposing); }
    }
}
