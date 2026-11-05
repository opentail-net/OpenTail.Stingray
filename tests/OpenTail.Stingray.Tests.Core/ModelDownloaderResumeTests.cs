using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Text;
using OpenTail.Stingray.Core.Catalog;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// E3: the downloader's behaviour when a server does not behave the way a resume expects. Each case runs against a local
/// HTTP listener set to one behaviour (no network). The partial file must never be corrupted, doubled or silently dropped.
/// </summary>
public sealed class ModelDownloaderResumeTests
{
    private enum Mode { HonourRange, IgnoreRange, RangeNotSatisfiable, ServiceUnavailable }

    [Fact]
    public async Task ServerIgnoresRange_RestartsFromScratch_AndWritesExactlyTheFile()
    {
        byte[] full = Encoding.UTF8.GetBytes(string.Concat(Enumerable.Repeat("resume me ", 300)));
        using var server = new Listener(Mode.IgnoreRange, full);
        string dir = NewDir();
        try
        {
            string dest = Path.Combine(dir, "a.bin");
            await File.WriteAllBytesAsync(dest + ".part", full[..1000]);
            using var http = new HttpClient();
            var outcome = await ModelDownloader.DownloadAsync(http, server.Url + "/a.bin", dest + ".part", full.Length, null, TestContext.Current.CancellationToken);
            Assert.Equal(DownloadOutcome.Downloaded, outcome);
            // A 200 answer to a Range request must not be appended to the old bytes: that would double the first 1000.
            Assert.Equal(full, await File.ReadAllBytesAsync(dest + ".part"));
            Assert.Equal("1000-", server.RangesSeen.Single().Range);
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public async Task RangeNotSatisfiable_OnACompletePartial_IsAlreadyComplete_AndTheFileIsUntouched()
    {
        byte[] full = Encoding.UTF8.GetBytes(string.Concat(Enumerable.Repeat("complete ", 200)));
        using var server = new Listener(Mode.RangeNotSatisfiable, full);
        string dir = NewDir();
        try
        {
            string part = Path.Combine(dir, "a.bin.part");
            await File.WriteAllBytesAsync(part, full);
            using var http = new HttpClient();
            // The listing gave no size, so the local size check cannot short-circuit this; the 416 is what says "nothing left".
            var outcome = await ModelDownloader.DownloadAsync(http, server.Url + "/a.bin", part, null, null, TestContext.Current.CancellationToken);
            Assert.Equal(DownloadOutcome.AlreadyComplete, outcome);
            Assert.Equal(full, await File.ReadAllBytesAsync(part));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public async Task ServerError_KeepsThePartialForTheNextAttempt()
    {
        byte[] full = Encoding.UTF8.GetBytes(string.Concat(Enumerable.Repeat("partial kept ", 200)));
        using var server = new Listener(Mode.ServiceUnavailable, full);
        string dir = NewDir();
        try
        {
            string part = Path.Combine(dir, "a.bin.part");
            byte[] head = full[..500];
            await File.WriteAllBytesAsync(part, head);
            using var http = new HttpClient();
            await Assert.ThrowsAnyAsync<HttpRequestException>(() =>
                ModelDownloader.DownloadAsync(http, server.Url + "/a.bin", part, full.Length, null, TestContext.Current.CancellationToken));
            // A failed attempt must leave the bytes already on disk, so the next run resumes instead of starting over.
            Assert.Equal(head, await File.ReadAllBytesAsync(part));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    private static string NewDir()
    {
        string dir = Path.Combine(Path.GetTempPath(), "stingray-resume-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private sealed class Listener : IDisposable
    {
        private readonly HttpListener _listener = new();
        private readonly Mode _mode;
        private readonly byte[] _data;
        public List<(string Path, string? Range)> RangesSeen { get; } = [];
        public string Url { get; }

        public Listener(Mode mode, byte[] data)
        {
            _mode = mode;
            _data = data;
            var probe = new TcpListener(IPAddress.Loopback, 0);
            probe.Start();
            int port = ((System.Net.IPEndPoint)probe.LocalEndpoint).Port;
            probe.Stop();
            Url = $"http://127.0.0.1:{port}";
            _listener.Prefixes.Add(Url + "/");
            _listener.Start();
            _ = Task.Run(Loop);
        }

        private async Task Loop()
        {
            while (_listener.IsListening)
            {
                HttpListenerContext ctx;
                try { ctx = await _listener.GetContextAsync(); }
                catch (Exception) { return; }
                string? range = ctx.Request.Headers["Range"];
                lock (RangesSeen) RangesSeen.Add((ctx.Request.Url!.AbsolutePath, range?.Replace("bytes=", "", StringComparison.Ordinal)));
                switch (_mode)
                {
                    case Mode.RangeNotSatisfiable:
                        ctx.Response.StatusCode = 416;
                        break;
                    case Mode.ServiceUnavailable:
                        ctx.Response.StatusCode = 503;
                        break;
                    case Mode.IgnoreRange:
                        ctx.Response.StatusCode = 200;
                        ctx.Response.ContentLength64 = _data.Length;
                        await ctx.Response.OutputStream.WriteAsync(_data);
                        break;
                    default:
                        int start = range is null ? 0 : int.Parse(range["bytes=".Length..].TrimEnd('-'), System.Globalization.CultureInfo.InvariantCulture);
                        ctx.Response.StatusCode = range is null ? 200 : 206;
                        ctx.Response.ContentLength64 = _data.Length - start;
                        await ctx.Response.OutputStream.WriteAsync(_data.AsMemory(start));
                        break;
                }
                ctx.Response.Close();
            }
        }

        public void Dispose() => _listener.Close();
    }
}
