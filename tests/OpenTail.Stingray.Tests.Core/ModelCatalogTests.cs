using System.Net;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;
using OpenTail.Stingray.Core.Catalog;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// The model catalog's own rules (docs/3-product-and-runtime/103-front-door-design.md) and the
/// installer's download / resume / SHA-256 behaviour, against a local HTTP server (no network).
/// </summary>
public sealed class ModelCatalogTests
{
    [Fact]
    public void CatalogFollowsItsRules()
    {
        Assert.Equal(ModelCatalog.Entries.Count, ModelCatalog.Entries.Select(e => e.Id).Distinct(StringComparer.OrdinalIgnoreCase).Count());
        foreach (string task in ModelCatalog.Tasks)
        {
            int n = ModelCatalog.ForTask(task).Count();
            Assert.InRange(n, 1, 4); // One default plus at most three alternatives.
        }

        // Files are stored flat in the model home, so names must not collide across entries.
        var names = ModelCatalog.Entries.SelectMany(e => e.Files).Select(f => f.FileName).ToList();
        Assert.Equal(names.Count, names.Distinct(StringComparer.OrdinalIgnoreCase).Count());

        foreach (var e in ModelCatalog.Entries)
        {
            Assert.Contains(e.Task, ModelCatalog.Tasks);
            Assert.StartsWith("stingray ", e.RunTemplate, StringComparison.Ordinal);
            Assert.False(string.IsNullOrWhiteSpace(e.Evidence));
            Assert.DoesNotContain("MEASURE", e.Speed, StringComparison.Ordinal);
            foreach (var f in e.Files)
            {
                Assert.Matches("^[0-9a-f]{64}$", f.Sha256);
                Assert.Matches("^[0-9a-f]{40}$", f.Revision); // Pinned to a commit, never "main".
                Assert.Contains("/resolve/" + f.Revision + "/", f.Url, StringComparison.Ordinal);
                Assert.True(f.Size > 0);
                Assert.DoesNotContain("..", f.RepoPath, StringComparison.Ordinal);
            }
        }
    }

    [Fact]
    public void FindResolvesIdsAndTasks()
    {
        Assert.Equal("qwen2.5-0.5b", ModelCatalog.Find("chat")!.Id);
        Assert.Equal("whisper-base", ModelCatalog.Find("WHISPER-BASE")!.Id);
        Assert.Null(ModelCatalog.Find("no-such-thing"));
    }

    [Fact]
    public async Task InstallResumesVerifiesAndReportsState()
    {
        byte[] a = Encoding.UTF8.GetBytes(string.Concat(Enumerable.Repeat("model weights ", 5000)));
        byte[] b = Encoding.UTF8.GetBytes("{\"config\":true}");
        using var server = new StubServer(new() { ["/o/r/resolve/" + Rev + "/dir/a.bin"] = a, ["/o/r/resolve/" + Rev + "/b.json"] = b });
        var entry = MakeEntry(server, a, b);

        string root = Path.Combine(Path.GetTempPath(), "stingray-catalog-test-" + Guid.NewGuid().ToString("N"));
        try
        {
            var home = new ModelHome(root);
            Assert.Equal(InstallState.Missing, home.StateOf(entry));

            // An interrupted download: the first 1000 bytes are already in a .part file.
            Directory.CreateDirectory(root);
            File.WriteAllBytes(Path.Combine(root, "a.bin" + ModelHome.PartialSuffix), a[..1000]);
            Assert.Equal(InstallState.Partial, home.StateOf(entry));
            Assert.Equal(a.Length - 1000 + b.Length, home.RemainingBytes(entry));

            using var http = new HttpClient();
            string main = await ModelInstaller.EnsureAsync(entry, home, http, null, null, TestContext.Current.CancellationToken);

            Assert.Equal(Path.Combine(root, "a.bin"), main);
            Assert.Equal(a, File.ReadAllBytes(main));
            Assert.Equal(b, File.ReadAllBytes(Path.Combine(root, "b.json")));
            Assert.False(File.Exists(main + ModelHome.PartialSuffix));
            Assert.Equal(InstallState.Installed, home.StateOf(entry));
            Assert.Equal(0, home.RemainingBytes(entry));
            Assert.Equal("1000-", server.RangesSeen.Single(r => r.Path.EndsWith("a.bin", StringComparison.Ordinal)).Range);
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task HashMismatchIsNotInstalledAndPartialIsDropped()
    {
        byte[] a = Encoding.UTF8.GetBytes("the real file");
        byte[] served = Encoding.UTF8.GetBytes("a tampered one");
        using var server = new StubServer(new() { ["/o/r/resolve/" + Rev + "/dir/a.bin"] = served });
        // The catalog expects a's hash; the server hands out something else of the listed size.
        var entry = MakeEntry(server, a, [1]) with { Files = [StubFile(server, "dir/a.bin", a) with { Size = served.Length }] };

        string root = Path.Combine(Path.GetTempPath(), "stingray-catalog-test-" + Guid.NewGuid().ToString("N"));
        try
        {
            var home = new ModelHome(root);
            using var http = new HttpClient();
            await Assert.ThrowsAsync<ModelHashMismatchException>(() =>
                ModelInstaller.EnsureAsync(entry, home, http, null, null, TestContext.Current.CancellationToken));
            Assert.False(File.Exists(Path.Combine(root, "a.bin")));
            Assert.False(File.Exists(Path.Combine(root, "a.bin" + ModelHome.PartialSuffix)));
            Assert.Equal(InstallState.Missing, home.StateOf(entry));
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task StaleRevisionFailsAndInstallsNothing()
    {
        byte[] a = Encoding.UTF8.GetBytes("pinned content");
        // The server only has the file at a different commit; the entry pins another one.
        using var server = new StubServer(new() { ["/o/r/resolve/ffffffffffffffffffffffffffffffffffffffff/dir/a.bin"] = a });
        var entry = MakeEntry(server, a, [1]) with { Files = [StubFile(server, "dir/a.bin", a)] };

        string root = Path.Combine(Path.GetTempPath(), "stingray-catalog-test-" + Guid.NewGuid().ToString("N"));
        try
        {
            var home = new ModelHome(root);
            using var http = new HttpClient();
            await Assert.ThrowsAsync<HttpRequestException>(() =>
                ModelInstaller.EnsureAsync(entry, home, http, null, null, TestContext.Current.CancellationToken, offline: false));
            Assert.False(File.Exists(Path.Combine(root, "a.bin")));
            Assert.Equal(InstallState.Missing, home.StateOf(entry));
            Assert.Contains("/resolve/" + Rev + "/", server.RangesSeen.Single().Path, StringComparison.Ordinal);
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task OfflineModeNeverTouchesTheNetworkButStillVerifiesLocalFiles()
    {
        byte[] a = Encoding.UTF8.GetBytes("local model bytes");
        using var server = new StubServer(new() { ["/o/r/resolve/" + Rev + "/dir/a.bin"] = a });
        var entry = MakeEntry(server, a, [1]) with { Files = [StubFile(server, "dir/a.bin", a)] };

        string root = Path.Combine(Path.GetTempPath(), "stingray-catalog-test-" + Guid.NewGuid().ToString("N"));
        try
        {
            var home = new ModelHome(root);
            using var http = new HttpClient();
            await Assert.ThrowsAsync<ModelOfflineException>(() =>
                ModelInstaller.EnsureAsync(entry, home, http, null, null, TestContext.Current.CancellationToken, offline: true));
            Assert.Empty(server.RangesSeen);

            // A hand-placed correct file (the "explicit local model" case) is accepted offline.
            File.WriteAllBytes(Path.Combine(root, "a.bin"), a);
            string main = await ModelInstaller.EnsureAsync(entry, home, http, null, null, TestContext.Current.CancellationToken, offline: true);
            Assert.Equal(Path.Combine(root, "a.bin"), main);
            Assert.Empty(server.RangesSeen);

            // A same-size but different file is rejected, offline or not.
            File.WriteAllBytes(Path.Combine(root, "a.bin"), Encoding.UTF8.GetBytes("LOCAL MODEL BYTES"));
            await Assert.ThrowsAsync<ModelHashMismatchException>(() =>
                ModelInstaller.EnsureAsync(entry, home, http, null, null, TestContext.Current.CancellationToken, offline: true));
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void ModelHomeHonoursOverride()
    {
        string? old = Environment.GetEnvironmentVariable(ModelHome.EnvironmentVariable);
        try
        {
            Environment.SetEnvironmentVariable(ModelHome.EnvironmentVariable, Path.GetTempPath());
            Assert.Equal(Path.GetFullPath(Path.GetTempPath()), ModelHome.Default().Root);
            Environment.SetEnvironmentVariable(ModelHome.EnvironmentVariable, null);
            Assert.EndsWith(Path.Combine("stingray", "models"), ModelHome.DefaultRoot(), StringComparison.Ordinal);
        }
        finally
        {
            Environment.SetEnvironmentVariable(ModelHome.EnvironmentVariable, old);
        }
    }

    private const string Rev = "0123456789abcdef0123456789abcdef01234567";

    private static CatalogEntry MakeEntry(StubServer server, byte[] a, byte[] b) => new(
        Id: "test", Task: "chat", Why: "test",
        Files:
        [
            StubFile(server, "dir/a.bin", a),
            StubFile(server, "b.json", b),
        ],
        Licence: "MIT", LicenceNeedsConsent: false, Hardware: "", Speed: "", Evidence: "", RunTemplate: "x {0}");

    /// <summary>A catalog file whose URL points at the local stub server.</summary>
    private static CatalogFile StubFile(StubServer server, string path, byte[] content) =>
        new CatalogFile("o/r", Rev, path, Convert.ToHexStringLower(SHA256.HashData(content)), content.Length) { BaseUrl = server.BaseUrl };

    /// <summary>Minimal HTTP/1.1 file server with Range support, on a loopback port.</summary>
    private sealed class StubServer : IDisposable
    {
        private readonly HttpListener _listener = new();
        private readonly Dictionary<string, byte[]> _files;
        public List<(string Path, string? Range)> RangesSeen { get; } = [];
        public string BaseUrl { get; }

        public StubServer(Dictionary<string, byte[]> files)
        {
            _files = files;
            var probe = new TcpListener(IPAddress.Loopback, 0);
            probe.Start();
            int port = ((IPEndPoint)probe.LocalEndpoint).Port;
            probe.Stop();
            BaseUrl = $"http://127.0.0.1:{port}";
            _listener.Prefixes.Add(BaseUrl + "/");
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
                string path = ctx.Request.Url!.AbsolutePath;
                string? range = ctx.Request.Headers["Range"];
                lock (RangesSeen) RangesSeen.Add((path, range?.Replace("bytes=", "", StringComparison.Ordinal)));
                if (!_files.TryGetValue(path, out var data))
                {
                    ctx.Response.StatusCode = 404;
                    ctx.Response.Close();
                    continue;
                }
                int start = 0;
                if (range is not null)
                {
                    start = int.Parse(range["bytes=".Length..].TrimEnd('-'), System.Globalization.CultureInfo.InvariantCulture);
                    ctx.Response.StatusCode = 206;
                }
                ctx.Response.ContentLength64 = data.Length - start;
                await ctx.Response.OutputStream.WriteAsync(data.AsMemory(start));
                ctx.Response.Close();
            }
        }

        public void Dispose() => _listener.Close();
    }
}
