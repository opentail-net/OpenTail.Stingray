using OpenTail.Stingray.Engine.Scout;
using System.Net;
using System.Security.Cryptography;
using OpenTail.Stingray.Cli.Scout;
using OpenTail.Stingray.Core.Catalog;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>
/// First run: a task command that finds its model missing offers to install it, with the same consent and confirmation as `setup`,
/// and never downloads without a yes. A tiny catalog entry and a fake server stand in for the real model; no network is used.
/// </summary>
public sealed class FirstRunTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "firstrun-" + Guid.NewGuid().ToString("N"));
    public void Dispose() { try { Directory.Delete(_root, recursive: true); } catch { } }

    private static readonly byte[] Bytes = Enumerable.Range(0, 5000).Select(i => (byte)(i % 251)).ToArray();
    private static string Sha(byte[] b) => Convert.ToHexString(SHA256.HashData(b)).ToLowerInvariant();

    private static CatalogEntry Entry(bool consent = false, string? sha = null) => new(
        "tiny", "chat", "test model",
        [new CatalogFile("o/r", "rev", "tiny.gguf", sha ?? Sha(Bytes), Bytes.Length)],
        consent ? "Custom licence: read it" : "MIT", consent, "1 MB", "n/a", "test", "stingray chat");

    private sealed class Prompt(params bool[] answers) : ISetupPrompt
    {
        private readonly Queue<bool> _answers = new(answers);
        public List<string> Asked { get; } = [];
        public bool Confirm(string question, bool defaultYes) { Asked.Add(question); return _answers.Count > 0 ? _answers.Dequeue() : defaultYes; }
    }

    private sealed class Server(byte[]? body = null) : HttpMessageHandler
    {
        public int Requests;
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage r, CancellationToken ct)
        {
            Interlocked.Increment(ref Requests);
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(body ?? Bytes) });
        }
    }

    private (bool Ok, ResolvedModelTask? R, string? Err) Resolve(Server s, Prompt p, CatalogEntry e, bool interactive = true, Func<string, string?>? env = null, string? modelFile = null, PreflightResult? fit = null)
    {
        bool ok = CatalogTaskResolver.TryResolveOrOffer("chat", null, modelFile, out var r, out var err, default,
            interactive, p, new ModelHome(_root), () => new HttpClient(s), e, env ?? (_ => null), (_, _) => fit);
        return (ok, r, err);
    }

    [Fact]
    public void Interactive_and_missing_asks_once_downloads_verifies_and_resolves()
    {
        var s = new Server(); var p = new Prompt(true);
        var (ok, r, err) = Resolve(s, p, Entry());
        Assert.True(ok, err);
        Assert.Single(p.Asked);
        Assert.StartsWith("Download", p.Asked[0]);
        Assert.Equal(1, s.Requests);
        Assert.Equal(Path.Combine(Path.GetFullPath(_root), "tiny.gguf"), r!.ModelPath);
        Assert.Equal(Bytes, File.ReadAllBytes(r.ModelPath));
        Assert.False(File.Exists(r.ModelPath + ".part"));
    }

    [Fact]
    public void Declining_downloads_nothing_and_leaves_the_one_line_fix()
    {
        var s = new Server(); var p = new Prompt(false);
        var (ok, r, err) = Resolve(s, p, Entry());
        Assert.False(ok);
        Assert.Null(r);
        Assert.Equal(0, s.Requests);
        Assert.Contains("stingray setup chat", err);
        Assert.False(File.Exists(Path.Combine(_root, "tiny.gguf")));
    }

    [Fact]
    public void Non_interactive_never_prompts_and_never_downloads()
    {
        var s = new Server(); var p = new Prompt(true);
        var (ok, _, err) = Resolve(s, p, Entry(), interactive: false);
        Assert.False(ok);
        Assert.Empty(p.Asked);
        Assert.Equal(0, s.Requests);
        Assert.Contains("Run: stingray setup chat", err);
    }

    [Fact]
    public void A_licence_that_needs_consent_is_asked_first_defaults_to_no_and_blocks_the_download()
    {
        var s = new Server(); var p = new Prompt(false);
        var (ok, _, _) = Resolve(s, p, Entry(consent: true));
        Assert.False(ok);
        Assert.Single(p.Asked);
        Assert.Contains("licence", p.Asked[0]);
        Assert.Equal(0, s.Requests);

        var s2 = new Server(); var p2 = new Prompt(true, true);
        var (ok2, _, err2) = Resolve(s2, p2, Entry(consent: true));
        Assert.True(ok2, err2);
        Assert.Equal(2, p2.Asked.Count);
        Assert.Contains("licence", p2.Asked[0]);
        Assert.StartsWith("Download", p2.Asked[1]);
    }

    [Fact]
    public void An_already_installed_model_is_used_without_a_prompt_or_a_request()
    {
        Directory.CreateDirectory(_root);
        File.WriteAllBytes(Path.Combine(_root, "tiny.gguf"), Bytes);
        var s = new Server(); var p = new Prompt(true);
        var (ok, r, _) = Resolve(s, p, Entry());
        Assert.True(ok);
        Assert.Empty(p.Asked);
        Assert.Equal(0, s.Requests);
        Assert.NotNull(r);
    }

    [Fact]
    public void An_interrupted_install_is_offered_again_and_completed()
    {
        Directory.CreateDirectory(_root);
        File.WriteAllBytes(Path.Combine(_root, "tiny.gguf.part"), Bytes[..1000]);
        var s = new Server(); var p = new Prompt(true);
        var (ok, r, err) = Resolve(s, p, Entry());
        Assert.True(ok, err);
        Assert.Equal(Bytes, File.ReadAllBytes(r!.ModelPath));
    }

    [Fact]
    public void A_file_with_the_wrong_hash_is_never_installed()
    {
        var s = new Server(); var p = new Prompt(true);
        var (ok, r, err) = Resolve(s, p, Entry(sha: new string('0', 64)));
        Assert.False(ok);
        Assert.Null(r);
        Assert.False(File.Exists(Path.Combine(_root, "tiny.gguf")));   // only a verified file ever has its final name
        Assert.Contains("stingray setup chat", err);
    }

    [Theory]
    [InlineData("STINGRAY_ALLOW_EXTERNAL", "off")]
    [InlineData("STINGRAY_OFFLINE", "1")]
    [InlineData("HF_HUB_OFFLINE", "1")]
    public void The_external_access_policy_stops_the_download_before_any_download_question(string variable, string value)
    {
        var s = new Server(); var p = new Prompt(true);
        var (ok, _, err) = Resolve(s, p, Entry(), env: n => n == variable ? value : null);
        Assert.False(ok);
        Assert.Empty(p.Asked);
        Assert.Equal(0, s.Requests);
        Assert.Contains("not installed", err);
    }

    [Fact]
    public void A_bad_model_file_path_is_a_plain_error_with_no_install_offer()
    {
        var s = new Server(); var p = new Prompt(true);
        var (ok, _, err) = Resolve(s, p, Entry(), modelFile: Path.Combine(_root, "does-not-exist.gguf"));
        Assert.False(ok);
        Assert.Empty(p.Asked);
        Assert.Contains("not found", err);
    }

    // ── the pre-download fit check (reads the pinned remote index; injected here) ──
    private static PreflightResult Fit(PreflightVerdict v) => new(v, "x", 40L << 30, 16L << 30, 4L << 30);

    [Fact]
    public void Download_anyway_can_be_chosen_interactively_for_a_model_that_will_not_fit()
    {
        var s = new Server(); var p = new Prompt(true);
        var (ok, _, err) = Resolve(s, p, Entry(), interactive: true, fit: Fit(PreflightVerdict.Blocked));
        // interactive: the user is asked "Download anyway?" (default no); here the scripted first answer is yes to that question
        Assert.True(ok, err);
        Assert.Contains(p.Asked, q => q.StartsWith("Download anyway"));
    }

    [Fact]
    public void Declining_download_anyway_downloads_nothing()
    {
        var s = new Server(); var p = new Prompt(false);
        var (ok, _, _) = Resolve(s, p, Entry(), fit: Fit(PreflightVerdict.Blocked));
        Assert.False(ok);
        Assert.Equal(0, s.Requests);
        Assert.Single(p.Asked);
        Assert.StartsWith("Download anyway", p.Asked[0]);
    }

    [Fact]
    public void A_fitting_model_adds_no_extra_question()
    {
        var s = new Server(); var p = new Prompt(true);
        var (ok, _, err) = Resolve(s, p, Entry(), fit: Fit(PreflightVerdict.Allowed));
        Assert.True(ok, err);
        Assert.Single(p.Asked);
        Assert.StartsWith("Download ", p.Asked[0]);
    }

    [Fact]
    public void An_unknown_or_unavailable_check_does_not_get_in_the_way()
    {
        foreach (var fit in new PreflightResult?[] { Fit(PreflightVerdict.Unknown), null })
        {
            var s = new Server(); var p = new Prompt(true);
            var (ok, _, err) = Resolve(s, p, Entry(), fit: fit);
            Assert.True(ok, err);
            Assert.Single(p.Asked);
            Directory.Delete(_root, recursive: true);
        }
    }

    [Fact]
    public void With_yes_or_non_interactive_a_model_that_will_not_fit_is_refused_not_downloaded()
    {
        foreach ((bool yes, bool interactive) in new[] { (true, true), (false, false), (true, false) })
        {
            var s = new Server();
            var outcome = SetupFlow.Run(Entry(), new ModelHome(Path.Combine(_root, "x" + yes + interactive)), new HttpClient(s), yes, acceptLicence: false, interactive, new Prompt(true), default,
                _ => null, (_, _) => Fit(PreflightVerdict.Blocked));
            Assert.False(outcome.Installed);
            Assert.Equal(0, s.Requests);
        }
    }
}