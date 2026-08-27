using System.Linq;
using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Server;
using OpenTail.Stingray.Sessions;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// Wire contracts for the named-session endpoints under <c>/v1/sessions</c>.
///
/// <para>These cover the outcomes that cannot be eyeballed: optimistic-concurrency conflicts and
/// idempotent replay. A turn is append-only and mutates committed revision, so "did the second
/// call re-run the model or replay the first result?" is the whole contract — and it is invisible
/// in a green build without a test that asserts it.</para>
///
/// <para><see cref="IServerSessionRuntime"/> is registered with <c>TryAddSingleton</c>, so
/// registering a stub first wins over the production relay. That is the only seam available:
/// the interface exposes a concrete <see cref="HotSessionRuntime"/> rather than an abstraction,
/// so an enabled runtime has to be a real one over a fake forward pass.</para>
/// </summary>
public sealed class SessionEndpointTests : IDisposable
{
    private readonly List<WebApplicationFactory<Program>> _factories = new();

    public void Dispose()
    {
        foreach (var factory in _factories) factory.Dispose();
    }

    private const int Eos = 31;

    // Mirrors the fakes the Sessions suite already uses; they are private nested types there, so
    // this is a local copy rather than a shared one.
    private sealed class Tokenizer : ITokenizer
    {
        public int VocabSize => 64;
        public int BosTokenId => 0;
        public int EosTokenId => Eos;
        public int UnknownTokenId => 0;
        public int PadTokenId => Eos;
        public bool AddBosToken => false;
        public IReadOnlyCollection<int> EogTokenIds => [Eos];
        public IReadOnlyList<int> Encode(string text) => [1, 2];
        public string Decode(IEnumerable<int> tokens) => "tok";
        public byte[] DecodeBytes(int token) => [];
    }

    private sealed class FakeCache : IRewindableSequenceKvCache
    {
        public int LogicalPosition { get; set; }
        public bool CanRewindTo(int logicalPosition) => logicalPosition >= 0 && logicalPosition <= LogicalPosition;
        public void RewindTo(int logicalPosition) => LogicalPosition = logicalPosition;
        public void Dispose() { }
    }

    private sealed class FakeForwardPass : IBatchedForwardPass
    {
        public bool SnapKvEnabled => false;
        public long KvBytesPerToken => 1;
        public int MaxSeqLen => 512;
        public bool PrefillDequantCacheActive => false;
        public ISequenceKvCache CreateCache() => new FakeCache();

        public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos = 0)
        {
            var retained = Assert.IsType<FakeCache>(cache);
            retained.LogicalPosition = startPos + tokens.Count;
            var logits = new float[64];
            logits[Eos] = 1f;   // stop immediately: these tests are about the HTTP contract
            return logits;
        }

        public float[]?[] PrefillPackedMulti(ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits) =>
            throw new NotSupportedException();

        // Decode runs through here. Returning EOS ends the turn after one step, which keeps these
        // tests about the HTTP contract while still letting a turn COMPLETE — throwing instead made
        // every turn come back "failed" with the revision unmoved, quietly voiding the
        // revision-conflict and replay assertions below.
        public float[][] BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches)
        {
            var rows = new float[tokens.Length][];
            for (int i = 0; i < tokens.Length; i++)
            {
                if (caches[i] is FakeCache cache) cache.LogicalPosition = positions[i] + 1;
                var logits = new float[64];
                logits[Eos] = 1f;
                rows[i] = logits;
            }
            return rows;
        }
    }

    /// <summary>Sessions unavailable — the production default when EnableSessions is not set.</summary>
    private sealed class UnavailableSessions : IServerSessionRuntime
    {
        public HotSessionRuntime? Runtime => null;
        public ColdSessionRuntime? ColdRuntime => null;
        public string? UnavailabilityReason => "test lane does not support sessions.";
    }

    private sealed class EnabledSessions : IServerSessionRuntime, IDisposable
    {
        private readonly ContinuousBatchingEngine _engine =
            new(new FakeForwardPass(), new Tokenizer(), "test", maxBatchSize: 1);

        public EnabledSessions() => Runtime = new HotSessionRuntime(_engine, new Tokenizer());
        public HotSessionRuntime? Runtime { get; }
        // Hot lane only: the durable wrapper has its own storage contract and is out of scope here.
        public ColdSessionRuntime? ColdRuntime => null;
        public string? UnavailabilityReason => null;
        public void Dispose() => _engine.Dispose();
    }

    /// <summary>
    /// Hot runtime plus a durable wrapper over a scratch directory, mirroring how
    /// <c>InferenceEngineLoader</c> builds one when <c>SessionStorageDirectory</c> is configured.
    /// </summary>
    private sealed class DurableSessions : IServerSessionRuntime, IDisposable
    {
        private readonly ContinuousBatchingEngine _engine =
            new(new FakeForwardPass(), new Tokenizer(), "test", maxBatchSize: 1);

        public DurableSessions(string storageDirectory)
        {
            StorageDirectory = storageDirectory;
            Directory.CreateDirectory(storageDirectory);
            Runtime = new HotSessionRuntime(_engine, new Tokenizer());
            ColdRuntime = new ColdSessionRuntime(Runtime, _engine, storageDirectory, ModelFormat.Gguf);
        }

        public string StorageDirectory { get; }
        public HotSessionRuntime? Runtime { get; }
        public ColdSessionRuntime? ColdRuntime { get; }
        public string? UnavailabilityReason => null;
        public void Dispose() => _engine.Dispose();
    }

    /// <summary>A scratch directory that removes itself, so a failing test cannot leak packs.</summary>
    private sealed class TempDirectory : IDisposable
    {
        public TempDirectory()
        {
            Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"opentail_sessions_{Guid.NewGuid():N}");
            Directory.CreateDirectory(Path);
        }

        public string Path { get; }

        public void Dispose()
        {
            try { if (Directory.Exists(Path)) Directory.Delete(Path, recursive: true); }
            catch (IOException) { }
        }
    }


    /// <summary>Fails a turn with a message carrying a filesystem path, as a real IOException would.</summary>
    private sealed class ThrowingForwardPass : IBatchedForwardPass
    {
        public const string SecretPath = @"C:\Users\dmitri\private-models\unreleased-sk-9137.gguf";

        public bool SnapKvEnabled => false;
        public long KvBytesPerToken => 1;
        public int MaxSeqLen => 512;
        public bool PrefillDequantCacheActive => false;
        public ISequenceKvCache CreateCache() => new FakeCache();

        public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos = 0)
            => throw new IOException($"Could not find file '{SecretPath}'.");

        public float[]?[] PrefillPackedMulti(ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits) =>
            throw new NotSupportedException();

        public float[][] BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches) =>
            throw new NotSupportedException();
    }

    private sealed class FailingSessions : IServerSessionRuntime, IDisposable
    {
        private readonly ContinuousBatchingEngine _engine =
            new(new ThrowingForwardPass(), new Tokenizer(), "test", maxBatchSize: 1);

        public FailingSessions() => Runtime = new HotSessionRuntime(_engine, new Tokenizer());
        public HotSessionRuntime? Runtime { get; }
        public ColdSessionRuntime? ColdRuntime => null;
        public string? UnavailabilityReason => null;
        public void Dispose() => _engine.Dispose();
    }

    private HttpClient CreateClient(IServerSessionRuntime sessions)
    {
        var factory = new WebApplicationFactory<Program>()
            .WithWebHostBuilder(builder => builder.ConfigureServices(services =>
            {
                services.AddSingleton<IInferenceEngine>(new FakeInferenceEngine("test-model"));
                services.AddSingleton(sessions);
            }));
        _factories.Add(factory);
        return factory.CreateClient();
    }

    private static async Task<Guid> CreateSessionAsync(HttpClient client)
    {
        var created = await client.PostAsync("/v1/sessions", content: null);
        Assert.Equal(HttpStatusCode.Created, created.StatusCode);
        using var json = JsonDocument.Parse(await created.Content.ReadAsStringAsync());
        return Guid.Parse(json.RootElement.GetProperty("id").GetString()!);
    }

    private static object TurnBody(string prompt, long revision, int maxTokens = 4, string? operationId = null) =>
        operationId is null
            ? new { append_prompt = prompt, expected_revision = revision, max_tokens = maxTokens }
            : new { append_prompt = prompt, expected_revision = revision, max_tokens = maxTokens, operation_id = operationId };

    // ── Unavailable lane ──────────────────────────────────────────────────

    /// <summary>
    /// Every route must refuse with the same shape when the load lane cannot support sessions.
    /// The turns route is included deliberately: it is the newest and the easiest to leave unmapped.
    /// </summary>
    [Fact]
    public async Task AllRoutes_SessionsUnavailable_Return409WithReason()
    {
        var client = CreateClient(new UnavailableSessions());
        var id = Guid.NewGuid();

        var responses = new[]
        {
            await client.PostAsync("/v1/sessions", content: null),
            await client.GetAsync($"/v1/sessions/{id}"),
            await client.GetAsync($"/v1/sessions/{id}/operations/{Guid.NewGuid()}"),
            await client.PostAsJsonAsync($"/v1/sessions/{id}/turns", TurnBody("hi", 0)),
            await client.PostAsJsonAsync($"/v1/sessions/{id}/skills", new { name = "s" }),
            await client.GetAsync($"/v1/sessions/{id}/skills"),
            await client.DeleteAsync($"/v1/sessions/{id}/skills/s"),
            await client.PostAsJsonAsync($"/v1/sessions/{id}/tool-calls/validate", new { name = "t" }),
            await client.DeleteAsync($"/v1/sessions/{id}"),
        };

        foreach (var response in responses)
        {
            Assert.Equal(HttpStatusCode.Conflict, response.StatusCode);
            using var json = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
            Assert.Equal("session_unavailable", json.RootElement.GetProperty("type").GetString());
            Assert.Contains("does not support sessions", json.RootElement.GetProperty("message").GetString(), StringComparison.Ordinal);
        }
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────

    [Fact]
    public async Task Session_CreateGetDelete_RoundTrips()
    {
        using var sessions = new EnabledSessions();
        var client = CreateClient(sessions);

        Guid id = await CreateSessionAsync(client);
        Assert.Equal(HttpStatusCode.OK, (await client.GetAsync($"/v1/sessions/{id}")).StatusCode);
        Assert.Equal(HttpStatusCode.NoContent, (await client.DeleteAsync($"/v1/sessions/{id}")).StatusCode);

        // Gone afterwards, and deleting twice is a 404 rather than a second success.
        Assert.Equal(HttpStatusCode.NotFound, (await client.GetAsync($"/v1/sessions/{id}")).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await client.DeleteAsync($"/v1/sessions/{id}")).StatusCode);
    }

    /// <summary>
    /// The optimistic-concurrency contract, end to end: read <c>committed_revision</c>, echo it back
    /// as <c>expected_revision</c>. That is the only pattern the API admits, and it used to fail on
    /// the SECOND turn — the server advertised the cursor's accepted-position count (6) while
    /// conflict detection compared against the store's turn counter (1), so it rejected the exact
    /// value it had just published: <c>409 "Expected revision 6, but current revision is 1"</c>.
    ///
    /// <para>The earlier restart test did not catch this because a persisted revision was ALSO a
    /// position count, so on the durable path the two wrong numbers agreed. This test runs against a
    /// live session, where they do not.</para>
    /// </summary>
    [Fact]
    public async Task Turn_AdvertisedCommittedRevision_IsAcceptedAsExpectedRevision()
    {
        using var sessions = new EnabledSessions();
        var client = CreateClient(sessions);
        Guid id = await CreateSessionAsync(client);

        var first = await client.PostAsJsonAsync($"/v1/sessions/{id}/turns", TurnBody("hello", 0));
        Assert.Equal(HttpStatusCode.OK, first.StatusCode);

        var get = await client.GetAsync($"/v1/sessions/{id}");
        Assert.Equal(HttpStatusCode.OK, get.StatusCode);
        using var doc = JsonDocument.Parse(await get.Content.ReadAsStringAsync());
        long advertised = doc.RootElement.GetProperty("committed_revision").GetInt64();

        var second = await client.PostAsJsonAsync($"/v1/sessions/{id}/turns", TurnBody("world", advertised));
        Assert.Equal(HttpStatusCode.OK, second.StatusCode);

        // And the value must keep advancing, so a third turn can repeat the pattern rather than the
        // token happening to be a fixed point.
        var get2 = await client.GetAsync($"/v1/sessions/{id}");
        using var doc2 = JsonDocument.Parse(await get2.Content.ReadAsStringAsync());
        long next = doc2.RootElement.GetProperty("committed_revision").GetInt64();
        Assert.True(next > advertised, $"committed_revision must advance ({advertised} -> {next}).");

        var third = await client.PostAsJsonAsync($"/v1/sessions/{id}/turns", TurnBody("again", next));
        Assert.Equal(HttpStatusCode.OK, third.StatusCode);

        // A stale revision must still conflict — the fix must not have disabled conflict detection.
        var stale = await client.PostAsJsonAsync($"/v1/sessions/{id}/turns", TurnBody("stale", advertised));
        Assert.Equal(HttpStatusCode.Conflict, stale.StatusCode);
    }

    [Fact]
    public async Task Turn_UnknownSession_Returns404()
    {
        using var sessions = new EnabledSessions();
        var client = CreateClient(sessions);

        var response = await client.PostAsJsonAsync($"/v1/sessions/{Guid.NewGuid()}/turns", TurnBody("hi", 0));
        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
    }

    /// <summary>
    /// A client that loses the turn response can reconnect using its operation id and recover the
    /// same bounded result without re-running the model. The endpoint is deliberately a lookup;
    /// it must not mutate the committed revision or claim a retry replay happened.
    /// </summary>
    [Fact]
    public async Task OperationLookup_CompletedHotTurn_ReturnsRetainedResult()
    {
        using var sessions = new EnabledSessions();
        var client = CreateClient(sessions);
        Guid id = await CreateSessionAsync(client);
        string operationId = Guid.NewGuid().ToString();

        var turn = await client.PostAsJsonAsync($"/v1/sessions/{id}/turns", TurnBody("hi", 0, operationId: operationId));
        Assert.Equal(HttpStatusCode.OK, turn.StatusCode);
        using var turnJson = JsonDocument.Parse(await turn.Content.ReadAsStringAsync());
        long revision = turnJson.RootElement.GetProperty("session").GetProperty("committed_revision").GetInt64();

        var lookup = await client.GetAsync($"/v1/sessions/{id}/operations/{operationId}");

        Assert.Equal(HttpStatusCode.OK, lookup.StatusCode);
        using var json = JsonDocument.Parse(await lookup.Content.ReadAsStringAsync());
        Assert.Equal(Guid.Parse(operationId).ToString("N"), json.RootElement.GetProperty("operation_id").GetString());
        Assert.Equal("completed", json.RootElement.GetProperty("state").GetString());
        Assert.Equal(revision, json.RootElement.GetProperty("session").GetProperty("committed_revision").GetInt64());
        Assert.Equal(1, json.RootElement.GetProperty("operation_revision").GetInt64());
        Assert.True(json.RootElement.GetProperty("completed_at").GetDateTimeOffset() >=
            json.RootElement.GetProperty("created_at").GetDateTimeOffset());
        Assert.Equal("", json.RootElement.GetProperty("text").GetString());
        Assert.Equal("", json.RootElement.GetProperty("thinking").GetString());
    }

    [Fact]
    public async Task OperationLookup_UnknownOperationOrSession_Returns404()
    {
        using var sessions = new EnabledSessions();
        var client = CreateClient(sessions);
        Guid id = await CreateSessionAsync(client);

        Assert.Equal(HttpStatusCode.NotFound,
            (await client.GetAsync($"/v1/sessions/{id}/operations/{Guid.NewGuid()}")).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound,
            (await client.GetAsync($"/v1/sessions/{Guid.NewGuid()}/operations/{Guid.NewGuid()}")).StatusCode);
    }

    // ── Skills & tool-call validation ─────────────────────────────────────

    [Fact]
    public async Task Skills_AttachListDetach_RoundTrips()
    {
        using var sessions = new EnabledSessions();
        var client = CreateClient(sessions);
        Guid id = await CreateSessionAsync(client);

        var attach = await client.PostAsJsonAsync($"/v1/sessions/{id}/skills", new
        {
            name = "weather",
            description = "Weather lookups",
            tools = new[] { new { name = "get_forecast", description = "Fetches a forecast" } },
        });
        Assert.Equal(HttpStatusCode.Created, attach.StatusCode);
        using (var attached = JsonDocument.Parse(await attach.Content.ReadAsStringAsync()))
        {
            Assert.Equal("weather", attached.RootElement.GetProperty("name").GetString());
            Assert.Equal("get_forecast", attached.RootElement.GetProperty("tools")[0].GetString());
        }

        var list = await client.GetAsync($"/v1/sessions/{id}/skills");
        Assert.Equal(HttpStatusCode.OK, list.StatusCode);
        using (var listed = JsonDocument.Parse(await list.Content.ReadAsStringAsync()))
        {
            var skills = listed.RootElement.GetProperty("skills");
            Assert.Equal(1, skills.GetArrayLength());
            Assert.Equal("weather", skills[0].GetProperty("name").GetString());
        }

        Assert.Equal(HttpStatusCode.NoContent, (await client.DeleteAsync($"/v1/sessions/{id}/skills/weather")).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await client.DeleteAsync($"/v1/sessions/{id}/skills/weather")).StatusCode);

        using var afterDetach = JsonDocument.Parse(
            await (await client.GetAsync($"/v1/sessions/{id}/skills")).Content.ReadAsStringAsync());
        Assert.Equal(0, afterDetach.RootElement.GetProperty("skills").GetArrayLength());
    }

    [Fact]
    public async Task ValidateToolCall_AuthorizesOnlyToolsFromAttachedSkills()
    {
        using var sessions = new EnabledSessions();
        var client = CreateClient(sessions);
        Guid id = await CreateSessionAsync(client);

        await client.PostAsJsonAsync($"/v1/sessions/{id}/skills", new
        {
            name = "weather",
            tools = new[] { new { name = "get_forecast" } },
        });

        var authorized = await client.PostAsJsonAsync(
            $"/v1/sessions/{id}/tool-calls/validate", new { name = "get_forecast", arguments = new { city = "SF" } });
        using (var doc = JsonDocument.Parse(await authorized.Content.ReadAsStringAsync()))
            Assert.True(doc.RootElement.GetProperty("authorized").GetBoolean());

        var unauthorized = await client.PostAsJsonAsync(
            $"/v1/sessions/{id}/tool-calls/validate", new { name = "delete_everything" });
        using (var doc = JsonDocument.Parse(await unauthorized.Content.ReadAsStringAsync()))
            Assert.False(doc.RootElement.GetProperty("authorized").GetBoolean());
    }

    [Fact]
    public async Task Skills_UnknownSession_Returns404()
    {
        using var sessions = new EnabledSessions();
        var client = CreateClient(sessions);
        var id = Guid.NewGuid();

        Assert.Equal(HttpStatusCode.NotFound,
            (await client.PostAsJsonAsync($"/v1/sessions/{id}/skills", new { name = "s" })).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await client.GetAsync($"/v1/sessions/{id}/skills")).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await client.DeleteAsync($"/v1/sessions/{id}/skills/s")).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound,
            (await client.PostAsJsonAsync($"/v1/sessions/{id}/tool-calls/validate", new { name = "t" })).StatusCode);
    }

    // ── Request validation ────────────────────────────────────────────────

    public static TheoryData<object, string> InvalidTurnBodies => new()
    {
        { new { append_prompt = "", expected_revision = 0L, max_tokens = 4 }, "append_prompt" },
        { new { append_prompt = "hi", max_tokens = 4 }, "expected_revision" },
        { new { append_prompt = "hi", expected_revision = -1L, max_tokens = 4 }, "expected_revision" },
        { new { append_prompt = "hi", expected_revision = 0L, max_tokens = 0 }, "max_tokens" },
    };

    [Theory]
    [MemberData(nameof(InvalidTurnBodies))]
    public async Task Turn_InvalidRequest_Returns400NamingTheField(object body, string expectedField)
    {
        using var sessions = new EnabledSessions();
        var client = CreateClient(sessions);
        Guid id = await CreateSessionAsync(client);

        var response = await client.PostAsJsonAsync($"/v1/sessions/{id}/turns", body);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        using var json = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("invalid_request_error", json.RootElement.GetProperty("type").GetString());
        // Naming the offending field is the difference between a usable 400 and a guess.
        Assert.Contains(expectedField, json.RootElement.GetProperty("message").GetString(), StringComparison.Ordinal);
    }

    // ── Optimistic concurrency ────────────────────────────────────────────

    /// <summary>
    /// A turn is append-only, so a stale expected_revision must be refused rather than silently
    /// applied on top of work the caller has not seen.
    /// </summary>
    [Fact]
    public async Task Turn_StaleRevision_Returns409Conflict()
    {
        using var sessions = new EnabledSessions();
        var client = CreateClient(sessions);
        Guid id = await CreateSessionAsync(client);

        var first = await client.PostAsJsonAsync($"/v1/sessions/{id}/turns", TurnBody("one", 0));
        Assert.Equal(HttpStatusCode.OK, first.StatusCode);

        // Revision 0 is now stale: the accepted turn advanced it.
        var stale = await client.PostAsJsonAsync($"/v1/sessions/{id}/turns", TurnBody("two", 0));

        Assert.Equal(HttpStatusCode.Conflict, stale.StatusCode);
        using var json = JsonDocument.Parse(await stale.Content.ReadAsStringAsync());
        Assert.Equal("session_revision_conflict", json.RootElement.GetProperty("type").GetString());
    }

    // ── Idempotency ───────────────────────────────────────────────────────

    /// <summary>
    /// The regression this pins: the handler used to default operation_id to a fresh
    /// <c>Guid.NewGuid()</c>, so a retried request looked like a brand-new operation and the turn
    /// ran twice. A client that times out and retries — the exact caller idempotency exists for —
    /// got no protection at all. Identical content at the same revision must replay.
    /// </summary>
    [Fact]
    public async Task Turn_RetriedWithoutOperationId_ReplaysInsteadOfReRunning()
    {
        using var sessions = new EnabledSessions();
        var client = CreateClient(sessions);
        Guid id = await CreateSessionAsync(client);

        var first = await client.PostAsJsonAsync($"/v1/sessions/{id}/turns", TurnBody("same", 0));
        Assert.Equal(HttpStatusCode.OK, first.StatusCode);
        using var firstJson = JsonDocument.Parse(await first.Content.ReadAsStringAsync());
        Assert.False(firstJson.RootElement.GetProperty("idempotent_replay").GetBoolean());
        string operationId = firstJson.RootElement.GetProperty("operation_id").GetString()!;
        long revisionAfterFirst = firstJson.RootElement.GetProperty("session").GetProperty("committed_revision").GetInt64();

        // Byte-identical retry, no client-supplied key.
        var retry = await client.PostAsJsonAsync($"/v1/sessions/{id}/turns", TurnBody("same", 0));

        Assert.Equal(HttpStatusCode.OK, retry.StatusCode);
        using var retryJson = JsonDocument.Parse(await retry.Content.ReadAsStringAsync());
        Assert.True(retryJson.RootElement.GetProperty("idempotent_replay").GetBoolean());
        Assert.Equal(operationId, retryJson.RootElement.GetProperty("operation_id").GetString());
        // The decisive assertion: a replay must not append a second turn.
        Assert.Equal(revisionAfterFirst,
            retryJson.RootElement.GetProperty("session").GetProperty("committed_revision").GetInt64());
    }

    /// <summary>
    /// The derived key must be content-addressed, not constant: two genuinely different turns at
    /// the same revision must not collide into one replayed operation.
    /// </summary>
    [Fact]
    public async Task Turn_DifferentPromptSameRevision_IsNotTreatedAsReplay()
    {
        using var sessions = new EnabledSessions();
        var client = CreateClient(sessions);
        Guid id = await CreateSessionAsync(client);

        var first = await client.PostAsJsonAsync($"/v1/sessions/{id}/turns", TurnBody("alpha", 0));
        Assert.Equal(HttpStatusCode.OK, first.StatusCode);
        using var firstJson = JsonDocument.Parse(await first.Content.ReadAsStringAsync());
        string firstOperation = firstJson.RootElement.GetProperty("operation_id").GetString()!;

        // Same revision, different text: a conflict is correct (the revision is now stale), but it
        // must NOT be reported as a replay of the first operation.
        var other = await client.PostAsJsonAsync($"/v1/sessions/{id}/turns", TurnBody("beta", 0));
        if (other.StatusCode == HttpStatusCode.OK)
        {
            using var otherJson = JsonDocument.Parse(await other.Content.ReadAsStringAsync());
            Assert.NotEqual(firstOperation, otherJson.RootElement.GetProperty("operation_id").GetString());
            Assert.False(otherJson.RootElement.GetProperty("idempotent_replay").GetBoolean());
        }
        else
        {
            Assert.Equal(HttpStatusCode.Conflict, other.StatusCode);
            using var otherJson = JsonDocument.Parse(await other.Content.ReadAsStringAsync());
            Assert.Equal("session_revision_conflict", otherJson.RootElement.GetProperty("type").GetString());
        }
    }

    /// <summary>A client-supplied key still wins, so two deliberate turns with identical text
    /// remain expressible.</summary>
    [Fact]
    public async Task Turn_ClientSuppliedOperationId_IsHonoured()
    {
        using var sessions = new EnabledSessions();
        var client = CreateClient(sessions);
        Guid id = await CreateSessionAsync(client);

        string key = Guid.NewGuid().ToString();
        var response = await client.PostAsJsonAsync($"/v1/sessions/{id}/turns", TurnBody("hi", 0, operationId: key));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using var json = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal(Guid.Parse(key).ToString("N"), json.RootElement.GetProperty("operation_id").GetString());
    }



    /// <summary>Runs one turn and returns the session's committed revision afterwards.</summary>
    private static async Task<long> RunTurnAsync(HttpClient client, Guid id, string prompt, long expectedRevision)
    {
        var response = await client.PostAsJsonAsync($"/v1/sessions/{id}/turns", TurnBody(prompt, expectedRevision));
        string body = await response.Content.ReadAsStringAsync();
        // Carry the body into the message: "expected OK, got Conflict" alone cannot distinguish a
        // revision conflict from an operation conflict from an unavailable runtime.
        Assert.True(response.StatusCode == HttpStatusCode.OK,
            $"turn '{prompt}' at revision {expectedRevision} returned {(int)response.StatusCode}: {body}");
        using var json = JsonDocument.Parse(body);
        return json.RootElement.GetProperty("session").GetProperty("committed_revision").GetInt64();
    }

    // ── Durable seam ──────────────────────────────────────────────────────

    /// <summary>
    /// The point of the durable lane: a completed turn is evicted to disk, and a LATER server —
    /// a fresh relay and a fresh hot runtime over the same directory, i.e. what a restart looks
    /// like — serves the session again without the caller knowing it was ever cold.
    /// </summary>
    [Fact]
    public async Task DurableSession_SurvivesRuntimeRestart_AndIsRestoredOnRead()
    {
        using var storage = new TempDirectory();

        Guid id;
        using (var first = new DurableSessions(storage.Path))
        {
            var client = CreateClient(first);
            id = await CreateSessionAsync(client);

            var turn = await client.PostAsJsonAsync($"/v1/sessions/{id}/turns", TurnBody("hi", 0));
            Assert.Equal(HttpStatusCode.OK, turn.StatusCode);

            // A completed turn must have produced an on-disk manifest; without it there is nothing
            // to restore and the assertion below would pass for the wrong reason.
            Assert.True(File.Exists(Path.Combine(storage.Path, $"{id:N}.manifest")),
                "a completed turn did not write a manifest, so the restore below proves nothing.");
        }

        // Fresh runtime over the same directory: nothing is in memory, so a successful read can
        // only have come from disk.
        using var second = new DurableSessions(storage.Path);
        var restoredClient = CreateClient(second);

        var response = await restoredClient.GetAsync($"/v1/sessions/{id}");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using var json = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal(id.ToString("N"), json.RootElement.GetProperty("id").GetString());
    }

    /// <summary>
    /// Restart continuation is not enough for a caller that lost the HTTP response: the same
    /// operation id must retrieve and replay the bounded committed result without running a
    /// second turn after a fresh server has restored the KV state from disk.
    /// </summary>
    [Fact]
    public async Task DurableSession_RestartRestoresOperationLookupAndIdempotentReplay()
    {
        using var storage = new TempDirectory();
        Guid id;
        string operationId = Guid.NewGuid().ToString();

        using (var first = new DurableSessions(storage.Path))
        {
            var client = CreateClient(first);
            id = await CreateSessionAsync(client);
            var turn = await client.PostAsJsonAsync($"/v1/sessions/{id}/turns", TurnBody("hi", 0, operationId: operationId));
            Assert.Equal(HttpStatusCode.OK, turn.StatusCode);
        }

        using var second = new DurableSessions(storage.Path);
        var restoredClient = CreateClient(second);

        var lookup = await restoredClient.GetAsync($"/v1/sessions/{id}/operations/{operationId}");
        Assert.Equal(HttpStatusCode.OK, lookup.StatusCode);
        using var lookupJson = JsonDocument.Parse(await lookup.Content.ReadAsStringAsync());
        Assert.Equal("completed", lookupJson.RootElement.GetProperty("state").GetString());
        long committedRevision = lookupJson.RootElement.GetProperty("session").GetProperty("committed_revision").GetInt64();

        var retry = await restoredClient.PostAsJsonAsync($"/v1/sessions/{id}/turns", TurnBody("hi", 0, operationId: operationId));
        Assert.Equal(HttpStatusCode.OK, retry.StatusCode);
        using var retryJson = JsonDocument.Parse(await retry.Content.ReadAsStringAsync());
        Assert.True(retryJson.RootElement.GetProperty("idempotent_replay").GetBoolean());
        Assert.Equal(committedRevision, retryJson.RootElement.GetProperty("session").GetProperty("committed_revision").GetInt64());

        using var capabilities = JsonDocument.Parse(await (await restoredClient.GetAsync("/capabilities")).Content.ReadAsStringAsync());
        var persistence = capabilities.RootElement.GetProperty("runtime").GetProperty("session_operation_result_persistence");
        Assert.True(persistence.GetProperty("available").GetBoolean());
    }

    /// <summary>
    /// Deleting a session must remove its bytes, not merely forget it. A delete that leaves packs
    /// behind is a slow disk leak that nothing else in the system will ever clean up.
    /// </summary>
    [Fact]
    public async Task DurableSession_Delete_RemovesManifestAndPacksFromDisk()
    {
        using var storage = new TempDirectory();
        using var sessions = new DurableSessions(storage.Path);
        var client = CreateClient(sessions);

        Guid id = await CreateSessionAsync(client);
        Assert.Equal(HttpStatusCode.OK,
            (await client.PostAsJsonAsync($"/v1/sessions/{id}/turns", TurnBody("hi", 0))).StatusCode);

        string manifest = Path.Combine(storage.Path, $"{id:N}.manifest");
        Assert.True(File.Exists(manifest));
        Assert.NotEmpty(Directory.GetFiles(storage.Path, $"*{id:N}*.pack"));

        Assert.Equal(HttpStatusCode.NoContent, (await client.DeleteAsync($"/v1/sessions/{id}")).StatusCode);

        Assert.False(File.Exists(manifest));
        Assert.Empty(Directory.GetFiles(storage.Path, $"*{id:N}*.pack"));
        Assert.Equal(HttpStatusCode.NotFound, (await client.GetAsync($"/v1/sessions/{id}")).StatusCode);
    }

    // ── Capability reporting ──────────────────────────────────────────────

    /// <summary>
    /// The capability report has three states and they are not interchangeable. Hot-only must not
    /// claim restart continuation — a client that saw "available" would trust its session across a
    /// deploy and silently lose it — but it must still advertise the lifecycle it genuinely serves.
    /// </summary>
    [Fact]
    public async Task Capabilities_HotOnlySessions_AdvertiseLifecycleButNotRestartContinuation()
    {
        using var sessions = new EnabledSessions();
        var client = CreateClient(sessions);

        using var json = JsonDocument.Parse(await (await client.GetAsync("/capabilities")).Content.ReadAsStringAsync());
        var root = json.RootElement;

        Assert.True(root.GetProperty("api").GetProperty("session_lifecycle").GetBoolean());
        var restart = root.GetProperty("runtime").GetProperty("session_restart_continuation");
        Assert.False(restart.GetProperty("available").GetBoolean());
        Assert.Contains("in memory", restart.GetProperty("detail").GetString(), StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// With storage configured the claim flips to available and states that bounded idempotent
    /// results come back with the restored CPU-dense session.
    /// </summary>
    [Fact]
    public async Task Capabilities_DurableSessions_ReportRestartContinuationAvailable()
    {
        using var storage = new TempDirectory();
        using var sessions = new DurableSessions(storage.Path);
        var client = CreateClient(sessions);

        using var json = JsonDocument.Parse(await (await client.GetAsync("/capabilities")).Content.ReadAsStringAsync());
        var root = json.RootElement;

        Assert.True(root.GetProperty("api").GetProperty("session_lifecycle").GetBoolean());
        var restart = root.GetProperty("runtime").GetProperty("session_restart_continuation");
        Assert.True(restart.GetProperty("available").GetBoolean());
        Assert.Contains("idempotent", restart.GetProperty("detail").GetString(), StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// A failed operation's reason is an exception message, so it can carry a filesystem path —
    /// which discloses a username, a project, or an unreleased model name. The operations endpoint
    /// is client-facing, so the location must not reach the wire even though the explanation
    /// should. Mirrors what DiagnosticSurfaceRedactionTests enforces on /status.
    /// </summary>
    [Fact]
    public async Task Operation_FailureReason_DoesNotLeakFilesystemPaths()
    {
        using var sessions = new FailingSessions();
        var client = CreateClient(sessions);
        Guid id = await CreateSessionAsync(client);

        var turn = await client.PostAsJsonAsync($"/v1/sessions/{id}/turns", TurnBody("hi", 0));
        Assert.Equal(HttpStatusCode.OK, turn.StatusCode);
        using var turnJson = JsonDocument.Parse(await turn.Content.ReadAsStringAsync());
        string operationId = turnJson.RootElement.GetProperty("operation_id").GetString()!;

        var response = await client.GetAsync($"/v1/sessions/{id}/operations/{operationId}");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        string body = await response.Content.ReadAsStringAsync();

        Assert.DoesNotContain("private-models", body, StringComparison.Ordinal);
        Assert.DoesNotContain("dmitri", body, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(@"C:\", body, StringComparison.Ordinal);

        // Redaction, not suppression: the explanation must survive with the location removed.
        using var json = JsonDocument.Parse(body);
        string? reason = json.RootElement.GetProperty("failure_reason").GetString();
        Assert.False(string.IsNullOrWhiteSpace(reason));
        Assert.Contains("[path]", reason, StringComparison.Ordinal);
    }
}
