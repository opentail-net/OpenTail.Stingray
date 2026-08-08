using System.Net;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Server;
using OpenTail.Stingray.Sessions;

namespace OpenTail.Stingray.Tests.Server;

/// <summary>
/// Bounded admission rejects excess generation work with HTTP 429 instead of silently retaining
/// an unbounded number of requests. Setting MaxQueuedRequests to zero retains the legacy
/// MaxConcurrentRequests-only behaviour.
/// </summary>
public sealed class ConcurrencyLimitTests
{
    private static HttpClient CreateClient(FakeInferenceEngine fake, int? maxConcurrent,
        int maxQueued = 0, int maxBatch = 1) =>
        new WebApplicationFactory<Program>()
            .WithWebHostBuilder(b => b.ConfigureServices(s =>
            {
                s.Configure<OpenTailStingrayServerOptions>(o =>
                {
                    o.MaxConcurrentRequests = maxConcurrent;
                    o.MaxQueuedRequests = maxQueued;
                    o.MaxBatchSize = maxBatch;
                });
                s.AddSingleton<IInferenceEngine>(fake);
            }))
            .CreateClient();

    private static object ChatRequest() => new
    {
        model = "test-model",
        messages = new[] { new { role = "user", content = "hi" } },
        max_tokens = 5,
        stream = false,
    };

    [Fact]
    public async Task MaxConcurrent1_OverlappingRequest_Rejected429()
    {
        var fake = new FakeInferenceEngine("test-model") { Hold = new TaskCompletionSource() };
        var client = CreateClient(fake, maxConcurrent: 1);

        // Request A enters the engine and blocks (holding the single admission slot).
        var aTask = client.PostAsJsonAsync("/v1/chat/completions", ChatRequest());
        Assert.True(fake.Entered.Wait(TimeSpan.FromSeconds(5)), "Request A never reached the engine.");

        // Request B overlaps A → must be fast-rejected with 429 (it never reaches the engine).
        var bResp = await client.PostAsJsonAsync("/v1/chat/completions", ChatRequest());
        Assert.Equal(HttpStatusCode.TooManyRequests, bResp.StatusCode);
        Assert.True(bResp.Headers.Contains("Retry-After"));
        var body = await bResp.Content.ReadAsStringAsync();
        Assert.Contains("STINGRAY_MAX_QUEUE", body);

        // Release A; it completes normally, and the slot frees for a subsequent request.
        fake.Hold!.SetResult();
        var aResp = await aTask;
        Assert.Equal(HttpStatusCode.OK, aResp.StatusCode);

        var cResp = await client.PostAsJsonAsync("/v1/chat/completions", ChatRequest());
        Assert.Equal(HttpStatusCode.OK, cResp.StatusCode);
    }

    [Fact]
    public async Task Unset_OverlappingRequests_BothSucceed_NoRejection()
    {
        // Default (no limit): the gate is a passthrough — overlapping requests are not rejected.
        var fake = new FakeInferenceEngine("test-model") { Hold = new TaskCompletionSource() };
        var client = CreateClient(fake, maxConcurrent: null);

        var aTask = client.PostAsJsonAsync("/v1/chat/completions", ChatRequest());
        var bTask = client.PostAsJsonAsync("/v1/chat/completions", ChatRequest());
        Assert.True(fake.Entered.Wait(TimeSpan.FromSeconds(5)), "No request reached the engine.");

        // Both requests are in flight against the (un-gated) engine; release and confirm neither
        // was rejected.
        fake.Hold!.SetResult();
        var responses = await Task.WhenAll(aTask, bTask);
        Assert.All(responses, r => Assert.Equal(HttpStatusCode.OK, r.StatusCode));
    }

    [Fact]
    public async Task MaxConcurrent1_NonGenerationEndpoints_NotGated()
    {
        // /v1/models and /health must stay reachable even while a generation request holds the slot.
        var fake = new FakeInferenceEngine("test-model") { Hold = new TaskCompletionSource() };
        var client = CreateClient(fake, maxConcurrent: 1);

        var aTask = client.PostAsJsonAsync("/v1/chat/completions", ChatRequest());
        Assert.True(fake.Entered.Wait(TimeSpan.FromSeconds(5)), "Request A never reached the engine.");

        var models = await client.GetAsync("/v1/models");
        Assert.Equal(HttpStatusCode.OK, models.StatusCode);
        var health = await client.GetAsync("/health");
        Assert.Equal(HttpStatusCode.OK, health.StatusCode);

        fake.Hold!.SetResult();
        Assert.Equal(HttpStatusCode.OK, (await aTask).StatusCode);
    }

    [Fact]
    public async Task BoundedQueue_RejectsOnlyAfterActiveAndWaitingCapacityIsConsumed()
    {
        // 2 active batch slots + 3 waiting slots. The fake holds every admitted request so
        // all five occupy the outer admission gate; the sixth must fail before it reaches it.
        var fake = new FakeInferenceEngine("test-model") { Hold = new TaskCompletionSource() };
        var client = CreateClient(fake, maxConcurrent: null, maxQueued: 3, maxBatch: 2);
        var admitted = Enumerable.Range(0, 5)
            .Select(_ => client.PostAsJsonAsync("/v1/chat/completions", ChatRequest()))
            .ToArray();

        // Wait for ALL five to reach the engine, not just the first. Waiting on `Entered` was a
        // race: it signals on the first request, while the other four can still be in the HTTP
        // pipeline holding no gate slot. The sixth request then found free capacity, was admitted,
        // and blocked on Hold until HttpClient's 100s timeout — surfacing as a TaskCanceledException
        // that pointed nowhere near the real cause and stretched this suite to ~100s.
        var deadline = Environment.TickCount64 + 30_000;
        while (fake.EnteredCount < 5 && Environment.TickCount64 < deadline)
            await Task.Delay(10);
        Assert.Equal(5, fake.EnteredCount);

        var rejected = await client.PostAsJsonAsync("/v1/chat/completions", ChatRequest());
        Assert.Equal(HttpStatusCode.TooManyRequests, rejected.StatusCode);
        Assert.Contains("max 5 active or waiting", await rejected.Content.ReadAsStringAsync());

        fake.Hold!.SetResult();
        var responses = await Task.WhenAll(admitted);
        Assert.All(responses, response => Assert.Equal(HttpStatusCode.OK, response.StatusCode));
    }

    /// <summary>Session runtime that reports unavailable — its handler answers 409 immediately, which
    /// is what makes this test discriminating: reaching the handler at all proves the route is
    /// ungated.</summary>
    private sealed class UnavailableSessions : IServerSessionRuntime
    {
        public HotSessionRuntime? Runtime => null;
        public ColdSessionRuntime? ColdRuntime => null;
        public string? UnavailabilityReason => "test lane does not support sessions.";
    }

    /// <summary>
    /// The named-session turns route generates, so it must sit behind the same bounded-admission
    /// gate as the stateless chat routes. It previously did not, which meant MaxQueuedRequests
    /// bounded the three chat routes while any number of sessions could enqueue prompts alongside
    /// them.
    ///
    /// <para>409-vs-429 is the discriminator: this session runtime reports unavailable, so the
    /// handler answers 409 the moment it is reached. A 429 can therefore only come from the
    /// admission filter rejecting the request before the handler runs.</para>
    /// </summary>
    [Fact]
    public async Task SessionTurn_IsSubjectToTheSameAdmissionGateAsChat()
    {
        var fake = new FakeInferenceEngine("test-model") { Hold = new TaskCompletionSource() };
        var client = new WebApplicationFactory<Program>()
            .WithWebHostBuilder(b => b.ConfigureServices(s =>
            {
                s.Configure<OpenTailStingrayServerOptions>(o =>
                {
                    o.MaxConcurrentRequests = 1;
                    o.MaxQueuedRequests = 0;
                    o.MaxBatchSize = 1;
                });
                s.AddSingleton<IInferenceEngine>(fake);
                s.AddSingleton<IServerSessionRuntime>(new UnavailableSessions());
            }))
            .CreateClient();

        // Hold the only admission slot with an ordinary chat request.
        var held = client.PostAsJsonAsync("/v1/chat/completions", ChatRequest());
        Assert.True(fake.Entered.Wait(TimeSpan.FromSeconds(5)), "The holding request never reached the engine.");

        var turn = await client.PostAsJsonAsync($"/v1/sessions/{Guid.NewGuid()}/turns",
            new { append_prompt = "hi", expected_revision = 0L, max_tokens = 4 });

        Assert.Equal(HttpStatusCode.TooManyRequests, turn.StatusCode);
        Assert.True(turn.Headers.Contains("Retry-After"));

        // Releasing the slot lets a session turn through again — to the handler's own 409, proving
        // the gate released rather than wedging the route shut.
        fake.Hold!.SetResult();
        Assert.Equal(HttpStatusCode.OK, (await held).StatusCode);

        var afterRelease = await client.PostAsJsonAsync($"/v1/sessions/{Guid.NewGuid()}/turns",
            new { append_prompt = "hi", expected_revision = 0L, max_tokens = 4 });
        Assert.Equal(HttpStatusCode.Conflict, afterRelease.StatusCode);
    }
}
