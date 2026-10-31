using System.Net;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Server;
using Xunit;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// Plan 2 P8: a client that disconnects mid-stream stops generation, frees the admission slot and the
/// admission slot, and leaves the server able to serve the next request. The engine is a token-by-token
/// fake, so the disconnect lands during generation, not before or after it.
///
/// <para>Single-model mode only. The chat endpoint acquires a <c>ModelRuntimeHandle</c> only in multi-model mode
/// (OpenAiEndpoints.cs, "if (inferenceService.IsMultiModel)"), and that path is NOT covered here: it needs a
/// fake loader, which is a separate follow-up. So this test proves the slot and the engine, not the handle.</para>
///
/// <para>Every wait is bounded. A missing cancellation shows up as an assertion failure, not a hang.</para>
/// </summary>
public sealed class ClientDisconnectTests : IDisposable
{
    private static readonly TimeSpan Bound = TimeSpan.FromSeconds(10);
    private const string Model = "test-model";

    private readonly List<WebApplicationFactory<Program>> _factories = new();
    private readonly CapturingLoggerProvider _logs = new();

    public void Dispose()
    {
        foreach (var factory in _factories) factory.Dispose();
    }

    private (HttpClient Client, IServiceProvider Services) CreateServer(TokenByTokenEngine engine)
    {
        var factory = new WebApplicationFactory<Program>()
            .WithWebHostBuilder(b => b.ConfigureServices(s =>
            {
                s.Configure<OpenTailStingrayServerOptions>(o =>
                {
                    o.MaxConcurrentRequests = 1;
                    o.MaxQueuedRequests = 0;
                    o.MaxBatchSize = 1;
                });
                s.AddSingleton<IInferenceEngine>(engine);
                s.AddSingleton<ILoggerProvider>(_logs);
            }));
        _factories.Add(factory);
        return (factory.CreateClient(), factory.Services);
    }

    private static object ChatRequest(bool stream) => new
    {
        model = Model,
        messages = new[] { new { role = "user", content = "hi" } },
        max_tokens = 5000,
        stream,
    };

    [Fact]
    public async Task ClientDisconnectMidStream_StopsGeneration_FreesTheSlot_AndTheNextRequestSucceeds()
    {
        var engine = new TokenByTokenEngine(Model, tokens: 10_000, perTokenDelay: TimeSpan.FromMilliseconds(10));
        var (client, services) = CreateServer(engine);

        // 1. Start a streaming generation and read the first SSE line, so generation is provably in flight.
        using var disconnect = new CancellationTokenSource();
        using var deadline = new CancellationTokenSource(Bound * 3);
        using var open = CancellationTokenSource.CreateLinkedTokenSource(disconnect.Token, deadline.Token);
        using var request = new HttpRequestMessage(HttpMethod.Post, "/v1/chat/completions")
        {
            Content = JsonContent.Create(ChatRequest(stream: true)),
        };
        using HttpResponseMessage response = await client.SendAsync(
            request, HttpCompletionOption.ResponseHeadersRead, open.Token);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        using var body = await response.Content.ReadAsStreamAsync(open.Token);
        using var reader = new StreamReader(body);
        string? firstLine = await reader.ReadLineAsync(open.Token);
        Assert.NotNull(firstLine);

        // 2. Positive controls: the engine has started, tokens are flowing, and the slot is held.
        Assert.True(engine.Started.Wait(Bound), "the engine never started generating");
        await WaitUntilAsync(() => engine.TokensYielded >= 2, "generation did not produce tokens");
        Assert.Equal(1, engine.ActiveRequests);

        // 3. Disconnect. Cancel the client request and drop the response.
        disconnect.Cancel();
        response.Dispose();

        // 4. Cancellation reaches the engine's active operation.
        await WaitUntilAsync(() => engine.CancellationObserved, "cancellation never reached the engine");

        // 5. No further generation after the cancel: the count is frozen.
        int atCancel = engine.TokensYielded;
        await Task.Delay(TimeSpan.FromMilliseconds(300));
        Assert.Equal(atCancel, engine.TokensYielded);
        Assert.Equal(0, engine.ActiveRequests);

        // 6. The admission slot is released: with MaxConcurrentRequests = 1, a leaked slot would answer 429.
        //    The slot is returned when the handler returns, which can be a moment after the stream stops, so retry briefly.
        HttpResponseMessage? next = null;
        await WaitUntilAsync(async () =>
        {
            next?.Dispose();
            next = await client.PostAsJsonAsync("/v1/chat/completions", ChatRequest(stream: false), deadline.Token);
            return next.StatusCode == HttpStatusCode.OK;
        }, "the admission slot was not released: the follow-up request kept getting a non-200 response");
        Assert.NotNull(next);
        next!.Dispose();

        // 7. No generation failure was logged. A cancellation is not a failure.
        Assert.DoesNotContain(_logs.Messages, m => m.Contains("Generation failed", StringComparison.Ordinal));
    }

    private static async Task WaitUntilAsync(Func<bool> condition, string failure)
    {
        var deadline = DateTime.UtcNow + Bound;
        while (!condition())
        {
            if (DateTime.UtcNow > deadline) Assert.Fail(failure);
            await Task.Delay(TimeSpan.FromMilliseconds(20));
        }
    }

    private static async Task WaitUntilAsync(Func<Task<bool>> condition, string failure)
    {
        var deadline = DateTime.UtcNow + Bound;
        while (!await condition())
        {
            if (DateTime.UtcNow > deadline) Assert.Fail(failure);
            await Task.Delay(TimeSpan.FromMilliseconds(20));
        }
    }
}

/// <summary>
/// Streams <c>tokens</c> tokens with a delay between each. Every delay is cancellable by the caller's
/// token, so a cancelled request stops here rather than running to the end.
/// </summary>
internal sealed class TokenByTokenEngine : IInferenceEngine
{
    private readonly int _tokens;
    private readonly TimeSpan _perTokenDelay;
    private int _active;
    private int _yielded;
    private int _calls;
    private volatile bool _cancellationObserved;

    public TokenByTokenEngine(string modelId, int tokens, TimeSpan perTokenDelay)
    {
        ModelId = modelId;
        _tokens = tokens;
        _perTokenDelay = perTokenDelay;
    }

    public ManualResetEventSlim Started { get; } = new(false);

    /// <summary>Tokens produced so far, across all calls.</summary>
    public int TokensYielded => Volatile.Read(ref _yielded);

    /// <summary>True once a generation call saw its cancellation token fire.</summary>
    public bool CancellationObserved => _cancellationObserved;

    public string ModelId { get; }
    public int QueueDepth => 0;
    public int ActiveRequests => Volatile.Read(ref _active);
    public bool PrefixCacheEnabled => false;
    public long PrefillTokensReused => 0;

    public async IAsyncEnumerable<GenerateChunk> GenerateChunksAsync(
        string prompt,
        SamplingParams sp,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default,
        string? canonicalHistoryPrefix = null)
    {
        // Only the first call is long and slow. Later calls (the follow-up request) are short, so the test
        // does not wait out a 10,000-token generation.
        int call = Interlocked.Increment(ref _calls);
        int tokens = call == 1 ? _tokens : 3;
        TimeSpan delay = call == 1 ? _perTokenDelay : TimeSpan.Zero;

        Interlocked.Increment(ref _active);
        Started.Set();
        try
        {
            for (int i = 0; i < tokens; i++)
            {
                try
                {
                    await Task.Delay(delay, ct);
                }
                catch (OperationCanceledException) when (ct.IsCancellationRequested)
                {
                    _cancellationObserved = true;
                    throw;   // real engines throw OperationCanceledException; the endpoint must not log it
                }

                Interlocked.Increment(ref _yielded);
                yield return new GenerateChunk(GenerateChunkKind.Text, "t");
            }
        }
        finally
        {
            Interlocked.Decrement(ref _active);
        }
    }

    public async IAsyncEnumerable<GenerateChunk> GenerateImageChunksAsync(
        string prompt,
        IReadOnlyList<byte[]> images,
        SamplingParams sp,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default,
        string? canonicalHistoryPrefix = null)
    {
        await Task.CompletedTask;
        yield break;
    }

    public async IAsyncEnumerable<string> InferAsync(
        string prompt,
        IInferenceParams? inferenceParams = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        yield break;
    }

    public async IAsyncEnumerable<GenerateChunk> InferChunksAsync(
        string prompt,
        IInferenceParams? inferenceParams = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        yield break;
    }

    public void Dispose() { }
}

/// <summary>Keeps every log message for assertion. Logs go to memory only.</summary>
internal sealed class CapturingLoggerProvider : ILoggerProvider
{
    private readonly List<string> _messages = new();

    public IReadOnlyList<string> Messages
    {
        get { lock (_messages) return _messages.ToArray(); }
    }

    public ILogger CreateLogger(string categoryName) => new Logger(this);

    public void Dispose() { }

    private sealed class Logger(CapturingLoggerProvider owner) : ILogger
    {
        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception,
            Func<TState, Exception?, string> formatter)
        {
            string text = formatter(state, exception);
            if (exception is not null) text += " " + exception.GetType().Name;
            lock (owner._messages) owner._messages.Add(text);
        }
    }
}
