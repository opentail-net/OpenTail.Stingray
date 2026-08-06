# OpenTail.Stingray.Server

ASP.NET Core endpoints, options, and DI extensions that expose [OpenTail.Stingray](https://www.nuget.org/packages/OpenTail.Stingray) as a drop-in **OpenAI- and Anthropic-compatible HTTP API**. Bring your own host (Kestrel, IIS, YARP, …); this package only ships the routes, request/response shapes, and DI wiring.

For the bare inference library, use [`OpenTail.Stingray`](https://www.nuget.org/packages/OpenTail.Stingray). For the standalone CLI, use [`OpenTail.Stingray.Cli`](https://www.nuget.org/packages/OpenTail.Stingray.Cli).

## Install

```
dotnet add package OpenTail.Stingray.Server
```

This transitively pulls in `OpenTail.Stingray` (the bundled inference engine + CPU/Vulkan/CUDA backends). You must be on the `Microsoft.NET.Sdk.Web` SDK — the package's `Microsoft.AspNetCore.App` framework reference is propagated.

## Quick start

```csharp
var builder = WebApplication.CreateBuilder(args);

builder.Services.AddOpenTailStingray(opt =>
{
    opt.ModelPath = "models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf";
    opt.GpuLayers = -1; // -1 = all layers on GPU; 0 = pure CPU
});

var app = builder.Build();
app.MapOpenTailStingray();
app.Run();
```

## Deployment safety

`OpenTail.Stingray.Server` supplies inference routes; it deliberately does **not** choose an
authentication scheme for the host. The bundled demo host therefore listens only on
`127.0.0.1:8080` and admits one generation at a time by default.

Do not bind it directly to a LAN or public interface without an authentication boundary.
For remote use, place it behind an authenticated reverse proxy, or add your application's
authentication/authorization middleware and protect the mapped routes before calling `Run()`.
Set `OpenTail.Stingray:MaxQueuedRequests` (or `STINGRAY_MAX_QUEUE`) to bound how many requests
may wait behind the active batch. The server returns HTTP 429 once `MaxBatchSize + MaxQueuedRequests`
is reached, rather than retaining unbounded HTTP connections. The default is 16 waiting requests.
`MaxConcurrentRequests` (or `STINGRAY_MAX_CONCURRENT`) remains available as a legacy fixed
in-flight cap and takes precedence when explicitly set.

Bind from configuration instead:

```csharp
// appsettings.json: { "OpenTail.Stingray": { "ModelPath": "...", "GpuLayers": -1 } }
builder.Services.AddOpenTailStingray(builder.Configuration);
```

## What you get

| Endpoint                       | Wire-compatible with     |
|--------------------------------|--------------------------|
| `POST /v1/chat/completions`    | OpenAI Chat Completions  |
| `POST /v1/completions`         | OpenAI Completions       |
| `POST /v1/messages`            | Anthropic Messages       |
| `POST /v1/responses`           | OpenAI Responses         |
| `GET  /v1/models`              | OpenAI Models            |
| `GET  /health`, `/metrics`     | Liveness + Prometheus    |
| `GET  /capabilities`           | Read-only protocol/configuration diagnostics |

Streaming (SSE) is enabled for every chat/completion endpoint, and the JSON pipeline is wired through a source-generated `JsonSerializerContext` so the package is AOT-friendly even though the project itself is not AOT-published.

When routes are mapped through `MapOpenTailStingray()`, `GET /capabilities` is a versioned
diagnostics snapshot for local tooling and support. It reports the mapped OpenAI/Anthropic
routes plus compatibility-relevant effective settings (backend, batching, reasoning, tool
grammar, and image-input support), but deliberately excludes model paths, credentials,
prompts, generated text, and inference-loop state.

## Configuration

`OpenTailStingrayServerOptions` is the single options record (`Options` pattern, validated on first request):

```csharp
public sealed class OpenTailStingrayServerOptions
{
    public string  ModelPath      { get; set; } = "";
    public int     GpuLayers      { get; set; }       // -1 = all, 0 = CPU-only
    public int     MaxContext     { get; set; } = 4096;
    public string? Architecture   { get; set; }       // override GGUF detection
    public Func<IServiceProvider, LoadedEngine>? EngineFactory { get; set; } // tests
    // …
}
```

Override `EngineFactory` in tests to inject a fake `IInferenceEngine`; the rest of the DI graph (chat-template renderer, metrics, JSON context) stays intact.

For CPU-only deployments, `OpenTail.Stingray:CpuThreads` (or `STINGRAY_CPU_THREADS`) controls the SIMD-kernel worker count. Leave it at `0` to use all logical processors. If the host is shared with other CPU- or memory-bandwidth-heavy work, benchmark a smaller value — oversubscribing these memory-bound kernels can lower tokens/sec.

## Links

- [Repository & docs](https://github.com/opentail-net/OpenTail.Stingray)
- [Design document](https://github.com/opentail-net/OpenTail.Stingray/blob/master/docs/OpenTail.Stingray-Design.md)
- [Issues](https://github.com/opentail-net/OpenTail.Stingray/issues)

---

## Acknowledgements

Forked from **[SharpInference](https://github.com/pekkah/SharpInference)** by Pekka Heikura (MIT), which remains actively developed upstream; copyright is retained in `LICENSE` alongside ours.

Interoperates with **[llama.cpp](https://github.com/ggml-org/llama.cpp)**'s GGUF format and quantization block layouts, and follows `llama-cli` flag names where the meaning matches — **no llama.cpp code is used**. **[LLamaSharp](https://github.com/SciSharp/LLamaSharp)** was studied as the reference for .NET inference API design; **no LLamaSharp code is used**, and unlike it this engine is managed C# end to end rather than P/Invoke bindings to native llama.cpp.

## License

MIT. Copyright (c) 2026 Pekka Heikura.
