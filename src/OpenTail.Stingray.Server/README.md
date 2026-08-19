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

## Complete API Surface

| Endpoint | Wire-compatible with | Capabilities |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions | Streaming SSE, Tool Calling, Structured JSON, Multimodal Vision |
| `POST /v1/completions` | OpenAI Completions | Text completion |
| `POST /v1/audio/transcriptions` | OpenAI Audio Transcriptions | Whisper STT with SRT, VTT, verbose_json |
| `POST /v1/audio/translations` | OpenAI Audio Translations | Whisper Speech Translation |
| `POST /v1/audio/speech` | OpenAI Audio Speech | Kokoro, Piper, MeloTTS, Chatterbox, F5-TTS with clause streaming |
| `POST /v1/images/generations` | OpenAI Image Generations | SD 1.5, SDXL, SD 3/3.5, FLUX, Z-Image with Base64 / URL outputs |
| `POST /v1/images/edits` | OpenAI Image Edits | Img2Img and Inpainting modifications |
| `POST /v1/images/variations` | OpenAI Image Variations | Image variation generation |
| `POST /v1/embeddings` | OpenAI Embeddings | Dense text embeddings |
| `POST /v1/rerank` | Cohere Rerank | Cross-encoder sequence reranking |
| `POST /v1/messages` | Anthropic Messages | Anthropic Messages API parity |
| `POST /v1/responses` | OpenAI Responses | OpenAI Responses endpoint |
| `GET  /v1/models` | OpenAI Models | Model enumeration |
| `GET  /health`, `/metrics` | Observability | Liveness + Prometheus metrics |
| `GET  /capabilities` | Diagnostics | Read-only protocol/configuration diagnostics |
| `POST/GET/DELETE /v1/sessions` | Sessions | Opt-in CPU-dense GGUF named-session lifecycle |
| `GET /v1/sessions/{id}/operations/{operationId}` | Operations | Bounded idempotent-result reconnect lookup |

Streaming (SSE) is enabled for every chat/completion endpoint, and the JSON pipeline is wired through a source-generated `JsonSerializerContext` so the package is AOT-friendly even though the project itself is not AOT-published.
