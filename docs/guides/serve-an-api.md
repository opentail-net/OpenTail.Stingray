# Guide: serve models over HTTP (OpenAI and Anthropic compatible)

**You get:** a local web server that existing OpenAI, Anthropic and Cohere clients can talk to by
changing only the base URL. Chat, embeddings, reranking, speech, transcription and image
generation use the same server.

**You need:** a model file (see the [chat guide](chat.md)) and the repository or the
`OpenTail.Stingray.Server` package.

## Start it from the repository

```bash
STINGRAY_MODEL=models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  dotnet run --project src/OpenTail.Stingray.Server.Host -c Release
```

On PowerShell set the variable first: `$env:STINGRAY_MODEL = "models/..."`.

The host listens on **`http://127.0.0.1:8080`** and answers one generation at a time by default.

Try it:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"Name three planets, comma-separated."}]}'
```

Add `"stream": true` for Server-Sent Events. Point an OpenAI client at `http://127.0.0.1:8080/v1`.
Clients that insist on an API key field can be given any value, because the server does not check
one (see "Before you expose it" below).

## Settings

Edit `appsettings.json` (or a git-ignored `appsettings.Local.json`) under `"OpenTail.Stingray"`, or
set an environment variable. Environment variables win.

| Setting | Environment variable | Meaning |
|---|---|---|
| `ModelPath` | `STINGRAY_MODEL` | The model file to serve |
| (image projector) | `STINGRAY_MMPROJ` | The `mmproj-*.gguf` for a vision model |
| `NGpuLayers` | `STINGRAY_N_GPU_LAYERS` | GPU layers (-1 is all, 0 is CPU) |
| `MaxBatchSize` | `STINGRAY_MAX_BATCH` | Requests generated together (default 1) |
| `MaxQueuedRequests` | `STINGRAY_MAX_QUEUE` | How many may wait; past that the server answers HTTP 429 (default 16) |

`stingray list-env` shows the `STINGRAY_*` variables your shell has set and flags any the build
does not read; the server host also warns at start-up about a mistyped one.

## Put it in your own ASP.NET app

```csharp
builder.Services.AddOpenTailStingray(builder.Configuration, o => o.ModelPath = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
var app = builder.Build();
app.MapOpenTailStingray();
app.Run("http://localhost:5080");
```

Runnable version: [samples/ChatServer](../../samples/ChatServer/Program.cs). Use the
`Microsoft.NET.Sdk.Web` SDK.

## What it serves

| Endpoint | For |
|---|---|
| `POST /v1/chat/completions` | Chat, streaming, tool calling, structured JSON output, vision |
| `POST /v1/completions` | Plain text completion |
| `POST /v1/messages` | Anthropic Messages API |
| `POST /v1/responses` | OpenAI Responses API |
| `POST /v1/embeddings`, `POST /v1/rerank` | [Search building blocks](search-embeddings-rerank.md) |
| `POST /v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/audio/translations` | [Speech](speech.md) |
| `POST /v1/images/generations`, `/edits`, `/variations` | [Images](images-and-video.md) |
| `GET /v1/models`, `/health`, `/metrics`, `/capabilities` | Discovery, liveness, Prometheus metrics, diagnostics |
| `/v1/sessions` | Opt-in named sessions that keep a conversation's state between calls |

The full table, with the formats each endpoint speaks, is in the
[Server README](../../src/OpenTail.Stingray.Server/README.md).

## Before you expose it

**The server has no authentication of its own.** That is deliberate: it only provides inference
routes. It listens on loopback only by default. Do not bind it to a LAN or public address without an
authenticated reverse proxy or your own authentication middleware in front. The queue limit above
(HTTP 429) protects it from unbounded waiting requests, and the Server README's "Deployment safety"
section is the authoritative guidance.

## Limits to know

- The server loads the model you configure. Serving several models with automatic loading and
  eviction is under construction ([00-current-work.md](../00-current-work.md), section 3).
- Image input over HTTP is narrower than the CLI's. For vision, use the
  [CLI](vision.md) until you have tested your model through the server.
- Sessions are an opt-in feature for CPU-dense GGUF models. Read the Server README before relying
  on them.
