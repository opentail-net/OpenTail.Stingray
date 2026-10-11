# ChatServer

> **Classification:** package consumer. It uses only the public `OpenTail.Stingray.Server` package API (the default build uses the source project; `scripts/verify-package-consumers.ps1` rebuilds it against the packed NuGet package).
> **Task:** an OpenAI-compatible HTTP endpoint inside your own ASP.NET Core app.

Eight lines of `Program.cs`: `AddOpenTailStingray(...)` registers the engine, `MapOpenTailStingray()` maps `/v1/chat/completions`, `/v1/models` and the other endpoints, and the app listens on `http://localhost:5080`.

## Prerequisites

- The .NET 10 SDK.
- One chat model in GGUF form. The quickest is the starter model: `stingray setup chat` (Qwen2.5 0.5B Instruct, 469 MB, Apache-2.0), which stores it under `%LOCALAPPDATA%\stingray\models` (Windows) or `~/.cache/stingray/models`. Any other GGUF works.

## Run

```bash
dotnet run --project samples/ChatServer -c Release -- <path-to-model.gguf>
```

The model path is the first argument (`args[0]`); there is no path-free mode in this sample. In another terminal:

```bash
curl localhost:5080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"Name three planets, comma-separated."}],"max_tokens":32,"temperature":0}'
```

## Expected output

A JSON chat completion. The run behind this README returned (abridged):

```json
{"object":"chat.completion","model":"qwen2","choices":[{"index":0,"message":{"role":"assistant","content":"Mercury, Venus, and Earth."},"finish_reason":"stop"}],"usage":{"prompt_tokens":36,"completion_tokens":8,"total_tokens":44}}
```

`GET /v1/models` lists the loaded model.

## Known limitations

- Single model, loaded from the first command-line argument; no authentication, TLS or rate limiting (it is a sample, not a deployment).
- The `model` field in a request is ignored; the response reports the architecture name (`qwen2`), not the file name.
- Other endpoints (audio, images, embeddings, Anthropic and Responses APIs) exist in the server package but this sample loads only a chat model, so it does not demonstrate them. See the [Server README](../../src/OpenTail.Stingray.Server/README.md).

## Verification

- **Run against real weights, 2026-10-11:** Qwen2.5 0.5B Instruct Q4_K_M from `stingray setup chat`; the request above returned the response shown. One request, one model; no load test.
- **Package contract:** builds against the packed `OpenTail.Stingray.Server` 1.0.8 package with no project references (`scripts/verify-package-consumers.ps1`, 2026-10-11).
