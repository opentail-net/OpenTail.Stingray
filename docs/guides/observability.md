# Observability

Stingray emits standard .NET `Meter` instruments and `ActivitySource` spans. The libraries
(`Engine`, `Server`) depend only on `System.Diagnostics`; **no OpenTelemetry package is referenced
by them**. `OpenTail.Stingray.Server.Host` adds the OpenTelemetry SDK and exports **only if you ask**.

Nothing is exported by default, and no prompt, completion, image, audio, tool content, file path or
client-supplied identifier is ever recorded.

Status: in development; names marked *standard* follow the OpenTelemetry GenAI conventions, which are
themselves still in "Development" upstream. See
[the plan](../3-product-and-runtime/2026-10-12-opentelemetry-observability-plan.md) for the mapping table and decisions.

## Turn it on (Server.Host)

| Variable | Effect |
|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` (or `_TRACES_` / `_METRICS_` variants) | export traces and metrics over OTLP |
| `OTEL_TRACES_EXPORTER=console`, `OTEL_METRICS_EXPORTER=console` | print to the console (local development, no infrastructure) |
| `OTEL_SERVICE_NAME` | service name (default `stingray`) |
| `OTEL_METRIC_EXPORT_INTERVAL` | metric export interval in ms (default **30000**; long on purpose, because on a CPU-bound run the exporter competes with the compute threads) |

```powershell
$env:OTEL_METRICS_EXPORTER = "console"; $env:OTEL_TRACES_EXPORTER = "console"
$env:STINGRAY_MODEL = "models\SmolLM2-135M-Instruct-Q4_K_M.gguf"
dotnet run --project src/OpenTail.Stingray.Server.Host -c Release
```

For a UI, point `OTEL_EXPORTER_OTLP_ENDPOINT` at any OTLP receiver (the .NET Aspire Dashboard needs Docker;
Prometheus can still scrape `/metrics` without it).

## Embedding the library

Add the sources to your own provider; the library needs nothing else:

```csharp
.AddMeter("OpenTail.Stingray.Engine", "OpenTail.Stingray.Server")
.AddSource("OpenTail.Stingray.Engine", "OpenTail.Stingray.Server")
```

## Metrics

Server meter `OpenTail.Stingray.Server` (attributes: `gen_ai.operation.name`, `gen_ai.request.model` (server-resolved id; path-like values become `other`), `opentail.endpoint`, `outcome` on request duration):

| Instrument | Unit | Meaning |
|---|---|---|
| `gen_ai.server.request.duration` | s | From the start of serving (after parsing and template rendering) to the response being complete. `outcome`: success, cancelled, incomplete, client_error, error |
| `gen_ai.server.time_to_first_token` | s | Start of serving to the first generated token (a cold model load happens *before* this timer starts) |
| `gen_ai.server.time_per_output_token` | s | Decode time over (output tokens − 1), one sample per request, delivery time excluded |
| `opentail.queue.wait` | s | Start of serving to engine admission |
| `opentail.requests.active` | {request} | Generation requests in flight |
| `opentail.requests.rejected` | {request} | Rejected before admission (`opentail.reason`=overload) |
| `opentailllm.*` | | Legacy instruments, kept for one release. `opentailllm.itl_ms` is mislabelled (it is total generation time) |

Engine meter `OpenTail.Stingray.Engine` (attributes: `outcome`, `opentail.forward_pass`; load also `opentail.architecture`):
`opentail.engine.requests`, `.prefill.duration` / `.prefill.tokens` (new, not reused, prompt tokens), `.decode.duration` / `.decode.tokens`, `.first_token.duration`, `.generation.duration`, `.model.load.duration`.

Throughputs are derived, not exported: prefill = `sum(prefill.tokens) / sum(prefill.duration)`; decode = `sum(decode.tokens) / sum(decode.duration)`.

## Traces

`POST /v1/chat/completions` (ASP.NET Core) → `chat <model>` (server operation span, same boundaries as `gen_ai.server.request.duration`) → `opentail.engine.generate` (engine span; the caller's context is handed across the engine thread explicitly). `model.load` is its own trace, never a child of the request that triggered the load.
Span tags carry counts, outcome and ids only.

## Per-response numbers without any backend

Send `"opentail_timing": true` on a non-streaming chat request to get `time_to_first_token_ms`, `total_ms` and, when the engine supplies them, `prompt_tokens`, `reused_prompt_tokens`, `generated_tokens`, `encode_ms`, `prefill_ms`, `decode_ms`.

## `/metrics` (Prometheus text)

Unchanged in purpose. Note the lifetime names changed from `opentail-llm_*` to `opentail_llm_*` (a hyphen is not valid in a Prometheus name); `stingray status` reads both.

## Known gaps

`ContinuousBatchingEngine` does not yet supply per-request stats; queue/prefill/decode child spans, span links for requests that waited on a shared load, speculative-decoding, prefix-cache and residency metrics, the NativeAOT CLI path and the real-model overhead A/B are not done. The CLI `run` command has its own decode loop and is not instrumented by the engine meter.
