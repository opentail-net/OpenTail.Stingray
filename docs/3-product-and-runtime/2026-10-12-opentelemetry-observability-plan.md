# OpenTelemetry Observability for Stingray

**Status:** Draft plan (2026-10-12). Not started. The owner has more input to add; the open questions are in §11 and nothing in Phases 1 onward should begin until Phase 0 settles them.  
**Target:** .NET 10 / C# 14, NativeAOT-safe, local-first.  
**Scope:** metrics and traces for the generation serving path (Server) and the inference engine, with an optional, off-by-default way to capture content. Audio, diffusion and vision pipelines are a later phase (§6, Phase 6).

## 1. Objective

Let an operator, or a developer embedding the library, answer these questions from standard OpenTelemetry data, without reading logs or attaching a profiler:

1. How fast is it? Time to first token, time per output token, tokens per second for prefill and for decode, per model and per backend (CPU, Vulkan, CUDA).
2. Where did the time go for one slow request: waiting in the queue, rendering the template, prefill, decode, or loading a model?
3. How loaded is it? Active requests, queue depth, rejections, resident models, evictions, KV-cache memory.
4. What is it costing? Input and output tokens per model and endpoint.
5. Is it failing, and how? Error types by endpoint and model.
6. Can benchmark tooling read structured numbers from the engine instead of scraping printed text? Telemetry as a benchmarking interface, so Performance League runs, backend comparisons and regression checks all draw on the same instrumentation.

The same instrumentation serves three audiences: a developer diagnosing a failure on one PC, anyone measuring inference performance (model load, prefill, decode, backend), and an operator exporting standard telemetry from a hosted server.

**Telemetry must not weaken the product promise that prompts stay in your process.** Content is never recorded unless the operator turns it on explicitly.

## 2. Repository facts (checked at HEAD, 2026-10-12)

- `src/OpenTail.Stingray.Server/ServerMetrics.cs` already defines `Meter("OpenTail.Stingray.Server", "1.0.0")` with these instruments: `opentailllm.requests`, `opentailllm.tokens`, `opentailllm.overload_rejections` (counters), `opentailllm.ttft_ms` and `opentailllm.itl_ms` (histograms). It also keeps its own bounded `LatencyHistogram` and renders a hand-written Prometheus page for `/metrics`.
- `ServingRequestTiming` is endpoint-owned: `BeginServingRequest()` starts after request parsing and template rendering, `MarkAdmitted()` and `MarkFirstToken()` are called from the OpenAI and Anthropic endpoints, and `RecordTokens()` is called when a response completes.
- `Endpoints/ResponseTimingExtension.cs` already returns `time_to_first_token_ms` and `total_ms` to a client that sets `opentail_timing: true`. That is a per-response answer, not an aggregate one.
- `Endpoints/ConcurrencyLimit.cs` holds `RequestConcurrencyGate` (a `SemaphoreSlim`); `ModelRuntimeManager.cs` owns model loading, residency and eviction, with counters for loads, evictions and admission rejections, and a `Snapshot()` of per-model stats.
- `src/OpenTail.Stingray.Engine/InferenceEngine.cs` is the generation entry point (`GenerateAsync`, `GenerateChunksAsync`, `GenerateImageChunksAsync`). The engine, Core and Sessions libraries expose no `Meter` or `ActivitySource`; roughly 68 files use `Stopwatch` for internal timing only. `result.Timing` (with `PrefillTokensPerSecond` and `DecodeTokensPerSecond`) is used by the golden-verification path.
- `stingray status` (`Cli/StatusCommand.cs`) reads `/health` and `/metrics` from a running server. Any change to the Prometheus output must keep it working.
- `Core/EngineEvents.cs` is a static, in-process, reflection-free event bus (events include model loaded, prefill completed, decode completed, MTP step and GC stats). Only the CLI raises them (`Cli/Program.cs`, `RunCommand.cs`); the server path never does.
- `Engine/InferenceEngine.cs` already measures encode, prefill, time to first token, decode tokens and reused tokens per request, but only writes them in a `LogDebug` call (around line 1407). `GenerateChunk` carries only `PromptTokens` and truncation flags, so none of this reaches the endpoints.
- Generation runs on a dedicated engine thread fed by a `BlockingCollection<Action>` (`InferenceEngine.cs`). The caller's execution context is not captured, so `Activity.Current` from the HTTP request does not reach prefill and decode.
- The CLI's AOT publish build (conditional on publishing) sets `MetricsSupport=false`, `EventSourceSupport=false` and `HttpActivityPropagationSupport=false`, so `Meter` instruments are switched off in the published CLI unless that changes.
- `Server/Endpoints/HealthEndpoints.cs` writes about half of its `/metrics` names with a hyphen (`opentail-llm_requests_total`), which is not a valid Prometheus metric name; the rest use `opentail_llm_`, and the Meter uses a third spelling, `opentailllm.`.
- Reported by an independent review and **not yet verified**: only the OpenAI, Anthropic and Responses endpoints call `ServingRequestTiming` (image, audio, embedding and rerank endpoints record nothing); nothing outside `ModelRuntimeManager` calls its `GetStats()`; in multi-model mode `/health` and `/metrics` read only the DI singleton engine.
- Speculative decoding exists: `SpeculativeDecoder` and `MtpDecoder.cs` in the Engine, driven from the CLI (`--model-draft`, `--spec-lookahead`, `--draft-lookup`; greedy only). Its docs describe per-token acceptance, but no acceptance figure is exported.
- The Vulkan backend has an env-gated profiler (`STINGRAY_PROFILE_GPU_SPLIT=1` in `VulkanBackend.cs`) that totals submit-and-wait time, staging-copy time and dispatch count in static fields. It has no timestamp queries, so per-kernel GPU time is not available today.
- `Directory.Build.props` sets `TreatWarningsAsErrors`, `IsTrimmable` and `EnableTrimAnalyzer` (Server and Server.Host are exempt from trimming), so any instrumentation in the libraries must be trim- and AOT-clean or the build fails.
- `Engine/Planning/ExecutionPlanner.cs` already records structured decisions with reasons and warnings (for example the DSpark placement decision and its reason), and hybrid and layer-split execution exist (`HybridForwardPass`, `VulkanLayerSplitForwardPass`). These are a ready source for "which backend ran, and why".
- There is no observer or hook interface in the engine: nothing like an `IEngineObserver` exists to subscribe to.
- `ContinuousBatchingEngine` already keeps prefix-cache counters (hits, bytes used and budgeted, tokens shared across sessions), and `stingray status` already reports a prefix-cache hit rate. They are not exported as metrics.
- Two opt-in, env-gated phase profilers exist: `MoePhaseTiming` (`STINGRAY_MOE_PHASE_TIMING=1`, per-phase wall time of the CPU MoE FFN, about 28 call sites) and the Vulkan GPU-split profiler above. They print at exit; nothing structured comes out.
- The benchmark scripts (`scripts/bench-*.ps1`, feeding `PerformanceLeague.md`) obtain their figures by regex-matching the CLI's printed output (`Select-String`), which breaks whenever a line changes.
- No OpenTelemetry package is referenced anywhere (`Directory.Packages.props` has none). The CLI sets `<EventSourceSupport>false</EventSourceSupport>` for NativeAOT size, which switches off EventCounters tooling such as `dotnet-counters`. In-process `Meter` and `ActivitySource` listeners are unaffected.

## 3. Gaps this plan closes

| # | Gap | Evidence |
|---|-----|----------|
| G1 | No dimensions on any instrument, so nothing can be sliced by model, endpoint, backend or outcome | Every `Add`/`Record` call in `ServerMetrics.cs` is tag-free |
| G2 | `opentailllm.itl_ms` is described as inter-token latency but records each request's total generation duration (`RecordGenerationDuration`) | `ServerMetrics.cs`, `RecordGenerationDuration` |
| G3 | Legacy `opentailllm.*` names; no alignment with the OpenTelemetry GenAI conventions, so standard dashboards do not apply | instrument names |
| G4 | No traces | no `ActivitySource` in `src/` |
| G5 | The engine and library emit nothing, so CLI and embedding users get no data | §2 |
| G6 | Model residency, evictions and admission rejections are tracked but not exported | `ModelRuntimeManager` counters and `Snapshot()` |
| G7 | The engine computes the useful numbers (encode, prefill, reused tokens, decode) but only logs them at debug level; they never reach the endpoints | `InferenceEngine.cs` `LogDebug`; `GenerateChunk` fields |
| G8 | Trace context is lost on the engine thread, so spans cannot be parented naively | `BlockingCollection<Action>` engine thread |
| G9 | `/metrics` has invalid (hyphenated) Prometheus names and three spellings of the prefix | `HealthEndpoints.cs`, `ServerMetrics.cs` |
| G10 | `EngineEvents` already carries prefill, decode and MTP events but only the CLI raises them, with no request correlation | `Core/EngineEvents.cs` |
| G11 | The published CLI turns `Meter` support off | `Cli.csproj` publish properties |

## 4. Design principles

1. **No third-party dependency in the libraries.** Instrument with `System.Diagnostics.Metrics` and `System.Diagnostics.ActivitySource` (in-box, `System.Diagnostics.DiagnosticSource`). The host application adds the OpenTelemetry SDK and exporter. This keeps the "no bloat, no sidecar" claim true.
2. **Zero cost when nobody listens.** Check `Counter.Enabled` / `ActivitySource.HasListeners()` before doing any work beyond a timestamp.
3. **Record per request, never per token.** One histogram sample per request for TTFT and time per output token (a per-request aggregate, not a rolling per-token average); one decode span, not one span per token. Build tag sets once per resolved model and endpoint and reuse them: `TagList` avoids an array allocation, but integer values still box, so prefer pre-resolved string attributes over values built in the loop.
4. **Low-cardinality attributes only** (§7).
5. **Privacy by default.** No prompt or completion text on any metric or span unless explicitly enabled.
6. **Align with the OpenTelemetry GenAI semantic conventions** for names and attributes, pinned to a stated version, because those conventions are still marked as in development and have changed names before (Phase 0 decision D1).
7. **Do not break what exists.** `/metrics`, `stingray status` and `opentail_timing` keep working; old instrument names are kept for one release with a changelog note before removal.
8. **Telemetry never breaks inference.** Our own listeners and sinks catch and swallow their exceptions; exporter failures stay inside the SDK. A test confirms instrumentation is harmless with no listener attached.
9. **Define every throughput.** Prefill, decode and end-to-end throughput are different measurements and are named and documented as such (§5). Rates are derived from summed tokens and summed durations, never from an average of per-request rates.
10. **NativeAOT-safe.** No reflection-based exporters in the libraries. The CLI already publishes with `PublishAot` (see `docs/guides/native-aot-deployment.md`); verify by publishing it and running it with a listener attached. There is no AOT step in CI today, so Phase 1 adds one if that is wanted (open question D10).

## 5. Instrument and span catalogue (proposed)

Names follow the GenAI conventions where one exists; `opentail.*` is used for Stingray-specific measures. Final names are fixed in Phase 0.

**Metrics: Server**

| Instrument | Type, unit | Attributes | Notes |
|---|---|---|---|
| `gen_ai.server.request.duration` | histogram, s | operation, model, endpoint, `error.type` | Replaces the request-duration summary |
| `gen_ai.server.time_to_first_token` | histogram, s | operation, model, endpoint | From the existing `ServingRequestTiming` |
| `gen_ai.server.time_per_output_token` | histogram, s | operation, model | `(end - first token) / (output tokens - 1)`, one sample per request. Fixes G2 |
| `gen_ai.client.token.usage` | histogram, `{token}` | operation, model, `gen_ai.token.type` = input / output | The conventions define token usage only as this metric name; decision D2 |
| `opentail.requests.rejected` | counter | reason (`overload`), endpoint | Replaces `opentailllm.overload_rejections` |
| `opentail.requests.active` | up/down counter | endpoint | |
| `opentail.queue.wait` | histogram, s | model | From `MarkAdmitted()` |

**Metrics: Engine (library, so CLI and embedders benefit)**

| Instrument | Type, unit | Attributes |
|---|---|---|
| `opentail.engine.prefill.duration`, `opentail.engine.prefill.tokens` | histogram, s and `{token}` | model, backend | Prompt-processing time and input tokens. Prefill throughput = input tokens / prefill time |
| `opentail.engine.decode.duration`, `opentail.engine.decode.tokens` | histogram, s and `{token}` | model, backend | Generation time and output tokens. Decode throughput = output tokens / decode time |
| `opentail.engine.model.load.duration` | histogram, s | model, backend |
| `opentail.engine.kv_cache.bytes` | observable gauge, By | model, backend |
| `opentail.engine.batch.size`, `opentail.engine.sequences.active` | histogram / gauge | model, only where continuous batching exposes them |
| `opentail.engine.speculative.acceptance` | histogram, ratio 0 to 1 | model, mode (draft model, lookup, MTP) | Accepted over proposed draft tokens, one sample per request. Needed to size draft models. Confirm in Phase 2 where `SpeculativeDecoder` and `MtpDecoder` can report it |
| `opentail.engine.speculative.tokens_per_step` | histogram | model, mode | The realised speed-up per verify step |
| `opentail.engine.prefix_cache.hits`, `opentail.engine.prefix_cache.shared_tokens` | counters | model | From the existing `ContinuousBatchingEngine` counters; hit rate is derived, not exported |
| `opentail.engine.prefix_cache.estimated_bytes` | observable gauge, By | model | Engine-accounted, so named as an estimate (see below) |
| `opentail.vulkan.dispatches`, `opentail.vulkan.submit_wait`, `opentail.vulkan.staging_copy` | histogram per request (count, s, s) | model | Per-request aggregates from the existing GPU-split counters, made cheap enough to be always on. Not per-dispatch spans |

**Metrics: Residency (from `ModelRuntimeManager`)**: `opentail.models.resident` (observable gauge), `opentail.models.loads`, `opentail.models.evictions`, `opentail.models.admission_rejections` (counters or observable counters over the existing cumulative counts).

**Percentiles.** Distribution views (P50 and P95 of how fast a request generated) come from the `gen_ai.server.time_per_output_token` histogram; a high percentile of seconds per token is a low percentile of tokens per second. No separate per-request rate is exported.

**Throughput definitions.** Prefill throughput is input tokens over prefill time; decode throughput is output tokens over decode time; end-to-end throughput is output tokens over total operation time (it also depends on prompt length, queueing, sampling and load policy). Dashboards derive rates as `sum(tokens) / sum(duration)`. A per-request rate histogram is deliberately not exported.

**Execution decisions (from `ExecutionPlanner`).** Selected backend and device, device-initialisation outcome, fallback count with a bounded reason category, unsupported-architecture count, and for hybrid execution the share of layers on the GPU (selecting a GPU does not mean the whole model runs on it). Reason categories are a fixed enumeration, never free text.

**Reliability and memory.**
- Every operation records an outcome of `success`, `error` or `cancelled`; cancellation is not counted as failure. Each failure is counted once, at the outermost boundary, not again by nested operations.
- Memory instruments say what they are: engine-accounted weight and KV-cache sizes are named `*.estimated_bytes`; measured process memory comes from the runtime's own instrumentation. An estimate is never presented as resident memory, and an unmeasurable value is reported as absent, not invented.

**Opt-in phase profile (the time breakdown, without kernel spans).** A time breakdown such as "attention 42%, matmul 37%, KV cache 12%" is valuable, but a span per kernel is not the way to get it (see "Ideas considered and not adopted"). Instead, one explicit profiling switch, off by default, would consolidate the existing env-gated profilers (`MoePhaseTiming`, the Vulkan GPU split) and report **per-phase durations summed over a request** as `opentail.profile.phase.duration{phase}` and as a text report. Rules: the overhead of profiling is measured and stated; it is never on by default; on GPU only what is measurable is reported (there are no timestamp queries today, so submit-wait versus staging-copy is the finest honest split).

**Traces.** Sources `OpenTail.Stingray.Server` and `OpenTail.Stingray.Engine`. One server span per request (`chat <model>`), with child spans `queue.wait`, `template.render`, `tokenize`, `prefill` and `decode`, plus `model.load` when a cold acquisition happens, with children `gguf.parse`, `tensor.map` and `device.upload` so a slow load shows which stage is slow. Attributes: `gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.response.finish_reasons`, `opentail.backend`. W3C `traceparent` is honoured automatically by ASP.NET Core.

### Ideas considered and not adopted

| Idea | Why not |
|---|---|
| Kernel-level spans (per matmul, rope, attention, softmax, dispatch) as the way to get a time breakdown | Thousands of spans per token would swamp the engine and any backend, and measuring each kernel costs real time. The opt-in aggregate phase profile (§5) gives the same breakdown at a stated cost |
| A span per Vulkan dispatch | Thousands of spans per token would swamp both the engine and any backend. Per-request aggregates (§5) answer "GPU or synchronisation overhead?" at no real cost. Per-kernel GPU time needs timestamp queries, which are not wired up |
| A span or measurement per generated token | Breaks principle 3; the per-request aggregate carries the same information |
| A SQLite (or sqlite-vec) sink | Adds a native dependency and a schema to maintain; the JSON-lines sink and OTLP cover local use without either |
| Reporting a weight-size estimate as resident memory | It overstates what is known; estimates are named and labelled as estimates (§5) |
| A rolling per-token latency average | Misleading under batching and streaming; time per output token per request is simpler and comparable across runs |

## 6. Phases

**Phase 0: Decisions.** Settle §11. Pin the semantic-convention version. Write the instrument catalogue as a checked-in table that tests assert against. *Done when* every open question has an answer recorded in this file.

**Phase 1: Engine diagnostics foundation (the first milestone).** One static diagnostics class in the Engine (`ActivitySource` and `Meter`, reused instances) with the contract from Phase 0. Instrument model loading and one representative text-generation path (`GenerateChunksAsync`): load, prefill, decode, token counts, outcome, selected backend. Start with the data contract: `GenerateChunk` carries only `PromptTokens` and truncation flags today, so extend its usage or stop chunk (or add a metrics chunk) to carry encode, prefill and decode durations, decode tokens, reused tokens and stop reason, which the engine already computes. Record the MTP and speculative acceptance figures from the existing `EngineEvents` MTP step events. Feed `ModelRuntimeManager` counters into observable gauges (G6). Keep instrumentation at these shared boundaries, not duplicated across the CLI, server and library. Ship a runnable sample plus tests that listen with a plain `MeterListener` and `ActivityListener`, and again with an SDK-backed exporter, and show the library works with no listener. *Done when* a developer can answer "why was this inference slow, which backend did it use, and where did it fail?" with no extra instrumentation of their own, the NativeAOT and trim builds still pass with warnings as errors, and the library references no OpenTelemetry package.

**Phase 2: Fix and extend the server meter.** The Server consumes the engine measurements. Add attributes (G1); correct time per output token (G2); add the GenAI-named instruments beside the old ones (G3); add active-request and queue-wait instruments. Repair the `/metrics` names (G9) and choose one source of truth for histograms, with `/metrics` kept as a compatibility view or replaced by the OpenTelemetry Prometheus exporter (D14). Cover the image, audio, embedding and rerank endpoints, not only chat. Tests with `MeterListener` assert instrument names, units and attribute sets, and that no forbidden attribute appears. *Done when* the Server test suite passes and `/metrics` and `stingray status` are unchanged.

**Phase 3: Traces.** Add the span tree to the Server endpoints and the engine. Capture the request's `ActivityContext` when work is queued and pass it explicitly to the engine thread (G8); in `ContinuousBatchingEngine`, where one loop serves many requests, use span links rather than parents. *Done when* one chat request produces one connected trace with the expected children, verified by an `ActivityListener` test.

**Phase 4: Host wiring.** Wire OTLP in `Server.Host` from the standard `OTEL_*` environment variables, with a console exporter for local development. Decision D3 chooses between documentation snippets and a small optional package (for example `OpenTail.Stingray.OpenTelemetry`) that registers the meters and sources. Also add a **local diagnostics experience with no infrastructure**: an in-process listener over the same instruments that feeds `stingray status` and, if wanted (D12), a human-readable report of the last run (model, backend and the planner's reasons, prompt and generated tokens, load, prefill and decode times, outcome) and a support bundle for bug reports. The report states what is known and what is unknown: an unmeasurable value is shown as unknown, and an estimate is labelled as one. It needs no extra package and no native dependency, and one instrumentation source feeds the listener, OpenTelemetry and the report. *Done when* a fresh `Server.Host` run exports to an OTLP endpoint with no code changes.

**Phase 5: Documentation and a worked example.** A guide `docs/guides/observability.md`, plus a sample showing the free local .NET Aspire Dashboard as a viewer (it needs Docker; document the console exporter and Prometheus scrape of `/metrics` as no-Docker alternatives).

**Phase 6: Other pipelines (later).** Audio: real-time factor, and per engine the split between autoregressive token (codebook) generation and waveform decode. Diffusion: `opentail.diffusion.steps`, `opentail.diffusion.step.duration`, images generated, output size in bounded buckets. Speech: audio seconds in and out, transcription or synthesis duration, `opentail.speech.realtime_factor`. Vision: image decode and encoder duration, vision tokens produced. Embeddings: throughput. Each behind the same conventions.

**Phase 7: Benchmark bridge and phase profile (later).** Let the benchmark scripts and the Performance League read structured numbers (prefill and decode duration and tokens, backend, outcome) from the engine, for example as machine-readable output from a run, instead of regex-matching printed text. Add the opt-in phase profile from §5. Telemetry does not replace controlled benchmarking: warm-up, repeated runs and reporting the spread still follow the existing performance docs. Regression detection and automated comparisons are a downstream use of this data and are out of scope for this plan.

## 7. Cardinality and privacy rules

- **Allowed metric attributes:** operation, model id, backend, endpoint, `error.type`, HTTP status class. In multi-model mode the model id comes from the client, so it is used only when it matches a configured model and is otherwise recorded as `other`; the existing server code already avoids per-model labels for this reason.
- **Forbidden on any metric or span attribute:** prompt or completion text, tool arguments or results, the canonical history prefix, image or audio bytes, request, session or user identifiers, client-supplied `user` or `metadata` fields, file paths and raw model paths (a model path exposes user directories), and exception messages (record the exception type name only, because messages can echo prompt content).
- **Spans** may carry request-scoped identifiers but never content by default.
- **Model identity is configurable.** The default is the catalogue id or the architecture family. A file name or path is never used unless explicitly enabled, because it can disclose a user name, company directory or private project.
- **Correlating metrics with traces** uses exemplars where the SDK supports them, never trace or request ids as metric attributes.
- **Content capture** (prompts, completions) is opt-in through one explicit setting, off by default, documented with a warning, and recorded as span events rather than attributes.
- A unit test enumerates the attribute keys on every instrument and fails on anything outside the allow-list.

## 8. Performance budget and verification

- With **no listener attached**, decode throughput must be indistinguishable from today's, measured on an existing small model (for example SmolLM2-135M) with the same method used in the performance docs, repeated runs, reporting the spread.
- With **a listener attached** (console or OTLP), the budget is a stated percentage agreed in Phase 0, measured the same way.
- All timing uses `Stopwatch.GetTimestamp()`; no allocation per token.
- A NativeAOT publish of the CLI must still succeed and run with the instrumentation in the libraries (manual today; see D10).

## 9. Compatibility and deprecation

- Keep `opentailllm.*` instruments and the Prometheus output for one release alongside the new ones.
- `opentailllm.itl_ms` changes meaning (from total generation time to a true per-token figure); record this in `CHANGELOG.md` and in the release notes.
- Removal of the old names follows the project's normal release-notes policy.

## 10. Risks

| Risk | Mitigation |
|---|---|
| GenAI semantic conventions change again | Pin the version (D1); keep names in one internal constants file |
| Label explosion from model or endpoint names | Allow-list test (§7); model id comes from the catalogue or loaded-model name only |
| Overhead on the decode path | §8 budget, per-request recording, `Enabled` checks |
| Streaming responses that fail midway | Record `error.type` and partial token counts; finish spans in a `finally` |
| Trace context is lost across the engine thread (and shared across requests in the batching engine) | Explicit `ActivityContext` hand-off; links in the batching engine (Phase 3) |
| The published CLI has `MetricsSupport=false` | Decide D9 first; either enable it for the CLI build or keep CLI data on `EngineEvents` |
| The OpenTelemetry SDK and OTLP exporter are not confirmed NativeAOT-clean under warnings as errors | Spike before committing the CLI or Server.Host AOT builds to them (D10) |
| Several models resident at once make "current model" ambiguous | Attributes are set per request from the resolved model, not from global state |
| NativeAOT trimming of exporter code | Keep the SDK in the host only; publish and run the CLI with AOT before release (D10: add a CI step) |

## 11. Open questions for the owner

To be filled in as the owner adds input. Provisional questions:

- **D1:** Which GenAI semantic-convention version do we pin to, and do we accept renames when they change?
- **D2:** Token usage: use `gen_ai.client.token.usage` on the server, or a Stingray-named server-side metric?
- **D3:** Separate optional package for OpenTelemetry wiring, or documentation snippets only?
- **D4:** Which backends do you expect users to send data to (Aspire Dashboard, Grafana/Prometheus, Azure Monitor, Datadog)?
- **D5:** Is `/metrics` the long-term interface, or should OTLP replace it for `stingray status` as well?
- **D6:** Default-on or opt-in? Recommendation: instrumentation always present (free when unlistened), export only when configured.
- **D7:** Content capture: never, or opt-in for debugging? Recommendation in this plan: opt-in, off by default. An independent review recommended never enabling capture at all, which is the stricter reading of the "prompts stay in your process" promise; decide which.
- **D8:** Priority order: server serving path first, then engine, then audio, diffusion and vision?
- **D9:** Does the CLI (`stingray chat`, `speak`) get telemetry too, or only the server and library? The published CLI sets `MetricsSupport=false`, so Meter-based telemetry there needs that changed (and the AOT warnings checked); the alternative is to keep CLI data on the existing `EngineEvents` bus.
- **D10:** Add a NativeAOT publish-and-run step to CI? Today only a manual check exists.
- **D11:** Instrument directly with `ActivitySource` and `Meter` at request boundaries (recommended), or put a provider abstraction in front (for example an `IInferenceTelemetry` with an OpenTelemetry provider and a null provider)? For: tidy tests, a place to add other sinks later. Against: `ActivitySource` and `Meter` already are a dependency-free abstraction with listeners as the providers; an extra interface adds DI plumbing to a library that is also used without DI, and a method such as `RecordToken` would be raised per token, which principle 3 rules out. If chosen, the interface must be boundary-level only.
- **D12:** Do you want a local diagnostics report (for example `stingray diagnostics`, a last-run report, a support bundle) on top of the same instruments, or is `stingray status` plus the console exporter enough? A JSON-lines snapshot file as well?
- **D13:** Should the Performance League scripts move to structured engine output (Phase 7), and is machine-readable output from `stingray run` (or the metrics themselves) the right interface?
- **D14:** Should the hand-written `/metrics` page be kept as a compatibility view over the same instruments, or replaced by the OpenTelemetry Prometheus exporter? Either way its hyphenated names must be fixed.

## 12. Acceptance checklist

- [ ] Every instrument in §5 exists, with the documented name, unit and attribute set, asserted by tests.
- [ ] No instrument carries an attribute outside the §7 allow-list (test).
- [ ] Time per output token is a true per-token figure; the old `itl_ms` meaning is documented as changed.
- [ ] One chat request yields one connected trace with the expected child spans.
- [ ] The library and CLI emit data with no OpenTelemetry package referenced.
- [ ] `/metrics` and `stingray status` behave as before.
- [ ] Decode throughput with no listener is within noise of today's; the with-listener cost is within the agreed budget.
- [ ] A NativeAOT publish of the CLI succeeds and runs with a listener attached.
- [ ] `docs/guides/observability.md` and the worked example exist and were run on a clean machine.
