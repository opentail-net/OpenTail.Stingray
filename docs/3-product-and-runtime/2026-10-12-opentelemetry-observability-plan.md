# OpenTelemetry Observability for Stingray

**Status:** Draft plan (2026-10-12), revised after review by three independent reviewers and a fresh read of the code. Not started. Phase 1 is deliberately small (one vertical slice); everything else follows it. Every open question in §11 now carries a proposed answer, and none is decided until the owner confirms.  
**Target:** .NET 10 / C# 14, NativeAOT-safe, local-first.  
**Scope:** metrics and traces for the generation serving path (Server) and the inference engine, with an optional, off-by-default way to capture content. Audio, diffusion and vision pipelines are a later phase (§6, Phase 7).

## 1. Objective

Let an operator, or a developer embedding the library, answer these questions from standard OpenTelemetry data, without reading logs or attaching a profiler:

1. How fast is it? Time to first token, time per output token, tokens per second for prefill and for decode, per model and per backend (CPU, Vulkan, CUDA).
2. Where did the time go for one slow request: waiting in the queue, rendering the template, prefill, decode, or loading a model?
3. How loaded is it? Active requests, queue depth, rejections, resident models, evictions, KV-cache memory.
4. What is it costing? Input and output tokens per model and endpoint.
5. Is it failing, and how? Error types by endpoint and model.
6. Can benchmark tooling read structured numbers from the engine instead of scraping printed text? Telemetry as a benchmarking interface, so Performance League runs, backend comparisons and regression checks all draw on the same instrumentation.

The same instrumentation serves three audiences: a developer diagnosing a failure on one PC, anyone measuring inference performance (model load, prefill, decode, backend), and an operator exporting standard telemetry from a hosted server.

**Telemetry must not weaken the product promise that prompts stay in your process.** No prompt, completion, image, audio or tool content is ever recorded in OpenTelemetry data (§7).

## 2. Repository facts (checked at HEAD, 2026-10-12)

- `src/OpenTail.Stingray.Server/ServerMetrics.cs` already defines `Meter("OpenTail.Stingray.Server", "1.0.0")` with these instruments: `opentailllm.requests`, `opentailllm.tokens`, `opentailllm.overload_rejections` (counters), `opentailllm.ttft_ms` and `opentailllm.itl_ms` (histograms). It also keeps its own bounded `LatencyHistogram` and renders a hand-written Prometheus page for `/metrics`.
- `ServingRequestTiming` is endpoint-owned: `BeginServingRequest()` starts after request parsing and template rendering, `MarkAdmitted()` and `MarkFirstToken()` are called from the OpenAI and Anthropic endpoints, and `RecordTokens()` is called when a response completes.
- `Endpoints/ResponseTimingExtension.cs` already returns `time_to_first_token_ms` and `total_ms` to a client that sets `opentail_timing: true`. That is a per-response answer, not an aggregate one.
- `Endpoints/ConcurrencyLimit.cs` holds `RequestConcurrencyGate` (a `SemaphoreSlim`); `ModelRuntimeManager.cs` owns model loading, residency and eviction, with counters for loads, evictions and admission rejections, and a `Snapshot()` of per-model stats.
- `src/OpenTail.Stingray.Engine/InferenceEngine.cs` is the generation entry point (`GenerateAsync`, `GenerateChunksAsync`, `GenerateImageChunksAsync`). The engine, Core and Sessions libraries expose no `Meter` or `ActivitySource`; roughly 68 files use `Stopwatch` for internal timing only. `result.Timing` (with `PrefillTokensPerSecond` and `DecodeTokensPerSecond`) is used by the golden-verification path.
- `stingray status` (`Cli/StatusCommand.cs`) reads `/health` and `/metrics` from a running server. Any change to the Prometheus output must keep it working.
- `Core/EngineEvents.cs` is a static, in-process, reflection-free event bus with an `IEngineEventSink` interface (`OnEvent(in EngineEvent e)`; the event is a plain struct with `Kind`, `Name`, `A`, `B`, `C`, `Ms` and `Detail`). With no subscriber `Emit` is one volatile read; a throwing sink is swallowed. Events: `ModelLoaded` (architecture, build time, forward-pass kind), `PrefillCompleted` (prompt tokens, wall time), `DecodeCompleted` (tokens generated and visible, wall time), `MtpStep` (position, drafts proposed, drafts accepted) and `GcStats`. Limits that matter: **sinks run synchronously on the emitting thread; an event carries no request identity; and only the CLI emits them** (`Cli/Program.cs`, `RunCommand.cs`), never the server path.
- `Engine/InferenceEngine.cs` already measures encode, prefill, time to first token, decode tokens and reused tokens per request, but only writes them in a `LogDebug` call (around line 1407). `GenerateChunk` carries only `PromptTokens` and truncation flags, so none of this reaches the endpoints.
- Generation runs on a dedicated engine thread fed by a `BlockingCollection<Action>` (`InferenceEngine.cs`). The caller's execution context is not captured, so `Activity.Current` from the HTTP request does not reach prefill and decode.
- The CLI's AOT publish build (conditional on publishing) sets `MetricsSupport=false`, `EventSourceSupport=false` and `HttpActivityPropagationSupport=false`, so `Meter` instruments are switched off in the published CLI unless that changes.
- `Server/Endpoints/HealthEndpoints.cs` writes about half of its `/metrics` names with a hyphen (`opentail-llm_requests_total`), which is not a valid Prometheus metric name; the rest use `opentail_llm_`, and the Meter uses a third spelling, `opentailllm.`.
- Reported by an independent review and **not yet verified**: only the OpenAI, Anthropic and Responses endpoints call `ServingRequestTiming` (image, audio, embedding and rerank endpoints record nothing); nothing outside `ModelRuntimeManager` calls its `GetStats()`; in multi-model mode `/health` and `/metrics` read only the DI singleton engine.
- Speculative decoding exists: `SpeculativeDecoder` and `MtpDecoder.cs` in the Engine, driven from the CLI (`--model-draft`, `--spec-lookahead`, `--draft-lookup`; greedy only). Its docs describe per-token acceptance, but no acceptance figure is exported.
- The Vulkan backend has an env-gated profiler (`STINGRAY_PROFILE_GPU_SPLIT=1` in `VulkanBackend.cs`) that totals submit-and-wait time, staging-copy time and dispatch count in static fields. It has no timestamp queries, so per-kernel GPU time is not available today.
- `Directory.Build.props` sets `TreatWarningsAsErrors`, `IsTrimmable` and `EnableTrimAnalyzer` (Server and Server.Host are exempt from trimming), so any instrumentation in the libraries must be trim- and AOT-clean or the build fails.
- `Engine/Planning/ExecutionPlanner.cs` already records structured decisions with reasons and warnings (for example the DSpark placement decision and its reason), and hybrid and layer-split execution exist (`HybridForwardPass`, `VulkanLayerSplitForwardPass`). These are a ready source for "which backend ran, and why".
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
2. **Negligible, measured cost.** An instrumentation call is not literally free just because nobody listens, so the claim is made measurable (§8): no tag construction, string building or allocation on the hot path when unobserved (check `Counter.Enabled` / `ActivitySource.HasListeners()` first), no per-token spans, and an agreed, tested overhead threshold with and without a listener.
3. **Per-request records and per-iteration samples; never per-token spans or allocations.** Request-level measurements (time to first token, time per output token, durations, token counts) are one sample per request, and a request is one trace with one decode span. Step-level measurements that serving engines such as vLLM export (an inter-token latency or tokens-per-iteration histogram) are allowed as one histogram sample per decode **iteration**, not per token per sequence, guarded by `Instrument.Enabled` and held to the §8 overhead budget; in continuous batching one iteration serves many sequences, so this is cheap. Build tag sets once per resolved model and endpoint and reuse them: `TagList` avoids an array allocation, but integer values still box, so prefer pre-resolved string attributes over values built in the loop.
4. **Low-cardinality attributes only** (§7).
5. **Privacy by contract.** No prompt or completion content on any metric or span, with no debugging flag to enable it (§7).
6. **Use the OpenTelemetry GenAI conventions where a name fits and has been verified, and keep Stingray's own contract separate.** The conventions have moved to their own repository, have no tagged release and mark every metric as in development (§5), so every instrument is recorded in a checked-in mapping table that states what Stingray measures, the exported name and unit, whether the name is standard or Stingray-specific, the convention commit it was checked against, and any deliberate deviation (Phase 0, D1). A later convention change must not silently change what a Stingray metric means.
7. **Do not break what exists.** `/metrics`, `stingray status` and `opentail_timing` keep working; old instrument names are kept for one release with a changelog note before removal.
8. **Telemetry never breaks inference.** Our own listeners and sinks catch and swallow their exceptions; exporter failures stay inside the SDK. A test confirms instrumentation is harmless with no listener attached.
9. **Define every throughput.** Prefill, decode and end-to-end throughput are different measurements and are named and documented as such (§5). Rates are derived from summed tokens and summed durations, never from an average of per-request rates.
10. **NativeAOT-safe, proven on the real binary.** No reflection-based exporters in the libraries. The CLI already publishes with `PublishAot` (see `docs/guides/native-aot-deployment.md`); compiling is not proof, so the check is to publish the real CLI, run it with a listener attached, and confirm the expected measurements and activities arrive. `EventSourceSupport`, `MetricsSupport` and `HttpActivityPropagationSupport` are separate switches with separate effects and are decided separately (D9, D10). There is no AOT step in CI today.
11. **Performance wins on the hot path; detail is tiered, not dropped.** When detail and speed conflict, speed wins by default, because Stingray's own measurements (the Performance League, backend comparisons) are only trustworthy if the instrumentation does not disturb what it measures. An instrument that does not fit the §8 overhead budget moves to a more expensive tier or is sampled; it is not deleted. The cost tiers (§4.1; not to be confused with the stability tier in §14): **(1) always on**, per-request records and per-iteration aggregates, which must fit the budget; **(2) costs something only when a listener is attached**, spans and per-iteration histograms; **(3) explicit diagnostic mode**, the phase profile and any sampling, off by default. Numbers taken in diagnostic mode are perturbed by the profiling itself, so they are labelled as such and never feed benchmark comparisons. Correctness and privacy are never traded for either detail or speed.

### 4.1 Performance priority and the cost tiers

**The rule.** Performance wins on the hot path. Detail is offered in tiers, so a user pays for it only when they ask. The default answer to "more detail or more speed?" is speed, with the detail moved to a tier, not dropped.

**Why.**
- Stingray is an inference engine. Its own measurements (the Performance League, backend comparisons) are only trustworthy if the instrumentation does not disturb what it measures. A telemetry layer that slows decoding by even a few percent makes every figure it reports slightly wrong.
- Choosing performance does not lose detail: an instrument that does not fit the §8 overhead budget goes to a more expensive tier or is sampled.
- Much of the detail is free anyway: the engine already measures encode, prefill and time to first token, so exporting them costs almost nothing.

**The cost tiers.**

| Tier | What it covers | When it costs anything | Default |
|---|---|---|---|
| 1. Always on | Per-request records and per-iteration aggregates; request-level histograms and counters; model-load duration; execution decisions recorded once per load; observable gauges and counters (read only when collected) | Must fit the §8 budget with no listener attached | On |
| 2. Listener-only | Spans (the request tree and model load); per-iteration histograms (inter-token latency, tokens per iteration, batch size); speculative-decoding counters and per-position counters; the Vulkan per-request aggregates (only if the underlying counters are cheap enough for tier 1, otherwise tier 3) | Only when a listener or exporter is attached, and then within the with-listener budget | Present, inactive until listened to |
| 3. Diagnostic mode | The opt-in phase profile (`MoePhaseTiming`, the Vulkan GPU split, ONNX Runtime profiling) and any sampling of fine-grained measurements | Real overhead, stated and measured | Off; explicit switch |

**Caveat for tier 3.** Numbers taken in diagnostic mode are perturbed by the profiling itself. They are labelled as such and never feed the Performance League or any benchmark comparison.

**Applying it.** Every instrument in the §5 mapping table has a tier. A borderline instrument is measured with the Phase 1 benchmark and then placed; it is not argued about in the abstract. Correctness and privacy are never traded for detail or speed.

## 5. Instrument and span catalogue (proposed)

Names follow the GenAI conventions where one exists and has been verified; `opentail.*` is used for Stingray-specific measures. Final names are fixed in Phase 0 through the mapping table below. **Only the rows for model load, prefill, decode, time to first token, token counts and outcome belong to Phase 1;** the rest of the catalogue is staged into later phases (§6).

### Semantic-convention status (verified 2026-10-12)

- The GenAI conventions have moved from the main `semantic-conventions` repository to **`open-telemetry/semantic-conventions-genai`**; the old documentation page is only a notice. That repository shows no releases or tags and no stated stability level, so "pin a version" is not possible in the usual way: pin a commit, and record it in the mapping table.
- Every metric there is marked **Development**. The metrics defined in `docs/gen-ai/gen-ai-metrics.md` are histograms in seconds: `gen_ai.client.operation.duration` (client), and `gen_ai.server.request.duration`, `gen_ai.server.time_to_first_token` and `gen_ai.server.time_per_output_token` (server), plus agent, workflow and tool metrics that do not apply here.
- **Token usage has been renamed.** `gen_ai.client.token.usage` and the `gen_ai.token.type` attribute do not appear in the current documents. Token metrics are now `gen_ai.client.inference.usage.input_tokens` and `.output_tokens` (counters) and `gen_ai.client.inference.operation.input_tokens` and `.output_tokens` (histograms), with `gen_ai.token.modality` (text, image, audio, unknown) and a required `gen_ai.provider.name`. They are **client** metrics, so using them on a server is a decision, not a given (D2).
- An earlier draft of this plan used the old token-usage name; it has been replaced.

### Mapping table (checked in, completed in Phase 0; initial rows; every instrument also records its cost tier, §4.1)

| Stingray measurement and its definition | Exported name | Unit | Standard or Stingray-specific | Checked against | Deviation | Tier |
|---|---|---|---|---|---|---|
| Server operation duration (§5 timing contract) | `gen_ai.server.request.duration` | s | Standard (Development) | convention commit to be recorded | none proposed | 1 |
| Time to first token (server) | `gen_ai.server.time_to_first_token` | s | Standard (Development) | as above | boundary stated in the timing contract | 1 |
| Time per output token: decode time over (output tokens minus one) | `gen_ai.server.time_per_output_token` | s | Standard (Development) | as above | none proposed | 1 |
| Prefill duration and new prompt tokens | `opentail.engine.prefill.duration`, `.tokens` | s, `{token}` | Stingray-specific (no standard equivalent) | n/a | n/a | 1 |
| Decode duration and output tokens | `opentail.engine.decode.duration`, `.tokens` | s, `{token}` | Stingray-specific | n/a | n/a | 1 |
| Input and output tokens per request (server) | `opentail.tokens.input`, `opentail.tokens.output` (proposed) | `{token}` | Stingray-specific until the conventions define server token metrics (D2) | n/a | documented | 1 |

### Timing contract (to be fixed in Phase 0 and asserted by tests)

Two correct-looking metrics must never describe different intervals. Proposed definitions, to be confirmed against the code (including where cold model acquisition falls relative to `BeginServingRequest()`):

| Measurement | Starts | Ends | Notes |
|---|---|---|---|
| HTTP duration | Request arrives | Response complete | Owned by ASP.NET Core's own instrumentation, not by Stingray |
| Server operation duration (`gen_ai.server.request.duration`) | `BeginServingRequest()`, after request parsing and template rendering | Response complete, including serialisation and streaming | Excludes parsing and template rendering |
| Queue wait | Engine submission | `MarkAdmitted()` | Admission wait only |
| Time to first token | `BeginServingRequest()` | First generated text or reasoning token | Includes queue wait; whether it includes a cold model load must be stated and tested |
| Prefill | Start of prompt processing inside the engine | End of prefill | Excludes queueing and tokenisation; counts new prompt tokens, not reused prefix tokens |
| Decode | First generated token | Last generated token | The first token belongs to time to first token; time per output token divides decode time by (output tokens minus one) |
| Total inference duration | Engine submission | Generation complete | A defined interval, not assumed equal to HTTP duration |

Rules for the awkward cases: a cancelled or disconnected request records outcome `cancelled` and the time to the cancellation point; a streamed response's operation duration runs to the last byte written; a request that waits for a model load records that wait separately (§5 traces) instead of folding it silently into prefill.

**Metrics: Server**

| Instrument | Type, unit | Attributes | Notes |
|---|---|---|---|
| `gen_ai.server.request.duration` | histogram, s | operation, model, endpoint, `error.type` | Replaces the request-duration summary |
| `gen_ai.server.time_to_first_token` | histogram, s | operation, model, endpoint | From the existing `ServingRequestTiming` |
| `gen_ai.server.time_per_output_token` | histogram, s | operation, model | `(end - first token) / (output tokens - 1)`, one sample per request. Fixes G2 |
| `opentail.tokens.input`, `opentail.tokens.output` | histogram, `{token}` | operation, model | Stingray-specific names proposed because the standard token metrics are client-side (`gen_ai.client.inference.*`); decision D2 |
| `opentail.requests.rejected` | counter | reason (`overload`), endpoint | Replaces `opentailllm.overload_rejections` |
| `opentail.requests.active` | up/down counter | endpoint | |
| `opentail.queue.wait` | histogram, s | model | From `MarkAdmitted()` |

**Metrics: Engine (library, so CLI and embedders benefit).** Phase 1 covers model load, prefill, decode, token counts and outcome only; the other rows are Phase 4 or later.

| Instrument | Type, unit | Attributes |
|---|---|---|
| `opentail.engine.prefill.duration`, `opentail.engine.prefill.tokens` | histogram, s and `{token}` | model, backend | Prompt-processing time and input tokens. Prefill throughput = input tokens / prefill time |
| `opentail.engine.decode.duration`, `opentail.engine.decode.tokens` | histogram, s and `{token}` | model, backend | Generation time and output tokens. Decode throughput = output tokens / decode time |
| `opentail.engine.model.load.duration` | histogram, s | model, backend |
| `opentail.engine.kv_cache.bytes` | observable gauge, By | model, backend |
| `opentail.engine.batch.size`, `opentail.engine.sequences.active` | histogram / gauge | model, only where continuous batching exposes them |
| `opentail.engine.speculative.drafts`, `.draft_tokens`, `.accepted_tokens` | counters | model, mode (draft model, lookup, MTP) | Phase 4 at the earliest. Cumulative counters, not a per-request ratio: acceptance rate is derived as `rate(accepted_tokens) / rate(draft_tokens)`. vLLM exports the same shape (`vllm:spec_decode_num_drafts`, `_num_draft_tokens`, `_num_accepted_tokens`), which is a useful precedent. **Not exported until drafts, drafted tokens and accepted tokens (and what counts as committed) have a precise written definition for each decoding mode**, so the number is comparable across modes. The existing `EngineEvents` MTP step events are the likely source |
| `opentail.engine.speculative.accepted_tokens_by_position` | counter | model, mode, `position` | Bounded by the lookahead length (a small fixed number), so cardinality stays low. vLLM and llama.cpp both export an accepted-tokens-per-position series (`spec_decode_num_accepted_tokens_per_pos`); it shows where drafts stop being accepted |
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

**Traces.** Sources `OpenTail.Stingray.Server` and `OpenTail.Stingray.Engine`. Span boundaries mirror the timing contract: ASP.NET Core's own span covers the whole HTTP request; `template.render` is a child of it that ends before the operation span starts; the operation span `chat <model>` is the interval `gen_ai.server.request.duration` measures, with children `queue.wait`, `tokenize`, `prefill` and `decode`. Attributes: `gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.response.finish_reasons`, `opentail.backend`. W3C `traceparent` is honoured automatically by ASP.NET Core.

**Model load is its own operation, not a child of a request.** `model.load` (with children `gguf.parse`, `tensor.map` and `device.upload`) is recorded once per physical load. A request that waits for it records the wait duration separately and carries a span link to the load, so one shared load is never attributed to an arbitrary request.

**Span lifecycle across the engine thread.** This is a design problem, not just an `ActivityContext` argument. The code that starts a span ends it, in a `finally`; exceptions and cancellation set the status and the `cancelled` or `error` outcome before the span ends; the dedicated engine thread receives an explicit parent context so concurrent requests stay isolated; and in `ContinuousBatchingEngine`, where one iteration serves several requests, each request's span links to the iteration instead of becoming its child. Tests must prove that request identities and timings do not leak between concurrent requests.

### Ideas considered and not adopted

| Idea | Why not |
|---|---|
| Kernel-level spans (per matmul, rope, attention, softmax, dispatch) as the way to get a time breakdown | Thousands of spans per token would swamp the engine and any backend, and measuring each kernel costs real time. The opt-in aggregate phase profile (§5) gives the same breakdown at a stated cost |
| A span per Vulkan dispatch | Thousands of spans per token would swamp both the engine and any backend. Per-request aggregates (§5) answer "GPU or synchronisation overhead?" at no real cost. Per-kernel GPU time needs timestamp queries, which are not wired up |
| Tag `Activity.Current` from an `EngineEvents` sink to attach engine data to the request span | The sink runs on the thread that emitted the event. That is the dedicated engine thread, where `Activity.Current` is not the request's activity (G8): the tag would silently be lost, or land on another request's span. The server also never emits these events today, and a CLI has no current activity |
| An `IChatClient` adapter (Microsoft.Extensions.AI) added now to get client-side GenAI telemetry from its `UseOpenTelemetry()` decorator | Worth remembering, but it is a separate product feature, not a telemetry task. Stingray has no `IChatClient` integration or plan today (nothing in `src/` or `docs/`). If it is ever built, embedders would get standard client-side spans and `gen_ai.client.*` metrics from the decorator, as Microsoft's agent-framework does with `OpenTelemetryChatClient` and `OpenTelemetryAgent`, without Stingray instrumenting anything for that path |
| A span, or an allocation, per generated token | Breaks principle 3. Per-request records cover request-level questions; step-level questions use one histogram sample per decode iteration (§14), never a span per token |
| A SQLite (or sqlite-vec) sink | Adds a native dependency and a schema to maintain; the JSON-lines sink and OTLP cover local use without either |
| Reporting a weight-size estimate as resident memory | It overstates what is known; estimates are named and labelled as estimates (§5) |
| A rolling per-token latency average | Misleading under batching and streaming; time per output token per request is simpler and comparable across runs |

## 6. Phases

**Phase 0: Decisions and contract.** Settle §11 (each question carries a proposed answer). Record the convention commit and complete the mapping table (§5). Write down the timing contract (§5) after checking where cold model acquisition falls relative to `BeginServingRequest()`. Agree the overhead thresholds (§8). Run two short spikes to settle D9 and D10: publish the CLI with `MetricsSupport` on and a listener attached; and bridge the **existing** server meter to OTLP in `Server.Host` (an `AddMeter("OpenTail.Stingray.Server")` plus OTLP exporter in the host) to confirm the SDK, the exporter and the trimming and AOT behaviour. Treat that spike's output as provisional and do not document it: `opentailllm.itl_ms` is mislabelled and no instrument has attributes until Phase 2. *Done when* every open question has an answer recorded in this file and the mapping table and timing contract are checked in as tables that tests assert against.

**Phase 1: Engine diagnostics foundation, one vertical slice (the first milestone).** Scope, and nothing more: model-load duration and outcome; one representative text-generation path (`GenerateChunksAsync`); prefill, decode, time to first token and total inference duration; input and output token counts and outcome; the resolved model and the actual execution backend. One static diagnostics class in the Engine (reused `ActivitySource` and `Meter`), with the contract from Phase 0. Start with the data contract: `GenerateChunk` carries only `PromptTokens` and truncation flags today, so extend its usage or stop chunk (or add a metrics chunk) to carry **one extensible per-request stats record** (a small class instance holding the encode, prefill and decode durations, token counts, reused tokens and finish reason that the engine already computes, allocated once per request and referenced only from the final chunk) instead of loose fields. It must not be a struct field on the per-token chunk: `GenerateChunk` is a struct yielded for every token, so growing it would add copying to the decode path, so a later metric adds a field and not a new plumbing path. The `ModelLoaded` event already reports build time, architecture and forward-pass kind, so the model-load measurement and the actual backend can come from the same emission point. Keep instrumentation at these shared boundaries, not duplicated across the CLI, server and library, and do not change the meaning of any existing public metric or timing response. Ship listener-based tests (`MeterListener` and `ActivityListener`) for names, units, attributes, outcome, cancellation and concurrent-request isolation; a test that telemetry failures cannot break inference; a runnable example that works with and without a listener; and the published NativeAOT CLI run with a listener attached, showing the expected instruments and activities arrive. Report the measured overhead using §8. *Done when* a developer can answer "why was this inference slow, which backend did it use, and where did it fail?" with no extra instrumentation of their own, the measurements correspond to the documented timing boundaries, the NativeAOT and trim builds pass with warnings as errors, and the library references no OpenTelemetry package. **Out of scope for Phase 1:** server migration, speculative acceptance, prefix-cache and residency metrics, Vulkan aggregates, batching, memory, other modalities, kernel profiling and the benchmark bridge.

**Phase 2: Fix and extend the server meter.** The Server consumes the engine measurements. Add attributes (G1); correct time per output token (G2); add the GenAI-named instruments beside the old ones (G3); add active-request and queue-wait instruments. Repair the `/metrics` names (G9) and make **one source of truth an explicit architecture decision**: record each measurement once and feed the `Meter` instruments, the Prometheus page and `stingray status` from it, with a test that all three agree on request counts and latency. Keep `/metrics` (Prometheus scraping is a useful integration and is not removed just because OTLP exists), and never require an OTLP collector for `stingray status` to work (D14). Cover the image, audio, embedding and rerank endpoints, not only chat. Extend the existing opt-in `opentail_timing` response extension (today only `time_to_first_token_ms` and `total_ms`) to carry the same per-request stats record (queue, prefill and decode time, new and reused prompt tokens, generated tokens, finish reason), so a client can see what a response cost without any telemetry backend; this is the shape Ollama users already know from its response fields (see §14, reference points). Tests with `MeterListener` assert instrument names, units and attribute sets, and that no forbidden attribute appears. *Done when* the Server test suite passes and `/metrics` and `stingray status` are unchanged.

**Phase 3: Traces.** Add the span tree to the Server endpoints and the engine, following the lifecycle design in §5 (explicit context hand-off to the engine thread (G8); span ownership and ending; cancellation and exceptions; model load as its own operation with linked waits; span links in `ContinuousBatchingEngine`). *Done when* one chat request produces one connected trace with the expected children, verified by an `ActivityListener` test.

**Phase 4: Engine extras.** The instruments deferred from Phase 1, each added only with a precise definition: `ModelRuntimeManager` residency and counters as observable gauges (G6), prefix-cache hits, shared tokens and estimated bytes, execution decisions from `ExecutionPlanner` (selected backend, fallback reason category, GPU layer share), KV-cache estimated bytes, and speculative acceptance only once proposed, accepted and committed tokens are defined per decoding mode (the `EngineEvents` MTP step already defines drafts proposed and drafts accepted per verify step, but not committed tokens, so that definition is still missing). *Done when* each instrument has a written definition in the mapping table and a test asserting it.

**Phase 5: Host wiring.** Wire OTLP in `Server.Host` from the standard `OTEL_*` environment variables, with a console exporter for local development. Decision D3 chooses between documentation snippets and a small optional package (for example `OpenTail.Stingray.OpenTelemetry`) that registers the meters and sources. Also add a **local diagnostics experience with no infrastructure**: an in-process listener over the same instruments that feeds `stingray status` and, if wanted (D12), a human-readable report of the last run (model, backend and the planner's reasons, prompt and generated tokens, load, prefill and decode times, outcome) and a support bundle for bug reports. The report states what is known and what is unknown: an unmeasurable value is shown as unknown, and an estimate is labelled as one. It needs no extra package and no native dependency, and one instrumentation source feeds the listener, OpenTelemetry and the report. *Done when* a fresh `Server.Host` run exports to an OTLP endpoint with no code changes.

**Phase 6: Documentation and a worked example.** A guide `docs/guides/observability.md`, plus a sample showing the free local .NET Aspire Dashboard as a viewer (it needs Docker; document the console exporter and Prometheus scrape of `/metrics` as no-Docker alternatives).

**Phase 7: Other pipelines (later).** Audio: real-time factor, and per engine the split between autoregressive token (codebook) generation and waveform decode. Diffusion: `opentail.diffusion.steps`, `opentail.diffusion.step.duration`, images generated, output size in bounded buckets. Speech: audio seconds in and out, transcription or synthesis duration, `opentail.speech.realtime_factor`. Vision: image decode and encoder duration, vision tokens produced. Embeddings: throughput. Each behind the same conventions.

**Phase 8: Benchmark bridge and phase profile (later).** Let the benchmark scripts and the Performance League read structured numbers (prefill and decode duration and tokens, backend, outcome) from the engine, for example as machine-readable output from a run, instead of regex-matching printed text. Add the opt-in phase profile from §5; it is the diagnostic tier (principle 11), and its numbers are labelled as perturbed and never used for Performance League or other benchmark comparisons. For the ONNX-based speech engines, the ONNX Runtime C# API already exposes `SessionOptions.EnableProfiling`, which writes per-node timings; the profile switch can enable it and summarise the result instead of re-measuring. Telemetry does not replace controlled benchmarking: warm-up, repeated runs and reporting the spread still follow the existing performance docs. Regression detection and automated comparisons are a downstream use of this data and are out of scope for this plan.

## 7. Cardinality and privacy rules

- **Allowed metric attributes:** operation, model id, backend, endpoint, `error.type`, HTTP status class. In multi-model mode the model id comes from the client, so it is used only when it matches a configured model and is otherwise recorded as `other`; the existing server code already avoids per-model labels for this reason.
- **Forbidden on any metric or span attribute:** prompt or completion text, tool arguments or results, the canonical history prefix, image or audio bytes, request, session or user identifiers, client-supplied `user` or `metadata` fields, file paths and raw model paths (a model path exposes user directories), and exception messages (record the exception type name only, because messages can echo prompt content).
- **Spans** may carry request-scoped identifiers but never content.
- **Model identity is configurable.** The default is the catalogue id or the architecture family. A file name or path is never used unless explicitly enabled, because it can disclose a user name, company directory or private project.
- **Correlating metrics with traces** uses exemplars where the SDK supports them, never trace or request ids as metric attributes.
- **No content in OpenTelemetry, with no debugging flag.** Prompt and completion text can hold credentials, personal data, proprietary code or business documents, and spans can be exported to systems with different access and retention rules; an operator who enables tracing should not thereby enable content capture. If content capture is ever needed, it is a separate, conspicuous local diagnostic feature with its own retention and disclosure controls, outside OpenTelemetry (D7).
- A unit test enumerates the attribute keys on every instrument and fails on anything outside the allow-list.

## 8. Performance budget and verification

- **The claim is made measurable, not absolute.** No tag construction, string building or per-token allocation when unobserved, no per-token spans, and `Counter.Enabled` / `ActivitySource.HasListeners()` checked first.
- **Procedure.** Interleaved A/B runs of an uninstrumented baseline and the instrumented build, at least 10 runs each, on the same model, prompt and machine state, using the method in the performance docs. Measure the baseline noise band first. Report the median and the spread. A difference counts only if it exceeds the baseline spread; if the spread is larger than the threshold, repeat on a quieter machine or with more runs.
- **Proposed thresholds (D15):** no listener, within the baseline noise band and no more than 1% on the median; an in-process listener, no more than 2%; an OTLP exporter attached, no more than 3%. Costs are reported separately for each of the three cases.
- All timing uses `Stopwatch.GetTimestamp()`; no allocation per token.
- **Do not grow the per-token chunk.** The per-request stats record is reached through one reference on the final chunk (Phase 1); the §8 benchmark would catch any decode regression from the chunk's size.
- **Start-up and binary size are part of the budget.** The README states the CLI starts in about 60 ms; measure start-up time and the published binary size with and without the instrumentation (and with `MetricsSupport` on), and create instruments lazily so nothing is initialised before it is used.
- **A CPU-bound run is the hard case for an exporter.** CPU inference keeps all cores busy, so an exporter's background work competes with the compute threads. Take the "with exporter" measurement on the CPU backend, and document a long default export interval (for example 30 to 60 seconds).
- **Published NativeAOT CLI, not just a build.** Publish the real CLI, run it with a listener attached, and show the expected measurements and activities arrive. Record which of `MetricsSupport`, `EventSourceSupport` and `HttpActivityPropagationSupport` had to change and the size and start-up effect of changing them (D9, D10).

## 9. Compatibility and deprecation

- Keep `opentailllm.*` instruments and the Prometheus output for one release alongside the new ones.
- `opentailllm.itl_ms` changes meaning (from total generation time to a true per-token figure); record this in `CHANGELOG.md` and in the release notes.
- Removal of the old names follows the project's normal release-notes policy.

## 10. Risks

| Risk | Mitigation |
|---|---|
| GenAI semantic conventions change again (they already moved repository and renamed the token metrics, with no release to pin) | Pin a commit (D1); the checked-in mapping table; names in one internal constants file |
| Two correct-looking metrics describe different intervals | The timing contract (§5), asserted by boundary tests including streaming, cancellation and cold acquisition |
| An existing metric's meaning changes under people | Phase 1 changes no existing metric; `itl_ms` is corrected in Phase 2 with a changelog note (§9) |
| Label explosion from model or endpoint names | Allow-list test (§7); model id comes from the catalogue or loaded-model name only |
| Overhead on the decode path | §8 budget, per-request recording, `Enabled` checks |
| Streaming responses that fail midway | Record `error.type` and partial token counts; finish spans in a `finally` |
| Trace context is lost across the engine thread (and shared across requests in the batching engine) | Explicit `ActivityContext` hand-off; links in the batching engine (Phase 3) |
| The published CLI has `MetricsSupport=false` | Decide D9 first; either enable it for the CLI build or keep CLI data on `EngineEvents` |
| The OpenTelemetry SDK and OTLP exporter are not confirmed NativeAOT-clean under warnings as errors | Spike before committing the CLI or Server.Host AOT builds to them (D10) |
| Several models resident at once make "current model" ambiguous | Attributes are set per request from the resolved model, not from global state |
| NativeAOT trimming of exporter code | Keep the SDK in the host only; publish and run the CLI with AOT before release (D10: add a CI step) |

## 11. Open questions for the owner

Each question carries a **proposed** answer from the independent reviews. None is decided until the owner confirms.

**Decided 2026-10-12 (owner said "go ahead" to the proposed answers; D1, D2, D9, D16 are needed for Phase 1; the rest stay proposed until their phase):**
- **D1:** pin a commit of `semantic-conventions-genai`, recorded in the mapping table (commit to be filled in when the table is written).
- **D2:** Stingray-specific `opentail.tokens.input` / `.output` on the server.
- **D9:** CLI telemetry yes, if the published NativeAOT binary passes the listener test. Spike evidence: with `MetricsSupport=true` a published AOT app delivers all `Meter` measurements (false: none); activities work either way; about +31 KB in a minimal app. Real-CLI size and start-up cost are measured in Phase 1.
- **D10 (spike):** OpenTelemetry SDK 1.19.1 + OTLP + console exporters publish NativeAOT with warnings as errors, with no warnings, and deliver a meter and a span to the console exporter (exe 8.3 MB). OTLP delivery to a collector and `Server.Host` itself are not yet tested.
- **D16:** `AcquireAsync` runs before `BeginServingRequest()` in the OpenAI endpoint (line order), so time to first token **excludes** a cold model load; the load is recorded separately (`model.load`). To be asserted by a test in Phase 1.

- **D1: Which GenAI convention version do we use?** *Proposed:* pin a commit of `semantic-conventions-genai` (there is no release to pin), verify each proposed standard name against it, and allow deliberate Stingray-specific metrics, all recorded in the mapping table.
- **D2: Token usage.** The current token metrics are client-side (`gen_ai.client.inference.*`). *Proposed:* Stingray-specific `opentail.tokens.input` and `.output` on the server, documented in the mapping table, revisited if the conventions define server token metrics.
- **D3: Separate optional package for OpenTelemetry wiring, or documentation snippets?** *Proposed:* start with host-side setup and a runnable example; add a dedicated integration package only if it materially simplifies use.
- **D4: Which backends do users send data to (Aspire Dashboard, Grafana/Prometheus, Azure Monitor, Datadog)?** *Proposed:* prove the OTLP path with the Aspire Dashboard first; keep documented Prometheus and console alternatives.
- **D5: Is `/metrics` the long-term interface?** *Proposed:* keep it for compatibility, as a view over the same measurements as the `Meter` instruments; never require a collector for `stingray status`.
- **D6: Default-on or opt-in?** *Proposed:* instrumentation present by default (negligible, measured cost); external export only when explicitly configured.
- **D7: Content capture.** *Proposed:* none in OpenTelemetry, no debugging flag. Reconsider only as a separate local diagnostic feature if there is a demonstrated need. Both independent reviews and this plan now agree.
- **D8: Priority.** *Proposed:* the engine and the text-serving path first; other modalities later.
- **D9: Does the CLI get telemetry?** The published CLI sets `MetricsSupport=false`. *Proposed:* yes, if the actual published NativeAOT binary passes the listener tests; otherwise keep CLI data on the existing `EngineEvents` bus.
- **D10: Add a NativeAOT publish-and-run step to CI?** *Proposed:* yes; a manual-only check is too weak for a plan whose core constraints include NativeAOT and trimming.
- **D11: Abstraction.** An observer hook already exists in the form of `EngineEvents` and `IEngineEventSink` (§2), so the real choice is among three: (a) instrument directly with `ActivitySource` and `Meter` at request boundaries; (b) route through `EngineEvents`, with a sink that updates `Meter` instruments; (c) add a new telemetry interface. *Proposed:* (a) for the server request path, because it needs request identity and the ASP.NET context, which `EngineEvents` does not carry and cannot supply on the engine thread; keep `EngineEvents` as the CLI, benchmark and local-diagnostics channel, and consider a thin sink that bridges only the events that have no request context (model loaded, MTP steps, GC stats). Avoid (c) unless a concrete need appears; any interface must be boundary-level, never per token.
- **D12: Local diagnostics.** *Proposed:* yes, starting with structured per-run output and the existing `stingray status` experience; defer a larger support-bundle feature.
- **D13: Benchmark bridge.** *Proposed:* yes, later, as a deliberate machine-readable CLI contract, not scraped logs or a side effect of exporting metrics.
- **D14: The `/metrics` page.** *Proposed:* keep it during migration, make it a view over the same measurements, and fix the hyphenated names.
- **D15: Overhead thresholds.** *Proposed:* no listener within the baseline noise band and at most 1%; in-process listener at most 2%; OTLP exporter at most 3% (§8).
- **D16: Timing contract.** Confirm the boundaries in §5, and check where cold model acquisition falls relative to `BeginServingRequest()` before fixing the time-to-first-token definition.
- **D17: Should Stingray also offer a vLLM-compatible Prometheus view** (vLLM-style metric names) so existing vLLM dashboards work? *Proposed:* not now. Revisit after Phase 4, once the real metric set exists; a compatibility view is cheap to add later if the seams in §14 are in place, but it adds a naming surface to maintain.
- **D18: Priority rule when detail and performance conflict.** *Proposed:* principle 11: performance wins on the hot path by default; detail moves to a more expensive tier (listener-only, then explicit diagnostic mode) instead of being dropped; diagnostic-mode numbers never feed benchmark comparisons.

## 12. Acceptance checklist

- [ ] For the phase being delivered, every instrument has the documented name, unit and attribute set, and an entry in the mapping table, asserted by tests.
- [ ] The measurements correspond to the documented timing boundaries, including streaming, cancellation and cold model acquisition (tests).
- [ ] No instrument or span carries an attribute outside the §7 allow-list, and no content appears anywhere in telemetry (tests).
- [ ] Concurrent requests do not cross-contaminate trace context or measurements (tests).
- [ ] A telemetry or listener failure cannot break inference (test).
- [ ] Time per output token is a true per-token figure; the old `itl_ms` meaning is documented as changed (Phase 2).
- [ ] One chat request yields one connected trace with the expected child spans, and a shared model load is a separate operation with linked waits (Phase 3).
- [ ] The library emits data with no OpenTelemetry package referenced.
- [ ] The published NativeAOT CLI, run with a listener attached, shows the expected instruments and activities.
- [ ] `/metrics`, the `Meter` instruments and `stingray status` agree on request counts and latency, and existing public behaviour is unchanged (Phase 2).
- [ ] Overhead is reported with no listener, an in-process listener and an OTLP exporter, using the §8 procedure, and is within the agreed thresholds.
- [ ] `docs/guides/observability.md` and the worked example exist and were run on a clean machine.
- [ ] Every instrument has a cost tier in the mapping table (§4.1); only tier 1 does work with no listener attached, and tier 3 output is labelled as perturbed and is never used in benchmark comparisons (tests).
- [ ] The growth seams in §14 exist (one stats record, observable callbacks over the batching-observability interface, per-instrument bucket advice, closed-enum attributes, the stability tier), so adding a metric is a declaration, a mapping-table row and a test.

## 13. First task brief (Phase 1), for a coding agent

Do not hand an agent this whole document as one task. Give it Phase 1 only:

- Implement Phase 1 after resolving the Phase 0 decisions that affect the initial engine contract (§11 D1, D2, D9, D16).
- **Scope:** reusable, dependency-light `ActivitySource` and `Meter` instrumentation in the engine; model loading and one representative text-generation path; correctly defined prefill, decode, time to first token, token-usage and total-duration measurements, plus outcome and resolved backend and model identity; no per-token spans or allocations.
- **Tests:** listener-based tests for names, units, attributes, outcome, cancellation and concurrent-request isolation; proof that telemetry failures cannot break inference; the published NativeAOT CLI run with a listener attached; a runnable example with and without a listener.
- **Constraints:** no mandatory OpenTelemetry SDK or exporter dependency in the engine libraries; no prompt, completion, image, audio or tool content in telemetry; no high-cardinality identifiers or raw filesystem paths as attributes; do not change the semantics of existing public metrics or timing responses; preserve existing CLI, server and `/metrics` behaviour; all builds warning-free under the repository's trimming and NativeAOT settings.
- **Report:** measured overhead per §8, and any unsupported capability or unresolved convention decision, stated plainly instead of approximated.
- **Not in this task:** server migration, other modalities, kernel profiling, the benchmark bridge.

## 14. Growth path to vLLM-class server metrics

**Goal of this section:** not to match vLLM now, but to make sure nothing in this plan blocks matching it later, and to say what each step would need.

**What vLLM exports (verified in `examples/vllm/vllm/v1/metrics/loggers.py`; about 38 metric names, plus a separate set for speculative decoding).** Scheduler state (`num_requests_running`, `num_requests_waiting`, `num_requests_waiting_by_reason`, `num_preemptions`); KV cache (`kv_cache_usage_perc` and block lifetime, idle and reuse-gap histograms); prefix and multimodal caches (queries and hits, internal and external); token counters (`prompt_tokens`, `generation_tokens`, `prompt_tokens_by_source`, `prompt_tokens_cached`, and a per-iteration `iteration_tokens_total` histogram); request histograms (prompt and generation tokens, queue, prefill, decode and inference time, time to first token, time per output token, `inter_token_latency_seconds`, and request parameters such as max tokens); `request_success` with a finish-reason label; plus LoRA, distributed KV transfer, engine sleep state and corrupted-request counters.

**Parity map.** Stingray's `IContinuousBatchingObservability` (in `Engine/IInferenceEngine.cs`) already exposes the scheduler and cache state below. Status is relative to this plan.

| vLLM group | Stingray equivalent and source | Status |
|---|---|---|
| `num_requests_running`, `num_requests_waiting` | `ActiveRequests`, `QueueDepth` | Ready as gauges (Phase 2) |
| `num_requests_waiting_by_reason`, `num_preemptions` | Bounded admission queue and overload rejections exist; no preemption concept was found in `ContinuousBatchingEngine` | Waiting-by-reason via a closed enum of rejection reasons; preemptions only if the engine gains the concept |
| `kv_cache_usage_perc` | `CommittedKvTokens` over `KvTokenBudget` | Ready (Phase 4) |
| KV block lifetime, idle and reuse-gap histograms | Depends on how Stingray manages KV blocks; not assessed | Needs design |
| `prefix_cache_queries`, `_hits`, `prompt_tokens_cached` | `PrefixCacheHits`, `PrefixCacheMisses` (queries = hits + misses), `PrefillTokensReused`, plus evictions, entries and bytes | Ready (Phase 4) |
| External prefix cache, distributed KV transfer, LoRA, engine sleep state | No equivalent | Not applicable unless such features appear |
| Multimodal cache hits and queries | Not assessed | Needs a look at the vision embedding path |
| `prompt_tokens`, `generation_tokens`, `request_prompt_tokens`, `request_generation_tokens`, `prompt_tokens_by_source` | Stats record: new versus reused prompt tokens, generated tokens | Ready (Phases 1 and 2); `source` is a closed enum |
| `request_queue_time`, `request_prefill_time`, `request_decode_time`, `request_inference_time`, `time_to_first_token`, `request_time_per_output_token` | The §5 timing contract and the stats record | Ready (Phases 1 and 2) |
| `inter_token_latency_seconds`, `iteration_tokens_total` | One histogram sample per decode iteration in `ContinuousBatchingEngine` | Possible after the §8 budget is measured; needs the per-iteration allowance in principle 3 |
| `request_params_max_tokens`, `request_max_num_generation_tokens`, `request_params_n` | Request parameters | Optional, low priority; bounded buckets |
| `request_success` with finish reason | Outcome counter with a closed finish-reason enum | Ready (Phase 2) |
| Speculative decoding (`num_drafts`, `num_draft_tokens`, `num_accepted_tokens`, per-position) | The §5 speculative counters | Phase 4, after the definitions are written |

**Design seams that keep the window open** (these are the only things to build now; each is small):

1. **One extensible per-request stats record** carried out of the engine, so adding a request-level metric adds a field, not a new plumbing path.
2. **Observable callbacks over `IContinuousBatchingObservability` and `ModelRuntimeManager.Snapshot()`** for gauges and cumulative counters. The callback enumerates the resident engines, so adding a gauge needs no change to the engine's hot path and multi-model setups work.
3. **One telemetry class per layer** (engine, server) as the single place instruments are declared; every instrument has a mapping-table row (§5) and a test, so "add a metric" is a declaration, a row and a test.
4. **Per-iteration histogram samples are allowed**, guarded by `Instrument.Enabled`, inside the §8 budget (principle 3); per-token spans and allocations stay banned.
5. **Explicit histogram bucket advice** on every latency and size histogram (`InstrumentAdvice<T>` bucket boundaries are available from .NET 9; the engine targets .NET 10), recorded in the mapping table, so buckets can be tuned later without renaming a metric.
6. **Closed-enum attributes only** (finish reason, wait reason, token source, decoding mode), enforced by the allow-list test, so richer breakdowns never open the door to unbounded cardinality.
7. **A stability tier:** new instruments land under an `opentail.experimental.*` prefix until a release promotes them, and the `Meter` version is bumped on a breaking change, so growing the set does not promise permanence.
8. **Prometheus name translation is checked once** (how OpenTelemetry names and units become Prometheus names) and covered by a test of the `/metrics` compatibility view (D14), so new instruments appear there predictably.

**Not goals:** matching vLLM's metrics that describe features Stingray does not have (LoRA, distributed KV transfer, external prefix cache), and any vLLM-compatible naming layer before D17 is decided.

### Reference points (checked 2026-10-12 in `C:\Git\OpenTail\examples\ai-telemetry-repos`)

The Ollama server source is not in that folder; what is there is Ollama client and instrumentation code, which shows what Ollama exposes and how people turn it into telemetry. Whether the Ollama server has native Prometheus or OpenTelemetry support is **not verified** here.

- **Ollama's telemetry model is per-response stats.** The `agent-spy` Ollama example reads `eval_duration` and `prompt_eval_duration` from each response (nanoseconds), and the `openllmetry` Ollama instrumentation reads `prompt_eval_count`, `eval_count` and `done_reason`. Telemetry is added by client-side wrappers, not by the server. So the per-request stats record (§14 seam 1) and the extended `opentail_timing` extension match what users of the most popular local runtime already expect. Any bridge or comparison must convert units: Ollama reports nanoseconds, OpenTelemetry uses seconds.
- **Naming churn is real in the wild.** `openllmetry` still ships `gen_ai.client.token.usage` and `gen_ai.client.operation.duration`, plus a non-standard `llm.chat_completions.streaming_time_to_generate`, while the current conventions use `gen_ai.client.inference.usage.*`. This is why the mapping table (§5) records, for every instrument, what it measures and which convention commit it was checked against.
- **Content capture is on by default in a popular library.** `openllmetry` records prompts and completions unless `TRACELOOP_TRACE_CONTENT=false` is set. Stingray's "no content in telemetry, by contract" rule (§7) is therefore a real difference, not a formality.
- **Other repos in the folder** (`openinference`, `openlit`, `langfuse`, `opik`, `helicone`, `agentops`, `tellm` and more) are worth a targeted look when choosing attributes in Phase 0, for example how they model GPU and cache measurements. They were not reviewed for this plan.

## 15. Progress log

- **2026-10-12, Phase 1a/1b (local commits `a261dfdb`, `fe856d42`).** `Engine/EngineTelemetry.cs` (`Meter`/`ActivitySource` `OpenTail.Stingray.Engine`): `opentail.engine.requests`, `.prefill.duration/.tokens`, `.decode.duration/.tokens`, `.first_token.duration`, `.generation.duration`, `.model.load.duration`. One sample per request from the existing `finally` in `InferenceEngine`; model load recorded in `RuntimeInstance.Create`. Attributes: `outcome`, `opentail.forward_pass` (class name, used as a backend proxy; a real backend/device name is still to do), `opentail.architecture` (load only). `GenerateChunk` is unchanged (stats-record plumbing to the server is Phase 2, to be measured). Fix found on the way: the main decode path counted the stop token as a generated token; it no longer does. Tests: `EngineTelemetryTests` (units, tags allow-list, cancellation, concurrency, model load); Fast suite 1113 passed, 0 failed.
- **Overhead (micro, scripted fake engine, 300 tokens/request, 12 x 500 runs):** no listener median 141.9 us, in-process listener 142.6 us, run-to-run range 72-266 us. The listener/no-listener difference is below the noise; the added work is about 6 `Stopwatch.GetTimestamp` calls and one `Enabled` check per request. This is **not** the §8 real-model A/B (no uninstrumented baseline build and no real model yet), so the §8 thresholds are not yet signed off.
- **Not done in Phase 1:** engine `ActivitySource` spans, real-CLI NativeAOT run with a listener (the CLI `run` command has its own decode loop and does not go through `InferenceEngine.GenerateChunksAsync`, so it needs its own instrumentation or the `EngineEvents` bridge), real-model overhead A/B.
- **2026-10-12, Phase 1c (`e52ee92c`):** listener-only `opentail.engine.generate` span (outcome and token counts only). Engine Fast suite 1114 passed.
- **2026-10-12, Phase 2a:** `ServerMetrics` now also records `gen_ai.server.request.duration`, `gen_ai.server.time_to_first_token` and `gen_ai.server.time_per_output_token` (seconds, bucket advice, attributes `gen_ai.operation.name` and a closed `opentail.endpoint`), beside the unchanged legacy `opentailllm.*`. Time per output token = decode time over (output tokens - 1), delivery excluded, from the real token count (`ServingRequestTiming.SetOutputTokens`). Chat endpoints (OpenAI chat, Anthropic messages, Responses) pass their endpoint name. Still to do in Phase 2: model attribute (allow-listed), active-request and queue-wait instruments, `error.type`/outcome, image/audio/embedding/rerank endpoints, `/metrics` name repair and the one-source-of-truth test, `opentail_timing` extension, the per-request stats record from the engine. Server Fast suite 457 passed.
- **2026-10-12, Phase 2b:** `opentail.requests.active` (up/down, by endpoint; decrement only if the increment was recorded), `opentail.queue.wait` (s), `opentail.requests.rejected{opentail.reason=overload}` added beside the legacy counters. Server Fast 458 passed. Remaining Phase 2: model attribute, `error.type`/outcome, non-chat endpoints, `/metrics` name repair + one-source test, `opentail_timing` extension, engine stats record.
- **2026-10-12, Phase 2c (G9):** `/metrics` lifetime metrics renamed `opentail-llm_*` to `opentail_llm_*` (valid Prometheus names; `stingray status` already parsed both), a test asserts every exposed name is a valid identifier, CHANGELOG notes the break. Server Fast 459 passed. Not yet done: the "one source of truth" agreement test across Meter, `/metrics` and `stingray status`.
- **2026-10-13, Phase 2d:** server instruments carry `gen_ai.request.model` = the server-resolved `engine.ModelId` (not the client string); path-like (`/`, `\`, `:`) or over-64-char values become `other`, empty becomes `unknown` (tested). Server Fast 466 passed. Remaining Phase 2: `error.type`/outcome, non-chat endpoints, `opentail_timing` extension, engine stats record, agreement test.
- **2026-10-13, Phase 2e:** `gen_ai.server.request.duration` carries `outcome`: `success` (generation loop finished), `cancelled` (request aborted before it finished), `incomplete` (ended without finishing and not aborted, e.g. an early error return). `error.type` is not used yet (needs per-endpoint exception categories). Server Fast 469 passed. Remaining Phase 2: non-chat endpoints, `opentail_timing` extension, engine stats record, agreement test.
- **2026-10-13, Phase 2f:** per-request stats record. `Engine/GenerationStats.cs` (prompt/reused/output tokens, encode, prefill, first-token, total ms; derived new-prompt tokens and decode ms) is carried by one new nullable reference field `Stats` on `GenerateChunk`, set only on the final `Stop` chunk of `InferenceEngine` (3 sites, one small allocation per request). The chunk struct grew by one reference (8 bytes); its copy cost on the per-token path has **not** been A/B measured yet (the micro-bench run before this change was 141.9 us per 300-token request; rerun it, and the real-model A/B, before signing off §8). `ContinuousBatchingEngine` does not set stats yet. The opt-in `opentail_timing` extension now also returns `prompt_tokens`, `reused_prompt_tokens`, `generated_tokens`, `encode_ms`, `prefill_ms`, `decode_ms` (omitted when the engine gave no stats; original two fields unchanged) for non-streaming OpenAI and Anthropic responses. Engine Fast 1115, Server Fast 470 passed.
- **2026-10-13, Phase 2g:** the non-chat endpoints (`/v1/embeddings`, `/api/embed`, `/v1/images/generations`, `/v1/images/edits`, `/v1/audio/speech|transcriptions|translations`, `/v1/rerank`) now record `gen_ai.server.request.duration` through one wrapper (`Endpoints/OperationMetricsFilter.cs`, wraps the endpoint's `RequestDelegate`; no handler body changed). Attributes: `gen_ai.operation.name` (embeddings, image_generation, image_edit, speech, transcription, translation, rerank), `opentail.endpoint`, `gen_ai.request.model`=`unknown` (model is resolved inside these handlers), `outcome` (success, client_error, error, cancelled). Server Fast 471 passed. Remaining Phase 2: the Meter/`/metrics`/`stingray status` agreement test; `error.type`.
- **2026-10-13, Phase 2h:** agreement test (`MeterInstrumentsAndMetricsSummaries_AgreeOnCounts`): for the same requests the legacy counters, the `/metrics` summaries and the new `gen_ai.server.*` instruments report the same counts (they share one recording method each). Server Fast 472 passed. **Phase 2 status:** done except `error.type` (deliberately deferred: needs per-endpoint exception categories; `outcome` covers success/cancelled/incomplete/error/client_error) and `ContinuousBatchingEngine` stats. Next: Phase 3 (traces).
- **2026-10-13, Phase 3a:** traces. Server: `ServerMetrics.ActivitySourceName` = `OpenTail.Stingray.Server`; each chat serving request starts an operation span `chat <model>` with the timing (child of the ASP.NET Core request span through `Activity.Current`; null and free with no listener), tagged `gen_ai.operation.name`, `gen_ai.request.model`, `opentail.endpoint`, `outcome`, `gen_ai.usage.output_tokens`, and (when the engine gave stats) `gen_ai.usage.input_tokens`, `opentail.reused_tokens`, plus `opentail.time_to_first_token_ms`. Engine: `InferenceEngine.GenerateChunksAsync` captures the caller's `Activity.Current.Context` before the hop to the engine thread and parents `opentail.engine.generate` with it explicitly (G8). Tests: parent/child relationship, and 4 concurrent callers each get exactly their own engine child span (no cross-over); allow-listed span tags. Engine Fast 1116, Server Fast 473 passed. Not done in Phase 3: `queue.wait`/`prefill`/`decode` child spans, `model.load` as its own trace with span links from waiting requests, span links in `ContinuousBatchingEngine`, exception/status mapping beyond `outcome`.
- **2026-10-13, Phase 3b:** `model.load` span (source `OpenTail.Stingray.Engine`) at `RuntimeInstance.Create`, tagged `opentail.architecture`, `opentail.forward_pass`, `outcome`; it is a new trace root, never a child of the request that triggered it. Finding: `ActivitySource.StartActivity` inherits `Activity.Current` even when a default or fresh-trace parent context is passed, so the start clears `Activity.Current` and a scope restores the caller's context on dispose (tested: the requester's activity is current again afterwards). Engine Fast 1117 passed. Still open in Phase 3: span links from requests that waited for a shared load (needs `ModelRuntimeManager`), `queue.wait`/`prefill`/`decode` child spans, links in `ContinuousBatchingEngine`.
- **2026-10-13, Phase 5/6 (first cut):** `Server.Host` wires OpenTelemetry (SDK 1.19.1, OTLP + console exporters, ASP.NET Core instrumentation) only when an `OTEL_*` variable asks; metric export interval defaults to 30 s unless `OTEL_METRIC_EXPORT_INTERVAL` is set; libraries still reference no OpenTelemetry package. **Verified end to end with the real binary** (SmolLM2-135M, console exporters, one chat request): all new metrics exported (`gen_ai.server.*`, `opentail.queue.wait`, `opentail.requests.active`, `opentail.engine.*`, plus ASP.NET/Kestrel), and the trace `POST /v1/chat/completions` -> `chat llama` -> `opentail.engine.generate` is connected (parent ids checked), with `model.load` in its own trace; `opentail_timing` returned real numbers (prompt 32, prefill 428 ms, decode 8 tokens in 209 ms). OTLP delivery to a real collector is still unverified. `docs/guides/observability.md` written. Not done: the support-bundle/local report (D12), Aspire Dashboard run, Phase 4 engine extras, Phase 7-8.
- **2026-10-13, Phase 4a:** observable instruments (`Server/ServerObservableMetrics.cs`; read only when a listener collects, no request-path work, a faulting read yields no measurement). Residency from `ModelRuntimeManager.GetStats()`: `opentail.models.resident|pending_loads|estimated_bytes`, `opentail.admission.queued`, counters `opentail.models.loads|load_failures|evictions|admission_rejections`. Engine (DI singleton, same limitation as `/metrics` in multi-model mode): `opentail.engine.queue.depth`, `.sequences.active`, and, when the engine is the continuous-batching one, `.kv_cache.committed_tokens`, `.prefix_cache.entries|estimated_bytes|hits|misses|evictions`. Bytes are named estimates. Server Fast 474 passed. Still open in Phase 4: speculative-decoding counters (definitions per mode first), KV budget/usage ratio, planner decisions (backend/fallback reason/GPU layer share), Vulkan aggregates.
