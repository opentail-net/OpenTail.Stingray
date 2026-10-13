# Telemetry instruments (the checked-in mapping table)

Every `Meter` instrument Stingray publishes is a row here, and tests fail if the code and this table disagree
(`EngineTelemetryContractTests`, `ServerTelemetryContractTests`). Adding a metric is a declaration, a row and a test.
Guide for users: [observability.md](../guides/observability.md). Plan and decisions:
[the plan](../3-product-and-runtime/2026-10-12-opentelemetry-observability-plan.md).

**Conventions checked against:** OpenTelemetry GenAI conventions, repository `open-telemetry/semantic-conventions-genai`
(no release or tag exists; every metric there is marked *Development*; checked 2026-10-12, files
`docs/gen-ai/gen-ai-metrics.md` and `gen-ai-token-metrics.md`). A later convention change must not silently change what a
Stingray metric means: this table is the contract. *Standard* = name and unit taken from the convention;
*Stingray* = our own, no standard equivalent.

**Cost tiers** (principle 11 of the plan): **1** always on and inside the overhead budget; **2** costs something only when a
listener is attached; **3** explicit diagnostic mode, off by default, perturbs what it measures and must never feed a
benchmark comparison. Observable instruments are read only when a listener collects.

**Allowed attributes** (closed or bounded values only): `gen_ai.operation.name`, `gen_ai.request.model` (server-resolved id;
path-like or over-long values become `other`), `opentail.endpoint` (fixed names), `error.type` (exception type name, only on failed non-chat operations), `outcome` (success, error, cancelled,
incomplete, client_error), `opentail.forward_pass` (class name or `cli`/`server`/`continuous_batching`), `opentail.architecture`,
`opentail.backend` (auto/cpu/cuda/vulkan/unknown), `opentail.backend.requested`, `opentail.fallback.reason`
(unavailable/other), `opentail.reason` (overload), `opentail.speculative.mode` (mtp), `opentail.profile.phase` (fixed ids), `opentail.diffusion.scheduler` (fixed names, currently `euler_flow`).
Never: prompt or completion text, tool data, image or audio bytes, request/session/user ids, file paths, exception messages.

## Instruments

| Meter | Name | Type | Unit | Tier | Standard or Stingray | Attributes | Definition |
|---|---|---|---|---|---|---|---|
| `OpenTail.Stingray.Server` | `gen_ai.server.request.duration` | histogram | s | 1 | Standard (Development) | operation, model, endpoint, outcome | Start of serving (after parsing and template rendering) to response complete; non-chat endpoints use their own start/finish around the handler |
| `OpenTail.Stingray.Server` | `gen_ai.server.time_to_first_token` | histogram | s | 1 | Standard (Development) | operation, model, endpoint | Start of serving to first generated text or reasoning token; a cold model load happens before this timer starts |
| `OpenTail.Stingray.Server` | `gen_ai.server.time_per_output_token` | histogram | s | 1 | Standard (Development) | operation, model, endpoint | Decode time (first token to generation end, delivery excluded) over (output tokens - 1); one sample per request |
| `OpenTail.Stingray.Server` | `opentail.queue.wait` | histogram | s | 1 | Stingray | operation, model, endpoint | Start of serving to engine admission |
| `OpenTail.Stingray.Server` | `opentail.requests.active` | up-down counter | {request} | 1 | Stingray | endpoint | Generation requests in flight; decremented only if the increment was recorded |
| `OpenTail.Stingray.Server` | `opentail.requests.rejected` | counter | {request} | 1 | Stingray | reason | Rejected before admission |
| `OpenTail.Stingray.Server` | `opentail.audio.duration` | histogram | s | 1 | Stingray | operation, endpoint | Audio seconds produced (speech) or consumed (transcription, translation) by a successful request |
| `OpenTail.Stingray.Server` | `opentail.audio.realtime_factor` | histogram | 1 | 1 | Stingray | operation, endpoint | Operation duration over audio duration for a successful request (below 1 is faster than real time); for transcription it includes reading the upload |
| `OpenTail.Stingray.Server` | `opentailllm.requests` | counter | requests | 1 | Legacy (kept one release) | none | Admitted inference requests |
| `OpenTail.Stingray.Server` | `opentailllm.tokens` | counter | tokens | 1 | Legacy (kept one release) | none | Tokens generated |
| `OpenTail.Stingray.Server` | `opentailllm.overload_rejections` | counter | requests | 1 | Legacy (kept one release) | none | Requests rejected because the bounded queue was full |
| `OpenTail.Stingray.Server` | `opentailllm.ttft_ms` | histogram | ms | 1 | Legacy (kept one release) | none | Time to first token |
| `OpenTail.Stingray.Server` | `opentailllm.itl_ms` | histogram | ms | 1 | Legacy (kept one release) | none | **Mislabelled:** records total generation duration per request, not inter-token latency; use `gen_ai.server.time_per_output_token` |
| `OpenTail.Stingray.Server` | `opentail.models.resident` | observable gauge | {model} | 1 | Stingray | none | Models currently resident |
| `OpenTail.Stingray.Server` | `opentail.models.pending_loads` | observable gauge | {load} | 1 | Stingray | none | Model loads in flight |
| `OpenTail.Stingray.Server` | `opentail.models.estimated_bytes` | observable gauge | By | 1 | Stingray | none | Estimated bytes of resident weights (an estimate, not measured memory) |
| `OpenTail.Stingray.Server` | `opentail.admission.queued` | observable gauge | {request} | 1 | Stingray | none | Requests waiting for model admission |
| `OpenTail.Stingray.Server` | `opentail.models.loads` | observable counter | {load} | 1 | Stingray | none | Model loads completed |
| `OpenTail.Stingray.Server` | `opentail.models.load_failures` | observable counter | {load} | 1 | Stingray | none | Model loads that failed |
| `OpenTail.Stingray.Server` | `opentail.models.evictions` | observable counter | {model} | 1 | Stingray | none | Models evicted |
| `OpenTail.Stingray.Server` | `opentail.models.admission_rejections` | observable counter | {request} | 1 | Stingray | none | Requests rejected by model admission |
| `OpenTail.Stingray.Server` | `opentail.engine.queue.depth` | observable gauge | {request} | 1 | Stingray | none | Requests waiting to start generation (DI singleton engine) |
| `OpenTail.Stingray.Server` | `opentail.engine.sequences.active` | observable gauge | {request} | 1 | Stingray | none | Requests currently generating (DI singleton engine) |
| `OpenTail.Stingray.Server` | `opentail.engine.kv_cache.committed_tokens` | observable gauge | {token} | 1 | Stingray | none | KV tokens reserved by admitted requests (continuous batching only) |
| `OpenTail.Stingray.Server` | `opentail.engine.kv_cache.budget_tokens` | observable gauge | {token} | 1 | Stingray | none | KV-token admission budget; long.MaxValue means unlimited |
| `OpenTail.Stingray.Server` | `opentail.engine.prefix_cache.entries` | observable gauge | {entry} | 1 | Stingray | none | Prefix snapshots retained |
| `OpenTail.Stingray.Server` | `opentail.engine.prefix_cache.estimated_bytes` | observable gauge | By | 1 | Stingray | none | Engine-accounted bytes of retained snapshots (an estimate) |
| `OpenTail.Stingray.Server` | `opentail.engine.prefix_cache.hits` | observable counter | {lookup} | 1 | Stingray | none | Lookups that found a reusable prefix |
| `OpenTail.Stingray.Server` | `opentail.engine.prefix_cache.misses` | observable counter | {lookup} | 1 | Stingray | none | Lookups with no reusable prefix |
| `OpenTail.Stingray.Server` | `opentail.engine.prefix_cache.evictions` | observable counter | {entry} | 1 | Stingray | none | Snapshots evicted to honour capacity |
| `OpenTail.Stingray.Engine` | `opentail.engine.requests` | counter | {request} | 1 | Stingray | forward_pass, outcome | Generation requests handled by the engine |
| `OpenTail.Stingray.Engine` | `opentail.engine.prefill.duration` | histogram | s | 1 | Stingray | forward_pass, outcome | Prompt processing time for the new (not reused) prompt tokens; excludes queueing and tokenisation |
| `OpenTail.Stingray.Engine` | `opentail.engine.prefill.tokens` | histogram | {token} | 1 | Stingray | forward_pass, outcome | New prompt tokens processed (reused prefix tokens excluded) |
| `OpenTail.Stingray.Engine` | `opentail.engine.decode.duration` | histogram | s | 1 | Stingray | forward_pass, outcome | First generated token to end of generation |
| `OpenTail.Stingray.Engine` | `opentail.engine.decode.tokens` | histogram | {token} | 1 | Stingray | forward_pass, outcome | Tokens generated (the stop token is not counted) |
| `OpenTail.Stingray.Engine` | `opentail.engine.first_token.duration` | histogram | s | 1 | Stingray | forward_pass, outcome | Engine request start to first generated token |
| `OpenTail.Stingray.Engine` | `opentail.engine.generation.duration` | histogram | s | 1 | Stingray | forward_pass, outcome | Engine request start to generation complete |
| `OpenTail.Stingray.Engine` | `opentail.engine.model.load.duration` | histogram | s | 1 | Stingray | architecture, forward_pass, backend, outcome | Time to build a runtime for a model |
| `OpenTail.Stingray.Engine` | `opentail.engine.placement.gpu_layer_share` | histogram | 1 | 1 | Stingray | backend | GPU layers over total layers at load (selecting a GPU does not mean the whole model runs on it) |
| `OpenTail.Stingray.Engine` | `opentail.engine.backend.fallbacks` | counter | {load} | 1 | Stingray | backend, backend.requested, fallback.reason | An explicitly requested backend was not used |
| `OpenTail.Stingray.Engine` | `opentail.engine.speculative.drafts` | counter | {step} | 2 | Stingray | speculative.mode | Speculative verify steps (MTP only; other modes not exported until defined) |
| `OpenTail.Stingray.Engine` | `opentail.engine.speculative.draft_tokens` | counter | {token} | 2 | Stingray | speculative.mode | Draft tokens proposed to the verifier in a step |
| `OpenTail.Stingray.Engine` | `opentail.engine.speculative.accepted_tokens` | counter | {token} | 2 | Stingray | speculative.mode | Draft tokens the verifier accepted (its own corrected token not counted); acceptance = rate(accepted)/rate(draft) |
| `OpenTail.Stingray.Engine` | `opentail.profile.phase.duration` | observable counter | s | 3 | Stingray | profile.phase | DIAGNOSTIC, perturbed: seconds per profiled phase; exists only when `STINGRAY_MOE_PHASE_TIMING=1` or `STINGRAY_PROFILE_GPU_SPLIT=1` |
| `OpenTail.Stingray.Engine` | `opentail.profile.vulkan.dispatches` | observable counter | {submit} | 3 | Stingray | none | DIAGNOSTIC: Vulkan queue submits since the last profile reset; exists only when `STINGRAY_PROFILE_GPU_SPLIT=1` |
| `OpenTail.Stingray.Diffusion` | `opentail.diffusion.steps` | counter | {step} | 1 | Stingray | scheduler, outcome | Denoising steps run (a failed loop counts the steps completed before the failure) |
| `OpenTail.Stingray.Diffusion` | `opentail.diffusion.step.duration` | histogram | s | 1 | Stingray | scheduler | Wall time of one denoising step (one model forward plus the update); one sample per step |
| `OpenTail.Stingray.Diffusion` | `opentail.diffusion.denoise.duration` | histogram | s | 1 | Stingray | scheduler, outcome | Wall time of one whole denoising loop |

Rows whose tier is 3 are *conditional*: the contract tests do not require them in a normal run, but fail if one appears that
is not listed.

## Spans

| Source | Span | Parent | Boundaries and tags |
|---|---|---|---|
| ASP.NET Core instrumentation (host) | `POST <route>` | caller (W3C `traceparent`) | whole HTTP request |
| `OpenTail.Stingray.Server` | `chat <model>` | the HTTP span | same boundaries as `gen_ai.server.request.duration`; tags `gen_ai.operation.name`, `gen_ai.request.model`, `opentail.endpoint`, `outcome`, `gen_ai.usage.output_tokens`, `gen_ai.usage.input_tokens`, `opentail.reused_tokens`, `opentail.time_to_first_token_ms` |
| `OpenTail.Stingray.Engine` | `opentail.engine.generate` | the caller's captured context (explicit hand-off across the engine thread) | engine request; tags `outcome`, `opentail.prompt_tokens`, `opentail.reused_tokens`, `opentail.output_tokens` |
| `OpenTail.Stingray.Engine` | `model.load` | none (own trace; a load is shared, never a child of one request) | tags `opentail.architecture`, `opentail.forward_pass`, `opentail.backend`, `outcome` |

## Timing contract

Two correct-looking metrics must never describe different intervals.

| Measurement | Starts | Ends | Notes |
|---|---|---|---|
| HTTP duration | request arrives | response complete | owned by ASP.NET Core's instrumentation, not Stingray |
| Server operation duration | `BeginServingRequest()` (after parsing and template rendering) | response complete, incl. serialisation and streaming | excludes parsing and rendering |
| Queue wait | start of serving | `MarkAdmitted()` | admission wait only |
| Time to first token | start of serving | first generated text or reasoning token | includes queue wait; **excludes a cold model load**, which `AcquireAsync` completes before serving starts (the load is recorded as `model.load`) |
| Prefill | start of prompt processing in the engine | end of prefill | new prompt tokens only; excludes queueing and tokenisation |
| Decode | first generated token | last generated token | the first token belongs to time to first token; time per output token divides by (tokens - 1) |
| Engine generation | engine request start | generation complete | a defined interval, not assumed equal to HTTP duration |

Cancelled or disconnected: `outcome=cancelled`. Generation end excludes delivery. Throughput is never an average of
per-request rates; derive it as `sum(tokens)/sum(duration)`.
