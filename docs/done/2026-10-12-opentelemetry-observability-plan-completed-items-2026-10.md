# Completed items split from 2026-10-12-opentelemetry-observability-plan.md (2026-10-31)

Finished items moved verbatim out of [../3-product-and-runtime/2026-10-12-opentelemetry-observability-plan.md](../3-product-and-runtime/2026-10-12-opentelemetry-observability-plan.md). Open items stay there. Each block is preceded by the heading it sat under. Relative links inside the blocks were rebased for this folder.

> from: ## 12. Acceptance checklist
- [x] A telemetry or listener failure cannot break inference (test). **Done 2026-10-31:** `EngineTelemetryFailureTests` (5) and `ServerTelemetryFailureTests` (2) install a `MeterListener` and an `ActivityListener` that throw on every event and require the output to equal an unobserved run (both engines, concurrent requests, two requests in a row on the batcher, model load, server request timing, operation, audio and load-wait recording). They found real defects, now fixed: a throwing `ActivityListener` failed `InferenceEngine` generation and model load (`Activity.Start`/`Dispose` run listener callbacks inline) and the server request path (`ServingRequestTiming`, and every raw metric record in `ServerMetrics`). Each test failed before its fix. The telemetry self-test span (`RunSelfTestSpan`) was found unguarded in the next review and is guarded too (`SelfTestSpan_ThrowingActivityListener_DoesNotThrow`). Not covered: observable-gauge callbacks and the OTLP exporter path.

> from: ### Open items
- [x] Plan header corrected on 2026-10-30 to match its progress log

> from: ### Open items
- [x] Acceptance: a telemetry or listener failure cannot break inference (line 293): done 2026-10-31, see that line


---

## More completed items moved from 2026-10-12-opentelemetry-observability-plan.md on 2026-11-01

Finished items moved verbatim out of [../3-product-and-runtime/2026-10-12-opentelemetry-observability-plan.md](../3-product-and-runtime/2026-10-12-opentelemetry-observability-plan.md). Open items stay there. Each block is preceded by the heading it sat under. Relative links inside the blocks were rebased for this folder.

> from: ## 12. Acceptance checklist
- [x] *(verified 2026-11-01: only `OpenTail.Stingray.Server.Host.csproj` references an OpenTelemetry package; `LibraryTelemetryDependencyTests` fails if any other source project does)* The library emits data with no OpenTelemetry package referenced.

> from: ## 12. Acceptance checklist
- [x] *(verified 2026-11-01: the CI jobs `NativeAOT telemetry self-test` publish the real CLI for linux-x64 and, since 2026-11-01, win-x64 and run `telemetry-selftest`, which fails unless the five engine instruments and a span reach an in-process listener; the server instruments are not in the CLI, so they are not part of this check)* The published NativeAOT CLI, run with a listener attached, shows the expected instruments and activities.

> from: ### Open items
- [x] *(verified 2026-11-01: only `OpenTail.Stingray.Server.Host.csproj` references an OpenTelemetry package; `LibraryTelemetryDependencyTests` fails if any other source project does)* The library emits data with no OpenTelemetry package referenced (line 296)

> from: ### Open items
- [x] *(verified 2026-11-01: the CI jobs `NativeAOT telemetry self-test` publish the real CLI for linux-x64 and, since 2026-11-01, win-x64 and run `telemetry-selftest`, which fails unless the five engine instruments and a span reach an in-process listener; the server instruments are not in the CLI, so they are not part of this check)* The published NativeAOT CLI, with a listener attached, shows the expected instruments and activities (line 297)
