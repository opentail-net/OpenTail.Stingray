# Completed items split from 2026-10-12-opentelemetry-observability-plan.md (2026-10-31)

Finished items moved verbatim out of [../3-product-and-runtime/2026-10-12-opentelemetry-observability-plan.md](../3-product-and-runtime/2026-10-12-opentelemetry-observability-plan.md). Open items stay there. Each block is preceded by the heading it sat under. Relative links inside the blocks were rebased for this folder.

> from: ## 12. Acceptance checklist
- [x] A telemetry or listener failure cannot break inference (test). **Done 2026-10-31:** `EngineTelemetryFailureTests` (5) and `ServerTelemetryFailureTests` (2) install a `MeterListener` and an `ActivityListener` that throw on every event and require the output to equal an unobserved run (both engines, concurrent requests, two requests in a row on the batcher, model load, server request timing, operation, audio and load-wait recording). They found real defects, now fixed: a throwing `ActivityListener` failed `InferenceEngine` generation and model load (`Activity.Start`/`Dispose` run listener callbacks inline) and the server request path (`ServingRequestTiming`, and every raw metric record in `ServerMetrics`). Each test failed before its fix. The telemetry self-test span (`RunSelfTestSpan`) was found unguarded in the next review and is guarded too (`SelfTestSpan_ThrowingActivityListener_DoesNotThrow`). Not covered: observable-gauge callbacks and the OTLP exporter path.

> from: ### Open items
- [x] Plan header corrected on 2026-10-30 to match its progress log

> from: ### Open items
- [x] Acceptance: a telemetry or listener failure cannot break inference (line 293): done 2026-10-31, see that line

