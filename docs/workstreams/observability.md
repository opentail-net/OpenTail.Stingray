# Workstream: OpenTelemetry observability

> This file summarises its owner plan. If the two disagree, the owner plan is authoritative; correct this file.

**Owner:** [../3-product-and-runtime/2026-10-12-opentelemetry-observability-plan.md](../3-product-and-runtime/2026-10-12-opentelemetry-observability-plan.md). **Guide:** [../guides/observability.md](../guides/observability.md). **Last verified:** 2026-10-30.

## Correct state

- **The plan's header was corrected on 2026-10-30** (it had said "Not started"). Its own progress log records work landed on 2026-10-13: Phase 2 (instruments, the `gen_ai.server.*` names, outcome tags, the per-request stats record, the agreement test), Phase 3a and 3b (server and model-load traces), Phase 4a (observable instruments in `Server/ServerObservableMetrics.cs`), the first cut of Phases 5 and 6 (`Server.Host` wiring, only when `OTEL_*` is set), and the CLI bridge (`Engine/EngineTelemetryBridge.cs`).
- The owner plan is authoritative for status.
- **Overhead A/B** on a real model (2026-10-13, SmolLM2-360M, interleaved): recorded in the plan's progress log.
- **Still open:** the acceptance criteria at the end of the plan (lines 289-302). Most are tests or checks that the log does not record as run. In particular: the NativeAOT CLI with a listener attached, the clean-machine run of the guide, and the cost tiers for each instrument.

## Open items

- [x] Plan header corrected on 2026-10-30 to match its progress log
- [ ] Acceptance: every instrument's name, unit and attributes asserted by tests (plan line 289)
- [ ] Acceptance: measurement boundaries (streaming, cancellation, cold model load) covered by tests (line 290)
- [ ] Acceptance: no attribute outside the §7 allow-list, and no content in telemetry (test, line 291)
- [ ] Acceptance: concurrent requests do not cross-contaminate trace context or measurements (line 292)
- [ ] Acceptance: a telemetry or listener failure cannot break inference (line 293)
- [ ] Time per output token is a true per-token figure; the old `itl_ms` meaning documented as changed (line 294)
- [ ] One chat request yields one connected trace; a shared model load is a separate operation with linked waits (line 295)
- [ ] The library emits data with no OpenTelemetry package referenced (line 296)
- [ ] The published NativeAOT CLI, with a listener attached, shows the expected instruments and activities (line 297)
- [ ] `/metrics`, the `Meter` instruments and `stingray status` agree on counts and latency (line 298)
- [ ] Overhead reported with no listener, in-process listener and OTLP exporter, within agreed thresholds (line 299)
- [ ] `docs/guides/observability.md` run on a clean machine (line 300)
- [ ] Every instrument has a cost tier in the mapping table; only tier 1 does work with no listener (line 301)
- [ ] The growth seams in §14 exist (line 302)

## Cadence

- Review after each observability change, and whenever the NativeAOT publish changes.
- Re-read at least every two weeks while any acceptance item is open.
- Close when every acceptance item is either tested or explicitly dropped in the plan.
