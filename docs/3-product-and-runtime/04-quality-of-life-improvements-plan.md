> **Reprioritized 2026-08-08 — now runway position 4.** Item 1 is now DONE (2026-08-15): both inventories are name-complete against source AND fully classified row-by-row, not just via a partial summary register. See [00-current-work.md](../00-current-work.md) Priority 3.
>
> Completed items moved to [done/04-quality-of-life-improvements-plan-completed-items-2026-10.md](../done/04-quality-of-life-improvements-plan-completed-items-2026-10.md) on 2026-10-30. This file keeps only open work.

# Quality-of-life improvements — current work

**Thread:** [configuration and sessions](04-quality-of-life-improvements-plan.md).

Overlap (ported from the retired workstream, 2026-10-31): item 4 (server observability) is tracked in the [OpenTelemetry plan](2026-10-12-opentelemetry-observability-plan.md). Item 6 depends on the outcome of [plan 010](010-forward-pass-context-isolation-for-session-forking-plan.md).

**Status:** planning/diagnostics baseline shipped; deterministic static-plan decision fixtures
and typed host-key classification are now in place. Environment ownership and execution parity remain.

## Remaining work

- [ ] 2. Extend effective configuration from static plan inputs to real server/loader startup values.
   Server environment precedence is centralized in `ServerEnvironmentOverrides`; `/status`
   publishes its non-sensitive applied-variable receipt and a `configuration.bound` snapshot
   after that precedence has been applied (admission, CPU/prefill/cache budgets, TurboQuant/KV,
   tool grammar, and session persistence as a boolean — never filesystem paths). Remaining work
   `configuration.resolved` now also records the concrete built-in loader route (actual backend,
   forward-pass family, model format, and resolved context), so a requested `auto`/hybrid setup is
   no longer mistaken for the route that actually loaded. Remaining work is actual device/cache
   allocation telemetry, without exposing values such as filesystem paths. `/capabilities` now
   also states the restart-session verdict explicitly rather than leaving session support ambiguous.
- [ ] 3. Extend the golden capability fixtures only where a loader actually executes the route. Static
   CPU/backend/dtype/batching/speculation decisions are covered; hardware rows remain release gates.
- [ ] 4. Finish server observability: VRAM/RAM breakdown, streaming timing, and per-request cache signal.
   The opt-in non-streaming timing extension now covers OpenAI, Anthropic, and Responses;
   it deliberately remains absent from streaming event contracts.
- [ ] 5. Add model/API compatibility corpus and protocol smoke coverage. The small llama-server
   compatibility surface (`/tokenize`, `/detokenize`, `/props`) now has fake-engine wire-contract
   coverage, including malformed-payload errors; expand only with a named consumer contract.
- [ ] 6. Keep sessions out of the product configuration surface until the proven CPU-dense restart lane
   has a named lifecycle and capability gate. The proof itself is now covered by the real-GGUF,
   cross-process replay acceptance test; it is not yet a product contract.

Inputs: [env-var-inventory.md](../reference/env-var-inventory.md), [reference/host-config-inventory.md](../reference/host-config-inventory.md),
and [reference/eligibility-check-inventory.md](../reference/eligibility-check-inventory.md).

Historical implementation record: [done/quality-of-life-improvements-plan-2026-07.md](../done/quality-of-life-improvements-plan-2026-07.md).

---

## Research findings (2026-10-31)

- **Formatting defect (fix this first).** Lines 1 to 5 are a broken blockquote. The sentence "Item 1 is now DONE (2026-08-15): both" stops on line 1 and resumes on line 4 after a second blockquote, so the status is hard to read. Merge it into one status line.
- **Items 2 to 6** were not re-verified in code in this pass. The `configuration.bound` and `configuration.resolved` snapshot names are the doc's own claims.
- **Web research:** not applicable. These items are internal configuration and observability work, and the external telemetry references are in [2026-10-12-opentelemetry-observability-plan.md](2026-10-12-opentelemetry-observability-plan.md).

**Next check (done 2026-11-01, plan 5 E1):** the names are real. `/status` carries `configuration.bound` and `configuration.resolved` (`ServerStatusConfiguration.Bound` and `.Resolved` in `ServerStatusSnapshot.cs`, serialised snake_case). They were not found by grepping for the dotted names, because the code has them as record members; the map below was read from the code.

---

## Item 2 map: where each effective-configuration value comes from (2026-11-01, plan 5 E1)

Two different things carry the name "effective configuration", and the item mixes them:

- **The CLI's `EffectiveConfigurationSnapshot`** (`Engine/EffectiveConfiguration.cs`; `stingray plan` and `StaticPlanCommand`): a dictionary of values, each with a `Source` (CLI flag, profile, environment). It is built from CLI settings, a profile and the environment, **before** any model loads. The server does not build or publish one.
- **The server's `/status` `configuration` object** (`ServerStatusSnapshot.cs`): `bound` is the server options after host binding and the legacy environment overrides (`ServerEnvironmentOverrides`), with no per-value source; `resolved` is the receipt of the route the built-in loader actually took.

Row by row, read from `ServerStatusSnapshot.Create`, `InferenceEngineLoader.DescribeRuntime` and `ExecutionPlanner`:

| Value in `/status` | Where it comes from | Real startup value? | What the plan resolves that `/status` does not publish |
|---|---|---|---|
| `resolved.backend`, `resolved.forward_pass` | `DescribeRuntime(forwardPass)`: the class of the forward pass that was built | **Yes** | none |
| `resolved.model_format`, `resolved.context_size` | the loaded format and `forwardPass.MaxSeqLen` | **Yes** | none |
| `resolved.gpu_layers`, `placement.gpu_layers` | the layers the plan placed on the GPU (the receipt) | **Yes**, once a built-in load has happened | before a load, or with a custom engine factory, it is `options.NGpuLayers` |
| `placement.backend`, `placement.context_size` | the resolved value when present, else the configured option | **Yes** after a load, **configured** before | none |
| `bound.cpu_threads` | `options.CpuThreads`, where 0 means automatic | **No: configured** | the planner resolves the count (`request.ThreadCount > 0 ? ... : capabilities.RecommendedThreadCount`, `ExecutionPlanner.cs:131`); the number actually used is not published |
| `bound.kv_type` | `options.KvType` as configured | **No: configured** | `ExecutionPlan.KvDtype` (`ResolveKvDtype`, `ExecutionPlanner.cs:48`) is the dtype that was planned; not published |
| `placement.spec_type` | `options.SpecType` | **No: configured** | the plan''s `Speculation.Mode` can differ (DSpark enabled by placement, or off); not published |
| `placement.max_batch_size`, `placement.continuous_batching` | `options.MaxBatchSize`, or a live engine reporting continuous batching | partly: the batching flag also reads the live engine | the plan''s `Batching` section; not published |
| `bound.turbo_quant`, `bound.prefill_chunk_tokens`, `bound.kv_budget_mb`, `bound.prefix_cache_mb`, `bound.prefill_dequant_cache_mb`, admission limits, session flags | options after environment precedence | **Configured**, which is what these are | the live values do appear elsewhere: `batching.prefill_chunk_tokens`, `batching.kv_token_budget` and `cache.capacity_bytes` come from the engine |
| `configuration.environment_overrides` | names of the variables that changed an option | **Yes** (names only, no values) | none |
| `memory.device_vram_bytes`, `memory.device_vram_free_bytes` | always `null`, by design: the server boundary cannot see device allocations | **No** | `ExecutionPlan.EstimatedVramMb` and `EstimatedRamMb` are estimates, not allocations |

What remains for item 2, in the order it would be done: publish the plan''s resolved thread count, KV dtype and speculation mode next to the configured ones (the plan object is in `InferenceEngineLoader` where `ServerRuntimeResolution` is built); then give `bound` per-value sources the way the CLI snapshot does; the device and cache allocation telemetry stays blocked on an engine placement-memory surface, as `ServerStatusSnapshot` itself says.

## Thread state (from the retired workstreams folder, 2026-10-31: configuration-and-sessions.md)

### Correct state

- **Configuration (plan 04):** item 1 is done (the inventories are name-complete and classified row by row, dated 2026-08-15). Items 2 to 6 are open. The plan's status line says the planning and diagnostics baseline shipped.
- **Session forking (plan 010):** planning only, as its status section says. Checked 2026-10-30: `IForwardPass.CreateContext()` still returns `this` (`src/OpenTail.Stingray.Core/IForwardPass.cs:14`), but nothing in `src/` calls it (checked 2026-10-31): sessions fork through `HotSessionRuntime.Fork` and the ref-counted KV cache, not through this method, so plan 010 as written is obsolete and its fate is an owner decision. The guard the plan cites, `ThrowIfForwardPassNotIsolated` in `InferenceSession.cs`, is not in the repository, so that reference is stale.
- **Overlap:** plan 04's item 4 (server observability) is covered by the [observability plan](2026-10-12-opentelemetry-observability-plan.md). Its item 6 (sessions out of the product configuration) depends on plan 010's outcome.

### Open items

- [ ] 04 item 2: extend effective configuration from static plan inputs to real server and loader startup values
- [ ] 04 item 3: extend the golden capability fixtures only where a loader actually executes the route
- [ ] 04 item 4: server observability (VRAM and RAM breakdown, streaming timing, per-request cache signal). Covered by the observability plan
- [ ] 04 item 5: model and API compatibility corpus, with protocol smoke coverage
- [ ] 04 item 6: keep sessions out of the product configuration surface until the proven CPU-dense restart lane is proven

### Cadence

- Review when a configuration or session change lands, and when the observability plan closes.
- Re-read at least every two weeks while any item is open.
- Close when plan 04 items 2 to 6 are each done or dropped, and plan 010 is either implemented or written off.
