# OpenTail.Stingray quality-of-life improvements — product and implementation plan

**Status:** proposal for review  
**Owner:** unassigned  
**Drafted:** 2026-07-26  
**Scope:** user-facing setup, planning, diagnostics, configuration, model lifecycle,
compatibility validation, observability, sessions, and local auto-tuning  
**Out of scope:** inference-kernel optimization unless required to make a selected plan
correct or observable

---

## 0. Progress

Legend: `[ ]` not started · `[~]` partially done · `[x]` complete.

| Item | State | Evidence |
|---|---|---|
| Phase 0 · 1 — inventory settings | `[~]` | All three surfaces enumerated: `env-var-inventory.md` (**141** `STINGRAY_*`), `cli-option-inventory.md` (**94** `[CommandOption]` declarations), `host-config-inventory.md` (**41** bindable keys). **276 settings total.** Enumeration is done; every Class column is still unfilled, and classification is the blocker — it needs the behaviour owner, not name-inference. |
| Phase 0 · 2 — eligibility checks + owners | `[~]` | `docs/eligibility-check-inventory.md` — **14** feature gates located with file:line and what each gates. Name-matched and deliberately conservative, so it is a FLOOR, not a census; the "current owners" half is unanswered because these predicates live in constructors, static initialisers and kernel dispatch rather than anywhere a user can query. |
| Phase 0 · 3-5 | `[~]` | Shared immutable planning contracts and stable decision codes now exist in Engine and `plan --json` emits the contract. Golden fixtures and recorded startup selections still depend on loader consumption. |
| **Mechanical work: COMPLETE** | — | The small items in this plan are done. What remains is either multi-day (see below) or blocked on classifying the 276 enumerated settings, which needs the behaviour owner. The working loop stopped here rather than manufacturing work. |
| Phase 1 · 3 — effective config | `[~]` | `list-env` prints active `STINGRAY_*` settings with values, flags unknown names, and redacts credentials. `plan --json` now adds a typed, source-tracked static snapshot for its planning knobs (`cli > profile > environment > default`), including conflicts, invalid environment fallbacks, ignored and inapplicable settings. It is **not yet** the server host's effective configuration or a complete stable-setting inventory. |
| Phase 1 · 1-3, 7-8 — inspect, static plan, eligibility, JSON | `[~]` | `inspect` and `plan` are now distinct read-only commands. Both read the GGUF index without loading weights and report compatibility, CPU/CUDA/Vulkan discovery and feature eligibility; only `plan` calculates TierPlanner placement. `plan --json` includes source-tracked effective planning configuration and an immutable Engine `ExecutionPlan` with stable decision codes. It is not yet consumed by a loader, has no golden fixture matrix, and does not cover every feature family. |
| Phase 1 · 4 — doctor | `[x]` | Runtime identity, **CPU instruction sets** (AVX2+FMA absence is a blocking error), CUDA/Vulkan discovery, **configuration conflicts** with suggestions, **filesystem free space**, structural GGUF validation, **actionable remediation text** (warnings/errors only), text + JSON, deterministic exit status, `--bundle`, and **`--deep` mode** (64 MiB host allocation smoke test + Vulkan backend initialization smoke test). **Test-covered** (`DoctorCommandTests` 6 + `DoctorBundleTests` 3). |
| Phase 1 · 5 — ExecutionPlan loader & run --auto | `[x]` | `ExecutionPlan` & `ExecutionPlanBuilder` (`OpenTail.Stingray.Engine`), `InferenceEngineLoader.LoadFromPlan` (`OpenTail.Stingray.Server`), `run --auto`, `--goal`, `--explain` compact startup summary (`RunCommand.cs`). **Test-covered** (`ExecutionPlanTests`). |
| Phase 2 · 4 — support bundle | `[x]` | `doctor --bundle x.zip`. Allowlist of three entries (`doctor.json`, `settings.txt`, `manifest.txt`); setting NAMES only, values never written; no prompts, generated text, token IDs, credentials, model bytes or absolute paths; manifest declares its own exclusions; local file, no upload. **Test-covered** (`DoctorBundleTests`, 3 cases incl. a planted secret asserted absent from every entry). |
| Phase 2 · 1-3 — status snapshot | `[~]` | Versioned `GET /status` document (`ServerStatusSnapshot.cs`, schema v1) added alongside `/health`/`/metrics`: placement, traffic, TTFT/inter-token/queue/request latency (from the same bounded histograms as `/metrics`, exposed as count/mean/p50/p95/p99), prefix-cache hit/miss/reuse, host working-set/managed-heap memory, continuous-batching KV occupancy, and derived operator warnings (saturated admission, prefix-cache-unavailable, KV near budget). Unlimited (`long.MaxValue`) budgets serialize as absent, never as the literal sentinel. `status [--url <URL>] [--watch] [--json]` (`StatusCommand.cs`) now prefers `/status` and falls back to the `/health`+`/metrics` scrape on older/partial hosts; unknown fields render as omitted rows, never as a false zero. Opt-in per-response `opentail_timing` extension (TTFT + total ms) added to non-streaming OpenAI/Anthropic responses (`ResponseTimingExtension.cs`). **Test-covered** (`StatusDocumentTests` 10 cases, `ResponseTimingExtensionTests` 5 cases, `DiagnosticSurfaceRedactionTests` 4 cases, `ServingRequestTimingTests` 4 cases, `StatusCommandTests` +5). **Fixed defect**: `_generationDuration`/`GenerationDurationSummary` (and therefore `/status`'s `inter_token_mean_ms` and the `opentail_timing` response extension) was measured from first-token to `ServingRequestTiming.Dispose()` time — which on the non-streaming path fires *after* the full JSON response is serialized and written to the client, so a slow client or a large response inflated the reported decode/inter-token rate with delivery time that had nothing to do with the engine. Added `ServingRequestTiming.MarkGenerationComplete()`, called immediately after each token-producing loop (before any response serialization/delivery) across all four handlers (OpenAI/Anthropic × streaming/non-streaming); `Complete()` keeps its old dispose-time behavior only as a fallback for requests that error/cancel before the loop finishes. Regression-pinned by `ServingRequestTimingTests.MarkGenerationComplete_ExcludesTimeSpentAfterItFromGenerationDuration` (real sleeps distinguish decode time from delivery time deterministically). Remaining for `[x]`: device VRAM/RAM breakdown (needs a live backend memory-query surface at the server boundary, not yet exposed), streaming/Responses-API timing extension, per-request cache-hit signal. |
| Phase 3 · read-only slice | `[~]` | `list-models` enumerates GGUFs on disk with size, and `--deep` opens each index to report arch/version/tensor count, reporting corrupt files as UNREADABLE rather than failing the listing. **Test-covered** (`ListModelsCommandTests`, 4 cases incl. corrupt file, empty dir, missing dir). NOT a model store: no manifest, aliases, downloads, verification or removal. |
| Phase 4 · read-only slice | `[~]` | `show-template` renders a model's chat template against a sample conversation (`--system`, `--no-thinking`, `--raw` for the GGUF source). Verified on the reference model: correct ChatML with system message and generation prompt. **Test-covered** (`ShowTemplateCommandTests`, 3 cases). NOT the compatibility lab: no fixture corpus, tool-call envelopes, streaming order, or SDK smoke tests. |
| Phases 3 (rest), 4 (rest), 5-6 | `[ ]` | **Skipped by size**, not abandoned: model store + downloads, fixture-based compatibility lab, persistent sessions, measured auto-tuning. |

### Shipped alongside, not in the original plan

Both address success metric §11 "silent ignored stable settings: zero", which nothing in the
phase list actually delivered:

- [x] `KnownEnvironmentVariables` (Core) — the 141 names as a `FrozenSet`, with unknown-name
      detection and suffix-scored "did you mean" suggestions.
- [x] Startup misconfiguration warning in **both** entry points (CLI and `Server.Host`). A
      misspelled variable is otherwise indistinguishable from unset, so the run silently ignores
      the user's configuration.
- [x] `KnownEnvironmentVariablesTests.ListMatchesSource` — re-scans `src/` and fails in **both**
      directions (unlisted variable, or listed-but-dead entry). Without it the warning decays into
      noise and trains users to ignore it.

### Skipped by size (NOT abandoned)

Left unchecked and untouched by the working loop because each is multi-day, not because they were
judged unnecessary. They remain the substance of the plan:

- Phase 1 — `ExecutionPlan`, `inspect`, `plan`, `doctor`, `run --auto`
- Phase 2 — mostly done: versioned status snapshot, support bundle, a non-streaming opt-in
  per-response timing extension, and a cross-surface redaction suite all shipped; streaming/
  Responses-API timing, a per-request cache-hit signal, and device VRAM/RAM breakdown remain
- Phase 3 — model lifecycle
- Phase 4 — compatibility lab
- Phase 5 — persistent sessions
- Phase 6 — measured auto-tuning

### Conflict found: §7.3's precedence chain contradicts the shipped code

**§7.3 states** `CLI pin > profile file > host configuration > environment > planner/default`,
placing environment BELOW host configuration.

**The server host does the opposite.** `ServiceCollectionExtensions` binds
`OpenTailStingrayServerOptions` from the `OpenTail.Stingray` configuration section, and then the inline
`AddOpenTailStingray(configuration, opts => ...)` delegate applies **17** `STINGRAY_*` overrides on
top of it. The host's own comment states the consequence plainly: *"Inline configure runs last →
wins."* So today **environment beats host configuration**, for at least `ModelPath`,
`MmprojPath`, `MaxBatchSize`, `MaxQueuedRequests`, `MaxConcurrentRequests`, `PrefillChunkTokens`,
`KvBudgetMb`, `PrefixCacheMb` and the prefill dequant budget.

**Implementing §7.3 as written is therefore a silent behaviour change**, not a clean-up: any
deployment that sets those variables alongside an `appsettings.json` would have its configuration
silently start losing to the file. That is the same failure mode as the six "a fast path existed
but was not taken" bugs in the performance log — behaviour changing under a user with nothing
saying so.

Decide explicitly, and record the decision here:

- [ ] **A.** Keep the shipped order (environment overrides host config) and correct §7.3 to match.
      Environment-last is the conventional container/12-factor expectation, which is an argument
      for this being the RIGHT order rather than an accident.
- [ ] **B.** Adopt §7.3's order and treat the change as breaking: call it out in release notes and
      warn at startup when a host-config key is being overridden by an environment variable.
- [ ] **C.** Keep both orders available behind an explicit, documented setting. Least attractive —
      it makes precedence itself configurable, which is how this became hard to reason about.

**Pinned as of now:** `ConfigurationPrecedenceTests` (Tests.Server, 4 cases) records the CURRENT
behaviour — the inline `configure` delegate wins over the bound configuration section, which is the
mechanism by which environment beats host config. It also pins that an untouched key keeps its
configured value, so a delegate that assigned unconditionally (erasing config for every unset
variable) would fail. The test documents that it records the order rather than endorsing it; if the
order is changed deliberately, the test changes in the same commit.

### Open defects in the in-flight `plan` command (StaticPlanCommand)

Found by review, not yet fixed — the file is another author's in-flight work.

- [x] **The planner read two variables the engine does not.** `STINGRAY_CTX_SIZE` and
      `STINGRAY_SPEC_TYPE` were removed from the static resolver rather than inventing runtime
      compatibility. `Resolve_DoesNotInventEnvironmentInputsThatRunDoesNotRead` pins this.
- [x] **`STINGRAY_TQ=1` / `STINGRAY_TOOL_GRAMMAR=1` parsing.** Resolved by the shared
      typed resolver: `1` and `true` produce Boolean JSON values. Malformed typed environment
      values now retain a structured `invalid` diagnostic and fall back safely, rather than
      aborting `plan`. Covered by `StaticPlanConfigurationTests`.
- [x] **`inspect` was registered as an alias for `plan`.** It is now a distinct read-only command:
      `inspect` emits model identity/capabilities/compatibility and does not calculate placement;
      `plan` retains placement and configuration decisions. Both share the metadata reader and
      common facts, so they do not duplicate GGUF parsing.
- [ ] **Precedence now differs by surface.** The planner resolves `cli > profile > environment >
      default`; the server resolves `environment > host config`. Both defensible, but together
      "does environment beat a file?" has opposite answers depending on which surface you are on.
      This makes the §7.3 decision above more urgent, not less.

### Blocking dependency

**Classification of the 141 variables gates most of Phase 1.** §7.3's precedence chain
(`CLI > profile > host > environment > default`) cannot be specified until it is known which
variables are even in that chain. Until each is marked `stable` / `expert` / `diagnostic` /
`bench` / `experimental`, profiles and effective-config output cannot be correct.

### Two constraints the plan does not mention

- **NativeAOT.** The repo sets `EnableAotAnalyzer` with `TreatWarningsAsErrors`. Every
  serializable plan type needs source-generated registration in `OpenTailStingrayJsonContext`. Cheap if
  designed in, a wall if discovered late.
- **`inspect` overlaps existing commands.** `list-metadata` and `list-tensors` already cover much
  of §6.1 and should be extended rather than joined by a third GGUF reader.

---

## 1. Executive decision

OpenTail.Stingray already has a broad inference surface: CPU, CUDA, and Vulkan backends; hybrid
and MoE placement; several KV-cache formats; speculative decoding; continuous batching;
OpenAI and Anthropic APIs; continuation and prefix caching; multimodal inference; image
generation; and Prometheus metrics.

The next quality-of-life investment should **not** be another collection of independent
flags. It should be an explainable layer that turns user intent into one validated,
reproducible execution plan.

The recommended first product slice is:

```text
opentail inspect <model>
opentail plan <model> --goal balanced --ctx 32k
opentail doctor
opentail run <model> --auto
```

All four commands must use the same planning and validation components. The plan displayed
to the user must be the plan the loader executes.

The longer-term product direction is:

> OpenTail.Stingray should tell the user what will run, why it was selected, what it costs,
> which features are unavailable, and how to reproduce or diagnose the result.

---

## 2. Problem statement

OpenTail.Stingray's capability is increasingly difficult to discover and operate:

- The CLI exposes many low-level options, while the engine also references approximately
  142 distinct `STINGRAY_*` environment variables under `src`.
- Some options are backend-, model-, dtype-, context-, batch-, or placement-dependent.
- An option may be unsupported, ignored, automatically replaced, or harmful to quality.
- Model loading is path-oriented; acquisition, shard verification, and companion artifacts
  remain separate user tasks.
- `/metrics` is useful for monitoring, but answering “why is this request slow?” still
  requires knowledge of internal counters, logs, and execution paths.
- Chat templates, reasoning formats, constrained output, and tool calling create a second
  compatibility matrix independent of basic model loading.
- Performance decisions are often based on static estimates even though this project has
  strong measurement discipline and could calibrate against the actual machine.

The central problem is therefore not missing configurability. It is **missing synthesis**:
the engine knows many relevant facts, but the user has to reconstruct the final decision.

---

## 3. Goals

1. Make the first successful local generation require a model reference and little else.
2. Make every automatic decision inspectable and reproducible.
3. Detect unsupported, conflicting, ignored, or quality-risking configuration before a
   long model load where possible.
4. Let ordinary users choose an intent rather than an implementation detail.
5. Preserve expert control: an explicit low-level setting pins that decision unless it is
   invalid.
6. Give users useful runtime status without requiring a Prometheus installation.
7. Make support reports reproducible, privacy-preserving, and easy to collect.
8. Validate real OpenAI/Anthropic client behavior, not merely endpoint names.
9. Make multi-turn state reuse reliable and visible, especially for hybrid recurrent
   architectures.
10. Eventually produce measured recommendations for the current model and machine.

---

## 4. Non-goals

- Replacing the CLI with a desktop GUI.
- Adding an OpenTail-specific model packaging format in the first release.
- Hiding all advanced controls or removing existing llama.cpp-compatible flags.
- Claiming exact memory or performance predictions when the planner only has an estimate.
- Automatically downloading anything during `inspect` or `doctor` without an explicit
  model reference or opt-in.
- Uploading diagnostics, hardware data, prompts, or model metadata.
- Building a multi-model router before single-model setup and diagnosis are coherent.
- Treating more API endpoints as progress unless a demonstrated client workflow needs them.
- Turning experimental performance flags into stable user configuration by default.

---

## 5. Product principles

### 5.1 Intent first, knobs second

The common path should accept a goal:

| Goal | Primary objective | Allowed trade-offs |
|---|---|---|
| `quality` | Preserve the highest practical numerical and context quality | Lower throughput and higher memory use |
| `balanced` | Safe default for interactive local use | Moderate, disclosed compression or offload |
| `throughput` | Maximise steady-state generation/serving throughput | More memory use and non-bit-identical fast paths if explicitly permitted |
| `long-context` | Maximise usable context within a memory budget | KV compression or more host placement, with quality warnings |
| `low-memory` | Fit inside an explicit RAM/VRAM ceiling | Reduced context, offload, or compression |

Explicit expert settings remain available and override the corresponding automatic decision.

### 5.2 Explain every consequential decision

Every plan decision should carry:

- a stable decision code;
- the selected value;
- the alternatives considered;
- the reason for the selection;
- whether it came from a user pin, profile, environment, default, or auto-planner;
- memory and quality consequences where known;
- a confidence level for estimates.

### 5.3 Plan once, execute that plan

`plan` must not reimplement loader rules for display. Introduce one immutable execution-plan
model consumed by both diagnostics and the real engine loader. Any unavoidable load-time
change must produce a visible plan amendment and reason.

### 5.4 Safe and local by default

Diagnostics stay on the machine. Network access is explicit. A support bundle is reviewed
locally before sharing and excludes prompts, generated text, tokens, credentials, and model
contents by default.

### 5.5 Honest measurement

Auto-tuning must use warmups, repeated samples, reported dispersion, and a separately
validated control. Results are tied to the exact model, build, hardware, driver, and
effective plan. A noisy result is reported as inconclusive rather than promoted.

### 5.6 Preserve the inner loop's microarchitectural budget

For OpenTail's decode and prefill kernels, “no allocations, locks, logging, or I/O” is
necessary but radically insufficient. A source-level change that appears free can consume
registers, ports, loads/stores, arithmetic capacity, branch-prediction budget, or alter
unrolling and register allocation. Any of those changes can materially reduce throughput.

This has already been measured in this project:

- Iteration 57 lost **43%** from adding eight `Vector256<int>` accumulators: no allocation,
  lock, logging, or I/O was involved; the additional live state pushed the loop beyond the
  16 architectural YMM registers.
- Iteration 58 lost **48.6%** from scalar arithmetic interleaved into a vectorised loop.

Therefore the invariant is:

> No QoL feature may add anything to an inference inner loop's register, port, instruction,
> load/store, branch, or dependency budget.

Diagnostics must be collected at request, batch, turn, or process boundaries; rendered from
existing state; or enabled only through a separately measured profiling build/path. They must
not introduce per-token counters, timestamps, conditionals, temporary vectors/scalars, or
instrumentation into the hot loop. Any proposed hot-path change—however harmless it looks in
source—requires an isolated benchmark, representative end-to-end measurement, and the
project's established correctness/parity gates before it can ship.

---

## 6. Target command-line experience

The command name below is written as `opentail`; the final executable name may remain
`opentail-llm-cli` until packaging is decided.

### 6.1 Inspect a model

```text
opentail inspect models/qwen.gguf
opentail inspect hf:Qwen/Qwen3-8B-GGUF:Q4_K_M
opentail inspect models/qwen.gguf --json
```

Expected output:

- GGUF identity, architecture, quantization, parameter count, tensor inventory, and context;
- tokenizer, chat-template, reasoning, tool-use, vision, and MTP capabilities;
- required or optional companion artifacts;
- supported OpenTail backends and unsupported-reason codes;
- corruption, missing tensor, metadata, and shard diagnostics;
- no weight allocation unless a deep validation mode explicitly requests it.

### 6.2 Produce an execution plan

```text
opentail plan models/qwen.gguf --goal balanced --ctx 32768
opentail plan models/qwen.gguf --goal long-context --memory-limit 14GB
opentail plan models/qwen.gguf --for server --concurrency 8
opentail plan models/qwen.gguf --profile workstation.json --json
```

Illustrative output:

```text
Hardware       RTX 4070 Ti 12 GiB; 64 GiB RAM; AVX2
Model          Qwen3 8B Q4_K_M
Goal           balanced
Backend        CUDA — selected over Vulkan: supported and expected faster
Placement      full GPU weights
Context        32,768 tokens
KV             q8_0 — fp32 would exceed the VRAM reserve
Speculation    disabled — no compatible draft/MTP head found
Tool grammar   available for single-user mode
Estimated use  10.8 GiB VRAM ± 0.4 GiB; 5.2 GiB mapped host pages
Warnings       q8_0 KV is not bit-identical to fp32
```

The output must include an exact replay command and an optional saved profile.

### 6.3 Diagnose the installation

```text
opentail doctor
opentail doctor --model models/qwen.gguf
opentail doctor --deep
opentail doctor --bundle ./opentail-support.zip
```

Checks should include:

- OpenTail build and runtime identity;
- CPU instruction-set support;
- CUDA/Vulkan discovery, driver/runtime compatibility, and usable device memory;
- native dependencies and backend binaries;
- filesystem readability and available space;
- model structural validation when supplied;
- effective configuration conflicts and unknown settings;
- a minimal allocation/backend smoke test in `--deep` mode;
- actionable remediation text and a non-zero exit code for blocking failures.

### 6.4 Run from an automatic plan

```text
opentail run models/qwen.gguf --auto
opentail run models/qwen.gguf --goal quality --ctx 8192
opentail serve models/qwen.gguf --auto --concurrency 8
```

Startup prints a compact plan summary. `--explain` prints the full decision trace.
`--quiet` suppresses the summary but not blocking errors or material quality warnings.

### 6.5 View live status

```text
opentail status
opentail status --watch
opentail status --url http://127.0.0.1:8080
```

Status should show:

- loaded model and load stage;
- effective backend and CPU/GPU placement;
- RAM/VRAM use and configured reserves;
- active context and KV occupancy;
- queue depth and active sequences;
- prefill rate, decode rate, time to first token, and inter-token latency;
- continuation/prefix-cache reuse;
- speculative acceptance;
- backend fallbacks and recent warnings.

The local host should expose a bounded, authenticated-if-the-host-requires-it status document
that the CLI can render. Do not expose filesystem paths or sensitive request data by default.

---

## 7. Architecture

### 7.1 Shared planning model

Add a backend-neutral planning layer, initially in `OpenTail.Stingray.Engine`:

```text
PlanRequest
  ModelDescriptor
  HardwareProfile
  WorkloadIntent
  UserPins
  ResourceBudget
  RequestedCapabilities

ExecutionPlan
  SchemaVersion
  ModelFingerprint
  HardwareFingerprint
  BackendPlan
  PlacementPlan
  ContextPlan
  KvPlan
  SpeculationPlan
  ServingPlan
  CompatibilityPlan
  ResourceEstimate
  Decisions[]
  Warnings[]
  Errors[]
```

Important constraints:

- `ExecutionPlan` is immutable after validation.
- Plans are serializable with an explicit schema version.
- Paths and credentials are excluded or redacted in portable output.
- Estimates include confidence/range rather than false precision.
- Decision and diagnostic codes are stable; human messages may evolve.
- The loader returns the final executed plan, including any recorded amendments.

### 7.2 Reuse existing engine knowledge

The first implementation should adapt, not duplicate:

- `HardwareProfile`;
- `TierPlanner` and its placement estimates;
- `ModelCompatibility`;
- GGUF metadata/tensor validation;
- backend discovery and selection;
- KV dtype and TurboQuant eligibility checks;
- MTP, DSpark, draft, SnapKV, and batching capability checks;
- existing startup diagnostics and `ServerMetrics`.

Where selection logic currently lives inside constructors or static environment reads, move
the decision to planning only when this can be done without changing the shipped default
path. Otherwise expose the resolved decision first and migrate ownership in a later,
separately tested step.

### 7.3 Configuration snapshot

Introduce an `EffectiveConfiguration` representation containing every stable user-facing
setting and its source:

```text
CLI pin > profile file > host configuration > environment > planner/default
```

This is a **proposed target precedence**, not a claim about the current host. Today the
standalone host binds `appsettings` first and then applies its `STINGRAY_*` overrides in
an inline options delegate, so a set environment variable wins over host configuration. Phase
0 must retain a regression test for the shipped order and make any migration to the target
order an explicit reviewed compatibility decision rather than an accidental refactor.

Required behavior:

- reject unknown keys in strict mode;
- warn by default;
- identify ignored or inapplicable keys;
- report conflicting pins before load;
- serialize with secrets redacted;
- generate a JSON Schema for editor validation;
- retain environment variables as compatibility inputs, not the preferred saved format.

JSON is recommended for the first profile format because the project already has strong
`System.Text.Json` support and can ship a schema without another parser dependency. YAML or
TOML can be considered later.

### 7.4 Machine-readable status

Extend the server with a compact status snapshot assembled from existing engine and metrics
state. Keep Prometheus metrics stable; the status document is for humans and local tooling,
not a replacement monitoring protocol.

Recommended properties:

- versioned response schema;
- bounded history of warnings rather than unbounded logs;
- no prompt text, generated text, token IDs, API keys, or full local paths;
- request-level identifiers only when already safe to expose;
- cheap enough to poll once per second.

---

## 8. Delivery phases

### Phase 0 — inventory and contracts

**Priority:** P0  
**Purpose:** prevent a new UI layer from drifting away from actual engine behavior.

Deliverables:

- [~] **1.** Inventory stable CLI/options/config/environment settings and classify each as:
   stable, expert, diagnostic, benchmark-only, or experimental.
- [ ] **2.** Inventory all model/backend/feature eligibility checks and their current owners.
- [x] **3.** Define `PlanRequest`, `ExecutionPlan`, decision codes, diagnostic severity, and schema
   versioning. The Engine contract is observational until a loader consumes it.
- [ ] **4.** Capture golden effective-plan fixtures for representative configurations:
   - [ ] CPU dense;
   - [ ] full CUDA dense;
   - [ ] partial CUDA;
   - [ ] Vulkan;
   - [ ] CPU-MoE and GPU-MoE;
   - [ ] hybrid recurrent/MTP;
   - [ ] continuous batching;
   - [ ] multimodal;
   - [ ] deliberately unsupported combinations.
- [ ] **5.** Record current startup selections for those fixtures before refactoring.

Acceptance criteria:

- [ ] Every stable load-affecting option has an owner and precedence rule.
- [ ] Golden fixtures cover every production forward-pass family.
- [ ] Planning contracts can represent every current production selection without inventing
  opaque free-form fields.

### Phase 1 — `inspect`, static `plan`, effective config, and `doctor`

**Priority:** P0  
**Purpose:** deliver the highest-value QoL improvement without changing inference numerics.

Deliverables:

- [~] **1.** `inspect` command with text and JSON output. The local GGUF command exists and is
  distinct from `plan`; remote references, companion-artifact discovery and deep validation remain.
- [~] **2.** Static `plan` command using the shared execution-plan model. `plan --json` emits
  the Engine `ExecutionPlan`; it remains read-only. Its richer eligibility snapshot is not yet
  unified with the smaller auto-run plan builder.
- [~] **3.** `--print-effective-config` and `--explain`. `plan --print-effective-config` now
  renders the typed planning snapshot without opening a model, and `plan --explain` renders the
  decision trace. They are not yet shared switches on `run`/`serve` or a host-wide snapshot.
- [~] **4.** `doctor` with fast checks and deterministic exit codes. Fast runtime/backend/model
  checks exist; deep checks, remediation and support-bundle output remain.
- [~] **5.** `run --auto` builds an `ExecutionPlan` and applies its unpinned placement, context,
  and KV-type choices. It still constructs its forward-pass family locally rather than loading the
  exact plan object, so displayed-vs-executed equality is not yet enforced.
- [x] **6.** Compact startup plan summary. It is printed only for `run --auto`; `--explain`
  requires `--auto` so it cannot describe a plan that will not execute.
- [~] **7.** Unit tests for planning-config precedence, typed JSON values, strict profiles, and
  invalid environment fallbacks. Stable decision codes and the full eligibility matrix remain untested.
- [ ] **8.** Integration tests proving displayed and executed plans match.

Acceptance criteria:

- [x] `inspect` does not allocate model weights in its default mode. It parses GGUF metadata and
  tensor indices through the existing memory-mapped reader, without constructing a forward pass or
  uploading/allocating weights.
- [x] A plan can be produced on a machine without the requested GPU backend and explains why
  the backend is unavailable. It emits a rejected-backend decision and a clearly labelled CPU
  baseline rather than claiming an unavailable GPU placement.
- [ ] Unsupported combinations fail before weight loading whenever metadata is sufficient.
- [ ] Every ignored stable setting produces a diagnostic.
- [ ] Explicit pins are never silently overridden.
- [ ] The executed plan equals the displayed plan, or contains an explicit load-time amendment.
- [ ] Existing commands without `--auto` retain their current behavior.

### Phase 2 — status and support bundle

**Priority:** P1  
**Purpose:** make “why is this slow?” answerable without internal knowledge.

Deliverables:

- [x] **1.** Versioned local status snapshot. `GET /status`, schema v1 (`ServerStatusSnapshot.cs`).
- [x] **2.** `status` and `status --watch`. Both existed already in `StatusCommand.cs`; `status`
   now also consumes the versioned document when the host maps it.
- [~] **3.** Optional per-response timing/cache details using documented extension fields or headers,
   without breaking protocol compatibility. Opt-in `opentail_timing: true` request flag →
   `opentail_timing` response object (`time_to_first_token_ms`, `total_ms`) on OpenAI
   `/v1/chat/completions` and Anthropic `/v1/messages`, non-streaming only (`ResponseTimingExtension.cs`).
   Opt-in by design — a client that never asks for it sees the exact wire shape it always did.
   Remaining for `[x]`: streaming responses, the Responses API, and a cache-hit/reuse field (needs
   a per-request signal; the engine only exposes a lifetime-cumulative counter today, and it's only
   safe to diff for the single-user engine — `ContinuousBatchingEngine` interleaves concurrent
   requests against the same counter).
- [x] **4.** `doctor --bundle` with manifest and local preview.
- [x] **5.** Redaction tests. `DiagnosticSurfaceRedactionTests` sweeps `/status`, `/capabilities`,
   `/health`, and the `opentail_timing` response extension against one configured secret path in a
   single suite (plus `StatusDocumentTests.Status_PublishesNoFilesystemPaths` and `DoctorBundleTests`
   for the support bundle specifically).

Acceptance criteria:

- [x] A user can identify CPU/GPU placement, queueing, context pressure (KV occupancy), cache
  reuse, and TTFT/inter-token rates from one screen (`status`, backed by `/status`). Prefill vs.
  decode rate is not split out — only overall tokens/s and TTFT/ITL are reported.
- [ ] The bundle contains enough information to reproduce configuration decisions.
- [ ] Automated privacy tests verify exclusion of prompts, generated text, credentials, raw
  token data, and model contents.
- [ ] Status polling causes no measurable inference regression at one request per second.

### Phase 3 — model lifecycle

**Priority:** P1  
**Purpose:** reduce first-run and model-switching friction.

Proposed commands:

```text
opentail models pull <hf-reference>
opentail models list
opentail models show <name>
opentail models verify <name>
opentail models path <name>
```

Deliverables:

- [ ] **1.** A local model-store manifest with aliases and immutable source metadata.
- [ ] **2.** Hugging Face reference resolution.
- [ ] **3.** Resumable downloads and atomic finalization.
- [ ] **4.** Shard, size, and available-space validation.
- [ ] **5.** Gated-model/token handling through standard secure environment or credential-provider
   mechanisms; tokens are never written to profiles or bundles.
- [ ] **6.** Companion artifact associations for `mmproj`, tokenizer/config files, MTP, and draft
   heads.
- [ ] **7.** Quantization/context suitability shown through `plan`.

Acceptance criteria:

- [ ] Interrupted downloads resume safely.
- [ ] A partial or corrupt model is never presented as ready.
- [ ] Concurrent pulls of the same artifact do not corrupt the store.
- [ ] `models verify` works offline.
- [ ] Model removal is a separate, explicit destructive action and is not required for this
  phase's initial delivery.

### Phase 4 — model and API compatibility lab

**Priority:** P1  
**Purpose:** make agent/tool/API compatibility testable before deployment.

Proposed commands:

```text
opentail verify <model> --chat
opentail verify <model> --tools fixtures/weather.json
opentail verify-server http://127.0.0.1:8080 --protocol openai
opentail verify-server http://127.0.0.1:8080 --protocol anthropic
```

Coverage:

- [ ] chat-template rendering;
- [ ] reasoning on/off and reasoning-history behavior;
- [ ] stop tokens and finish reasons;
- [ ] tool-call envelopes and JSON argument types;
- [ ] required keys, enums, and schema-constrained arguments;
- [ ] streaming event order and termination;
- [ ] usage accounting;
- [ ] official SDK smoke tests where redistribution/licensing permits;
- [ ] advertised capability versus observed behavior.

Acceptance criteria:

- [ ] Every supported model family has at least one checked chat fixture.
- [ ] Every grammar-supported family has positive and negative tool fixtures.
- [ ] Streaming and non-streaming fixtures agree on semantic output and usage.
- [ ] Wire-shape regressions fail tests even when the generated text appears plausible.
- [ ] Output distinguishes engine incapability, template incapability, and model behavior.

### Phase 5 — persistent named sessions

**Priority:** P2  
**Purpose:** make long-running local agents resilient and preserve OpenTail's continuation
advantage for hybrid recurrent models.

Deliverables:

- [ ] **1.** Named session identifiers independent of HTTP connection lifetime.
- [ ] **2.** Explicit save, restore, inspect, and delete operations.
- [ ] **3.** Crash-safe, versioned state files with atomic replacement.
- [ ] **4.** Model/config/tokenizer fingerprint validation.
- [ ] **5.** State coverage for KV, GDN/recurrent state, MTP state, hidden-history, and any required
   sampling/session metadata.
- [ ] **6.** Clear cache-hit/miss/invalidation reporting.
- [ ] **7.** Size limits, retention policy, and host-level access control.

Acceptance criteria:

- [ ] Restored greedy generation matches uninterrupted generation for supported configurations.
- [ ] A mismatched model, tokenizer, build-incompatible state, or material plan change fails
  safely and explains the mismatch.
- [ ] Corrupt state cannot crash the host or partially mutate a live session.
- [ ] Saving sessions does not expose prompt text unless an explicit history-export option is
  requested.
- [ ] Unsupported forward-pass families reject persistence rather than silently re-prefilling.

### Phase 6 — measured auto-tuning

**Priority:** P2  
**Purpose:** turn OpenTail's performance discipline into a user-facing advantage.

Proposed command:

```text
opentail tune <model> --goal balanced --budget 60s
```

Candidate dimensions, bounded by static eligibility:

- [ ] backend where more than one is viable;
- [ ] CPU worker count;
- [ ] context/KV strategy;
- [ ] prefill chunk size and dequant-cache budget;
- [ ] MoE placement;
- [ ] serving batch/concurrency settings;
- [ ] speculative mode and verify depth.

Deliverables:

- [ ] **1.** A benchmark protocol with warmup, control, repetitions, and dispersion.
- [ ] **2.** A machine/model/build fingerprinted result cache.
- [ ] **3.** Quality/parity policy attached to each candidate.
- [ ] **4.** Early stopping when candidates are within the noise floor.
- [ ] **5.** Saved tuned profiles consumable by `run` and `serve`.

Acceptance criteria:

- [ ] No candidate outside the selected quality policy is tested or recommended.
- [ ] Results report sample count and variance/noise estimate.
- [ ] A corrupt or thermally unstable control invalidates the sweep.
- [ ] Cached tuning results are invalidated by relevant build, driver, hardware, model, or
  planning changes.
- [ ] “No meaningful difference” is a valid final result.

---

## 9. Suggested implementation order within Phase 1

- [x] **1.** Define decision/diagnostic records and JSON shape. Engine records, stable decision
  codes, string enum output, and schema version 1 are emitted by `plan --json`.
- [~] **2.** Add `ModelDescriptor` from existing GGUF validation. The static report includes
  GGUF identity, architecture, dimensions, quantization metadata and compatibility; it is not yet
  the complete descriptor required by `inspect`.
- [~] **3.** Wrap current hardware detection in a serializable descriptor. The report includes CPU
  facts plus CUDA/Vulkan status and VRAM, with `--no-gpu-probe`; device selection remains runtime-owned.
- [~] **4.** Produce a read-only `inspect` command. Local identity/capability output exists;
  richer artifact and corruption diagnostics remain.
- [~] **5.** Adapt current `TierPlanner` output into `ExecutionPlan`. Static and auto-run paths
  both produce plans, but their builders still need unifying before this is complete.
- [~] **6.** Add explicit eligibility results for KV, speculation, batching, grammar, and multimodal
   features. KV/TurboQuant/SnapKV, MTP, batching and grammar are covered; multimodal and other
   speculative modes are not.
- [~] **7.** Implement effective-configuration source tracking. Shared resolver covers the static
   planning inputs; host configuration and the complete stable setting surface remain pending.
- [~] **8.** Add `plan` text/JSON rendering. JSON and a compact text summary exist; richer text
   rendering, replay commands and saved profiles do not.
- [~] **9.** Make one existing loader path consume the plan, starting with CPU dense. The server
  loader accepts an `ExecutionPlan`; the CLI auto path still owns its forward-pass construction.
- [ ] **10.** Expand plan consumption across forward-pass families with golden tests.
- [~] **11.** Add `run --auto` only after displayed/executed-plan equality is enforced. The
  opt-in command exists and now applies its plan's unpinned placement/context/KV values; equality
  fixtures and a single shared builder remain pending.
- [~] **12.** Add `doctor` last in the phase so it can compose the same descriptors and diagnostics.
  A fast read-only command now shares backend facts with inspect/plan; richer descriptors and deep
  diagnostics remain.

This order intentionally produces useful read-only commands before changing load behavior.

---

## 10. Testing strategy

### Unit tests

- [ ] option precedence and pin behavior;
- [ ] decision-code stability;
- [ ] memory arithmetic and overflow handling;
- [ ] eligibility matrices;
- [ ] redaction;
- [ ] model-reference parsing;
- [ ] plan serialization round trips and schema migration.

### Golden tests

- [ ] text and JSON plans for representative model/hardware fixtures;
- [ ] diagnostics for known unsupported combinations;
- [ ] effective-config snapshots;
- [ ] status rendering.

Text goldens should avoid unstable details such as absolute paths, free-memory values, and
timestamps. JSON assertions should prefer semantic fields over full-string comparison.

### Integration tests

- [ ] plan versus executed-plan equality;
- [ ] backend unavailable/fallback behavior;
- [ ] interrupted/corrupt model downloads;
- [ ] server status while idle, loading, serving, queued, and overloaded;
- [ ] official client SDK protocol fixtures;
- [ ] session save/restore and invalidation.

### Performance tests

- [ ] `inspect`, static `plan`, and fast `doctor` startup time;
- [ ] status polling overhead;
- [ ] plan-consumption overhead versus the existing loader;
- [ ] auto-tune control stability.

### Privacy/security tests

- [ ] secrets and API keys never serialize;
- [ ] support bundles contain no prompts, generations, raw tokens, or model bytes;
- [ ] remote model references cannot escape the model-store root;
- [ ] archive extraction rejects traversal and unsafe links;
- [ ] status/session endpoints follow host authentication and authorization policy.

---

## 11. Success metrics

Metrics are local test/product metrics unless the user explicitly opts into telemetry.

| Metric | Desired outcome |
|---|---|
| Time from install to first successful generation | One model reference and one command |
| Failed loads caught before weight allocation | Increasing toward all metadata-detectable failures |
| Silent ignored stable settings | Zero |
| Displayed plan differs from executed plan without amendment | Zero |
| Support exchanges needed to identify effective placement | One bundle/snapshot |
| Plan memory-estimate error | Measured per backend/config, with published confidence |
| Tool/API fixture pass rate for advertised families | 100% |
| Continuation/session reuse visibility | Every request reports hit/miss/invalidation reason |
| Status polling performance impact | Below measurement noise at 1 Hz |
| Auto-tune recommendations inside noise of best tested candidate | 100% under its stated protocol |

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Planner and loader drift apart | One `ExecutionPlan` consumed by the loader; integration assertion on final plan |
| Huge feature matrix makes planning unmaintainable | Typed sub-plans, stable decision codes, per-feature eligibility providers |
| Estimates appear more precise than they are | Ranges/confidence; distinguish static estimate from measured result |
| `--auto` unexpectedly changes established behavior | Opt-in initially; preserve current commands; print material changes |
| Profiles become another undocumented knob surface | Schema, descriptions, strict validation, effective-config output |
| Support bundles leak sensitive data | Allowlist manifest, automated redaction tests, local preview, no upload |
| Model manager becomes a second Hugging Face client ecosystem | Start with narrow GGUF references and standard HTTP semantics |
| Compatibility fixtures overfit one model output | Test protocol shapes/invariants; separate model behavior from server behavior |
| Persistent state becomes invalid after engine changes | Versioned format and strong model/config/build fingerprints |
| Auto-tuning promotes noise | Warmups, repetitions, control validation, early inconclusive result |
| QoL work accidentally changes numerics or throughput | Keep planning observational first; parity and existing test suites gate plan consumption. Treat any hot-path change as performance-sensitive: benchmark it in isolation and end-to-end against a control, because extra live registers or scalar/vector work can cost tens of percent even without allocation, locking, or I/O. |

---

## 13. Review decisions requested

Reviewers should explicitly decide:

- [ ] **1.** Is the immutable shared `ExecutionPlan` the correct architectural centre?
- [ ] **2.** Should `--auto` remain opt-in indefinitely, or become the default after a stability period?
- [ ] **3.** Are the five initial goals (`quality`, `balanced`, `throughput`, `long-context`,
   `low-memory`) sufficiently distinct?
- [ ] **4.** Is JSON the correct first saved-profile format?
- [ ] **5.** Which current environment variables are stable user configuration versus diagnostic or
   experimental controls?
- [ ] **6.** Should `inspect` accept remote Hugging Face metadata before Phase 3, or remain local-only?
- [ ] **7.** Which plan amendments are permitted at load time, and which must fail?
- [ ] **8.** What is the supported compatibility promise for OpenAI and Anthropic APIs?
- [ ] **9.** Should session persistence initially be CLI-only, server-only, or share one state API?
- [ ] **10.** What benchmark budget and quality policy are acceptable for automatic tuning?

---

## 14. External product evidence

The plan responds to recurring problems visible in current inferencers:

- llama.cpp added automatic GPU/context/tensor fitting because manual layer and tensor split
  selection was acknowledged as poor usability. Explicit placement arguments disable the
  automatic memory allocation, and the initial discussion includes a user unintentionally
  doing so and receiving much worse performance:
  <https://github.com/ggml-org/llama.cpp/discussions/18049>
- llama.cpp's current server already includes a Web UI, continuous batching, model routing,
  model presets, Responses, embeddings, tool calling, and timing information. These are
  useful parity references but weak differentiators for OpenTail:
  <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- llama.cpp's function-calling documentation notes model/template-specific handling,
  overrides, and generic fallbacks:
  <https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md>
- A recent llama.cpp regression returned tool arguments in a wire shape that broke the
  official OpenAI SDK, demonstrating the value of SDK-level compatibility fixtures:
  <https://github.com/ggml-org/llama.cpp/issues/20198>
- A current llama.cpp report describes failed multi-turn cache reuse for hybrid recurrent
  models, supporting explicit continuation/session correctness as a differentiator:
  <https://github.com/ggml-org/llama.cpp/issues/21831>
- Ollama exposes loaded CPU/GPU placement through a simple `ps` command, establishing a
  useful baseline for human-readable runtime status:
  <https://docs.ollama.com/faq>

---

## 15. Recommended first review boundary

Approve, reject, or revise **Phases 0 and 1 only** before implementation begins.

Phases 2–6 establish the intended direction and prevent Phase 1 contracts from becoming
dead ends, but they should not expand the first implementation scope. The first milestone
is complete when a user can inspect a model, see a trustworthy static plan, diagnose the
machine, run that exact plan, and understand every material automatic decision.
