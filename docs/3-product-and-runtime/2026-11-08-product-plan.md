# Product and runtime plan

Status 2026-11-08. This is the one plan for the product surface: configuration, observability, the compatibility corpus, the first-run and catalog work, documentation, and samples. It replaces the separate QoL, documentation-architecture, samples, first-run and forking plans, which are now in [done/](../done) as superseded. Four owner docs stay as design records and are linked from here:

- [032-multi-model-inference-runtime-plan.md](032-multi-model-inference-runtime-plan.md): multi-model runtime design.
- [2026-10-09-public-developer-surface-and-front-door-plan.md](2026-10-09-public-developer-surface-and-front-door-plan.md): the catalog design ("front-door design").
- [2026-10-12-opentelemetry-observability-plan.md](2026-10-12-opentelemetry-observability-plan.md): observability design and acceptance list.
- [2026-10-09-checkpoint-scout-and-ai-admission-plan.md](2026-10-09-checkpoint-scout-and-ai-admission-plan.md): scout design.

Contracts that every item must satisfy: `Tests.Core` before merging; no secrets, filesystem paths or user content in telemetry or `/status`; the status matrix (STATUS.md) only shows what has been verified (rule 10); "ported, not verified" families stay out of the catalog and the docs (rule 14); CLI, library, README and tests read the same catalog.

---

## Part A: Start now, in order

### A1. Effective configuration reaches the real startup values

- **Why:** `ServerEnvironmentOverrides` centralises the precedence, and `/status` already publishes a `configuration.bound` snapshot and the resolved loader route. Missing: actual device and cache allocation, which the operator needs to match the budget against the machine.
- **Next step:** publish allocation figures (device bytes, cache bytes) from the loader in `/status`, using only numbers, never paths or values from the environment.
- **Depends on:** the device memory figure in [../00-current-work.md](../00-current-work.md), item A5, so the figures are the measured ones.
- **Done when:** `/status` shows allocation for a loaded model, a test checks that no path or raw environment value appears, and the figure matches the ledger within the tolerance in A5.

### A2. Golden capability fixtures where a loader runs the route

- **Why:** static CPU, backend, dtype, batching and speculation decisions are covered. Hardware rows are release gates and cannot run here.
- **Next step:** add a fixture only where a loader actually executes the route. Do not add fixtures for routes that only the static planner reaches.
- **Done when:** every route the loader executes has a fixture, and the list of routes without one is written in the test file.

### A3. Server observability: streaming timing and per-request cache signal

- **Why:** the non-streaming timing extension covers OpenAI, Anthropic and Responses. It is deliberately absent from streaming event contracts. Remaining: streaming timing and the per-request cache signal. VRAM and RAM breakdown depends on A1.
- **Next step:** decide the streaming timing fields with the observability plan's §7 allow-list, then add them behind the same opt-in switch.
- **Done when:** the fields are in the allow-list, a test checks each one, and the privacy canary test still passes.

### A4. Protocol compatibility corpus, per case

- **Why:** the corpus has 63 cases for chat, messages and responses, including streaming, tool definitions, wrong shapes and malformed bodies. Cancellation is covered by `ClientDisconnectTests`. The llama-server surface (`/tokenize`, `/detokenize`, `/props`) has fake-engine coverage.
- **Next step:** record a status and an error type for every case in `protocol-smoke.json`. Add streaming tool-call and error cases for Anthropic and Chat Completions. Expand only when a named consumer needs the case.
- **Done when:** every case has a recorded status and error type.

### A5. Sessions stay out of the product configuration surface

- **Why:** the restart lane is proven by a real-GGUF, cross-process replay test. It is not yet a product contract.
- **Next step:** no code. Keep sessions out of the configuration surface until a named lifecycle and capability gate exist. Revisit with B2.
- **Done when:** the gate exists, or the owner confirms it is not wanted.

### A6. Catalog: one source for the CLI, library, README and tests

- **Why:** the first-run seam works (`chat`, `speak` and `transcribe` offer to install a missing model). The catalog design is not yet the single source. Each consumer still has its own list.
- **Rules (from the design):** one default per task; at most three alternatives per task, each with a user-level reason ("faster", "fits in 8 GB"); admission requires a 🟢 status and an automated test on the exact files, sha-checked; everything else works through `pull -r` or `-m`.
- **Next step:** write the catalog file (id, task, why, HF repo, file, sha256, size, licence, RAM and VRAM, speed note, the test that proves it). Generate the lookup at build time so it works under NativeAOT. Point the CLI `models` command at it first.
- **Done when:** the CLI, the library and the README read the same catalog, and a test downloads and checks one entry per task.

### A7. Docs-as-tests

- **Why:** `scripts/verify-package-consumers.ps1` compiles the README snippets and the samples (17 checks on 1.0.9). It does not run the README commands.
- **Next step:** run every README command and snippet in a test, with cached downloads, so the README fails in CI before it lies.
- **Done when:** every README command and snippet is exercised by a test.

### A8. Documentation architecture: ownership, index, status-matrix check

- **Why:** the ownership map exists in its plan, but no audit record was kept. There is no test that checks the status matrix against the evidence.
- **Next step:** (1) write the ownership audit record, one line per doc: owner and status. (2) Re-check the front door beyond the README recipes (`README.md`, `RUNNING.md`, `WHAT-YOU-CAN-DO.md`). (3) Merge the overlapping user guides and references. (4) Add a documentation index, and a test that the status matrix cites a dated source for every row (rule 10).
- **Done when:** the audit record exists, the index is generated or tested, and the status-matrix test fails on an uncited row.

### A9. Samples: priority 1, then the automated checks

- **Why:** the samples plan lists a priority-1 set, each checked against its evidence rules (section 3 of that plan). Priority 2 needs extra setup. Section 7 asks for automated verification of every sample.
- **Next step:** take the priority-1 samples in the order of that plan's section 5. Each one needs a README contract (section 6) and a verification run (section 7).
- **Done when:** every priority-1 sample runs in the verification script, and priority 2 is either built or recorded as deferred.

### A10. Forking: real-model test on the GPU, hybrid and GDN passes

- **Why:** the CPU dense real-model fork test exists. The same test for the GPU, hybrid and GDN passes does not.
- **Next step:** add the real-model variant to the GPU, hybrid and GDN fork tests, reusing `HotSessionForkRealModelTests`. Correctness only: no speed claim, so the iGPU result is acceptable (rule 13 applies only to performance).
- **Done when:** each pass has a real-model fork test that passes, with its checkpoint named in the test.

### A11. Front-door close-out and release-quality review

- **Why:** the front-door and API documents (C2), and the close-out review (E) of the public-surface plan, are open. They wait on A6 and A7.
- **Next step:** after A6 and A7, review the front door against the catalog, the README and the API docs, and record each finding with a fix or a reason.
- **Done when:** the review record exists, and every finding is closed or recorded.

---

## Part B: Decisions and deferred work

1. **Durable session restore across a restart, in multi-model mode** (multi-model plan, phase 7). Deferred. Needs a design before any code: what is restored, and what the restore guarantees. Owner decision to start.
2. **Fair scheduling: measure, then decide** (multi-model phase 6). The measurement is in the merged plan (B2). The default stays off until the measurement shows no loss.
3. **Speak default licence.** The Piper `en_US-lessac` licence restricts use more than the known-good plan's wording says. Kokoro is Apache-2.0 and is the current default for `speak`. Owner decision: keep Kokoro as the default, and whether Piper stays as an alternative in the catalog.
4. **Scout calibration and Hugging Face discovery** (scout phases 4 and 5b). Deferred by the scout plan. Start only when the owner wants repository discovery, and calibrate against the misses recorded in its calibration notes.
5. **Piper licence acceptance** for the first-run check: owner action, see [../00-current-work.md](../00-current-work.md), item B6.

---

## Part C: Blocked

- **Clean-machine release verification** for the observability and release-quality items. Needs a second machine.
- **HTTP image API** (`/v1/images/generations`, `/edits`): parked by the owner; stays 501.
- **Browser UI for the local server** (Qwen phase 7F): parked by the owner.

---

## Already done, not to repeat

- The `setup`, `pull` and first-run seam for `chat`, `speak`, `transcribe`.
- `ServerEnvironmentOverrides` precedence and the `configuration.bound` snapshot.
- The 63-case protocol corpus and the `ClientDisconnectTests` cancellation test.
- The env-var inventory: every registered name is listed.
