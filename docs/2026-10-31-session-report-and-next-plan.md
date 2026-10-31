# Session report and next plan, 2026-10-31

An unattended working session on OpenTail.Stingray (parts A, B and C of the plan agreed at the start, then two external review
rounds and the queued housekeeping). This page records what was done with its evidence, what turned out to be wrong in the
documents, what was measured, and what to do next. It is a report plus a proposed plan; it is **not** the backlog.
[00-current-work.md](00-current-work.md) stays the single backlog, and every item below that is still open should be moved there
when someone takes it on.

> **Status after this report was written (same day).** Several steps of section 7 have since been done, so read this page as the
> record of the first half of the day, and use [plan 2](2026-10-31-autonomous-plan-2.md) for current state (its open items; the finished ones are in its [done half](done/2026-10-31-autonomous-plan-2-completed-items-2026-10.md)). Done since:
> the first-decode warm-up (step 2, PR 73); the OpenTelemetry allow-list test and server canary (step 6, PR 80; the NativeAOT-with-listener,
> overhead and OTLP items are still open); the flaky timing test, the test that dirtied a tracked image and the missing CI suites (step 8,
> PRs 74 and 75). Step 3 is partly done (Qwen2.5 admitted; Gemma 2 and Phi-3 recorded as not admissible; Granite open) and step 4 is done (`rope_freqs` and long contexts exercised, the 32-token boundary tested, the `startPos > 0` cost measured). Not done: steps 5 and 7.

Scope: OpenTail.Stingray only. Skipped on purpose, as instructed: CUDA and discrete-GPU items, items blocked on an independent
reference, "ported, not verified" families (CLAUDE.md rule 14), the voice-licence `speak`/`transcribe` recipes, and the Q8_KS
numerics decision.

## 1. What was done (PRs 52 to 70, all merged)

| PR | What | Evidence |
|---|---|---|
| 52, 54 | Closed the "Q5_K_M is off the batched Vulkan trunk" gap; confirmed four "slower" CPU rows were noise | SmolLM2-1.7B Q5_K_M prefill 66.8-85.5 tok/s against 84.7-86.0 for Q4_K_M (503 tokens, 3 runs each); rwkv-6, Phi-3-mini, granite-vision, qwen2.5 re-run 3 times each, none a regression |
| 53 | Gemma 4 handoff threshold is 32 tokens, not 19; `GemvQ4K8x8Q8K` citation confirmed | `STINGRAY_HYBRID_CPU_PREFILL_MIN_TOKENS` default 32 in `EngineEnvironment.cs`; the 19 was the sweep prompt's length |
| 55 | CLI names itself `stingray` in help and messages | `CliBrandingTests` (help usage line, source scan, script scan); Tests.Cli all passing |
| 56 | Concurrent fork test on a real model; fixed a thread-affinity bug in the heavy-test lock | `HotSessionForkRealModelTests` concurrent variant, 10 of 10 runs, shared decode batch asserted; `HeavyTestBase` released its mutex from the wrong thread in 5 projects |
| 57, 59, 60 | Three rounds of fixes from external reviews (licence text, stale statuses, scripts, README block, gate race) | each claim re-checked against the code before editing |
| 58, 60 | Opt-in fair turn gate for models that share a device (`FairExecutionGate`) and its cancellation fix | 10 gate tests, 3 engine tests, 2 real-model tests (loader and DI), mutation-checked; see section 6 for limits |
| 61 | A failing telemetry listener can no longer break inference; no-content canary test | each failure test was red before its fix; `EngineTelemetryPrivacyTests` catches a planted leak |
| 62 | One-row Q4_K decode kernel: **negative result**, reverted | 3% slower in all 6 interleaved pairs on two models, byte-identical output |
| 63 | Corrected the image-token prefill backlog item | CPU `PrefillEmbeddings` already batches; the remaining per-token paths are listed |
| 64 | **Dense `llama` models are admitted to the CPU-prefill handoff on the Vulkan hybrid** | see section 3 |
| 65-68 | Env-var inventory made name-complete: 287 of 287 (was 192) | four batches, every row written from the variable's read-site; `InventoryTable_CoversTheRegistry_ExceptTheListedGap` now enforces it (mutation-checked) |
| 69 | Pinned the three moving upstream links in the research docs to exact commits | two `llama.cpp` files to `eb4e244b`, one Hugging Face config to `f7f5b7f5`; URLs return HTTP 200 |
| 70 | Fixed three stale `Vulkan.Fast` tests found by a sweep of every test project | Path 3 has been the default since 2026-10-11; tests still pinned Path 1 |

State at the end: every test project on `main` builds and passes, apart from one known flaky timing test (section 5). Totals from
the final sweep: Core 1179, Cli 745, ForwardPass.Fast 1150, Server.Fast 494, Sessions.Fast 129, Vulkan.Fast 48; heavy and
hardware-gated suites skip on this machine.

## 2. What turned out to be wrong or stale in the documents

Each of these was a confident statement that did not survive reading the code. All are corrected in the owning document.

- **Plan 010 (session forking).** It proposed implementing `IForwardPass.CreateContext()` on seven passes. Nothing in `src/` calls
  it; sessions fork through `HotSessionRuntime.Fork` and the reference-counted KV cache, and the `InferenceSession` class the plan was
  written around no longer exists. The per-backend work is not needed for forking. (Owner decision below.)
- **"Q5_K_M models are off the batched Vulkan trunk"** (22.3 against 235 tok/s). True on 2026-10-11 before Path 3; false since.
- **"Gemma 4 prompts under the handoff threshold (19 tokens)".** The threshold is 32; 19 was the prompt.
- **Backlog item 5, "`VulkanLayerSplitForwardPass` prefills per token"**, and the batched-MoE plan's "correction" of it. Both were
  wrong. The layer split already hands long fresh prompts to the CPU batched pass. The real gap was dense `llama` models with a
  partial offload on `HybridForwardPass`, which had no handoff receipt (section 3).
- **Backlog item 3, "image-token prefill runs per token (about 9 tok/s)".** The CPU path already batches a whole image; three
  narrower per-token paths remain (section 7).
- **The env-var inventory called itself name-complete** with 192 rows for 287 registered names.
- **Two of my own earlier PRs broke `Tests.Core`** (an architecture-literal ratchet and a made-up backticked path). Fixed in #68.

## 3. The main finding: dense `llama` partial offload prefilled 12x too slowly

`HybridForwardPass` (`-g N`) only uses the CPU-prefill handoff for families with a parity receipt in
`PrefillHandoffFamilies.Receipts`, and receipts are per code path. For the Vulkan hybrid only `qwen3` had a dense receipt, so any
plain `llama`-architecture model (SmolLM2, Mistral, Llama 2 and 3, Falcon3, Orpheus) was refused and prefilled token by token.

- **Before:** SmolLM2-1.7B `-g 12`, 503-token prompt: 18.1-18.3 tok/s, against 227 tok/s on the CPU alone.
- **Evidence for admission** (`HybridCpuPrefillHandoffTests`, default path with no override, real weights): K/V byte-identical and
  prefill logits equal to the CPU pass on SmolLM2-1.7B (`-g 12`, `-g 4`), Mistral-7B-Instruct v0.3 (`-g 4`), orpheus-3b (`-g 8`) and
  Falcon3-3B (`-g 4`, `-g 16`); decode cosines at or above 0.998 over 40 teacher-forced steps.
- **After** (CLI, handoff off against the new default, 3 interleaved pairs): SmolLM2-1.7B `-g 12` 178-182 tok/s (about 10x);
  Mistral-7B `-g 4` on 242 tokens 44-49 tok/s against 7.6-7.7 (about 6x).
- **Cost:** the first decode after a handoff pays a one-time warm-up of about 1.9 s (24 tokens decode at 6-7 tok/s, 160 tokens at
  12.5-13.0; the fitted steady rate is about 15.0 against 15.3-15.9 without the handoff). Whole-run wall time is still lower:
  33 s to 8 s at 24 tokens, 40 s to 17 s at 160. The warm-up cause is a hypothesis (GPU pipelines created on first use, which the old
  per-token prefill absorbed); it was not profiled.
- **Known deviation, recorded in the receipt itself:** Falcon3-3B at `-g 8` has one decode step (38 of 40) at cosine 0.9874, below the
  0.99 bar. It is deterministic, specific to that split, and the argmax agrees; `-g 4` and `-g 16` of the same file pass.
- **Not exercised:** Llama-3.1-style `rope_freqs` scaling, LongRoPE, sliding-window llama variants, contexts above 1024.

## 4. Other measured results

- **Negative:** the one-row Q4_K decode kernel (#62). The microbenchmark said the two-row kernel was 38% slower per row; end to end it
  is faster. Reverted. Only `MatVecQ4K` was tried, not `MatVecDual`.
- **Noise rather than regressions:** the four single-run CPU "slowdowns" (rwkv-6 first run was a cold file read).

## 5. Process findings (worth keeping)

1. **Mutation-check every new test.** Several tests I wrote passed the first time and would have passed on broken code (the
   "turn ending while a caller registers" branch was never reached by 3000 racy rounds). A deliberate breakage that makes the test
   fail is the only proof it can fail. A race-only branch needs a test seam, not more iterations.
2. **A failed build means you ran a stale binary.** It happened twice this session (an analyzer error, a bad string escape). Always
   build first and read the error count.
3. **Do not assert strict ordering over real threads.** A participant preempted between releasing and re-asking legitimately queues
   behind the others. Drive such tests on one thread, or assert invariants (no leaked turn, no stranded waiter).
4. **Bound every await in a test.** Two older tests would have hung on a regression instead of failing.
5. **Run `Tests.Core` before merging any `src/` or `docs/` change.** It holds the architecture-literal ratchet and the
   documentation-link checks, and I broke both without noticing until a full sweep.
6. **`sed` with `~` as the delimiter breaks if `~` appears in the text.** It silently produced `63 MB)~63 MB)` as the first line of a
   shell block in a README. Use a different delimiter or an exact-string replacement.
7. **Heavy test classes run every row whose model exists.** Use the env-selected handoff test to run one model.
8. **Tests that write into the repository.** `LtxVaeNoiseIsolationTest` rewrites the tracked
   `docs/diffusion-samples/ltx_vae_noise_isolation.png` on every run, leaving a dirty tree. It should write to a temp directory.
9. **Known flaky test (not caused by this session):** `SimdKernelsLegacyQ8_1Tests.FastKernels_AreFasterThanDequantFallback` is a
   wall-clock ratio test; it failed once in five full runs under load and passes alone.

## 6. What is built and what is not

- **Fair turn gate:** built, opt-in, off by default. One engine iteration per turn, longest waiter first. Wired through
  `RuntimeInstance`, the loader and the server options `ExecutionTurnGate` and `FairModelScheduling` (both tested through DI on real
  weights). **Not built:** mediation for the single-request `InferenceEngine`; any throughput or latency measurement; per-device
  gates chosen automatically. Fairness is per iteration, not per unit of compute time, so a long prefill iteration costs far more than
  a decode iteration.
- **OpenTelemetry:** the listener-failure item is done for the engines and the server; the no-content item is half done (engine
  canary only). **Not done:** a single test that enumerates every instrument and span key against the section 7 allow-list; a
  server-side prompt canary through the HTTP endpoints; the published NativeAOT binary check; overhead numbers; OTLP exporter
  behaviour; the observability guide run on a clean machine.

## 7. Proposed plan for the next session, in order

Each item says what to read first and how to know it worked. Sizes are my estimates, not measurements.

1. **Take the owner decisions in section 8.** Nothing else depends on them except item 6, but item 3 changes the default behaviour of
   a shipped path, so decide it first. (minutes)
2. **Hide the one-time warm-up after a handoff.** Read where `HybridForwardPass` creates its GPU pipelines (the `??=` pipeline fields
   in `VulkanBackend`) and profile the first decode to confirm the hypothesis before changing anything. If confirmed, create the
   pipelines at load or finish the handoff with one throwaway token. Verify: the 24-token decode rate matches the 160-token rate, and
   greedy output is unchanged. (half a day)
3. **Extend handoff receipts to other dense families on the Vulkan hybrid.** Same recipe as #64: run the env-selected test
   (`STINGRAY_HANDOFF_TEST_MODEL`, `STINGRAY_HANDOFF_TEST_GPU_LAYERS`, `STINGRAY_HYBRID_CPU_PREFILL=all`), then add a permanent row
   and a receipt only on a pass. Candidates with a local checkpoint: Qwen2.5-1.5B, Phi-3-mini (LongRoPE tensors, see the structural
   rule), gemma-2-2b-it, Granite. Each is about 20 to 45 s per configuration. Verify: the permanent rows pass without the override.
   (a day, mostly waiting on runs)
4. **Close the gaps the `llama` receipt names:** a Llama-3.1 `rope_freqs` checkpoint, contexts above 1024
   (`STINGRAY_HANDOFF_TEST_CTX`), and a prompt at the 32-token minimum. Also note the handoff is refused for `startPos > 0`, so a
   second chat turn on the hybrid still prefills its new tokens one at a time; measure how much that matters before building
   anything. (half a day)
5. **Measure the fair turn gate on a real workload before anyone considers enabling it.** Two resident models, one prefill-heavy and
   one decode-heavy, with and without `FairModelScheduling`: per-model latency and total throughput, three interleaved runs. Then
   decide about mediating `InferenceEngine`. Verify: numbers in `032-multi-model-inference-runtime-plan.md`, slice 5. (a day)
6. **Finish the OpenTelemetry acceptance list:** the one allow-list test over every instrument and span, the server-side canary,
   then the NativeAOT publish check with a listener attached and the overhead procedure from the plan's section 8. The plan's
   checklist already says exactly which items are open. (a day or two)
7. **Remaining per-token image paths:** the CLI segment path (batch each contiguous run with `PrefillEmbeddings`), then a batched
   entry point for the Vulkan and CUDA passes. Any new caller needs a real-checkpoint output comparison, because batched prefill
   quantises activations differently. Gemma 4 through the server cannot benefit until the batched trunk supports per-layer head
   dimensions. (two days)
8. **Housekeeping:** make `LtxVaeNoiseIsolationTest` write to a temp directory; fix or quarantine the flaky `SimdKernels` timing
   test; add a CI step that builds and runs every test project (the sweep script I used did this in one background command and found
   three real failures); add "run `Tests.Core`" to the merge checklist. (half a day)

Not on this list because nothing was learned about it: the rest of the CPU and diffusion performance backlog in
[00-current-work.md](00-current-work.md) section 4, and pickup row 8 (long-context decode) in
[4-performance/2026-10-30-performance-pickup.md](4-performance/2026-10-30-performance-pickup.md), which was not re-measured.

## 8. Decisions for the owner

1. **`IForwardPass.CreateContext()`.** No caller exists in `src/`; only two test doubles use it. Remove it (a breaking change for
   external implementers of a public interface), keep it as an unused default, or give it a real caller. Plan 010 stays unimplemented
   until this is decided.
2. **`FairModelScheduling`.** It is off by default. Should it ever default on? That needs the measurement in item 5 first; no
   throughput claim has been made.
3. **Keep the dense-`llama` admission on the Vulkan hybrid** despite the Falcon3-3B `-g 8` step (cosine 0.9874 against a 0.99 bar)?
   I admitted it because six configurations on four checkpoints pass, the effect is a 6x to 10x prefill gain, and the deviation is
   disclosed in the receipt. Reverting is one line: delete the `LlamaArchitecture.Descriptor.Id` entry in
   `PrefillHandoffFamilies.Receipts` and its permanent test rows.

### Decisions taken on the owner's behalf (2026-10-31, provisional and reversible)

The owner asked for the work to continue without waiting for answers, so each decision above was taken with the option that is cheapest
to undo. The owner can overrule any of them; each line says what overruling costs.

1. **`CreateContext()`: keep it as is, unused.** Removing it breaks external implementers of a public interface for no gain, and plan
   010 has no caller to wire. Revisit only if a real feature needs per-request contexts. Overruling costs one interface edit.
2. **`FairModelScheduling`: stays off by default.** No throughput number exists. It may only default on after the measurement in the
   follow-up plan (item P4) shows no loss in total throughput. Overruling costs a default flip.
3. **Dense-`llama` admission on the Vulkan hybrid: kept.** Six configurations on four checkpoints pass, the gain is 6x to 10x, and the
   Falcon3 `-g 8` deviation is in the receipt. Reverting is one line.
