# Current work: what to do next

As of 2026-11-08. This file is the entry point and the parent of the open work in Queue 1, which was the separate open-work plan and is now merged here. The Qwen 3.8 port has its own plan. Finished items are in [done/](done). The previous version of this file is in git history.

## How to use this file

1. Work the **Queue** top to bottom. Do not reorder because another item looks more interesting.
2. Each item has a **Next step** and a **Done when** in its plan. If an item stalls, write the blocker in the plan and move to the next one (CLAUDE.md).
3. An item is done only when its **Done when** line is true, with real weights and an independent reference where one applies. Then update the STATUS.md row, the RUNNING.md row if a command changed, and commit. Move the finished text to [done/](done).
4. Heavy runs one at a time (64 GB host). Perf claims need an idle machine, interleaved A/B, three or more runs and equal output (CLAUDE.md rule 7).
5. A new item needs a named failure, a user-visible limit, a measurable gain or a recurring cost.
6. **One parent per item.** An item's text and its checkbox live in one parent document. This file is the parent of Queue 1 (Parts A to C), of the Queue 3 and Queue 5 items with no open parent, and of the owner actions. Queue 2, Queue 4 and Queue 5 item 8 point to their parent and do not copy its state.

## Queue

### 1. Open work (in order)

#### How to read this

- **Part A** is work that can start now on this machine. Do the items in order. Earlier items either unblock later ones or make their results trustworthy (test reliability and the memory figures come first, because later measurements depend on them).
- **Part B** is work that waits for you (an approval, a choice, or a download) or for a quiet machine.
- **Part C** is blocked or parked. It is listed so it is not lost. Nothing in it should be started.
- Each item has **Why**, **Next step** and **Done when**. "Done when" is the test for closing the item. When an item is done, move its text to a file in [done/](done) with the date and the evidence, and delete it from this plan.
- Plan is finished when Parts A and B are empty or moved to C with a reason.

**Rules that apply to every item:** one merged PR per item; run `Tests.Core` before merging; a performance claim needs a quiet machine, interleaved A/B runs, three or more repetitions and equal output (CLAUDE.md rule 7); nothing that needs a discrete GPU, CUDA, a second machine, AVX-512 or ARM64 (rule 13); every new test is mutation-checked (break the code on purpose and confirm the test fails); model weights may be deleted when stale, no other file may be.

**Already decided, provisional and reversible** (the owner may overrule; the cost of overruling is in brackets):
- `CreateContext()` on the forward-pass interface stays, unused (one interface edit to remove).
- Dense `llama` models are admitted to the CPU-prefill handoff on the Vulkan hybrid (one line in the receipt table to revert).
- `FairModelScheduling` stays off by default (flip the default).

---

#### Part A: Start now

##### A1. Cancellation test fails as "Failed" under low CPU

- **Why:** `HotSession_CancellationRollsBackCacheAndDoesNotCommitRevision` failed once in 40 with `DOTNET_PROCESSOR_COUNT=2` (expected Cancelled, got Failed). `HotSession_CancellationDuringPrefillLeavesTheCursorUnchanged` failed once in 40. Neither fails alone (0 in 120). The same class failed on CI in `HotSessionMetricsMetadataTests.HotSession_TurnResult_ReportsCancelledFinishReasonOnCancellation`. Fixes already merged cover the queued-request and between-prefill-chunk cases. Not yet covered: a cancel during prefill or decode that the consumer sees as a normal end.
- **Next step:** loop the `Tests.Sessions.Fast` executable at `DOTNET_PROCESSOR_COUNT=2`, with a diagnostic that prints `FailureReason` for every non-Cancelled outcome of a cancelled turn. Capture the exception. Write a deterministic repro first, then fix.
- [x] **Done when:** 120 loops pass in a row under the same contention.
- **Status 2026-11-08: done.** `HotSession_CancellationRollsBackCacheAndDoesNotCommitRevision` passed 40 of 40 (baseline) and 120 of 120 more at `DOTNET_PROCESSOR_COUNT=2`, while another AI's `Tests.Core` run was active. The assertion already prints `FailureReason`, which is the diagnostic the next step asked for. Mutation check: `InMemorySessionStore.Cancel` recording `Failed` makes the test fail with "expected Cancelled, got Failed; failure reason [MUTATION]". The mutation was reverted.
- **Note:** a hung test process can lock the output DLLs (MSB3021 or MSB3027 on the next build). `dotnet-stack report -p <pid>` shows where it is stuck.

##### A2. Fixed timing ceilings in two tests

- **Why:** `ServingRequestTimingTests.MarkGenerationComplete_ExcludesTimeSpentAfterItFromGenerationDuration` asserts the sample is under a fixed 0.15 s after a `Thread.Sleep(30)`. On the loaded 2-core runner the sleep overran (164 ms). `HotSessionTests.HotSession_AbandonedConsumer_DoesNotStallOtherSessions` assumes request A is still running when session B completes. On Windows a 2 ms sleep costs about 15 ms, so that assumption breaks.
- **Next step:** (1) compare against measured elapsed time, with a `Stopwatch` around the sleeps, or give the metrics a clock a test can fake. (2) In the abandoned-consumer test, make A's decode step block on an event (the fake's `OnDecodeStep` hook) until B has finished, so the ordering is guaranteed instead of timed.
- [ ] **Done when:** both tests pass in a loop at `DOTNET_PROCESSOR_COUNT=2`, and each fails when its fix is reverted.
- **Status 2026-11-08: partly done.**
  - [x] (1) Done. `ServingRequestTimingTests` already bounds the sample by a `Stopwatch` around the decode window, with no fixed ceiling. 120 of 120 pass at `DOTNET_PROCESSOR_COUNT=2`. Mutation check: making `MarkGenerationComplete` a no-op makes it fail with "delivery time leaked into the sample" (248.7 ms against a 34.0 ms window). Reverted.
  - **Deferred 2026-11-10 (blocker):** the batcher is one loop, so (2) needs a per-sequence hold seam in `ContinuousBatchingEngine`. Checked 2026-11-10: no such seam exists, and the test `FakeForwardPass` has none. Adding a product-code seam for a test is a design decision for the owner, so this part stays open. The current test passes 120 of 120 under the 8 s margin.
  - [ ] (2) Not done, and the planned event-based version is not possible as written. The engine's batcher is one loop, so blocking request A's decode step on an event would also block B, and B could never finish. Removing the 2 ms step delay was tried: A then retires before B finishes (the non-vacuity guard fails in 40 of 40 runs), so A's lifetime does need wall-clock time. The test stays as it was: 8 s of margin at 2 ms per step, 120 of 120 pass at `DOTNET_PROCESSOR_COUNT=2`. A real fix needs a fake that can hold one sequence without holding the batcher, which is not built yet.

##### A3. Finder sweep for the remaining heavy test projects

- **Why:** 161 test files still have their own model finder. The shared one is `Engine.Verification.ModelLocator.Find` and `FindDeep`. Finders that look only in the repo's `models` folder skip tests whose checkpoints sit on other drives, so the tests report "skipped" when they could run.
- **Already done:** ForwardPass, ForwardPass.Fast, Tests.Server (5 finders), Tests.Vulkan (4), and sampled classes in Audio and Diffusion.
- **Next step, in this order:** Core, Embeddings, Cli, then the rest of Server and Vulkan. For each project, run with `STINGRAY_RUN_HEAVY_TESTS=1` and list every skip with its reason. Point each finder at `ModelLocator`. Then Audio and Diffusion in full (hours of runtime; run last).
- [ ] **Done when:** every heavy project's skips are listed with a reason, and no finder duplicates the shared one.
- **Status 2026-11-08: started, one file.**
  - **Confirmed bug.** `DeepSeek2GpuParityTests.FindModelPath` (Vulkan) walked up from the test binary and checked only `<repo>/models` and `<repo>/models/_models`. The checkpoint `DeepSeek-V2-Lite-Chat.Q2_K.gguf` (6.4 GB) is on `F:\_models`. In the 2026-11-08 Vulkan run the test reported "not present", so it silently skipped a real checkpoint. Changed to `ModelLocator.FindDeep`, the same call `Gemma4GpuForwardPassTests` uses.
  - **Verified:** build is clean; the class still skips by default with the heavy-gate message. **Not verified:** that the new lookup finds the `F:` file. That needs a heavy run, which loads the 6.4 GB model, so it is left for a quiet-machine run.
  - **Scope, strict definition:** 95 test files build their own `"models"` path and do not use `ModelLocator`. By project: Diffusion 29, Cuda 28, Vulkan 18, Core 9, Audio 4, Server.Fast 3, Cli 2, Server 1, ForwardPass.Fast 1. The plan's 161 used a broader definition.
  - **Before changing the others:** check each project's gate first. Vulkan, Diffusion and Audio use `HeavyTestBase`, so a finder change there only runs in heavy mode. Core, Cli and Server.Fast were not checked, and a finder change there could start real model runs in default runs. Core was checked 2026-11-10 (see the note below).
  - **Core done 2026-11-10 (per-class runs; no full-project run completed).** Nine finder files now use `ModelLocator`. `PublicApiContractsTests`, `JinjaTojsonTests` and `RealHeaderSemanticsTests` use `FindDeep`; `Gemma4TokenizerTests`, `ChatTemplateCorpusTests`, `TekkenTokenizerTests`, `PreTokenizerParityTests` and `RealPackageTouchpointTests` use `Roots()`, which removes the hard-coded `E:\models`. Core has no heavy gate, so these run in default runs. Per-class results on this machine: `PublicApiContractsTests` 33 run, 10 visible skips (SmolLM2 absent; the 9 MiniLM tests run for real against `/f/_models/all-MiniLM-L6-v2-Q8_0.gguf`); `ChatTemplateCorpusTests` 237 pass, 1 visible skip (SmolLM2 fixture), 89 s; `TekkenTokenizerTests` 15 pass; `PreTokenizerParityTests` 31 pass, 49 s; `RealHeaderSemanticsTests` 7 pass; `JinjaTojsonTests` 5 pass; `RealPackageTouchpointTests` 3 visible skips (SmolLM2 package absent); `Gemma4TokenizerTests` 3 pass, 1 visible skip. Build is clean under `TreatWarningsAsErrors`. `ModelCatalogTests` was checked and is not a finder (temp-directory tests only).
  - **Open question found by the sweep:** the Gemma 4 E4B Q4_K_M export at `/f/_models` (`tokenizer.ggml.eos_token_id` 106) does not list `<eos>` among its special tokens. Once the finder could see it, `Gemma4_Tokenizer_LoadsFromGguf` failed with `KeyNotFoundException`. The test now skips visibly on that condition. Not checked: the token type of id 1 in that file. Owner decision: should `<eos>` count as a special token for this export?
  - **Core default-run cost (not bounded yet):** a full `Tests.Core` run with no `-class` filter, after this change, ran for more than 15 minutes at about 14 GB working set. I stopped it. The classes that load real checkpoints are not identified yet. A gate or an explicit list is needed before a full run is used as a check.

##### A4. Non-finite logits in hybrid MoE decode

- **Why:** 1 in 18 control runs produced a NaN; 12 later runs did not. The control test asserts finite logits and names the run and step.
- **Next step:** run the control test in a loop with per-layer finite checks on, until it fires. Record the run index, the decode step, and the first layer that goes non-finite. Then compare that layer's inputs with the CPU pass. If the NaN comes from a fused kernel, check its scratch sizing and reads past the end of its buffer.
- [ ] **Done when:** the cause is fixed with a regression test that fails before the fix. If it is not reproduced, the instrumented repro is committed and names the layer.
- **Do not** repeat the 12-run check without the per-layer instrumentation. It adds no information.

##### A5. Device memory figure understates real use

- **Why:** on a Ryzen 7 5700G (Vulkan, every layer offloaded, context 512), dense Qwen3-0.6B Q8_0 held 1152.7 MiB device-local against a planned figure of 761.8 MiB. Hybrid-GDN Qwen3.5-0.8B Q8_0 held 1612.2 MiB against 827.8 MiB. The ledger matched the OS counters within 1% (`scripts/gpu-process-memory.ps1` reads the OS counters).
- **Causes, confirmed in the code:**
  1. The output head. Both Vulkan passes dequantise every output table except Q4_K and Q6_K (and Q3_K in the hybrid pass) to F32, Q8_0 included (`GpuForwardPass.cs` around line 1198; `VulkanHybridGdnForwardPass.UploadEmbeddingWeight`). A tied head is one F32 copy, but the planner counts two Q8_0 copies.
  2. `ModelDescription.MeasureLayerGpuBytes` sums a fixed list of plain-transformer tensor names. A hybrid layer's `attn_qkv`, `attn_gate` and `ssm_*` tensors are uploaded but not counted (195.3 MiB for Qwen3.5-0.8B).
  3. The weight figure leaves out KV cache, recurrent state and scratch.
- **Still open:** the Vulkan output head and the staging buffer. The planner changes that do not depend on Vulkan are already merged.
- **Next step:** count the head and the staging buffer in the planner for the Vulkan path. Rerun the ledger measurement for both models.
- [ ] **Done when:** the planned device figure is within 5% of the ledger for both models.

##### A6. Warm-up after a dense-llama handoff

- **Why:** the first decode after a CPU-prefill handoff on the Vulkan hybrid costs about 1.9 s (24 tokens at 6-7 tok/s; 160 tokens at 12.5-13 tok/s; steady rate about 15 tok/s). The cause is a hypothesis: GPU pipelines created on first use. It is not profiled.
- **Next step:** find where `HybridForwardPass` creates its GPU pipelines (the lazily created pipeline fields in `VulkanBackend`). Profile the first decode to confirm or reject the hypothesis. If confirmed, create the pipelines at load, or finish the handoff with one throwaway token.
- [ ] **Done when:** the 24-token decode rate matches the 160-token rate, and greedy output is unchanged.

##### A7. Handoff receipts for more dense families

- **Recipe:** run the env-selected test with `STINGRAY_HANDOFF_TEST_MODEL`, `STINGRAY_HANDOFF_TEST_GPU_LAYERS` and `STINGRAY_HYBRID_CPU_PREFILL=all`. Add a permanent row and a receipt only on a pass.
- **Candidates with a local checkpoint:** Qwen2.5-1.5B, gemma-2-2b-it, Granite (its scale fields are applied on the hybrid now). Not Phi-3: its fused `attn_qkv` has no path on the hybrid. About 20 to 45 s per configuration.
- [ ] **Done when:** the permanent rows pass without the override.

##### A8. Handoff cases not yet exercised

- **Runnable now:** contexts above 4096 on a dense family already on disk; Qwen2.5 Coder and VL.
- **Blocked (Part C):** LongRoPE and sliding-window llama, no checkpoint on this host.
- [ ] **Done when:** each runnable case has a passing row or a recorded refusal.

##### A9. MoE activation numerics

Source: [1-correctness/2026-10-02-prefill-numerics-investigation.md](1-correctness/2026-10-02-prefill-numerics-investigation.md), "Follow-ups from the MoE audit".

- [ ] **a. Small-prompt path:** measure the folded-dispatch path for prompts of 32 tokens or fewer. Decide whether it needs a special case. Record the measurement either way.
- [ ] **b. Per-weight-type comparison:** compare Q8_KS with ggml Q8_K for each weight type. The first two boxes are closed (`Q8ActivationQuantizationTests`). Next step: pick one weight type and compare the dot products on identical rows.
- [ ] **Done when:** each weight type has its comparison recorded, with the error next to the reference.

##### A10. Sweep the stale open boxes

Each file below has open checkboxes. For each box: check it against the code. Close it with the evidence, or keep it and write the blocker next to it. No box is left unexplained.

- [ ] [4-performance/2026-11-08-performance-plan.md](4-performance/2026-11-08-performance-plan.md): 16 boxes.
- [ ] [4-performance/2026-11-08-performance-plan.md](4-performance/2026-11-08-performance-plan.md): 11 boxes.
- [ ] [2-coverage/ported-families-todo.md](2-coverage/ported-families-todo.md): 14 boxes.
- [ ] [3-product-and-runtime/2026-11-08-product-plan.md](3-product-and-runtime/2026-11-08-product-plan.md): 10 boxes; all five live when read on 2026-11-03.
- [ ] [3-product-and-runtime/2026-11-08-product-plan.md](3-product-and-runtime/2026-11-08-product-plan.md): one open box (real-model fork on the GPU, hybrid and GDN passes).

##### A11. Housekeeping

- [x] Make `LtxVaeNoiseIsolationTest` write to a temp directory.
  - **Status 2026-11-08: already done.** The test writes to `Path.GetTempPath()` (`LtxVaeNoiseIsolationTest.cs:93`). Close the box.
- [x] Fix or quarantine the flaky `SimdKernels` timing test.
  - **Done 2026-11-10:** the test that asserted nothing is renamed to `SimdKernelsIq1STests.FastKernel_TimingIsRecordedNotAsserted`, so the name says what it does (it records a speed of about 0.8 to 0.9 times the fallback and gates nothing). `ForwardPass.Fast` builds in Release, and `-class OpenTail.Stingray.Tests.ForwardPass.Fast.SimdKernelsIq1STests` runs 3 of 3 with 0 failed and 0 skipped. No other reference to the old name.
  - **Status 2026-11-08: no flake found in the timing tests that assert.** `SimdKernelsBF16Tests.FastKernel_IsFasterThanDequantFallback` asserts `speedup > 1.0`. It passed 40 of 40 at `DOTNET_PROCESSOR_COUNT=2`, with speedups of 7 to 10 times. `SimdKernelsIq1STests.FastKernel_IsFasterThanDequantFallback` (renamed 2026-11-10, see the line above) asserts no speed at all: it prints a speedup that ran from 0.89 to 1.90 over 40 runs, and only its correctness checks can fail. The name says "IsFaster", so it is misleading. Fix: rename it, or assert a speed bound that the margins support. Not changed here.
- [ ] Add a CI step that builds and runs every test project. A sweep found three real failures in one background command.
  - **Status 2026-11-08: partly done.** `ci.yml` builds the whole solution (line 42) and runs 12 test projects by name. It runs the `.Fast` projects for Server, Sessions, ForwardPass and Audio, plus Vulkan.Fast and Mlx.Fast. Heavy and GPU projects are excluded on purpose (comment, lines 59-63). Not in CI at all: `Tests.Diffusion` and `Tests.Embeddings`. Run with defaults on this machine, neither has a failure. Diffusion: 535 tests, 260 passed, 275 skipped (259 behind the heavy flag, 16 for missing checkpoints), 0 failed, 489 s. Embeddings: 46 tests, 6 passed, 40 skipped (8 benchmarks, 32 checkpoints missing), 0 failed, 1.8 s. The 489 s run is slow for CI, so a subset would be needed. Not added to CI here.
  - **Progress 2026-11-10:** `Tests.Embeddings` is now in both CI jobs (Linux suite list and the Windows `$suites` array), with floor 5. It has 46 discovered tests and 6 that pass without model checkpoints. The floor counts executed passing tests, not discovered ones (floor 6 passes, floor 7 fails), so the floor is 5. The CI comments say skipped tests count toward the floors, which is wrong for any suite with skipped tests. The box stays open: `Tests.Diffusion` is still not in CI (489 s, see the status above).
- [x] Add "run `Tests.Core`" to the merge checklist.
  - **Status 2026-11-08: done.** The merge checklist is `.github/PULL_REQUEST_TEMPLATE.md`. It now has a line for `Tests.Core`. It did not name it before, and it asked only for the `.Fast` projects.

##### A12. OpenTelemetry acceptance boxes

Source: [3-product-and-runtime/2026-10-12-opentelemetry-observability-plan.md](3-product-and-runtime/2026-10-12-opentelemetry-observability-plan.md), section 12.

- [ ] **a. No-listener overhead:** show it is within the plan's bound (1%, and the baseline noise band), with a method that can isolate it. Also measure the NativeAOT CLI size and start-up effect, using the overhead procedure in the plan's section 8. Quiet machine, interleaved.
- [x] **b. Cost tiers:** every instrument has a cost tier (plan section 4.1). Test that only tier 1 does work with no listener attached, and that tier 3 output is labelled as perturbed.
  - **Status 2026-11-08: met by a passing test.** `EngineTelemetryTierGatingTests` (ForwardPass.Fast, 1 test, 0 skipped, 0.21 s) checks that the tier-2 gate is closed with no listener, with a tier-1-only listener, and that it opens with a tier-2 listener and closes again after it. It also checks that tier-3 instruments are absent without diagnostic mode and that their definitions start with "DIAGNOSTIC". Not re-checked here: that every instrument in code has a row in the tier table; `EngineTelemetryContractTests` is the place for that.
  - **a is not done:** the overhead bound needs a quiet machine and a method that isolates the no-listener cost. Left open.
- [ ] **c. Growth seams:** the seams in plan section 14. Add each one when a new metric needs it. Do not add them speculatively.
- [ ] **Done when:** a and b have measured or tested evidence on their line, and c is met or deferred with a reason.

##### A13. Heavy runs that need a quiet machine (one at a time)

- **Why:** two golden and parity classes have not been run in full: the `glm4moe` golden (GLM-4.5-Air Q2_K, 43 GB) and `LlamaFourGreedyParityTests` with `LlamaFourLongContextParityTests` (Llama-4-Scout, 44 GB). The Scout run mapped 43.9 GiB of experts and drove free RAM to 304 MiB before. Run each one alone.
- **Next step:** check free RAM and close other work. Run one class. Stop if free RAM goes under 2 GiB.
- [ ] **Done when:** each class has a result recorded, pass or fail with the output.

---

#### Part B: Waiting on you or a machine

##### B1. Resident-bytes accounting (choice needed)

- **Why:** `ModelRuntime.AcceleratorResidentBytesEstimate` is set once at load from the planned figure and never updated. No live GPU-residency counter exists in `src`. For MoE models the planned figure leaves out the routed experts, on purpose (`TierPlanner.MeasureLayerBytes` excludes `ffn_*_exps`). The Vulkan GDN pass sizes its cache from VRAM minus its uploads minus a 2 GiB margin, and picks CPU MoE when the predicted slots are under half the experts. So at plan time it is not known whether a GPU expert cache exists.
- **Choice:** (A) a plan-time ceiling, `min(routed expert bytes, expert cache budget)`, for Vulkan MoE plans. Simple, but every MoE model claims the rest of the device and blocks co-residency. (B) two figures: the planned figure, and a live figure from the loader. **Recommended: B.**
- [ ] **Done when:** the choice is recorded, and the chosen figure is in the runtime and the status output, with a test.

##### B2. Fair turn gate measurement

- **Why:** the gate is built and off by default. No throughput number exists for it.
- **Next step:** two resident models, one prefill-heavy and one decode-heavy, with and without `FairModelScheduling`. Per-model latency and total throughput, three interleaved runs.
- **Depends on:** B1 (co-residency depends on the memory figure).
- **Decision after:** whether to route `InferenceEngine` through the gate.
- [ ] **Done when:** the numbers are in [3-product-and-runtime/032-multi-model-inference-runtime-plan.md](3-product-and-runtime/032-multi-model-inference-runtime-plan.md).

##### B3. Phi-3.5-MoE: which checkpoint is real

- **Why:** two files share the name and size: `K:\_other_models` and `H:\_models`. They first differ at byte 11,648,045,057. The test's expected tokens were produced from the K: copy. The shared locator now picks the H: copy first, and the test fails there: the greedy text diverges at position 2. A second reference build (`llama-completion`, b10306) gives the same text as the engine on the H: copy for the first eleven tokens. So the engine is consistent with the H: file, and the H: file is a different checkpoint from the one the expected tokens came from. Neither copy is the 18,055,104,352-byte Hub file the source doc names, so the Hub hash cannot say which is the real download.
- **Owner decision:** which copy is the real download. The other may be corrupt or a different conversion.
- **Then:** pin the catalog and the test to the chosen file by hash. Delete the other copy only after checking its hash.
- [ ] **Done when:** `PhiMoeGreedyParityTests` passes on the pinned file and the catalog hash matches.

##### B4. Gemma 2 9B golden

- **Why:** the Gemma 1 path is done (golden 22/24 near-tie; a real defect was found and fixed). Gemma 2 9B is open.
- **Blocker:** the 9B file needs a download, and F: has 3.7 GB free. Put it on E: or H:, never F: (see [2-coverage/2026-10-31-gemma-family-plan.md](2-coverage/2026-10-31-gemma-family-plan.md)).
- **Next step:** check free space on E: and H:. Download only if the golden is still the goal.
- [ ] **Done when:** a Gemma 2 9B golden is recorded against the verified file.

##### B5. Bonsai2 PRISM claim (approval needed)

Source: [1-correctness/2026-10-03-unverified-port-claims.md](1-correctness/2026-10-03-unverified-port-claims.md). Six of its seven claims are blocked (Part C). Bonsai2 can move: 7.1 GiB.

- **Blocker:** its reference is stock llama.cpp b11556. The sandbox classifier refused to run that binary on 2026-11-03. **Approval needed:** allow the download and run of that pinned binary from the scratchpad, or name another reference.
- **Then:** teacher-forced golden and perplexity. At 0.5 tok/s, run once, overnight.
- [ ] **Done when:** the golden and perplexity are recorded against the pinned reference, and the claim is updated with the number.

##### B6. Piper licence for the first-run check

- **Why:** `scripts/qualify-tasks.ps1` passes chat, and speak and transcribe are skipped. The Piper voice is research-use only. Claude may not accept a licence for you.
- **Owner action:** run `pwsh scripts/qualify-tasks.ps1 -AcceptLicence` once.
- [ ] **Done when:** all three tasks pass in the script's output.

##### B7. MoE expert GEMM for prefill: choose the numerics

Source: [4-performance/2026-10-04-q4k-expert-gemm-plan.md](4-performance/2026-10-04-q4k-expert-gemm-plan.md), section 7.

- **Option A:** repack each used expert into scratch and reuse the existing repacked GEMM. Faster to build, but not bit-identical to the per-token path. It needs the policy decision.
- **Option B:** an exact tiled GEMM that keeps the per-token arithmetic. Bit-identical, more code.
- [ ] **Decision needed:** is an approximate but reproducible MoE prefill acceptable (option A), or must prefill stay bit-identical to per-token decode (option B)? Step 0 of the plan (the NLL gate) gives the evidence and touches no engine code; it can run first.

##### B8. Q5 API follow-ups: list or close

- **Why:** the item has no sub-items.
- [ ] **Decision needed:** list the open items from [reference/request-parameter-audit-2026-10-31.md](reference/request-parameter-audit-2026-10-31.md) and turn each into a row here, or close the item.

##### B9. Performance and scout follow-ups (after A5 and A6)

Start these after A5 and A6, because the memory figures and the handoff warm-up change what the measurements mean. Quiet machine only.

- [ ] **Allocation leftovers in the perf sweep (items 1.5 and 11.3):** measure whether allocation is a visible cost. If not, record "not worth it" with the measurement and close.
- [ ] **Scout, Hub limits:** cap and expose the Hub index limits. Small hardening; do it first.
- [ ] **Scout, semantics-changing metadata:** needs a signature schema change and a re-emit of the checked-in signatures. Do after the hardening.
- Both scout items need unit tests with no Hub calls.

---

#### Part C: Blocked or parked

Each item here needs hardware, a second machine, a checkpoint, or a reference that does not exist on this machine. Do not start them.

- **CUDA and discrete GPU (rule 13):**
  - Confirm the residual and logit scale on a CUDA machine for Granite and Gemma 3 (the Vulkan guard does not cover CUDA).
  - The CUDA tensor-parallel forward pass ([4-performance/gpu/2026-11-06-cuda-tensor-parallelism-plan.md](4-performance/gpu/2026-11-06-cuda-tensor-parallelism-plan.md)).
  - [9-external-hardware/90-external-hardware-work.md](9-external-hardware/90-external-hardware-work.md) (20 boxes).
  - [4-performance/audio/080-tts-asr-gpu-residency-checklist.md](4-performance/audio/080-tts-asr-gpu-residency-checklist.md) (31).
  - [4-performance/diffusion/diffusion-performance.md](4-performance/diffusion/diffusion-performance.md) (19).
  - [4-performance/gpu/vulkan-gemm/vulkan-gemm.md](4-performance/gpu/vulkan-gemm/vulkan-gemm.md) (11).
- **MLX and Metal:** native numerical tests need a working Metal device ([4-performance/gpu/2026-11-06-mlx-metal-backend-plan.md](4-performance/gpu/2026-11-06-mlx-metal-backend-plan.md)).
- **AVX-512 and ARM64 receipts:** no such machine. The docs already say so.
- **Clean-machine release verification:** the released package on a machine with no repo, no checkout SDK and no GPU driver changes. CI's NativeAOT self-test covers the AOT binary only. Also the second-machine boxes in [reference/nuget-release-checklist.md](reference/nuget-release-checklist.md) (9 boxes).
- **Checkpoints that do not fit in 63 GiB RAM, or do not exist:**
  - DeepSeek-V4 (see [2-coverage/2026-10-03-deepseek-v4-review-plan.md](2-coverage/2026-10-03-deepseek-v4-review-plan.md), 11 boxes); DeepSeek-V4.1; DeepSeek-V3.2.
  - MiniMax-H3 (9 boxes); [2-coverage/2026-10-03-streamed-residency-plan.md](2-coverage/2026-10-03-streamed-residency-plan.md) (6).
  - DeepSeek V1 port (no checkpoint on any drive).
  - Nemotron-H MoE (no checkpoint on this machine, and no free disk for one).
  - LongRoPE and sliding-window llama (A8).
  - Six of the seven claims in the unverified-port list (B5).
- **HunyuanVideo and Wan 2.2:** the owner has already decided "duplicate first" (reconciliation item R2, [1-correctness/2026-10-09-documentation-claim-reconciliation-plan.md](1-correctness/2026-10-09-documentation-claim-reconciliation-plan.md)). The work is blocked by the checkpoints and, for HunyuanVideo v1, by the missing independent reference.
- **Ported but not verified (rule 14):** these families stay internal and unadvertised. Status in [2-coverage/ported-families-todo.md](2-coverage/ported-families-todo.md).
- **Parked by owner:** the HTTP image API stays at 501 (`/v1/images/generations` and `/edits` are not connected to the real pipelines); Gemma 2 GPU soft-capping; the Qwen3-Coder NaN hunt (keep the canary line; reopen only if it fires); the browser UI for the local server.

### 2. Qwen 3.8 Flash Next (`qwen4exp`), in progress

Source: [2026-11-08-qwen38-remaining-work-plan](2-coverage/2026-11-08-qwen38-remaining-work-plan.md). Qwen 3.8 stays not admitted and not advertised until Phase 8 (CLAUDE.md rule 14). Its owner decisions (parity criterion, unsloth fork reference, validator behaviour, admission) are in that plan.

The parent of its open items is that plan's unchecked boxes. They are not copied here.

### 3. Items with no open parent

These items have no open owner document, so this file is their parent. Decide whether each one goes into Queue 1 or is dropped.

- [ ] **Memory returned to the OS after unload.** Two-model `singleslot` host, four switch cycles, working set after each unload; trace the private-memory growth of about 130 MB per cycle. Source: former hub item E4 (its 2026-11-05 measurements are in git history of this file).
- [ ] **Admit one real model with the playbook.** Candidate `ornith-ai/Ornith-1.5-35B-A3B-GGUF`. Start with `stingray scout -r <repo> -f <file> --budget <host RAM> -c <ctx>` before any download. Source: [scout plan](3-product-and-runtime/2026-10-09-checkpoint-scout-and-ai-admission-plan.md), [playbook](reference/architecture-admission-agent-playbook.md).
- [ ] **Configuration ownership.** List every config switch with its owner and mark the obsolete-looking ones for an owner decision. Source: [QoL plan](3-product-and-runtime/2026-11-08-product-plan.md).

Moved to their parents, no longer tracked here:
- Gemma 4 26B-A4B word-3 divergence: parent is [2026-11-07 TensorSharp reference comparison](1-correctness/2026-11-07-gemma4-tensorsharp-reference-comparison.md), which records the root cause as fixed.
- Protocol corpus status per case: parent is the product plan, item A4.
- Multi-model durable session restore: parent is the [032 multi-model plan](3-product-and-runtime/032-multi-model-inference-runtime-plan.md), phase 7.

### 4. Coverage ports (behind Qwen; rule 14 applies)

Ported families stay not admitted and not advertised until a real checkpoint and an independent reference verify them. Status ladder: [ported-families-todo](2-coverage/ported-families-todo.md).

Each port's open boxes live in its parent. They are not copied here:

1. **GLM-5.x**: parent is the [programme plan](2-coverage/2026-11-07-coverage-completion-programme-plan.md), Phase 2B.
2. **DeepSeek V4**: parent is the programme plan, Phase 3.
3. **DiffusionGemma**: parent is its [port plan](2-coverage/2026-10-03-diffusiongemma-port-plan.md), the Level 4 box.
4. **MiniMax-H3** and **Muse-Glimmer**: parents are their port plans, [MiniMax-H3](2-coverage/2026-10-03-minimax-h3-port-plan.md) and [Muse-Glimmer](2-coverage/2026-10-03-muse-glimmer-port-plan.md).
5. **Candidate reassessment**: parent is the programme plan, Phase 4.

Lower priority, not planned: AI21 Jamba, Kimi Linear, Arcee AFM, ServiceNow Apriel, Ant Ling (`bailingmoe2`), MiniMax-M2 (too large), DeepSeek-OCR v1 (no checkpoint), LTX-2.3 / 2.5 (a later campaign), ONNX shared-pipeline extraction (SenseVoice, Paraformer).

### 5. Performance

Resume from [4-performance/2026-11-08-performance-plan.md](4-performance/2026-11-08-performance-plan.md). Rules: [performance-measurement-rules](reference/performance-measurement-rules.md). Do not reopen the closed Q4_K repacked-GEMM investigation. When the owner resumes, this is the order:

- [ ] 1. SmolLM2 prefill at about 0.89x of llama.cpp: the Q4_K Path-2 GEMM is 65% of trunk time; then RoPE and attention.
- [ ] 2. int8 alignment with ggml Q8_K (ADR-0003 gap): the Q8_KS variant of the repacked Q4_Kx8 GEMM.
- [ ] 3. Batched prefill for partially offloaded dense models and the remaining per-token paths.
- [ ] 4. MTP on Qwen3.6-27B (UD-Q3_K_XL) is slower than plain (0.77-0.81x). Check the tensor-dtype coverage of the shared-decode kernels first. Source: [ADR-0002](reference/adr-0002-hybrid-gdn-mtp-chunked-prefill.md).
- [ ] 5. Raw-quant matvec bandwidth on Vulkan (gpt-oss MXFP4 decode, about 18 GB/s).
- [ ] 6. FLUX.1 DiT: 132 s against the 99.8 s sd.cpp target ([diffusion-performance Part 2](4-performance/diffusion/diffusion-performance.md#part-2-flux1-vulkan-gemm-handoff)).
- [ ] 7. Image-token prefill in VLMs, the remaining per-token segment path.
- 8. Cross-model DRY and perf-doc consistency: parent is [diffusion-performance Phase 9](4-performance/diffusion/diffusion-performance.md), an open box. Not copied here.
- [ ] 9. MusicGen / AudioGen performance and DRY passes.
- [ ] 10. CosyVoice3 ODE step count: the fewer-steps A/B by ear and with a Whisper round trip (never run; the default is still 10).
- [ ] 11. Stable Audio 3 `IStableAudio3Engine` consolidation.

The rest of the performance boxes are in [perf-sweep-plan](4-performance/2026-11-08-performance-plan.md), [diffusion-performance](4-performance/diffusion/diffusion-performance.md), [vulkan-gemm](4-performance/gpu/vulkan-gemm/vulkan-gemm.md) and [080](4-performance/audio/080-tts-asr-gpu-residency-checklist.md).

## Owner decisions and actions outside the plans

- [ ] **Release 1.0.10.** Not ready: no tag, no version bump, no pack. Everything since 1.0.9 stays unpublished until the owner says so.
- [ ] **Stable Audio 3 Small quality.** Waiting on the owner's listening verdict on `docs/audio-samples/sa3_small_*_POSTTRAINED_*`. If the post-trained clips are better, add GGUF loading.

The other owner decisions are in Part B under Queue 1, each with a recommended default.

## Rules

1. **Order.** Take items in this order.
2. **A stalled item** gets its blocker written down, in its plan and in [1-correctness/bugstofix.md](1-correctness/bugstofix.md) when it is a bug. Then the next item starts.
3. **Evidence.** "Done" needs the check in the item's Done-when line, with real weights and an independent reference where one applies.
4. **Estimates are guesses** unless an item says it was investigated. Record the real time taken when an item closes.
5. **Disk.** Do not download a checkpoint unless the item says so. Check `models/`, `models/_models/` and the other `_models` folders first.
