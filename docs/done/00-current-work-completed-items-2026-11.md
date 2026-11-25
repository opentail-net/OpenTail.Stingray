# Current work: outstanding items, quickest first - completed items

Ticked items moved from [docs/00-current-work.md](../00-current-work.md) on 2026-11-03. Each keeps its section heading and nested lines.

## 13. MoE variants of the recurrent families

  - [x] Admit Granite 4.0-H tiny/small (MoE) with PPL parity. Small DONE 2026-09-28 (top-k renormalisation fix; +1.2% residual at -c 2048 logged).

## 4. Performance

- [x] 2. **Stale: re-measured 2026-11-01 at 0.97x** (61.45 against 63.45 t/s, medians of 4 interleaved rounds, Q6_K, 550 tokens, CPU; detail in [plan 6 D2](../00-current-work.md#open-work-as-of-2026-11-03)). Qwen3.6-35B-A3B prefill was 0.63x of llama.cpp (Phase 8 of
   [done/2026-09-25-hf-top-downloads-coverage-plan.md](2026-09-25-hf-top-downloads-coverage-plan.md)).

## 1. Transcription `model` field (completed 2026-11-05)

  - [x] Transcription `model` selects an installed ASR pipeline, and an unknown name returns 400 that names the field (validation: PR 143; selection through `IWhisperPipelineProvider.Create(model)`: PR 165). Tests in `TranscriptionModelFieldTests` (13 pass).

## Evidence for the engine (added 2026-11-05, from the external review)

- [x] **E1. Long-running inference stability.** Done 2026-11-05: `LongRunSoakTests` (opt-in, `STINGRAY_SOAK=1`) passed on real Qwen3-0.6B Q8_0 on CPU, 30 greedy repeats in one process, identical output, working set flat after warm-up. Repeated prompts, long contexts and several sessions, looking for memory growth, stale state, and KV-cache corruption over time. Done when a soak run at temperature 0 gives identical output on every repeat, and resident memory is flat after warm-up.
  First step: write a soak test on the fake engine in `tests/OpenTail.Stingray.Tests.Core` that loops N prompts and asserts identical output and flat allocations. Then the same loop against a real small model on CPU.
- [x] **E2. CPU performance regression gate.** Done 2026-11-05, as a local gate: `scripts/perf-gate.ps1` measures prefill and decode for qwen2.5-0.5b and 1.5b (median of 3) and fails on a drop of more than 15% against `scripts/perf-baseline.json`, per machine. Hosted CI has no model files, so it cannot run this gate; run it before a release or after a kernel change. Checked: unchanged code passes (exit 0); a baseline inflated 2x fails all four metrics (exit 1).

## Owner action (does not block the list above)

- [x] **Run once:** `pwsh scripts/qualify-tasks.ps1`. Run 2026-11-06: chat, speak and transcribe all PASS with no licence flag (the default speak model is Kokoro, Apache-2.0; the Piper licence applies only if Piper is chosen with `-m piper-lessac`).

## Moved 2026-11-08 from 00-current-work.md

  - [x] *(stale, closed 2026-11-03: `lfm2moe` was admitted 2026-10-01; the batched drift was flash-64 attention's online softmax, not a batched-path bug; see STATUS and [bugstofix item 12](done/bugstofix-resolved-2026-10.md))* Admit LFM2-MoE (`lfm2moe`) with PPL parity. BLOCKED 2026-09-28 (timeboxed): PPL +6.3% and per-token vs batched disagree by 9%; see bugstofix.

## Sections moved from 00-current-work.md (2026-11-25)

Finished sections moved verbatim out of [../00-current-work.md](../00-current-work.md). Open sections stay there.

##### A1. Cancellation test fails as "Failed" under low CPU

- **Why:** `HotSession_CancellationRollsBackCacheAndDoesNotCommitRevision` failed once in 40 with `DOTNET_PROCESSOR_COUNT=2` (expected Cancelled, got Failed). `HotSession_CancellationDuringPrefillLeavesTheCursorUnchanged` failed once in 40. Neither fails alone (0 in 120). The same class failed on CI in `HotSessionMetricsMetadataTests.HotSession_TurnResult_ReportsCancelledFinishReasonOnCancellation`. Fixes already merged cover the queued-request and between-prefill-chunk cases. Not yet covered: a cancel during prefill or decode that the consumer sees as a normal end.
- **Next step:** loop the `Tests.Sessions.Fast` executable at `DOTNET_PROCESSOR_COUNT=2`, with a diagnostic that prints `FailureReason` for every non-Cancelled outcome of a cancelled turn. Capture the exception. Write a deterministic repro first, then fix.
- [x] **Done when:** 120 loops pass in a row under the same contention.
- **Status 2026-11-08: done.** `HotSession_CancellationRollsBackCacheAndDoesNotCommitRevision` passed 40 of 40 (baseline) and 120 of 120 more at `DOTNET_PROCESSOR_COUNT=2`, while another AI's `Tests.Core` run was active. The assertion already prints `FailureReason`, which is the diagnostic the next step asked for. Mutation check: `InMemorySessionStore.Cancel` recording `Failed` makes the test fail with "expected Cancelled, got Failed; failure reason [MUTATION]". The mutation was reverted.
- **Recurred 2026-11-25 on CI** (run 45570543687, `Tests.Sessions.Fast`): the same assertion, after 10 s, with "The test did not release decode." The test waited on `DecodeStarted` through an async continuation, which needs a thread-pool thread while the fake decode holds one. It now waits on a `ManualResetEventSlim` signalled by the fake's decode, and cancel and release run back to back on the test thread. Not reproduced locally: 0 of 30 full-suite runs under `DOTNET_PROCESSOR_COUNT=2` before the change, 0 of 20 with three busy loops, and 0 of 30 after; the change removes a dependency, it is not a proven root cause.
- **Note:** a hung test process can lock the output DLLs (MSB3021 or MSB3027 on the next build). `dotnet-stack report -p <pid>` shows where it is stuck.


##### A6. Warm-up after a dense-llama handoff

- **Why:** the first decode after a CPU-prefill handoff on the Vulkan hybrid costs about 1.9 s (24 tokens at 6-7 tok/s; 160 tokens at 12.5-13 tok/s; steady rate about 15 tok/s). The cause is a hypothesis: GPU pipelines created on first use. It is not profiled.
- **Next step:** find where `HybridForwardPass` creates its GPU pipelines (the lazily created pipeline fields in `VulkanBackend`). Profile the first decode to confirm or reject the hypothesis. If confirmed, create the pipelines at load, or finish the handoff with one throwaway token.
- [x] **Done when:** the 24-token decode rate matches the 160-token rate, and greedy output is unchanged.
  - **Status 2026-11-10: met on Qwen2.5-1.5B Q4_K_M, not by a fix; the original slow first decode did not reproduce.** Hybrid handoff (`-g 8`, `STINGRAY_HYBRID_CPU_PREFILL=all`, the path shows `HybridForwardPass` and no refusal), temperature 0, eight runs each: 24 new tokens 22.3 to 26.0 tok/s (median about 24.8); 160 new tokens 20.9 to 28.3 tok/s (median about 26.2). The 6 to 7 tok/s figure in the item could not be reproduced on this model, and the model it came from is not recorded in the docs. The single 12.9 tok/s run earlier was noise. SmolLM2-1.7B gave 15.7 to 21.6 tok/s for 24 tokens over three runs. Greedy output is unchanged because no code changed. The hypothesis (GPU pipelines created on first use) was not profiled, because no slow first decode showed up to profile. Re-open if the 6 to 7 tok/s case is found with its model name.


##### A7. Handoff receipts for more dense families

- **Recipe:** run the env-selected test with `STINGRAY_HANDOFF_TEST_MODEL`, `STINGRAY_HANDOFF_TEST_GPU_LAYERS` and `STINGRAY_HYBRID_CPU_PREFILL=all`. Add a permanent row and a receipt only on a pass.
- **Candidates with a local checkpoint:** Qwen2.5-1.5B, gemma-2-2b-it, Granite (its scale fields are applied on the hybrid now). Not Phi-3: its fused `attn_qkv` has no path on the hybrid. About 20 to 45 s per configuration.
- [x] **Done when:** the permanent rows pass without the override.
  - **Done 2026-11-18.** Ran `HybridCpuPrefillHandoffTests.Handoff_IsByteExact_AndDecodeAgreesWithSequentialPrefill` on the Vulkan device, with `STINGRAY_RUN_HEAVY_TESTS=1`: 25 cases, 0 failed, 2 skipped because Mistral-7B and Llama-3.1-8B checkpoints are absent (23 ran). The Qwen2.5-1.5B rows passed at `-g 4` (prefill cosine 0.999028; minimum decode 0.99921) and `-g 8` (0.998840; 0.99896). Granite 3.3 2B passed at `-g 4` (0.999037; 0.99872) and `-g 8` (0.999099; 0.99832). The full existing theory rows now supply the permanent test evidence; no test override was used. Earlier `PrefillHandoffFamiliesTests` and `ArchitectureRegistryTests` receipts remain as documented below.


##### A8. Handoff cases not yet exercised

- **Runnable now:** contexts above 4096 on a dense family already on disk; Qwen2.5 Coder and VL.
  - **Status 2026-11-10 (run; one case fails and is recorded below).** Env-selected handoff test, heavy on, no override needed for the receipts: (1) **Qwen2.5-1.5B, context above 4096:** 5,252 prompt tokens at context 8192 (`STINGRAY_HANDOFF_TEST_PROMPT_CHARS=24000`, `_CTX=8192`, `-g 4`): passes, prefill cosine 0.998909; 396 s, mostly the token-by-token sequential side. This is an env-selected run, not a permanent row, because the long-prompt test is not a theory row. (2) **Qwen2.5 Coder 1.5B `-g 4`:** passes, prefill cosine 0.999704, lowest decode cosine 0.99969; added as a permanent row (same `qwen2` receipt). (3) **Qwen2.5-VL-7B-Instruct text model `-g 4`: FAILS, recorded refusal.** Prefill logits cosine 0.989063 against the CPU, below the bar; decode cosines 0.9985 to 0.9999 over 150 tokens, so the K/V path agrees and the mismatch is in the prefill logits. Likely cause, not verified: Qwen2.5-VL's multi-section (M-RoPE) rotary encoding, which the hybrid may not apply to the text tokens; `Qwen3VlVulkanMRopeParityTests` covers a related path. Not admitted: the `qwen2vl` family has no receipt, so the planner does not hand off to it without the override. Follow-up: check the hybrid's rotary path against M-RoPE before any receipt.
- **Blocked (Part C):** LongRoPE and sliding-window llama, no checkpoint on this host.
- [x] **Done when:** each runnable case has a passing row or a recorded refusal.
  - **Done 2026-11-18:** Qwen2.5-1.5B long-context and Qwen2.5 Coder handoff pass; Qwen2.5-VL text-only handoff is explicitly refused after prefill-logit cosine 0.989063, below the 0.99 bar. LongRoPE and sliding-window llama remain asset-blocked in Part C.


##### A9. MoE activation numerics

Source: [1-correctness/2026-10-02-prefill-numerics-investigation.md](../1-correctness/2026-10-02-prefill-numerics-investigation.md), "Follow-ups from the MoE audit".

- [x] **a. Small-prompt path:** measure the folded-dispatch path for prompts of 32 tokens or fewer. Decide whether it needs a special case. Record the measurement either way.
  - **Done 2026-11-10: no special case needed.** `tests/OpenTail.Stingray.Tests.ForwardPass/MoeSmallPromptDispatchMeasurementTests.cs` (heavy; OLMoE-1B-7B Q4_K_M, CPU, min of 5 interleaved repeats, both arms in one process after warm-up). Batched MoE prefill vs folded sequential (the per-token decode-style dispatch): 8 tokens 100.0 vs 172.6 ms (sequential 1.73 times slower); 16 tokens 140.5 vs 340.6 ms (2.42); 32 tokens 226.1 vs 683.2 ms (3.02); 64 tokens 403.7 vs 1406.7 ms (3.48). Routing the short prompts through the folded decode-style dispatch would be slower, so the batched path stays for N of 32 or fewer. Found on the way, not this item's question: with default settings the two arms disagree on the greedy token at 8 tokens, and the largest logit gap is about 1.2 at every length (see item b).
- [x] **b. Per-weight-type comparison:** compare Q8_KS with ggml Q8_K for each weight type. The first two boxes are closed (`Q8ActivationQuantizationTests`).
  - **Status 2026-11-10: Q4_K done; Q3_K, Q5_K, Q6_K and Q8_0 open.** `tests/OpenTail.Stingray.Tests.ForwardPass.Fast/Q4KDotActivationErrorTests.cs`: 64 synthetic Q4_K rows of 2048 columns, dequantised by `Dequantize`, against a Gaussian activation with three outlier channels. Three numbers per row: the exact dot in double; Stingray's `DotQ4K_Q8KS` on Q8_KS activations (the production path); the dot with ggml-style Q8_K activations (one scale per 256). Error is normalised by the sum of absolute products. Normalised error, mean (max):
    - amax/rms 3: Q8_KS 1.61E-04 (6.5E-04), ggml Q8_K 2.19E-04 (8.1E-04)
    - amax/rms 25: Q8_KS 3.18E-04 (1.2E-03), ggml Q8_K 9.85E-04 (3.8E-03)
    - amax/rms 90: Q8_KS 1.20E-03 (5.8E-03), ggml Q8_K 2.68E-03 (8.9E-03)
    So the finer Q8_KS scale gives the smaller error on every Q4_K row tested, by 1.4 to 3.1 times. The result is int8 activation noise, not a defect. It agrees with the 2026-11-03 note: aligning Q8_KS with Q8_K would make the Q4_K dot product less accurate, so the per-type alignment is not wanted for Q4_K.
  - **Done 2026-11-18: comparison recorded for all five types.** `PerWeightActivationQuantizationErrorTests` uses 64 synthetic 2048-column rows per type and each amax/rms level (3, 25, 90), dequantises the same weights for both activation formats and compares normalized dot error against an FP64 dot. Q8_KS mean error (max) vs ggml Q8_K mean error (max):
    - Q3_K: 1.830E-04 (6.065E-04), 3.133E-04 (1.261E-03), 8.188E-04 (2.714E-03); Q8_K: 2.435E-04 (6.604E-04), 9.743E-04 (3.349E-03), 2.579E-03 (1.516E-02).
    - Q5_K: 1.988E-04 (8.341E-04), 3.764E-04 (1.358E-03), 9.619E-04 (3.618E-03); Q8_K: 2.290E-04 (8.766E-04), 8.891E-04 (3.736E-03), 3.168E-03 (1.276E-02).
    - Q6_K: 1.661E-04 (4.109E-04), 3.654E-04 (1.399E-03), 1.039E-03 (5.295E-03); Q8_K: 2.253E-04 (7.660E-04), 9.279E-04 (2.896E-03), 2.488E-03 (9.710E-03).
    - Q8_0: 1.677E-04 (6.633E-04), 3.296E-04 (9.203E-04), 9.331E-04 (4.113E-03); Q8_K: 1.870E-04 (7.102E-04), 7.278E-04 (2.188E-03), 2.617E-03 (7.419E-03).
    Q8_KS had the lower mean error on every tested type and level. These synthetic rows do not explain the historical SmolLM2 batched-Q8 versus llama.cpp +0.69% PPL delta (corrected attribution in A9c); the model-level investigation is now correctness item 25. `PerWeightActivationQuantizationErrorTests`: 12 passed, 0 failed, 0 skipped.
- [x] **Done when:** each weight type has its comparison recorded, with the error next to the reference.


##### A16. A failing estimator leaks a loaded engine (verified 2026-11-24)

- **Why:** `RunLoad` (`src/OpenTail.Stingray.Server/ModelRuntimeManager.cs` around lines 630-660) calls `_loader(model)` and then `_estimateBytes(model)` and the `ModelRuntime` constructor. If the estimator throws after the loader succeeded, the `catch` only fails the task. The `LoadedEngine` (engine, `RuntimeInstance`, GGUF mapping) is never disposed.
- **Next step:** compute the estimate before calling the loader (`_estimateBytes` takes only the `ModelId`, so the order does not change its input). Then a throwing estimator never loads anything.
- [x] **Done when:** a test whose estimator throws shows that the loader's engine is not left undisposed (either the loader is not called, or its engine's `Dispose` ran). The test fails on the code as it was before the fix.
  - **Done 2026-11-24.** Test `Acquire_EstimatorThrows_DoesNotLeaveALoadedEngineUndisposed`: before the fix the loader ran once and its engine was never disposed. The estimate is now computed before the loader runs. Mutation (old order restored) fails the test. Passes after the fix. Evidence as for A15 (same build and runs).
  - **Checked 2026-11-24 (iteration 47): not reachable, no change.** The gap was a `ModelRuntime` constructor that throws after a successful load. The constructor only assigns fields and calls `Touch()` (a clock read); nothing in it can throw, so the load path cannot leak there.


##### A17. Lost wake-up for single-slot waiters (verified 2026-11-24, narrow window)

- **Why:** `ModelRuntimeHandle.Dispose` (`src/OpenTail.Stingray.Server/ModelRuntime.cs` around lines 306-316) calls `OnHandleReleased` outside `_lock`, then `NotifyResidencyChanged` (`ModelRuntimeManager.cs` around line 435), which swaps `_residencyChanged` with `Interlocked.Exchange` and does not take the lock. A waiter reads `HandleCount > 0` and then reads `_residencyChanged.Task`. If a release lands between those two reads, the waiter gets the new, never-completed task and waits for a later event.
- **Next step:** capture `_residencyChanged.Task` before the handle-count checks. Any release after the capture then completes the captured task.
- [x] **Done when:** a stress test (many iterations, one held handle released concurrently with a second acquire) completes within its timeout every time, and the fix is in. A deterministic test needs a seam; record it if one is not built.
  - **Done 2026-11-24 (iteration 47): the window is now tested deterministically.** Test `SingleSlot_HandleReleasedAfterTheBusyCheck_StillWakesTheWaiter` (`tests/OpenTail.Stingray.Tests.Server.Fast/ModelRuntimeManagerTests.cs`) releases the held handle from a test seam, `ModelRuntimeManager.SingleSlotWaitHookForTests` (null in production), at the point between the busy check and the wait. Mutation (the old read of the residency signal after the busy check): the test fails with `TimeoutException`; with the fix it passes. `ModelRuntimeManagerTests` 68 of 68; `Server.Fast` full run 715 total, 0 failed, 2 skipped (exit 0).
  - **Earlier note (2026-11-24, before the seam):** the fix was verified by reading only. `AcquireAsync` captures the residency task (`Volatile.Read(ref _residencyChanged).Task`) before the busy checks, and the single-slot wait uses that captured task (`src/OpenTail.Stingray.Server/ModelRuntimeManager.cs`). The release decrements the handle count before it notifies, so a task captured earlier cannot miss the release. Stress test `SingleSlot_HandleReleasedWhileAnotherModelIsAcquiring_NeverLeavesTheWaiterHanging` (200 rounds, 5 s timeout each) passes on the new code (5 runs) **and on the old code (1 run)**, so it is a smoke test only, not evidence of the fix. The seam between the busy check and the wait is now built (see the done note above).


##### A18. `ModelRuntime.Dispose` skips the model when the engine dispose throws (verified 2026-11-24)

- **Why:** `ModelRuntime.Dispose` (`src/OpenTail.Stingray.Server/ModelRuntime.cs` around lines 265-279) calls `Instance?.Dispose()` and then `Engine.Dispose()` with no try/finally. If either throws, `instance.Model.Dispose()` never runs, so the GGUF mapping leaks, and the state is already `Disposed`, so a retry does nothing.
- **Next step:** put the model disposal in a `finally`, and keep the `Disposed` state set once.
- [x] **Decided 2026-11-24: keep the original `ModelRuntime.Dispose`.** A `try`/`finally` that frees the GGUF mapping after an engine `Dispose` throws was tried and reverted: when an engine's dispose throws, the state of its worker is unknown, so freeing the mapping could be a use-after-free while the worker still reads the weights. The original code skips the model on that path, so the GGUF mapping leaks there. That leak is the recorded trade-off. **Owner may reverse this** (for example, if a process that hits this path must reclaim memory more than it must avoid the rare risk); the reversal needs a decision on what a throwing engine means for its worker.


##### A19. Fresh cold callers ignore in-flight loads in the byte budget (fixed 2026-11-24)

- **Why:** the admission estimate (`TryEnsureAdmissibleLocked`) counts resident bytes only; the design note on `WakeOldestAdmissibleQueuedWaiter` accepts this for queued waiters (one wake at a time). A fresh caller does not go through that gate, so two cold callers can pass the budget against the same headroom.
- **Next step:** count the estimated bytes of `_pendingLoads` in the admission estimate, after A15 is in.
- [x] **Done when:** a test with two cold callers and a budget that fits one shows the second one waits or is refused.
  - **Done 2026-11-24.** `EstimateAdmissionLocked` (`src/OpenTail.Stingray.Server/ModelRuntimeManager.cs`) adds the pre-load estimate of every in-flight load of another model to the candidate's bytes, and to its accelerator bytes when the candidate has an accelerator footprint on the same target. It is used by both admission checks and by the queue's wake scan. Test `FreshCallerIsRefusedWhileAnotherModelIsStillLoadingAndFillsTheBudget` (`tests/OpenTail.Stingray.Tests.Server.Fast/ModelRuntimeManagerTests.cs`, new `ByteCapacityBudget` fake, which counts resident bytes only, like the real host budget): before the fix the second caller was admitted (`No exception was thrown`); the mutation (in-flight term removed) fails it again; passes after the fix.
  - **Evidence:** `ModelRuntimeManagerTests` 66 total, 0 failed, three runs; the A19 test 1 of 1; full `Server.Fast` 703 total, 0 failed, 2 skipped (exit 0). Release build 0 warnings, 0 errors. No existing admission test changed; the test file has additions only.
  - **Decisions taken (recorded):** (1) the 1.25 safety margin of the budget now applies to in-flight bytes too. This is more conservative than a single load, and it is the default kept here. (2) A pending load stores no estimate; it is computed when needed, as `Snapshot` already does. A stored estimate is a small follow-up. (3) The wake scan counts in-flight loads too. **Gap closed 2026-11-24:** test `QueuedWaiterIsNotWokenWhileAnotherModelIsStillLoadingAndFillsTheBudget_AdmittedOnceThatLoadIsReleased` (`tests/OpenTail.Stingray.Tests.Server.Fast/ModelRuntimeManagerTests.cs`). It queues a waiter while another model loads, releases an unrelated resident handle so the wake scan runs (the scan evicts the idle model, which shows that it ran), and asserts the queued waiter is still refused. The **wake-only mutation** (the in-flight term removed from the wake scan alone, `WakeOldestAdmissibleQueuedWaiter`) fails it at `Assert.False(acquireC.IsCompleted)`; the fresh-admission mutation also fails it. Restored; the test passes. Evidence: `ModelRuntimeManagerTests` 67 total, 0 failed, three runs; full `Server.Fast` 704 total, 0 failed, 2 skipped. Release build 0 warnings, 0 errors.
  - **Caveat:** the real `HostResourceBudget` checks host headroom from GC information. Before this fix an in-flight load was invisible to it as well, so this matters on a real host too.


##### A20. Evict-then-fail in the queue wake path (policy; default: keep as documented)

- **Why:** `WakeOldestAdmissibleQueuedWaiter` calls `EvictIdleOthersLocked(null)` before it picks a winner, so when nobody is admissible the idle models are still evicted. `TryEnsureAdmissibleLocked` does the same and is documented that way.
- **Recommended default (owner-level, taken without asking):** keep the documented behaviour. Record it; change only if the owner asks.
- [x] **Decision recorded 2026-11-24:** keep as documented.


##### A22. A nested GGUF array overflows the stack (verified 2026-11-24, high impact: a crash, not an exception)

- **Why:** `ReadGgufArray` (`src/OpenTail.Stingray.Core/GgufModel.cs` around lines 645-657) and `SkipGgufValue` (lines 688-695) call each other with no depth limit. An array whose element type is also an array nests one level per 12 bytes of file, so about 1 MB of header gives roughly 80,000 levels and a stack overflow. A stack overflow ends the process and cannot be caught.
- **Next step:** pass a depth counter through both methods and refuse above a small limit (8 is enough for every real GGUF) with `InvalidDataException`.
- [x] **Done when:** a file with a 64-deep nested array is rejected with `InvalidDataException` (the test fails before the fix, because nothing is thrown) on both the read path and the shard-skip path. The test must not overflow the stack itself.
  - **Done 2026-11-24.** Depth limit 8 (`MaxArrayNestingLevels`, `src/OpenTail.Stingray.Core/GgufModel.cs`): 8 nested levels are accepted, the ninth is refused. Tests `Open_ArrayNestedSixtyFourDeep_IsRefused` and `Open_ShardSkippedArrayNestedSixtyFourDeep_IsRefused` (`tests/OpenTail.Stingray.Tests.Core/GgufUntrustedInputTests.cs`) fail before the fix (`No exception was thrown`); with the depth check disabled 2 of 12 fail. **Open:** the shard-skip test goes through `GgufModel.Open` with split file names; `ParseIndex`'s skip path is not tested directly. **Checked 2026-11-24 by reading:** `ParseIndex` calls `ParseShard` for every shard (`GgufModel.cs` around line 416), and `ParseShard` calls `ParseTensorInfos`, so the row-length rule of A24 and the rank check of A23 apply to every shard that `ParseIndex` reads.


##### A23. A tensor rank is allocated before it is checked (verified 2026-11-24)

- **Why:** `ParseTensorInfos` (`src/OpenTail.Stingray.Core/GgufModel.cs` around lines 436-442) runs `new long[nDims]` on the file's `uint` before the `nDims > 8` check at line 454. A rank of `0xFFFFFFFF` asks for a huge array and fails with an out-of-memory or overflow exception, not `InvalidDataException`.
- **Next step:** move the `nDims > 8` check before the allocation.
- [x] **Done when:** a tensor entry with rank `0xFFFFFFFF` throws `InvalidDataException` (fails before the fix) and nothing is allocated for it.
  - **Done 2026-11-24.** The rank check now runs before `new long[nDims]`. Test `Open_TensorRankOfUInt32Max_IsRefusedBeforeAllocation` failed before the fix with `OverflowException` at the allocation, and passes after.


##### A24. A quantized tensor whose first dimension is not a block multiple is under-counted (verified 2026-11-24, medium-high: a possible read past the mapped view)

- **Why:** `DTypeInfo.ByteSize` (`src/OpenTail.Stingray.Core/Tensor.cs` line 313) computes `(elementCount / blockSize) * bytesPerBlock`, which floors. A Q4_K tensor with dims `[1, N]` reports too small a size, so the bounds check in `ParseTensorInfos` (around lines 475-511) passes. Kernels that read a whole block per row can then read past the end of the mapped file. GGML itself requires the first dimension to be a multiple of the block size.
- **Next step:** in `ParseTensorInfos`, reject a quantized tensor whose element count is not a multiple of its block size, with `InvalidDataException`. Unquantized types (block size 1) are unaffected.
- [x] **Done when:** a Q4_K tensor with first dimension 1 at the end of the file is rejected with `InvalidDataException` (fails before the fix), and an F32 tensor of the same shape still loads.
  - **Done 2026-11-24, with the rule corrected on review.** The first version checked the total element count. Review showed that is not enough: a `[100, 256]` Q4_K tensor has 25600 elements (a whole number of blocks), but its 100-element rows straddle blocks. GGML's rule is on the first dimension, the row length. The check is now `Dimensions[0] % blockSize`, for quantized types only. Tests: `Open_Q4KTensorWithElementCountNotBlockMultiple_IsRefused` (`[1, 100]`), `Open_Q4KTensorWhoseRowLengthIsNotBlockMultiple_IsRefusedEvenWhenTheTotalIsABlockMultiple` (`[100, 256]`, which fails with the element-count mutation: `No exception was thrown`), and the F32 control `Open_F32TensorWithSameShape_Loads`. The `[100, 256]` case is the one that matters for the read past the view.


##### A25. Bad `general.alignment` and `general.architecture` values throw the wrong exception (verified 2026-11-24, low impact)

- **Why:** `Convert.ToInt32(alignObj)` (`GgufModel.cs` lines 324-325 and 516-517) accepts any value type: alignment 0 gives `DivideByZeroException` in `AlignUp`, a negative value gives a wrong data start, a value above `int.MaxValue` gives `OverflowException`, and a string gives `FormatException`. `(string)archVal` (around line 345) throws `InvalidCastException` for a non-string architecture.
- **Next step:** accept only an integer type whose value is a positive power of two no larger than 1 MiB for alignment, and a string for the architecture; anything else throws `InvalidDataException`.
- [x] **Done when:** alignment 0, -1, 3 (not a power of two), 2^40 and the string `"x"` each throw `InvalidDataException`, and a non-string architecture does too (each test fails before the fix).
  - **Done 2026-11-24.** `ParseAlignment` accepts only an integer GGUF value that is a positive power of two no larger than 1 MiB; `general.architecture` must be a string. Tests `Alignment_*` (zero, negative, not a power of two, 2^40, string) and `Architecture_Integer` fail before the fix with `DivideByZeroException`, no exception, `OverflowException`, `FormatException` or `InvalidCastException`, and pass after. Control `Alignment_PowerOfTwoWithinLimit_Loads` passes before and after.
  - **Evidence for all four:** `GgufUntrustedInputTests` 13 total, 0 failed; `GgufModelTests` 29, `GgufIndexTests` 13, `GgufModelIntegrationTests` 2, all pass. Core build 0 warnings, 0 errors. `Server.Fast` 704 total, 0 failed, 2 skipped. The Core project as a whole was not run (it is over 15 minutes).


##### A26. A request asking for more new tokens than the engine allows is not refused with 400 (verified 2026-11-24, by reading and a test)

- **Why:** the validator only checked that `max_tokens` is positive. The engine refuses any request above `MaxBufferedOutputChunks - 4` = 4092 (`src/OpenTail.Stingray.Engine/ContinuousBatchingEngine.cs`, `ValidateOutputBound`), with an `ArgumentOutOfRangeException`. No exception handler maps it, so from reading the code the server answers 500 (a server failure) for a client's oversized request. For a streaming request the 200 is already sent before the engine's first step, so the stream is cut short instead. **Not tested end to end:** the 500 and the cut-off stream are read from the code, not observed.
- **Fix:** `ContinuousBatchingEngine.MaxNewTokensLimit` is public; `RequestParameterValidation` refuses `max_tokens`, `max_completion_tokens`, `max_output_tokens` and the Anthropic `max_tokens` above it with a 400 that names the field and the limit.
- **Behaviour change (recorded):** a request above 4092 is now a 400. Before, it was a 500 or a cut-off stream, so no client could have relied on it being served. `ClientDisconnectTests` sent `max_tokens = 5000`, which was never servable; it now sends 4000, which keeps the generation long enough for the disconnect test.
- [x] **Done when:** a request above the limit is refused by the validator with a 400 message (`MaxTokens_AboveTheEngineLimit_IsRefusedWith400NotPassedToTheEngine`, `tests/OpenTail.Stingray.Tests.Server.Fast/ServerLibraryTests.cs`, fails when the check is removed: `Assert.NotNull() Failure: Value is null`), and the at-limit value is still accepted.
  - **Evidence:** `Server.Fast` full run 705 total, 0 failed, 2 skipped (exit 0); `ClientDisconnectTests` 2 of 2 on two runs. The engine's own limit is unchanged in value (only exposed), so engine behaviour is the same.
  - **Not done:** an end-to-end HTTP check that a 5000-token request gets 400 (the validator test covers the decision, not the HTTP response). Add one if the owner wants the status observed rather than read.


##### A27. A stop string can be arbitrarily long, and each token pays for its length (verified by reading 2026-11-24; capped)

- **Why:** `StopSequences` (`src/OpenTail.Stingray.Server/Endpoints/StopSequences.cs`) holds back up to `stop.Length - 1` characters and, for every token, rebuilds that text as a string, searches each stop string in it, and runs `LongestStopPrefixAtEnd` over every stop string. So the cost per token grows with the stop string's length, and the body limit (30 MB by default) allowed a single stop string of that size. **Not measured:** the cost is read from the code.
- **Fix (done):** `StopSequenceList` refuses a stop string longer than 256 characters with a 400 that names the field and the limit (`RequestParameterValidation.cs`). **Owner decision, recommended default taken:** 256 characters; real stop sequences are a few characters. The owner may choose another cap.
- [x] **Done when:** a 257-character stop string is refused and a 256-character one is accepted (`StopSequence_LongerThan256Chars_IsRefused_AtTheLimitIsAccepted`, `tests/OpenTail.Stingray.Tests.Server.Fast/ServerLibraryTests.cs`; fails with `Assert.NotNull() Failure: Value is null` when the check is removed; passes now). Server.Fast 706 total, 0 failed, 2 skipped.


##### A31. Device-to-host reads are slow on the iGPU: the download staging buffer is not host-cached (performance lead, measured 2026-11-24, not changed)

- **Measured:** `Download` of a 9.6 MiB device tensor takes 52 ms (about 185 MB/s); of 15.7 MiB, 85 ms. An allocating `Upload` of the same sizes takes 2.6 and 3.7 ms. A device-to-pinned copy takes 0.5 ms, and a host write into pinned memory 1.4 ms; reading pinned memory from the host takes 305 ms for 9.6 MiB. `PrefillAttentionGateTests.PinnedActivationRoundTrip_WithDeviceCopies_IsTimed` and the plan's round-trip decomposition (iteration 42) hold the figures.
- **Cause, read in code:** `VulkanBackend.Download` reads a staging buffer created by `GpuBuffer.CreateStaging` with `HostVisible | HostCoherent` only (`GpuBuffer.cs` line 69), with no `HostCached`. On many drivers that memory type is write-combined, and host reads from it are slow. The cause is inferred from the rate and the code, not from the memory-type list of this device, which is not queried yet.
- **Size of the effect:** decode-time readbacks are small. Logits for a 50k vocabulary are about 200 KiB per token, about 1 ms per token at the measured rate (an estimate, not measured), which is a few percent at 20 tokens per second. The per-layer router readback is 64 floats and latency-bound. The effect is large only for big readbacks, such as activations for a host-side MoE step (Phase 2b).
- **Next step:** query the device's memory types for a `HostVisible | HostCached` type (`VulkanBackend` memory-type helpers); if one exists, use it for the download staging buffer behind a measured switch; measure `Download` of the 9.6 and 15.7 MiB blocks before and after, and of the logits block; keep the change only if the read rate improves measurably. If no such type exists, record that and close the item.
- [x] **Done when:** the memory-type query is recorded, `Download` is measured before and after the change on this iGPU, and the change is kept or dropped on the numbers.
  - **Status 2026-11-24 (iteration 43): done; the change is kept and is on by default.**
  - **Memory types (`MemoryTypeQueryTests`, this iGPU):** 16 types. Type 1 is `HostVisible | HostCoherent` (heap 1, 32,153 MiB of system memory), which is what the ordinary staging buffer takes (`FindMemoryType` returns the first match). Type 3 is `HostVisible | HostCoherent | HostCached` on the same heap. So the cached type exists, and the ordinary staging type lacks it.
  - **Timing (`DownloadStagingTimingTests`, median of 5 after 2 warm-ups, alternated twice):** a 9.56 MiB block 51.83 ms ordinary against 1.19 ms cached (about 44 times); a 15.73 MiB block 85.39 ms against 1.82 ms (about 47 times); the 50,304-float logits block 1.12 ms against 0.09 ms (about 12 times). The data came back unchanged in every run.
  - **Change:** `GpuBuffer.CreateStagingHostCached` (falls back to `CreateStaging` if no matching type exists), `VulkanBackend.DownloadStagingHostCached` (default `true`; set `false` to revert), and one helper, `EnsureDownloadStaging`, used by both download paths.
  - **Regression evidence:** the Vulkan test project in default mode: 369 total, 0 failed, 358 skipped (heavy-gated). Heavy-gated, the download classes pass: `PrefillAttentionGateTests` 4 of 4, `ExpertSlotLeaseTests` 1 of 1, `ExpertSlotManagerConcurrencyTests` 1 of 1, `DownloadStagingTimingTests` 1 of 1, `MemoryTypeQueryTests` 1 of 1. CLI A/B on OLMoE (`stingray run`, `--temp 0`, `-n 32`, the baseline build copied before the change): the full-GPU path (`-g -1`) gave the same sentence in both builds. The hybrid path (`-g 8`) varies from run to run on the baseline build too (3 runs: two "beauty" and one "creativity"), so it is not a difference from this change; see A32.
  - **Plan effect:** with the cached download the Phase 2b round trip is about 3.8 ms at 1,224 rows and 5.5 ms at 2,014 (cached download plus allocating upload), against break-evens of 26 and 83 ms. The attention-only projection becomes a gain of about 3.7% and 7.5% of prefill. That is a change of decision input, recorded in the plan; starting Phase 2b remains the owner's decision.


##### B1. Resident-bytes accounting (choice recorded: B)

- **Why:** `ModelRuntime.AcceleratorResidentBytesEstimate` is set once at load from the planned figure and never updated. No live GPU-residency counter exists in `src`. For MoE models the planned figure leaves out the routed experts, on purpose (`TierPlanner.MeasureLayerBytes` excludes `ffn_*_exps`). The Vulkan GDN pass sizes its cache from VRAM minus its uploads minus a 2 GiB margin, and picks CPU MoE when the predicted slots are under half the experts. So at plan time it is not known whether a GPU expert cache exists.
- **Choice:** (A) a plan-time ceiling, `min(routed expert bytes, expert cache budget)`, for Vulkan MoE plans. Simple, but every MoE model claims the rest of the device and blocks co-residency. (B) two figures: the planned figure, and a live figure from the loader. **Recommended: B.**
- **Choice recorded 2026-11-24: B**, the recommended option, taken by the AFK loop because the owner is away (owner decision deferred to the loop by the standing AFK rule). Admission keeps the planned figure unchanged. The live figure is additive: Vulkan only, measured per backend as the device-local bytes that the backend's live allocations hold; CUDA stays "not measured" until a CUDA device can verify it. Status shows both figures.
- **Stage status 2026-11-24:** stages 1 to 3 are done (see the Done-when line below). B2 is now unblocked.
- [x] **Done when:** the choice is recorded, and the chosen figure is in the runtime and the status output, with a test.
  - **Done 2026-11-24 (Vulkan only; CUDA and CPU report "not measured").** Stage 1: `VulkanBackend.LiveDeviceLocalBytes` sums the driver's `allocationSize` of live device-local buffers allocated through that backend instance (`GpuBuffer.Create` and `Dispose`; host-visible staging excluded). Stage 2: `LoadedEngine.LiveAcceleratorBytes` is read from the loaded backend when a Vulkan load completes; no API change was needed. Stage 3: `ModelRuntime.LiveAcceleratorResidentBytes` and `ModelRuntimeStats`/`MultiModelRuntimeStats` carry it (null when unmeasured; the sum covers only measured runtimes). The `InsufficientAcceleratorMemory` message prints both "planned" and "live". Admission still uses the planned figure, and the planned sums are unchanged.
  - **Evidence:** Release builds with 0 warnings and 0 errors (Server, Server.Fast, Vulkan, Tests.Vulkan). `Server.Fast` full run: 697 total, 0 failed, 2 skipped, exit 0, 16.7 s. The new Vulkan test `VulkanLiveDeviceLocalBytesTests` with `STINGRAY_RUN_HEAVY_TESTS=1`: 1 of 1 passes on the integrated AMD device. Mutation check (the free no longer subtracts): the test fails with `Expected 5376, Actual 4199680`, then the line was restored and the test passes.
  - **Limits:** the live figure is read once at load, so later allocations are not reflected. A fresh backend already holds 5,376 device-local bytes. It is not measured for real checkpoints in an assertion yet (the real-model test runs the loader but does not assert the figure). A real-model value check is a follow-up.


##### B2. Fair turn gate measurement

- [x] **Status 2026-11-24: done.** The numbers are in the 032 plan under Slice 5 (measured 2026-10-31 with the 1.5B and 0.5B Qwen2.5 pair, and re-run on a quiet machine on 2026-11-24). The gate stays off for CPU models. A B2 harness was started on 2026-11-24 with SmolLM2 models, but it stopped because those checkpoints are not on this machine, and it was not needed: the existing `FairGateMeasurementTests` already runs this protocol.
  - **Open (new):** the prefill model took 7.9 s in the 2026-10-31 record and 44.8 s on 2026-11-24, for the same three prompts. Not investigated. Check whether this is a regression before anything else uses it as a baseline.
  - **Not measured:** the shared-GPU case (two engines on one Vulkan device with no arbitration). The 032 plan says this is the one case where the gate could help. This machine has an iGPU, so it could give partial evidence, but per CLAUDE.md rule 13 it would not settle the question.

- **Why:** the gate is built and off by default. No throughput number exists for it.
- **Next step:** two resident models, one prefill-heavy and one decode-heavy, with and without `FairModelScheduling`. Per-model latency and total throughput, three interleaved runs.
- **Depends on:** B1 (co-residency depends on the memory figure).
- **Decision after:** whether to route `InferenceEngine` through the gate.
- [x] **Done when:** the numbers are in [3-product-and-runtime/032-multi-model-inference-runtime-plan.md](../3-product-and-runtime/032-multi-model-inference-runtime-plan.md). Met 2026-11-24. Decision (the plan's own conclusion, recorded here): the single-request `InferenceEngine` is not routed through the gate.


##### B4. Gemma 2 9B golden

- **Why:** the Gemma 1 path is done (golden 22/24 near-tie; a real defect was found and fixed). Gemma 2 9B is open.
- **Blocker:** the 9B file needs a download, and F: has 3.7 GB free. Put it on E: or H:, never F: (see [2-coverage/2026-10-31-gemma-family-plan.md](../2-coverage/2026-10-31-gemma-family-plan.md)).
- **Next step:** check free space on E: and H:. Download only if the golden is still the goal.
- **Progress 2026-11-10:** the blocker above no longer holds: E: has 128 GB free. Scout (`--budget 40G -c 4096`): architecture `gemma2` admitted, no static blockers, estimated peak 13.4 GiB, fits. Downloaded `bartowski/gemma-2-9b-it-GGUF` at revision `d731033f3dc4`, file `gemma-2-9b-it-Q4_K_M.gguf` (5.37 GiB) to `E:\_models\gemma-2-9b-it-GGUF`. SHA-256 verified by the downloader: `13b2a7b4115b...`, matching the hash scout read. Golden capture is next: three teacher-forced 48-token cases with the 2B golden's prompts, from the vendored llama-server. It needs the GPU and CPU free, so it runs after the Vulkan heavy suite.
- [x] **Done when:** a Gemma 2 9B golden is recorded against the verified file.
  - **Done 2026-11-10.** Golden `tests/OpenTail.Stingray.Tests.ForwardPass/Goldens/gemma2-9b.golden.json` (3 teacher-forced 48-token cases: `prose-1`, `prose-2`, `prose-3`, the 2B prompts; llama-server 10306 (6b5c2efb4); file sha256 `13b2a7b4115b…`). `stingray admit-arch -m E:/_models/gemma-2-9b-it-GGUF/gemma-2-9b-it-Q4_K_M.gguf --golden …`: `gemma2: PASS (NearTie)`. `prose-1` 47/48 matched (one near-tie where the reference was undecided, margin 0.066); `prose-2` and `prose-3` Exact 48/48. Stepwise vs prefill: argmax agrees. Decode 2.2 to 3.9 tok/s on CPU. The golden has no machine paths (checked). Row updated in `docs/models/gemma.md`.


##### A10. Sweep the stale open boxes

Each file below has open checkboxes. For each box: check it against the code. Close it with the evidence, or keep it and write the blocker next to it. No box is left unexplained.

- [x] [4-performance/2026-11-08-performance-plan.md](../4-performance/2026-11-08-performance-plan.md): the "16 boxes" and "11 boxes" entries are stale. On 2026-11-10 the file has no `- [ ]` line at all, so nothing is left to check or close here. Closed with that evidence.
- [x] [3-product-and-runtime/2026-11-08-product-plan.md](../3-product-and-runtime/2026-11-08-product-plan.md): the "10 boxes" entry is stale: the file has no `- [ ]` line on 2026-11-10. Its one real open item is the section `### A10. Forking: real-model test on the GPU, hybrid and GDN passes`, which is not a checkbox. Blocker: each pass needs its own real-model fork test (the `HotSessionForkRealModelTests` pattern) on a named checkpoint. Nothing is written yet. Sweep 2026-11-25: the file has no checkboxes left; the forking item is a section, kept with the blocker above.
  - **Blocked 2026-11-10 (engine work, not a test): the session fork path cannot run on these passes.** `HotSessionForkRealModelTests` builds `ContinuousBatchingEngine`, whose constructor takes `IBatchedForwardPass`. Only the CPU `ForwardPass`, `CudaForwardPass` and `Qwen4ExpBatchedForwardPass` implement it. `GpuForwardPass`, `HybridForwardPass` and `HybridGdnForwardPass` implement `IForwardPass` only. A real-model fork test on the Vulkan GPU, hybrid or GDN pass therefore needs batched decode and prefix-cache fork support in those passes first; the test alone would not compile against the engine. Owner decision: build that batched support, or defer the item.
- [x] [2-coverage/ported-families-todo.md](../2-coverage/ported-families-todo.md) (sweep closed 2026-11-25: the 7 open tasks each keep a blocker; lines 78–93 are the template): 16 unchecked lines; the "14 boxes" count was stale. Lines 78 to 93 are the per-family checklist, a template that applies to every family, so they are kept as the template and not as tasks. The other seven are kept, each with its blocker: line 109 Qwen 3.8 broader real-weight verification (needs heavy runs; the trunk is on E:); line 125 `glm-dsa` real-weight parity (disk budget, rescoped to hyperparameter parsing); line 126 `glm5next` Phase 2B (not started); line 137 Muse-Glimmer vision tower (deferred by its plan); line 150 DiffusionGemma real-weight ladder (checkpoint on E:; needs a heavy run and a llama.cpp reference); line 160 MiniMax-H3 (points to its plan's checklist); line 175 DeepSeek V1 (re-checked 2026-11-10: still no V1 checkpoint; the note's list of present files is now corrected).



## Criteria and items completed under open sections (2026-11-25)

Finished items moved verbatim out of [../00-current-work.md](../00-current-work.md). The open parts of each section stay there.

> from: ##### A5. Device memory figure understates real use

- [x] **Done when:** the planned device figure is within 5% of the ledger for both models. (Qwen3.5-0.8B: −1.3% after the fix; Qwen3-0.6B: 1.2% before the fix, unchanged by construction; see the 2026-11-24 status above.)

> from: ##### A11. Housekeeping

- [x] Make `LtxVaeNoiseIsolationTest` write to a temp directory.
  - **Status 2026-11-08: already done.** The test writes to `Path.GetTempPath()` (`LtxVaeNoiseIsolationTest.cs:93`). Close the box.

> from: ##### A11. Housekeeping

- [x] Add "run `Tests.Core`" to the merge checklist.
  - **Status 2026-11-08: done.** The merge checklist is `.github/PULL_REQUEST_TEMPLATE.md`. It now has a line for `Tests.Core`. It did not name it before, and it asked only for the `.Fast` projects.

> from: ##### A12. OpenTelemetry acceptance boxes

- [x] **b. Cost tiers:** every instrument has a cost tier (plan section 4.1). Test that only tier 1 does work with no listener attached, and that tier 3 output is labelled as perturbed.
  - **Status 2026-11-08: met by a passing test.** `EngineTelemetryTierGatingTests` (ForwardPass.Fast, 1 test, 0 skipped, 0.21 s) checks that the tier-2 gate is closed with no listener, with a tier-1-only listener, and that it opens with a tier-2 listener and closes again after it. It also checks that tier-3 instruments are absent without diagnostic mode and that their definitions start with "DIAGNOSTIC". Not re-checked here: that every instrument in code has a row in the tier table; `EngineTelemetryContractTests` is the place for that.
  - **a is not done:** the overhead bound needs a quiet machine and a method that isolates the no-listener cost. Left open.

> from: ##### A14. Partial-offload estimate test fails at HEAD

- [x] **Done when:** the test passes, with the cause recorded (planner fix or corrected premise), and the measured figure is in the plan. (Met 2026-11-24: both the planner fix and the premise correction; numbers above.)

> from: ##### A15. Single-slot admits a second model while the first is still loading (verified 2026-11-24)

- [x] **Done when:** a test with a blocking loader shows that `B`'s loader is not called while `A`'s load is in progress, and that `B` loads after `A`'s handle is released. The test fails on the code as it was before the fix (both loaders run concurrently).
  - **Done 2026-11-24.** Test `SingleSlot_ColdLoadOfAnotherModel_WaitsWhileFirstLoadIsInFlight_NeverRunsTwoLoadersAtOnce` (`tests/OpenTail.Stingray.Tests.Server.Fast/ModelRuntimeManagerTests.cs`): before the fix, B's loader ran while A's was in flight (`Expected 0 Actual 1`). Mutation (the pending-load term removed) fails it again. Passes after the fix.
  - **Also needed for the fix (a real race that the pending-load check alone exposed):** a freshly loaded model could be evicted by a waiter before its caller took the handle, so the caller reloaded it. A claim window now protects a successful load while callers still await it (`_awaitingClaim`, with an `AwaitingCallers` count per load). Test `SingleSlot_CallerCancelledWhileItsLoadIsInFlight_DoesNotStrandTheModel` (found and fixed in review): the first version left the model protected forever when its only caller cancelled before the load finished; the counter fixes that. Test `SingleSlot_FirstOfTwoCallersOnALoadCancels_SecondClaimsIt_AndNextModelWaitsOnlyWhileItIsHeld` guards the two-caller case, but it uses 100 ms delays to order its callers, so it is not a proof of ordering.
  - **Evidence:** `ModelRuntimeManagerTests` 64 total, 0 failed, 3 runs. Full `Server.Fast` 701 total, 0 failed, 2 skipped (exit 0). Release build 0 warnings, 0 errors.
  - **Open:** a single-slot load that the admission queue starts (`StartLoadLocked` from `WakeOldestAdmissibleQueuedWaiter`) has no caller at start, so it is unprotected until its queued caller retries. That matters only with `ResourceBudget` and `AdmissionWaitTimeout` set. The change touches residency accounting in a concurrency path; an independent review is recommended before any further change here.

> from: ##### A28. A prompt longer than the context window is a 500, not a 400 (verified by reading 2026-11-24; fixed 2026-11-24)

- [x] **Done when:** a prompt one token longer than the context is refused with a 400 that names the limit, a prompt at the limit still runs, and a test fails before the change (the status is 500 or an unmapped exception).
  - **Done 2026-11-24 for the continuous-batching engine and the chat, Responses and Anthropic endpoints.** The batcher refuses a prompt longer than `MaxSeqLen` at admission (`ContinuousBatchingEngine.cs`, before any KV reservation or prefill) with the existing `ContextLimitExceededException`; `IInferenceEngine.CheckPromptFitsContext` lets streaming endpoints refuse before the 200 header; `ContextLimitRefusal` maps it to a 400 with `code: context_length_exceeded` (OpenAI's code; the Anthropic body has no code field).
  - **Tests:** engine `PromptOneTokenLongerThanTheContext_IsRefusedAtAdmission_BeforeAnyPrefill` failed before (`No exception was thrown`), passes after; `PromptExactlyAtTheContext_IsAdmitted_AndGenerates` passes before and after. Endpoint tests (`tests/OpenTail.Stingray.Tests.Server.Fast/ContextLimitEndpointTests.cs`): the non-streaming chat, Responses and Anthropic cases failed before with `InternalServerError`, and the streaming chat case with `OK` (headers already sent); all four pass after. Mutations: the admission condition disabled fails the engine test; the chat non-streaming catch disabled fails its test. Server.Fast 710 total, 0 failed, 2 skipped; ForwardPass.Fast `ContinuousBatchingContextLimitTests` 2 of 2.
  - **Gaps, recorded:** (1) `InferenceEngine` (the single-request engine, `RuntimeInstance.cs` around line 267) has no admission check or `CheckPromptFitsContext` override, (Was open; **closed 2026-11-24, iteration 35.** Reachability, read the same day: `RuntimeInstance.cs` line 257 builds the continuous batcher only when `plan.Batching.Mode` is `Continuous`; otherwise line 267 builds `InferenceEngine`, so a server request on such a plan reaches it. Fix: `InferenceEngine.CheckPromptFitsContext` (tokenizes, refuses above `MaxSeqLen`; the streaming endpoints call it before the header) and a refusal in `GenerateCore` after tokenization, before any prefill. Tests: `InferenceEngineContextLimitTests` 5 of 5 (refusal before prefill; a prompt of exactly the window is admitted and prefilled, with zero output tokens; decode stops at the window; the pre-check refuses and accepts). Mutation: disabling both refusals fails the two refusal tests. Neighbouring engine and telemetry classes green. **Difference kept, recorded:** a prompt of exactly the window gives zero output tokens here, because the decode loop samples only within its budget; the continuous batcher emits one prefill token. Matching it would mean restructuring the decode loop, so it is not done. Server.Fast, 714 total, 0 failed, 2 skipped, is not a test of this path because it uses a fake engine; the engine-level tests are the evidence.) (2) Image and video requests go through other entry points, and the pre-check counts text tokens only. (3) Closed 2026-11-24: Responses and messages streaming now have endpoint tests (`ContextLimitEndpointTests.StreamingResponses_*` and `StreamingMessages_*`: 400 JSON before the SSE header). Removing both pre-checks makes both fail; restored, Server.Fast 714 total, 0 failed, 2 skipped.

> from: ##### A29. Ordinary generation is not stopped at the context window, so decode can run past it (HIGH, possible memory corruption; fixed 2026-11-24, evidence below)

- [x] **Done when:** a fake forward pass with a small `MaxSeqLen` records every decode position; a prompt plus `MaxNewTokens` longer than the window never gets a decode position at or beyond `MaxSeqLen` (the test fails before the fix, where the recorded position reaches or passes the window), a prompt exactly at the window still produces its first token, and the output stops at the window. Then one real-model check on a small GGUF (Qwen2.5-0.5B is on F:\_models) with a context of 64 and a long `max_tokens`, showing the stop, recorded in the item.
  - **Fix (engine, `ContinuousBatchingEngine.cs`):** the renewal block now stops every decoding sequence with `Position + 2 > MaxSeqLen` (the step still runs; the token it samples is discarded by `StopAfterStep`, and `StopAtContextEnd` reports the stop as length, not budget). A sequence that reaches `MaxSeqLen` before a step is retired first. The renewer is consulted only when the window allows the step.
  - **Tests (ForwardPass.Fast, Release, run 2026-11-24):** `ContinuousBatchingDecodeBoundTests` 3 of 3 (prompt 10 in a window of 16 decodes at positions 10 to 15 and stops with 6 tokens; prompt 16 makes no decode step; prompt 15 makes one step at 15). Before the fix the coder's run decoded at 14, 15, 16, 17, 18 and beyond and failed. `ContinuousBatchingRetainedStateTests` 6 of 6, two of them added in this pass: a retained prompt of 64 in a window of 64 is refused before any prefill, and a retained 63-token turn stops at the window with one output token and `MaterializedPosition` 64. Mutations: disabling the decode-window stop fails 3 tests (2 decode-bound, 1 retained); removing the retained refusal clause fails its test. `ContinuousBatchingContextLimitTests` 2 of 2, `ContinuousBatchingTurnGateTests` 3 of 3, `ContinuousBatchingConstraintTests` 7 of 7, `Tests.Server.Fast` 710 total, 0 failed, 2 skipped.
  - **Real model (CPU backend, Qwen2.5-0.5B-Instruct Q4_K_M, context 64, `max_tokens` 200, only the window can stop it, `STINGRAY_RUN_REAL_CONTEXT_TEST=1`):** prompt 17, generated 47 (= 64 - 17), length stop, 3.5 s of wall clock with weights loaded. This was not run on the pre-fix code, so the before-fix evidence is the fake-forward-pass tests only.
  - **Decision (mine, recorded for the owner):** a retained (session) turn whose prompt fills the window exactly is refused at admission with `context_length_exceeded` (400), not run. Its first token would have no KV slot, and a retained turn may log only what the cache materialised. A fresh request of that length still runs and returns one token with a length stop (the A28 test). The two paths differ on purpose; a different trade-off is the owner's call.
  - **Open:** (1) the CPU batched decode has no position check of its own (`ForwardPass.BatchedIO.cs` around line 307); the engine now prevents the call, and a defensive check is not added. (2) The Vulkan and CUDA batched passes were not read in full. A pattern search of `src/OpenTail.Stingray.Vulkan` and `src/OpenTail.Stingray.Cuda` for a position guard (`Position exceeds`, `(uint)pos >=`) found none, so they rely on the engine bound. A full read is still open. (3) The single-request `InferenceEngine` bounds decode with `decodeBudget = min(MaxNewTokens, MaxSeqLen - startPos)` (`InferenceEngine.cs` line 1481). Tested 2026-11-24 with a fake forward pass (`InferenceEngineContextLimitTests.DecodeStopsAtTheWindow_WhenTheBudgetIsLargerThanTheRoom`: a 5-token prompt in an 8-token window decodes 3 tokens of 10 requested, with a length stop). No real-model run. (4) Closed 2026-11-24 for chat and messages: `FinishReasonEndpointTests` (new) runs a fake that stops at its budget and checks `finish_reason: length` (chat, `OpenAiEndpoints.cs` line 1069) and `stop_reason: max_tokens` (messages, `DetermineStopReason` in `AnthropicEndpoints.cs`). Passes with 2 of 2. **Open, owner review, not changed:** `ResponsesEndpoints.cs` never reads `TruncatedByMaxTokens`; it hard-codes `"completed"` for the response status at lines 204 and 243, so a run cut at `max_output_tokens` is reported as completed. The official Response object page (read 2026-11-24, most of it unread) lists `incomplete` as a status and `max_output_tokens` as an `incomplete_details.reason`, but the part read does not say a length stop sets `incomplete`, and the response-level stream event names were not found in it. Changing the wire needs that confirmed, plus a choice for the streaming events. Recommended default when the owner decides: status `incomplete` with `incomplete_details.reason` `max_output_tokens` for a length stop; the stream event name (`response.incomplete` is the likely one, unverified) needs the rest of the spec read first. Not done here.

> from: ##### A32. The hybrid path's greedy output varies between runs of the same build, and a background prefetch error appears at exit (observed 2026-11-24; not investigated)

- [x] **Done when:** the variation is either explained by a logged difference in expert serving, or shown to be absent on repeated runs; and the shutdown error is gone or explained.
  - **Met 2026-11-24 (iteration 47), in this scope:** the variation is explained by logged expert serving (GPU slot versus CPU fallback, mixed by upload timing; the iteration-44 trace and the forced-CPU runs above), and the shutdown error is gone (fixed in `b65c8e51`). Not included: a deterministic mode for the expert path. That is owner decision (h), and it stays recorded with the evidence above.

> from: ##### B9. Performance and scout follow-ups (after A5 and A6)

- [x] **Scout, Hub limits:** cap and expose the Hub index limits. Small hardening; do it first.
  - **Done 2026-11-24.** `scout --backlog --limit N`: a value above 200 is still capped to 200 (the Hub listing cap), but the report now carries `requested_limit` and `limit_note`, and the text output prints the note. `limit` keeps the value sent to the Hub; the schema stays at version 1 (additive). `--max-index-mb` is refused above a 512 MB ceiling (4 times the 128 MB default, derived from `RemoteGgufReader.MaxIndexBytesCeiling`); the help text and `061-coverage-tooling.md` state both values. Evidence: `OpenTail.Stingray.Tests.Cli` built Release with 0 warnings and 0 errors; full run 778 total, 0 failed, 25 skipped, exit 0 (14.98 s). New no-network tests: above-cap note, in-range no note, ceiling validation, help text against the derived constants. Mutation check (note replaced by null): 2 tests fail, then restored.
  - **Done 2026-11-24:** `--max-quants` has a ceiling of 160 (4 × the default 40 in `QuantPicker.DefaultMaxFiles`, the same rule as `--max-index-mb`). Above it the command is refused, and the help text and `cli-option-inventory.md` / `061-coverage-tooling.md` say so. Test `MaxQuants_RejectsAboveCeiling_AcceptsCeiling_NoNetwork` (no network). Cli test exe 779 total, 0 failed, 25 skipped, exit 0 (16.7 s), Release build 0 warnings and 0 errors. The 160 is the orchestrator's choice; the owner can change it.
  - **Machine note (owner):** `global.json` pins SDK 10.0.301, and this machine has only SDK 11.0.100 working (`dotnet --list-sdks`). The 10.0.300, 10.0.301 and 10.0.403 folders exist but have no `dotnet.dll`, so a build run inside the repo cannot find the pinned SDK. This run built and tested from outside the repo, where `global.json` does not apply, with SDK 11.0.100. The 10.x pin is not verified here. Repairing the SDK install is a machine change, so it waits for the owner.

> from: ### 3. Items with no open parent

- [x] **Configuration ownership.** List every config switch with its owner and mark the obsolete-looking ones for an owner decision. Source: [QoL plan](../3-product-and-runtime/2026-11-08-product-plan.md).
  - **Done 2026-11-24 (the listing and the marking; the retirement is the owner's decision).** The inventory is name-complete: all 296 registered `STINGRAY_*` names are in [reference/env-var-inventory.md](../reference/env-var-inventory.md), each with a class. The owner of every switch is the project owner (there is no per-switch owner in the repo), so the owner column is the owner's. A mechanical pass marks **33 candidates**: each has one live read or dump hook in source (checked line by line; none is comment-only), no test reference, and at most two doc mentions. They are listed in the new section "Owner-review candidates, 2026-11-24" of that inventory. **Nothing was removed.** Owner decision: retire, keep, or document each candidate; a read site must be read before any removal.
  - **CI note 2026-11-24:** main was red from 89af44a5 to fe94cd91 because this candidate table had the shape of the variable-class table, which KnownEnvironmentVariablesTests reads (the Core suite failed). The list is bullets now (becaf125); the latest main run for becaf125 is green (12:27 UTC).


## Leftover checked items under open sections (2026-11-25)

> from: ##### A11. Housekeeping

- [x] Fix or quarantine the flaky `SimdKernels` timing test.
  - **Done 2026-11-10:** the test that asserted nothing is renamed to `SimdKernelsIq1STests.FastKernel_TimingIsRecordedNotAsserted`, so the name says what it does (it records a speed of about 0.8 to 0.9 times the fallback and gates nothing). `ForwardPass.Fast` builds in Release, and `-class OpenTail.Stingray.Tests.ForwardPass.Fast.SimdKernelsIq1STests` runs 3 of 3 with 0 failed and 0 skipped. No other reference to the old name.
  - **Status 2026-11-08: no flake found in the timing tests that assert.** `SimdKernelsBF16Tests.FastKernel_IsFasterThanDequantFallback` asserts `speedup > 1.0`. It passed 40 of 40 at `DOTNET_PROCESSOR_COUNT=2`, with speedups of 7 to 10 times. `SimdKernelsIq1STests.FastKernel_IsFasterThanDequantFallback` (renamed 2026-11-10, see the line above) asserts no speed at all: it prints a speedup that ran from 0.89 to 1.90 over 40 runs, and only its correctness checks can fail. The name says "IsFaster", so it is misleading. Fix: rename it, or assert a speed bound that the margins support. Not changed here.

> from: ##### A12. OpenTelemetry acceptance boxes

- [x] **c. Growth seams:** deferred 2026-11-10, with a reason: the plan says add each seam only when a new metric needs it, and no metric was added in this session, so none was needed. Revisit when one is.

