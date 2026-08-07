# Handover — OpenTail.Stingray session-native runtime work

**Written for whoever picks this up next (human or AI).** Read this before touching anything.
Everything here is stated as of the handover; verify rather than trust, for the reasons in §6.

> **2026-08-07 follow-up.** This remains a historical handover, not a live git-state report: the
> uncommitted-file list in §2 no longer matches the working tree. The previously unverified
> `HotSession_ExactAppendAtPageBoundary_MatchesFullGreedyReplay` now compiles and passed as part
> of `HotSessionGreedyReplayTests` (**2/2**, Release, 2026-08-07) on the local SmolLM2 GGUF. This
> also covers its real-model hot multi-turn/full-replay sibling. The formerly outstanding
> `ColdSession_RealModel_CrossProcessRestore_MatchesFullGreedyReplay` has now passed too: it
> persists in one real test process, restores in a fresh process/runtime, and compares greedy
> continuation against full replay. This proves only the CPU-dense SmolLM2 reference lane; product
> API exposure and other backend/cache families remain open.

---

## 1. Where the work is

Two programmes. The first is finished; the second is mid-flight.

| programme | doc | status |
|---|---|---|
| CPU/Vulkan kernel programme | `docs/cpu-architecture-kernel-opportunities.md` | **CLOSED.** Items 1–4 and 6 resolved, item 5 re-scoped. Do not reopen it; read its closing table first if tempted. |
| Session-native runtime | `docs/session-native-inference-runtime-plan.md` | **ACTIVE.** This is the work. |

**The plan document is the authoritative state, not this file and not any prompt.** It is written
append-only with numbered sections (§3.4.x for Milestone 0 baselines, §8.x for Milestone 1). Every
result — including negatives and retractions — is recorded there. Start by reading the last ~200
lines and the checkbox list.

Progress: Milestone 0 essentially complete (18/22 items; the rest are judgement calls or need
llama-swap). Milestone 1 invariants at **12 of 21**. Milestones 2+ untouched.

---

## 2. Git state at handover

The user has committed everything up to and including the kernel programme and most session work.
**Five files remain uncommitted**, all from the most recent Milestone 1 test work:

```
M docs/session-native-inference-runtime-plan.md
M src/OpenTail.Stingray.Sessions/SessionContracts.cs          # added ExpectedRevision/ActualRevision
M tests/OpenTail.Stingray.Tests.Sessions/HotSessionGreedyReplayTests.cs
M tests/OpenTail.Stingray.Tests.Sessions/HotSessionTests.cs
M tests/OpenTail.Stingray.Tests.Sessions/InMemorySessionStoreTests.cs
```

**DO NOT COMMIT ANYTHING.** That is the user's call, without exception. It was the standing rule for
the whole engagement and nothing was ever committed by the agent.

### There is one DRAFT that has never compiled

`HotSessionGreedyReplayTests.HotSession_ExactAppendAtPageBoundary_MatchesFullGreedyReplay` was
written and then the session ended before `dotnet build` ran. **Treat it as unverified draft.** It
may not compile (it references `PagedKvCache.PageSize`, `ImmutableArray<ExecutionSegment>` and
`ContinuousBatchingEngine` — check the using directives in that file are sufficient). Build it
first, then run it, then mutation-test it before believing it.

Last **verified** full suite: **2291 total, 3 failed**. A later run was in flight when the session
ended and its result was never read.

---

## 3. The three known failures — do not chase them

| test | verdict |
|---|---|
| `Gemma4VulkanNarrowedKvE2ETests.Gemma4_E4B_Q4_0_VulkanNarrowedKv_MatchesFp32Argmax` | pre-existing Gemma4 Vulkan divergence |
| `Gemma4VulkanPleE2ETests.Gemma4_E4B_Q4_0_VulkanForward_LongDecodeIsCoherent` | same pair |
| `ConcurrencyLimitTests.BoundedQueue_RejectsOnlyAfterActiveAndWaitingCapacityIsConsumed` | load-dependent flake; passes 4/4 in isolation, times out under full-suite load |

Any count other than these three is a regression you introduced.

---

## 4. Standing rules (these were non-negotiable)

1. **Never commit.** The user's call.
2. **Never run two benchmarks or model loads concurrently.** Single CPU/GPU box.
3. **Never build while a test run is live** — it overwrites DLLs under execution. Check with
   `tasklist | grep -ci OpenTail.Stingray.Tests` first.
4. **Clean build + full suite before claiming anything done.** `dotnet build -c Release` has
   `TreatWarningsAsErrors`, so 0 warnings is a real gate.
5. **Verify a new test FAILS on a deliberate mutation before trusting it.** See §6 — this caught
   real problems repeatedly and is not ceremony.
6. **Measure isolated first, then end-to-end, with interleaved arms** and enough samples to clear
   the noise floor (this box's run-to-run spread is ~±5%).
7. **Run the perplexity gate BEFORE defaulting a numerics-changing path**, never after.
8. **Document every result including negatives and retractions** in the relevant doc.

---

## 5. What is open, precisely

### Actionable now
- **Page-boundary append** — the draft in §2. Build, run, mutation-test.
- **Milestone 1's non-invariant items**: session deletion surface, bounded ledger retention,
  detachable result retrieval, rolling memory reservation (plan §8.0, §8 body).

### Blocked on a *named* prior gap — do not write tests for these yet
Four invariants require `AcceptedPositionCount != MaterializedPositionCount`. `HotSession.BuildNextCursor`
ends with `new SessionCursor(log, accepted, accepted, accepted, accepted, StateCoverage.Full)` — all
four counts identical. **The runtime cannot represent an unmaterialised suffix at all**, so these
would assert on a state that never occurs and pass vacuously:
first-token materialisation lag; session starting with an unmaterialised suffix; mismatch inside
that suffix; reservation renewal at the pending token. They are blocked on **implementing plan
§4.2**, not on test-writing. See §8.3 of the plan.

### Blocked on a decision only the user can make
- **"Changed leading block"** — ambiguous. "Block" may mean the leading execution *segment* or a
  page-aligned KV *block* (`PagedKvCache.PageSize` = 16). The neighbouring invariant says "page
  boundary" explicitly, weakly suggesting the latter. Ask; do not guess. (Plan §8.5.)
- **Two Milestone 0 stopping decisions**: whether three local role models are useful enough in a
  manual interleave, and whether vision is interactive on a reference host (which has not been
  declared). Both are product judgement, deliberately left unticked.
- **llama-swap** is not in the tree. Fetching third-party binaries was not done unprompted.

### Known gaps worth raising, not currently in the plan
- **Compressed KV cannot back a session.** Verified by execution (`TurboQuantSessionCompositionTests`).
  `ContinuousBatchingEngine` reaches the model only via `PrefillWithCache` and `BatchForwardMulti`,
  both of which throw with a TurboQuant cache. So "bounded residency in bytes" — a defining property
  in plan §0 — has no lever today except retaining fewer tokens. One 16K session costs 6 GiB.
  Plan §3.4.13/§3.4.14.
- **llama-server's KV is fp16; ours is fp32** — literally half the residency, measured on the same
  model. `STINGRAY_KV_DTYPE=bf16` reduces *precision* while still storing fp32, so it buys
  nothing in bytes. Plan §3.4.23.
- **`ActivateSeq`'s stop-token ordering fix is inspection-only.** A test now reaches that branch but
  re-inverting the ordering does **not** fail it — the race window is too small there. Plan §8.9.

---

## 6. Hard-won lessons — re-learning these is expensive

These are not general advice; each cost real time here.

1. **Reading is not running.** Four separate "the audit is complete" conclusions were wrong and only
   surfaced on execution. Most recently: a "three quantities" refactor derived from two real crash
   traces was still wrong — fixing both merely advanced the crash to five more unmodelled concepts.
   For any *gated* code path, the only trustworthy characterisation comes from opening the gate and
   running it.
2. **A passing suite proves nothing about a path no available model exercises.** Say so explicitly
   rather than implying coverage.
3. **Mutation-test, and mutate the right half.** For a test asserting what *survives* a rejection,
   break the survival, not the rejection. Several tests here were green and vacuous until a
   non-vacuity guard was added — one was vacuous twice, for two different reasons (disposing an
   async enumerator *cancels* the request; the fake retires 64 tokens in 3 ms).
4. **Reaching a branch ≠ detecting its defect.** See §5's `ActivateSeq` note.
5. **A/B needs a positive control that the treatment was applied.** A clean null result turned out to
   be a stale binary: `tools/*` projects are outside `OpenTail.Stingray.slnx`, so a solution-level build
   does not rebuild them. `session-bench` now self-stamps its assemblies' build times. `attn-bench`
   has the same exposure and no such stamp.
6. **An impossible number means the accounting is wrong, not the measurement.** A STREAM probe
   implied decode was at 106% of memory bandwidth; the cause was write-allocate undercounting, and
   correcting it made all four kernels agree at ~35–36 GB/s.
7. **fp32 accumulation ORDER is not a rounding detail in a deep residual network.** Reducing MoE
   expert contributions in expert order instead of top-k slot order — same kernels, same math —
   moved logits by 0.20 and changed the sampled token.
8. **Hardcoded constants are not automatically faster than runtime parameters.** A strided GEMM beat
   the shape-hardcoded one by 7.6% at the identical shape; hoisted row pointers beat folded loop
   bounds. The opposite had been assumed and written into a comment.
9. **Python edit scripts that assert on several anchors and write at the end will silently discard
   earlier replacements when a later assert fails.** This bit twice. Prefer one anchored edit at a
   time, and grep for the symbol afterwards rather than trusting the "done" message.

---

## 7. Box specifics

Zen 3, **12 logical processors** (`Environment.ProcessorCount` reads 12 — `docs/HANDOVER.md`'s
"6-core/12-thread" is right; the "Ryzen 7 5700G 8c/16t" attribution repeated elsewhere is wrong).
Measured memory ceiling **~35–36 GB/s**. No AVX-512, no VNNI, no OpenBLAS, no NVIDIA GPU.
**CPU prefill is ~2x faster than Vulkan here**, so CPU work is worth about twice Vulkan work.

`docs/HANDOVER.md` also describes SmolLM2-1.7B as "8 KV heads, GQA" — **wrong**, the GGUF says
`head_count_kv = 32` (MHA). That document is useful but not authoritative; check the metadata.

Models present: SmolLM2-1.7B-Q4_K_M (8192 ctx, headDim 64), Qwen3-0.6B-Q8_0 and Qwen3-8B-Q4_K_M
(40960 ctx, headDim **128** — so flash-64 never engages), OLMoE-1B-7B (4096 ctx), gemma-4-E4B q4_0
(per-layer head dims → sequential trunk). **No model on disk can serve a 16K context on the fast
attention path**; that is a model-availability constraint, not an effort one.

---

## 8. Useful commands

```bash
# Always check before building
tasklist | grep -ci OpenTail.Stingray.Tests

dotnet build -c Release                      # 0 warnings is the gate
dotnet test -c Release --no-build            # full suite, ~620-740s

# One class (MTP filter syntax — NOT --filter)
dotnet test tests/OpenTail.Stingray.Tests.Sessions -c Release --no-build -- --filter-class "*HotSessionTests*"

# tools/* are OUTSIDE the solution — build explicitly or you will run a stale binary
dotnet build tools/session-bench -c Release
```

`DOTNET_TC_QuickJitForLoops=0` for any timing run, or tiered JIT invalidates the numbers.
