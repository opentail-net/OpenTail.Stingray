> **ARCHIVED, 2026-08-15.** Resolved entries split out of [../bugstofix.md](../bugstofix.md),
> which keeps only the still-open items. Two of these (the `ForwardPass.cs:5606` batch-tier
> divergence and the Q8-vs-F32 replay-oracle investigation) have their own full writeups already
> archived at [031-concurrent-decode-batch-tier-divergence-bug.md](031-concurrent-decode-batch-tier-divergence-bug.md)
> and [029-prefill-batch-composition-numerics-bug.md](029-prefill-batch-composition-numerics-bug.md)
> respectively — kept here too only for the original bug-list context.

---

```json
[
  {
    "file": "src/OpenTail.Stingray.Cpu/MicroGemmQ4K.cs",
    "line": 64,
    "summary": "RESOLVED. The scale/min splice bug was real but turned out to be only half the defect: the byte-to-value nibble grouping was also wrong. The kernel treated each 16-byte region as 32 independent per-byte low/high-nibble pairs sharing one scale/min; ggml's actual q4_K layout (dequantize_row_q4_K/get_scale_min_k4 in ggml-quants.c) is four 64-element super-chunks, each consuming 32 bytes and TWO scale/min pairs -- the LOW nibble of all 32 bytes forms the first 32 elements (pair j), the HIGH nibble of the SAME 32 bytes forms the next 32 (pair j+1). Fixed both: added a GetScaleMinK4 helper mirroring Dequantize.GetScaleMinK4 exactly, and rewrote the inner loop to the correct four-super-chunk/two-nibble-half structure.",
    "failure_scenario": "RESOLVED. Was: any matmul routed through this kernel (gated behind STINGRAY_Q4K_MICRO_GEMM/STINGRAY_CPU_MICRO_GEMM, default off) silently corrupted Q4_K weight values with no crash and no test coverage. New test tests/OpenTail.Stingray.Tests.Core/MicroGemmQ4KTests.cs compares the kernel's output against Dequantize.ToFloat32 + a naive matmul reference; confirmed it fails hard against the pre-fix code (off by 4-5x) and passes against the fix. Tests.Core (519/519) and Tests.ForwardPass.Fast (602/602) both pass clean."
  },
  {
    "file": "src/OpenTail.Stingray.Engine/ChoiceConstraint.cs",
    "line": 81,
    "summary": "ALREADY FIXED (found already resolved when picked up, not fixed in this pass). Current code: IsComplete => _currentNode.IsTerminal, with a comment describing exactly this bug and why the zero-children requirement was wrong. No outstanding work here.",
    "failure_scenario": "N/A -- resolved before this entry was next picked up."
  },
  {
    "file": "src/OpenTail.Stingray.Core/Grammar/GrammarStateMachine.cs",
    "line": 146,
    "summary": "RESOLVED. Added transitions for '{' (nested object), '[' (array), and 'n' (null literal, treated as loosely-spelled like the pre-existing boolean handling). Wired the previously-dead PushFrame/RecordPropertyEmitted/AreAllRequiredPropertiesEmitted machinery into TryAcceptChar/CanAcceptChar: object keys are now tracked automatically as they're read, and a closing '}' is only legal once all of the CURRENT frame's required properties have been emitted -- enforced per nesting level, not just at the root. Also added enum enforcement: a string value with EnumValues can only continue with characters that are a prefix of at least one allowed value, and can only close once the accumulated text is an exact match. GrammarFrame was redesigned from a class (HashSet-backed required/emitted tracking) to a cheap struct (bitmask-backed, bounded to 64 tracked properties per nesting level) specifically because it's now pushed/popped inside the SAME hot path as the JsonSchemaGrammarMasker.cs:78 allocation bug below -- fixed together so the nesting fix didn't make that bug worse. GrammarStateMachine.Clone() is preserved for compatibility but a new non-allocating CopyFrom(source) was added for the masker's reused scratch instance.",
    "failure_scenario": "RESOLVED. Was: any DTO schema with a nested object, array, or null-typed property either deadlocked generation or was silently unconstrained at that point; required and enum constraints were never enforced, letting a model emit {} for a schema requiring fields. New tests in tests/OpenTail.Stingray.Tests.Core/JsonSchemaGrammarMaskerTests.cs (Test17-Test21) exercise nested-object required-property enforcement, array-of-objects round trips, null literals, the exact '{} for a schema requiring fields' scenario, and enum rejection through the ACTUAL masker Filter()/Accept() path (not just the state machine directly). One pre-existing test (Test14) had its scenario redesigned: it used to prove ValueStart was a dead end by replaying an opener token as its own illegal-continuation probe, which stopped being a genuine dead end once nested objects are legal at every value position (by design) -- redesigned around an enum-prefix mismatch instead, which remains a genuine dead end. All 22 tests pass; full Tests.Core (525/525 non-skipped), Tests.Sessions.Fast (394/394), and Tests.Server.Fast (260/260) pass clean."
  },
  {
    "file": "src/OpenTail.Stingray.Engine/ForwardPass.cs",
    "line": 5606,
    "summary": "RESOLVED (was open as SimdKernels.cs:1092). Original hypothesis (SimdKernels.MatMulBatched's small-batch tiered dispatch, MatVec4In/MatVec2In, batch-composition-dependent in DECODE) was disproven by a controlled raw-API diagnostic: BatchForwardMulti at N=8 in isolation showed zero divergence. The real mechanism was in PREFILL: ContinuousBatchingEngine.RunPrefillStep packs multiple UNRELATED sessions' prompt tokens into one combined batch via ForwardPass.PrefillPackedMulti, and SimdKernels.MatMulBatched's OpenBLAS-SGEMM-vs-tiered-dot-product kernel choice was gated purely on that COMBINED batch size (N >= MinBatchForBlas, default 16) -- with no awareness that N spans multiple independent prompts. Fixed by adding an allowBlas parameter (default true) to SimdKernels.MatMulBatched and ForwardPass.MatMulBatchedCached/MatMulBatchedDualCached; PrefillPackedMulti's six matmul call sites now pass allowBlas:false. Full writeup: 031-concurrent-decode-batch-tier-divergence-bug.md.",
    "failure_scenario": "RESOLVED. Was: any deployment serving roughly 5-15 concurrent HotSession requests admitted together could have one session's prefill (and downstream decode) silently diverge from what the identical request would produce alone or with different concurrent traffic. Now fully green; see 031 for full regression detail."
  }
]
```

**`JsonSchemaGrammarMasker.cs:78` — RESOLVED.** Was: 100k+ allocations per decode step via `GrammarStateMachine.Clone()` once per vocabulary entry inside `Filter()`'s masking loop (CLAUDE.md hot-path violation). Fixed together with the `GrammarStateMachine.cs:146` entry above: `GrammarFrame` became a bitmask-backed struct instead of a `HashSet`-backed class, and `JsonSchemaGrammarMasker` now keeps one reused "scratch" `GrammarStateMachine` reset per candidate via the new non-allocating `GrammarStateMachine.CopyFrom(source)`, instead of allocating a fresh clone per vocabulary entry. `Test22_FilterDoesNotAllocateProportionallyToVocabSize` measures allocation stays flat rather than scaling with vocab size — passes against the fix.

**Investigated and RESOLVED, not a production bug**: `ForwardPass.PrefillCore`'s Q8-quantized prefill vs `BatchForwardMulti`'s exact-F32 decode looked like a cache-correctness defect but both paths are individually correct and deliberate — the actual defect was the test's own replay oracle grading the session against an unachievable reference. Fixed by rewriting the oracle to replay through the session's own `PrefillWithCache`/`BatchForwardMulti` entry points instead. Full writeup: 029-prefill-batch-composition-numerics-bug.md.
