```json
[
  {
    "file": "src/OpenTail.Stingray.Cpu/DeepSeekMoeGraph.cs",
    "line": 36,
    "summary": "SelectTopKExperts indexes the MoE router weight wGateInp as column-major, contradicting the row-major [outDim,inDim] convention used everywhere else in this codebase, including the existing MoE router path in HybridForwardPass.cs that loads the same tensor name.",
    "failure_scenario": "DeepSeek-style MoE layers route tokens to essentially random experts with random gating weights, silently degrading output quality with no crash. Duplicated verbatim in src/OpenTail.Stingray.Engine/DeepSeekMoeGraph.cs:36."
  },
  {
    "file": "src/OpenTail.Stingray.Cpu/MlaAttention.cs",
    "line": 42,
    "summary": "CompressKvLatent/UncompressKv index the kv_a_proj/kv_b_proj weight matrices column-major, the same transpose bug as the MoE router, inconsistent with the codebase's row-major weight convention.",
    "failure_scenario": "Every DeepSeek MLA layer computes wrong compressed KV latents and wrong K/V for attention, silently corrupting generation output for any DeepSeek-V2/V3/R1 model with no crash. Its only test is a NaN/Inf smoke check, not a numeric correctness check. Duplicated in src/OpenTail.Stingray.Engine/MlaAttention.cs:42,90,102."
  },
  {
    "file": "src/OpenTail.Stingray.Cpu/MicroGemmQ4K.cs",
    "line": 64,
    "summary": "RESOLVED. The scale/min splice bug was real but turned out to be only half the defect: the byte-to-value nibble grouping was also wrong. The kernel treated each 16-byte region as 32 independent per-byte low/high-nibble pairs sharing one scale/min; ggml's actual q4_K layout (dequantize_row_q4_K/get_scale_min_k4 in ggml-quants.c) is four 64-element super-chunks, each consuming 32 bytes and TWO scale/min pairs -- the LOW nibble of all 32 bytes forms the first 32 elements (pair j), the HIGH nibble of the SAME 32 bytes forms the next 32 (pair j+1). Fixed both: added a GetScaleMinK4 helper mirroring Dequantize.GetScaleMinK4 exactly, and rewrote the inner loop to the correct four-super-chunk/two-nibble-half structure.",
    "failure_scenario": "RESOLVED. Was: any matmul routed through this kernel (gated behind STINGRAY_Q4K_MICRO_GEMM/STINGRAY_CPU_MICRO_GEMM, default off) silently corrupted Q4_K weight values with no crash and no test coverage. New test tests/OpenTail.Stingray.Tests.Core/MicroGemmQ4KTests.cs compares the kernel's output against Dequantize.ToFloat32 + a naive matmul reference; confirmed it fails hard against the pre-fix code (off by 4-5x) and passes against the fix. Tests.Core (519/519) and Tests.ForwardPass.Fast (602/602) both pass clean."
  },
  {
    "file": "src/OpenTail.Stingray.Cpu/IqCodebooks.cs",
    "line": 33,
    "summary": "IQ2_XXS/IQ3_XXS/IQ3_S/IQ4_XS grid tables are procedurally fabricated from a bit-pattern formula rather than the real calibrated GGML codebook constants, and the matching decoders skip the real format's 9-bit index/7-bit sign-mask/per-group scale.",
    "failure_scenario": "Any GGUF quantized with these IQ formats dequantizes to plausible-looking but unrelated weight values, silently degrading generation to incoherence; the existing test only checks shape/NaN-absence, not reference values."
  },
  {
    "file": "src/OpenTail.Stingray.Engine/ModelCompatibility.cs",
    "line": 460,
    "summary": "\"deepseek2\" was newly admitted as a supported architecture, but none of the six IForwardPass implementations were updated to load the MLA-specific attn_kv_a_mqa/attn_kv_b tensors instead of attn_k/attn_v.",
    "failure_scenario": "Loading a real deepseek2 GGUF throws InvalidOperationException(\"Missing tensor: blk.0.attn_k.weight\") immediately on load, despite the architecture now being listed as admitted."
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
    "file": "src/OpenTail.Stingray.Sessions/InferenceSession.cs",
    "line": 223,
    "summary": "AppendAsync/GenerateAsync check _state == Suspended before acquiring the session mutex and never re-check after acquiring it, unlike EnsureActiveAsync which double-checks under lock.",
    "failure_scenario": "A racing SuspendAsync can Release() the KV sequence between the check and the lock acquisition; CpuKvSequence.Append never checks _disposed, so the subsequent append silently succeeds on a stale, cleared sequence while _tokenHistory keeps accumulating on top of the old count — desyncing logical history from physical KV state with no exception."
  },
  {
    "file": "src/OpenTail.Stingray.Sessions/InferenceSession.cs",
    "line": 799,
    "summary": "RestoreFromSnapshot mutates _tokenHistory/_checkpointGeneration before calling the KV sequence's Append, with no try/catch around that call and no rollback if it throws.",
    "failure_scenario": "If newKv.Append throws (e.g. KV pool exhausted), _tokenHistory/_checkpointGeneration are already updated to the snapshot's values while _kvSequence still points at the pre-restore sequence — leaving the session's logical and physical state permanently mismatched with no compensating recovery."
  },
  {
    "file": "src/OpenTail.Stingray.Sessions/CrossSessionPrefixSynthesizer.cs",
    "line": 127,
    "summary": "The prefix-cache namespace is hardcoded to (\"default-model\",\"default-kv\") for every session instead of being derived from the session's actual ModelId, defeating the namespace's documented purpose of preventing cross-model page sharing.",
    "failure_scenario": "Two sessions running different models (or different KV quantization configs) are treated as prefix-compatible and can share physical KV pages through Publish/MatchPrefix — exactly the invalid cross-model sharing IPrefixCacheIndex's namespace type exists to prevent."
  },
  {
    "file": "src/OpenTail.Stingray.Engine/ForwardPass.cs",
    "line": 5606,
    "summary": "RESOLVED (was open as SimdKernels.cs:1092). Original hypothesis (SimdKernels.MatMulBatched's small-batch tiered dispatch, MatVec4In/MatVec2In, batch-composition-dependent in DECODE) was disproven by a controlled raw-API diagnostic: BatchForwardMulti at N=8 in isolation showed zero divergence. The real mechanism was in PREFILL: ContinuousBatchingEngine.RunPrefillStep packs multiple UNRELATED sessions' prompt tokens into one combined batch via ForwardPass.PrefillPackedMulti, and SimdKernels.MatMulBatched's OpenBLAS-SGEMM-vs-tiered-dot-product kernel choice was gated purely on that COMBINED batch size (N >= MinBatchForBlas, default 16) -- with no awareness that N spans multiple independent prompts. A short prompt prefilled alone stayed under 16 (non-BLAS); the same prompt packed with 7 others crossed 16 and took BLAS instead -- a different, non-bit-identical (but individually correct) summation order, producing a tiny per-layer drift that compounded over decode steps and eventually flipped a close-margin greedy argmax. Confirmed via raw PrefillPackedMulti repro with Q8Prefill toggled off (divergence persisted, ruling out Q8) and MinBatchForBlas forced to 1000 (divergence disappeared, confirming BLAS crossover as the sole variable). Fixed by adding an allowBlas parameter (default true) to SimdKernels.MatMulBatched and ForwardPass.MatMulBatchedCached/MatMulBatchedDualCached; PrefillPackedMulti's six matmul call sites now pass allowBlas:false so packed multi-session prefill never lets combined batch size decide BLAS eligibility. Solo Prefill/PrefillWithCache unaffected (keep the default). Full writeup: docs/031-concurrent-decode-batch-tier-divergence-bug.md.",
    "failure_scenario": "RESOLVED. Was: any deployment serving roughly 5-15 concurrent HotSession requests admitted together could have one session's prefill (and downstream decode) silently diverge from what the identical request would produce alone or with different concurrent traffic. HotSessionConcurrencyStressTests.cs (Boundary_8Concurrent_StillDiverges, Stress_5ConcurrentSessions, Stress_10ConcurrentSessions, Stress_40ConcurrentSessions) is now fully green; full regression (Tests.ForwardPass.Fast, Tests.Sessions.Fast, Tests.Server.Fast, Tests.TurboQuant, Tests.Cli, real-model ContinuousBatchingTests) passes except three pre-existing, unrelated failures confirmed via git-stash bisection to fail identically without this change."
  }
]
```

**Cut for the cap but still real and worth a look**, roughly in priority order: `DeepSeekMoeGraph.cs:172` (MoE `ExpertOffsets[MaxExperts]` off-by-index — silently drops expert gather/scatter on workspace reuse), `InferenceSession.cs:325/529` (empty `catch {}` swallowing `TruncateTo` failures with zero diagnostics), `InferenceSession.cs:612-668` (checkpoint generation not restored on `Rollback`, causing spurious `StaleContinuationException`), `SessionBranchingExtensions.cs:53` (`int.TryParse(chunk.Text,...)` — `GeneratedTokens` is always empty for real text). Two candidates were explicitly REFUTED by verification: the `KvMemoryGovernor` TOCTOU race (safe — serialized through the session's own mutex with re-validation) and the `InferenceSession` double-release (safe — `Dispose()` is idempotency-guarded). Two more (`SpeculativeDecoder.cs` StepSampled/PLD bugs, `InferenceSession.cs` unguarded `Fork()` on CUDA) were confirmed as real defects but currently unreachable/latent — no wired call path exercises them yet.

**`JsonSchemaGrammarMasker.cs:78` — RESOLVED.** Was: 100k+ allocations per decode step via `GrammarStateMachine.Clone()` once per vocabulary entry inside `Filter()`'s masking loop (CLAUDE.md hot-path violation). Fixed together with the `GrammarStateMachine.cs:146` entry above, since that fix's frame-stack state runs through this same hot loop and would have made the allocation problem worse if not designed jointly: `GrammarFrame` became a bitmask-backed struct instead of a `HashSet`-backed class, and `JsonSchemaGrammarMasker` now keeps one reused "scratch" `GrammarStateMachine` (created once in the constructor) reset per candidate via the new non-allocating `GrammarStateMachine.CopyFrom(source)`, instead of allocating a fresh clone per vocabulary entry. `Test22_FilterDoesNotAllocateProportionallyToVocabSize` in `JsonSchemaGrammarMaskerTests.cs` measures `GC.GetAllocatedBytesForCurrentThread()` around a `Filter()` call over a 2000-token vocabulary and asserts allocation stays flat (well under 32 bytes/token) rather than scaling with vocab size — passes against the fix.

**Investigated and RESOLVED, not a production bug**: `ForwardPass.PrefillCore`'s Q8-quantized prefill vs `BatchForwardMulti`'s exact-F32 decode looked like a cache-correctness defect (`HotSessionGreedyReplayTests.HotSession_MultiTurn_MatchesFullGreedyReplay_OnRealModel` was failing on it) but both paths are individually correct and deliberate — the actual defect was the test's own replay oracle grading the session against an unachievable "replay everything as one cold Q8 prefill" reference. Fixed by rewriting the oracle to replay through the session's own `PrefillWithCache`/`BatchForwardMulti` entry points instead. A related, confirmed asymmetry (`ForwardPass.Prefill`'s length-1-prompt shortcut to `Forward`, which `PrefillWithCache` deliberately lacks) is a plausible root cause for the still-open `ContinuousBatchingTests.PrefillWithCache_SingleToken_MatchesForward` failure — not yet confirmed. Full writeup: docs/029-prefill-batch-composition-numerics-bug.md.
