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
    "summary": "Q4_K scale/min unpacking reads scales[chunk]/scales[chunk+8] as if the 12-byte packed field were 16 contiguous bytes, instead of the real 6-bit cross-byte-spliced format this repo's own Dequantize.cs correctly implements.",
    "failure_scenario": "For sub-block indices 4-7, the read goes past the real scales buffer into the qs (weight nibble) region. Any matmul routed through this kernel (gated by an env var) silently corrupts about half the dequantized Q4_K weight values per super-block, with no crash and no test coverage catching it."
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
    "summary": "TokenChoiceTrie.ChoiceState.IsComplete requires zero children, so a valid choice that is a token-prefix of a longer allowed choice can never be accepted as a stopping point.",
    "failure_scenario": "With AllowedChoices [\"APPROVED\", \"APPROVED_WITH_CHANGES\"], once decoding reaches the terminal-but-has-children \"APPROVED\" node, MaskLogits masks everything except tokens continuing toward the longer choice, forcing the model to always emit the longer choice. Confirmed reachable: InferenceSession.cs's generation loop gates its only early-stop signal on this exact property."
  },
  {
    "file": "src/OpenTail.Stingray.Core/Grammar/GrammarStateMachine.cs",
    "line": 146,
    "summary": "TryAcceptChar has no transitions for nested-object start ({), array start ([), or the null literal (n), and the required-property/enum enforcement machinery (PushFrame/PopFrame/GrammarFrame) has zero callers from JsonSchemaGrammarMasker or JsonSchemaGrammarCompiler.",
    "failure_scenario": "Any DTO schema with a nested object, array, or null-typed property either deadlocks generation or is silently unconstrained at that point; required and enum constraints from the schema are never enforced, letting a model emit {} for a schema requiring fields. No test exercises nested/required/enum rejection through the actual masker path."
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

**Cut for the cap but still real and worth a look**, roughly in priority order: `DeepSeekMoeGraph.cs:172` (MoE `ExpertOffsets[MaxExperts]` off-by-index — silently drops expert gather/scatter on workspace reuse), `InferenceSession.cs:325/529` (empty `catch {}` swallowing `TruncateTo` failures with zero diagnostics), `InferenceSession.cs:612-668` (checkpoint generation not restored on `Rollback`, causing spurious `StaleContinuationException`), `SessionBranchingExtensions.cs:53` (`int.TryParse(chunk.Text,...)` — `GeneratedTokens` is always empty for real text), `JsonSchemaGrammarMasker.cs:78` (100k+ allocations per decode step via `Clone()` per vocab token — CLAUDE.md hot-path violation). Two candidates were explicitly REFUTED by verification: the `KvMemoryGovernor` TOCTOU race (safe — serialized through the session's own mutex with re-validation) and the `InferenceSession` double-release (safe — `Dispose()` is idempotency-guarded). Two more (`SpeculativeDecoder.cs` StepSampled/PLD bugs, `InferenceSession.cs` unguarded `Fork()` on CUDA) were confirmed as real defects but currently unreachable/latent — no wired call path exercises them yet.

**Investigated and RESOLVED, not a production bug**: `ForwardPass.PrefillCore`'s Q8-quantized prefill vs `BatchForwardMulti`'s exact-F32 decode looked like a cache-correctness defect (`HotSessionGreedyReplayTests.HotSession_MultiTurn_MatchesFullGreedyReplay_OnRealModel` was failing on it) but both paths are individually correct and deliberate — the actual defect was the test's own replay oracle grading the session against an unachievable "replay everything as one cold Q8 prefill" reference. Fixed by rewriting the oracle to replay through the session's own `PrefillWithCache`/`BatchForwardMulti` entry points instead. A related, confirmed asymmetry (`ForwardPass.Prefill`'s length-1-prompt shortcut to `Forward`, which `PrefillWithCache` deliberately lacks) is a plausible root cause for the still-open `ContinuousBatchingTests.PrefillWithCache_SingleToken_MatchesForward` failure — not yet confirmed. Full writeup: docs/029-prefill-batch-composition-numerics-bug.md.
