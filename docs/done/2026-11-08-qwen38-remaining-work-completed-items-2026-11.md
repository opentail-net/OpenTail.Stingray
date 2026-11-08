# Qwen 3.8 remaining work (2026-11-08) - completed items

Ticked items moved from [2026-11-08-qwen38-remaining-work-plan.md](../2-coverage/2026-11-08-qwen38-remaining-work-plan.md) on 2026-11-08. Each keeps its text and its section heading. Open items stay in that plan.

## Phase 2. Batching

- [x] **Queue limit.** Maximum queue length and rejection when full. Implemented in ae47d135 (`maxQueuedRequests`,
      `EngineQueueFullException`; tests `ContinuousBatchingQueueLimitTests`, 2). Mapping to a server status: see D3.

## Phase 3. Prefix reuse and retained state

- [x] Copy-on-write for QSA history rows within a sequence and its clones (`Qwen4ExpSequenceState.CopyLists`,
      commit ad42fa92; tests `Qwen4ExpSequenceSharingTests`, 4 synthetic).

## Phase 4. MTP speculative decoding

- [x] `Qwen4ExpForwardPass.ForwardMtp` written (planned graph, steps 1-8). `Qwen4ExpMtpForwardTests.DraftStep_IsFiniteAndDeterministic`
      passes. This is a "step runs" result, not a correctness result.
- [x] MTP head loads from its own file (`Qwen4ExpMtpLoadTests`, real files).

## Phase 5. Image and video input

- [x] 5.1 Vision encoder verified against the pinned reference (`Qwen38Vl_Rainbow448_MatchesLlamaMtmdDebug`, 16/16).
- [x] 5.2 Image question answered, matching the reference (`Qwen4ExpImageAnswerTests`, heavy flag, `QWEN_TEST_IMAGE`).

## Phase 6. Tool calls

- [x] Wire-format parsing for the Qwen tool-call shape (`Qwen4ExpToolCallParseTests`, 15 cases).
- [x] Streaming fragments for the reasoning split (tool-call scan is text-only, by design).
- [x] Validator exists as a library (`ToolCallArgumentValidator`), incomplete and malformed calls are not executed.
- [x] **Malformed JSON tool calls** (`ParseJsonCallBlock`, `DeepSeekToolCallAdapter.ParseBlock`). Fixed in 660b4ea8; tests `ToolCallJsonHardeningTests`, 20. Open: a non-object root (`[]`) throws
      `InvalidOperationException`; a non-string `name` (`{"name":42}`) throws from `GetString()`; a duplicate property
      throws `ArgumentException` from `ToDictionary`; an array or scalar `arguments` silently becomes an empty dictionary.
      Confirm with tests first, then fix. Runnable now.
- [x] **Validator numeric and enum edge cases.** Fixed in 660b4ea8; tests `ToolCallValidatorEdgeCaseTests`, 16. Before the fix: `double.TryParse` accepted `NaN` and `Infinity`; `Math.Floor(d) == d`
      was true for infinity; `Convert.ToString(true)` gave `"True"` where the enum literal is `"true"`.

## Doc hygiene

- [x] Programme plan, "Qwen 3.8 remaining work, tabled 2026-11-08" table: superseded by this file (linked at the end of the programme plan).
