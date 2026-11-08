# Plan 2 (2026-10-31): carry-on work, done without owner input - completed items

Ticked items moved from [docs/2026-10-31-autonomous-plan-2.md](../00-current-work.md#open-work-as-of-2026-11-03) on 2026-11-03. Each keeps its section heading and nested lines.

## Checklist

  - [x] Test the slot-cache candidates: placement timing confirmed (8 of 8 runs bit-identical with prefetcher and warm-up off); expert warm-up alone and predictive prefetch alone are not the cause

  - [x] Record a decision about the bar: 0.97 per-step floor plus a median above 0.999 for this row only (PR 83)

- [x] **P2** Handoff receipts on the Vulkan hybrid (Qwen2.5 admitted; Phi-3, gemma-2 and Granite recorded as not admissible, with the reasons; P2b and P2c fixed; P2d blocked on hardware) (one passing permanent row each; a failing family stays Unverified with its cosine recorded)
  - [x] Qwen2.5 (receipt on the Vulkan hybrid; three permanent rows)
  - [x] Phi-3: tried, cannot run on the hybrid (fused `attn_qkv`, full offload only), stays Unverified; the full-GPU handoff path for it is untested
  - [x] gemma-2: tried, fails (min cosine 0.90 handoff vs sequential); the cause is the Vulkan hybrid and full GPU passes, not the handoff (see P2b); stays Unverified
  - [x] Granite: a dense checkpoint was on `H:\_models` all along (`ibm-granite_granite-3.3-2b-instruct`, missed by my first search because its name starts with `ibm-`). It is not a receipt candidate: the Vulkan hybrid ignores Granite's scale fields and produces garbage, see P2c. The earlier vision-LM result was a property of that file
  - [x] **P2c** The Vulkan hybrid applies none of the model scale fields (embedding, residual, logit, attention-scale override). Found with the dense Granite: `-g 8` answered `C. to - > and \ C、` instead of `Paris` (sequential hybrid against CPU: min cosine -0.36; CPU perplexity 7.81, full GPU 7.77); gemma-3-4b the same (0.47); ERNIE-0.3B (no scales) 0.998. **Superseded the same day by [plan 3, Q2](../00-current-work.md#open-work-as-of-2026-11-03): the hybrid now applies the scale fields and the guard covers soft-capping only.** Original fix: Fixed: `GpuForwardPass.VulkanHybridUnsupportedReason` refuses such models, and `run` prints a clear error with exit code 1 instead of a stack trace. Tests: 6 unit tests, 2 real-model refusals, a 13-checkpoint guard that no receipt checkpoint is caught; three production mutations each fail tests. The clean CLI error was verified by hand only
  - [x] **P2b** Silent GPU degradation for CPU-only architectures. Found: real runs were already safe (the gemma2 descriptor declares `SupportedBackends = Cpu` and the planner falls back to the CPU for `-g 8` and `-g -1`), but the override was silent, `stingray plan` reported 26 GPU layers (it did not consult the descriptor), and `perplexity -g -1` built the GPU pass directly and measured a 22 percent worse gemma-2 without saying so. Fixed: the planner raises a `cpu_only_architecture` warning when explicit GPU layers are overridden, `run` prints it as one note, `plan` consults the descriptor (and shows the reason), and `perplexity -g -1` refuses a CPU-only architecture. Tests: 3 planner and 3 plan tests, four production mutations each fail tests; the `run` note and the perplexity refusal were checked by hand only. Not changed: code that constructs the GPU passes directly (tests, library users) still accepts gemma2, and the dense GPU path still lacks soft-capping
  - Note: SmolLM2 and Mistral-7B were no longer in `models/` at 17:27 and are on no `_models` drive; re-pull before A/B work that needs them

## Moved 2026-11-08 from 2026-10-31-autonomous-plan-2.md

  - [x] Test the slot-cache candidates: placement timing confirmed (8 of 8 runs bit-identical with prefetcher and warm-up off); expert warm-up alone and predictive prefetch alone are not the cause
  - [x] Record a decision about the bar: 0.97 per-step floor plus a median above 0.999 for this row only (PR 83)
- [x] **P6** `Audio.Fast` and `Vulkan.Fast` in CI
  - [x] `ci.yml` (floors 70 and 38) (PR 74)
  - [x] `release.yml` (PR 74)
  - [x] Whisper+VAD test skips visibly without `silero_vad.onnx` (PR 74)
- [x] **P7** Documented request parameters audited (unknown JSON properties stay accepted); findings F1 to F4 of the audit are resolved (PRs 82, 84, 85 and 86); F5 (request-level tests for supported fields such as top_p and the Anthropic sampling fields reaching the engine) is only partly covered, since the range tests send them but do not assert they arrive
  - [x] Audit of every field the chat and messages endpoints read, with its tests and classification (PR 82)
  - [x] P7b: `max_completion_tokens`, `seed`, `top_k` implemented; `n` other than 1 and out-of-range sampling values refused with a 400 naming the field; `parallel_tool_calls`, `user` and `metadata` documented as intentionally ignored (30 tests, mutation-checked)
  - [x] P7c: `stop` and `stop_sequences` (one filter around the engine stream in all four handlers; generation is cancelled, not just the text hidden). 35 tests: 19 on the filter including 400 random inputs against a character-level reference, 16 through the endpoints; five production mutations each fail tests (the endpoint stream not filtered 4, no `stop_sequence` reason 2, no cancellation 7, no holdback 3, no validation 6)
  - [x] P7d: `stream_options.include_usage` (final usage chunk) and Anthropic `tool_choice` (auto and none honoured; any and tool refused with a 400). Limit: the `none` wiring has no failing endpoint test
  - [x] P7e: error bodies now also carry the nested `error` object on both APIs, additively (the flat fields stay; Anthropic's top-level `type` is not changed to `error`)
- [x] **P8** HTTP-disconnect cancellation, end to end: single-model half by the delegate session (PR 81, taken over unchanged), multi-model half with the runtime handle added here
  - [x] Check which existing tests already cover it (done in PR 81)
  - [x] Test: generation started, client disconnects, cancellation reaches the engine, slot released, next request completes (single-model host, delegate's test)
  - [x] Test: the same in a multi-model host with a fake loader (`NamedModelOptions.EngineFactory`): positive control that the runtime holds one handle while generating, then the handle is back to zero after the disconnect and the model serves the next request
  - [x] Mutation-checked: never disposing the runtime handle fails only the multi-model test; ignoring the client disconnect token fails both; the delegate's three mutations are in PR 81's description
- [x] **P4** Fair gate measured (two resident models, interleaved, 3 runs). Machine 10 to 16 percent busy, not idle; stated with the numbers. Result: off by default stays right, and the single-request engine is not worth mediating (plan 032, Slice 5)
  - [x] P4b `STINGRAY_PREFILL_CHUNK` and the server option `PrefillChunkTokens` never reached the batching engine; wired, 4 real-model tests, three production mutations each fail tests
- [x] **P9** Reproducibility helper used by the benchmark scripts (PR 78; self-test run again here: all checks pass)
- [x] **P10** `stingray config --show` (narrow), with a redaction test (PR 79; source is environment only and the output says so)
