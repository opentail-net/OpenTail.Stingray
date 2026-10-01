# Correctness backlog — open items only

Last updated 2026-10-01. **Open items only.** When an item closes, move its entry (and its plan, if any) to
[../done/](../done) — everything closed so far is in
[../done/bugstofix-resolved-2026-10.md](../done/bugstofix-resolved-2026-10.md) and
[../done/bugstofix-resolved-2026-08.md](../done/bugstofix-resolved-2026-08.md). Item numbers are stable
and are not renumbered.

State key: 🔴 open defect · 🟡 open investigation · 🔵 blocked on an asset or hardware · ⚪ deferred (no concrete model target).

| # | Area | State | Next action | Latest evidence |
| --- | --- | --- | --- | --- |
| 15 | LLaVA-NeXT AnyRes (residual) | ⚪ | 2026-10-01: geometry, token counts and separators match llama.cpp; view-order defect found and fixed (overview now last, as mtmd); first-token top-9 agree on 2 images. Unverified: pixel-level resize, per-view embeddings, OneVision | Plan: `../done/15-llava-next-anyres-parity-plan.md` (Resolution) |
| 19 | RWKV6 CPU | ⚪ | Target chosen: `E:/_models/rwkv6-world-1b6` (Finch 1.6B, Q8_0 + Q4_K_S, arch `rwkv6`). Implementation project: scalar WKV6 oracle, graph, parity vs llama.cpp | `../done/17-ggml-op-coverage-verification-plan.md` |
| 20 | RWKV7 CPU | ⚪ | Target chosen: `E:/_models/rwkv7-goose-world3-1b5` (Goose World3 1.5B, Q8_0 + Q4_K_S, arch `rwkv7`). Same shape of project | same |
| 22 | Stale numpy-golden vision parity tests | 🟡 | `LlavaVisionEmbedderParityTests` (min cosine -0.21), `Qwen25Vl/MimoVl/Exaone4 ...EmbedderParityTests.Forward_MatchesNumpyReference` fail in the full Vision run (2026-10-01: 192 total, 4 failed, 47 skipped). Not caused by item 15 (encoders untouched; LLaVA encoder was changed 2026-09-27 to match llama.cpp, fixture dates from 2026-09-01). Replace the numpy fixtures with `llama-mtmd-debug` oracles (no new Python) or retire the tests | `tests/OpenTail.Stingray.Tests.Vision` |
| 23 | DeepSeek-V2-Lite greedy parity regressed | 🟡 | 2026-10-01: `DeepSeek2GreedyParityTests.DeepSeek2_GreedyContinuation_MatchesLlamaServer` (Q2_K GGUF, llama-server reference captured 2026-09-26) now diverges at generated token 9 (expected 6636, got 18684). Reproduced identically on the code BEFORE the 2026-10-01 Q4_K matvec scheduling change, so it predates it; it passed when admitted (commit 113734d2). Not bisected yet. | `git bisect` from 113734d2 with `-class OpenTail.Stingray.Tests.ForwardPass.DeepSeek2GreedyParityTests` |

(Item 21, generic `SOLVE_TRI`, is closed; see the resolved file.)

## Tracked items

- [ ] **15 (residual). LLaVA-NeXT AnyRes**: the main defect (view order) is fixed and geometry/token counts/first-token agree with llama.cpp (plan: [../done/15-llava-next-anyres-parity-plan.md](../done/15-llava-next-anyres-parity-plan.md), Resolution). Still unverified: pixel-level resize/pad parity and per-view embedding parity (no oracle: `llama-mtmd-debug preproc` prints geometry only), and LLaVA-OneVision.
- [ ] **19. RWKV6 CPU coverage**: No WKV6 kernel or RWKV6 model graph exists. Add this only against a concrete model target; require a scalar WKV6 oracle, recurrent-state/reset tests, and real-checkpoint parity before admission. Current source audit: [GGML op coverage verification](../done/17-ggml-op-coverage-verification-plan.md).
- [ ] **20. RWKV7 CPU coverage**: No WKV7 kernel or RWKV7 model graph exists. Track independently from RWKV6 because the recurrence differs; require its own scalar oracle, recurrent-state/reset tests, and real-checkpoint parity before admission. Current source audit: [GGML op coverage verification](../done/17-ggml-op-coverage-verification-plan.md).
- [ ] **22. Stale numpy-golden vision parity tests**: see the dashboard row. Per-model residual notes from the closed sweep (Exaone4 min token cosine 0.825366, Llava -0.21, MimoVL 0.249, Qwen2.5-VL 0.283; fresh MTMD aggregates agree for Exaone4/MimoVL/Qwen2.5-VL) are in [../done/01-real-weight-landscape-sweep-rerun-plan.md](../done/01-real-weight-landscape-sweep-rerun-plan.md). LLaVA-1.5 7B main checkpoint is absent for the Llava fixture.
- [ ] **23. DeepSeek-V2-Lite greedy parity regression**: see the dashboard row (bisect from 113734d2). The earlier investigation is closed: [../done/032-deepseek2-mla-yarn-moe-routing-investigation.md](../done/032-deepseek2-mla-yarn-moe-routing-investigation.md); do not restart it without new evidence.
