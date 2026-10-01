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
| 22 | Stale numpy-golden vision parity tests | 🟡 | `LlavaVisionEmbedderParityTests` (min cosine -0.21), `Qwen25Vl/MimoVl/Exaone4 ...EmbedderParityTests.Forward_MatchesNumpyReference` fail in the full Vision run (2026-10-01: 192 total, 4 failed, 47 skipped). Not caused by item 15 (encoders untouched; LLaVA encoder was changed 2026-09-27 to match llama.cpp, fixture dates from 2026-09-01). Replace the numpy fixtures with `llama-mtmd-debug` oracles (no new Python) or retire the tests | `tests/OpenTail.Stingray.Tests.Vision` |

(Items 19, 20, 21 and 23 are closed; see the resolved file.)

## Tracked items

- [ ] **15 (residual). LLaVA-NeXT AnyRes**: the main defect (view order) is fixed and geometry/token counts/first-token agree with llama.cpp (plan: [../done/15-llava-next-anyres-parity-plan.md](../done/15-llava-next-anyres-parity-plan.md), Resolution). Still unverified: pixel-level resize/pad parity and per-view embedding parity (no oracle: `llama-mtmd-debug preproc` prints geometry only), and LLaVA-OneVision.
- [ ] **22. Stale numpy-golden vision parity tests**: see the dashboard row. Per-model residual notes from the closed sweep (Exaone4 min token cosine 0.825366, Llava -0.21, MimoVL 0.249, Qwen2.5-VL 0.283; fresh MTMD aggregates agree for Exaone4/MimoVL/Qwen2.5-VL) are in [../done/01-real-weight-landscape-sweep-rerun-plan.md](../done/01-real-weight-landscape-sweep-rerun-plan.md). LLaVA-1.5 7B main checkpoint is absent for the Llava fixture.
