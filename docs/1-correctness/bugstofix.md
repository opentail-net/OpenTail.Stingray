# Correctness backlog — open items only

Last updated 2026-10-02. **Open items only.** When an item closes, move its entry (and its plan, if any) to
[../done/](../done) — everything closed so far is in
[../done/bugstofix-resolved-2026-10.md](../done/bugstofix-resolved-2026-10.md) and
[../done/bugstofix-resolved-2026-08.md](../done/bugstofix-resolved-2026-08.md). Item numbers are stable
and are not renumbered.

State key: 🔴 open defect · 🟡 open investigation · 🔵 blocked on an asset or hardware · ⚪ deferred (no concrete model target).

| # | Area | State | Next action | Latest evidence |
| --- | --- | --- | --- | --- |
| 15 | LLaVA-NeXT AnyRes (residual) | ⚪ | 2026-10-01: geometry, token counts and separators match llama.cpp; view-order defect found and fixed (overview now last, as mtmd); first-token top-9 agree on 2 images. Unverified: pixel-level resize, per-view embeddings, OneVision | Plan: `../done/15-llava-next-anyres-parity-plan.md` (Resolution) |

(Items 19, 20, 21, 22, 23 and 24 are closed; see the resolved file.)

## Tracked items

- [ ] **15 (residual). LLaVA-NeXT AnyRes**: the main defect (view order) is fixed and geometry/token counts/first-token agree with llama.cpp (plan: [../done/15-llava-next-anyres-parity-plan.md](../done/15-llava-next-anyres-parity-plan.md), Resolution). Still unverified: pixel-level resize/pad parity and per-view embedding parity (no oracle: `llama-mtmd-debug preproc` prints geometry only), and LLaVA-OneVision.

