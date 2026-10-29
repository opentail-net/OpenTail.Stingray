# Correctness backlog — open items only

Last updated 2026-10-20. **Open items only.** When an item closes, move its entry (and its plan, if any) to
[../done/](../done) — everything closed so far is in
[../done/bugstofix-resolved-2026-10.md](../done/bugstofix-resolved-2026-10.md) and
[../done/bugstofix-resolved-2026-08.md](../done/bugstofix-resolved-2026-08.md). Item numbers are stable
and are not renumbered.

State key: 🔴 open defect · 🟡 open investigation · 🔵 blocked on an asset or hardware · ⚪ deferred (no concrete model target).

| # | Area | State | Next action | Latest evidence |
| --- | --- | --- | --- | --- |
| 15 | LLaVA-NeXT AnyRes (residual) | ⚪ | 2026-10-18: no local defect isolated. Rebuilt Vision tests (0 warnings/errors); `LlavaImagePreprocessorGeometryTests` passes 16/16. Reviewed pinned llama.cpp b10306 `mtmd-image.cpp`: best-resolution selection and row-major slicing match the implemented geometry. Pixel/per-view parity remains unverified: available mtmd debug `preproc` reports geometry only, and the earlier generated comparison image is not retained. | Plan: `../done/15-llava-next-anyres-parity-plan.md` (Resolution) |

(Items 19, 20, 21, 22, 23 and 24 are closed; see the resolved file.)

## Tracked items

- [ ] **15 (residual). LLaVA-NeXT AnyRes**: the main defect (view order) is fixed and geometry/token counts/first-token agree with llama.cpp (plan: [../done/15-llava-next-anyres-parity-plan.md](../done/15-llava-next-anyres-parity-plan.md), Resolution). On 2026-10-18, the Vision project rebuilt with 0 warnings/errors and all 16 `LlavaImagePreprocessorGeometryTests` passed. Reviewed llama.cpp `b10306` `tools/mtmd/mtmd-image.cpp`; best-resolution selection and row-major slicing agree with Stingray. Still unverified: pixel-level resize/pad parity and per-view embedding parity. The installed `llama-mtmd-debug preproc` exposes geometry only, and the generated comparison image from the prior real-model run is not retained; no reproducible local numerical mismatch is available to fix. LLaVA-OneVision remains uncovered.

