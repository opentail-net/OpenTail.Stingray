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

- [ ] **15 (residual). LLaVA-NeXT AnyRes**: the main defect (view order) is fixed and geometry/token counts/first-token agree with llama.cpp (plan: [../done/15-llava-next-anyres-parity-plan.md](../done/15-llava-next-anyres-parity-plan.md), Resolution). On 2026-10-18, the Vision project rebuilt with 0 warnings/errors and all 16 `LlavaImagePreprocessorGeometryTests` passed. Reviewed llama.cpp `b10306` `tools/mtmd/mtmd-image.cpp`; best-resolution selection and row-major slicing agree with Stingray. Still unverified: pixel-level resize/pad parity and per-view embedding parity. The installed `llama-mtmd-debug preproc` exposes geometry only, and the generated comparison image from the prior real-model run is not retained; no reproducible local numerical mismatch is available to fix. End-to-end evidence (2026-11-01, llama.cpp `llama-mtmd-cli` vs `stingray` on `llava-v1.6-mistral-7b.Q5_K_M.gguf` + `mmproj-model-f16.gguf`, greedy, 40 tokens): on a 256×256 PNG (one 336 view, 576 image tokens) the output matches word for word with `--repeat-penalty 1.0`; the default penalty 1.1 changes the first divergence (" a" to " an"), so the repeat penalty, not the vision path, was the cause. On a 512×512 PNG (`docs/diffusion-samples/sd15-apple-512.png`, AnyRes) both sides build 2880 image tokens and 2896 prompt tokens. The text diverges at the second word: llama.cpp "A red apple…", Stingray "A close-up of a red apple…". A forced-prefix check puts " close" at -1.3143 and " red" at -1.3200 (gap 0.006 nats), a near-tie that numerical differences can flip, so no defect is shown. Still unverified: pixel-level resize/pad parity and per-view embedding parity (the reference debug tool accepts only synthetic images and exposes no real-image pixels or embeddings), so this item stays open. LLaVA-OneVision remains uncovered.

