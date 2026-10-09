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
| 24 | DeepSeek-V2-Lite greedy receipt vs Q3_K kernel | 🔴 | **User decision needed.** `e7b7aa8a` (2026-10-02, Q3_K x Q8_K dot rewritten in ggml's exact AVX2 shape; Granite-Vision-3.2 decode 0.48x -> 0.74x) flips `DeepSeek2GreedyParityTests` at generated token 9 (18684 vs 6636), the same knife-edge token the F32 prefill flips. Options: revert `e7b7aa8a`, or accept the flip as a documented knife-edge | Bisected 2026-10-02: passes at `88204c4f`, fails at `e7b7aa8a`. See tracked item |

(Items 19, 20, 21, 22 and 23 are closed; see the resolved file.)

## Tracked items

- [ ] **15 (residual). LLaVA-NeXT AnyRes**: the main defect (view order) is fixed and geometry/token counts/first-token agree with llama.cpp (plan: [../done/15-llava-next-anyres-parity-plan.md](../done/15-llava-next-anyres-parity-plan.md), Resolution). Still unverified: pixel-level resize/pad parity and per-view embedding parity (no oracle: `llama-mtmd-debug preproc` prints geometry only), and LLaVA-OneVision.
- [ ] **24. DeepSeek-V2-Lite receipt broken by the Q3_K kernel rewrite (2026-10-02)**. Bisect: `DeepSeek2GreedyParityTests` (real run, ~6s) passes at `88204c4f` and fails at `e7b7aa8a`, whose Q3_K x Q8_K dot is now integer- and float-exact to ggml's x86 `ggml_vec_dot_q3_K_q8_K` (verified line by line) and *closer* to the exact scalar reference on 27,686 real rows (mean rel 3.2e-7 vs 8.3e-7). Our Q8_K quantizer is also identical to `quantize_row_q8_K_ref`. So the old kernel matched llama-server on token 9 by error cancellation with other non-exact kernels (this checkpoint mixes Q2_K, Q3_K and IQ4_NL).
  - Tried 2026-10-02, not committed: making the Q8_0 activation quantizer ggml-exact (`id = 127/max`, fp16-rounded scale) plus an exact-shape AVX2 IQ4_NL dot makes DeepSeek pass again but breaks `GraniteHybridGreedyParityTests` (granite-4.0-h-350m Q8_0, passes at HEAD). With only the IQ4_NL dot exact, Granite passes and DeepSeek fails later (token 11). The receipts pull in opposite directions; likely because llama.cpp on AVX2 runs *repacked* Q8_0/IQ4_NL GEMV kernels (different accumulation order), so matching plain `vec_dot` is not matching what llama-server executed. Not verified.
  - Parked work (applies cleanly to `0298f88e`+): `docs/4-performance/patches/2026-10-02-deepseek-iq4nl-avx2-dot.patch` (AVX2 IQ4_NL x Q8_0 dot; the current one is fully scalar) and `...-deepseek-moe-folded-q2k-iq4nl.patch` (Q2_K/IQ4_NL routed experts on the folded MoE decode path). Together: DeepSeek-V2-Lite Q2_K decode 12.7 -> 25-27 t/s (llama.cpp tg64 36.4). Kernel check: AVX2 IQ4_NL vs scalar mean rel 6.4e-7 over 70,739 real rows.
  - Decision for the user: (a) revert `e7b7aa8a` (receipt passes, lose the Q3_K speedup), or (b) keep it and re-classify token 9 as a documented knife-edge (as was done for the F32-prefill flip), then land the parked DeepSeek decode work.
