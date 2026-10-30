# Completed items split from 2026-10-02-tensorsharp-takeaways-plan.md (2026-10-30)

Finished items moved verbatim out of [../2-coverage/2026-10-02-tensorsharp-takeaways-plan.md](../2-coverage/2026-10-02-tensorsharp-takeaways-plan.md). Open items stay there. Each block notes the section it came from.

> from: ## 1. Spin-then-park CPU worker pool (performance experiment)
  - [x] Port behind a switch (e.g. `STINGRAY_CPU_POOL=spin`) and route `SimdKernels`'s matvec
     `Parallel.For` sites through it.

> from: ## 1. Spin-then-park CPU worker pool (performance experiment)
  - [x] Prove the dispatch happened (a call counter).

> from: ## 1. Spin-then-park CPU worker pool (performance experiment)
  - [x] Run an interleaved A/B on 135M / 360M / Qwen2.5-0.5B decode and one 7B; outputs must be
     bit-identical, since only scheduling changes.

> from: ## 1. Spin-then-park CPU worker pool (performance experiment)
  - [x] Keep it only if it is measurably better; record the result either way.

