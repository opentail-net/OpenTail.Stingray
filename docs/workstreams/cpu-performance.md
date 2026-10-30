# Workstream: CPU performance

> This file summarises its owner plans. If the two disagree, the owner plan is authoritative; correct this file.

**Owners:** [../4-performance/perf-sweep-plan.md](../4-performance/perf-sweep-plan.md) (the phased sweep, 22 open) and [../4-performance/2026-10-04-q4k-expert-gemm-plan.md](../4-performance/2026-10-04-q4k-expert-gemm-plan.md) (CPU MoE expert GEMM, plan only). **Index:** [../4-performance/2026-10-30-performance-pickup.md](../4-performance/2026-10-30-performance-pickup.md). **Last verified:** 2026-10-30.

## Correct state

- **Text models on the CPU are mixed.** The 2026-10-11 recheck estimates a CPU prefill median of 0.93x and a decode median of 0.69x against stored llama.cpp numbers (see the pickup doc).
- **The sweep's phases, from its headings:** Phase 1 (Voxtral) declared a plateau on 2026-09-12. Phase 2 (the Qwen3.6 hybrid GDN family) and Phase 3 (small-model decode) are open. Phase 6 (Gemma family prefill) is open.
- **Expert GEMM for CPU MoE prefill:** the plan says nothing is implemented. The code search on 2026-10-30 found no expert-GEMM kernel.
- **Tried and not repeated:** the list is in the pickup doc ("Already tried").

## Open items

- [ ] Perf sweep: the 22 open items in [perf-sweep-plan.md](../4-performance/perf-sweep-plan.md), phase by phase
- [ ] Q4_K expert GEMM: the three open items in its plan (options A or B, the plan's section 5, and the owner's decisions)
- [ ] Pickup rows 8 and 9 (long-context decode; OLMoE) still have no owning plan

## Cadence

- Review after each dated measurement, recorded in `PerformanceLeague.md`.
- Re-read at least every two weeks while any item is open.
- Close when every sweep phase is either closed or moved to `done/`, and the expert-GEMM plan is decided.
