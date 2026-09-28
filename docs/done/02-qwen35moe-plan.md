> **CLOSED 2026-09-28.** Architecture and parity coverage complete:
> - Ornith-1.0 9B and Qwen3.8-27B validated end-to-end with exact greedy match against llama.cpp.
> - Chunked GDN recurrence parallelized over heads and tokens (see `103-quickest-first-plan.md` item 15, reaching ~0.96x llama.cpp).
> - Parity defect item 11.d analyzed, resolved, and pinned by `GdnKernelsTests`.
> - Retained-session conformance tracked under general session runtime improvements.

# Qwen3.5 MoE / Gated DeltaNet — current work

**Status:** closed / archived to `docs/done/`. The target uses Gated DeltaNet linear-attention recurrence plus MoE.

- [x] 1. Run the existing path against the reference GGUF and capture load, greedy parity, context, batching, and hybrid-placement evidence (Ornith-1.0 9B and Qwen3.8-27B validated on CPU and full CUDA offload).
- [x] 2. Turn a failure into a narrow tensor/operation discrepancy before designing a kernel.
- [x] 3. GDN state-lifecycle conformance & chunked parity (item 11.d closed in `103-quickest-first-plan.md`).
- [x] 4. Benchmark only after correctness passes (completed in `103-quickest-first-plan.md` item 15, reaching ~0.96x llama.cpp).

References: [qwen35moe-tensor-layout.md](../reference/qwen35moe-tensor-layout.md) and
[done/qwen35moe-plan-superseded.md](../done/qwen35moe-plan-superseded.md).
