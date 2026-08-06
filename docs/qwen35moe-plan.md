# Qwen3.5 MoE / Gated DeltaNet — current work

**Status:** the original SSM/Mamba plan was superseded. The target uses Gated DeltaNet
linear-attention recurrence plus MoE.

1. Run the existing path against the reference GGUF and capture load, greedy parity, context,
   batching, and hybrid-placement evidence.
2. Turn a failure into a narrow tensor/operation discrepancy before designing a kernel.
3. Add GDN state-lifecycle conformance tests, including retained-session compatibility.
4. Benchmark only after correctness passes.

References: [qwen35moe-tensor-layout.md](qwen35moe-tensor-layout.md) and
[done/qwen35moe-plan-superseded.md](done/qwen35moe-plan-superseded.md).
