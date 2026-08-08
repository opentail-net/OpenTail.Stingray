> **Reprioritized 2026-08-08 — now runway position 2.** Items 1 and 2 are largely covered in
> practice: Ornith-1.0 9B (arch `qwen35`, which ships GDN tensors and therefore takes the hybrid
> Gated-DeltaNet path) has been validated end to end on CPU and full CUDA offload. What remains is
> item 3 (GDN state-lifecycle conformance, including retained-session compatibility) and item 4
> (benchmark). This family stays high on the runway because it is widely distributed on Hugging
> Face — see [00-current-work.md](00-current-work.md).

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
