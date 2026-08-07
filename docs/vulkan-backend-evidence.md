# Vulkan backend — measured evidence (AMD APU)

**Measured:** 2026-08-07. Ryzen 7 5700G with integrated AMD Radeon Graphics, Vulkan reporting a
15.7 GiB placement budget. SmolLM2-1.7B-Q4_K_M, greedy (`--temp 0`), 46-token prompt, 48 tokens
generated.

## This backend was nearly written off by mistake

An earlier assessment in this session recorded "zero GPU evidence possible on this machine" after
`doctor` reported no NVIDIA CUDA driver. That conclusion was wrong: it generalised from CUDA to all
GPU backends without checking. `doctor` had in fact been reporting
`backend.vulkan: OK — Vulkan GPU (AMD Radeon(TM) Graphics)` the whole time, on the line below the
CUDA warning. An entire backend axis of the release matrix was declared blocked on an assumption
that took one command to falsify.

## It works, and it is not faster

| Backend | Prefill | Decode |
|---|---:|---:|
| CPU (`-g 0`) | 96.1 t/s | 23.7 t/s |
| Vulkan (`-g -1 --backend vulkan`) | 84.2 t/s | 24.0 t/s |

All 24 layers upload to VRAM and generation is coherent, so the path is functional. But **decode is
a dead heat and prefill is ~12% slower**.

That is the expected result for an **APU**, and worth stating so nobody reads it as a Vulkan defect:
integrated Radeon graphics share system DRAM with the CPU. Decode is bound by that memory (and by
the dequant/dot work behind it — see `cpu-speculative-decoding-findings.md`), so moving the work to
a GPU on the *same* memory bus cannot help. A discrete card with its own VRAM is a different
measurement entirely, and nothing here predicts it.

**Operational reading:** on an APU, `-g -1` is not an optimisation. The Vulkan path earns its place
by being correct and available, not by being quicker, and a default that prefers GPU-when-present
would make this machine slower at prefill.

## Open: greedy output diverges from CPU

At `--temp 0` the two backends produce different text, diverging around token 12:

- CPU: "...allows the **operating system** to efficiently utilize memory by storing frequently
  accessed data in cache, reducing the time spent on accessing slower main memory..."
- Vulkan: "...allows the **system** to efficiently utilize memory by minimizing the number of times
  data needs to be read from or written to main memory..."

This is **not yet evidence of a defect**. Greedy decoding is a maximal amplifier: one argmax flip
early on rewrites everything after it, so text divergence is what any two backends with different
floating-point orderings will produce. Both continuations are fluent and on-topic, which is what
would *not* be true of a real numerical fault.

**It is also not yet evidence of correctness.** Establishing that requires comparing prefill logits
directly — cosine plus argmax agreement at position 0 — rather than reading generated text, exactly
as the Flash 128/256 decision required. That measurement has not been run.

## Not measured

- Prefill logit cosine / argmax agreement against the CPU path (the actual parity check).
- Longer contexts, other models, MoE, or batching on Vulkan.
- Any discrete-GPU comparison. Every number here is APU-specific.
