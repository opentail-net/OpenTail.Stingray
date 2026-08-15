> **ARCHIVED, 2026-08-15.** Completed measurement/investigation snapshot, not an open plan.
> Filed here as evidence per the [archive convention](README.md).

---

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
the dequant/dot work behind it — see `done/cpu-speculative-decoding-findings.md`), so moving the work to
a GPU on the *same* memory bus cannot help. A discrete card with its own VRAM is a different
measurement entirely, and nothing here predicts it.

**Operational reading:** on an APU, `-g -1` is not an optimisation. The Vulkan path earns its place
by being correct and available, not by being quicker, and a default that prefers GPU-when-present
would make this machine slower at prefill.

## Logit parity vs CPU — measured (supersedes the text comparison below)

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


## RESOLVED + OPEN 2026-08-07 — logit parity measured

`VulkanCpuLogitParityTests` compares prefill logits directly rather than generated text.
SmolLM2-1.7B-Q4_K_M, 17-token prompt:

| metric | result |
|---|---|
| **argmax** | **CPU 23 == Vulkan 23** — agree |
| cosine | **0.99195** |
| maxAbs | 1.476 |

**Resolved:** the greedy text divergence is not a decision-level disagreement. Both backends pick
the same next token; the earlier "operating system" vs "the system" split is downstream
amplification of later differences, which is what greedy decoding does to any two backends with
different floating-point orderings.

**Open, and deliberately not closed:** cosine 0.992 is roughly **12x looser** than the CPU-side
approximations this repo already accepts — int8 activation prefill at 0.999504 and Flash-128 at
0.999345. No cause has been established. The FP16 narrowed-KV store is opt-in via
`STINGRAY_KV_DTYPE` and was not in use, so it does not explain this.

The test asserts argmax equality (verified, and the property the sampler actually depends on) and
holds cosine at an **empirical floor of 0.99** — a regression guard on the measured baseline, not a
claim that 0.992 is right. The threshold was deliberately not fitted to today's number: had this
divergence been a real defect, a tolerance tuned to make it pass would have concealed it
permanently. Tighten it only once the gap is explained.

Worth noting the ordering: an initial threshold of 0.999 was written **before** measuring, and it
failed. Choosing the bound first is what turned "Vulkan looks fine" into a specific open question.


## CORRECTION 2026-08-07 — the 0.992 "open question" was my prompt, not Vulkan

The section above recorded Vulkan/CPU cosine 0.99195 as roughly 12x looser than the CPU-side
approximations, with no cause established. **That framing was wrong, and the cause was the test
prompt.** Both that measurement and the first parity test used the synthetic id sequence
`[1, 2, 3, 5, 7, 11, ...]`. Arbitrary low-numbered ids are an out-of-distribution token sequence,
and the int8 activation prefill handles them badly.

A sweep over prompt lengths, comparing three ways rather than two, shows it plainly:

| n | Vulkan vs CPU-int8 | Vulkan vs CPU-F32 | CPU-int8 vs CPU-F32 |
|---:|---:|---:|---:|
| 1 | 0.998132 | 0.998132 | 1.000000 |
| 2 | 0.975049 | 0.988505 | 0.987387 |
| **8** | **0.570589** | **0.997638** | **0.547410** |
| 32 | 0.997180 | 0.998700 | 0.996181 |
| 128 | 0.997206 | 0.998126 | 0.996627 |

At n=8 Vulkan agrees with exact-F32 CPU at **0.9976** while the CPU's own int8 path disagrees with
its own F32 path at **0.547**. The outlier was never Vulkan. Repeating the sweep with real
tokenizer output puts every pairing at 0.997-0.999 across all lengths.

Two corrections follow:

1. **Vulkan's divergence is normal.** On real text it sits at the same scale as the CPU's own
   int8-vs-F32 gap. There is no unexplained Vulkan looseness.
2. **The int8 prefill's out-of-distribution sensitivity is real and broader than the shipped
   mitigation.** That mitigation skips int8 only when a prompt is *entirely* control/user-defined
   tokens. This sequence is not — it spans ids up to ~900 — and still produced cosine 0.547 at 8
   tokens. Synthetic token sequences in any test that compares numerics are therefore actively
   misleading, which is worth knowing well beyond this file.

The parity test now uses real tokenizer output and asserts a contract that survives it: cosine
above 0.99, and argmax disagreement permitted **only when it is a near-tie** (CPU rates its own pick
less than 2% of the logit range above Vulkan's pick). Exact argmax equality was the wrong
cross-backend contract — near-tied candidates flip on any FP ordering difference and neither
backend is wrong when the model had no real preference.
