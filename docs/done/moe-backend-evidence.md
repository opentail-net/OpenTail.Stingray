> **ARCHIVED, 2026-08-15.** Completed measurement/investigation snapshot, not an open plan.
> Filed here as evidence per the [archive convention](README.md).

---

# MoE (mixture-of-experts) — measured CPU evidence

**Measured:** 2026-08-07. `OLMoE-1B-7B-0924-Instruct-Q4_K_M`, Ryzen 7 5700G (Zen 3, AVX2 only),
CPU backend (`-g 0`). This axis of the release matrix was previously unevidenced.

## Functional

Greedy generation is coherent and the routed-expert path runs end to end:

```
Prefill: 28 tokens, 33.6 t/s | Decode: 40 tokens, 27.2 t/s
```

Output for "Explain in one sentence what a mixture-of-experts layer does" was on-topic and fluent.
Decode at 27.2 t/s on a 7B-parameter MoE (1B active) is consistent with routed experts keeping the
per-token weight traffic near the active-parameter count rather than the total.

## Prefill numerics: int8 vs exact F32

| input class (n=32) | cosine |
|---|---:|
| prose | 0.999039 |
| code | 0.998837 |
| whitespace | 0.996447 |
| CJK | 0.992699 |
| repeated `the` x8 | **1.000000** |

**The comparison is not vacuous.** MoE has its own prefill gating (`_hp.IsMoE` with
`MoeBatchedPrefillSupported`), so if int8 never engaged, every row would read 1.000000 and the table
would claim "MoE numerics are fine" while testing nothing. Prose reads 0.999039, not 1.0, which
demonstrates the int8 path is genuinely active. That check was run first, deliberately — this
document series has recorded three prior cases where a treatment silently failed to apply.

The `1.000000` on the repeated-token row is the expected result of the single-distinct-token fix:
such prompts are routed to exact F32, so they match the reference bit-for-bit. It confirms the fix
reaches the MoE path, which is gated separately from the dense one.

Overall MoE prefill sits in the same 0.992-0.999 band as dense CPU prefill. No MoE-specific numerics
problem was found.

## Not measured

- MoE on Vulkan or CUDA (no CUDA driver on this machine; Vulkan MoE untested).
- Expert-offload paths (`--cpu-moe`, the SLRU VRAM expert cache, the 3-tier memory hierarchy) —
  those need a GPU with constrained VRAM to exercise meaningfully.
- Corpus perplexity for MoE; only per-prompt cosine here.
- Longer contexts and batched/continuous-batching behaviour under MoE.
