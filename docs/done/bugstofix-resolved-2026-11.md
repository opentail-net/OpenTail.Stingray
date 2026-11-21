# Correctness items resolved — November 2026

## 25. CPU batched-int8 prefill numerical parity — characterized

**Resolution (2026-11-21):** the CPU batched-Q8 prefill path is a faster, lower-precision alternative whose numerical effect depends on model, prompt, and metric. The parity investigation did not isolate a correctness defect in the Q8 implementation or establish a generally more accurate mode. Keep sequential per-token prefill as the default because it follows the same computation path as token-by-token decode; keep batched Q8 opt-in with `STINGRAY_CPU_PREFILL_Q8=1` (see [ADR 0003](../reference/adr-0003-cpu-int8-prefill-default.md)). Reopen this item if a reproducible discrepancy is found in the Q8-eligible route or requirements change.

Evidence is recorded in [A9c](../00-current-work.md) and `Q8PrefillGreedyParityTests`:

- Paired 2048-token `wiki.test.raw` runs: SmolLM2 `[1024,+)` PPL was 6.9839 per-token, 6.9437 batched F32, and 6.9891 batched Q8 (llama.cpp b10306: 6.9414). Qwen2.5-0.5B was 11.9818 per-token and 11.9693 Q8 (llama.cpp: 12.0055).
- First-64 teacher-forced checks: SmolLM2 mean absolute NLL error was 0.3097 per-token vs 0.1141 Q8; Qwen2.5-0.5B was 0.0575 vs 0.0565. Differences vary by model and metric.
- Raw greedy/logprob checks cover SmolLM2, Qwen2.5, Granite 4 H 350M MoE, Phi-2, and Gemma 1.1. Q8 sometimes matches more reference tokens, but other prompts favor per-token or both modes agree; neither is a universal numerical winner. Gemma's `The answer is` check has a shared divergence after eight tokens, while its math prompt matches all 16 reference tokens in both modes with identical per-position logprobs.
- `Q8PrefillGreedyParityTests` built without warnings/errors and passed 5/5 on 2026-11-21 with local model assets.
