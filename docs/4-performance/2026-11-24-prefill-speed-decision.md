# Prefill speed on the CPU default route: not doing it (2026-11-24)

**Status:** closed by owner decision on 2026-11-24. No code change. Reopen only under the conditions in the last section.

## Decision

We are not doing further prefill speed work on the CPU default route. The experiments that could be run without changing numerics did not move speed. The one large gain, the batched int8 path (about 2.7× on SmolLM2), changes numerics, and its default is an owner decision under [ADR-0003](../reference/adr-0003-cpu-int8-prefill-default.md). That decision is unchanged.

## Context

Performance item 1 in [00-current-work](../00-current-work.md) says SmolLM2 prefill runs at about 0.89× llama.cpp. The question asked on 2026-11-24 was whether the default CPU prefill could be made faster without changing its numerics.

**Correction to earlier notes.** For ordinary prompts the default is not a per-token loop. Prefill goes through the batched `PrefillCore` path. With Q8 off, Q4_K matmuls use the batched F32 path: dequantised weights, with the dequant cache on by default when the F32 copy fits in a quarter of available memory. The per-token loop runs only for control-only or single-token prompts and some recurrent models. The "per-token default" label in ADR-0003 and older notes is misleading. Fix it when ADR-0003 is next edited.

## Evidence

Machine: Ryzen 7 5700G (AVX2, no VNNI, no AVX-512), 63 GB RAM, some background load (about 10%). Model: SmolLM2-1.7B Q4_K_M. Three runs per setting unless noted.

| Experiment | Result (tok/s) |
|---|---|
| Profile, default route, 2550-token prompt (sampling, 76 working samples) | 75 overall; batched F32 GEMM about 80% of working samples, attention about 18% |
| Default, 871-token prompt | 80.6, 80.0, 86.0 |
| Spin thread pool (`STINGRAY_CPU_POOL=spin`) | 23.0, 8.3, 23.1. Slower; generated text identical |
| F32 GEMM row chunks, 8 per thread vs 2 | 81.8, 82.1, 81.3 vs 80.9, 89.4, 82.9. No change; generated text identical |
| Dequant cache on vs off (`STINGRAY_PREFILL_DEQUANT_MB=0`) | about 82 vs 85; one outlier at 32.8 unexplained |
| Batched int8 path (`STINGRAY_CPU_PREFILL_Q8=1`) | 196.3, 217.6, 222.8. About 2.7× the default |

Accuracy evidence already in ADR-0003, not rerun here: batched int8 is +0.074% perplexity against per-token on SmolLM2 (`[1024,+)` bucket), and it is slightly better than per-token on Qwen2.5-0.5B (11.9693 against 11.9818). Neither path is a general accuracy winner.

## What we did not do

- No bit-identical kernel was built. The candidate is a repacked Q4_K kernel matching single-token `MatVec` exactly. Nothing was tried, and nothing suggests it is cheap.
- No Qwen timing, and no timing on other hardware.

## Reopen only if

- The owner decides to make the batched int8 path the default. That needs a repeat-run accuracy comparison first.
- A new Q4_K kernel is shown to be bit-identical to single-token `MatVec` and measurably faster on repeat runs.
- The work moves to hardware with VNNI or AVX-512, where the int8 dot is cheaper. This was not tested.

## Limits

One machine. One model for timing. Profile sampling is coarse (about 76 working samples). Timings are wall-clock with background load, not a clean perf baseline.
