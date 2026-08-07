# CPU prefill quality gate — corpus perplexity evidence

**Measured:** 2026-08-07. Qwen3-8B Q4_K_M, wikitext-2 subset (120 KB, 1792 scored tokens),
`perplexity -c 2048 -g 0 --batched --batch-chunk-size 512` on a Ryzen 7 5700G (Zen 3, AVX2 only).
Perplexity is deterministic for a fixed code path, so these figures are reproducible rather than
sampled; the throughput figures are single-sample on a machine in light use and are soft.

## int8 activation prefill: faster AND marginally better

| configuration | ppl [256,1024) | ppl [1024,+) | throughput |
|---|---:|---:|---:|
| `STINGRAY_CPU_PREFILL_Q8=1` (default) | **6.0579** | **7.5318** | **23.59 tok/s** |
| `STINGRAY_CPU_PREFILL_Q8=0` | 6.0789 | 7.5660 | 7.12 tok/s |

The int8 path is **3.3x faster and 0.35% better in perplexity**, not worse. That disconfirms the
natural assumption that it trades quality for speed, and supports the default-ON setting with
corpus evidence instead of inference.

"Better" here should not be read as "more accurate": both routes are approximations with different
rounding orders, and a 0.35% difference at this corpus size is small. The defensible claim is that
int8 prefill **does not degrade** corpus perplexity on this model, which is what the gate needs.

## The flag controls more than its name says

`STINGRAY_CPU_PREFILL_Q8=0` does **not** isolate int8 activations. The Q4Kx8 repacked GEMM is
gated by the same flag (`ForwardPass.cs`: `if (SimdKernels.Q8PrefillEnabled) { GetRepackedQ4Kx8(...)
TryMatMulBatchedQ4Kx8(...) }`), so disabling it also disables the repacked path — separately
documented as worth 1.80x end-to-end.

So the 3.3x above is **int8 + repack combined**, not int8 alone, and anyone disabling this flag to
study int8 numerics in isolation will misattribute both the speed and the quality result. Ranking
which of the two contributes the perplexity delta requires separating the gates first.

## Calibration for other numerics decisions

These numbers give a yardstick for what a perplexity movement means on this model and corpus:

| change | perplexity effect | shipped? |
|---|---|---|
| int8 activation prefill (+ repack) | **-0.35%** (better) | yes, default ON |
| Q4Kx8 repack alone (recorded elsewhere) | 16.0488 -> 16.0484, ~0% | yes, default ON |
| Flash-64 head dims 128/256 | **+0.52%** (worse) | no — opt-in only |

Flash-128 is a regression against every configuration currently shipped, including the exact
sequential path (6.0896 vs 6.0789), which is why it stayed behind
`STINGRAY_PREFILL_ATTN_WIDE_HEADS`.

## Two methodology traps hit while producing this

1. **`perplexity` defaults to token-by-token `Forward`, which never enters batched prefill.** A gate
   run without `--batched` produced byte-identical NLL across both arms — a confident-looking null
   from a treatment that was never applied. The `--batched` flag's own description states this
   plainly. Identical-to-six-decimals is a signal that the treatment did not land, not a result.
2. **Q8=0 changes the code path more than intended** (see above), so the arms differ in two ways.

## Not yet done

- Full wikitext-2 test split rather than a 120 KB subset.
- Multiple context lengths (only `-c 2048` here; the two position buckets give a partial view).
- Separating the repack gate from the int8 gate so each can be attributed independently.
