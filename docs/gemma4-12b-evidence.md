# Gemma 4 12B — measured CPU evidence

**Measured:** 2026-08-07. `gemma-4-12b-it-qat-q4_0` (6.50 GiB), Ryzen 7 5700G (Zen 3, AVX2 only),
CPU backend. Geometry from the GGUF: 48 layers, 3840d, **key/value_length 512 global and 256 SWA**,
per-layer KV head counts, **sliding_window 1024**, context 32768.

## Functional

Loads in 2.5s and generates coherent, on-topic text. No crash, no degenerate output.

## Prefill runs at decode speed — a ~4x penalty

| model | prefill | decode | prefill ÷ decode |
|---|---:|---:|---:|
| SmolLM2-1.7B (dense) | 96.1 | 23.7 | **4.1x** |
| OLMoE-1B-7B (MoE) | 33.6 | 27.2 | 1.2x |
| **Gemma-4-12B** | **3.5** | **4.0** | **0.9x** |

Batched prefill exists to amortise weight reads across tokens. Gemma's prefill running *slower than
its own decode* means it is not batching at all.

Confirmed in code, not inferred from timing: `ForwardPass` computes
`perLayerHdUnsupported = _layerHeadDim is not null && ...` and, when set, prefill is literally
`for (i..N) Forward(tokens[i], startPos + i)` — the decode path in a loop. Decode itself is healthy
at 27.3 GB/s, 74% of this machine's DRAM ceiling, so the model is not slow; only its prefill is.

This is the quantified cost of the outstanding "per-layer-head-dimension batched prefill" work.
`STINGRAY_PER_LAYER_HD_PREFILL=1` forces the batched path and is documented as producing wrong
output for gemma4 (it assumes model-wide KV heads and no sliding window); it exists to make the
remaining work measurable, and running it would size the prize.

## Long-position self-consistency: attempted, and VACUOUS

A prefill-versus-token-by-token invariant was run at n=1200, deliberately crossing the 1024
sliding window where SWA layers evict while global layers still see everything and two KV strides
are live — the configuration in which the earlier `PagedKvCache` stride bug lived.

Result: `cos=1.00000000 maxAbs=0.00000 argmax same`.

**That result is worthless, and the perfection is the tell.** Because Gemma takes the
`perLayerHdUnsupported` branch, `Prefill(1200)` *is* a `Forward()` loop internally. The test
compared that loop against an identical loop written in the test. Two runs of the same code agree
bit-for-bit; nothing about long-position behaviour was exercised.

The oracle-free invariant is a good instrument precisely because it needs no external reference —
but it only works where the two paths are genuinely different. For any model on the sequential
fallback (Gemma 4, and MoE without batched prefill support) it degenerates to comparing a function
with itself. **A test that cannot fail is not evidence**, and this one would have been reported as
"Gemma long-position parity: exact" had the bit-identical result not been suspicious.

## What real parity evidence requires here

1. **An external reference.** `tools/llama.cpp` (b8585) is present and supports this model family, so
   a same-prompt, same-seed greedy comparison against `llama-cli` is the available check — that is
   genuine reference parity rather than self-comparison.
2. **Or** a second internal path that genuinely differs: comparing the sequential route against the
   forced batched route (`STINGRAY_PER_LAYER_HD_PREFILL=1`) would be meaningful, but only once that
   path is correct for per-layer head dims — today it is documented-wrong by construction.

Neither has been run. Long-position parity for Gemma 4 12B is therefore **unevidenced**, not passing.
