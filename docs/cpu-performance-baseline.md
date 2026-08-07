# CPU performance baseline — all locally available models

**Measured:** 2026-08-07 on a quiet machine. Ryzen 7 5700G (Zen 3, 6c/12t, AVX2 only, no AVX-512,
no VNNI), CPU backend (`-g 0`), greedy, ~405-465 token prompt, 24 tokens generated. Models
interleaved round-robin so drift affects every model equally rather than clustering on whichever
ran last. Figures are best-of across rounds; contention only slows a run, so the maximum is the
least-contaminated estimator.

| Model | Prefill t/s | Decode t/s | prefill ÷ decode |
|---|---:|---:|---:|
| SmolLM2-1.7B Q4_K_M | 141.3 | 21.3 | 6.6x |
| Qwen3-0.6B Q8_0 | 93.0 | 47.4 | 2.0x |
| OLMoE-1B-7B Q4_K_M (MoE) | 105.6 | 28.2 | 3.7x |
| Qwen3-8B Q4_K_M | 36.7 | 6.3 | 5.8x |
| **gemma-4-12B q4_0** | **3.8** | **3.7** | **1.0x** |

Best-of **three** interleaved rounds. An earlier revision of this table published best-of-two while
claiming best-of-all-rounds, which understated three rows — OLMoE's prefill by 13% (93.4 vs 105.6).
That swing is itself the argument for the third round: two samples of a throughput figure are not
enough to bound it, even interleaved on an otherwise quiet machine.

## Reading it

**Every model batches its prefill except Gemma 4.** Prefill-to-decode ratios sit between 1.9x and
6.5x for the others; Gemma is 1.0x, meaning prefill runs at exactly decode speed. That is the
signature of no batching at all, confirmed in code at the `perLayerHdUnsupported` gate.

**Sizing it against the nearest size class** is more informative than the ratio alone. Gemma 4 12B versus Qwen3-8B (both unchanged by round 3):

- decode: 3.7 vs 6.3 t/s — **1.7x slower**, which parameter count alone explains.
- prefill: 3.8 vs 36.7 t/s — **9.7x slower**, which it does not.

So roughly a **~5.7x prefill penalty beyond what model size accounts for**. That is the cost of the
missing per-layer-head-dim batched prefill, expressed against a real comparator rather than against
itself.

**Qwen3-0.6B's low 1.9x ratio** is a small-model artifact, not a defect: at 0.6B the fixed
per-prefill overhead is a larger share of a short run, and decode is unusually fast because the
weights nearly fit in cache. It is the one row where the ratio should not be read as a batching
signal.

**OLMoE's decode was measured over only 7 tokens** — it emitted EOS early — so that figure carries
more per-run overhead than the others and should be treated as approximate. The prefill number is
sound.

## Caveats

- Single machine, AVX2 only, CPU only. Vulkan on this APU is slower than CPU (see
  `vulkan-backend-evidence.md`); CUDA is untested here.
- Prompt is ~400-465 tokens depending on tokenizer; per-model token counts differ slightly, which
  matters at the margin but not at the scale of the differences above.
- These are throughput figures and are the one class of measurement that a second process on the
  same box will corrupt. Re-measure on a quiet machine before treating any change as a regression.
