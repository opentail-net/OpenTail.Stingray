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


## KNOWN DEFECT 2026-08-07 — int8 prefill collapses on low-magnitude input

The corpus evidence above says int8 prefill does not degrade quality on wikitext. That remains
true, and it is not the whole picture. A sweep across input classes found a **user-reachable
collapse**:

| input class | cos @ n=8 | cos @ n=32 |
|---|---:|---:|
| prose | 0.9973 | 0.9697 |
| code | 0.9776 | 0.9953 |
| hex | 0.9980 | 0.9981 |
| base64 | 0.9968 | 0.9984 |
| repeated word | 0.9965 | 0.9972 |
| CJK | 0.9986 | 0.9971 |
| punctuation runs | 0.9972 | 0.9602 |
| **whitespace only** | **-0.1241** | 0.9776 |

A whitespace-only prompt of 8 tokens yields a final-logit cosine of **-0.124** against exact F32,
with a different argmax. The logits point in roughly the opposite direction. Deterministic and
exactly reproducible across runs.

**The shipped mitigation does not cover it.** `ForwardPass` skips int8 only when a prompt is
composed ENTIRELY of control/user-defined tokens. Whitespace tokens are ordinary vocabulary
entries. The mitigation keys on token **type**; the failure is driven by activation **magnitude** —
per-row int8 scaling degrades badly when a row's dynamic range collapses toward zero, and near-empty
input is exactly that. The earlier all-control-token case was very likely the same underlying
failure observed through a different door.

**Reachability is real but narrow:** a blank-ish prompt, a document with a long whitespace run, an
empty template slot. Ordinary text is unaffected. Note also that several ordinary classes sit below
0.99 (prose at n=32: 0.9697; punctuation at n=32: 0.9602), which matters when calibrating any
tolerance against this path — the earlier "int8 is 0.35% better on corpus perplexity" result is an
average over a corpus and does not bound per-prompt divergence.

Pinned by `Q8PrefillLowMagnitudeInputTests`, deliberately **skipped rather than asserted green**: a
test written to pass against -0.124 would bless the defect. The suggested fix is to gate int8 on
activation dynamic range rather than token type, so the guard tracks the property that actually
causes the failure. Un-skip when that lands.


## Attempted fix 2026-08-07 — "gate int8 on activation dynamic range" does NOT work as stated

The recommended fix was to gate int8 on activation dynamic range instead of token type. Calibrating
that gate before implementing it shows **the obvious form of it cannot work**, and the measurement
is recorded so nobody implements it on the strength of the reasoning alone.

Hypothesis: per-row symmetric int8 uses `scale = max|x| / 127`, so a row whose energy sits in small
components while one outlier sets the scale loses those components. The discriminator would then be
the outlier ratio `max|x| / rms(x)`, measured on the prompt's embeddings — a pure function of the
tokens, which is required here (the three existing gate call sites are all prompt-level, and a
statistic computed over a batch would reintroduce the arrival-dependence and chunk-dependence that
`ContinuousBatchingTests.PrefillWithCache_Chunked_MatchesFull` exists to prevent).

Measured embedding outlier ratio against the observed int8 damage:

| class | worst ratio | mean ratio | int8 cosine |
|---|---:|---:|---:|
| whitespace | 10.78 | 10.78 | **-0.124** |
| prose | 10.78 | 5.94 | 0.997 |
| code | **12.58** | 6.11 | 0.978 |
| repeat | 10.78 | 8.62 | 0.997 |
| cjk | 4.84 | 3.79 | 0.999 |

**It does not separate.** `code` has a higher worst-case ratio than whitespace and is healthy;
whitespace shares its worst-case ratio with prose and repetition. Any threshold that catches
whitespace also catches ordinary prose and code. A gate built on this statistic would either be
inert or would push common inputs onto the slow exact path while still not being principled.

Why the proxy fails: the embeddings are not where the pathology lives. Whitespace embeddings are
unremarkable; the activations only become hostile to int8 after passing through norms and
projections. The statistic has to be measured **where the quantisation happens**, not at the input.

Note the Q8 path already quantises each token's activations independently ("quantize every token's
activations once up front"), so this is not cross-row contamination — a specific row genuinely
quantises badly, several layers in.

### What a real fix requires

1. Instrument the actual quantisation point and find which layer's activations first go pathological
   for whitespace input, and by what statistic.
2. Gate **per row**, not per batch or per prompt — that keeps the decision a function of the token
   alone and preserves both the chunked/unchunked equivalence and the "a token's logits must not
   depend on its neighbours" contract.
3. Only then choose a threshold, from the separating statistic rather than a plausible one.

Until that exists, `STINGRAY_CPU_PREFILL_Q8=0` is the mitigation, and the defect stays pinned by the
skipped `Q8PrefillLowMagnitudeInputTests`.


## Second attempt 2026-08-07 — measured AT the quantisation point; still does not separate

The previous entry concluded the embedding proxy was measured in the wrong place and that the
statistic had to be taken where the quantisation happens. That was done: a temporary hook recorded
`amax/rms` for every activation row at the moment of int8 conversion, across a full prefill
(192 quantisation calls per prompt).

| class | median | p90 | max |
|---|---:|---:|---:|
| prose (healthy, cos 0.997) | 25.44 | 43.15 | **90.43** |
| whitespace (collapses, cos -0.124) | 41.59 | 86.23 | **89.76** |

**Partial separation, insufficient for a gate.** Whitespace is systematically more
outlier-dominated — median 41.6 against 25.4 — but the two classes reach the *same maximum*.
Healthy prose contains rows at ratio 90.4 that quantise fine, so a per-row threshold anywhere below
that fires on ordinary text, and anywhere above it never fires at all.

This rules out the whole family of per-row activation-dynamic-range gates, which was the requested
fix. The requested fix cannot be built from this statistic, at either measurement point.

What survives is weaker but real: the **distribution** differs (median, p90), not the extreme. A
prompt-level gate on median outlier ratio might work, but it would need calibrating against the
mildly-degraded classes too (code at 0.978, punctuation at 0.960), and only two classes have been
measured this way. Building a numerics gate on a two-point separation would repeat the error this
document already records twice.

### Where the investigation actually stands

The per-row outlier hypothesis is **disproved**, not merely unconfirmed. The failure is not
explained by any single row quantising badly, because prose rows with identical outlier ratios are
fine. Plausible remaining directions, none yet tested:

- The damage may be **accumulative across layers** rather than located in one bad row, in which case
  no input-side statistic predicts it and the gate must be based on an output check.
- It may involve the **interaction between quantised activations and specific weight rows**, which
  no activation-only statistic can see.
- It may be specific to prompts whose rows are near-**identical** (whitespace repeats one token),
  making the batch degenerate in a way per-row statistics do not capture.

Note the third is testable cheaply and was not tested: `repeat1` ("the the the...") is nearly
identical-token and measured healthy at 0.997, which argues against it, but it is not the same token
repeated exactly.

The temporary probe was removed from `SimdKernels` — it was diagnostic scaffolding in a hot
production path and had no business surviving the measurement.


## FIXED 2026-08-07 — the cause was exact token repetition

The third untested direction was right, and the first two were wrong. Prefilling **eight copies of a
single token** collapses int8 regardless of which token:

| 8-token prompt | cosine before fix |
|---|---:|
| `space` x8 | -0.124 |
| `newline` x8 | 0.101 |
| `tab` x8 | 0.030 |
| `,` x8 | 0.031 |
| `9` x8 | -0.013 |
| **`the` x8** | **0.470** |
| **`scheduler` x8** | **0.324** |
| *mixed whitespace, 3 distinct ids* | 0.9945 |
| *distinct prose tokens* | 0.9973 |

It was never about whitespace, magnitude, or control tokens. Ordinary words collapse just as hard.
The whitespace reproduction was misleading because whitespace happens to tokenise into long runs of
one repeated id.

The boundary is binary:

| prompt | cosine |
|---|---:|
| 1 distinct token (lengths 2, 4, 8, 16, 32, 64) | 0.40 - 0.48 |
| 2 distinct | 0.9959 |
| 3 distinct | 0.9980 |
| 7x`the` + **1** other | 0.9953 |
| 15x`the` + **1** other | 0.9946 |

**One differing token restores it completely, at any length.** That sharpness is why the fix is a
distinct-count test and not a numerical threshold.

### Why identical tokens break int8

With one repeated token, the rows entering each matmul differ only by positional effects: the signal
lives entirely in small differences riding on a large common component. Per-row int8 scales to the
common component and quantises those differences away. Ordinary prompts carry their signal in the
differences *between* tokens, which are large enough to survive.

This also explains the original all-control-token report — a short prompt of repeated structural
tokens is the same degeneracy seen through a different door.

### The fix

`ForwardPass.IsSingleDistinctTokenPrompt` routes such prompts to the exact F32 path, applied at the
same three sites as the existing all-control guard (`Prefill`, `PrefillWithCache`, and the packed
multi-prompt admission path). It is a function of the token ids alone — deliberately, since a
statistic over activations or over batch-mates would make a token's numerics depend on its
neighbours, which is what the packed-admission and chunked-prefill tests exist to prevent.

Cost is an O(n) scan with early exit on the first differing token, so ordinary prompts pay one
comparison.

After the fix every previously collapsing prompt returns **cosine 1.000000** — exactly, because it
now takes the same F32 route as the reference — and healthy prompts are unchanged (0.995-0.998).

### What the two disproved hypotheses cost, and were worth

Both earlier attempts (embedding outlier ratio; activation outlier ratio at the quantisation point)
were measured and disproved rather than reasoned about. Had either been implemented on the strength
of its plausible mechanism, it would have shipped a gate that fires on healthy prose and code while
still missing `the` x8. The disproofs are retained above precisely because the reasoning behind them
is attractive enough to be tried again.

Regression coverage: `Q8PrefillLowMagnitudeInputTests`, now un-skipped, asserting both the
whitespace reproduction and a repeated ordinary word — a guard that only special-cased whitespace
would have left the real defect in place.
