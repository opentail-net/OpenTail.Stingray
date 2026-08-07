# CPU speculative decoding — measured findings

**Machine:** Ryzen 7 5700G (Zen 3, 6c/12t, AVX2/FMA only, no VNNI/AVX-512), 36.8 GB/s measured
DRAM ceiling. **Target:** `Qwen3-8B-Q4_K_M` (4.68 GiB). **Draft:** `Qwen3-0.6B-Q8_0`.
**Prompt:** realistic prose, 85 tokens. Greedy (`--temp 0`), `-g 0`, `--no-thinking`, `-n 128`.

Running log — append new iterations, do not rewrite old ones.

## Why this investigation exists

Dense decode streams every weight once per emitted token, so it is DRAM-bound. Measured:

| Model | decode t/s | GB/s | % of 36.8 ceiling |
|---|---|---|---|
| SmolLM2-1.7B Q4_K_M | 26.4 | 27.8 | 76% |
| Qwen3-8B Q4_K_M | 6.8 | 34.2 | **93%** |

At 93% of the memory ceiling there is essentially nothing left for kernel micro-optimisation on
the 8B. The only lever that beats a bandwidth wall is emitting more tokens per weight read, which
is exactly what speculative decoding promises: verify k drafted tokens in one pass over the weights.

## Iteration 1 — baseline vs draft-model speculation

| Configuration | decode t/s | notes |
|---|---:|---|
| No speculation | **6.8** | baseline |
| `-md Qwen3-0.6B-Q8_0` (draft-n 4, default) | **4.3** | acceptance 62%; draft 2695ms / verify 26520ms / commit 363ms |
| same, `STINGRAY_SPEC_BATCH_VERIFY=0` | 4.2 | verify 27715ms |
| same, `STINGRAY_BATCHED_MATVEC_TIER=1` | 4.3 | verify 26342ms (flag was already on — see below) |

**Speculative decoding is currently a 37% LOSS on CPU, not a win**, despite a healthy 62%
acceptance rate. The draft model is not the problem: drafting costs 2.7 s of a ~30 s run.

**The verify pass does not amortise weight reads.** At 62% acceptance with draft-n 4 the run makes
roughly 37 verify steps, so ~717 ms per verify against 147 ms for a single-token decode — about
4.9×, i.e. indistinguishable from k sequential decodes. Speculation therefore pays the draft cost
and buys nothing.

Confirmed by direct A/B: forcing the sequential fallback with `STINGRAY_SPEC_BATCH_VERIFY=0`
changes nothing (4.2 vs 4.3 t/s). The batched path is being taken — `ForwardPass.SupportsBatchVerify`
is true for this model and `BatchVerify` builds a genuine N-token batch calling
`SimdKernels.MatMulBatched` once per projection — it simply is not faster than looping.

### Where the amortisation is lost

`MatMulBatched` (SimdKernels.cs ~line 98) routes small batches away from BLAS:

```csharp
if (batchSize < MinBatchForBlas || !BlasInterop.IsAvailable)
```

Both terms hold here. OpenBLAS is not installed on this machine (`OpenBLAS: not found` on every
run; no `tools/openblas`), and `MinBatchForBlas` defaults to **16** while speculative verify uses
N = draft-n + 1 = **5**. The threshold is not wrong: the BLAS route dequantises Q4_K to an F32 temp
buffer, roughly 4× the bytes, which only pays above ~16 tokens. So BLAS is the wrong tool for a
5-token verify regardless of whether it is installed.

The in-register alternative exists and is enabled: the fp32 multi-input tier
(`MatVec4In`/`MatVec2In`, "one weight row read, dotted against 4 activation vectors"), bit-identical
per output slot and pinned by `SimdKernelsQ8KSTests.MatVec4In_BitwiseMatchesSingleMatVec`. Setting
`STINGRAY_BATCHED_MATVEC_TIER=1` changed nothing because it is **already on by default**
(`!= "0"`).

Its own documentation says why that is not enough (SimdKernels.cs ~line 258): 4-way concurrent
decode moved 28.90 → 30.28 tok/s, and the note states the tier is *"deliberately NOT described as
amortising weight reads N×, because it demonstrably does not"*. The present measurement is an
independent confirmation of that on a different workload.

**So the open question is not "should we batch the verify" — it is why a kernel that reads each
weight row once and dots it against 4 activations does not convert into ~4× less DRAM traffic.**
That is the thread to pull next.

### Incidental defect found

SimdKernels.cs line ~142 says *"Default OFF until measured end to end"* for
`BatchedMatVecTierEnabled`, but the property 120 lines below defaults it **ON** (`!= "0"`), and its
own summary says "Default ON as of the measurement in the session runtime plan §3.4.9". The inline
comment is stale and actively misleading — it is what sent this iteration down a redundant A/B.

## Method notes

- Interleave arms; discard any sweep taken while the machine is otherwise in use.
- Prompt choice matters and is a trap: highly repetitive text massively flatters prompt-lookup
  drafting. Acceptance rates from `"the quick brown fox…"` repeated N times are not transferable.
  Prose and code prompts are kept alongside the repetitive one for exactly this reason.
- Decode t/s here comes from the CLI's own `Decode:` line, which times the decode loop only.

## Open threads

1. Why `MatVec4In` does not yield ~4× less weight traffic. Is it reached for Q4_K at N=5, and does
   the FFN down-projection take the same route? A counter exists (`BatchedMatVecTierCalls`).
2. `--draft-lookup` (no draft model) on prose vs code vs repetitive text — acceptance and t/s.
3. Draft length sweep (`--draft-n` 2/4/8) — the optimum shifts with acceptance rate.
4. Whether any CPU configuration makes speculation profitable, or whether the honest answer is
   "CPU speculative decoding is off until the verify amortises", which would be worth surfacing
   as a CLI warning rather than letting users discover a 37% slowdown.

## Iteration 2 — the tier IS reached; the loss is not a dispatch bug

Reference material added under `examples/` (gitignored, third-party): `flame/how-to-optimize-gemm`,
`flame/blislab`, `google/XNNPACK`, alongside the BLIS paper in `docs2/blis.txt`.

**Measured, not inferred: `BatchVerify` reaches the multi-input tier on every matmul.** A probe on
Qwen3-0.6B (28 layers) read `SimdKernels.BatchedMatVecTierCalls` around one `BatchVerify` of 5
tokens: **196 calls = 7 matmuls x 28 layers**, exactly the full set. `BatchedMatVecTierEnabled` is
True. And `MatVec4In` for Q4_K genuinely calls `DotQ4K_4In(row, i0, i1, i2, i3, ...)` — one row
read, four dots, four outputs. So the plumbing is correct end to end.

**The gap is therefore in the kernel's throughput, not in dispatch.** Bounding it from measured
numbers: a single-token decode streams 4.68 GiB in 147 ms (31.8 GiB/s, 93% of ceiling). A 4-token
group that truly amortised would cost one weight pass, ~143 ms, with a compute floor of ~110 ms at
Zen 3 peak. Measured verify is ~725 ms — **5x above the floor**, and compute at peak cannot explain
it.

### A discarded measurement, recorded so it is not repeated

An attempt to time `MatVec` vs `MatVec4In` directly reported ratio 3.25x (throughput gain only
1.23x), which would have been a clean answer. **It is invalid and was thrown away.** Three defects,
any one fatal:

1. It reported 2.0 GiB/s for `MatVec` where production decode achieves 31.8 GiB/s — a 16x
   discrepancy that alone disqualifies it.
2. The matrix was 8192x4096 Q4_K = ~18 MB against a 16 MB L3, so it measured a largely
   cache-resident kernel, not DRAM streaming — the opposite of the regime under investigation.
3. Weights were filled with random bytes. Q4_K's FP16 block scales then decode to arbitrary values
   including denormals and NaNs, which can collapse FP throughput. Synthetic quantised weights must
   come from a real tensor, or be constructed with valid scales.

The lesson generalises: a kernel microbenchmark for a memory-bound path must be validated against
the production throughput it claims to model **before** its ratio is believed.

### Next

Redo the kernel comparison with a real Q4_K tensor lifted from a GGUF, sized well beyond 16 MB L3,
and sanity-check that the single-input arm reproduces ~31 GiB/s before trusting any ratio from it.
Only then is the "is `DotQ4K_4In` compute-bound?" question answerable.

## Iteration 3 — the Q4_K dot is COMPUTE-bound, which explains everything

Corrected benchmark: 40 real Q4_K tensors lifted from `Qwen3-8B-Q4_K_M`, 1.35 GiB streamed per
arm (well past the 16 MB L3), pages faulted in by a warm pass before timing, weights genuinely
quantised rather than random bytes.

| Arm | time | rate |
|---|---:|---:|
| `MatVec` — 1 output | 57 ms | 23.9 GiB/s |
| `MatVec4In` — 4 outputs | 214 ms | 6.3 GiB/s |

**Ratio 3.78x for 4 outputs → effective throughput gain 1.06x. The fused 4-input kernel does not
amortise.**

Fitting `T(n) = M + n·C` to the two points gives **M = 4.7 ms, C = 52.3 ms**: the kernel is
**~92% compute, ~8% memory**. Amortising the weight read cannot save more than that 8% no matter
how it is implemented — and the measured 1.06x is exactly that ceiling.

### This inverts the premise of the whole investigation

Iteration 1 reasoned "decode runs at 93% of the DRAM ceiling, therefore it is bandwidth-bound,
therefore the lever is emitting more tokens per weight read". The first half is a measurement; the
second half was an **inference**, and it is wrong. Decode reaching a rate close to the memory
ceiling does not establish that memory is the constraint — here the dequantise-and-dot arithmetic
is, at roughly 92% of the time, and the byte rate is merely what falls out of it.

This is why every attempt to make the verify cheaper failed identically: `BatchVerify`, the
multi-input tier, and `MatVec4In` all attack the 8%.

### Consequences

1. **CPU speculative decoding cannot pay on this hardware.** Verifying k tokens costs ~k× the
   compute regardless of how the weight read is scheduled, so it can only lose by the drafting
   cost. The measured 4.3 vs 6.8 t/s is the expected result, not a bug to fix.
2. **The lever is dot throughput, not data movement.** This CPU (Zen 3) has no VNNI, so Q4_K dots
   run the AVX2 `vpmaddubsw` → `vpmaddwd` → `vpaddd` chain. The VNNI widening committed in
   `f0be474` replaces that with a single `vpdpbusd` per step — 1 uop on P01 versus 2 uops pinned to
   P0 plus a second instruction. On Zen 4+ that should cut C substantially, and **only once C falls
   does batching (and therefore speculation) start to pay**. That path remains unmeasured for want
   of hardware, and this result raises its value considerably.
3. Prompt-lookup drafting and draft-length sweeps are not worth running on this machine. They vary
   acceptance rate, and no acceptance rate rescues a verify whose cost is linear in k.

### Caveat on the harness

The 1-input arm measured 23.9 GiB/s against production decode's 31.8 GiB/s — same order, but 25%
apart, so this harness does not reproduce production exactly (it streams the 40 largest tensors
rather than a real layer sequence). The **ratio** is the load-bearing number and both arms share
the harness, so it is far more robust than either absolute figure. Anyone re-running this should
still treat 3.78x as the finding and 23.9 GiB/s as a validity check that passed only loosely.

## Iteration 4 — the compute-bound finding survives a proper test (and nearly didn't)

Iteration 3's split came from fitting `T(n) = M + n·C` to **two** points with **two** parameters —
exactly determined, zero residual by construction, no validation whatsoever. `MatVec2In` supplies
an independent third point the model never saw. Prediction before measuring: T(2) = 109 ms.

The first re-run appeared to **refute** it: T(1)=120, T(2)=127, T(4)=242 ms, implying gain4 = 1.98x
rather than 1.06x — real amortisation. Five samples show why that was wrong:

| run | T(1) | T(2) | T(4) | GiB/s at n=1 |
|---|---:|---:|---:|---:|
| A | 57 | — | 214 | 23.9 |
| B | 120 | 127 | 242 | **11.3** |
| C | 85 | 98 | 187 | 15.9 |
| D | 52 | 101 | 211 | 26.1 |
| E | 53 | 104 | 202 | 25.6 |

`T(1)` varies by 2.3x across identical runs. Because `gain = n / (T(n)/T(1))`, a contended — and
therefore inflated — `T(1)` manufactures apparent amortisation. Run B's 11.3 GiB/s against
production's 31.8 marks it as contended, not as evidence. Contention only ever slows a run, so
best-of is the sound estimator.

**Best-of-5: T(1)=52, T(2)=98, T(4)=187 ms.** Fitting on T(1) and T(4) alone gives
**M = 7.0 ms (13% memory), C = 45.0 ms (87% compute)**, which then **predicts T(2) = 97 ms against
an actual 98 ms — 1.0% error**. That is genuine out-of-sample validation.

**gain2 = 1.06x, gain4 = 1.11x, amortisation ceiling 13%.** Iteration 3's conclusion stands, now on
evidence that can actually carry it.

Two method points worth keeping:

1. A single re-run that contradicts a finding is not a refutation any more than a single run that
   confirms it is proof. The refuting run here was the contended one, and taking it at face value
   would have reversed a correct conclusion.
2. State the prediction before measuring. "T(2) = 109 ms" was written down first, which is what made
   the actual 98 ms interpretable rather than something to rationalise after the fact.
