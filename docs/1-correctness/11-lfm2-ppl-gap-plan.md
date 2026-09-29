# LFM2 PPL gap â resolution plan

**Logged:** 2026-09-27 (`docs/103-quickest-first-plan.md` item 11a; timeboxed out).  
**Entry in:** `docs/1-correctness/bugstofix.md`, item **11** (LFM2 `lfm2` perplexity gap).  
**Checkpoint:** `LFM2-1.2B-Q8_0.gguf`.  
**Numbers (bugstofix.md):** our PPL 10.9543 vs `llama-perplexity --chunks 1` 10.9277 (wikitext second-half, -c 2048).

This plan incorporates and critiques two ChatGPT-authored proposals (2026-09-29).

---

## What is already known

| Finding | Status |
|---|---|
| Layer bisection: layer 0 (short conv) matches to print resolution (1e-4) | Done |
| From layer 2 onward: real differences, 5e-4â1.6e-3 absolute, growing gradually; no single broken layer | Done |
| Q8_0 activation scheme (32-element blocks, fp16 block scale): tried matching ggml exactly â changed nothing | Done (ruled out) |
| `llama-eval-callback` 4-decimal print resolution â 1e-4 noise floor; all printed bisection at this level is unreliable | Established |

**Remaining suspects:**

1. **KV cache precision** â llama.cpp defaults to an F16 KV cache; this engine's CPU `ForwardPass` always uses F32.
2. **Float summation order in attention V-accumulation / softmax** over long context.
3. **A small distributed FP rounding difference** that cannot be eliminated without contorting the engine.

---

## Step 0 â Freeze the canonical benchmark (do this first)

There is a genuine documentation inconsistency in the repository:

- `ModelCompatibility.cs` line 289â290: **Stingray 10.9195, llama.cpp 10.9543** (Stingray BETTER â this was the admission measurement).
- `bugstofix.md` entry: **Stingray 10.9543, llama.cpp 10.9277** (Stingray WORSE â logged separately as item 11a).

These are not the same result with opposite interpretation; they are **different experiments**
(different wikitext ranges, possibly different context slices or chunking modes). Before investing
in any code change, establish one canonical row:

```
Checkpoint SHA256:
Stingray commit:
llama.cpp commit:
Dataset SHA256:
Context: -c 2048
Scored token range: (e.g. second 1024 tokens of the first chunk)
BOS: prepended per model metadata
Chunk mode: --chunks 1
Stingray PPL:
llama.cpp PPL:
```

> Do not investigate anything else until this baseline is reproducible on the current HEAD.
> Update `ModelCompatibility.cs` and `bugstofix.md` to use the same row.

---

## Step 1 â llama.cpp F32 KV experiment (zero code change, highest value)

Run the identical `llama-perplexity` command with `-ctk f32 -ctv f32` to force F32 KV storage
and compare:

```
llama.cpp F16 KV (default):  10.9277  (expected from canonical Step 0)
llama.cpp F32 KV:            ?
Stingray  F32 KV:            10.9543  (expected from canonical Step 0)
```

### Interpreting the outcome

Record the result, then reason carefully. Do NOT claim an outcome proves more than it does:

**If llama F32 KV â Stingray F32 KV â 10.954:**  
KV precision *accounts for approximately the full gap*. It is not proven to be the only cause
(other FP differences could cancel), but it becomes the dominant suspect. Proceed to Step 5 to
quantify Stingray's F16 KV round-trip, then decide whether to implement it.

**If llama F32 KV â 10.9277 (unchanged from F16):**  
KV precision is not a meaningful contributor. Skip Step 5. Proceed to Step 3.

**If llama F32 KV is between 10.928 and 10.954:**  
KV precision accounts for approximately `(10.9277 - llama_f32) / (10.9543 - 10.9277)` of the
observed gap. Both Step 3 and Step 5 are warranted to close the remainder.

> **Priority note:** Run this before touching any Stingray code. It costs ~15â30 minutes of
> compute and is the highest-signal experiment in the plan.

---

## Step 2 â PPL at multiple context sizes (almost free, very revealing)

This experiment was missing from the earlier version of the plan and should be done alongside
Step 1. Run PPL at several context sizes on both sides and record:

| Context | Stingray | llama.cpp (F16 KV) | llama.cpp (F32 KV) | Gap |
|---|---|---|---|---|
| 256  | ? | ? | ? | ? |
| 512  | ? | ? | ? | ? |
| 1024 | ? | ? | ? | ? |
| 1536 | ? | ? | ? | ? |
| 2048 | ? | ? | ? | ? |

### Why this matters more than it looks

If the gap is **context-dependent** (e.g., negligible at 256, growing to 0.24% at 2048), that
strongly implicates KV accumulation or attention summation across a long context window.

If the gap is **context-independent** (essentially the same 0.24% at 256 and 2048), KV cache
accumulation becomes much less compelling and the fault likely lies in something per-layer
rather than something that grows with position.

This is one of the cheapest experiments in the plan and can collapse the search space before
any binary dump infrastructure is built.

---

## Step 3 â Per-position NLL comparison (zero new code needed)

`PerplexityCommand.cs` already has `--dump-nll <path>` (line 81â83): writes one line per scored
position (`targetPos tokenId nll`) to a file.

```powershell
# Stingray side
stingray perplexity -m LFM2-1.2B-Q8_0.gguf -f scripts/kvarn-gate/wiki.test.raw -c 2048 --dump-nll stingray-nll.txt

# llama.cpp side â llama-perplexity itself doesn't have --dump-nll;
# use llama-server /v1/completions teacher-forced loop, or the scratch
# harness ZzMakeGlmIdsTmp pattern extended for LFM2.
```

From the dump, compute:
- Mean ÎNLL, median ÎNLL, max |ÎNLL|
- Top-20 positions by |ÎNLL|
- Cumulative PPL by position bucket: [1,256), [256,1024), [1024,+)

This answers the most important structural question:

> Is the 0.24% PPL gap caused by a **broad tiny degradation** over nearly every token
> (numerical accumulation hypothesis), or by **a small number of pathological positions**
> (discrete per-layer bug)?

These imply completely different investigations:
- Uniform small ÎNLL across all positions â numerical/FP accumulation (KV precision, FMA order)
- Sharp spikes at specific positions â inspect those tokens and their attention layers

Note also whether the high-|ÎNLL| positions cluster at high position indices. If the worst
tokens are consistently in the [1024,+) range, that is additional evidence for a long-context
numerical cause.

---

## Step 4 â Binary tensor dump (prerequisite for layer bisection)

The 4-decimal print resolution of `llama-eval-callback` makes printed bisection unreliable for
this model's drift magnitude. Extend `ZzLayerDumpTmp.cs` (or a sibling scratch harness) to
write raw float32 `.bin` files (header: `[int32 len][float32 x len]`) for comparison.

**Two contexts, not one:**

- **Context A (short):** existing 338-token wikitext prompt, last-token row. This is the
  existing diagnostic baseline.
- **Context B (long):** a deterministic context reaching ~1536â2048 tokens. The per-position
  NLL dump (Step 3) should guide which specific long-context position to capture. Pick the
  final scored position plus 2â3 positions with the largest |ÎNLL|.

> If Step 2 shows the gap is context-independent, Context A alone is sufficient and Context B
> can be skipped. If the gap is context-dependent, Context B is the more relevant diagnostic.

**Target stages per layer (last-token row of each context):**
- Short-conv layers: `sc_norm`, `sc_out`
- Attention layers: `attn_norm`, `attn_out` (post-softmax V-sum), `o_proj`
- Every layer: `post_attn_resid`, `post_ffn_resid`

**Comparison metrics â report all of these, never just a signed sum:**
- `max_abs_error`, `mean_abs_error`, `rms_error`
- `relative_L2`, `cosine_similarity`
- `max_relative_element` (for catching per-element outliers)

---

## Step 5 â Layer-by-layer bisection from binary dumps

With full-precision dumps:

1. Confirm `l0_sc_out` matches at near-floating-point precision â if so, the short-conv kernel
   is cleared.
2. Walk layers comparing `post_attn_resid` then `post_ffn_resid`.
3. **At each layer, record the full error profile.** Do NOT use `max_abs_error > 1e-4` as a
   causal threshold â the whole problem is that errors are around that scale. Instead, look
   for the **first layer where error materially increases relative to the previous stage**. A
   pattern like `layer 0: 1e-6, layer 1: 3e-6, layer 2: 8e-5, layer 3: 4e-4` points to layer 3;
   do not call layer 2 a fault boundary just because it crossed an arbitrary line.
4. Once a diverging layer is identified, drill into sub-stages:
   - `attn_norm` â if mismatch here: norm precision issue
   - Q/K/V projections â if mismatch: quantized projection kernel
   - `q_norm` / `k_norm` (QK RMSNorm before RoPE) â if mismatch: QK-norm implementation
   - `q_rope` / `k_rope` â if mismatch after QK-norm match: RoPE
   - Raw QK scores (per context position) â attention score accumulation
   - Softmax weights `P[j]` â softmax accumulation / exp precision
   - `attn_out` (pre-`o_proj`) â V-accumulation loop
   - `o_proj` (post-`wo`) â `wo` projection

---

## Step 6 â Simulated F16 KV experiment in Stingray (only if Step 1 shows KV precision matters)

The CPU `ForwardPass` always uses F32 KV storage. The BF16 KV store path
(`cache.IsBf16Store`, `cache.Bf16KeyAt`, etc.) exists only for GPU backends and is
**not available on the CPU path**. `STINGRAY_KV_STORE=bf16` does not affect CPU perplexity runs.

**Two distinct experiments â keep them explicit and separate:**

**Diagnostic round-trip (do this first):** when writing K/V into the F32 CPU cache, convert
to `ushort` (IEEE 754 half-precision) and immediately widen back. This gives the numerical
effect of F16 precision without changing the cache layout, pointer math, or any reader.
Cost: ~10â20 lines; protected by an env gate such as `STINGRAY_DEBUG_KV_F16_ROUNDTRIP=1`.
**This is a diagnostic only â it must not ship as a permanent feature.**

**Production F16 KV store (only if diagnostic confirms the gap):** implement a real F16 CPU
KV store following `PagedKvCache`'s existing BF16 pattern: `ushort*` pages, `WriteKvF16`,
`F16KeyAt`, `F16ValueAtHead`, F32-widen on read. This is a non-trivial change to a heavily-
calledpath; do not start it before the diagnostic round-trip confirms it is worth the cost.

Run the PPL table:
```
Stingray F32 KV (default):         10.9543
Stingray simulated F16 KV:         ?
Stingray simulated BF16 KV:        ?
llama.cpp F32 KV:                   ? (from Step 1)
llama.cpp F16 KV (default):        10.9277
```

Also test K and V independently: K precision affects QKâsoftmax nonlinearly; V precision
affects the weighted sum linearly â they may contribute differently.

---

## Step 7 â Attention arithmetic experiments (only if Step 5 points here)

Only pursue these if the binary dump (Steps 4â5) specifically identifies the attention score
computation, softmax, or V-accumulation as the diverging operation.

For each candidate, construct an isolated test using **real captured LFM2 Q/K/V vectors**
(not synthetic random data) and compute the output three ways:

**For QK dot-product:**
- A: current `SimdKernels.DotF32` (AVX2 tree)
- B: scalar loop (to match ggml's reduction order)
- C: double-precision oracle

**For softmax:**
- A: current `SimdKernels.SoftmaxInPlace`
- B: double-accumulation `sum(exp(x-max))`
- C: verify ggml uses the same `max-subtract + exp + sum + divide` order (it does â confirm)

**For weighted V sum:**
- A: current `Fma.MultiplyAdd` path
- B: scalar `out += weight * value`
- C: double accumulator oracle

### Interpreting the results correctly

Double precision is an **oracle for numerical sensitivity**, not a correctness oracle against
llama.cpp. The key comparison is Stingray arithmetic vs ggml arithmetic/order, not
Stingray vs mathematically ideal. If B (scalar) agrees with llama.cpp substantially better
than A (FMA/AVX2 tree), there is a real numerical-order difference worth quantifying.
**Do not change production code based on this alone** â first establish the magnitude and
determine whether it accounts for the PPL gap.

FMA vs scalar differences on a 1.2B model for a short context are typically well below 1e-5
per element. If they exist, they likely explain only a fraction of the gap. Step 1â2 (KV
precision and context-size scaling) are far more likely to be the dominant contributor.

---

## Step 8 â Fix, regression pass, and close

Once the root cause is confirmed:

**If the fix is CPU F16 KV cache:**
- Implement the production F16 KV store (see Step 6) after the diagnostic round-trip confirms it.
- Make it general (not LFM2-specific) â test on at least LFM2, a standard llama-family model,
  and one other admitted hybrid (e.g., Granite-H or Nemotron-H).
- Run `Lfm2ParityTests` and re-run wikitext PPL on the canonical Step 0 setup.

**If the fix is a distributed numerical difference:**
- If the scalar/double oracle closes the gap, align the V-accumulation loop order with ggml.
- If the gap cannot be eliminated without contorting the engine, document the finding:
  > LFM2 is numerically equivalent but not bit-identical; the residual PPL difference is
  > attributable to [F32 KV vs F16 KV / FMA accumulation order / ...]; accepted.
- Add a pinned regression test that asserts the *discovered invariant* (e.g., F32 KV PPL
  remains â¤ 10.96; F16 KV PPL â¤ 10.93) rather than merely today's number.

**If a discrete operation mismatch is found (Steps 4â5):**
- Fix the specific op, run parity tests, re-run PPL. This is the cleanest outcome.

---

## What NOT to do

- **Do not re-run the 4-decimal printed bisection** as primary evidence. Binary dumps only (Step 4).
- **Do not retry the Q8_0 activation scheme.** Tried and ruled out.
- **Do not reopen the short-conv kernel** without binary evidence of a discrepancy there.
- **Do not change the F32 KV default globally** without PPL regression across â¥ 3 other architectures.
- **Do not replace all softmax with double precision** â the cost is disproportionate for a 0.24% gap.
- **Do not chase differences at 1e-4** in printed callback output.
- **Do not use `STINGRAY_KV_STORE=bf16`** expecting it to affect the CPU path â it doesn't.
- **Do not conflate the two inconsistent PPL measurements** (Step 0 must resolve this first).
- **Do not use 1e-4 as a layer-bisection threshold.** Record all layers' error profiles and look for inflection, not threshold crossing.
- **Do not start implementing a production F16 KV store** before the diagnostic round-trip (Step 6) confirms it reduces the gap.
- **Do not accept `cosine â 0.9994` or `max_abs < 1e-3` as proof of correctness** â report all metrics.

---

## Execution order and effort estimates

| Step | Action | Effort | Key decision it enables |
|---|---|---|---|
| 0 | Freeze canonical benchmark | 30 min | All further numbers are unambiguous |
| 1 | llama.cpp `-ctk f32 -ctv f32` PPL run | 30 min | KV precision: major, minor, or negligible contributor |
| 2 | PPL at 256/512/1024/1536/2048 context sizes | 1â2 hr | Context-dependent vs context-independent gap |
| 3 | Per-position NLL dump + diff (`--dump-nll`) | 1 hr | Distributed drift vs discrete spikes; cluster analysis |
| 4 | Binary tensor dump at short AND long context | 2â3 hr | Removes 4-decimal floor from bisection |
| 5 | Layer bisection from binary dumps | 1â3 hr | First failing operation identified |
| 6 | Simulated F16 KV round-trip in Stingray CPU | 1â2 hr | Quantifies KV contribution in Stingray |
| 7 | FMA/scalar/double arithmetic experiments | 1â3 hr | Only if Step 5 points at attention math |
| 8 | Fix + regression + close | depends | â |

Stop at whichever step definitively answers the question. Steps 1 and 2 together may close
the investigation before any code change is needed.

---

## Files affected (contingent on root cause)

| File | Expected change |
|---|---|
| `src/OpenTail.Stingray.Engine/ForwardPass.Attention.cs` | Possible: KV cache precision default |
| `src/OpenTail.Stingray.Engine/PagedKvCache.cs` | Possible: CPU F16 KV store implementation |
| `src/OpenTail.Stingray.Core/ModelGraph.cs` | Possible: per-arch KV dtype flag |
| Scratch harness `ZzLayerDumpTmp.cs` | Binary dump extension at short and long context |
| `docs/1-correctness/bugstofix.md` | Update entry when closed |
| `docs/STATUS.md`, `ModelCompatibility.cs` | Update PPL numbers once canonical measurement taken |

---

## Success criteria

Success is defined primarily by understanding the cause, not by hitting a specific PPL number:

1. **Canonical benchmark reproducible** (Step 0 completed; both `ModelCompatibility.cs` and
   `bugstofix.md` use the same numbers from the same experiment).

2. **Root cause demonstrated** via one of:
   - Steps 1+6 show KV precision accounts for the gap â Stingray simulated F16 KV â llama.cpp F16 KV
   - Steps 4+5 isolate a specific op with measurable divergence matching the observed PPL delta
   - Steps 1+2+3 show the gap is context-independent and uniformly small â accepted as inherent FP difference

3. **Parity tests maintained:** `Lfm2ParityTests` (â¥ 14 teacher-forced positions) still pass.

4. **No PPL regression** on â¥ 2 other architectures if KV cache changes are made.

5. **Pinned invariant test** records whatever the investigation finds (e.g., F32 KV PPL
   â¤ 10.96; F16 KV PPL â¤ 10.93; or simply: greedy/teacher-forced parity holds), so that
   future changes cannot silently regress LFM2 PPL without a test failure.

> Note: the investigation could legitimately conclude that the discrepancy is inherent numerical
> difference and accept it. LFM2 is already admitted because greedy and teacher-forced parity
> pass. The important thing is establishing *why* the gap exists, not blindly closing it.
