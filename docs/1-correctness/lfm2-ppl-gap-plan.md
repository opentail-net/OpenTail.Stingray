# LFM2 PPL gap â resolution plan

**Logged:** 2026-09-27 (`docs/103-quickest-first-plan.md` item 11a; timeboxed out).  
**Entry in:** `docs/1-correctness/bugstofix.md`, item "LFM2 (`lfm2`) perplexity 0.24% worse than llama.cpp".  
**Checkpoint:** `LFM2-1.2B-Q8_0.gguf`.  
**Numbers (bugstofix.md):** our PPL 10.9543 vs `llama-perplexity --chunks 1` 10.9277 (wikitext second-half, -c 2048).

This plan incorporates a ChatGPT-authored proposal (2026-09-29) and critiques / integrates its
best ideas against the actual codebase.

---

## What is already known

| Finding | Status |
|---|---|
| Layer bisection: layer 0 (short conv) matches to print resolution (1e-4) | Done |
| From layer 2 onward: real differences, 5e-4â1.6e-3 absolute, growing gradually; no single broken layer | Done |
| Q8_0 activation scheme (32-element blocks, fp16 block scale): tried matching ggml exactly â changed nothing | Done (ruled out) |
| `llama-eval-callback` 4-decimal print resolution â 1e-4 noise floor; all printed bisection at this level is unreliable | Established |

**Remaining suspects:**

1. **KV cache precision** â llama.cpp defaults to an F16 KV cache; this engine's CPU `ForwardPass` always uses F32.
2. **Float summation order in the attention V-accumulation / softmax** over long context.
3. **A small distributed FP rounding difference** that cannot be eliminated without contorting the engine.

---

## Step 0 â Freeze the canonical benchmark (do this first)

There is a genuine documentation inconsistency in the repository:

- `ModelCompatibility.cs` line 289â290: **Stingray 10.9195, llama.cpp 10.9543** (Stingray BETTER â this was the admission measurement).
- `bugstofix.md` entry: **Stingray 10.9543, llama.cpp 10.9277** (Stingray WORSE â logged separately as item 11a).

These are different experiments (different wikitext ranges, possibly different context slices or
chunking modes). Before investing in any code change, establish one canonical row:

```
Checkpoint SHA256:
Stingray commit:
llama.cpp commit:
Dataset SHA256:
Context: -c 2048
Scored token range: (e.g. [1024, +))
BOS: prepended per model metadata
Chunk mode: --chunks 1
Stingray PPL:
llama.cpp PPL:
```

This is cheap (one perplexity run each side) and ensures no further investigation chases a
number that no longer reflects the current code.

> Do not investigate anything else until this baseline is reproducible on the current HEAD.

---

## Step 1 â llama.cpp F32 KV experiment (zero code change, highest value)

This is the highest-value single experiment and requires no Stingray code changes at all.

Run the identical `llama-perplexity` command with `-ctk f32 -ctv f32` to force F32 KV storage
in llama.cpp and compare:

```
llama.cpp default (F16 KV):  10.9277  (expected)
llama.cpp F32 KV:            ?
Stingray F32 KV:             10.9543  (expected)
```

### Three possible outcomes

**Outcome A â gap is entirely KV precision:**
```
llama F16/F16 = 10.9277
llama F32/F32 â 10.9543
Stingray F32  â 10.9543
```
The investigation is essentially complete. The gap is KV representation,
not a Stingray bug. Fix: implement a CPU F16 KV mode (see Step 5);
document the F32 default as intentionally higher-precision.

**Outcome B â KV precision ruled out:**
```
llama F16/F16 = 10.9277
llama F32/F32 â 10.9277
Stingray F32  = 10.9543
```
Proceed to Step 3 (binary tensor dump).

**Outcome C â partial contributor:**
```
llama F16/F16 = 10.9277
llama F32/F32 = 10.93x
Stingray F32  = 10.9543
```
KV precision accounts for part of the gap. Still pursue Step 3 for the remainder.

> **Priority note:** Run this before touching any Stingray code. It costs ~15â30 minutes of
> compute and answers the most important question for free.

---

## Step 2 â Per-token NLL comparison (zero new code needed)

`PerplexityCommand.cs` already has `--dump-nll <path>` (line 81â83): writes one line per scored
position (`targetPos tokenId nll`) to a file. This was added during the Youtu-VL investigation
and is wired and working.

Use it now:

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

### Why this matters before diving into kernels

A 0.24% PPL gap could be:
- Tiny error distributed over nearly every token â numerical accumulation hypothesis
- 20â50 tokens with substantially worse likelihood â a discrete per-layer bug

These imply completely different investigations. If the delta is smooth and uniformly
distributed across positions, distributed floating-point rounding becomes much more
plausible. If there are sharp spikes at specific positions, inspect those tokens and layers.

---

## Step 3 â Binary tensor dump (prerequisite for layer bisection)

The 4-decimal print resolution of `llama-eval-callback` is the main bisection blocker.
Extend `ZzLayerDumpTmp.cs` (or a sibling scratch harness) to write raw float32 `.bin` files
(header: `[int32 len][float32 x len]`) for the same 338-token LFM2 wikitext prompt.

**Target stages per layer (last-token row):**
- Short-conv layers: `sc_norm`, `sc_out`
- Attention layers: `attn_norm`, `attn_out` (post-softmax V-sum), `o_proj`
- Every layer: `post_attn_resid`, `post_ffn_resid`

**Comparison metrics** â report all of these, never just a signed sum:
- `max_abs_error`, `mean_abs_error`, `rms_error`
- `relative_L2`, `cosine_similarity`
- `max_relative_element` (for catching per-element outliers)

**First target layer: the first attention layer** (layer 1 or 2 depending on the architecture's
short-conv / attention interleaving). The short-conv evidence is already favourable; start at
the first attention block.

> Do not continue to Step 4 until binary dumps are available. Printed comparisons at 1e-4
> resolution are ambiguous for exactly this model's drift magnitude.

---

## Step 4 â Layer-by-layer bisection from binary dumps

With full-precision dumps:

1. Confirm `l0_sc_out` matches exactly â if so, the short-conv kernel is cleared.
2. Walk layers: compare `post_attn_resid` then `post_ffn_resid` for each layer.
3. At the first layer where `max_abs_error > 1e-4` and stable, drill into sub-stages:
   - `attn_norm` â if mismatch here: norm precision issue
   - Q/K/V projections (Qraw, Kraw, V) â if mismatch: quantized projection kernel
   - `q_norm` / `k_norm` (QK RMSNorm before RoPE) â if mismatch: QK-norm implementation
   - `q_rope` / `k_rope` â if mismatch after QK-norm match: RoPE
   - Raw QK scores (per context position) â if mismatch after RoPE match: attention score accumulation
   - Softmax weights `P[j]` â if mismatch after QK match: softmax accumulation / exp precision
   - `attn_out` (pre-`o_proj`) â if mismatch after softmax match: V-accumulation loop
   - `o_proj` (post-`wo`) â if mismatch after `attn_out` match: `wo` projection

This decision tree gives an unambiguous path to the first failing operation.

---

## Step 5 â Simulated F16 KV experiment in Stingray (only if Step 1 shows KV precision matters)

The CPU `ForwardPass` always uses F32 KV storage. The BF16 KV store path
(`cache.IsBf16Store`, `cache.Bf16KeyAt`, etc.) exists only for the GPU
(`GpuForwardPass`/Vulkan/CUDA) backends and is NOT available on the CPU path.
Do NOT try to use `STINGRAY_KV_STORE=bf16` for this experiment â it does not apply to the
CPU forward pass.

Instead, add an experimental round-trip mode: when writing K/V into the F32 CPU cache,
convert to `ushort` (IEEE 754 half-precision) and immediately widen back. This gives
the numerical effect of F16 precision without changing the cache layout, pointer math,
or any reader.

Cost: ~10â20 lines in the KV write path; protected by an env gate such as
`STINGRAY_DEBUG_KV_F16_ROUNDTRIP=1`. This should NOT be a permanent feature â it is a
diagnostic only to isolate the contribution of KV precision.

Run the PPL table:
```
Stingray F32 KV (default):      10.9543 (expected)
Stingray simulated F16 KV:      ?
Stingray simulated BF16 KV:     ?
llama.cpp F32 KV:               ? (from Step 1)
llama.cpp F16 KV:               10.9277 (reference)
```

Also test K and V independently (K precision affects QKâsoftmax nonlinearly;
V precision affects weighted sum linearly â they may contribute differently).

---

## Step 6 â Attention arithmetic experiments (only if Step 4 points here)

Only pursue these if the binary dump (Step 3/4) specifically identifies the attention score
computation, softmax, or V-accumulation as the diverging operation.

For each candidate, construct an isolated test using captured real LFM2 Q/K/V vectors
(not synthetic random data) and compute the output three ways:

**For QK dot-product:**
- A: current `SimdKernels.DotF32` (AVX2 tree)
- B: scalar loop (to match ggml's reduction order exactly)
- C: double-precision oracle

**For softmax:**
- A: current `SimdKernels.SoftmaxInPlace`
- B: double-accumulation `sum(exp(x-max))`
- C: verify ggml uses the same `max-subtract + exp + sum + divide` order (it does â confirm)

**For weighted V sum:**
- A: current `Fma.MultiplyAdd` path
- B: scalar `out += weight * value`
- C: double accumulator oracle

If B (scalar) agrees with llama.cpp substantially better than A (FMA/AVX2 tree),
there is a real numerical-order difference. **Do not change production code** based on
this alone â first establish the magnitude and decide if it accounts for the PPL gap.

> **Priority note:** FMA vs scalar differences on a 1.2B model's attention for a 338-token
> context are typically well below 1e-5 per element. If they exist, they likely explain
> only a fraction of the 0.24% PPL gap. Step 1 (KV precision) is far more likely to be
> the dominant contributor.

---

## Step 7 â Fix, regression pass, and close

Once the root cause is confirmed from Steps 1â4:

**If the fix is CPU F16 KV cache (confirmed by Steps 1 and 5):**
- Implement a real F16 CPU KV store following `PagedKvCache`'s existing BF16 pattern:
  `ushort*` pages, `WriteKvF16`, `F16KeyAt`, `F16ValueAtHead`, F32-widen on read.
- Make it general (not LFM2-specific) â test on at least LFM2, a standard llama-family
  model, and one other admitted hybrid (Granite-H or Nemotron-H).
- Run `Lfm2ParityTests` and re-run wikitext PPL. Target: â¤ 10.93 (â¤ 0.05% above llama.cpp).
- Run the full ForwardPass suite.

**If the fix is a distributed numerical difference (confirmed by Step 6):**
- If the scalar/double oracle closes the gap, align the V-accumulation loop order with ggml.
- If the gap cannot be eliminated without contorting the engine, document the finding:
  > LFM2 is numerically equivalent but not bit-identical; the residual 0.24% PPL difference
  > is attributable to [F32 KV vs F16 KV / FMA accumulation order / ...]; accepted.
- Add a pinned regression test that asserts the *discovered invariant* (e.g., F32 KV PPL
  remains â¤ 10.96; F16 KV PPL â¤ 10.93) rather than just today's number.

**If a discrete operation mismatch is found (confirmed by Step 4):**
- Fix the specific op, run parity tests, re-run PPL. This is the cleanest outcome.

---

## What NOT to do

- **Do not re-run the 4-decimal printed bisection** as primary evidence. Binary dumps only (Step 3).
- **Do not retry the Q8_0 activation scheme.** Tried and ruled out.
- **Do not reopen the short-conv kernel** without binary evidence of a discrepancy there.
- **Do not change the F32 KV default globally** without PPL regression across â¥ 3 other architectures.
- **Do not replace all softmax with double precision** â the cost is disproportionate for a 0.24% gap.
- **Do not chase differences at 1e-4** in printed callback output.
- **Do not use `STINGRAY_KV_STORE=bf16`** expecting it to affect the CPU path â it doesn't.
- **Do not conflate the two inconsistent PPL measurements** (Step 0 must resolve this first).

---

## Execution order and effort estimates

| Step | Action | Effort | Cost if skipped |
|---|---|---|---|
| 0 | Freeze canonical benchmark | 30 min | All further numbers are ambiguous |
| 1 | llama.cpp `-ctk f32 -ctv f32` PPL run | 30 min | Miss the cheapest root-cause signal |
| 2 | Per-token NLL dump + diff (existing `--dump-nll`) | 1 hr | Don't know if error is distributed or spiked |
| 3 | Binary tensor dump infrastructure | 2â3 hr | Bisection stuck at 4-decimal floor |
| 4 | Layer bisection from binary dumps | 1â3 hr | Don't know which op diverges |
| 5 | Simulated F16 KV in Stingray CPU path | 1â2 hr | Can't quantify KV contribution without Step 1 |
| 6 | FMA/scalar/double arithmetic experiments | 1â3 hr | Only if Step 4 points specifically at attention math |
| 7 | Fix + regression + close | depends on cause | â |

Stop at whichever step definitively answers the question.

---

## Files affected (contingent on root cause)

| File | Expected change |
|---|---|
| `src/OpenTail.Stingray.Engine/ForwardPass.Attention.cs` | Possible: KV cache precision default |
| `src/OpenTail.Stingray.Engine/PagedKvCache.cs` or equivalent | Possible: CPU F16 KV store implementation |
| `src/OpenTail.Stingray.Core/ModelGraph.cs` | Possible: per-arch KV dtype flag |
| Scratch harness `ZzLayerDumpTmp.cs` | Binary dump extension (untracked) |
| `docs/1-correctness/bugstofix.md` | Update entry when closed |
| `docs/STATUS.md`, `ModelCompatibility.cs` | Update PPL numbers when consistent measurement taken |

---

## Success criteria

The bugstofix entry can be marked `[x]` when any of:

1. **Root cause found and fixed:** wikitext PPL (second-half, -c 2048) is â¤ 10.93; `Lfm2ParityTests` still pass (â¥ 14 teacher-forced positions); ForwardPass suite shows no regression.

2. **Root cause established and accepted:** binary dumps + Step 1 experiments conclusively attribute the 0.24% gap to F16 vs F32 KV representation (or inherent FMA accumulation order), and a pinned invariant test records this for future reference. The entry is documented as "known numerical difference, not a defect."

3. **A discrete op mismatch found and fixed:** same PPL and parity bar as (1).
