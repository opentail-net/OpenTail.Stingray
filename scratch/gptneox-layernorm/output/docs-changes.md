# Proposed documentation changes for the GPT-NeoX (Pythia) admission

Do NOT apply these directly — this file is input for a human reviewer to fold into the real docs
(`docs/01-gguf-model-coverage-plan.md`, and possibly `docs/00-current-work.md`), per the task plan's
instruction not to touch those files myself.

## 1. `docs/01-gguf-model-coverage-plan.md` — add a new §1h section

Insert after §1g (`apertus` — ADMITTED ...), following the same structure/tone as §1g and §1d
(Granite). Suggested content:

---

### 1h. `gptneox` — ADMITTED 2026-08-08, 22-of-24-token exact match, second new-kernel
architecture built this session

`EleutherAI/pythia-160m` (Apache-2.0), via `mradermacher/pythia-160m-GGUF`, Q8_0 (174.6 MB, deleted
after this receipt per the working pattern). `tokenizer.ggml.pre = olmo` was already covered by the
ported pretokenizer cascade (§3), so this exercised only the architecture axis.
`general.architecture = gptneox`.

**Result: 22 of 24 tokens EXACT, stronger than every prior receipt this session** (Apertus 11/24,
OLMoE 2-token prefix). llama.cpp's reference continuation for `"The capital of France is"` (prompt
ids `[510, 5347, 273, 6181, 310]`) is `" located in the city of Paris.\n\nThe city is also home to
the famous French football club, the Paris Saint"`. This engine matches token-for-token through
`"...football club, the "` and diverges only at the last two tokens: it picks `" F"` (logit 830.052)
where the reference implies `" Paris"` (this engine's own logit for it: 830.045) — a genuine
near-tie, 0.007 apart on logits around 830, confirmed via a top-5 logit dump at the divergence point,
not assumed. Same category of evidence the OLMoE (§1b) and Apertus (§1g) receipts accepted. See
`GptNeoxGreedyParityTests.cs`.

**What GPT-NeoX needs beyond the plain llama trunk — confirms the §1f item-1 assessment exactly, plus
one thing item 1's writeup didn't call out (fused QKV):**

1. **LayerNorm-with-bias** (mean-subtract + variance-normalize + learned scale + learned bias) on
   every norm in the model — `SimdKernels.LayerNorm`, dispatched via a new `ForwardPass.FastNorm`
   wrapper whenever `ModelHyperparams.HasNormBias` is set (detected from tensor inventory,
   `blk.0.attn_norm.bias`, same pattern as the existing `HasAttnBias`).
2. **Non-gated GELU FFN** (`down(gelu(up(x)))`, no `ffn_gate` tensor) — `SimdKernels.GeluInPlace`,
   the tanh-approximate GELU, gated on `ModelHyperparams.HasFfnBias` and distinguished from Apertus's
   xIELU (also non-gated) by checking `_xieluAlphaN is not null` first in `DenseFfn`/`PrefillCore`.
3. **True parallel residual** (`use_parallel_residual=true`): attention and FFN both read the SAME
   pre-attention layer input, and the layer output is a 3-way sum `input + attn_out + ffn_out`, not
   two sequential normalize-then-add steps. Implemented as an isolated
   `if (_hp.UseParallelResidual) { ... } else { <existing sequential path, untouched> }` branch in
   both `RunTrunk` (decode) and `PrefillCore` (batched prefill) — independently, since these are
   genuinely separate code paths in this engine, not a shared helper.
4. **Fused `attn_qkv.weight`/`attn_qkv.bias`** (not called out explicitly in §1f's item-1 writeup):
   this Pythia GGUF ships one 2304-wide fused QKV weight+bias pair per layer rather than separate
   `attn_q`/`attn_k`/`attn_v` tensors. `ForwardPass`'s constructor now splits the fused weight by
   byte/row offset (three `TensorRef`s pointing into the same underlying tensor, read-only, no
   ownership issue) and the fused bias by element offset (see the defect-2 writeup below for why
   this needed care on the *write* side, i.e. `Dispose()`).

Partial RoPE (`gptneox.rope.dimension_count=16` of `headDim=64`) and the `layer_norm_epsilon` metadata
key both turned out to be **already-generic mechanisms** needing no new code — `ModelHyperparams.RopeDim`
already reads `{arch}.rope.dimension_count` for every architecture (built for qwen35moe), and
`RmsNormEps` already has a two-key fallback chain that happens to land on the right key for GPT-NeoX.
Both were verified at runtime (not assumed), per the plan doc's standing rule.

**Two real defects found and fixed while building this receipt — both in refactored plumbing, not in
the "obvious" formula, both silent (fluent-looking output, then a delayed crash), matching the
Apertus xIELU precedent (§1g) for how this class of bug hides:**

1. **Flipped residual-save direction, corrupting layer 0 only.** `PrefillCore`'s per-token attn-norm
   setup loop was refactored (to plumb the new norm-bias parameter through) from
   `Copy(batchResidual, batchHidden, embDim)` (save this layer's input into the residual slot) into
   `Copy(batchHidden, batchResidual, embDim)` — the direction silently reversed. Invisible for layers
   1-11 (both buffers already agree by then, from the previous layer's own residual-store step), but
   `batchResidual` starts zeroed and is never populated before layer 0 runs, so the flip fed layer
   0's LayerNorm an all-zero input instead of the token embedding. LayerNorm on zero input reduces to
   just the learned bias vector — nonzero, so the result still looked like a plausible norm output;
   nothing crashed, nothing was NaN, the model just silently discarded the prompt at layer 0.
   Confirmed with a per-layer L2-norm trace (0.0000 before the fix, 0.5308 — the embedding's actual
   norm — after). Fixed by restoring the original copy direction.
2. **Three bias pointers aliased into one allocation, corrupted the native heap on teardown.** The
   fused-QKV-bias split loaded the whole 2304-float fused bias into ONE buffer and pointed
   `_bq[i]`/`_bk[i]`/`_bv[i]` at three offsets within that single allocation — correct for reading,
   but `ForwardPass.Dispose()` unconditionally frees all three independently (every other bias array
   in this engine is an independent per-tensor allocation, so the free loop has no reason to expect
   aliasing). Freeing `_bk[i]`/`_bv[i]` — pointers into the middle of `_bq[i]`'s block, not block
   starts — corrupted the process heap. Detection was fully deferred: model load, prefill, and a full
   23-step greedy decode all completed and produced plausible-if-wrong output with zero errors; the
   process only died with `STATUS_HEAP_CORRUPTION` (0xC0000374) when `Dispose()` reached the first
   `NativeMemory.Free` on an aliased pointer, nowhere near the code that caused it. Fixed by copying
   each Q/K/V slice into its own `Alloc()`'d buffer instead of aliasing. Also fixed, found by
   inspection while in the area: `_bAttnNorm`/`_bFfnNorm`/`_bOutputNorm`/`_bFfnUp`/`_bFfnDown` (all
   new for this architecture) were allocated but never freed in `Dispose()` — a plain leak, unrelated
   to the crash but same neighborhood.

**Licensing note, carried over from §1f/§1c:** `starcoder2` (BigCode OpenRAIL-M) and `stablelm`
(mixed CC-BY-SA/CC-BY-NC / Stability AI Community License) remain license-blocked; only `falcon`
(TII, Apache-2.0, not yet built) and `gptneox` (this receipt) are actually clean of the four
architectures item 1 in §1f originally grouped together. `falcon` should reuse essentially all of the
LayerNorm/non-gated-FFN plumbing this receipt built — the remaining unknown is whether Falcon needs
anything gptneox didn't (e.g. Falcon's multi-query variant, or its own bias conventions);
not investigated as part of this receipt.

**NOT wired (explicitly out of scope, same pattern as every other receipt this session):**
`PrefillCoreTq` (TurboQuant batched prefill), `PrefillWithCache` (continuous-batching admission),
`BatchForwardMulti`/`PrefillPackedMulti` (multi-sequence batched decode), and the CUDA/Vulkan
backends. A GPT-NeoX model run through any of those paths today will not use LayerNorm/parallel
residual/GELU and will produce wrong output (most likely: outright missing-tensor exceptions, since
those paths don't probe for `HasNormBias` etc at all). Track before enabling `gptneox` anywhere
outside the CPU dense single-user path (`ForwardPass.Prefill`/`Forward`).

---

## 2. §1c — mark `gptneox` ADMITTED

In the numbered candidate list (currently line ~169-180) and the "Reclassified 2026-08-08" bullet
(currently line ~185-192), update the `gptneox` mentions to note it moved from "new-kernel work" to
ADMITTED (§1h), the same way `apertus`'s line already reads "ADMITTED — see §1g" elsewhere in this
doc. Suggested wording for the reclassification bullet's last sentence: "`gptneox` ADMITTED 2026-08-08
— see §1h; `stablelm` remains license-blocked, so this reclassification's payoff for `starcoder2`
specifically is unrealized pending a permissive-license check on that one."

## 3. §1f — mark item 1 partially built

Item 1 ("LayerNorm-with-bias + non-gated FFN") currently reads as fully prospective. Update its
opening sentence to note gptneox's slice of this item is now built (§1h) and admitted, with `falcon`
remaining as the one other clean, not-yet-built beneficiary of the same kernel work (`starcoder2`
and `stablelm` still license-blocked per the existing table). Also worth a one-line cross-reference
from §1f's "Recommended order" paragraph (currently the last paragraph before §1g) noting item 1 is
no longer purely prospective.

## 4. `docs/OpenTail.Stingray-Design.md` (not reviewed in detail for this task — flagging only)

If that document enumerates supported architectures or forward-pass variants anywhere (I did not
check), it likely needs a `gptneox` mention alongside `apertus`/`granite`/`smollm3`. Not investigated
as part of this task — the plan scoped doc changes to `docs/01-gguf-model-coverage-plan.md` and
`ModelCompatibility.cs` (the latter's real-tree copy already has the actual edit, see
`scratch/gptneox-layernorm/output/ModelCompatibility.cs`).
