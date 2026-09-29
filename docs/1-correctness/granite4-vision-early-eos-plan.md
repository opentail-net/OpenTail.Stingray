# Granite 4.0 3B Vision â early EOS / empty decode plan

**Bug logged:** 2026-09-28 (RUNNING.md verification).  
**Entry in:** `docs/1-correctness/bugstofix.md`.  
**Model:** `granite-4.0-3b-vision-Q4_K_M.gguf` + `mmproj-granite-4.0-3b-vision-f16.gguf`.  
**Projector type:** `granite4-vision` (WindowQFormer deepstack).

---

## Failure description

Prefill succeeds (145 soft tokens, 20480-dim = 8 streams Ã 2560, at 44.3 t/s).  
Decoding stops after 0â3 tokens (e.g. a single period `.`). This is the EOS or an EOG token being
generated at position 1â3 of decode, not a crash.

**Control:** `granite-vision-3.2-2b` (standard MLP projector, no deepstack) works.

---

## Critical timeline â what changed between âworkingâ and âbrokenâ

| Date | Commit | What it did |
|---|---|---|
| 2026-09-27 | `0e7c406e` | Granite 4.0 Vision fix: per-token QFormer stream packing, deepstack injection, anyres/newlines. Documented as working against llama.cpp. |
| 2026-09-28 | `31d3846b` | Batched CPU image prefill: replaced the per-token `ForwardEmbedding()` loop with `PrefillEmbeddings()` â `PrefillCore()` for CPU. |
| 2026-09-28 | RUNNING.md verification | Granite 4.0 Vision now produces empty / early-EOS output. |

`31d3846b` is therefore the **primary regression vector**. This is not a Granite encoder bug â the
encoder (44.3 t/s, 145 tokens, 20480-dim) is working. It is a post-projector prefill bug.

---

## What the pipeline does for granite4-vision

1. **Encoder output:** `Granite4Adapter.EmbedImage` produces `nTok Ã 20480` floats
   (8 QFormer blocks Ã 2560-dim, concatenated along the feature axis).
2. **Injection point before `31d3846b`:** per-token `ForwardEmbedding()` loop, one call per soft token;
   `ForwardEmbedding` splits: first 2560 â `_hidden`, remainder â `_deepstackSlices`; `RunTrunk` injects
   per-layer.
3. **Injection point after `31d3846b` (CPU path):** `PrefillEmbeddings()` â `PrefillCore()` with
   `embeddingRows` pointing to the full 20480-wide buffer. `PrefillCore`:
   - Copies only `_embDim = 2560` floats per row into `batchHidden` (line 40â41) â correct.
   - At each mapped layer, reads `embeddingRows + n * embeddingWidth + dsIdx * _embDim` â
     mathematically correct.
4. **DeepstackMapping:** read from GGUF as `{arch}.deepstack_mapping` where `arch` is the text
   backbone's `general.architecture` string.
5. **`ScaleRawEmbeddings`:** set to `isGraniteFamily && !metadata.ContainsKey($"{arch}.deepstack_mapping")`
   (ModelGraph.cs:1312), so it is **false** for Granite 4.0 Vision. `PrefillCore` therefore does NOT
   scale the raw embedding rows. `ForwardEmbedding` also does not scale them (`if (_hp.ScaleRawEmbeddings)`).
   Both paths should agree on this.
6. **`RecurrentBatchedPrefillApplies`:** `true` for Granite 4.0 (no short-conv, no Mamba-2), so the
   `batched` flag in `PrefillEmbeddings` is `true` when `nTok > 1`, and all 145 image tokens
   are sent through `PrefillCore`.

---

## Hypotheses, ordered by current confidence

### H1 (HIGH) â Batched PrefillCore regression introduced by `31d3846b`

This is the primary suspect. The `PrefillEmbeddings` â `PrefillCore` batched path was not
tested for granite4-vision before committing. The Qwen3-VL validation noted in the commit only
checked `cosine â 0.9994` against the per-token path â acceptable for a standard VLM, but
not necessarily acceptable when the first generated token's logits sit near an EOS/EOG
decision boundary.

The `PrefillCore` deepstack injection (lines 136â139 of `ForwardPass.PrefillCore.cs`) is
mathematically identical to the `RunTrunk` per-token path on paper, but the **batched trunk
may use different int8 activation quantization** for the QK/V matmuls between image tokens
(the `STINGRAY_CPU_PREFILL_Q8` path). That numerical difference, compounded over 28 layers
and 145 image tokens, could shift the post-image logit distribution enough to make EOS/EOG
highly probable.

**Immediate diagnostic (no code change):** force the Granite deepstack case back through the
old per-token route by adding `|| _hp.NumDeepstack > 0` to the batched exclusion in `PrefillEmbeddings`:

```csharp
bool batched = count > 1 && RecurrentBatchedPrefillApplies && _layerHeadDim is null
    && !_usesUnweightedNorm && _tqKvCache == null && !_hp.HasPerLayerTokenEmbd
    && _hp.NumDeepstack == 0           // <- diagnostic gate
    && (!_hp.IsMoE || MoeBatchedPrefillSupported);
```

This is the cheapest, most decisive experiment. Run the exact failing command. If Granite 4
Vision starts producing full text descriptions, `31d3846b` is confirmed as the regression.

### H2 (MEDIUM) â `embeddingWidth` passed as 0 or wrong to `PrefillCore`

`PrefillEmbeddings` calls `PrefillCore(..., embeddingRows: p, embeddingWidth: width)` where
`width = rows.Length / count` = total floats / nTok. For granite4 this is `145 * 20480 / 145`
= 20480 â correct. But if `rows.Length` is anything other than `nTok * embd`, the slice math
would produce wrong offsets. Verify `width == 20480` with a one-line diagnostic.

### H3 (MEDIUM) â DeepstackMapping is null (wrong GGUF architecture key)

`ModelGraph.cs:1315` reads `{arch}.deepstack_mapping` where `arch` = the text GGUF's
`general.architecture`. If the key in this GGUF uses a different arch prefix
(e.g. `granite3` vs `granite` vs `granite3moe`), `DeepstackMapping` is null and
`NumDeepstack = 0`. Then `ForwardEmbedding` receives a 20480-wide row but only copies 2560
floats â the remaining 7 deepstack slices are silently dropped. Generation might still work
but would produce qualitatively wrong output (7 visual streams not injected). However this
would be a prefill error, not an EOS trigger, so it is lower priority for explaining the specific
early-EOS symptom.

**Also note:** `ScaleRawEmbeddings` (ModelGraph.cs:1312) is `isGraniteFamily && !metadata.ContainsKey(...)`.
If `deepstack_mapping` is NOT in the metadata (H3 is true), then `ScaleRawEmbeddings = true`,
meaning every image soft token would be multiplied by `EmbeddingScale` (e.g. 12.0). This would
cause catastrophically large activations and almost certainly trigger early EOS. **This makes H3
acutely important as a secondary check.**

### H4 (MEDIUM) â Chat template injects an EOG token in the assistant prefix

Granite's Jinja template is complex (it caused the parser to hang at one point).
The assistant-turn opening may include an EOG/EOS-family token that arrives as the last token
before decode, causing immediate termination. This would appear as `EOS = first generated token`,
not `EOS = second generated token`.

This was probably not introduced by `31d3846b` (it's a tokenizer/template issue), but it's
the easiest to test: run `--raw-prompt` or a text-only prompt and see if text decodes normally.
The pre-`31d3846b` success already rules out a pure template bug, so this is lower priority.

### H5 (LOW) â Off-by-one in deepstack injection indices

If the GGUF mapping stores 0-indexed values (0..7) and the code condition `dsIdx >= 1` (with
`(dsIdx-1) * _embDim` offset) skips the zeroth stream, all 7 injections would address wrong
slices. This would not cause immediate EOS but would produce wrong hidden states.

---

## Investigation steps (ordered by value / cost)

### Step 1 â Per-token A/B gate: the decisive experiment (~15 min)

Add `&& _hp.NumDeepstack == 0` to the `batched` condition in `PrefillEmbeddings`. Run:

```
stingray -m models/_models/granite-4.0-3b-vision-Q4_K_M.gguf \
  --mmproj models/_models/mmproj-granite-4.0-3b-vision-f16.gguf \
  --image photo.png -p "Describe this picture."
```

Expected result matrix:

| Per-token result | Interpretation |
|---|---|
| Full description (> 10 tokens) | **`31d3846b` regression confirmed.** Proceed to Step 6 for the real fix. |
| Still 0â3 tokens | Batched path not the primary cause. Proceed to Step 2 (arch key / mapping). |
| 0â3 tokens but different token(s) | Partial; check both H3 and H1 in parallel. |

### Step 2 â Capture post-image logits from both paths (~20 min)

Immediately after the image prefill and before the first decode step, print:
- Top-10 token IDs and strings
- EOS/EOG logit (check `tok.EogTokenIds`)
- `.` period token logit
- Top-1 minus top-2 margin
- Logit cosine between per-token and batched paths

This tells us whether the EOS/`.` is the consequence of corrupted post-image logits or of correct
logits + bad decode-loop stop handling. If EOS = top-1 on the batched path but top-25 on the
per-token path, the problem is fully inside the batched prefill arithmetic.

### Step 3 â Verify `hp.NumDeepstack` and `embeddingWidth` (~10 min)

Add diagnostics before `PrefillEmbeddings` in `RunImagePrompt`:
```csharp
Console.Error.WriteLine($"[granite4-diag] NumDeepstack={hp.NumDeepstack} embDim={hp.EmbeddingDim} embd={embd}");
```

Expected: `NumDeepstack=7`, `embDim=2560`, `embd=20480`.

If `NumDeepstack=0`, H3 is confirmed as a concurrent bug. Proceed to Step 4.

### Step 4 â Read the GGUF arch key for deepstack_mapping (~10 min)

Run `stingray list-metadata --model models/_models/granite-4.0-3b-vision-Q4_K_M.gguf | findstr deepstack`
and also check `general.architecture`. Compare the key prefix against what
`ModelGraph.cs:1315` expects (`{arch}.deepstack_mapping`). If the GGUF uses `granite3.deepstack_mapping`
but `arch` is `granite`, the lookup silently returns null.

### Step 5 â Text-only Granite control (~5 min)

Run the same text model without `--mmproj`:
```
stingray -m models/_models/granite-4.0-3b-vision-Q4_K_M.gguf -p "Count to 5."
```
If this also gives 0â3 tokens, there is a separate Granite text-model bug (template EOG, H4).
If this produces normal output, the fault is isolated to the vision path.

### Step 6 â Deepstack hidden-state trace (~1 h, only if Steps 1â3 are inconclusive)

For one image token (token index 72 = midpoint of 145), log the hidden-state L2-norm at:
- Before deepstack injection at each mapped layer
- After deepstack injection
- After FFN at the same layer

Compare per-token vs batched. The first layer where the norms diverge identifies the
specific computation that differs.

### Step 7 â First-decode-step instrument (~15 min)

In `DecodeLoop`, print for the first 5 positions:
```
position | chosen token ID | token string | is EOG? | top-5 alternatives | EogTokenIds
```

This disambiguates:
- Case A: first token is already an EOG â post-image logits are corrupted (confirmed H1)
- Case B: first token is `.`, second token is EOG â model state plausibly generated `.` as start of a sentence, then EOG'd; bad prefill slightly more likely than template
- Case C: decode loop exits despite non-EOG tokens â stop-token detection bug (very unlikely)

---

## Immediate fix (if Step 1 confirms H1)

The safest and fastest fix is to fall back to per-token `ForwardEmbedding` for any model with
`_hp.NumDeepstack > 0`. This restores the behaviour proven to work at commit `0e7c406e`:

```csharp
// ForwardPass.Decode.cs, PrefillEmbeddings
bool batched = count > 1 && RecurrentBatchedPrefillApplies && _layerHeadDim is null
    && !_usesUnweightedNorm && _tqKvCache == null && !_hp.HasPerLayerTokenEmbd
    && _hp.NumDeepstack == 0           // deepstack batched path not yet validated
    && (!_hp.IsMoE || MoeBatchedPrefillSupported);
```

This:
- Restores Granite 4.0 Vision to correct operation immediately
- Keeps Qwen3-VL on the per-token path (Qwen3-VL also uses deepstack, so this is conservative)
- Does NOT regress any non-deepstack VLM (they have `NumDeepstack == 0`)
- Does NOT require understanding why the batched path is numerically different

Document it with a comment: `// TODO: validate batched deepstack prefill against per-token path and re-enable`.

---

## Secondary fix (if Step 3/4 confirms H3 concurrent with H1)

If `DeepstackMapping` is null because the GGUF arch key doesn't match:

```csharp
// ModelGraph.cs
DeepstackMapping = GetIntArray(metadata, $"{arch}.deepstack_mapping")
    // fall-through aliases: granite4 checkpoints may ship under different arch strings
    ?? GetIntArray(metadata, "granite.deepstack_mapping")
    ?? GetIntArray(metadata, "granite4.deepstack_mapping")
    ?? GetIntArray(metadata, "granite3.deepstack_mapping")
    ?? /* Qwen3-VL synthesized mapping */ ...
```

Also fix `ScaleRawEmbeddings` to check all the alias keys:
```csharp
ScaleRawEmbeddings = isGraniteFamily
    && !metadata.Keys.Any(k => k.EndsWith(".deepstack_mapping"));
```

---

## Making batched deepstack prefill correct (longer-term, after correctness is restored)

Once the per-token fallback is in place, investigate whether `PrefillCore`'s batched path
can be brought back for deepstack models. The requirement is:

1. Same top greedy token as per-token path
2. Same first 10 greedy continuation tokens
3. No premature EOS/EOG
4. Logit cosine â¥ 0.9999 (higher bar than text-prefill because one wrong logit ends generation)

Likely causes of numerical divergence between the paths:
- `STINGRAY_CPU_PREFILL_Q8` activates int8 quantized QK/V matmuls in `PrefillCore`; the per-token
  path uses F32. For 145 image tokens this creates a compounded quantization error.
- The batched causal attention processes all 145 positions simultaneously, accumulating in a
  different order than 145 sequential single-token calls.

Fix candidates:
- Disable `STINGRAY_CPU_PREFILL_Q8` activation quantization for deepstack embeddings (narrower change)
- Make `PrefillCore` use F32 matmuls for vision-embedding inputs (controlled by `embeddingRows != null`)

Add a real end-to-end regression test before re-enabling:
```
Granite4VisionBatchedPrefillParityTest:
  - Load real checkpoint + mmproj
  - Encode deterministic image
  - Run per-token path â reference logits and first N greedy tokens
  - Run batched path â assert same top token, same N greedy tokens, cosine â¥ 0.9999
```

---

## Files affected

| File | Change |
|---|---|
| `src/OpenTail.Stingray.Engine/ForwardPass.Decode.cs` | Add `_hp.NumDeepstack == 0` to `batched` condition in `PrefillEmbeddings` |
| `src/OpenTail.Stingray.Core/ModelGraph.cs` | Possible: arch key alias for `deepstack_mapping` and `ScaleRawEmbeddings` |
| `tests/â¦/Granite4VisionParityTests.cs` | New: batched vs per-token parity gate |

---

## What NOT to do

- **Do not touch `Granite4VisionEncoder`, `Granite4ImagePreprocessor`, or the QFormer code.**
  The encoder produced 145 tokens at 44.3 t/s without error; the parity tests from `0e7c406e`
  already confirmed the projector output against `llama-mtmd-cli`. The regression is downstream.
- **Do not change `<image>` marker handling.** The `PlaceholderMarker` / `ImageOpenMarker` logic
  was fixed and documented. Reopening it risks introducing a second bug while chasing the first.
- **Do not disable batched prefill for all VLMs.** Only deepstack models need the exclusion.
- **Do not assume the Qwen3-VL cosine â0.9994 tolerance applies here.** For a model where one
  token decides whether generation starts, the tolerance must be much tighter.

---

## Success criterion

The bug is closed when:
1. `stingray -m granite-4.0-3b-vision-Q4_K_M.gguf --mmproj mmproj-... --image photo.png -p "Describe this picture."` produces a multi-sentence image description (> 10 tokens) matching the image content.
2. A regression test (`Granite4VisionBatchedPrefillParityTest` or similar) gates against re-introduction.
3. `docs/STATUS.md` and `RUNNING.md` are updated with the current command and confirmed behaviour.
4. The batched exclusion for deepstack models is documented with a `// TODO: validate and re-enable` comment.
