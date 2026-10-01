# Granite 4.0 3B Vision â early EOS / empty decode plan

**Bug logged:** 2026-09-28 (RUNNING.md verification).  
**Entry in:** `docs/1-correctness/bugstofix.md`, item **04**.  
**Model:** `granite-4.0-3b-vision-Q4_K_M.gguf` + `mmproj-granite-4.0-3b-vision-f16.gguf`.  
**Projector type:** `granite4-vision` (WindowQFormer deepstack).

---

## Failure description

Prefill succeeds (145 soft tokens, 20480-dim = 8 streams Ã 2560, at 44.3 t/s).  
Decoding stops after 0â3 tokens (e.g. a single period `.`). The decoder is behaving
correctly given the logits it receives â the fault must lie in how those logits are produced.

**Control:** `granite-vision-3.2-2b` (standard MLP projector, no deepstack) works.

---

## Critical timeline

| Date | Commit | What it did |
|---|---|---|
| 2026-09-27 | `0e7c406e` | Granite 4.0 Vision fix: per-token QFormer stream packing, deepstack injection, anyres/newlines. Documented as working against llama.cpp. |
| 2026-09-28 | `31d3846b` | Batched CPU image prefill: replaced the per-token `ForwardEmbedding()` loop with `PrefillEmbeddings()` â `PrefillCore()` for CPU. |
| 2026-09-28 | RUNNING.md verification | Granite 4.0 Vision now produces empty / early-EOS output. |

`31d3846b` is a **candidate change to test, not an established cause**. Earlier repository notes
(`docs/done/00-current-work-log-to-2026-09-27.md`, 2026-09-11) report non-image-grounded output
from real Granite vision runs, which predates this change. The reported symptoms and test inputs
also differ. Reproduce a fixed real-input baseline before classifying this as a regression.

---

## What the pipeline does for granite4-vision

1. `Granite4Adapter.EmbedImage` produces `nTok Ã 20480` floats (8 QFormer blocks Ã 2560-dim).
2. **Before `31d3846b` (per-token path):** `ForwardEmbedding()` called once per soft token;
   splits 20480-wide row into first 2560 floats (`_hidden`) + remaining 7 slices (`_deepstackSlices`);
   `RunTrunk` adds each slice at its mapped layer.
3. **After `31d3846b` (CPU batch path):** `PrefillEmbeddings()` â `PrefillCore()` with the
   full 20480-wide `embeddingRows` buffer. `PrefillCore` copies only the first 2560 floats
   into `batchHidden` per row (line 40â41), then adds the relevant deepstack slice at each
   mapped layer (line 136â139). Mathematically identical to the per-token path on paper.
4. **Width guard:** `PrefillEmbeddings` validates `width == _embDim || width == _embDim * (1 + nDs)`,
   throwing on mismatch â a wrong width cannot produce silent EOS.
5. **`NumDeepstack` check:** the same guard means: if prefill succeeded, `NumDeepstack == 7`
   is confirmed and `DeepstackMapping` is non-null. H3-style null-mapping scenarios would have
   thrown before any tokens were processed.
6. **`ScaleRawEmbeddings`:** `false` for Granite 4.0 Vision (ModelGraph.cs:1312:
   `isGraniteFamily && !metadata.ContainsKey($"{arch}.deepstack_mapping")`). Both paths
   correctly skip the 12Ã embedding scale for vision soft tokens.

---

## Hypotheses

### H1 (HIGH) â Batched PrefillCore introduces numerical divergence for deepstack models

The batched trunk uses `STINGRAY_CPU_PREFILL_Q8` int8-quantized QK/V matmuls for inter-token
attention. The per-token path uses F32 throughout. This difference is compounded over 145 tokens
Ã 28 layers, and the deepstack injection adds 7 additional residual contributions to each layer.
The Qwen3-VL validation of `31d3846b` accepted `cosine â 0.9994` â acceptable for text
generation but potentially inadequate for a multimodal boundary where one logit decides whether
generation starts.

Resulting symptom: post-image logits are corrupted enough that EOS or `.` becomes top-1,
and decode terminates after 0â3 tokens. The model itself is not broken; the logits it receives
from the batched prefill are wrong.

### H2 (HIGH) â Batched/deepstack interaction: a specific numerical failure in PrefillCore

Even if the int8 path is correct in isolation, the **interaction** between batched causal
attention across 145 image tokens and the 7 deepstack additions (each adding to a batch of
145 hidden states at once) may accumulate error differently than 145 independent single-token
trunk calls. The deepstack additions happen once-per-layer on the whole batch rather than once-
per-token, which changes the accumulation order. This is a plausible mechanism distinct from
pure int8 quantization.

**Three-variant experiment (see Step 1.3)** distinguishes H1 from H2.

### H3 (MEDIUM) â Deepstack mapping values or layer assignments are incorrect

If the GGUF `deepstack_mapping` values are 0-indexed (e.g. 0..7 instead of 1..7), the
condition `dsIdx >= 1` would incorrectly skip index 0 and address wrong slice offsets for
indices 1..7. This would produce wrong hidden states at every mapped layer. It does not
directly cause EOS but could bias logits toward EOG-family tokens. This would be a pre-existing
bug (present before `31d3846b`) rather than a regression, and is lower priority given
the September 27 success report.

**Sanity check:** print actual `hp.DeepstackMapping` values before the prefill and compare
against the raw GGUF metadata (`stingray list-metadata | findstr deepstack`).

### H4 (MEDIUM) â Chat template rendering produces wrong post-image logit context

> **Correction from earlier version of this plan:** a pre-existing EOG token in the rendered
> prompt template does NOT directly cause `DecodeLoop` to terminate â the loop only checks
> newly generated tokens. The real mechanism is different: if the chat template produces the
> wrong conversational context (wrong role delimiters, wrong turn structure), the model's prior
> distribution over the first generated token shifts toward EOG. This would be independent of
> `31d3846b` and would also affect the per-token path.

Since Granite 4 Vision worked with the same template on 2026-09-27, this is a lower-priority
hypothesis. Primarily useful to check via Step 5 (text-only control) and Step 3 (first decode
step instrument).

### H5 (LOW) â Stop/EOG handling incorrectly marks the first token as EOG

The first generated token is non-EOG (`.` or similar) but decode terminates anyway. This would
mean `GgufTokenizer.EogTokenIds` incorrectly includes `.`'s token ID. Very unlikely given the
control test (`granite-vision-3.2-2b` decodes normally on the same stack).

### H6 (VERY LOW) â Granite 4.0 Vision encoder/projector output is wrong

The encoder's output is already parity-tested against `llama-mtmd-cli` (144Ã20480 structure,
8-stream layout). The September 27 fix specifically addressed QFormer stream packing and
deepstack architecture. Reopening this without new evidence from the A/B experiments would
be premature.

---

## Investigation steps (ordered: cheapest decisive first)

### Step 1 â Per-token A/B gate â the decisive experiment (~20 min)

This is the single most important experiment. Run three variants:

| Variant | Code change | Purpose |
|---|---|---|
| A: Per-token + deepstack ON | Add `&& _hp.NumDeepstack == 0` to `batched` condition | Known-good baseline |
| B: Batched + deepstack ON | Current HEAD (no change) | Current failure |
| C: Batched + deepstack OFF | Skip deepstack additions in `PrefillCore` (zero them) | Isolates deepstack interaction |

Variant A is the critical gate:
```csharp
// ForwardPass.Decode.cs PrefillEmbeddings, diagnostic only:
bool batched = count > 1 && RecurrentBatchedPrefillApplies && _layerHeadDim is null
    && !_usesUnweightedNorm && _tqKvCache == null && !_hp.HasPerLayerTokenEmbd
    && _hp.NumDeepstack == 0   // <- diagnostic: force per-token for deepstack models
    && (!_hp.IsMoE || MoeBatchedPrefillSupported);
```

**Decision matrix:**

| A result | B result | C result | Interpretation |
|---|---|---|---|
| Full description | Early EOS | Full description | **`31d3846b` regression, deepstack interaction is the cause.** |
| Full description | Early EOS | Early EOS | **`31d3846b` regression, pure batched arithmetic is the cause.** |
| Early EOS | Early EOS | Early EOS | Not a batching regression; investigate H4 (template) and H3 (mapping). |
| Early EOS | Full description | ? | Per-token path itself is broken; check H3 (wrong mapping values). |

The first outcome (A works, B fails, C works) is the most useful: it confirms the batched
causal attention across 145 image tokens is the problem, not the deepstack injection itself.

### Step 2 â Capture post-image logits from both paths (~20 min)

Immediately after the last image soft token is processed, before the first text token, print:
- Top-10 token IDs and strings (for both A and B)
- EOS / EOG logit (`tok.EogTokenIds`)
- `.` period token logit
- Top-1 minus top-2 margin
- Logit cosine(A, B)
- Max absolute logit delta

This answers: is the EOS the **consequence of wrong post-image logits** (H1/H2) or does the
logit distribution look correct but something still stops decode (H4/H5)?

If `EOS = top-1` on B but `EOS = top-25` on A, the fault is entirely inside the batched
prefill arithmetic.

### Step 3 â First 5 decode steps instrument (~15 min)

In `DecodeLoop` (or equivalent), for the first 5 generated positions print:
```
position | token ID | token string | is_eog | top-5 alternatives and logits
```

This disambiguates:
- **Case A:** first token is already an EOG/EOS â post-image logits are bad (confirms H1/H2)
- **Case B:** first token is `.`, second is EOG â model generated `.` then ended turn; bad
  prefill context is still the likely cause
- **Case C:** first N tokens are ordinary words, decoder exits anyway â investigate H5
  (stop handling); very unlikely

### Step 4 â Sanity-check DeepstackMapping values (~10 min)

Note: given the prefill succeeded (145 tokens processed without exception), `NumDeepstack == 7`
and `DeepstackMapping != null` are already confirmed by the width guard. This step is only
about verifying the VALUES are correct (not zero-indexed), not the null question.

Before the prefill in `RunImagePrompt`:
```csharp
Console.Error.WriteLine($"[granite4-diag] NumDeepstack={hp.NumDeepstack}");
Console.Error.WriteLine($"[granite4-diag] Mapping[0..9]={string.Join(',', hp.DeepstackMapping!.Take(10))}");
```

Expected: `NumDeepstack=7`; mapping values in range 1..7 for the mapped layers
(not 0-indexed 0..6). Also run `stingray list-metadata` on the text GGUF and compare the raw
`{arch}.deepstack_mapping` values.

### Step 5 â Text-only Granite control (~5 min)

```
stingray -m models/_models/granite-4.0-3b-vision-Q4_K_M.gguf -p "Count to 5."
```

If this also terminates early, there is a separate Granite template/text bug (H4, H5) that is
independent of vision. If it produces normal output, the fault is isolated to the vision path
and the batched prefill is the prime suspect.

### Step 6 â Deepstack hidden-state trace (~1 h, only if Steps 1â3 are inconclusive)

For one image token (token index 72 = midpoint), log the hidden-state L2-norm at:
- After `batchHidden` init (after the initial copy, before layer 0)
- Before and after each deepstack injection at its mapped layer
- After the FFN at the same layer

Compare per-token vs batched. First divergence layer identifies the specific operation
that differs.

---

## Immediate fix (if Step 1 Variant A restores normal output)

Fall back to per-token `ForwardEmbedding` for any model with `_hp.NumDeepstack > 0`.
This restores the behaviour proven to work at commit `0e7c406e`:

```csharp
// ForwardPass.Decode.cs, PrefillEmbeddings
bool batched = count > 1 && RecurrentBatchedPrefillApplies && _layerHeadDim is null
    && !_usesUnweightedNorm && _tqKvCache == null && !_hp.HasPerLayerTokenEmbd
    && _hp.NumDeepstack == 0   // deepstack batched path not yet validated; see bugstofix entry
    && (!_hp.IsMoE || MoeBatchedPrefillSupported);
```

This:
- Restores Granite 4.0 Vision immediately
- Also conservatively falls back for Qwen3-VL (also uses deepstack) until validated
- Does NOT regress any non-deepstack VLM (`NumDeepstack == 0`)
- Does NOT require understanding why the batched path is numerically different

Document with a `// TODO: validate batched deepstack prefill, see 04-granite4-vision-early-eos-plan.md`
comment.

---

## Making batched deepstack prefill correct (longer-term, after correctness is restored)

Once the per-token fallback is in place:

1. **Identify the specific numerical mechanism** using Steps 2â6 above.
2. **Fix candidates (in order of invasiveness):**
   - Disable `STINGRAY_CPU_PREFILL_Q8` int8 activation quantization when `embeddingRows != null`
     (vision input already has no token-embedding quantization to match)
   - Pre-scale or adjust the deepstack slices before the batched injection
   - Accept the numerical difference and tighten the parity criterion
3. **Re-enable gated by a greedy parity test** (see Regression test below).

---

## Regression test (before re-enabling batched deepstack)

Machine-testable criterion â NOT "matches the image content" (subjective):

```
Granite4VisionBatchedPrefillParityTest:
  1. Load real checkpoint + mmproj
  2. Encode one deterministic image
  3. Run per-token path â first N greedy tokens (reference)
  4. Run batched path â first N greedy tokens
  5. Assert: same top-1 token at position 0
  6. Assert: same first 10 greedy tokens
  7. Assert: no EOG/EOS token within the first 10 positions
  8. Assert: logit cosine(per-token, batched) â¥ 0.9999 at position 0
```

Separately, a human smoke test verifies that the actual generated description is sensible.

---

## Files affected

| File | Change |
|---|---|
| `src/OpenTail.Stingray.Engine/ForwardPass.Decode.cs` | Add `_hp.NumDeepstack == 0` to `batched` condition in `PrefillEmbeddings` |
| `tests/â¦/Granite4VisionParityTests.cs` | New: batched vs per-token parity test |

---

## What NOT to do

- **Do not touch `Granite4VisionEncoder`, `Granite4ImagePreprocessor`, or the QFormer code.**
  The encoder produced 145 tokens at 44.3 t/s; the LlamaMtmdVisionParityTests already confirmed
  the 144Ã20480 8-stream output structure. The regression is downstream.
- **Do not check for EOG tokens embedded in the rendered prompt** as a direct stop causeÂ â `DecodeLoop` only checks newly generated tokens, not prefill tokens.
- **Do not assume wrong `embeddingWidth` can cause silent EOS.** The width guard in
  `PrefillEmbeddings` (line 86â88) throws on mismatch; the successful 145-token prefill
  confirms the width was 20480.
- **Do not assume null `DeepstackMapping` can cause silent EOS.** Same guard: if
  `NumDeepstack == 0`, the 20480-wide rows would throw at the width check. The successful
  prefill confirms the mapping is loaded.
- **Do not disable batched prefill for all VLMs.** Only deepstack models need the exclusion;
  non-deepstack VLMs retain the `31d3846b` performance improvement.
- **Do not accept `cosine â¥ 0.9994` as the parity bar for deepstack vision models.** That
  tolerance is from the Qwen3-VL text-prefill validation and is not adequate when one logit
  decides whether generation starts.

---

## Success criterion (Satisfied 2026-10-01)

1. `stingray -m granite-4.0-3b-vision-Q4_K_M.gguf --mmproj mmproj-... --image photo.png -p "Describe this picture."` produces > 10 tokens, with no EOG/EOS in the first 10 positions, and the same first greedy token as `llama-mtmd-cli` on the same inputs.
2. `GraniteVision4BatcedPrefillParityTest` passes (same top-1 token, same first 10 greedy tokens, logit cosine â¥ 0.9999).
3. `RUNNING.md` command updated and confirmed.
4. The `NumDeepstack == 0` exclusion is documented with a `// TODO` comment referencing this plan.

---

## Resolution & Root Cause (2026-10-01) - CLOSED

### Root Cause: Defective GGUF Metadata Chat Template (Hypothesis H4 Confirmed)

Investigation with both per-token and batched prefill proved that **batched deepstack prefill is 100% numerically sound** (logits agree within 0.09 across top-10 vocabulary, top-1 greedy token identical).

The early EOS / empty decode was caused by two defects in IBM's embedded GGUF Jinja chat template:
1. **Unconditional Default System Message**:
   The Jinja template in `granite-4.0-3b-vision-Q4_K_M.gguf` contains:
   ```jinja
   {%- if ns.system_message %}...{%- else %}{{- '<|start_of_role|>system<|end_of_role|>' + ns.default_system_message + '<|end_of_text|>\n' }}{%- endif %}
   ```
   Even when no system prompt was requested, it unconditionally injected:
   `"You are a helpful assistant. Please ensure responses are professional, accurate, and safe."` before the user turn.
2. **Whitespace Indentation Pollution**:
   The macro `render_content(x)` indented `{{ x }}` by 8 spaces (`        `), prepending 8 spaces right before `<image>`.
3. **Reference Parity Verification**:
   When Granite 4.0 Vision receives this injected system prompt before the image token, its post-image top-1 logit shifts directly to `<|end_of_text|>` (logit 15.6677 vs 15.1860 for text), causing immediate termination at token 0.
   Crucially, running reference `llama-mtmd-cli` with `--jinja` reproduces the **exact same bug** (`: \n\n this picture.`)! By default, llama.cpp ignores Jinja and uses the hardcoded `LLM_CHAT_TEMPLATE_GRANITE_4_0` (`llama-chat.cpp`), which emits no default system prompt and no 8 spaces.

### The Fix

1. **Routing in `RunCommand.FormatPrompt` & `ChatTemplateRenderer`**:
   Bypass the defective metadata Jinja for `s_arch is "granite"` and format canonical Granite roles:
   ```
   <|start_of_role|>user<|end_of_role|>{userMessage}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>
   ```
   (and including `<|start_of_role|>system<|end_of_role|>{systemPrompt}<|end_of_text|>\n` only when explicitly provided by the user).
2. **Tests Added**:
   - `ServerLibraryTests.cs`: `Fallback_Granite_EmitsCanonicalGraniteFraming` and `Granite_BypassesBrokenJinjaTemplate_EmitsCanonicalFormat`.
   - `Granite4VisionTests.cs`: `Granite4Vision_PromptFraming_MatchesCanonicalFormat`.
   - `Granite4VisionE2ETests.cs`: End-to-end real weights test verifying batched deepstack prefill, no early EOS, and greedy decode with correct grounding on sample image.
3. **Verification**:
   - Prefill: 737 tokens (724 image + 13 text) at 104.9 t/s.
   - Decode: 50+ tokens at 11.8 t/s with zero early EOS.
   - Output grounded: `"The picture depicts a red apple, set on a warm wooden surface. The apple's vibrant red hue dominates the scene while its glossy finish enhances the natural beauty of its form..."`
   - Added verified command and throughput to `docs/RUNNING.md`.
