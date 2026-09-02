# 059 — HunyuanVL (HunyuanOCR) vision encoder implementation plan

**UPDATE (2026-09-02): all four gaps implemented.** `src/OpenTail.Stingray.Vision/
HunyuanVlVisionEncoder.cs` now has the two tensor-name fixes (`mm.2.weight`/`mm.model.fc.weight`),
a real strided-Conv2D merger (`ApplyStridedConv2DMerge`, adapted from `Glm4VisionEncoder.
ApplyPatchMerger`'s index math per the DRY note below), the real bilinear position-embedding
resize (`AddResizedPosEmbd`, exact formula from `clip.cpp`), and `image_newline` row insertion +
`mm.image_begin`/`mm.image_end` sequence wrapping with the real `(outX+1)*outY+2` token-count
formula, applying `mm.post_norm` to the whole wrapped sequence as the real reference does. Also
fixed two related bugs found while implementing: `HunyuanVlVisionModel.cs`'s `projectionDim`
inference read the same wrong `mm.model_proj.weight` name (now `mm.model.fc.weight`), and
`UnifiedVisionPipeline.cs`'s GGUF-autodetect fallback branch had an always-false condition
(`t.Name == "mm.pre_norm.weight" && t.Name == "mm.model_proj.weight"` — a single tensor can never
equal two different names — fixed to two separate `.Any()` checks with the correct name).

**Verification status: GOLDEN-VERIFIED (2026-09-02).** `dotnet build` clean (0 warnings/errors);
`MultimodalRealWeightsTests` (11 tests, all real-weights vision architectures including
`HunyuanVl_RealWeights_LoadsAndEmbedsImage`) run clean: 11/11 passed, 0 regressions in any other
architecture. Beyond the differentiation bar, wrote `scripts/hunyuanvl_ref.py` (a real numpy port
of `hunyuanvl.cpp`'s `build()` + `clip.cpp`'s bilinear position-embedding resize, reading the same
local `models/mmproj-hunyuanocr-q8_0.gguf`) and `HunyuanVlVisionEmbedderParityTests.cs`
(`Forward_MatchesNumpyReference`, per-token cosine-similarity check, same pattern as GLM-4.6V/
Llava/Pixtral/Exaone4/Qwen2.5-VL/MimoVl this session) — **passes** (min per-token cosine > 0.97,
mean abs diff < 5e-2) on a deliberately non-native 160×128 test image (patchesX=10, patchesY=8,
chosen specifically so the bilinear position-embedding resize path is actually exercised, not
skipped — a native 2048×2048 image would coincide exactly with the stored 128×128 grid). Real
token count confirmed matching the formula: `(outX+1)*outY+2 = (5+1)*4+2 = 26`. This architecture
can now be marked confirmed-working in the README matrix.

---

**Original status: PLAN ONLY, not started.** Written after reading the complete real reference
(`examples/llama.cpp/llama.cpp/tools/mtmd/models/hunyuanvl.cpp`, 63 lines, read in full) and the
real `PROJECTOR_TYPE_HUNYUANVL` position-embedding-fill branch in `clip.cpp`, cross-checked line
by line against the current `src/OpenTail.Stingray.Vision/HunyuanVlVisionEncoder.cs` and the real
local checkpoint's exact tensor names/shapes (`models/mmproj-hunyuanocr-q8_0.gguf`, confirmed via
`list-tensors`/`list-metadata`). This is real, scoped feature work — same category as the
windowed-attention gap closed earlier this session — not a quick tensor-rename fix, which is why
it was deliberately deferred rather than attempted opportunistically mid-pass. `docs/00-current-
work.md`'s "New finding" section already documents `HunyuanVl` returning an all-zero embedding
(matches this plan's finding #3 below almost exactly: a silently no-op'd projector stage).

## Why this is bigger than it first looked

The differentiation-suite failure ("Embedding A is completely zero") made it LOOK like a single
missing tensor. It's actually four independent, real gaps, only one of which is a simple rename.
None of them were fixable without reading the real reference — the current C# code's structure
(pixel-shuffle-then-linear projector, no newline/wrap tokens, raw position-embedding add with no
resize) was a reasonable GUESS at a "Llava-style" projector, but HunyuanVL's real projector is
architecturally different in ways a tensor-name audit alone wouldn't surface.

## The four real gaps, in priority order (each independently blocking correct output)

### 1. Wrong tensor names for TWO of the three projector stages (the dominant bug — same class as YoutuVl's `mm.1`-vs-`mm.2` fix this session)

`HunyuanVlVisionEncoder.cs` reads:
- `"mm.1.weight"` for the second conv stage — **real name is `mm.2.weight`** (confirmed via
  `list-tensors`: `mm.0.weight`/`mm.2.weight` exist, `mm.1.*` does not).
- `"mm.model_proj.weight"` for the final LLM-hidden-size projection — **real name is
  `mm.model.fc.weight`** (confirmed present; `mm.model_proj.*` does not exist).

Both `LoadTensorF32` calls silently return `null` for a missing name, and the encoder's own
`VisionOps.MatVec(..., null, ...)`-equivalent calls then no-op (leave the destination buffer at
its zero-initialized default) — this is *exactly* why the differentiation test sees an all-zero
embedding: the LAST TWO stages of the three-stage projector never ran at all, the whole time.
**Fix: two one-line name corrections** (`mm.2.weight`/`mm.2.bias`, `mm.model.fc.weight`/
`mm.model.fc.bias`).

### 2. The projector's first stage is a REAL strided Conv2D, not a pixel-shuffle-then-Linear (same class as GLM-4.6V's patch-merger fix this session)

Real (`hunyuanvl.cpp` lines 24-31): `ggml_conv_2d(ctx0, model.mm_0_w, cur, merge, merge, 0, 0, 1,
1)` — a genuine strided 2D convolution (kernel=stride=`n_merge`=2) over the `[W,H,C]` ViT-output
grid, `C=1152 -> 2304`. The current C# code instead calls `VisionOps.PixelShuffle2x2` (a plain
channel-concat of 4 adjacent patches, **spatial-position OUTER / channel INNER** ordering — see
its own implementation: each of the 4 sub-blocks is copied as one contiguous `inDim`-length run)
followed by a plain `MatVec` treating `mm.0.weight` as an ordinary `[outDim,inDim]` linear matrix.

That ordering is backwards from the real conv weight's raw layout. `mm.0.weight`'s real GGUF shape
is `[2,2,1152,2304]` (`ne0`=kw fastest, `ne1`=kh, `ne2`=cin, `ne3`=cout — the SAME raw-byte
convention already established this session for GLM-4.6V's `mm.patch_merger.weight` and Hunyuan's
own `v.patch_embd.weight`), meaning for a fixed output channel the real per-position index order is
**channel OUTER, spatial (dy,dx) INNER** — the opposite of what `PixelShuffle2x2` produces. Even
after fixing the `mm.2`/`mm.model.fc` names above, this ordering mismatch would still corrupt
every value passing through the first conv.

**Fix: implement a real strided-Conv2D merger**, structurally identical to
`Glm4VisionEncoder.ApplyPatchMerger` (already written and golden-verified this session) — for each
`scale x scale` (`scale=n_merge`) block of ViT-output tokens: `out[o] = bias[o] + sum over
(c,dy,dx) of weight[o,c,dy,dx] * hidden[srcRow,srcCol,c]`. Reuse that exact method (or extract it
to a shared `VisionOps` helper — see "DRY note" below) rather than reimplementing the index math a
third time.

### 3. Learned position embedding needs a real, specific bilinear resize — currently just truncated/added raw

The real checkpoint's `v.position_embd.weight` is `[1152, 16384]` — a **128×128 native grid**
(`16384 = 128²`), NOT sized for whatever the actual input image's patch grid happens to be. Real
`clip.cpp` (`PROJECTOR_TYPE_HUNYUANVL` branch, ~line 4809-4869) resizes this down to the real
`(pw,ph)` grid on every forward pass using a specific formula — NOT a generic/library bilinear
resize, an exact one that must be replicated:

```
sx = (out_w + 0.1) / n_grid          sy = (out_h + 0.1) / n_grid     // n_grid = 128
for y in 0..out_h:
    fy = (y + 0.5) / sy - 0.5        // pixel-center convention, align_corners=False
    y0 = clamp(floor(fy), 0, n_grid-1);  y1 = clamp(y0+1, 0, n_grid-1)
    wy1 = clamp(fy - y0, 0, 1);  wy0 = 1 - wy1
    for x in 0..out_w:
        fx = (x + 0.5) / sx - 0.5
        x0 = clamp(floor(fx), 0, n_grid-1);  x1 = clamp(x0+1, 0, n_grid-1)
        wx1 = clamp(fx - x0, 0, 1);  wx0 = 1 - wx1
        dst[y,x] = wy0*wx0*src[y0,x0] + wy0*wx1*src[y0,x1]
                 + wy1*wx0*src[y1,x0] + wy1*wx1*src[y1,x1]
```

The current C# code (`if (_posEmbd != null) { ... x[i] += _posEmbd[i] ... }`) just adds the RAW
128×128-grid-sized array element-wise, truncated to `nP*_embd` — correct ONLY in the coincidental
case where the input image happens to be exactly 128×128 patches (2048×2048 pixels at patch=16;
essentially never in practice, since real preprocessing resizes based on min/max-pixel-count
constraints, not a fixed square). **Fix: implement this exact bilinear resize** before the add,
replacing the current truncating add.

### 4. Missing `image_newline` row markers and `image_begin`/`image_end` sequence wrapping

Real (`hunyuanvl.cpp` lines 37-53): after the two-stage conv projector, a learned
`v.image_newline` embedding vector (`[4608]`, matching the conv output width) is **inserted after
every row** of the merged token grid (`ggml_concat` along the width dimension, then reshaped into
a flat sequence) — a real, structural token-count change: `(ow+1) * oh` tokens, not `ow*oh`. Then,
AFTER the final `mm.model.fc` projection, the whole sequence is wrapped with a single
`mm.image_begin` token prepended and a single `mm.image_end` token appended (both `[1024]`,
LLM-hidden-size vectors — real learned embeddings, not text tokens), and only THEN is
`mm.post_norm` (RMSNorm) applied — to the WHOLE wrapped sequence, including the begin/end markers.

The current C# code does none of this: no newline insertion (so `tokenCount` itself is wrong,
missing the `+1` per row), no begin/end wrapping (missing 2 more tokens), and applies
`mm.post_norm` only to the un-wrapped per-position tokens. **Fix: implement newline insertion (a
real per-row array insert, not a strided view) and begin/end concatenation, matching the real
`tokenCount = (outX+1)*outY + 2` formula, before the final RMSNorm.**

## Additional real facts confirmed while reading (not bugs, just worth recording so the next pass
doesn't have to re-derive them)

- ViT trunk itself matches the current C# structure closely and is likely ALREADY CORRECT: separate
  (not fused) `attn_q/k/v` with bias, plain `NORM_TYPE_NORMAL` LayerNorm (`ln1`/`ln2`, both with
  bias) — not RMSNorm — for the ViT's own per-block norms, and a plain non-gated GELU FFN
  (`ffn_up`/`ffn_down` only, both with bias; confirmed no `ffn_gate` tensor exists in this
  checkpoint, so the encoder's existing `if (b.FfnGateW != null)` SwiGLU branch is correctly
  dead/unused here — not a bug). `VisionOps.Attention`'s real softmax is already used and already
  correct (shared with other, already-golden-verified encoders this session).
- No RoPE at all — `build_vit`'s `add_pos` callback is `nullptr`; position is added exactly ONCE
  (via the resized `pos_embd`, added before the first block) and never touched again per-layer.
- No separate `v.pre_ln`/`v.post_ln` tensors exist in this checkpoint (confirmed via
  `list-tensors`) — the encoder's existing `_preLnW`/`_postLnW` fields correctly stay `null` and
  those (optional, tensor-presence-gated) steps correctly no-op already. Not a bug.
- `patch_size=16` (not 14), single (not dual) patch-embed conv, with bias.
- `n_merge` (`hparams.n_merge`) real default is 2, overridable via `clip.vision.spatial_merge_size`
  metadata — matches the encoder's existing `_nMerge` field, no change needed there.
- `mm.pre_norm` (RMSNorm, applied to the ViT's raw output before the projector) is already read
  and applied correctly under the name `mm.pre_norm.weight` — no bug there.

## DRY note (per this project's own house rule — perf/DRY pass only after correctness, but worth
flagging now since the pattern already exists)

The real strided-Conv2D merger needed for gap #2 is structurally IDENTICAL to
`Glm4VisionEncoder.ApplyPatchMerger` (added this session, golden-verified). Once HunyuanVL's own
version is implemented and verified, consider extracting a shared
`VisionOps.StridedConv2DMerge(...)` helper — this exact "real strided conv2d, GGUF raw-byte
`[cout,cin,kh,kw]` layout" pattern has now appeared in at least two architectures (GLM-4.6V,
HunyuanVL) and is a likely-recurring shape for future VLM ports. Do not extract prematurely before
HunyuanVL's own version is correctness-verified, per CLAUDE.md rule 7's DRY-after-correctness
ordering.

## Suggested implementation + verification order

1. Fix the two tensor names (gap #1) — cheapest, isolate first to confirm it alone doesn't already
   fix everything (it won't, since gaps #2-4 are also real, but verify via the differentiation
   test moving from "all-zero" to *some* non-degenerate-but-still-likely-wrong output, confirming
   the no-op chain is broken).
2. Implement the real strided-Conv2D merger (gap #2), reusing/adapting
   `Glm4VisionEncoder.ApplyPatchMerger`'s index math.
3. Implement the bilinear position-embedding resize (gap #3) — self-contained, testable in
   isolation (a small standalone script or unit test comparing against a hand-computed example
   would catch transposition/off-by-one errors before wiring it into the full forward pass).
4. Implement newline insertion + begin/end wrapping + moving `mm.post_norm` to apply after
   wrapping (gap #4) — the real `tokenCount` formula changes, so this needs to update however
   `UnifiedVisionPipeline`'s `HunyuanVlAdapter` (or equivalent) consumes the encoder's token count
   too — check that call site while implementing, not just the encoder itself.
5. **Golden-verify, following this session's established methodology**: write
   `scripts/hunyuanvl_ref.py` (real numpy port of `hunyuanvl.cpp`'s `build()` + the bilinear
   position-embedding resize, reading the same local `mmproj-hunyuanocr-q8_0.gguf`) and a matching
   `HunyuanVlVisionEmbedderParityTests.cs`, the same pattern used for GLM-4.6V/Llava/Pixtral/
   Exaone4/Qwen2.5-VL/MimoVl this session. Do not declare this fixed on differentiation-test
   passing alone — that bar was already passing (in the weak "not all-zero" sense) for several
   OTHER architectures this session that turned out to have severe, real bugs underneath it.
6. Re-run the full `MultimodalRealWeightsTests` suite for regressions before committing, per
   established convention this session.

## How to reproduce / verify tensor facts independently

```bash
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- list-tensors -m models/mmproj-hunyuanocr-q8_0.gguf
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- list-metadata -m models/mmproj-hunyuanocr-q8_0.gguf
```

The checkpoint is already present locally (`models/mmproj-hunyuanocr-q8_0.gguf`) — no download
needed to start this work, unlike most of this session's other diffusion/audio investigations.

## House rules for whoever picks this up (from this project's `CLAUDE.md`)

- **No subagents** — do all work directly in the main session for this project.
- **Check the real reference before "fixing" code that looks wrong** — every claim in this plan
  was confirmed by reading `hunyuanvl.cpp` in full and the real `clip.cpp` position-fill branch
  directly, plus direct `list-tensors`/`list-metadata` inspection of the real local checkpoint —
  not guessed. Do the same for anything not already covered here.
- **Golden-verify, don't stop at differentiation-passing** — see step 5 above.
- **DRY pass only after correctness** — see the DRY note above; don't extract the shared conv
  helper before HunyuanVL's own version is verified correct.
