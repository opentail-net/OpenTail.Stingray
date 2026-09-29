# Plan: LLaVA-NeXT / LLaVA-OneVision AnyRes (`llava_uhd`) parity

**Entry in:** `docs/1-correctness/bugstofix.md`, item **15**.

## 0. Current state

Classic LLaVA-1.5 is already proven:

- single `336×336` image;
- CLIP-style ViT + 2-layer MLP projector;
- patch/CLS ordering fixed;
- direct image-token splice for the LLaMA-2/Vicuna vocabulary;
- end-to-end CPU and Vulkan image generation matches llama.cpp;
- `LlamaMtmdVisionParityTests.Llava15_Rainbow336_MatchesLlamaMtmdDebug`.

The repository also contains an AnyRes implementation in
`src/OpenTail.Stingray.Vision/LlavaImagePreprocessor.cs`, modeled on llama.cpp's `llava_uhd`:

1. one overview image;
2. optional best-pinpoint resize;
3. tiled `imageSize × imageSize` views;
4. row-major tile ordering;
5. model-specific mean/std;
6. concatenation of the resulting soft-token sequences.

`UnifiedVisionPipeline.LlavaAdapter` reads `clip.vision.image_grid_pinpoints` and passes those
pinpoints to `PreprocessViews(...)`.

This implementation was proven indirectly while fixing Granite Vision 3.2, but no real LLaVA-NeXT or
LLaVA-OneVision checkpoint has exercised the LLaVA adapter's AnyRes path end to end. The purpose of
this item is to prove or disprove the existing LLaVA AnyRes implementation against a real
LLaVA-NeXT/OneVision checkpoint before changing it. Do not assume Granite verification is sufficient.

## 1. Scope

### In scope

Verify the complete AnyRes path for at least one real LLaVA-NeXT checkpoint. Preferably also run a
LLaVA-OneVision checkpoint if it uses the same adapter path and can be obtained without excessive
model size. Cover:

- `clip.vision.image_grid_pinpoints`;
- best-resolution selection;
- overview generation;
- pinpoint resize/padding;
- tile extraction and ordering;
- per-view preprocessing/normalization;
- per-view vision encoder execution;
- soft-token concatenation;
- image-to-text prompt expansion;
- actual image token count;
- separator presence/absence;
- final decoder behavior.

### Not in scope

- redesigning the LLaVA vision encoder;
- changing the proven LLaVA-1.5 single-tile path;
- Granite Vision correctness;
- new AnyRes algorithms unrelated to llama.cpp;
- GPU optimization or CUDA parity;
- adding new model families.

## 2. Phase 0 — Select a real checkpoint deliberately

Choose a checkpoint whose mmproj declares real LLaVA-NeXT/AnyRes metadata. It must contain
`clip.vision.image_grid_pinpoints` or equivalent metadata consumed by the current adapter. Prefer a
configuration clearly larger than `336×336` so the test exercises multi-tile slicing. Do not use a
square `336×336` image as the main proof; it could collapse to the already-proven LLaVA-1.5 behavior.

Record base GGUF, mmproj GGUF, `clip.vision.image_size`, patch size, projector dimension, complete
`image_grid_pinpoints` list, and selected pinpoint for each test image.

## 3. Phase 1 — Unit-test AnyRes geometry independently

Before running the full model, test pure preprocessing decisions.

### 3.1 Pinpoint selection

Exercise `LlavaImagePreprocessor.SelectBestResolution(...)` against llama.cpp's
`select_best_resolution` formula. For each wide, tall, and approximately square test image record
source width/height, candidate resolutions, selected resolution, effective pixels, and wasted pixels.
This isolates resolution selection from neural computation.

### 3.2 Tile geometry

Given a selected resolution, verify overview is first; tile count is exactly
`(selectedW / imageSize) × (selectedH / imageSize)`; tiles are row-major; top-left precedes top-right;
first-row tiles precede second-row tiles; and no tile is duplicated or omitted. Use a synthetic source
image with visually distinct quadrants so tile swaps are obvious.

## 4. Phase 2 — Verify exact resize/pad semantics

The current implementation deliberately differs between:

- Overview: `ResizeBilinear(..., imageSize, imageSize)` with `PAD_NONE`.
- Refined/tiled image: `ResizePadCeil(..., bestW, bestH, ResizeBicubic, black)` followed by tile
  extraction.

Verify both independently against llama.cpp's `llava_uhd` preprocessing. Check scale calculation,
aspect preservation, integer rounding, padded-region size, black padding values, bicubic versus
bilinear selection, channel ordering, and normalization timing. Produce deterministic pixel/CHW
comparisons before involving the ViT.

## 5. Phase 3 — Verify the view-ordering contract

The current implementation produces overview, then tile `(row 0, col 0)`, `(row 0, col 1)`, and so on
in row-major order. Confirm this is the LLaVA-NeXT `mtmd` contract for the selected checkpoint; do
not infer ordering from Granite. Use `llama-mtmd-debug` output, llama.cpp source, or a reference image
whose tiles have unmistakably different content. Record the exact per-view sequence. A useful
diagnostic image has four distinct large quadrant patterns so swapping tiles changes encoder inputs
visibly.

## 6. Phase 4 — Verify whether separators exist

Treat separator behavior as an explicit question. The current LLaVA preprocessor documents
`MTMD_SLICE_TMPL_NONE` and concatenates overview/tile embeddings without marker tokens. Determine
from the actual checkpoint/reference whether the multimodal sequence contains no separator, a
newline/separator embedding, a learned view separator, or another per-view marker.

Verify:

`number of views × tokens per view + separator tokens = actual injected soft-token count`

Reproduce the reference exactly. Do not add a separator merely because another LLaVA-style model
uses one.

## 7. Phase 5 — Per-view vision-encoder parity

Run each individual view through `LlavaVisionEncoder` and compare against llama.cpp. For every view
record view index, width/height, patch grid, token count, embedding sum, row-0 values, and preferably
per-token cosine against the reference. Expected sequence is overview, tile 0, tile 1, and so on. The
existing LLaVA-1.5 golden test remains the encoder control. If every view matches independently but
the complete AnyRes image fails, investigate view composition/order/prompt insertion rather than the
ViT first.

## 8. Phase 6 — Complete AnyRes embedding parity

Construct the full soft-token sequence exactly as the adapter does and compare C# overview/tile
embeddings with llama.cpp's reference view sequence. Verify identical view count, token count per
view, concatenation order, and no missing/duplicated views or erroneous/missing separators. Add a
permanent regression test for complete AnyRes embedding output using a real checkpoint and genuinely
tiled image.

## 9. Phase 7 — End-to-end image-prompt parity

Run the real LLaVA-NeXT checkpoint with `stingray ... --image image.png -p "..."` and compare against
`llama-server --mmproj ...` or `llama-mtmd-cli ...`. Use an image forcing multiple views. Record
original prompt, image placeholder handling, view count, soft-token count, final expanded token count,
first/last image-token positions, first following text-token position, generated token IDs, and decoded
output. First establish that the same multimodal sequence reaches the decoder; prose similarity is
not the initial acceptance signal.

## 10. Phase 8 — Two-image/asymmetric-image controls

After one real AnyRes image works, add adversarial controls:

- Wide image: forces horizontal multi-tile layout.
- Tall image: forces vertical multi-tile layout.
- Distinct quadrants: every tile has visibly different content.
- Small image: requires no AnyRes and should use the single overview path.

These distinguish resolution-selection, tile-count, row/column-order, padding, and single/multi-view
transition errors.

## 11. Phase 9 — Compare token counts against llama.cpp

For at least one real LLaVA-NeXT image, reproduce the exact multimodal token count:

`source image → overview → refined tiles → views → tokens/view → AnyRes image token count → prompt expansion`

The C# count must equal llama.cpp's. A wrong image-token count can yield finite, plausible text while
the model receives the wrong sequence.

## 12. Phase 10 — Diagnose mismatches by stage

If the real-weight run does not match, classify in this order:

1. selected pinpoint differs;
2. overview pixels differ;
3. padded/refined image differs;
4. tile boundaries differ;
5. tile order differs;
6. per-view preprocessing differs;
7. per-view encoder differs;
8. concatenation differs;
9. separator/view-marker semantics differ;
10. prompt expansion differs;
11. decoder output differs.

Do not immediately modify `LlavaVisionEncoder`: the LLaVA-1.5 test is strong existing evidence for the
encoder.

## 13. Phase 11 — Real-weight regression test

Add a dedicated real-weight AnyRes test, for example
`LlamaMtmdVisionParityTests.LlavaNext_AnyRes_Rainbow...`. Pin at least selected resolution, view
count, token count, total embedding length, embedding sum, first-row values, and an end-to-end image
answer where practical. Keep the existing LLaVA-1.5 test untouched as the single-tile control. The
regression must prevent silent fallback to single-tile-only behavior.

## 14. Phase 12 — Only then consider code changes

- **Existing implementation is correct:** make no unnecessary code changes; update item 15 to verified
  and record real-weight evidence.
- **Preprocessing is wrong:** fix only the smallest incorrect stage, then rerun preprocessing parity,
  per-view encoder parity, combined embedding parity, and end-to-end image.
- **Composition/prompt semantics are wrong:** fix sequence construction, separator handling, or token
  expansion without touching the proven encoder.

Evidence must identify the first divergent stage before implementation changes.

## Success criteria

Item 15 is complete when a real LLaVA-NeXT checkpoint exercises `llava_uhd`; actual
`image_grid_pinpoints` are read and reproduced; best-resolution selection, overview preprocessing,
refined-image padding/resampling, tile extraction and ordering match llama.cpp; separator/view-marker
semantics are verified; each view has sane encoder parity; combined AnyRes embedding count/order and
expanded prompt image-token count match llama.cpp; the real image yields the expected
coherent/reference-consistent result; a permanent regression prevents return to single-tile-only
behavior; and LLaVA-1.5 remains green as the single-tile control.

## Key rule

**Do not treat "Granite AnyRes works" as proof that LLaVA-NeXT AnyRes works.** Granite proves the
general `llava_uhd` machinery can work for one model family. Item 15 specifically proves the
LLaVA-NeXT/OneVision checkpoint's metadata, preprocessing, view composition, and multimodal sequence
semantics.

Preferred debugging order:

`real checkpoint → metadata → views → pixels → per-view embeddings → combined soft tokens → prompt token sequence → decoder output`

Do not change the established LLaVA-1.5 path while doing this investigation.
