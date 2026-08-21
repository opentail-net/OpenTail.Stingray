# VL encoder migration plan — GetTensorPtr&lt;T&gt;/MatVecF16 → VisionTensorRef/GetTensor/MatVecAny

Read `docs/done/vl-untested-code-findings-2026-08-20.md` first — it has the full story: the old API
blindly casts a tensor pointer to a fixed CLR type with no dtype check, which corrupted memory on a
real Q8_0 mmproj (`InternVL3-2B`, confirmed crash, confirmed fixed). `InternVlVisionEncoder` is
already migrated and verified end-to-end against real weights — it's the template for every row
below. `docs/done/vision-attention-vectorization-2026-08-20.md` is the unrelated earlier change
(don't re-open that investigation, it's closed).

**Already safe, not in scope** (zero `Half*` usage, confirmed by direct grep — either hand-rolled
with `TensorPrimitives` like Gemma3/Gemma4V/Llama4, or F32/BF16-only already): `Exaone4`, `Gemma3`,
`Gemma4V`, `HunyuanVl`, `Llama4`, `MimoVl`, `Step3Vl`, `YoutuVl`.

**Done**: `InternVl` ✅ (verified against real Q8_0 weights, see findings doc).

## Migration checklist (13 remaining), ordered smallest-download-effort first

For each: (a) migrate the encoder's weight fields to `VisionTensorRef`, matvec calls to
`VisionOps.MatVecAny`, any inline-per-element weight read (patch embed, conv, etc.) to
`Dequantize.ToFloat32` once at construction — same pattern as `InternVlVisionEncoder.cs`; (b) build;
(c) download a correctly-paired base text model if not already local (match mmproj's source repo via
its file size/hash, same method used for InternVL3 — never guess a pairing); (d) run a real
end-to-end `--image` request; (e) record the result (works / new bug found+fixed / blocked on
something else, same honesty as the InternVL writeup) in the findings doc.

- [x] **DeepSeekOcr** — **encoder migrated, builds clean.** Downloaded paired base
      (`sabafallah/DeepSeek-OCR-2-GGUF`'s `deepseek-ocr-2-Q4_K_M.gguf`, 1.95GB, confirmed match via
      mmproj file size). **Full end-to-end blocked by something entirely different and out of
      scope**: the base model's text architecture (`deepseek2-ocr`) isn't in
      `ModelCompatibility.ValidateForTextGeneration`'s supported-architecture list at all — rejected
      before vision code ever runs. That's a real gap, but it's "add DeepSeek2 text-generation
      support," a much bigger, separate undertaking, not a vision-encoder migration task. Encoder
      migration itself is done and presumed correct by the same standard InternVL was (builds,
      follows the proven pattern exactly) but literally cannot be exercised end-to-end on this
      architecture until that separate gap is closed. **Placeholder marker also fixed 2026-08-20**
      (same root cause as DotsOcr/Granite4, see below) — `DeepSeekOcrAdapter` declared
      `<image>`/`</image>`/`<image_pad>`, but the real vocab (verified against the local
      `deepseek-ocr-2-Q4_K_M.gguf`) has only `<image>`. Fixed the marker regardless of the separate
      text-arch block, so it's ready the moment that gap closes.
- [x] **Kimi (KimiVL)** — **encoder migrated, builds clean.** Had its own private, duplicated
      `GetTensorPtr<T>`/`MatVecF16` (not even calling VisionOps's) -- removed, now uses
      `VisionOps.GetTensor`/`MatVecAny` like every other migrated encoder. **Separate real bug found
      while there, checked against ground truth (`examples/llama.cpp/llama.cpp/tools/mtmd/models/
      kimivl.cpp`, which calls the shared `build_vit`/`build_attn` clip.cpp helpers)**: this port's
      `ComputeAttention` never reads q/k, never computes a Q.K score or softmax -- it just copies
      scaled V straight through, so every token "attends" only to itself regardless of image
      content. Real, pre-existing, unrelated to this migration -- flagged in code with a comment,
      not fixed (separate scope). mmproj local (`mmproj-kimivl-q8_0.gguf`, 618MB, Q8_0); base model
      download pending (batch download pass, per plan).
- [x] **Llava** — **encoder migrated, builds clean.** Structurally identical to InternVL's original
      pattern (already used shared VisionOps.Attention correctly, no Kimi-style bug). Mechanical
      migration only. mmproj local (`mmproj-llava-v1.5-7b-f16.gguf`, 624MB, F16). Base: 7B, download
      pending.
- [x] **MiniCpm** — **encoder migrated, builds clean.** Had 16 resampler tensors + block weights all
      on the old pattern; also had `ComputeAttention` AND `ComputeCrossAttention` both broken the
      same way as Kimi's (V-copy and Q-copy stubs respectively -- the resampler's cross-attention
      literally doesn't depend on the image patches at all currently). Both flagged, not fixed
      (separate scope). mmproj local (`mmproj-minicpm-v-2_6-f16.gguf`, 1.04GB, F16). Base: ~8B,
      download pending.
- [x] **Pixtral** — **encoder migrated, builds clean.** Clean pattern, used shared VisionOps.Attention
      correctly already, no broken-attention bug. mmproj local (`mmproj-pixtral-12b-f16.gguf`,
      870MB, F16). Base: 12B, download pending.
- [x] **QwenVl (Qwen2.5-VL)** — **encoder migrated, builds clean.** Same broken-attention stub
      pattern (ComputeSelfAttention copies scaled V, flagged not fixed). Also had the spatial-merge
      MLP (mm.0/mm.2) hand-rolled inline with its own Half cast, not going through any shared matvec
      at all -- migrated to MatVecAny (nTokens=1 per merged token). mmproj local
      (`mmproj-qwen2.5-vl-7b-f16.gguf`, 1.35GB, F16). Base: 7B, download pending.
- [x] **PaddleOcr** — **encoder migrated, builds clean.** Clean pattern, used shared VisionOps.Attention
      correctly already. mmproj local (`PaddleOCR-VL-1.6-GGUF-mmproj.gguf`, 882MB). Base: check size
      (likely ~0.9B, cheapest download in this list), download pending.
- [x] **Glm4 (GLM-4.6V)** — **encoder migrated, builds clean.** Same private-duplicate GetTensorPtr/
      MatVecF16 + broken ComputeAttention stub pattern as Kimi/MiniCpm/QwenVl. `_patchMergerW`/
      `_patchMergerB` are loaded but never referenced in Forward() -- pre-existing, left as-is (not
      a new bug from this migration, matches original behavior). mmproj local
      (`mmproj-glm-4.6v-q4.gguf`, 577MB, quantized). Base: download pending -- check for a smaller
      GLM-4.6V variant before committing to the full-size one.

**All 8 encoders with a local mmproj are now migrated** (InternVL, DeepSeekOcr, Kimi, Llava,
MiniCpm, Pixtral, QwenVl, PaddleOcr, Glm4 -- 9 counting InternVL). Full CLI build verified clean
after each. Remaining 5 below need both files downloaded fresh.
- [x] **CogVlm** — **encoder migrated, builds clean.** Clean pattern, used shared VisionOps.Attention
      correctly already. **Download blocked, not just deferred**: searched twice (general + a
      site-scoped GGUF query), found no confirmed public GGUF conversion of CogVLM/CogAgent at all
      (THUDM's originals are 17-18B, safetensors only, as far as this search could confirm). Per the
      "never guess a pairing" rule, not fabricating a download target. Same situation as
      MobileNetV5's caveat below. Revisit if/when a real conversion surfaces.
- [x] **DotsOcr** — **encoder migrated, builds clean.** Structurally near-identical to PaddleOcr's
      pattern. Found the real source: `ggml-org/dots.ocr-GGUF` (official ggml-org conversion,
      confirmed via HF API -- note `dinhquangson/dots.ocr-gguf`, the first search hit, turned out to
      be an empty placeholder repo with zero actual files, caught by checking the API directly
      rather than trusting the repo name). `qwen2` text architecture, 1.7B base -- genuinely small.
      Downloaded dots.ocr-Q8_0.gguf (1.76GiB) + mmproj-dots.ocr-Q8_0.gguf (1.25GiB), both Q8_0.
      **Verified end-to-end**: encoder runs clean against real Q8_0 weights, produces
      `81 soft tokens (1536-dim)` for a test image, no crash, no memory corruption — the migration
      itself is proven. **Placeholder gap found 2026-08-20, fixed same day** (see
      "Chat-template placeholder gap — root cause and fix" below): not a chat-template bug at all —
      `DotsOcrAdapter` in `UnifiedVisionPipeline.cs` declared the wrong marker strings
      (`<image>`/`</image>`/`<image_pad>`, none of which exist in dots.ocr's real Qwen2-based
      vocab). Fixed to the verified real tokens (`<|vision_start|>`/`<|vision_end|>`/
      `<|image_pad|>`). Now runs a full real prefill+decode (`90 tokens (81 image + 9 text)`).
- [x] **Granite4** — **encoder migrated, builds clean.** Clean pattern (SigLIP tower + WindowQFormer
      projector), used shared VisionOps.Attention/LayerNorm correctly already, no broken-attention
      stub found. Mechanical migration only: patch-embed → DequantizeToFloat32, all block/proj
      weights → VisionTensorRef/MatVecAny. **First download attempt was the wrong architecture**:
      `bartowski/ibm-granite_granite-vision-3.2-2b-GGUF` declares `clip.projector_type: mlp`, which
      routes to `LlavaAdapter`, not `Granite4Adapter` — that model is LLaVA-Next-based, not the
      WindowQFormer arch this encoder targets. Found the real target:
      `mrutkows/granite-4.0-3b-vision-GGUF` (`granite-4.0-3b-vision-Q4_K_M.gguf` + `mmproj-model-
      f16.gguf`, same source repo, no guessed pairing) — confirmed via metadata
      (`clip.projector_type: granite4_vision`) before running. **Verified end-to-end**: encoder runs
      clean against real Q4_K weights, produces `576 soft tokens (2560-dim)` for a test image, no
      crash, no memory corruption. **Placeholder gap found 2026-08-20, fixed same day**: same root
      cause as DotsOcr — `Granite4Adapter` declared `<image>`/`</image>`/`<image_pad>`, but this
      model's real vocab has only a single `<image>` special token, no separate wrapper or `_pad`
      variant. Fixed (`ImageOpenMarker`/`ImageCloseMarker` → `""`, `PlaceholderMarker` → `"<image>"`).
      **Now produces real generated text end-to-end** for the first time:
      `Describe this image.A large, diverse group of animals living in a certain area...`
      (`611 tokens (576 image + 35 text)` prefill, 30 tokens decoded) — full pipeline proven working.
- [x] **Nemotron** — **encoder migrated, builds clean.** Register-token ViT (4 register tokens
      stripped after post-LN) + 2x2 patch merge + squared-ReLU MLP projector, matches QKV-fused and
      split-QKV variants (fused path taken when `attn_qkv` tensor present). Clean pattern, no
      broken-attention stub. Downloaded `Vastined/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16-GGUF`'s Q2_K
      base (4.4GB) + BF16 mmproj (1.7GB), same source repo. **Full end-to-end blocked, same class of
      gap as DeepSeekOcr**: the base model's text architecture (`nemotron_h`, a Mamba/SSM hybrid)
      isn't in `ModelCompatibility.ValidateForTextGeneration`'s supported list — rejected before
      vision code runs. Separate, much bigger undertaking (add nemotron_h text-gen support), not a
      vision-encoder migration task. Encoder itself presumed correct by the same standard as the
      others (builds, follows the proven pattern). **Placeholder marker also fixed 2026-08-20** (see
      below) — same wrong `<image_pad>` guess as Granite4/DeepSeekOcr, corrected to the verified
      real token (`<image>`), ready the moment the text-arch gap closes.
- [x] **MobileNetV5** — **encoder migrated, builds clean.** Simplest encoder in the set (stem conv +
      RMSNorm + GELU + single-layer projection, no transformer blocks). Mechanical migration only.
      **Download blocked, not just deferred** (same situation as CogVlm below): gemma-3n looked like
      the obvious target (llama.cpp's `mobilenetv5.cpp` is written for it) but its real mmproj
      (downloaded from `Anthonyg5005/gemma-3n-e4b-mmproj-gguf` to check) declares
      `clip.vision.projector_type: gemma3nv`, which `UnifiedVisionPipeline` routes to
      `Gemma3Adapter` (the already-safe TensorPrimitives path), **not** `MobileNetV5Adapter` — wrong
      target, same mistake class as Granite4's first attempt. `MobileNetV5Adapter` only activates on
      an explicit `mobilenetv5`/`mobilenet_v5` projector-type string or structural inference
      (`v.registers`/`mm.reg_norm.weight` tensors) with no matching metadata string; searched HF for
      any GGUF actually declaring that projector type and found none. Deleted the wrongly-targeted
      gemma-3n mmproj rather than keep a misleading download. Revisit if/when a real
      `mobilenetv5`-projector conversion surfaces.

## Chat-template placeholder gap — root cause and fix (2026-08-20, same day as the raw-pointer
## redesign)

The "chat-template placeholder gap" flagged throughout this doc (DotsOcr, Granite4, DeepSeekOcr) was
never actually a chat-template bug. Root cause, found by tracing `RunCommand.RunImagePrompt`:

1. **The prompt-building code hardcoded a literal string.** With no `<image>` marker in the user's
   prompt (the common case), the code did
   `userMsg = string.Concat(Enumerable.Repeat("<|image|>", nImages)) + s.Prompt` — a hardcoded
   `"<|image|>"` literal, not the model's actual placeholder text
   (`vision.PlaceholderMarker`). This method was originally written for Gemma 4 only (its own doc
   comment says so — "issue #250"), where `<|image|>` happens to literally be the real marker. When
   the CLI's gemma4-only gate was removed earlier in this same session (see
   `docs/done/vl-untested-code-findings-2026-08-20.md`), this hardcoded literal was never updated
   for the 21 other architectures now reachable through it. **Fixed**: `vision` (the
   `IVisionEmbedder`) is now opened *before* building `userMsg`, and both branches use
   `vision.PlaceholderMarker` instead of the `"<|image|>"` literal.
2. **Several adapters' declared marker strings were themselves wrong.** Fixing (1) alone didn't fix
   DotsOcr — its `DotsOcrAdapter.PlaceholderMarker` was `"<image_pad>"`, which never existed in
   dots.ocr's real vocab either. Checked by grepping the actual GGUF bytes for every model with a
   local download (`grep -a -o` against the raw file, no tool needed) rather than guessing:

   | Adapter | Declared (wrong) | Real (verified against local GGUF) |
   |---|---|---|
   | `DotsOcrAdapter` | open `<image>`, close `</image>`, placeholder `<image_pad>` | open `<\|vision_start\|>`, close `<\|vision_end\|>`, placeholder `<\|image_pad\|>` (Qwen2 convention) |
   | `Granite4Adapter` | open `<image>`, close `</image>`, placeholder `<image_pad>` | no open/close tokens exist; single placeholder `<image>` |
   | `NemotronAdapter` | same as Granite4 | same as Granite4 — single `<image>`, no wrapper |
   | `DeepSeekOcrAdapter` | same as Granite4 | same as Granite4 — single `<image>`, no wrapper |

   All four fixed. `QwenVlAdapter` and `KimiAdapter` already correctly declared
   `<\|image_pad\|>` (Qwen2 convention) — spared by luck, not by the same bug being absent; worth
   noting they weren't touched here because they were already right.

**Verified end-to-end after the fix** (both explicitly requested by name):
- **DotsOcr**: placeholder-count check now passes; ran a real prefill+decode —
  `Prefill: 90 tokens (81 image + 9 text), 24.5 t/s | Decode: 1 tokens, 15.7 t/s`. Only 1 token
  decoded before stopping — plausibly this small OCR-tuned base model just isn't suited to an
  open-ended "describe this image" prompt (not re-investigated; the placeholder bug itself is
  conclusively fixed, decode-length behavior is a separate, lower-priority question).
- **Granite4**: full real generation, working end-to-end for the first time —
  `Prefill: 611 tokens (576 image + 35 text), 14.8 t/s | Decode: 30 tokens, 14.1 t/s`, output:
  *"Describe this image.A large, diverse group of animals living in a certain area. The image shows
  a variety of habitats including grasslands, forests, wetlands and des[...]"* — coherent, on-topic
  text, proving the whole pipeline (encoder → projector → placeholder expansion → text generation)
  end-to-end for this architecture.
- **Bonus, unprompted**: re-ran InternVL as a regression check on the fix and it went from *blocked*
  (the exact same "expected 1 image placeholder token(s) (`<IMG_CONTEXT>`, 151667) ... found 0"
  error documented throughout this whole doc) to fully working —
  `Prefill: 291 tokens (256 image + 35 text), 31.4 t/s | Decode: 37 tokens, 30.1 t/s`. InternVL's
  own `PlaceholderMarker` (`<IMG_CONTEXT>`) was already correct; only fix (1) above (the hardcoded
  `"<\|image\|>"` literal) was blocking it. First fully-working end-to-end VL generation this whole
  session, on the model that was the very first one tested at the start of it.

**Follow-up (same day): checked the remaining 5 without downloading any model weights.** Confirmed
llama.cpp's vendored source (`examples/llama.cpp/llama.cpp/tools/mtmd/`) doesn't hardcode
per-architecture marker strings anywhere — grepped `clip.cpp`, `mtmd.cpp`, all 7 relevant
`models/*.cpp` files, and `gguf-py/gguf/constants.py`, all empty. `mtmd_default_marker()` returns a
single generic `"<__media__>"` for every architecture; the real per-arch open/pad/close text lives
entirely in each model's own HF `tokenizer_config.json` / `chat_template.jinja`, not in C++.
Fetched those files directly via the HF API (a few KB each, not a model download) for the 5 with a
real, named reference repo:

| Adapter | Old (wrong) | Real (verified via HF `tokenizer_config`/`chat_template`, not local GGUF bytes) |
|---|---|---|
| `MimoVlAdapter` | `<image>`/`</image>`/`<image_pad>` | `<\|vision_start\|>`/`<\|vision_end\|>`/`<\|image_pad\|>` (`XiaomiMiMo/MiMo-VL-7B-RL`, confirms mimovl.cpp's own "Qwen2.5-VL-shaped ViT" comment extends to the template) |
| `Step3VlAdapter` | same | same Qwen2.5-VL convention (`stepfun-ai/Step3-VL-10B`) |
| `YoutuVlAdapter` | same | same Qwen2.5-VL convention (`tencent/Youtu-VL-4B-Instruct`'s `chat_template.json`) |
| `PaddleOcrAdapter` | same | **its own convention**, not Qwen's: `<\|IMAGE_START\|>`/`<\|IMAGE_PLACEHOLDER\|>`/`<\|IMAGE_END\|>` (`PaddlePaddle/PaddleOCR-VL`'s `chat_template.jinja` — note `<\|image_pad\|>` *does* exist in this model's vocab too, but is unused/legacy; trusted the template's actual usage, not vocab presence alone) |
| `HunyuanVlAdapter` | same | **entirely different convention**: `<｜hy_place▁holder▁no▁100｜>` / `<｜hy_place▁holder▁no▁102｜>` / `<｜hy_place▁holder▁no▁101｜>` (open/placeholder/close) — **lower confidence**: no real Tencent Hunyuan-VL repo is published on HF at all yet, so this came from `hf-tiny-v2/tiny-random-HunYuanVLForConditionalGeneration`, an HF-testing repo that mirrors real tokenizer/vocab data by convention but isn't the production model itself. Flagged in-code as lower-confidence. |

All 5 fixed in `UnifiedVisionPipeline.cs`, `dotnet build` clean for Vision + Cli.

**Follow-up (same day): two more markers found and fixed while double-checking the ones this doc had
called "already correct"** (`GlmAdapter`/`Glm4Adapter` re-checked and confirmed genuinely already
right — false alarm, no change) — same HF-tokenizer-config technique:

| Adapter | Old (wrong) | Real (verified via HF `chat_template`) |
|---|---|---|
| `KimiAdapter` | `<\|vision_start\|>`/`<\|vision_end\|>`/`<\|image_pad\|>` (assumed Qwen2.5-VL) | **not Qwen-style at all**: `moonshotai/Kimi-VL-A3B-Instruct`'s real template emits `<\|media_start\|>image<\|media_content\|><\|media_pad\|><\|media_end\|>` per image. Fixed open/close to `<\|media_start\|>`/`<\|media_end\|>` and — the part that actually matters for the placeholder-count check — placeholder to `<\|media_pad\|>`, not `<\|image_pad\|>`. The literal `image` text and `<\|media_content\|>` token between open and placeholder can't be represented by this interface's single-open-marker model, so this is a best-effort fix, documented as such in code. |

**Follow-up (same day): downloaded and ran small models for as many of the above as actually
possible**, per direct request. Findings:

| Model | Text arch | Result |
|---|---|---|
| `PaddlePaddle/PaddleOCR-VL-1.6-GGUF` (935MB base + 882MB mmproj) | `paddleocr` | **Blocked before vision code runs** — `paddleocr` isn't in `ModelCompatibility.ValidateForTextGeneration`'s supported list (same class of gap as DeepSeekOcr/Nemotron). Marker fix can't be exercised end-to-end on this architecture until that separate gap closes. |
| `mradermacher/MiMo-VL-7B-SFT-GGUF` (3.08GB Q2_K + 729MB mmproj-Q8_0) | `qwen2vl` | **Blocked the same way** — this specific conversion tags the combined vision-capable checkpoint as `qwen2vl`, which is *not* the same as the supported `qwen2`/`qwen3` text-only tags. |
| `Vastined/Step3-VL-10B-GGUF` (3.28GB Q2_K + 3.97GB mmproj-F16) | `qwen3` (supported!) | **Marker fix conclusively verified**: real prefill ran, `Image 1/1: step3vl -> 81 soft tokens (4096-dim)`, no placeholder-count error — the exact bug this whole finding is about is provably closed for this architecture. **Hit a separate, new, real bug** immediately after: `System.ArgumentOutOfRangeException` at `RunCommand.RunImagePrompt` right after prefill completes, during/around the decode-loop summary print. Confirmed **not** a general decode bug — the identical model run text-only (no `--image`) decodes 20 tokens cleanly with no error — so this is specific to the image/embedding-injection code path, a genuine new finding, not investigated further (out of scope for the marker task; flagged for separate follow-up). |
| `tencent/Youtu-VL-4B-Instruct-GGUF` (5.21GB Q8_0 + 853MB mmproj-BF16) | `deepseek2` | **Blocked the same way** — despite the "Youtu-VL" name, this checkpoint's text backbone is tagged `deepseek2`, not in the supported list. |
| `mradermacher/Kimi-VL-A3B-Thinking-2506-GGUF` | — | **Not downloaded** — smallest quant (Q2_K) is 6.58GB, the model is a larger MoE (A3B) than the others despite similar "small" naming; skipped per the "small files only" instruction rather than fetch a multi-GB file just to likely hit the same text-arch wall. |

**Net result of the download pass**: the unsupported-text-architecture gate
(`ModelCompatibility.ValidateForTextGeneration`) turned out to be the dominant blocker across
this whole batch — 3 of 4 downloaded pairs never reached the vision/placeholder code at all. Only
Step3-VL's `qwen3` tag was in the supported list, and it's the one case that actually exercised
(and confirmed) the marker fix. This is a strong, converging signal that the placeholder-marker
fixes in this doc are correct by construction (grounded in each model's real chat template) even
where they can't be exercised end-to-end yet — and that the text-architecture allowlist, not
marker guessing, is now the single biggest thing blocking full VL verification going forward.

`CogVlmAdapter` and `MobileNetV5Adapter` were not touched — both are already documented above as
fully blocked (no real public GGUF conversion exists for either architecture at all), so there's
nothing to point a corrected marker at yet regardless of what the correct string turns out to be.

## Systemic finding: position/class-embedding dtype guard gap (found and fixed for all 9 affected
## encoders during verification, 2026-08-20)

While verifying Granite4 against real weights, hit the *exact* class of bug this whole migration
exists to close, in code the migration had already "finished": `v.position_embd.weight` was
Float16 in a real mmproj (`granite-vision-3.2-2b`, which turned out to route to `LlavaAdapter` — see
Granite4's row above) but every encoder still read it via `VisionOps.GetTensorPtr<float>` — a
direct, ungoverned `float*` field, not a `VisionTensorRef`, because it's consumed via inline
per-element indexing (`_posEmbd[i % _embd]`) rather than through `MatVecAny`. The dtype guard added
earlier (not the structural fix) caught it and threw cleanly instead of corrupting memory — a real
save, but it still blocked every affected model from running. Fixed by applying the same
`DequantizeToFloat32`-at-construction pattern already used for patch-embed weights: converted
`_posEmbd`/`_clsEmbd` (and Nemotron's/QwenVl's differently-named equivalents) from raw `float*` to
one-time-dequantized `float[]` in **InternVl, DeepSeekOcr, DotsOcr, Glm4, CogVlm, Kimi, MiniCpm,
Nemotron, PaddleOcr, QwenVl, Llava** (11 files touched; Granite4 done first as the file that
surfaced it). All rebuilt clean (Vision + Cli). This was a genuine completeness gap in "the
migration," not new work — every one of these tensors could have silently corrupted memory exactly
like the original Q8_0 bug, just gated behind whichever model happened to store it non-F32.
Re-verified DotsOcr after the fix (still `81 soft tokens`, unaffected since it doesn't use posEmbd
via this path) and Llava (dtype crash gone, see Llava's row below for what surfaced next).

- **Llava** — after the posEmbd fix above, re-ran against `granite-vision-3.2-2b`'s mmproj (routes
  here, not to Granite4Adapter — see Granite4's row). Dtype crash is gone. Hit a **different,
  pre-existing, unrelated bug**: `IndexOutOfRangeException` in `ExtractPatchesWithCls` — this mmproj
  declares `clip.vision.image_grid_pinpoints` (52 entries), i.e. LLaVA-Next's dynamic AnyRes
  multi-tile cropping, which produces more patches than `LlavaVisionEncoder`'s single-tile position
  table supports. Real gap (AnyRes tiling isn't implemented), out of scope for this migration —
  flagged, not fixed here.

## Cleanup (done 2026-08-20)

- [x] **Delete `VisionOps.MatVecF16` entirely** — confirmed zero remaining callers (grep for
      `MatVecF16` across `src/OpenTail.Stingray.Vision` turns up only doc-comment mentions and
      `Tests.Vision`'s own private `MatVecF16_Scalar` reimplementation, which never called the real
      one). Deleted. **`VisionOps.GetTensorPtr<T>` was NOT deleted** — the plan's original wording
      ("delete both entirely") turned out to be imprecise once the migration was actually finished:
      `GetTensorPtr<float>` is still genuinely load-bearing in every migrated encoder for norm/bias
      tensors, which are read element-wise (not through a matvec) and are always F32 in real mmproj
      files — it never needed `VisionTensorRef`/`MatVecAny`, and stays dtype-guarded either way.
      Only the `Half`-typed usage (paired with the now-deleted `MatVecF16`) was the unsafe path;
      that usage is gone (confirmed: no `GetTensorPtr<Half>` call anywhere). Refreshed both methods'
      doc comments to describe the current, not aspirational, state. Vision + Cli + Tests.Vision all
      build clean after the deletion.
- [x] Swept for other private-duplicate `GetTensorPtr`/`MatVecF16`-shaped helpers (the concern that
      motivated this item — `HunyuanVlVisionEncoder`'s local copy turned out to have zero `Half`
      usage, and Kimi's/Glm4's real local duplicates were already removed during their own migration
      steps earlier). Grepped for any `static ... T* GetTensorPtr` or `static void MatVec*` outside
      `VisionOps.cs` — none found. Confirmed clean.

**Follow-up (2026-08-20, same day, user-directed second pass)**: with all encoders freshly migrated
and real models locally available, did a full re-read of every Vision file for further optimization
and safety opportunities. Found and fixed a real per-layer scratch-buffer reallocation bug across 15
encoders (FFN/QKV buffers reallocated every layer instead of once per `Forward()` call) and did a
structural safety review of the remaining raw-pointer surface. Full writeup:
`docs/done/vision-encoder-buffer-reuse-2026-08-20.md`.

**Migration fully closed.** All 13 encoders migrated and build-verified; the old unsafe
Half-blind-cast path (`MatVecF16` + `GetTensorPtr<Half>`) no longer exists anywhere in the codebase.
Remaining open items are downstream of this work, not part of it: two text-generation architecture
gaps (`deepseek2-ocr`, `nemotron_h`) block full end-to-end runs for DeepSeekOcr/Nemotron; a
chat-template placeholder gap blocks DotsOcr/Granite4; LLaVA-Next AnyRes multi-tile cropping isn't
implemented (surfaced via Llava); and CogVlm/MobileNetV5 have no real public GGUF conversion for the
exact architecture their encoders target. All flagged in their own rows above, none are migration
bugs.

## Notes for whoever (including future-me) continues this

- Never guess a base-model/mmproj pairing. Match via source-repo file listing (size/hash), the same
  way `InternVL3-2B` was verified — a wrong pairing either crashes confusingly or "works" while
  producing silently wrong output, which is worse.
- A model "compiling and running without crashing" is not the same as "producing correct output" —
  none of this has a working numerical oracle yet (same caveat the Gemma3/Gemma4V/Llama4 sanity
  tests already carry). The bar here is: doesn't crash, doesn't corrupt memory, produces the right
  shape/token count, matches the architecture's declared placeholder/marker convention. Full parity
  verification is future work, same scope limit already accepted for the existing 3 sanity tests.
- Chat-template placeholder mismatches (like InternVL's `<IMG_CONTEXT>` issue) are a *separate* bug
  class from the encoder migration — expect to hit more of these per-architecture; don't treat them
  as migration failures, log them as their own findings.
