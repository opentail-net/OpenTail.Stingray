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
      architecture until that separate gap is closed.
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
- [ ] **DotsOcr** — no local mmproj. Both files needed. DotsOCR is a relatively small (~3B) model
      family per its public release — good candidate despite needing both files.
- [ ] **Granite4** — no local mmproj. Both files needed. IBM Granite 4 Vision — check smallest
      available variant.
- [ ] **Nemotron** — no local mmproj. Both files needed. Check smallest NVIDIA Nemotron-VL variant.
- [ ] **MobileNetV5** — no local mmproj. Both files needed. Named for a MobileNet-scale vision tower
      — likely genuinely small if a GGUF conversion exists at all; verify one is actually published
      before committing to it.

## Cleanup (only after every row above is checked off)

- [ ] Delete `VisionOps.GetTensorPtr<T>` and `VisionOps.MatVecF16` entirely — no caller left once
      migration is complete. Do not delete early "just to be tidy"; every unmigrated encoder still
      depends on them being present and dtype-guarded.
- [ ] Sweep for any other direct `Half*`/blind-cast pattern this list missed (the `HunyuanVlVisionEncoder`
      false-positive during discovery — it has its own *local* copy of the same pattern instead of
      calling `VisionOps.GetTensorPtr` directly, but turned out to have zero `Half` usage in practice;
      worth a final grep pass for other private duplicates before deleting the shared method).

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
