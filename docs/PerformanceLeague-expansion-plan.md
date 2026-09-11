# PerformanceLeague expansion plan — README coverage claims

**Why this exists:** `README.md`'s "What actually works today" matrix and feature list claim
coverage for dozens of models across LLM/vision/ASR/TTS/diffusion that `PerformanceLeague.md` has
never measured. This plan closes that gap systematically — every claimed-working model gets either
a real measurement or an explicit, dated reason it can't be measured yet (missing checkpoint,
unsupported quant, etc.).

**Hard rule carried over from the original backfill plan, re-stated because it was violated once
this session and corrected by the user:** all timing/benchmark runs happen **strictly serially**,
one process at a time, on this single machine — concurrent `stingray.exe`/`llama-bench.exe`/
`audiocpp_cli.exe` processes corrupt every number via CPU contention. Bug **investigation/fixing**
(code reading, root-causing, writing fixes, building) may be parallelized across subagents per the
user's explicit override of the project's normal no-subagents rule, for the rest of this session
only — but never two things that both do timed inference at once.

**Downloads:** explicit standing authorization to download more checkpoints into `models/_models/`
(F: drive, hundreds of GB free) as needed to close gaps below.

---

## Phase 1 — Vision architectures (biggest real gap)

`PerformanceLeague.md` has exactly one vision row (an internal scalar-vs-SIMD kernel micro-bench)
plus 4 VLM *text backbones* tested in text-only mode (InternVL3-2B, Granite-4.0-3B-Vision,
Granite-Vision-3.2-2B, dots.ocr) — **zero real vision-encode-path measurements exist**. README
claims 17+ architectures implemented, 11 with real-weight golden-verification (correctness, not
speed). This phase adds real `--image --mmproj` timing for whichever of those 11+ have both a
checkpoint AND an mmproj file already on this machine.

Checkpoints + mmproj already present in `models/_models/` (checked 2026-09-11):

- [x] InternVL3-2B + `mmproj-internvl3-2b-q8_0.gguf` — DONE 2026-09-11. Real vision-encode
      measured on both OT and llama-mtmd-cli (C++); numbers are not directly comparable (OT gives
      combined prefill+decode t/s, llama-mtmd-cli only prints vision-encoder-only ms) — caveat
      written up in PerformanceLeague.md's new "Vision-Language Model Real Image Encoding" section.
- [x] Granite-4.0-3B-Vision + `mmproj-granite-4.0-3b-vision-f16.gguf` — DONE 2026-09-11 (CPU only).
      Real timing measured but found a NEW bug: degenerate non-image-grounded output (see
      current-work.md). Logged, not blocking further phase 1 work — moving on.
- [x] Granite-Vision-3.2-2B + `mmproj-granite-vision-3.2-2b-f16.gguf` — DONE 2026-09-11. Same
      degenerate-output bug as Granite-4.0-3B-Vision — strengthens the "shared Granite-family
      vision bug" hypothesis in current-work.md. Moving on.
- [x] dots.ocr + `mmproj-dots.ocr-Q8_0.gguf` — DONE 2026-09-11. Vision encoder runs but decode
      emits 1-token degenerate `<|endofassistant|>` output — logged as a real gap, likely
      prompt-format mismatch for this OCR-specialized checkpoint rather than the Granite bug class.
- [x] Gemma-3-4B-it + `mmproj-gemma-3-4b-it-f16.gguf` — DONE 2026-09-11. Real, WORKING vision-encode
      (2nd confirmed-working checkpoint alongside InternVL3-2B) — genuinely describes the image
      content correctly across all 3 best-of-3 runs. Also found an unrelated Jinja chat-template
      gap for this checkpoint (logged, didn't affect this measurement).
- [x] Kimi-VL-A3B-thinking + `mmproj-kimi-vl-a3b-thinking-Q8_0.gguf` — DONE 2026-09-11. Crashes:
      `Missing tensor: blk.0.attn_q.weight` (deepseek2 MLA-tensor gap, logged).
- [x] MiMo-VL-7B-sft + `mmproj-mimo-vl-7b-sft-Q8_0.gguf` — DONE 2026-09-11, crash FIXED (turned
      into a clean error). Root cause: a genuine 3584-vs-4096-dim mmproj/text-backbone mismatch,
      an upstream conversion defect not fixable here — but `RunImagePrompt` no longer crashes on
      it, reporting the exact mismatch instead.
- [x] Step3-VL-10B + `mmproj-step3-vl-10b-F16.gguf` — DONE 2026-09-11, crash FIXED for real. Root
      cause: `Step3VlVisionEncoder` looked up the wrong GGUF tensor name for the final projector
      (`mm.model_proj.weight` vs the real `mm.model.fc.weight`). Fixed and verified — now runs to
      completion with real timing (10.0/9.9 t/s prefill/decode).
- [x] YouTu-VL-4B + `mmproj-youtu-vl-4b-BF16.gguf` — DONE 2026-09-11. Same
      `Missing tensor: blk.0.attn_q.weight` crash as Kimi-VL-A3B-thinking.
- [x] Nemotron-Nano-12B-v2-VL + `mmproj-nemotron-nano-12b-v2-vl-bf16.gguf` — DONE 2026-09-11.
      Crashes: `HybridGdnForwardPass dense FFN requires hp.IntermediateDim > 0`, logged.
- [x] DeepSeek-OCR-2 + `mmproj-deepseek-ocr-2-q8_0.gguf` — DONE 2026-09-11. Runs, real timing
      (41.9/30.9 t/s), garbled output as expected for unverified arch.
- [x] PaddleOCR-VL-1.6 + `mmproj-paddleocr-vl-1.6.gguf` — DONE 2026-09-11. Runs, real timing
      (51.5/39.2 t/s), degenerate output as expected for unverified arch.

Need a real image to feed `--image` — use an existing repo asset (check `docs/diffusion-samples/`
for something real and non-sensitive, e.g. the Z-Image-Turbo or Wan2.1 sample PNGs already
generated this session) rather than downloading anything new for this.

Checkpoints claimed in README but **not present locally** — download only if time allows after the
above (lower priority, since coverage of what's already downloaded matters more than breadth):
- [ ] Pixtral 12B, LLaVA-1.5/NeXT/OneVision, MiniCPM-V 2.6, GLM-4V/4.5V/OCR, Exaone 4.5-VL,
  Hunyuan-VL, Llama 4 Scout's vision path (mmproj already present:
  `mmproj-llama-4-scout-17b-16e-instruct-f16.gguf`, but the *text* checkpoint was explicitly
  cancelled this session — 93GB, doesn't fit in 64GB RAM; the vision-only mmproj path might still
  be small enough to try independently, check its size first)

## Phase 2 — ASR gaps

- [x] **FunASR Paraformer** — DONE 2026-09-11. Confirmed broken via direct real test execution
      (not guessed): `paraformer-q8.gguf` genuinely lacks `pf.vocab` GGUF metadata. Not a quick
      fix (needs a fresh correct GGUF conversion) — logged as a real gap in current-work.md.
- [x] **SenseVoice** — DONE 2026-09-11. Confirmed genuinely unwired: only a doc-comment mention
      in `FunAsrPipeline.cs`, no dedicated model spec/config/code path. README's prose overstates
      this; logged as a real gap.
- [x] **Parakeet TDT** — DONE 2026-09-11. Confirmed CTC-only: `Parakeet/` directory has no
      TDT-specific decoder file. README's "CTC/TDT" phrasing overstates coverage; logged as a
      real gap.

## Phase 3 — Diffusion / video (large, low-priority given per-run cost)

Given Z-Image-Turbo took ~14.5 min and Wan2.1 (2 frames) took ~71 min this session, treat this
phase as opportunistic, not a commitment — each of these could be a multi-hour single run.

- [x] LTX-Video (`ltx-video-2b-v0.9.1.safetensors` present) — DONE 2026-09-11. Real run, real
      181KB PNG output, 100.5s at 256×256/1-frame/25-steps — first timing number ever recorded
      for this checkpoint. Much faster than Z-Image-Turbo/Wan2.1 (smaller 2B DiT, single frame).
- [ ] FLUX.1-schnell — checked 2026-09-11: no checkpoint present in `models/_models/`. Would need
      a fresh download; deferred (disk-space-at-a-time discipline, lower priority than closing
      gaps on what's already local).
- [x] SD3/3.5 — DONE 2026-09-11. Backfilled the real 656.9s/20-step/256×256 number (the doc's own
      most-recent same-config measurement) with an explicit caveat that it predates 4 real
      correctness bugs fixed 2026-09-05 and no fresh post-fix timing exists.
- [x] MiniMax-Music3 — DONE 2026-09-11. Backfilled the real 3352.9s (~56min)/200-frame number
      from current-work.md/docs/066 (corrected from this plan's earlier "582.9s" — the real
      sourced number is 3352.9s, post-fix for a frame-index off-by-one bug).
- [ ] HunyuanVideo — README explicitly says 🔴, blocked on a missing VAE decoder — do not attempt,
      this isn't a "run it and see" gap, it's a known incomplete port

## Phase 4 — Bug fixes (parallelizable via subagents, NOT benchmarking)

- [x] **Granite Vulkan correctness bug — FIXED 2026-09-11** — root cause found this session (subagent investigation):
      `GpuForwardPass.cs`'s `RunStandardLayers` never threads `AttentionScaleOverride`,
      `ResidualScale`, or `LogitScale` into the Vulkan dispatch path, while the CPU path
      (`ForwardPass.Decode.cs`/`PrefillCore.cs`/`Attention.cs`) applies all three. Exact file/line
      references already gathered. Next: implement the fix (a subagent can do this — reading+
      writing code, not running timed inference), then verify via a real CPU-vs-Vulkan comparison
      run (that verification run itself must be serial, done directly, not by a subagent).
- [x] **DeepSeek-V2-Lite** — DONE 2026-09-11. Ran with `--allow-unverified-arch`, confirmed the
      documented garbled output matches README's own finding exactly, got real throughput numbers
      anyway (0.92x near-parity decode). Added to PerformanceLeague.md's new "DeepSeek family" section.
- [ ] **F5-TTS's blocked CPU backend in `audio.cpp`** — real, scoped, from the original backfill
      pass. Not attempted yet this session.
- [x] **Chatterbox Turbo's missing tokenizer asset** — FIXED 2026-09-11. Extracted the real GGUF
      tokenizer metadata into the 3 sidecar files the loader needs, verified end-to-end with a
      real WAV output. Turns out "streaming" was never the real blocker — this build has no
      streaming mode for Chatterbox Turbo at all (deliberate offline-only design limit, same as
      QwenTTS/CosyVoice3), so the vocab fix alone doesn't unlock a TTFA comparison — that would
      need real streaming support added to `audio.cpp` itself, out of scope here.
- [ ] **`stingray embed`'s missing real tokenizer** for the ONNX path (char-per-token placeholder,
      not real WordPiece/BPE) — fixed this session to fail gracefully, but the underlying issue (no
      real tokenizer wired) is still open. Lower priority than the other bugs — the graceful
      degradation already makes this safe to use for short inputs.
- [ ] **`stingray embed`'s fake GGUF stub** — the big one, requires wiring a real forward pass into
      `EmbeddingEngine`. Scoped as a real, larger task, not attempted this session beyond
      documentation.

---

*Companion to `docs/PerformanceLeague-backfill-plan.md` (the original CPU/Vulkan LLM+TTS+ASR sweep,
now essentially complete) — this doc covers the README-claims-driven expansion phase.*
