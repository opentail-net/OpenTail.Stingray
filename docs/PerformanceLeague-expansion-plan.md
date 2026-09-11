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

- [ ] InternVL3-2B + `mmproj-internvl3-2b-q8_0.gguf` — real vision-encode timing (text backbone
      already benchmarked; add the image path)
- [ ] Granite-4.0-3B-Vision + `mmproj-granite-4.0-3b-vision-f16.gguf` — same, plus this is the
      checkpoint with the confirmed Vulkan correctness bug (Phase 4) — test CPU only until that's
      fixed, to avoid reporting a broken-output timing as if it were valid
- [ ] Granite-Vision-3.2-2B + `mmproj-granite-vision-3.2-2b-f16.gguf` — same caveat as above
- [ ] dots.ocr + `mmproj-dots.ocr-Q8_0.gguf`
- [ ] Gemma-3-4B-it + `mmproj-gemma-3-4b-it-f16.gguf`
- [ ] Kimi-VL-A3B-thinking + `mmproj-kimi-vl-a3b-thinking-Q8_0.gguf` — text backbone rejected as
      unsupported architecture (`deepseek2`) earlier this session; check if `--allow-unverified-arch`
      makes it run (README claims DeepSeek2 family works via that flag) before writing this off
- [ ] MiMo-VL-7B-sft + `mmproj-mimo-vl-7b-sft-Q8_0.gguf` — text backbone rejected as unsupported
      (`qwen2vl`); same check as above
- [ ] Step3-VL-10B + `mmproj-step3-vl-10b-F16.gguf` — untested so far, check architecture support
- [ ] YouTu-VL-4B + `mmproj-youtu-vl-4b-BF16.gguf` — text backbone already confirmed unsupported
      architecture; check if the flag helps
- [ ] Nemotron-Nano-12B-v2-VL + `mmproj-nemotron-nano-12b-v2-vl-bf16.gguf` — untested so far
- [ ] DeepSeek-OCR-2 + `mmproj-deepseek-ocr-2-q8_0.gguf` — untested so far
- [ ] PaddleOCR-VL-1.6 + `mmproj-paddleocr-vl-1.6.gguf` — text backbone confirmed unsupported
      architecture (`paddleocr`); check the flag, else log as a real gap

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

- [ ] **FunASR Paraformer** — checkpoint (`paraformer-q8.gguf`) is confirmed broken (missing
      `pf.vocab` GGUF metadata) per earlier finding. Check if a fresh re-download or a different
      quant fixes this before giving up on it — README claims it as 🟢 working.
- [ ] **SenseVoice** — `sensevoice-small.int8.onnx` present, no wired C# pipeline found in an
      earlier sweep of `src/OpenTail.Stingray.Audio`. Re-confirm this (grep again) — README doesn't
      actually list SenseVoice with a status badge in the matrix (only in the feature-list prose),
      so it may be more aspirational than the matrix's other 🟢 entries. If genuinely unwired, log
      as a real gap rather than force it.
- [ ] **Parakeet TDT** (as opposed to the CTC variant already benchmarked) — README's feature list
      says "Parakeet FastConformer CTC/TDT" — check if a TDT-specific decode path exists in
      `src/OpenTail.Stingray.Audio/Parakeet/` or if only CTC is actually implemented despite the
      README wording.

## Phase 3 — Diffusion / video (large, low-priority given per-run cost)

Given Z-Image-Turbo took ~14.5 min and Wan2.1 (2 frames) took ~71 min this session, treat this
phase as opportunistic, not a commitment — each of these could be a multi-hour single run.

- [ ] LTX-Video (`ltx-video-2b-v0.9.1.safetensors` present) — README claims 🟢 golden-verified
      correctness; no perf number exists anywhere. Real T5-XXL + DiT + VAE pipeline per README.
- [ ] FLUX.1-schnell — check if a checkpoint is present; README says real but has a known unfixed
      tiling artifact (correctness gap, not blocking a timing measurement)
- [ ] SD3/3.5 — `docs/057-sd35-performance-handoff.md` already has a real number (215s/4-steps at
      256x256, cited in this doc's earlier session) — check if that's already been backfilled into
      PerformanceLeague.md as a citation (like Stable Audio 3 Medium was) and do so if not
- [ ] MiniMax-Music3 — real, working, user-confirmed per README, with an existing measured number
      (582.9s for a 200-frame/~8s generation) in `docs/066-minimax-music3-future-plan.md` — backfill
      this as a citation, same pattern as Stable Audio 3 Medium, no new run needed
- [ ] HunyuanVideo — README explicitly says 🔴, blocked on a missing VAE decoder — do not attempt,
      this isn't a "run it and see" gap, it's a known incomplete port

## Phase 4 — Bug fixes (parallelizable via subagents, NOT benchmarking)

- [ ] **Granite Vulkan correctness bug** — root cause found this session (subagent investigation):
      `GpuForwardPass.cs`'s `RunStandardLayers` never threads `AttentionScaleOverride`,
      `ResidualScale`, or `LogitScale` into the Vulkan dispatch path, while the CPU path
      (`ForwardPass.Decode.cs`/`PrefillCore.cs`/`Attention.cs`) applies all three. Exact file/line
      references already gathered. Next: implement the fix (a subagent can do this — reading+
      writing code, not running timed inference), then verify via a real CPU-vs-Vulkan comparison
      run (that verification run itself must be serial, done directly, not by a subagent).
- [ ] **DeepSeek-V2-Lite unsupported-architecture rejection** — confirmed this session that running
      without `--allow-unverified-arch` gives a clean rejection. README says this family works
      (accepted as "final" with a known routing-flatness caveat, run via that exact flag). Retry
      with the flag and get a real number if it produces coherent-enough output to be worth timing.
- [ ] **F5-TTS's blocked CPU backend in `audio.cpp`** — real, scoped, from the original backfill
      pass. Not attempted yet this session.
- [ ] **Chatterbox Turbo's missing streaming tokenizer asset** — real, scoped, from the original
      backfill pass. Likely a quick locate-or-regenerate fix.
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
