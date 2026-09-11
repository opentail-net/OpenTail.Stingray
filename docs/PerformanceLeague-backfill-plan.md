# PerformanceLeague backfill plan

**Goal:** close as many blank (`—`) Ratio cells in `PerformanceLeague.md` as possible by running the
real vendored C++ references (`tools/llama.cpp/llama-bench.exe`, `examples/whisper.cpp`,
`examples/audio.cpp`, `examples/s2.cpp`) against checkpoints already on this box, downloading missing
checkpoints into `models/_models` (F: drive, 227GB free, no space restriction) and symlinking into
`models/` where a harness expects a flat path, and — wherever OT itself was never benchmarked for a
scenario the C++ side can now cover — measuring OT too, since a new measurement is a good outcome
either way.

Binaries already built and confirmed working: `tools/llama.cpp/llama-bench.exe` (CPU, RPC backend),
`examples/whisper.cpp/build/bin/Release/{whisper-cli,whisper-bench}.exe`,
`examples/audio.cpp/build/bin/audiocpp_cli.exe`, `examples/s2.cpp/build/s2.exe`.

No CUDA device on this machine — CUDA rows stay out of scope (already correctly labeled † / no
numbers yet in the doc). Vulkan-vs-llama.cpp stays out of scope too: llama.cpp's Vulkan backend
isn't built here and building+validating it is a separate project, not a bench run.

---

## Phase 0 — setup

- [x] Confirmed `llama-bench.exe -m <gguf> -p <n> -n <n> -t 6 -ngl 0` matches the existing
      methodology; also fixed a repo-wide build break (`NU1902` on `Microsoft.Build.Tasks.Git`
      10.0.301, bumped `Microsoft.SourceLink.GitHub` to 10.0.401 in `Directory.Build.props`) that
      was blocking all OT-side CLI measurement.
- [x] OT CLI invocation confirmed: `stingray.exe -m <gguf> -f docs/benchmark-prompt.txt -n 24 -g 0
      --temp 0 --single-turn --no-display-prompt`, reports `Prefill:`/`Decode:` t/s directly.
      `docs/benchmark-prompt.txt` is the actual prompt file the original baseline used.
- [x] No new symlinks needed — every backfill so far ran directly against `models/_models/<file>`.

## Phase 1 — Qwen3 family — DONE

- [x] Qwen3-0.6B-Q8_0: re-measured at matched 493-tok prompt, both sides. Prefill 226.1 vs 265.28
      t/s (0.85x); decode 41.2 vs 62.95 t/s (0.65x). Superseded the old 47.4 t/s decode-only figure.
- [x] Qwen3-4B-Q4_K_M: new coverage, both sides. Prefill 61.4 vs 84.36 t/s (0.73x); decode 10.0 vs
      16.64 t/s (0.60x).
- [x] Qwen3-8B-Q4_K_M: downloaded (`Qwen/Qwen3-8B-GGUF`, Q4_K_M). Prefill 48.3 vs 47.28 t/s
      (**1.02x — real parity, best CPU LLM prefill ratio in the doc**); decode 6.3 vs 7.67 t/s
      (0.82x).
- [ ] Qwen3-Coder-30B-A3B-Q4_K_M: not attempted (large MoE, deprioritized this pass) — logged in
      Known Measurement Gaps.
- [x] Qwen3.6-35B-A3B / MTP / Carnice APEX: left as historical †, no attempt to re-measure CUDA
      (correct, no CUDA device here).
- [x] Qwen3.6-27B-MTP: searched HF for a public repo id for "Qwen3.6"/"Carnice APEX" — none found;
      logged as likely-gated/private in Known Measurement Gaps, not a build/download gap.

## Phase 2 — Gemma family — DONE (CPU rows)

- [x] Downloaded Gemma-4-12B-it Q4_K_M (`unsloth/gemma-4-12B-it-GGUF`) — closest available quant to
      the doc's QAT Q4_0 reference; noted the quant difference inline. Prefill 3.5 vs 27.81 t/s
      (**0.13x**); decode 4.2 vs 5.69 t/s (0.74x). Prefill≈decode signature (missing batched
      prefill) reconfirmed at this quant.
- [x] Downloaded Gemma-4-E4B-it Q4_K_M (`unsloth/gemma-4-E4B-it-GGUF`) — new CPU coverage (doc
      previously only had † CUDA/Vulkan rows for E4B). Prefill 9.8 vs 80.50 t/s (**0.12x** — same
      missing-batched-prefill bug, now confirmed on a second model size); decode 9.7 vs 13.07 t/s
      (0.74x).
- [x] Gemma-3-4B (already present, `gemma-3-4b-it-Q4_K_M.gguf`) — different family from the doc's
      Gemma-4 rows; left out, not worth adding as it doesn't backfill anything.

## Phase 3 — OLMoE family — DONE

- [x] Downloaded OLMoE-1B-7B-0924-Instruct Q4_K_M (`bartowski/OLMoE-1B-7B-0924-Instruct-GGUF`).
      Prefill 124.8 vs 180.82 t/s (0.69x); decode (full 24-tok run, not early-EOS-truncated like
      the original row) 25.3 vs 50.58 t/s (**0.50x** — worse than the old approximate 28.2 t/s
      figure suggested; the early-EOS truncation had undersold this gap).

## Phase 4 — Llama family (Llama-4 Scout) — IN PROGRESS, not yet backfilled

- [ ] Llama-4 Scout 17B-16E Q4_K_M download (`unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF`) is
      ~93GB across 2 shards — far larger than anticipated. First attempt failed mid-transfer
      (`ResponseEnded` after ~40MB); retried and still running in the background as of this update.
      Logged as an explicit sized gap in Known Measurement Gaps rather than blocking the rest of
      the doc on it. Re-run llama-bench + stingray CLI once the shards land.

## Phase 5 — SmolLM2 long-context prefill gap — DONE

- [x] Ran llama-bench `-p 267,773,1621,3218 -n 0` → filled the entire llama.cpp row of the CPU
      prefill scaling table. Finding: OT's 0.33x-ish prefill gap holds flat (0.24-0.27x) across
      context length — not a context-scaling effect.
- [x] Ran llama-bench `-p 0 -n 24 -d 267,773,1621,3218` → filled the entire llama.cpp row of the
      CPU decode scaling table. Finding: decode ratio degrades from 0.88x (short ctx) to 0.67x
      (3.2k ctx) — a real, newly-measured trend not visible in the doc before this pass.

## Phase 6 — TTS/ASR: close remaining blank-ratio pipelines — DONE (both confirmed still-blocked)

- [x] Checked `examples/audio.cpp`'s family registry directly (`--task tts --family <x> --help`
      family list): `f5_tts` IS registered (family + `model_specs/f5_tts.json` load fine), but a
      real run (`--voice-ref b.wav --reference-text ... --text "Hello, I will make some lunch,
      darling!"`) fails with `ggml_graph_compute_with_ctx unavailable (CPU backend not loaded)` —
      genuinely still blocked, now with a fresh 2026-09-10 confirmation and exact error text.
      `parler` is not registered as a family at all — no model_spec, no loader; a bigger lift than
      the doc previously implied (new family registration, not just enabling a build).
- [x] Checked `examples/piper`, `examples/kokoro.cpp`, `examples/MeloTTS.cpp`, `examples/TTS.cpp` —
      no `.exe` in any build tree, confirmed still unbuilt/blocked on external SDKs.
- [x] Both gap rows now carry a 2026-09-10 re-verification date instead of looking stale.

## Phase 7 — Vision encoder — left out of scope (as anticipated)

- [x] Confirmed `tools/llama.cpp/llama-mtmd-cli.exe`/`llama-mtmd-debug.exe` exist and are built, but
      did not attempt a cross-reference timing row: the existing doc row compares scalar-vs-SIMD
      internally, which is a different kind of comparison than a C#-vs-C++ oracle diff, and building
      a fair mtmd-cli timing harness for just this one row wasn't worth the scope creep this pass.

## Phase 8 — write-up — DONE

- [x] Every backfilled row got a 2026-09-10 Performance Check date and a Source note describing the
      exact command/binary used (no separate run-log file was created — the per-row Source/note
      text carries the same information inline, which was sufficient here).
- [x] Known Measurement Gaps table: removed every row that got closed this pass (SmolLM2 long-ctx,
      Qwen3-8B prefill/decode, Gemma prefill-batching-as-a-blank-ratio, OLMoE); refreshed the
      TTS/ASR rows with today's re-check date; added Llama-4-Scout (in-progress download),
      Qwen3.6/Carnice (unlocatable, likely gated), and Qwen3-Coder-30B (deprioritized) as new,
      explicit, dated gaps.
- [x] Updated the doc's `*Last updated:*` footer to 2026-09-10 with full source/command list.
- [x] Added a "Strengths & weaknesses, after the 2026-09-10 backfill" section: Qwen3-8B prefill
      parity (1.02x) as the headline strength, Gemma prefill collapse confirmed on two sizes as the
      headline weakness, decode-degrades-with-context as a newly-measured trend.

## Final status, 2026-09-11: sweep essentially complete

By the end of this pass:
- **Every LLM checkpoint with CPU coverage also has a Vulkan iGPU row** (or a documented, explicit
  reason it can't — e.g. Qwen3.8-27B's embedding-table-exceeds-2GB-buffer limitation): SmolLM2,
  Qwen3-0.6B/4B/8B/Coder-30B, OLMoE, Gemma-4-12B/E4B, Qwen3.6-27B/35B, Qwen3.8-27B, Ornith-9B,
  Mistral-7B, Ministral-8B, InternVL3-2B, Granite-4.0-3B-Vision, Granite-Vision-3.2-2B, dots.ocr.
- **Every TTS/ASR/VAD pipeline subdirectory in `src/OpenTail.Stingray.Audio` (all 30) has at least
  one real measurement**: AudioGen, Chatterbox, Citrinet, CosyVoice, F5TTS, FishSpeech, FunASR,
  HiggsAudio, Kokoro, MarbleNet, MeloTTS, MmsTts, MossTts, MusicGen, NemotronAsr, NeuTts, OmniVoice,
  Orpheus, Parakeet, Parler, PersonaPlex, Piper, QwenASR, QwenTTS, Vad (Silero), VibeVoice, VoxCpm2,
  VoxtralRealtime, Whisper, Xtts.
- **Five real bugs found and documented**, independent of the perf backfill's original goal:
  1. `stingray embed`'s GGUF path is a hash-based stub that never loads real weights (silent,
     serious — retracted a false measurement because of it).
  2. `stingray embed`'s ONNX path is real but crashes on a real BERT checkpoint (missing
     `token_type_ids`).
  3. Vulkan-specific Granite architecture correctness bug (coherent CPU output, garbled/degenerate
     Vulkan output on the identical prompt) — confirmed on two independent checkpoints.
  4. Two independent degenerate-ASR-output cases (Qwen3-ASR, FunASR-Nano) — fast but wrong,
     flagged as more urgent than any perf gap.
  5. Three tests found silently no-op'ing (Parakeet-CTC, Orpheus, and the general pattern flagged
     as likely affecting others) due to a `models/`-only search helper missing `models/_models/`
     checkpoints — fixed via symlinks matching the existing convention.
- **Remaining, explicitly out of scope this pass**: video/image diffusion beyond one Z-Image-Turbo
  attempt (hunyuanvideo, ltx-t5, wan2.1 have no wired end-to-end test found; likely far slower than
  anything measured so far on CPU) and Carnice APEX (unlocatable checkpoint).

## Extended sweep, 2026-09-10 (session continued, per "test every remaining model" instruction)

Went beyond the original plan's scope to sweep every untested checkpoint under `models/`/
`models/_models`, writing temporary `*PerfBaselineDebugTest.cs` harnesses for any real, wired OT
pipeline lacking one. All added as new coverage (with C++ comparison where one exists, honest
non-degeneracy/degenerate-output caveats where none does):

- [x] Llama-4-Scout: **cancelled** by explicit user instruction (~93GB vs. 64GB total RAM — would
      never fit). Partial download deleted. Not pursuing on this hardware.
- [x] Qwen3-Coder-30B-A3B-Instruct: downloaded (unsloth Q4_K_M), backfilled both sides — decode
      **1.08x, beats llama.cpp**.
- [x] Qwen3.6-27B-MTP and Qwen3.6-35B-A3B (hybrid-GDN architecture): downloaded, backfilled both
      sides. Real finding: hybrid-GDN prefill ratios (0.05-0.16x) are the worst of any dense/MoE
      architecture measured — worse than Gemma-4's missing-batched-prefill gap. Carnice APEX
      remains unlocatable (likely gated).
- [x] Qwen3-Embedding-0.6B: new "Embeddings" section, first measurement (~38,900 tok/s). Surfaced
      an unrelated real bug in passing: `stingray embed -o <file>` crashes on reflection-based JSON
      serialization (CLAUDE.md rule 4 violation) — flagged, not fixed (out of scope).
- [x] Paraformer (`paraformer-q8.gguf`): attempted — checkpoint itself is broken (missing
      `pf.vocab` GGUF metadata), not a code bug. Not pursued further.
- [x] Voxtral-Mini-4B-Realtime (ASR): new coverage via raw building-block harness (no pipeline/CLI
      exists yet). **50x slower than C++ reference (0.02x) — worst ratio in the doc**, on an
      otherwise-correct transcript.
- [x] Qwen3-ASR 0.6B (safetensors): new coverage. Fast (RTF 0.225) but transcript is **degenerate**
      ("aspects" only) — flagged as a correctness bug, more urgent than any perf gap.
- [x] CosyVoice2-0.5B, XTTS-v2, Whisper Tiny (HF safetensors), MusicGen-small, AudioGen-medium,
      Stable Audio 3 Small Music, Stable Audio 3 Small SFX, ACE-Step Turbo: all new OT-only
      coverage (no C++ reference exists for any of them). ACE-Step Turbo came back with **114x
      RTF — the worst absolute RTF in the entire doc**, despite "Turbo" naming.
- [x] Stable Audio 3 Medium: backfilled by citation from an already-measured prior-session number
      (`docs/00-current-work.md`), not a new run.
- [x] DSpark speculative decoding (Qwen3-4B + `dspark_qwen3_4b_block7`): new coverage. Confirms
      speculation is a loss on this hardware on a *second* independent target/draft pair (−75%,
      worse than the existing −37% n-gram/draft-model finding), with a lower acceptance rate too
      (23% vs 62%) — not yet disambiguated why DSpark specifically pays a bigger penalty.
- [x] TTS Streaming Latency gap (user asked directly why every C++ TTFA cell is blank): root-caused
      properly — `audiocpp_cli --mode streaming` exists but `qwen3_tts`/`cosyvoice3` explicitly
      refuse it at runtime, `chatterbox_turbo` is blocked on a missing tokenizer asset, and
      `--metrics` itself refuses in streaming mode regardless of family. Found one real exception:
      `examples/s2.cpp/build/bin/s2.exe --stream-file` genuinely streams with real metrics, but
      doesn't print an explicit first-chunk timestamp, so a true TTFA comparison isn't computable
      from it without a CLI change — left un-inferred rather than presenting a guess as a
      measurement.
- [ ] Z-Image-Turbo (image diffusion): attempted via `stingray image` CLI — still running as of
      this update (in background). Genuinely a different domain (image, not audio/LLM) but all
      pieces existed locally so it was tried anyway per "more models tested is always better."

## Not completed this pass (explicit)

- Qwen3-Coder-30B-A3B and Qwen3.6/Carnice APEX family (deprioritized / unlocatable) — **superseded**:
  Qwen3-Coder-30B-A3B and Qwen3.6 family both closed above; only Carnice APEX remains unlocatable.
- Building Piper/Kokoro/MeloTTS/MMS-TTS/Parler standalone C++ CLIs (multi-SDK build effort, out of
  scope for a bench-backfill pass — re-verified still blocked, not re-attempted).
- Video/image diffusion checkpoints (`hunyuanvideo`, `ltx-t5`, `wan2.1`) beyond the one
  Z-Image-Turbo attempt — a different domain than this doc's LLM/TTS/ASR focus, and likely far
  slower per-run than anything measured so far on CPU.
- `minimax-music3` — not attempted (real GPU-parity benchmark already exists for it per
  `CLAUDE.md` rule 13, outside this doc's scope).

---

*Execution note: work top to bottom, phase by phase. Per CLAUDE.md, no subagents — all runs done
directly in this session. Per standing "stopping is for wimps" directive: if one download or build
stalls, log the blocker precisely in this checklist and move to the next phase rather than halting.*
