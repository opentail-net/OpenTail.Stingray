# Audio re-check plan: correctness + performance (written 2026-09-24)

Scope: `src/OpenTail.Stingray.Audio`, which has 34 engine folders and ~70k lines. The README
matrix lists 30 audio capabilities, and there is an RTF table of 11 TTS engines. Checkpoints for
almost everything are on disk (`models/` and `models/_models` → `F:\_models`). The only ones
missing are **MarbleNet** and **RVC**, so nearly every check below can use real weights.

## Rules (the ones that paid off on the 2026-09-24 diffusion pass)

- **Real weights, visible timing.** Record the per-class wall-clock time. A pass in ~0.2s is a
  silent no-op, not a pass (CLAUDE.md rule 12).
- **Correctness against a reference, not by ear alone.** Where a vendored reference exists
  (`examples/whisper.cpp`, `qwen3-asr.cpp`, `cosyvoice.cpp`, `kokoro.cpp`, `parakeet.cpp`,
  `paraformer.cpp`, `audio.cpp`, onnxruntime), compare tensors or transcripts. For TTS the cheap
  automatic check is a round trip: synthesize, transcribe with our Whisper, compare the text.
- **One heavy process at a time.** Perf numbers come from ≥2 runs. Before starting, check that no
  other test `.exe` is running.
- **Profile before optimizing.** Every big 2026-09-24 win came from a profile, and it was never
  the obvious suspect: a scalar single-threaded attention loop (Qwen Image), a per-call weight
  re-read (Wan/safetensors), a scalar 3D-conv loop (LTX VAE).
- Scratch output goes to the temp scratchpad (rule 9). Samples go into `docs/audio-samples/`,
  which is local-only and not committed.

## Phase 0: regression sweep for the 2026-09-24 shared-code changes (do first)

Shared code changed today:
- `QuantizedWeightCache`: F32 pack-once (also via `ReadF32` for safetensors), dequantize-once
  for `allowQ8: false`, and the new `PackedSgemmF32.GemmQuant` for any block-quantized weight
  with ≥16 rows.
- `OpenClipGEncoder` now has 20×64 heads (was 16×80).
- SD3's T5 is unmasked. `T5Encoder` itself is unchanged, but its doc comment was corrected.
- `WanAttention` now chunks query rows (this also backs `DiffusionOps.MultiHeadAttention`).

Steps:
1. Grep the Audio project for `QuantizedWeightCache`, `DiffusionOps.MultiHeadAttention`,
   `WanAttention`, `T5Encoder` and `OpenClipGEncoder`, to list the affected engines. Expected at
   least: Stable Audio 3, MiniMax-Music3, MusicGen/AudioGen (T5), Parler (T5).
2. For each affected engine, re-run its real-weight test and one sample. Compare against the
   last documented sample and timing.

## Phase 1: correctness triage (ordered by risk)

1. **Open 🟡 rows**: RVC (checkpoint missing, so download it first), OmniVoice, MOSS-TTS-Nano
   (listed twice in the README, so merge the rows), Higgs Audio, Stable Audio 3 Medium. For each:
   re-read the dated finding in `docs/audio-review-progress.md`, re-run, and bisect against a
   reference where one exists.
2. **⚪ rows** (run, but never validated): PersonaPlex, MOSS-TTS-Nano. Give each its first real
   reference check.
3. **Engines with code but no README row**: Parakeet, QwenASR + ForcedAligner, Silero VAD,
   VibeVoice TTS/ASR (long open investigation, see the progress doc's tail), CosyVoice 2,
   Whisper large-v3/medium. Add a verified row for each, or write down the precise gap.
4. **🟢 engines**: one real-weight re-run each, to confirm nothing drifted since the matrix was
   written. For ASR, check word error rate on a fixed clip. For TTS, check the Whisper round-trip
   text plus a listen.
5. **Known bug classes from the diffusion pass**, checked in every engine (grep first, then
   confirm):
   - scalar or single-threaded attention/conv loops;
   - T5/attention padding masks that differ from the reference (per model, see the
     `T5Encoder.EncodeGpu` doc comment);
   - hand-coded head counts or dims that don't match the checkpoint or reference config;
   - `flipSinToCos` in timestep embeddings;
   - int8-activation (`allowQ8`) paths on precision-sensitive layers. Note that `MatVecQ3K`
     quantizes activations even with `allowQ8: false` (relErr ~4e-3, measured 2026-09-24).
   - tests that write their output twice and overwrite the first (the LTX seed-43 trap).

## Phase 2: performance (worst RTF first)

1. The slowest rows in the RTF table: FishSpeech 16.3×, F5-TTS 13.0×, CosyVoice 3 7.5×,
   Parler 5.7×, Chatterbox 4.7×, then QwenTTS / XTTS-v2 (~3×). For each: profile with
   temporary per-stage timers, fix the top one or two costs, re-measure with ≥2 runs, and keep a
   change only if it is measurably faster.
2. Engines with no RTF number yet: measure a baseline.
3. Likely shared levers:
   - route engine-local linears (many are `ReadF32` + dot loops) through
     `QuantizedWeightCache`, which gives the packed-GEMM / `GemmQuant` wins for free;
   - replace local attention loops with the shared tiled kernel
     (`DiffusionOps.MultiHeadAttention`);
   - swap scalar Conv1d/ConvTranspose1d loops for im2col + GEMM;
   - cache per-call small-tensor reads (biases, norm gammas).
4. DRY pass (CLAUDE.md rule 7): fold duplicated Linear/LayerNorm/attention helpers into
   `Primitives/*Kernels.cs`, then re-run the affected golden tests.

## Phase 3: record

Update the README status matrix and the RTF table, plus `docs/audio-review-progress.md`, with
dated and sourced findings. Commit per engine.

## Effort estimate

- Phase 0: small.
- Phase 1 items 1–3: medium to large. The 🟡 items are open-ended, so timebox each one and
  write down the blocker rather than stall (CLAUDE.md "stopping is for wimps").
- Phase 2: medium per engine.

## Status

- [ ] Phase 0: shared-change regression sweep
- [ ] Phase 1: correctness triage
- [ ] Phase 2: performance
- [ ] Phase 3: record
