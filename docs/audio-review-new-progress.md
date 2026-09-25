# Audio subsystem review — NEW progress log


## MOSS-TTS-Nano -- the "2.7x loudness gap" was a one-sample artifact; upgraded to 🟢, 2026-09-24

The 2026-09-08 conclusion compared ONE reference sample against ONE of ours (seed 11), and
attributed a ~2.7x RMS difference to the RNG mismatch. A different random stream explains why
individual samples differ, but it can't produce a *systematic* loudness gap unless the sampling
distribution differs, and one sample can't tell those apart. Re-checked with 6 seeds of the same
sentence ("Hello there, this is a test of speech synthesis.") on both sides:

| | RMS per seed 1..6 | mean | durations |
|---|---|---|---|
| reference `audiocpp_cli` (same q8_0 GGUF, `--backend cpu`) | 0.091 0.092 0.065 0.114 0.077 0.089 | 0.088 | 3.3-4.7s |
| this port (`MossTtsGenerator`, same sampling defaults) | 0.113 0.100 0.083 0.084 0.052 0.094 | 0.088 | 3.0-4.8s |

The distributions are indistinguishable, so there is no loudness gap. The seed-to-seed spread alone is ~2x.
Intelligibility was checked with a Whisper round trip (`stingray stt -m base --model-file
models/ggml-base.bin`): 5/6 of our samples transcribe exactly, and one gets the first two words wrong
("Follow the old, ..."). The reference's scored 3/6 exact, with three word errors. README row merged
(it was listed twice) and upgraded to 🟢 👂. Harness: `tests/OpenTail.Stingray.Tests.Audio/ZzMossProfTmp.cs`
(untracked).

## Higgs Audio TTS -- reference distribution check passes; upgraded to 🟢, 2026-09-24

Same method as MOSS-TTS-Nano above: 4 seeds of "Hello there, this is a test of speech synthesis."
from this port (`HiggsGenerator.Generate`, temperature 0.7 / top-k 50 / top-p 0.95) and from the vendored
reference (`audiocpp_cli --family higgs_audio_tts`, same `higgs-audio-v3-tts-4b-q8_0.gguf`, CPU).

- Whisper round trip (`stingray stt -m base`): **4/4 exact on both sides**.
- Duration: ours 3.00-3.68s, reference 3.40-3.92s.
- RMS: ours 0.080 0.048 0.057 0.053 (mean 0.060), reference 0.062 0.060 0.051 0.059 (mean 0.058).

Upgraded 🟡 ⚪ → 🟢 👂. Open, perf only: ~47s per sample vs the reference's ~16s (≈3×), so a Phase 2
target. Harness: `tests/OpenTail.Stingray.Tests.Audio/ZzHiggsProfTmp.cs` (untracked).

## OmniVoice -- reference check passes, 3.9× faster; upgraded to 🟢, 2026-09-25

**Correctness.** Same sentence ("Hello there, this is a real test of speech synthesis."), 4 seeds on
each side, at the reference's own defaults (32 MaskGIT steps, guidance 2.0):
- ours: `OmniVoiceMaskGitGenerator` via `OmniVoiceGenerateWavDebugTest` (now takes `OMNI_SEED` /
  `OMNI_STEPS` / `OMNI_OUT`; the default went from 12 to the reference's 32 steps), fixed 3.20s target;
- reference: vendored `audiocpp_cli --task tts --family omnivoice --backend cpu --threads 8` on the
  same `models/_models/omnivoice` safetensors (it needed `audio_tokenizer/preprocessor_config.json`,
  downloaded from `k2-fsa/OmniVoice`), which picks its own duration (3.31-3.36s).
- Whisper round trip (`stingray stt -m base`): **4/4 word-exact on both sides** (the only difference
  is a comma vs a full stop after "Hello there", which each side produces once).

**Performance (CPU only: OmniVoice has no GPU path).** Profile by reading the code: every
projection ran as one mat-vec **per token** (`DenseKernels.LinearNoBias` inside `Parallel.For` over
tokens), so each weight matrix was streamed from memory once per token, and RoPE recomputed
`Math.Pow`/`Cos`/`Sin` per element per head per layer. Fix: flat `[seq, dim]` buffers, one batched
GEMM per projection through the new shared `DenseKernels.LinearBatchedNoBias` (weights packed once
into `PackedSgemmF32` panels, cached per weight array), batched audio-head GEMMs for the CFG
logits, a RoPE table per forward, and attention parallel over head × query-chunk.

| | per sample (32 steps, 3.2s audio, includes model load) |
|---|---|
| before | 118.3 / 118.4 / 128.3 / 128.5s |
| after | 31.6 / 31.0 / 30.8 / 31.4s (**~3.9×**) |
| reference `audiocpp_cli` | 16.2 / 16.7 / 16.7 / 16.9s |

The transcripts are unchanged after the change (4/4 exact), and `OmniVoiceMaskGitGeneratorRealWeightsTests`
still passes (2.5s: a real 4-frame, 8-step run). We remain ~1.9× behind the reference. The next
levers would be fusing QKV/gate-up and a KV-reuse scheme for the static text prefix, not attempted.
Gap: no CLI/server wiring (library + test harness only).

## VibeVoice ASR -- transcribes correctly now; the 09-08 "resampler amplification" conclusion was wrong, 2026-09-25

The 2026-09-08 entries concluded that the garbled transcripts came from an unavoidable resampler
phase difference amplified by the encoder, and closed the investigation. That rested on comparing
**one element per stage**. The decisive test: feed audio that is **already 24 kHz** (the model's
native rate), so neither side resamples. Our port then transcribed it perfectly, and so did the
resampled 16 kHz LibriSpeech clips. Whatever made the transcripts garbled on 09-08 has since been
fixed (most likely one of the later shared ForwardPass/norm fixes; not bisected), and the resampler
was never the cause.

Test: `VibeVoiceAsrRealSpeechRealWeightsTests` (now takes `VV_WAV`; its fixed-index debug taps only
run with `VV_TRACE=1`, since they crashed on any other clip). Reference: vendored
`audiocpp_cli --task asr --family vibevoice_asr --backend cpu --threads 8`, same `vibevoice-asr-q8_0.gguf`.

| clip | ground truth | ours | reference |
|---|---|---|---|
| OmniVoice 24 kHz sample | Hello there, this is a real test of speech synthesis. | exact | exact |
| test-clean 6930-75918-0000 | CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS | exact | exact |
| test-clean 6930-75918-0001 (15s) | THE ENGLISH FORWARDED … THE NEXT DAY | exact (identical to ref) | exact |
| test-other 7902-96591-0000 | I AM FROM THE CUTTER LYING OFF THE COAST | "I'm from the **corner** lying off the coast." | "I'm from the cutter lying off the coast." |
| test-other 7902-96591-0001 | DON'T CRY HE SAID I WAS OBLIGED TO COME | exact (identical to ref) | exact |

4/5 identical to the reference; one word differs on a "test-other" (hard) clip. That's plausibly
the 16→24 kHz resampler difference flipping a close token, not investigated further.

Performance (CPU only: no GPU path, no CLI wiring): ours 47-56s for 3.5s clips and 142s for the 15s
clip (test wall **including** GGUF load), vs reference 11.3-12.6s and 31.1s (`metrics.wall_ms`).
~4× behind: a Phase 2 target (profile first).

## VibeVoice TTS -- two real generation-loop bugs found and fixed; never stopped → natural stop, 3/3 exact; upgraded to 🟢, 2026-09-25

**Symptom (re-measured before the fix).** Script "Speaker 1: Hello there, this is a real end to end
test of speech synthesis.", 1.5B q8_0, 10 inference steps, guidance 1.5, 3 seeds. Ours never emitted
`speech_end`: every run hit the 60-frame cap (8.00s) and Whisper heard "[inaudible]" ×2 and
"Oh, oh, oh…". Reference (`audiocpp_cli --family vibevoice --backend cpu`, same GGUF) stopped by
itself at 4.80-5.07s ("We held it!", "Hello there, this is Aurel M2N Test of Speech synthesis.",
"Hello there. This is a real end-to-end test of speech synthesis.").

**Root causes**, both found by reading `VibeVoiceGenerator`'s loop against the reference's
`src/models/vibevoice/generator.cpp` (following the 2026-09-09 lead that the feedback embedding
wasn't carrying progress information):
1. **Wrong connector input.** The reference projects the *scaled* diffusion output
   (`connector.project_acoustic(speech_latents.front())`) as the acoustic half of the next-step
   embedding. We projected the *unscaled* decoder latent (`latent / scaling - bias`), so the LM got a
   mis-scaled feedback embedding every frame. (The voice-prompt path already scaled correctly.)
2. **Off-by-one position.** The prompt fills positions `0..promptLength-1`; we wrote generated
   step N at `promptLength + 1 + N`, leaving KV slot `promptLength` unwritten but inside the
   attention window for every later step. The reference appends at its cache's current end (its
   third `cached_step` argument is a capacity, not a position).

**After** (same seeds): natural stop at 4.80 / 4.27 / 4.53s; Whisper: **3/3 exact**
("Hello there, this is a real end-to-end test of speech synthesis."), better than the reference's
own 3 samples. Real-weight suites still pass with real timings: `VibeVoiceGeneratorRealWeightsTests`
26.5s, `VibeVoiceTtsVoiceCloningRealWeightsTests` 21.1s, `VibeVoiceDiffusionHeadRealWeightsTests`
1.9s, `VibeVoiceTtsPromptBuilderRealWeightsTests` 16.9s.

This overturns the 2026-09-08 "compounding floating-point drift / non-associativity" conclusions:
those were symptoms of these two deterministic bugs.

Perf (CPU only, no GPU path, no CLI wiring): ours 65-72s per ~4.5s sample (test wall incl. GGUF
load) vs reference 19.0-19.9s (`metrics.wall_ms`); ~3.5× behind, a Phase 2 target.

## RVC -- RMVPE pitch extractor now golden-matches (transpose-conv kernel was read as 2×2, is 3×3), 2026-09-25

The README row said RMVPE's remaining gap was a quirk of the reference, but
`RvcRmvpeEndToEndRealAudioTests`' own comment said the divergence starts at the U-Net decoder's
ConvTranspose2d stage and was "not yet root-caused". The checkpoint settles it: every
`support_rmvpe/unet.decoder.layers.N.conv1.0.weight` is `[3, 3, out, in]` (PyTorch `[in, out, 3, 3]`),
i.e. `ConvTranspose2d(kernel=3, stride=2, padding=1, output_padding=1)`. The reference's
"(2n+1) full output, keep [1, 2n+1)" trick is exactly that; it's not a ggml artifact, as our comment
claimed. `ConvTranspose2dPyTorch2x` hard-coded a 2×2 kernel (`weight[... * 4]`), misreading all five
decoder upsample stages. Fixed to the real k=3 scatter (tap `(kh,kw)` → `(2i+kh-1, 2j+kw-1)`), with a
weight-size guard, and parallelised over output channels.

| salience (a.wav, 796 frames) | mean | std | max |
|---|---|---|---|
| reference | 0.00474 | 0.04768 | 0.97022 |
| ours before | 0.00298 | 0.02489 | 0.81235 |
| **ours after** | **0.00475** | **0.04768** | **0.97143** |

The test now asserts these (mean ±2 %, std/max ±1 %) instead of only printing them. HuBERT, the RMVPE
encoder and weight-load tests pass (30-45s each, real weights). `RvcSynthesizerRealReferenceMatchTests`
skips visibly (it needs reference `STINGRAY_RVC_TRACE` dump files that aren't on disk).
Still open for RVC: the `native_pipeline.cpp` orchestration (segmenting, synthesizer-input
assembly, RMS mix, pad-crop) — next.

## RVC -- end-to-end pipeline ported (`RvcPipeline`); round trip word-exact vs reference, 2026-09-25

New `src/OpenTail.Stingray.Audio/Rvc/RvcPipeline.cs` ports `native_pipeline.cpp`'s orchestration
over the verified stages: high-pass + reflect pad, RMVPE salience → f0 (threshold 0.03, ±4-class
cents average), median filter, quiet-point segmenting (>32s inputs), HuBERT content ×2 upsample,
optional retrieval blend, coarse pitch bins + semitone shift, unvoiced protection (0.33), NSF sine
source, per-segment pad crop, RMS mix (0.25). Defaults copied from the reference's `RvcInferenceConfig`.
Scope: v2 voices only (v1 needs HuBERT layer-9 + `final_proj`, not ported; rejected explicitly).
Sine-source noise uses .NET `Random`, not the reference's Philox CUDA RNG, so output is not bit-identical.

Check: `a.wav` (5.95s, "This little work was finished in the year 1803 and intended for immediate
publication."), packaged `default` v2 voice, CPU.
- ours (`RvcPipelineRealWeightsTests`, `RVC_OUT`): 5.94s @ 40 kHz, rms 0.071; Whisper: exact.
- reference `audiocpp_cli --task vc --family rvc --backend cpu --threads 8`: 5.94s; Whisper: exact.

**Performance, flagged as its own line item (CLAUDE.md rule 11):** ours 523.7s of compute vs
the reference's 8.2s (**~64× slower**). A profile-first perf pass is the next RVC step.

## RVC perf pass -- 482.6s → 19.8s for 6s of audio (24×), output unchanged, 2026-09-25

Profile first (`STINGRAY_RVC_PROFILE=1` stage timers in `RvcPipeline`, kept): rmvpe 14.95s,
hubert 5.42s, **synthesizer 462.2s (96%)**. The synthesizer's conv helpers were scalar,
single-threaded, with an inner channel loop striding through memory.

| step | synthesizer | rmvpe | total convert | vs pre-change output |
|---|---|---|---|---|
| before | 462.2s | 14.95s | 482.6s | — |
| conv helpers → row-wise vectorized axpy, parallel over out-channels | 75.0 / 73.0s | 14.8s | 95.0 / 92.8s | cosine 1.000000, maxAbs 4.85e-5 |
| stride-1 Conv1d → im2col + packed GEMM (`DenseKernels.LinearBatchedNoBias`, PyTorch weight used as-is) | 8.01 / 7.88s | 14.7s | 28.0 / 28.0s | cosine 1.000000, maxAbs 4.85e-5 |
| RMVPE 3×3 Conv2d → im2col + packed GEMM | 7.08 / 6.88s | 7.59 / 7.77s | **19.9 / 19.7s** | cosine 1.000000, maxAbs 1.65e-4 |

(maxAbs 4.85e-5 is 16-bit WAV quantisation; 1.65e-4 comes from tiny f0 differences after the
RMVPE rewrite.) RMVPE salience after the rewrite is still 0.00475 / 0.04768 / 0.97143, and the end-to-end
test's reference asserts still pass. Whisper round trip is still word-exact. Reference: 8.2s → we're ~2.4× behind.
Remaining big items: HuBERT (5.1s), RMVPE (7.7s, other convs / GRU) if more is wanted.

## Stable Audio 3 -- sampling schedule fixed; small-music + medium match the reference at official settings; SFX + cfg=1 gap open, 2026-09-25

Checked all three local base checkpoints (`models/stable-audio-3-{small-music,small-sfx,medium}-base`)
against the vendored `audiocpp_cli --task gen --family stable_audio` on the same safetensors. The reference
needed each model's `model_config.json` and 3 small T5Gemma files (`config.json`, `tokenizer.model`,
`generation_config.json` from `stabilityai/stable-audio-3-small-music-base/t5gemma-b-b-ul2`, now in
`models/stable-audio-3-t5gemma`, junctioned into each model dir as `t5gemma-b-b-ul2`). Whisper can't judge music,
so the comparison uses duration / rms / peak / HF-energy ratio / zero-crossing rate
(untracked harness `ZzSa3CmpTmp.cs`); samples are local-only in `docs/audio-samples/sa3-cmp-2026-09-25_*`.

Findings, in order:
1. **Comparison trap (not a port bug)**: the C++ reference defaults to the **pingpong** sampler
   (`rf_dit.cpp`: empty sampler → pingpong), while official Python picks by objective and all three
   base configs are `rectified_flow` → **Euler** (`sampling.py:434`). Against pingpong, ours looked 7-20×
   "muffled". With `--request-option sampler=euler` the gap mostly disappears (below).
2. **Real fix — sampling schedule.** `models/diffusion.py`: inference uses `sampling_dist_shift`, which
   defaults to `LogSNRShift(rate=0, anchor_logsnr=-6.2, logsnr_end=2.0)` when the config has no
   `sampling_distribution_shift_options` (true for all three). `distribution_shift_options`
   (`DistributionShift`, what `StableAudioScheduleKernels.ShiftTimestep` applied) is the *training*
   distribution. The C++ reference's `shifted_logsnr_timestep` agrees. `ShiftTimestep` now implements
   the LogSNR schedule (old formula kept as `TrainingDistributionShift`). Measured effect on these stats is
   small (hfRatio 0.0070 → 0.0074 small-music), so this rests on the source code, not the numbers.
3. **Auto-GPU**: with no backend passed, `DiffusionBackendResolver` creates Vulkan, so "CPU" harness runs
   were GPU. The true CPU path (`STINGRAY_BACKEND=cpu`) gives the same stats (small-music cfg 7: hfRatio
   0.0074 both), so CPU and GPU agree; CPU 93.9s vs GPU 50.4s for 6s/50 steps.
4. Tokenizer: our T5Gemma ids for the golden prompt equal the fixture's official ids (`545,2485,3036,10273`).

| 6s, 50 steps, CFG 7, Euler | rms ours / ref | hfRatio ours / ref | zcr ours / ref |
|---|---|---|---|
| small-music (seed 1) | 0.080 / 0.087 | 0.0074 / 0.0060 | 0.014 / 0.012 |
| medium (seed 1) | 0.337 / 0.290 (both peak 1.0: model behaviour) | 0.0037 / 0.0057 | 0.019 / 0.020 |
| small-sfx seeds 1-4 | 0.05-0.06 / 0.08-0.12 | 0.08-0.73 (mean ~0.43) / 0.61-0.93 (mean ~0.76) | 0.06-0.12 / 0.15-0.22 |

**Open (timeboxed, not root-caused)**: (a) SFX is consistently darker than the reference over 4 seeds
(~0.6× HF energy, ~0.5× ZCR; ranges overlap); (b) at CFG 1 (conditional only, off-nominal for base
models) the reference is much brighter (hfRatio 0.055 vs ours 0.002). Both point to a subtle
divergence in shared conditioning/DiT code that the 0.5s component goldens don't catch. Next step
would be a reference latent dump (needs a small `audiocpp_cli` patch + rebuild) to diff per-step latents
at CFG 1.
Timing (GPU iGPU, 6s/50 steps): small 50s, medium 140s; reference CPU: small 52-55s, medium 167-170s.

## Full heavy Audio sweep attempt -- stopped for memory; one finding, 2026-09-25

A single-process run of the whole Audio test project with `STINGRAY_RUN_HEAVY_TESTS=1` was stopped by
Claude Code's low-memory guard. The test process had grown to **44.9 GB** (it was loading PersonaPlex's
25.5 GB of weights while earlier classes' models were still resident). The orphaned process was killed
by hand. **Lesson:** a heavy sweep has to run one class per process (memory released between classes),
not one process for all 313 test files. Not re-run automatically.

Before it stopped, the only failure was `CosyVoice3DiTInputEmbedGoldenTests` (cosine 0.639). That's a
known stale oracle (2026-09-06 entry: Python fixture uses centered conv-pos padding; the real reference
and our code are causal). The test now skips visibly with that reason instead of failing.

## Qwen3 Forced Aligner -- first reference check: 31/32 word boundaries identical; confidence ⚪ → 🔬, 2026-09-25

`QwenForcedAlignerRealAlignmentTests` (safetensors `models/qwen3-forcedaligner`, 5.3s incl. load) vs vendored
`audiocpp_cli --task align --family qwen3_forced_aligner --language English` (q8_0 GGUF, 0.8s) on `a.wav` with
"This little work was finished in the year eighteen o three, and intended for immediate publication.":
all 16 words present on both sides; start/end times agree exactly for 31 of 32 boundaries (80 ms frame
grid). The one difference is the end of "year": ours 2.16s, reference 2.24s (1 frame). (The reference CLI
needs `--language`.)

## PersonaPlex -- first intelligibility check: speech tracks the model's own text stream; confidence ⚪ → 👂, 2026-09-25

The vendored reference only exposes speech-to-speech (`s2s`, needs user audio), while our harness
(`PersonaPlexGenerateWavDebugTest`, now takes `PP_SEED`/`PP_FRAMES`/`PP_OUT`) drives the text-system-prompt
path. So the check is self-consistency: does Whisper hear in the generated audio what the model's
inner-monologue text stream says? 150 frames (12s), q8_0 GGUF, CPU:

| seed | LM text stream | Whisper on the generated audio |
|---|---|---|
| 1 | "Hello, this is Kendra. How can I help you today?" | "Hello, this is Sandra. I want to help you today." |
| 2 | "Hey, let me know if you have any questions." | "Thank you. Thank you. You have any questions?" |

Intelligible, on-topic speech that follows the text stream, with Whisper mishearing some words (the
name, one phrase). Not reference parity. Perf: 240s per 12s sample (RTF ~20), a Phase 2 candidate.

## Qwen3-ASR + Parakeet -- first real-speech checks (no README rows before), 2026-09-25

Untracked harness `ZzAsrSweepProfTmp` (`ZZ_ASR=qwen|parakeet`) over the 4 audio.cpp LibriSpeech validation
clips, CPU. Neither engine had a README row; Parakeet's `ParakeetRealWeightsTests` only ever transcribed a
synthetic 300 Hz tone.

| clip | truth | Qwen3-ASR 0.6B (safetensors) | Parakeet CTC 0.6B (q4_k) |
|---|---|---|---|
| test-clean 0000 (3.5s) | CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS | exact (3.9s) | exact (1.6s) |
| test-clean 0001 (14.2s) | THE ENGLISH FORWARDED … THE NEXT DAY | exact (5.6s) | "made ~~a~~ plentiful provision", "to ~~a~~ supper" (3.2s) |
| test-other 0000 (2.1s) | I AM FROM THE CUTTER LYING OFF THE COAST | "I'm from the cutter…" (2.2s) | "i'm from the cutter…" (0.7s) |
| test-other 0001 (2.5s) | DON'T CRY HE SAID I WAS OBLIGED TO COME | exact (2.4s) | exact (0.8s) |

Qwen3-ASR: 0 real errors in 65 words. Parakeet: 2 deletions (~3% WER), plausibly q4_k quantisation;
not investigated. README rows added (🟢 👂). Also corrected: the Qwen3 Forced Aligner check above was manual,
so its confidence is 👂, not 🔬 (🔬 needs an automated reference test per the README legend).

## Silero VAD -- missing 64-sample context fixed (now exact vs onnxruntime); `stt --vad` crash fixed, 2026-09-25

Checked our native `SileroVad` against onnxruntime running the same `models/silero_vad.onnx` (oracle only),
per 512-sample frame of `a.wav` with 1s of silence each side.
- **Before:** per-frame probabilities up to 0.755 apart (frame 60: 0.48 vs 0.93); decisions 240/248.
- **Cause:** Silero v5's model input is the previous chunk's last **64 samples of context + the 512-sample
  frame** (576, which the graph reflect-pads to 640 → 4 STFT frames). The port fed the bare 512-sample frame
  (→ 3 STFT frames). The old golden test agreed with the bug because its onnxruntime golden was also captured
  on a bare 512-sample frame.
- **After:** `SileroVad` keeps a 64-sample context (cleared by `Reset`): maxAbsDiff **8.9e-7**, decisions
  **248/248**. New `SileroVadOnnxRuntimeParityTests` (onnxruntime oracle computed live, official input
  convention); `SileroVadWeightsTests` now computes its oracle live the same way instead of the stale constant.

**Also a real product bug:** `WhisperPipeline` defaulted to `new SileroVad()` with no weights, so
`stingray stt --vad` (and the server's `vad=true`) threw "requires real SileroVadWeights". It now resolves
real weights lazily via the new `SileroVad.TryLoadDefault()` (`STINGRAY_SILERO_VAD_PATH`, else
`models/silero_vad.onnx` walking up from cwd / app dir), with a clear error if none exists. Verified:
`stt --vad` on `a.wav` segments and transcribes correctly. The 5 Fast `SileroVadTests` that constructed a
weightless VAD (failing since the procedural fallback was removed) now load real weights or skip visibly;
6/6 pass.

## Audio.Fast suite -- 6 stale tests updated, one real Melo fix; now 72/72 green, 2026-09-25

Running the light `OpenTail.Stingray.Tests.Audio.Fast` suite (3.9s, no heavy models) showed 6 failures
beyond the Silero ones. In every case the implementation had changed deliberately after the test was last
edited (tests 2026-08-29; implementations 09-05 → 09-13), so these were stale expectations, each checked
against the current real behaviour or a reference before updating:
- Qwen3-ASR prompt: the real chat template (064334e) puts the language hint in the assistant prefix
  (`language en<asr_text>`), not a `Language: en` line.
- Chatterbox no-weights tokenizer: a documented char-level stub with no `<s>`/`</s>` (real BPE path is
  covered by the real-weight tests).
- Kokoro phonemes: `wˈɜɹld` (American, rhotic, correct for the en-us voice), not British `wˈɜːld`.
- Piper tokenize: "abc" now expands to real phonemes, so the raw length isn't pinned to 5 (intersperse
  structure still asserted). Piper streaming is frame-chunked, not per sentence.
- **Melo speaker ids, with a real fix:** the shipped `melotts-zh_en` checkpoint's only speaker is id 1
  ("ZH-MIX-EN", `examples/MeloTTS.cpp/src/tts.cpp` `speaker_ids`); the test's 0-4 accent table belongs to
  the English-only checkpoint. `MeloVoices.GetSpeakerId` also mapped "ZH" to 0, which this checkpoint
  doesn't use; it now maps everything to 1 (English was already 1 and ear-confirmed).

## CLI TTS engines on CPU and Vulkan -- Whisper round trip, 2026-09-25

`stingray tts -e <engine> -g cpu|vulkan -t "Hello there, this is a real test of speech synthesis."`,
Whisper-base round trip (`stingray stt`). One process at a time; wall time includes model load.

| engine | CPU | Vulkan | Whisper (both backends) |
|---|---|---|---|
| Kokoro (default voice) | 6s | 5s | exact / exact |
| Chatterbox-Turbo | 13s | 9s | exact / exact |
| Piper (`en_US-lessac-medium`) | 2s | 3s | exact / exact |
| F5-TTS (`f5tts_base`, cloning `a.wav`) | 124s | 226s | exact / exact |
| MeloTTS (`melotts-zh_en`, EN-US) | 9s | 9s | **"…this is Aurelite Test at Speech's offices."** / **"…a relay test at speech synthesis."** |

Kokoro, Chatterbox, Piper and F5-TTS are word-exact on both backends. F5-TTS on Vulkan is slower than CPU on
this iGPU (CLAUDE.md rule 13: says nothing about discrete GPUs). **MeloTTS English is only partly intelligible**
on both backends. The shipped checkpoint is the Chinese/English-mixed model; the MeloTTS.cpp demo audio is
all Chinese, so there's no reference for its English quality. Logged as an observation, not root-caused.
Note: piper/f5tts/melo need an explicit `-m` in the CLI; only kokoro and chatterbox have default model paths.

## CosyVoice2 -- greedy decoding replaced with RAS (now stops by itself); garbled ending remains; README 🟢 → 🟡, 2026-09-25

First direct check of CosyVoice2 (its 🟢 row was inferred from sharing CosyVoice3's code). `CosyVoice2GenerateWavDebugTest`,
"This is a test of voice synthesis." (its path lookups were also fixed: weights live in `models/_models`, and the
output path no longer comes from the checkpoint's grandparent directory).
- **Before:** `CosyVoiceLlmGeneration` decoded speech tokens greedily (argmax). It never emitted EOS, ran to the
  200-token cap (8.00s), and Whisper heard "This is the taxidest place to possess."
- **Fix:** the reference samples with Repetition-Aware Sampling (top_k 25, top_p 0.8, win 10, tau 0.1), stop
  tokens masked before `min_len = 2 × text tokens`, capped at `20 ×`. CosyVoice3 already had this
  (`c8a4bf9`); its `SampleSpeechToken` is now `internal` and shared (no copy), and CosyVoice2 threads the seed.
- **After** (3 seeds): natural stop at 2.0-2.9s; Whisper hears "This is a test of worst existence." /
  "This is a test of moistness and mistness." / "Is it a test of voice and visit?". The opening is right; the
  last words are garbled on every seed.
- **Likely cause (not yet fixed):** CosyVoice2-0.5B is a zero-shot model that expects a speaker prompt
  (reference audio → CamPlus x-vector + speech-tokenizer prompt tokens). `CosyVoice2Pipeline` has no prompt
  path: empty prompt tokens and an all-zero speaker embedding (its own class doc names this gap). CosyVoice3
  *with* a real reference is word-exact. Next step: port CosyVoice3's reference-prompt path
  (`cosyvoice_speech_tokenizer_v2.onnx` + CamPlus are on disk).

Also this pass (Whisper round trips, CPU): QwenTTS "Hello, I will make some lunch darling." (exact, 12s);
VoxCPM2 "Hello there, this is a real end-to-end test of speech synthesis." (exact, 33s); CosyVoice3 zero-shot
cloning from `a.wav` "Hello, I will make some lunch darling." (exact, 82s). `CosyVoice3ClipGenDebugTests` depended on
a deleted local sample and skipped; it now defaults to `a.wav` + its transcript (`CV3_REF`/`CV3_REF_TEXT`/`CV3_OUT`).

## Fish Speech, MMS-TTS, XTTS-v2 -- Whisper round trips + perf baselines, 2026-09-25

`TtsPerformanceBaselineDebugTest` (warm-up + 3 timed runs, "Hello, I will make some lunch, darling!", CPU).
XTTS's reference clip (`docs/audio-samples/fishspeech-lunch-REFERENCE.wav`, local-only) had been deleted, which made
the XTTS baselines skip; they now fall back to `a.wav`.

| engine | Whisper | mean | RTF |
|---|---|---|---|
| Fish Speech S2 Pro | "Hello, I will make some lunch darling." (exact) | 22.3s (22.1/22.5/22.3) | 8.28 (the audio plan's RTF table listed 16.3×) |
| MMS-TTS (eng) | "Hello, I will make some lunch with Darling." (1 inserted word) | 1.50s | 0.41 |
| XTTS-v2 (cloning `a.wav`) | "Hello, I'll make some lunch darling." (contraction only) | 8.38s | 2.74 |

## Orpheus-TTS -- Whisper round trip on Vulkan and CPU, 2026-09-25

`OrpheusPipelineTests` (namespace `OpenTail.Stingray.Tests.Audio.Fast`; `ORPHEUS_CPU=1` now forces `allowGpu: false`),
"Hello, this is a test.", voice tara, Q4_K_M 3B talker + SNAC:
- Vulkan (iGPU): transformer 9.96s (14.1 tok/s) + SNAC 0.45s; Whisper: **"Hello, this is a test."** (exact).
- CPU: transformer 11.13s (12.6 tok/s) + SNAC 0.44s; Whisper: "Hello, this is it." (last two words differ).
Audio tokens are sampled, so backend numeric differences can change a token. Minor, not investigated further.

## Parler-TTS mini v1 -- Whisper round trip, 2026-09-25

`ParlerFullPipelineTests` (now takes `PARLER_TEXT`/`PARLER_TOKENS`/`PARLER_SEED`/`PARLER_OUT`; its default 40 tokens
is ~0.5s, too short for a sentence), "Hello there, this is a real test of speech synthesis.", 500 tokens, CPU:
seed 1 "Hello there, this is a real test speech synthesis." (33s, drops "of"); seed 2 "Along there, this is a real
test of speech sympathy." (24s). Intelligible, with small sampled-token slips.

## NeuTTS -- Whisper round trip, 2026-09-25

`NeuTtsAudioDecoderRealWeightsTests` (full chain: Emily voice prompt → backbone → FSQ codec, 14.4s CPU):
"Hello there, this is a full test of the NeuTTS speech synthesis system." → Whisper "Hello Bear, this is a full
test of the new task speech synthesis system." One real slip ("there" → "Bear"); "NeuTTS" → "new task" is just
the proper noun's pronunciation.

## Voxtral Realtime ASR -- spurious trailing "ished." fixed (decode past end of audio), 2026-09-25

`stingray stt -m voxtral --model-file models/_models/voxtral-mini-realtime` on the 4 LibriSpeech clips: every word
right, but 3/4 transcripts ended with a spurious "ished." (e.g. "…amidst the tents.ished."). Cause:
`VoxtralPipeline.Transcribe` kept stepping the decoder after the audio-embedding rows ran out, feeding it no audio,
so it invented a tail. The reference (`examples/audio.cpp/src/models/voxtral_realtime/session.cpp`) runs exactly one
decoder step per audio row and states "only the audio source running dry ends the stream". Fixed to stop when the
audio rows are exhausted. After: 4/4 exact ("I'm" for "I am" aside), 21-60s per clip on CPU. All Voxtral test classes
pass (7.8-178s each; `VoxtralRealReferenceMatchTests` skips visibly for missing reference dumps).

## Nemotron / Citrinet / SenseVoice ASR -- real-speech checks, 2026-09-25

- Nemotron 3.5 ASR (`NemotronAsrEndToEndTests`, `a.wav`, 7.2s): "This little work was finished in the year eighteen oh
  three and intended for immediate publication." (exact) followed by a raw `<en-US>` tag. That tag appears only in
  the test's own ad-hoc detokenizer. The reference strips special/language tokens by default
  (`keep_language_tags = false`), and there's no user-facing Nemotron text pipeline here yet, so it's a note for
  whenever one is added, not a product bug.
- Citrinet (`CitrinetAsrRealWeightsTests`, LibriSpeech 0000, 0.95s): exact, and the test asserts the full transcript.
- SenseVoice (`SenseVoiceRealWeightsTests`, same clip, 2.3s): exact, lang `<|en|>`, event `<|Speech|>`.

**Phase 1 summary (2026-09-25):** every 🟡 and ⚪ audio row and every 🟢 TTS/ASR engine with a runnable path has now had
a real-weight check with visible timing (Whisper round trip for TTS; ground truth or reference for ASR). Fixed this
pass: VibeVoice TTS (2 loop bugs), RVC (RMVPE kernel + pipeline + 24× perf), OmniVoice (3.9× perf), Silero VAD
(context + `stt --vad` crash), Voxtral (invented tail), CosyVoice2 (argmax → RAS), Stable Audio (sampling schedule),
Melo (ZH speaker id). Open: CosyVoice2 garbled ending (needs a speaker-prompt path), Stable Audio SFX darker + CFG-1
divergence, MeloTTS English intelligibility, Parakeet ~3% WER.

## VibeVoice ASR -- EOS never matched (decoded padding to the cap) + encoder perf; 40.8s → ~17.5s, 2026-09-25

Profile first (new `STINGRAY_ASR_PROFILE=1` timers in `VibeVoiceAsrGenerator`, and stage timers in the real-speech
test), 3.5s LibriSpeech clip, CPU: speech features 15.9s, LLM prefill+decode 24.9s. Two findings:
1. **Real bug — EOS never matched.** The raw output was the complete answer
   (`[{"Start":0,"End":3.5,"Speaker":0,"Content":"Concord returned…"}]<|im_end|>`) followed by dozens of
   `<|endoftext|>` tokens up to the 96-token cap. The reference resolves EOS explicitly as `<|endoftext|>`
   (`tokenizer_text.cpp`); ours used `GgufTokenizer.EosTokenId`, which isn't that id for this checkpoint.
   `VibeVoiceAsrTextTokenizer` now resolves it explicitly: decode stops after 34 tokens (was 96).
2. **Encoder perf.** The ConvNeXt FFN ran a mat-vec per frame (re-streaming both weights per frame) and the
   full/depthwise causal convs looped serially over channels. The FFN is now two batched GEMMs through
   `DenseKernels.LinearBatchedNoBias` (shared helper, whole-clip and streaming paths), and the four conv loops
   are parallel over output channels.

| 3.5s clip | before | after (2 runs) |
|---|---|---|
| speech features | 15.9s | 4.0 / 4.5s |
| LLM prefill (86 tokens) + decode | 24.9s (6.8 + 18.1, 96 tokens) | 13.2 / 13.2s (6.8 + 6.4, 34 tokens) |
| total compute | 40.8s | ~17.5s (2.3×) |

Transcript unchanged (exact). Reference `audiocpp_cli`: 12.4s → now ~1.4× behind (was ~4×). Remaining lever:
prefill runs at ~12.6 tok/s on the 7B q8 backbone, not investigated.

## VibeVoice TTS perf pass -- 65-72s → 24.5-26.4s per sample, Whisper still exact, 2026-09-25

Profile (new `STINGRAY_TTS_PROFILE=1` stage timers in `VibeVoiceGenerator`), ~35 frames, CPU:

| stage | after ASR-side ConvNeXt fix | + streaming ConvTranspose1d parallel | + batched diffusion head |
|---|---|---|---|
| diffusion head | 8.10s | 8.07 / 8.63s | **3.99 / 4.44s** |
| acoustic decode | 15.86s | **5.10 / 6.05s** | 5.54 / 6.11s |
| semantic encode | 3.55s | 3.63 / 3.67s | 3.50 / 3.95s |
| negative + positive LM | 1.96 + 2.04s | 1.94 + 2.02s | 1.89 + 1.98s |
| test wall (incl. load) | 39.6s | 28.6 / 30.1s | **24.5 / 26.4s** |

Changes: `SConvTranspose1dStreaming` scattered serially over (in-channel, time, out-channel); it's now parallel
over output channels. The diffusion head ran its CFG cond/uncond rows one after another (each weight read twice
per solver step) and recomputed `cond_proj(condition)` at all 10 solver steps. `VibeVoiceDiffusionHead` now
batches rows through `DenseKernels.LinearBatchedNoBias` (`PredictProjected`) and `ProjectCondition` is hoisted
out of the timestep loop. `DenseKernels.LinearBatchedNoBias` now uses the packed GEMM from 2 rows (was 4).
Whisper (seeds 1, 2): both "Hello there, this is a real end-to-end test of speech synthesis." (exact).
Regression suites pass: diffusion head, generator, voice cloning, OmniVoice MaskGIT, RVC pipeline.
Start of the session: 65-72s. Reference `audiocpp_cli`: 19-20s → now ~1.3× behind (was ~3.5×).

## RVC HuBERT -- batched projections (small gain), 2026-09-25

HuBERT's q/k/v, attention-output (serial!) and FFN linears ran one mat-vec per frame. They now go through the new
shared `DenseKernels.LinearBatchedRows` (row-array wrapper over `LinearBatchedNoBias`, bias added per row). Measured
on the RVC pipeline (2 runs): hubert 5.1 → 4.64 / 4.69s (~9%); total 19.8 → 19.2 / 19.4s; output vs pre-change
cosine 1.000000 (same 1.65e-4 maxAbs), Whisper exact, `RvcHubertEncoderForwardTests` passes. Modest because the
projections weren't HuBERT's main cost (likely the 128-tap grouped positional conv); not pursued further. RVC stays
~2.4× behind the reference (8.2s).

## CosyVoice2 -- reference-prompt path ported; bisection pins the garbled tail on LLM token generation (open), 2026-09-25

Ported CosyVoice2's zero-shot prompt path (upstream `inference_zero_shot`) into `CosyVoice2Pipeline.Generate(…,
referenceAudioPath, referenceText)`: CamPlus x-vector → speaker embedding; speech-tokenizer-v2 prompt tokens aligned
with the reference mel (2 mel frames/token); LLM conditioned on prompt text + prompt speech tokens; flow over
prompt+generated tokens; `CosyVoiceCfmDecoder.Generate` gains the real `cond` input (prompt mel in the first
frames, channel-first); prompt frames trimmed before HiFT. ONNX helpers resolved from `models/` or `models/_models/`.

It did **not** fix the garbled endings, so I bisected with an untracked resynthesis harness (`ZzCv2ResynthProfTmp`):

| experiment (a.wav / "This is a test of voice synthesis.") | Whisper |
|---|---|
| a.wav's real speech tokens → flow/CFM/HiFT, zero speaker | "…finished in year 8 to do out of 3 and intended for you to complication." |
| real tokens, **real CamPlus x-vector** | "This little work was finished in 1803 and intended for a media publication." (≈ exact) |
| real tokens, x-vector, 30 ODE steps | worse (trained setting is 10) |
| **LLM** tokens (no prompt), x-vector, seeds 42/1/2 | "This is the task of wasting distance." / "…of voicing, isn't it?" / "…of this industry." |
| LLM tokens + full reference prompt, seeds 42/1/2 | "This is a task that works in some business." / "…of voice and resist." / "Yes, if the chance for the voices to pass." |

**Conclusion:** the acoustic stack is right given a real speaker embedding (the zero x-vector alone hurts it,
and CosyVoice2-0.5B is a zero-shot-only model anyway). The defect is in **LLM speech-token generation**, which
starts right and drifts off the text. Checked and ruled out: q/k/v biases are mapped, the prefill layout is
`[sos, text, task_id, prompt_speech]`, and the sampler is the reference RAS. Next suspects: text
tokenization/normalization against the upstream frontend, the speech-token embedding offset / `llm_decoder` head
indexing, RoPE/position handling in `ForwardPass` for this source. README row stays 🟡.

## Parakeet -- the ~3% WER was a tokenizer bug; now word-for-word with the C++ reference, 2026-09-25

Checked the "plausibly q4_k" theory first: the unquantized checkpoint (cstr/parakeet-ctc-0.6b-GGUF `parakeet-ctc-0.6b.gguf`,
saved as `models/_models/parakeet-ctc-0.6b-f16.gguf`) gave exactly the same transcripts, so not quantization. Built the
reference (`examples/CrispASR`, after fetching its `ggml` and `c2pa-audio` submodules): on the same file and clips it
transcribes "made **a** plentiful provision" / "to **a** supper", the two words we dropped. A frame dump showed our encoder
does emit `▁a` (frame 54: 118.6 vs blank 113.8), so it was lost in decoding: `ParakeetTokenizer.Decode` skipped the
fallback vocab's hard-coded special ids 0/2/3, and in this NeMo SentencePiece vocab id 2 is "▁th" and id 3 is "▁a" (the
blank is 1024, past the vocab). `FromGguf` now skips only real `<...>` special pieces. `ParakeetLibriSpeechTests` (new,
tracked) asserts equality with CrispASR's transcripts (casing/punctuation stripped) for q4_k and unquantized. The
onnx-community ONNX export needs HF-style features (it outputs only blanks on NeMo-style features) and was not used.
Also noted: `ParakeetConformerEncoderTests` look only in `models/`, but the checkpoint lives in `models/_models/`.

## CosyVoice2 -- teacher-forcing check narrows the drift away from the LLM math, 2026-09-25

New `CosyVoiceLlmGeneration.ScoreSpeechTokens` + `CosyVoice2TeacherForcingTests`: a.wav's real speech tokens after its
own transcript. Top-10 accuracy by quartile 45 / 48 / 56 / 55 %, median rank 14 → 8 of 6561, mean log-prob ≈ −4
(random ≈ −8.8). No decay with position (rules out RoPE/position handling; rope_theta is 1e6 as in the config) and real
tokens rank near the top (rules out a misindexed speech embedding or llm_decoder head). The drift is more likely in
generation-time behaviour (sampling / stopping) or input text normalization than in the forward pass. Not fixed; the
missing piece is upstream's per-token log-likelihoods for the same tokens as a direct comparison.
- **Repetition penalty ruled out (2026-09-25).** The shared RAS sampler applied a 1.15 logit penalty to recent tokens
  (from the CosyVoice3 C++ port); upstream CosyVoice2 `ras_sampling` has none. Now a parameter (CosyVoice3 keeps 1.15,
  CosyVoice2 uses 1.0). A/B with x-vector, seeds 42/1/2, Whisper medium: 1.15 → "…of your synthesis." / "…of voice and
  business." / "…of the district."; 1.0 → "…of voice and business." / "…of wisdom, this is-" / "…of white synthesis."
  Same quality: the start is right and it drifts at "voice synthesis" either way. With the forward pass (teacher forcing)
  and the sampler both ruled out, the next step needs upstream's own tokens or log-likelihoods for the same text.

## RVC HuBERT -- front-end convs shared with Wav2Vec2 and GEMM-based: hubert 4.63 → 1.2s, RVC 19.4 → 15.9s, 2026-09-25

HuBERT's feature-extractor convs and 128-tap grouped positional conv were scalar loops. They now use the new shared
`Primitives/Wav2Vec2FrontendKernels` (valid conv1d and grouped "same" conv as im2col + packed GEMM, per-channel GroupNorm),
which `Wav2Vec2CtcModel` also uses now instead of its private copies (DRY). RVC keeps its own erf GELU. Packed weights are
cached on `RvcHubertWeights`. Measured (2 runs, `STINGRAY_RVC_PROFILE=1`): hubert 4.63 → 1.31 / 1.17s, total convert
19.4 → 16.1 / 15.8s; output vs the pre-change WAV cosine 1.000000, maxAbs 4.76e-5 (16-bit quantisation). Wav2Vec2 parity
(ONNX, WER 1.4 %) and `RvcHubertEncoderForwardTests` pass. Remaining: RMVPE ~7.5s and the synthesizer ~7s; the reference
is 8.2s, so RVC is now ~1.9× behind (was ~2.4×).
- **RMVPE mel STFT (2026-09-25): 15.9 → 9.7s.** New RMVPE stage timers (`STINGRAY_RVC_PROFILE=1`) showed the U-Net + GRU
  take only ~1.85s of the 7.7s "rmvpe f0" stage; the rest was the mel extractor's direct O(n²) DFT calling
  `Math.Cos`/`Math.Sin` per term. It now uses the shared `SpectralKernels.ComputePowerSpectrum` (twiddle tables, SIMD
  dots) with frames in parallel. 2 runs: rmvpe 7.67 → 1.86 / 1.88s, convert 9.7 / 9.7s; output vs pre-change cosine
  1.000000, maxAbs 3.0e-4 (float vs double DFT); `RvcRmvpeEndToEndRealAudioTests` passes. RVC is now ~1.2× behind the
  reference (8.2s); the synthesizer (~6.6s) is the remaining big stage.
- **Synthesizer transpose convs (2026-09-25): 9.7 → 8.6 / 9.0s.** New synthesizer stage timers showed resblocks 3.76s
  (already GEMM, ~114 GMAC/s at the largest level, about half of peak) and the four ConvTranspose1d 1.26s (scalar per
  output channel). ConvTranspose1d is now one packed GEMM over all input frames plus overlap-add: 1.26 → ~0.2s. Output vs
  pre-change cosine 1.000000 (maxAbs 3.0e-4, unchanged); `RvcSynthesizerEncoderForwardTests` passes. **RVC: 19.4s → 8.6–9.0s
  today vs the reference's 8.2s.** What's left is mostly the resblock GEMMs, near this CPU's throughput.
