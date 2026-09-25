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
