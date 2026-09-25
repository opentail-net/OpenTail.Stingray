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
