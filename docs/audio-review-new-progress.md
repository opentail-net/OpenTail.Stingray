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
