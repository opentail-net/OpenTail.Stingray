# Audio proof samples: speech, music and sound effects

Real, weight-driven audio from OpenTail Stingray's native C# engines (no Python, no cloud). Regenerated **2026-10-10** on a Ryzen 5700G with integrated graphics (no discrete GPU), one model at a time. Click a link to play the WAV. Times are wall-clock for the generation step only, and a discrete GPU will be much faster.

## Music (text to music)

| Engine | Prompt | Length | Time | Sample |
|---|---|---|---|---|
| **ACE-Step 1.5 Turbo** (Vulkan DiT) | "An upbeat electronic synthwave track with retro bass and energetic drums" (seed 1234) | ~2.7s | 12.6s | [music-acestep15-turbo-synthwave.wav](music-acestep15-turbo-synthwave.wav) |
| **Stable Audio 3 Small (music)** | "A smooth lo-fi jazz beat with soft electric piano and mellow vinyl crackle" (16 steps, seed 42) | 4s, 44.1kHz stereo | see note | [music-stableaudio3-small-lofi.wav](music-stableaudio3-small-lofi.wav) |
| **Stable Audio 3 Medium** | "A cinematic orchestral crescendo with sweeping strings and timpani" (12 steps, seed 42) | 4s, 44.1kHz stereo | see note | [music-stableaudio3-medium-orchestral.wav](music-stableaudio3-medium-orchestral.wav) |
| **MiniMax-Music3** (with vocals) | "Intimate acoustic folk, male vocal, fingerpicked guitar", lyrics "[Verse] Walking through the morning rain" (200 frames, seed 42) | 8s, 44.1kHz stereo | 564s (CPU) | [music-minimax-music3-folk-verse.wav](music-minimax-music3-folk-verse.wav) |
| **MusicGen Small** | "upbeat acoustic guitar melody, happy" (seed 42) | 5s, 32kHz | see note | [music-musicgen-small.wav](music-musicgen-small.wav) |

Note: the Stable Audio 3 checkpoints on this machine are the pre-trained `-base` models, which sound rougher than the released post-trained ones. The three test runs (ACE-Step, SA3 Small, SA3 Medium) took 92s together, including model loads.

## Sound effects (text to sound)

| Engine | Prompt | Length | Sample |
|---|---|---|---|
| **Stable Audio 3 Small (SFX)** | "a glass bottle shattering on a hard floor" (3s) | 3s, 44.1kHz stereo (47.7s run) | [sfx-stableaudio3-small-glass-shatter.wav](sfx-stableaudio3-small-glass-shatter.wav) |
| **AudioGen Medium** | "heavy rain falling on a metal roof" (seed 42) | 5s, 16kHz | [sfx-audiogen-medium-rain-on-metal-roof.wav](sfx-audiogen-medium-rain-on-metal-roof.wav) |
| **AudioGen Medium** | "a dog barking twice" (seed 7) | 5s, 16kHz | [sfx-audiogen-medium-dog-barking-twice.wav](sfx-audiogen-medium-dog-barking-twice.wav) |

(AudioGen and MusicGen run together in 98s and 20s respectively.)

## Speech (text to speech)

Every clip says **"Hello from {model} on OpenTail Stingray."** (the Chatterbox clip is the original, which set the pattern). Times are for this iGPU-only machine; RTF is generation time divided by audio length, so under 1x is faster than real time.

| Engine | Audio | Time (RTF) |
|---|---|---|
| Chatterbox-Turbo | [chatterbox-sample.wav](chatterbox-sample.wav) | 2.68s in 5.3s (2.0x) |
| Piper VITS (Lessac) | [piper-lessac-sample.wav](piper-lessac-sample.wav) | 2.39s in 1.2s (0.49x) |
| MMS-TTS | [mms-eng-sample.wav](mms-eng-sample.wav), [tts-mms.wav](tts-mms.wav) | 3.65s in 2.1s (0.58x) |
| Kokoro 82M (`af_heart`) | [kokoro-af_heart-sample.wav](kokoro-af_heart-sample.wav) | 2.78s in 3.9s (1.4x) |
| MeloTTS (zh/en) | [melotts-zh_en-sample.wav](melotts-zh_en-sample.wav) | 2.91s in 4.9s (1.7x) |
| Qwen3-TTS 0.6B | [tts-qwen3-tts.wav](tts-qwen3-tts.wav) | 3.92s in 11.2s (2.9x) |
| CosyVoice 3 | [tts-cosyvoice3.wav](tts-cosyvoice3.wav) | 4.68s in 20.2s (4.3x) |
| Orpheus 3B + SNAC | [tts-orpheus.wav](tts-orpheus.wav) | 3.24s in 19.7s (6.1x) |
| Parler-TTS Mini | [tts-parler.wav](tts-parler.wav) | 2.98s in 23.4s (7.9x) |
| Fish Speech S2 Pro (Q8_0) | [tts-fish-speech-s2-pro.wav](tts-fish-speech-s2-pro.wav) | 3.07s in 32.7s (10.7x) |

**Speech-to-text check.** Whisper Base transcribes "Hello from" correctly in every clip, but "OpenTail Stingray" is a coined word that it often mis-hears (for example "Open Tail Stingray" or "OpenTale"). Fish Speech comes back as "Hello from Fish Speech on Open Tail Stingray." and Orpheus as "Hello from Orpheus on OpenTalestingRay.". An earlier run with the plain sentence "Hello world, this is a speech synthesis test." gave a word-exact Whisper transcript for Qwen3-TTS, CosyVoice 3, Orpheus, Parler and Fish Speech (MMS dropped one word), and Parakeet CTC 0.6B agreed on the Qwen3-TTS clip. Those earlier clips were replaced by these ones.

## How to regenerate

Speech and ASR use the CLI (`-o` sets the output file):

```powershell
stingray tts -e qwentts -t "Hello from Qwen3 TTS on OpenTail Stingray." -o tts-qwen3-tts.wav
stingray tts -e cosyvoice|orpheus|parler|mms ...            # same shape, -t "Hello from <model> on OpenTail Stingray."
stingray tts -e fish -m K:/_other_models/s2-pro-q8_0.gguf -t "..." -o tts-fish-speech-s2-pro.wav
stingray stt -m base --model-file models/ggml-base.bin -i tts-qwen3-tts.wav
```

Music and sound-effect generation has no CLI command yet. These samples come from the sample-writing tests, run with `STINGRAY_RUN_HEAVY_TESTS=1` against the built test executables:

| Samples | Test class |
|---|---|
| ACE-Step, Stable Audio 3 Small/Medium | `OpenTail.Stingray.Tests.Diffusion.GenerateAudioDiffusionSamplesTests` |
| Stable Audio 3 SFX | `OpenTail.Stingray.Tests.Diffusion.ZZ_ScratchStableAudioScheduleFixRegenTests` (`Regen_SfxGlassShatter_OfficialRecommendedParams`) |
| MiniMax-Music3 | `OpenTail.Stingray.Tests.Diffusion.MiniMaxMusic3.ZZ_ScratchMiniMaxMusic3GenerateSampleTests` |
| MusicGen, AudioGen | `OpenTail.Stingray.Tests.Audio.MusicGen.MusicGenGenerationSmokeTests`, `...AudioGen.AudioGenGenerationSmokeTests` |

Image and video samples are in [diffusion-samples](../diffusion-samples/README.md). Verification status per engine is in [STATUS.md](../STATUS.md).
