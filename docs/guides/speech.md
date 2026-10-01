# Guide: speech, both ways

**You get:** text to speech (a `.wav` file) and speech to text (a transcript, with timestamps or
subtitles).

## Text to speech

```bash
stingray tts -e piper -m models/en_US-lessac-medium.onnx -t "Hello world" -o speech.wav
stingray tts -e kokoro -m models/_models/kokoro-82m-q8_0.gguf -v af_heart --voices-dir models/_models -t "Hello world" -o speech.wav
```

`-e` picks the engine. The available ones are `kokoro` (default), `piper`, `f5tts`, `chatterbox`,
`melo`, `cosyvoice`, `parler`, `qwentts`, `fishspeech`, `orpheus`, `mms` and `xtts`.

| Pick | When | Speed on the reference CPU (shorter is faster) |
|---|---|---|
| **Piper** | You want it fast and light | Faster than real time (about 0.4x of the audio length) |
| **Kokoro** | Good quality, many voices | About 1.3x the audio length |
| **Chatterbox-Turbo** | More expressive | About 2x |
| **Parler-TTS** | Describe the voice in words | About 7x |
| **Fish Speech S2 Pro** | Highest quality here | About 11x |
| **F5-TTS, XTTS-v2** | **Clone a voice** from a short sample | See [RUNNING.md](../RUNNING.md) |

"About 2x" means generating 1 second of audio takes about 2 seconds. A GPU (`-g`/`--backend`)
changes this; measured rows are in [RUNNING.md](../RUNNING.md).

Useful options: `-v` voice, `-s` speed multiplier, `--seed` for repeatable output on the engines
that sample, and for F5-TTS voice cloning `--ref-audio clip.wav --ref-text "what the clip says"`.
`stingray tts --help` has the full list.

## Speech to text

```bash
stingray stt -m base --model-file models/ggml-base.bin -i speech.wav
stingray stt -m base --model-file models/ggml-base.bin -i speech.wav -t translate   # to English
stingray stt -m parakeet --model-file models/parakeet-ctc-0.6b-q4_k.gguf -i speech.wav
```

`-m` picks the model: Whisper `tiny` (default), `base`, `small`, `medium`, `large-v3`, `turbo`;
or `parakeet`, `voxtral`, `sensevoice`, `paraformer`.

| Pick | When | Speed on the reference CPU |
|---|---|---|
| **Parakeet** | Fast English transcription | About 11x faster than real time |
| **SenseVoice / Paraformer** | Chinese and multilingual, very fast | 9x to 21x faster than real time |
| **Whisper base / small** | General use, 100 languages | About 2x to 3x faster than real time |
| **Whisper medium / large-v3** | Best Whisper accuracy | 1.3x and 0.7x of real time (large-v3 is slower than the audio) |

Options: `-l` language code, `-t translate` (to English), `--no-timestamps`, `--vad` (Silero voice
detection to skip silence), `-o file` to save the transcript or subtitles.

**Audio format:** the input is a 16 kHz WAV. Convert other formats first (for example with ffmpeg).

## Close the loop

Generate with Piper, then transcribe with Whisper. They agree word for word on the test clip, which
is a quick way to check your setup works end to end.

## From C#

The [README](../../README.md) has both calls (`PiperPipeline`, `WhisperPipeline`) as short, tested
snippets.

## Not wired to a command yet

Some audio models are library-only: PersonaPlex (duplex speech), RVC (voice conversion) and the
music and sound models. [WHAT-YOU-CAN-DO.md](../WHAT-YOU-CAN-DO.md) lists which.
