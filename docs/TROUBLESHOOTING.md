# Troubleshooting

Common problems, what causes them and what to do. Every item here was met while running the
models in [RUNNING.md](RUNNING.md). If yours is not listed, start with the three diagnostic commands
at the bottom.

## It runs, but slowly

**The log says `OpenBLAS: not found (fallback to sequential)`.** CPU decoding is about 6x slower and
prefill about 8x slower without it (26 tok/s fell to 4 tok/s on the test model). Install the
library as described in [reference/openblas-troubleshooting-guide.md](reference/openblas-troubleshooting-guide.md).
A healthy start prints `OpenBLAS: LOADED`.

**A small model is slower with `-g -1` (GPU) than without.** Small models lose to the CPU on an
integrated GPU, because each GPU call has a fixed cost. On the reference machine Qwen2.5 0.5B
decodes at about 26 tok/s on the integrated GPU and 63-69 tok/s on the CPU. Use the CPU (`-g 0`, or
leave `-g` off) for small models. Only a larger model on a real GPU is likely to gain; the GPU rows in
[RUNNING.md](RUNNING.md) show what was measured on the integrated GPU.

**The first run is much slower than the next ones.** Weights are memory-mapped, so a cold file
cache makes the first run read from disk (one 590M model decoded at 21 tok/s cold and 60 tok/s
warm). Run it twice before judging a speed.

**Prompt processing is slow on a hybrid model** (Granite 4.0-H, Nemotron-H, LFM2). Batched prompt
processing is on by default since 2026-09-28. `STINGRAY_RECURRENT_BATCHED_PREFILL=0` turns it back
off if you are comparing.

## It runs out of memory

- Check the peak RAM in [RUNNING.md](RUNNING.md) for your model before you start. A 30B
  mixture-of-experts model peaks at about 18 GB.
- **Close other heavy programs.** Two 30 GB processes at once is what exhausted the 64 GB test
  machine, not either alone.
- A mixture-of-experts model's experts are loaded up front only if they fit in about 80% of the RAM
  that is free at that moment; otherwise they are paged in as needed, which is slower but works.
  `STINGRAY_PREFAULT=1` forces everything in, `STINGRAY_PREFAULT=0` loads nothing up front.
- A smaller quantization (a Q4_K_M file rather than Q8_0) uses less memory.

## The output is wrong

**Repetitive nonsense from the start.** Some models (LFM2) need the start-of-text token, which
Stingray adds automatically when the model's metadata asks for it. Do not strip it off your prompt.

**A reasoning model prints a long "thinking" trace, loops, or never reaches an answer.** Models such
as Ornith and Qwen3.8 print a thinking trace first, so raise `-n` to see the answer. Greedy
decoding (`--temp 0`) can loop on reasoning models; the CLI prints a warning about this for some
of them (Granite 4.0-H Small did) and suggests `--temp 0.6 --top-p 0.95 --top-k 20`.

**An image-aware model ignores or rejects `--image`.** You need both the model and its
`mmproj-*.gguf` file, passed with `--mmproj`. Qwen-VL-style models (Qwen2.5-VL, Qwen3-VL,
PaddleOCR-VL) and Granite 4.0 Vision work on the CPU, and Qwen3-VL also on a fully offloaded
Vulkan run. On other GPU setups the CLI refuses the image instead of answering wrongly.

**A transcript is empty.** A pure tone or silence gives an empty transcript on purpose. Test with
real speech.

## It will not load

**`GGUF architecture 'xyz' is not supported`.** Stingray only runs architectures it has verified.
Check [STATUS.md](STATUS.md) for your model family. `stingray capabilities` lists what is supported
in the version you have installed, and `stingray inspect -m <file>` says what the file contains
and whether it is compatible. The open gaps are in [00-current-work.md](00-current-work.md).

**The file is not the model its name says.** One real example: a root `paraformer-q8.gguf` was
actually a Fun-ASR-Nano checkpoint. `stingray inspect -m <file>` shows the architecture and name
stored inside the file, which is the truth.

**A relative path is not found.** Commands in these docs are run from the repository root, and
relative paths such as `models/_models/...` resolve from there. Use absolute paths if you run it
from somewhere else.

**An image or video model complains about a missing file.** These models need several files each
(text encoders, a VAE, tokenizers), and the command lists them all. Use the exact command in
[RUNNING.md](RUNNING.md) as your template.

## Speech and audio

**The voice sounds wrong or garbled at the end of a sentence.** CosyVoice 2 is known to garble
endings ([STATUS.md](STATUS.md) rates it partial). Use CosyVoice 3, Piper, Kokoro or Fish Speech
instead.

**Text-to-speech is slower than real time.** That is expected for the larger engines on a CPU
(Kokoro about 1.3x, Chatterbox 2x, Parler 7x, Fish Speech 11x of the audio length on the reference
machine). Piper is faster than real time.

## Diagnostics

```bash
stingray doctor -m <model.gguf>    # runtime, backends and the model, without running inference
stingray inspect -m <model.gguf>   # identity, compatibility and capabilities of a file
stingray list-env                  # the STINGRAY_* settings this process sees (flags unknown ones)
```

When you report a problem, include the exact command, the model file name and size, and a support
bundle: `stingray doctor --bundle support.zip` writes a redacted one to attach. `--deep` adds
memory-allocation and backend smoke tests, and `--json` gives machine-readable output.
