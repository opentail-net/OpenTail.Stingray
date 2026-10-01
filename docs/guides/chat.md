# Guide: chat with a language model

**You get:** a local chat assistant or one-shot text generation, on the CPU or a GPU, from the
command line or from C#.

**You need:** one `.gguf` model file. For a first run, `stingray setup chat` downloads a checked
Qwen2.5 0.5B (469 MB). To pick something bigger, see [MODELS.md](../MODELS.md); each model's
memory need and speed is in [RUNNING.md](../RUNNING.md).

## Run it

```bash
stingray -m models/qwen2.5-0.5b-instruct-q4_k_m.gguf                      # interactive chat
stingray -m models/qwen2.5-0.5b-instruct-q4_k_m.gguf -p "What is a unit test?"   # one prompt
stingray pull -r Qwen/Qwen3-4B-GGUF                                       # download another model
```

Without `-p` you get an interactive chat. With `-p` it answers once. `--single-turn` forces one
answer and exit.

## The options you will actually use

| Option | What it does |
|---|---|
| `--system-prompt "..."` | Sets the system message |
| `-n 256` | Maximum tokens to generate (default 512) |
| `--temp 0` | Greedy, repeatable answers. The default is 0.7 |
| `--top-k`, `--top-p`, `--min-p`, `--seed` | The rest of the sampling controls; a fixed `--seed` makes a run repeatable |
| `-c 4096` | Context size (prompt plus answer) |
| `-g -1` | Put all layers on the GPU (`-g 0` is CPU only; `-g 20` offloads 20 layers) |
| `--repeat-penalty 1.1` | Discourages loops |
| `-f prompt.txt` | Read a long prompt from a file |
| `--chat-template` | Override the model's built-in chat template |

`stingray --help` lists everything. The flag names follow llama.cpp's `llama-cli` where they mean
the same thing.

## Pick CPU or GPU

- **Small models (under about 1B) are usually faster on the CPU** than on an integrated GPU.
- A larger model on a real GPU can be much faster. Try `-g -1`, and compare against `-g 0`.
- The measured speeds, per model and per backend, are in [RUNNING.md](../RUNNING.md).

## Reasoning models

Some models think before answering. `--thinking` and `--no-thinking` switch the mode where the
model supports it, `--hide-thinking` shows only the answer, and `--max-thinking-tokens N` caps the
trace. Greedy decoding can loop on these, so for long answers use `--temp 0.6 --top-p 0.95
--top-k 20`.

## From C#

The [README quick start](../../README.md) is a complete, compiled program (also
[samples/QuickStart](../../samples/QuickStart/Program.cs)). The chat sample
[samples/OpenTail.Stingray.Sample.Chat](../../samples/OpenTail.Stingray.Sample.Chat) shows a
multi-turn conversation.

## If it goes wrong

Slow, out of memory, repeating itself, or a model that will not load:
[TROUBLESHOOTING.md](../TROUBLESHOOTING.md).
