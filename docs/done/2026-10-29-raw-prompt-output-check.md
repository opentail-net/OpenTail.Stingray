# Raw-prompt output check: gpt2, starcoder2, xverse, gpt-oss (2026-10-29)

Closes backlog item §1.16 of [00-current-work.md](../00-current-work.md): "output correctness of gpt2, starcoder2, xverse
and gpt-oss is unverified (2026-10-08 smoke sweep)". That sweep saw odd first output (`<|imp…`, `<`, `<`, `<`) and
compared nothing with a reference.

**Result: no output bug in any of the four.** gpt2, starcoder2 and xverse produce exactly llama.cpp's greedy text.
gpt-oss matches llama.cpp's top-10 next-token ranking and diverges only at a near-tie. The odd sweep output was the
CLI's **ChatML fallback wrapper**, not the models.

## Method

- Machine: Ryzen 7 5700G (8c/16t, AVX2, DDR4), CPU only (`-g 0` / `-ngl 0`).
- Reference: llama.cpp b10306 (`6b5c2efb4`), vendored `tools/llama.cpp`.
- Prompt `The capital of France is` (starcoder2: `def fibonacci(n):`), greedy, `--repeat-penalty 1.0`, 32 tokens.
- llama.cpp: `llama-completion -no-cnv --no-display-prompt -t 8 --temp 0` (raw completion).
- Stingray: `stingray -g 0 --temp 0 --repeat-penalty 1.0` with a pass-through `--chat-template` that emits only the user
  text. `STINGRAY_RAW_PROMPT=1` is equivalent.
- gpt-oss near-tie: `llama-server` `/completion` with `n_probs` against Stingray `--verbose-prompt` (top-10 log-probs
  at the first generated position).

## Output

| Model | Checkpoint | Result |
|---|---|---|
| gpt2 | `gpt2.Q8_0.gguf` | **Identical** 32-token text. |
| starcoder2 | `starcoder2-3b.Q4_K_M.gguf` | **Identical** 32-token code completion. |
| xverse | `xverse-7b-chat-q4_k_m.gguf` | **Identical**: ` not a country.`, then end-of-text after 4 tokens in both. |
| gpt-oss | `gpt-oss-20b-MXFP4.gguf` | Same first token (` Paris`); diverges at token 2 on a near-tie (below). |

**gpt-oss near-tie.** After `The capital of France is Paris`, the two leading candidates are `."\n` (14396) and `."`
(3692):

| | `."\n` | `."` | `.` |
|---|---:|---:|---:|
| llama.cpp (`llama-server`) | **−1.8311** | −1.8329 | −2.4853 |
| Stingray | −1.9100 | **−1.8468** | −2.3650 |

llama.cpp's margin is 0.002 nats. Both engines have the same top-3 tokens, and Stingray's log-probs are within 0.08
nats of llama.cpp's. Forcing the same token (`…Paris."`), the next-token top-10 has the same ten ids as llama.cpp.
The order matches except places 6-7 (` This`/` ->`), which are 0.005 nats apart in Stingray. Examples: `\n\n` −2.608
vs −2.646, ` in` −2.961 vs −2.991. This is the noise level the admission evidence already records
(teacher-forced vs `llama-server`: 22/24 exact argmax, worst gap 0.063 logits, inside llama.cpp's own `-fa` on/off
noise; STATUS gpt-oss row). It is unaffected by `STINGRAY_CPU_PREFILL_Q8=0`: gpt-oss does not use that tier.

## gpt-oss through its chat template

The backlog asked for gpt-oss to be checked with its chat template too. For "What is the capital of France?" the
prompt Stingray renders from the GGUF's embedded template was compared with `llama-server --jinja`
`/apply-template`. They differed in one place: `Current date: ` was empty in Stingray and `2026-10-29` in llama.cpp.
The template calls `strftime_now("%Y-%m-%d")`, which `JinjaChatTemplate` did not implement, so the call rendered as
empty. Llama 3.1's template uses the same function (`"%d %b %Y"`).

**Fixed in the same change:** `strftime_now` is now implemented with local time and a C-locale `strftime` subset
(`JinjaChatTemplate.Strftime`, `JinjaStrftimeNowTests`). After the fix the rendered prompts are identical. Stingray's
greedy answer starts `<|channel|>analysis<|message|>The user asks: …`, the expected harmony analysis channel.

## Speed (single runs unless noted, CPU, 8 threads)

| Model | Stingray prefill / decode t/s | llama.cpp prefill / decode t/s | Decode ratio |
|---|---:|---:|---:|
| gpt2 Q8_0 | 105.9 / 184.5 | 938.6 / 273.7 | 0.67x |
| starcoder2-3b Q4_K_M | 20.3 / 15.3 | 64.8 / 19.1 | 0.80x |
| xverse-7b Q4_K_M | 10.8 / 8.0 (4 tokens) | 46.8 / 10.0 (4 tokens) | 0.80x |
| gpt-oss-20b MXFP4 | 6.4-6.7 / 12.0-12.1 (2 warm runs) | 29.7 / 16.0 | 0.75x |

Prefill here is 5-8 tokens, so it is dominated by per-call overhead and not comparable to the 1k-token league rows.
The first gpt-oss Stingray run (cold page cache on a 12 GB file) measured 0.3 / 2.6 t/s; the warm runs above are
the figure.

## Why the 2026-10-08 sweep saw odd output

The sweep text says it ran "raw (no chat template)". It did not. With no `--chat-template` and no embedded template,
`RunCommand.FormatPrompt` falls back to hardcoded ChatML:

```
Prompt (escaped): <|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n
Prompt tokens (32): 27, 91, 320, 62, 9688, 91, 29, 7220, 198, ...   (gpt2: "<", "|", "im", "_", "start", ...)
```

GPT-2 and StarCoder2 have no `<|im_start|>` token, so the wrapper is tokenized as text and the model continues it.
That gives the `<|imp…` / `<` output the sweep saw.

**Fixed in the same change:** the CLI now warns when it falls back to ChatML on a vocabulary without `<|im_start|>`,
and points at `STINGRAY_RAW_PROMPT=1` (`RunCommand.UsesUnsupportedChatMlFallback`, `ChatMlFallbackWarningTests`).
Whether base models should default to raw completion instead is an owner decision. It changes default output, so it
is not changed here.
