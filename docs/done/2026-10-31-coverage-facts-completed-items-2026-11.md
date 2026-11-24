# 2026-10-31-coverage-facts - completed items 2026-11

Ticked items moved from [2026-10-31-coverage-facts.md](../2-coverage/research/2026-10-31-coverage-facts.md) on 2026-11-08. Each keeps its text and nested lines. Items with open follow-up text stay in the source doc.

- [x] DiffusionGemma config values and model-card numbers recorded (section 1), 2026-10-31
- [x] Qwen3.8-Flash-Next UD-IQ1_S size (72.5 GB) confirmed from two sources
- [x] Qwen1.5-MoE-A2.7B-Chat licence tag confirmed (`tongyi-qianwen-research`); read the LICENSE file before any commercial decision
- [x] Trinity-Mini (Apache 2.0) and Phi-3.5-MoE (MIT) licences confirmed from the repo cards

---

## Moved 2026-11-24 from 2026-10-31-coverage-facts.md

Open items stay in [2026-10-31-coverage-facts.md](../2-coverage/research/2026-10-31-coverage-facts.md).

_From: External facts behind the 2-coverage research (2026-10-31) / Open checks from this research (status 2026-11-08)_

- [x] GLM-5.3 UD-Q2_K_XL size: 236.44 GiB from the HF tree listing (section 2).

_From: External facts behind the 2-coverage research (2026-10-31) / Open checks from this research (status 2026-11-08)_

- [x] DeepSeek-V4-Flash Q2_K size: no plain `Q2_K` folder; `UD-Q2_K_XL` is 96.8 GB (section 2).

_From: External facts behind the 2-coverage research (2026-10-31) / Open checks from this research (status 2026-11-08)_

- [x] Hunyuan-A13B-Instruct licence: EU, UK and South Korea are excluded (section 3).

_From: External facts behind the 2-coverage research (2026-10-31) / Open checks from this research (status 2026-11-08)_

- [x] MiniMax-H3 licence text: retrieved. The Applicable Territory excludes the EU, UK, Korea and the USA (section 3).

_From: External facts behind the 2-coverage research (2026-10-31) / Open checks from this research (status 2026-11-08)_

- [x] Muse-Glimmer-30B licence: `apache-2.0` for `meta-models/Muse-Glimmer-30B` (section 3). The GGUF repo was not checked.

_From: External facts behind the 2-coverage research (2026-10-31) / Open checks from this research (status 2026-11-08)_

- [x] GLM-4.5-Air licence: HF tag `mit`. GLM-5.3 licence: custom, not MIT, with a US$10B Model-as-a-Service condition (section 3).

_From: External facts behind the 2-coverage research (2026-10-31) / Open checks from this research (status 2026-11-08)_

- [x] Llama 4 attention defaults: match the unsloth mirror of the Scout config (section 4). Meta's gated repo was not read.

_From: External facts behind the 2-coverage research (2026-10-31) / Open checks from this research (status 2026-11-08)_

- [x] File lists recorded: `unsloth/GLM-5.3-GGUF` has LICENSE (custom licence, section 3). `unsloth/DeepSeek-V4-Flash-GGUF` has no LICENSE file, and its HF tag is `mit`. `unsloth/Qwen3.8-Flash-Next-GGUF` has no LICENSE file. Its licence is the Qwen Community License 1.0, from the base repo `Qwen/Qwen3.8-Flash-Next` LICENSE (read 2026-11-08).

---

## Moved 2026-11-24 from 2026-10-31-coverage-facts.md

Open items stay in [2026-10-31-coverage-facts.md](../2-coverage/research/2026-10-31-coverage-facts.md).

_From: External facts behind the 2-coverage research (2026-10-31) / Open checks from this research (status 2026-11-08)_

- [x] llama-server logprob field names: read from the pinned `bed0a8566` source (`examples/llama.cpp/llama.cpp`, HEAD checked; no build). Request: `logprobs` (bool) and `top_logprobs` (int, default 20) set `n_probs` (`tools/server/server-common.cpp:1431-1436`); `top_logprobs` without `logprobs: true` is refused there. Response: `choices[].logprobs.content[]`, each entry with `id`, `token`, `bytes`, `logprob` and `top_logprobs` (`prob` and `top_probs` when post-sampling probabilities are requested) (`tools/server/server-task.cpp:296`, `434`). Streaming puts `logprobs` on the final chunk (`server-task.cpp:1150`). Met 2026-11-24 from source; the local `llama-server.exe` is still a stub, so no run was done.
