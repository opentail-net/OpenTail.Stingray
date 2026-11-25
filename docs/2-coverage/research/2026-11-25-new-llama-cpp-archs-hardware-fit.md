# New llama.cpp architectures: which models fit this host (2026-11-25)

**Question.** Which models from the architectures llama.cpp added since `bed0a8566` could run on this machine, and could be inferred through Stingray?

**Host.** Ryzen 7 5700G, 63 GB RAM (about 35–44 GB free in practice), integrated Radeon, no discrete GPU. The `models/` budget is about 57 GB, so download one checkpoint at a time.

**Method and limits.**
- Architectures were matched to their converter classes in `conversion/` and `gguf-py/`, then to model cards. The architecture name list is in `src/llama-arch.cpp` at `36d818e16`.
- Sizes come from third-party cards and articles, not from `config.json` files. Check the official `config.json` before any download.
- "Est. Q4" assumes about 0.6 GB per billion parameters (Q4_K_M, about 4.8 bits per weight). It is an estimate.
- Running a model needs a new architecture in Stingray. A GGUF file alone is not enough.
- Companion file: [2026-11-25-llama-cpp-refresh-and-tensorsharp-survey.md](2026-11-25-llama-cpp-refresh-and-tensorsharp-survey.md).

## 1. Fits this host

| Model | llama.cpp arch | Total / active | Est. Q4 | License | Notes |
|---|---|---|---|---|---|
| K2-Horizon-0.9B | `k2-horizon` | 0.9B dense (≈1.08B stored) | <1 GB | Apache 2.0 | GGUF on HF ([card](https://huggingface.co/IFM/K2-Horizon-0.9B-GGUF)). Smallest real test of the architecture. |
| K2-Horizon-MoVA-36B-A4B | `k2-horizon` (MoVA attention) | 36B / 4B. vLLM recipe says 37.44B stored and 5.95B active | ~22 GB | Apache 2.0 | [card](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B), [vLLM recipe](https://recipes.vllm.ai/IFM/K2-Horizon-36B). Needs MoVA attention. |
| Flint | `flint` | 39.8B / ~2.5B (about 2.2B without the output head) | ~24 GB | Apache 2.0 | [HF](https://huggingface.co/RecursiveSI/flint), [TechCrunch](https://techcrunch.com/2026/10/21/recursive-launches-flint-a-laptop-coding-model-it-says-was-largely-built-by-AI/). 262K context. No Transformers modeling code yet. |
| ZAYA1-8B | `zaya1` | 8.4B / ~0.76B | ~5 GB | Apache 2.0 | [HF](https://huggingface.co/Zyphra/ZAYA1-8B), [report](https://arxiv.org/pdf/2511.17127). Unusual attention: two KV heads, one for the current token and one for the previous. |
| Xing4.0-29B-A4B | `Xing4_0` | 29B / 4B | ~17 GB | not checked | [HF](https://huggingface.co/21g/Xing4.0-29B-A4B), [README](https://raw.githubusercontent.com/XingChen-AGI/Xing4.0-29B-A4B/main/README.md). Parameter counts disagree (29B official, 31.2B on aggregators). |
| SenseNova-U1-8B | `sensenova_u1` (vision and generation) | 8B dense | ~5 GB | Apache 2.0 per one outlet | [HF](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT). No visual encoder and no VAE, so the design differs from usual VLMs. |
| Zonos3-Mini (speech TTS) | `zonos3` | ~800M / ~100M | ~0.5 GB | Apache 2.0 | [HF](https://huggingface.co/Zyphra/ZONOS3-Mini). 44.1 kHz output, 43 languages. |
| tiny-voca (speech TTS) | `tiny-voca` | 4.5M | ~9 MB | Apache 2.0 | English only. Mirror: [ai-tldr](https://ai-tldr.dev/releases/n4ze3m-tiny-voca/index.md). Original: `huggingface.co/n4ze3m/tiny-voca`. |
| Maincoder-1B (code) | `maion-coder` | 1B dense | ~0.7 GB | not checked | [HF](https://huggingface.co/Maincode/Maincoder-1B). Llama-style config per mirrors. |
| Limite-1B-violetto | `limite` | ~1B (from the name only) | <1 GB | Apache 2.0 per MLX port | [MLX port](https://huggingface.co/pierjoe/limite-mlx). Size unconfirmed. |

## 2. Borderline

- **Kolibri-1** (`kolibri1`): 78.1B total, 3.46B active, 50 layers, 384 experts (one source). The Q4 weights are about 45 GB, so it only fits at a lower quant or with expert paging. Apache 2.0 weights. Sources: [article](https://pasqualepillitteri.it/en/news/20468/aleph-alpha-releases-kolibri-1-78b-open-weight-1m-context), [architecture gallery](https://sebastianraschka.com/llm-architecture-gallery/models/kolibri-1-78b/).
- **Moshi** (`moshi`): about 7.6B. Fits, but the licence is unclear. The arXiv page says CC BY-NC-SA 4.0, while release posts say CC-BY 4.0. There is no official GGUF, and it is full-duplex, which is a large job. Sources: [arXiv](https://arxiv.org/html/2410.00037v1), [release](https://kyutai.org/2024/09/18/moshi-release.html).
- **Metis-1.6-Logos** (`metis16`): size not found. The earlier Metis-1.5-think was 898M total and about 340M active. Probably fits. Source: [Metis-1.5-think](https://huggingface.co/Lernex/Metis-1.5-think), [Metis-1.6 report](https://arxiv.org/pdf/2610.37903).

## 3. Does not fit

- **Mistral-Large-4** (`mistral-large4`): 1.05T total, about 49–52B active. Source: [docs](https://docs.mistral.ai/models/mistral-large-4-0.md), [active-count discrepancy](https://mixed-news.com/en/?p=35786).
- **Kimi K3.1 Pro** (`kimi-k31`): about 2.8T total, 104B active. Source: [model card](https://www.mindstudio.ai/blog/kimi-k3-1-pro-model-card), [architecture gallery](https://sebastianraschka.com/llm-architecture-gallery/models/kimi-k3-1-pro/).
- **Muse-Spark-Open-816B-A42B** (`muse-spark`): 816B total, 42B active. The NVFP4 checkpoint is about a third of the 1.63 TB BF16 size, so roughly 545 GB. Source: [architecture gallery](https://sebastianraschka.com/llm-architecture-gallery/models/muse-spark-open-816b-a42b/).
- **Inkling-2-Small** (`inkling2`): 594B total, 17B active. Source: [model card](https://thinkingmachines.ai/model-card/inkling-2-small/). Licence status is unclear: one source says Apache 2.0 and another says "closed at launch".

## 4. Not checked

- **Beam**, **Clover MoE** and **vllm-decision2**: no sizes found in this pass.
- **Domyn-Edge** (`domynedge`): size not found. Its benchmarks compare against models of up to 4B, which suggests a small model. The companion model, Domyn-Small-v1.0, is a dense 10B model under MIT, with a 32K native window. Source: [paper](https://huggingface.co/papers/2607.20448.md).
- **gemma-embedding2**: an embedder. Not relevant to the generation path.
- **Clef** (`clef`): depends on an unmerged llama.cpp PR. Source: [ggml-org/Clef-GGUF](https://huggingface.co/ggml-org/Clef-GGUF).
- **Kokoro-82M** (`kokoro`): already in Stingray (listed in `CLAUDE.md`), so it is not new for us. Licence: Apache 2.0 weights. Source: [HF](https://huggingface.co/hexgrad/Kokoro-82M).

## 5. Suggested order, if a port is wanted

1. **K2-Horizon-0.9B.** Smallest real test of a new architecture against a llama.cpp golden.
2. **Flint** or **K2-Horizon-36B-A4B.** The best size and speed fit for a larger model. Both need a new architecture.
3. **ZAYA1-8B.** Fits easily. The two-KV-head attention needs careful golden checks.
4. **Zonos3-Mini** and **SenseNova-U1-8B** for speech and vision, after the text models.

Each candidate needs a golden against llama.cpp before admission (CLAUDE.md rule 14 and rule 8). Check every size against `config.json` before a download.
