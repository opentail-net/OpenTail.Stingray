# External facts behind the 2-coverage research (2026-10-31)

Companion to the research sections appended to the plans in [docs/2-coverage](..). Each row is a claim the plans make, the value found on the web, and where it was checked. Nothing here changes code.

## 1. DiffusionGemma (`google/diffusiongemma-26B-A4B-it`)

Config values, fetched 2026-10-31 from [config.json](https://huggingface.co/google/diffusiongemma-26B-A4B-it/blob/main/config.json):

| Key (text_config) | Value | Plan says | Match |
|---|---|---|---|
| `final_logit_softcapping` | 30.0 | 30 (Phase 8) | yes |
| `sliding_window` | 1024 | 1024 | yes |
| `hidden_size` | 2816 | 2816 | yes |
| `intermediate_size` | 2112 | 2112 for the dense FFN; the plan also uses 2112 for the self-conditioning MLP | dense yes; self-conditioning width not in config |
| `moe_intermediate_size` | 704 | 704 per expert (fused 1408) | yes |
| `num_experts` / `top_k_experts` | 128 / 8 | 128 / 8 | yes |
| `head_dim` / `global_head_dim` | 256 / 512 | 256 SWA, 512 full | yes |
| `layer_output_scale`, `enc_layer_output_scale` | no such keys | required tensors in Phase 1 | not in config |

Model card, fetched 2026-10-31 from [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it): 30 layers, sliding window 1024, canvas length 256, 8 of 128 experts active plus 1 shared, licence `apache-2.0`. It documents "Entropy-Bounded Denoising and Adaptive Stopping": entropy bound 0.1 for token selection, entropy threshold 0.005 for early stopping, and a temperature schedule of "Linear decay from 0.8 → 0.4".

GGUF, fetched 2026-10-31 from [unsloth/diffusiongemma-26B-A4B-it-GGUF](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF): Q4_K_M 16.8 GB (file listing; the guide says 16 GB), Q8_0 26.9 GB, BF16 50.5 GB, licence `apache-2.0`.

## 2. Large checkpoints named in the plans

| Checkpoint | Plan says | Web finding | Source |
|---|---|---|---|
| GLM-5.3, UD-Q2_K_XL | ~236.4 GiB in 7 shards | Not found. Unsloth lists **239 GB** for UD-IQ2_M of the full model. A 109 GB figure exists for GLM-5.3-Flash UD-Q2_K_XL, a different model. | [unsloth GLM-5.3 guide](https://unsloth.ai/docs/zh/mo-xing/glm-5.3.md), [repo discussion](https://huggingface.co/unsloth/GLM-5.3-GGUF/discussions/4) |
| Qwen3.8-Flash-Next, UD-IQ1_S | ~72.5 GB | Confirmed: 72.5 GB in Unsloth's quant table, and in two third-party blogs. UD-IQ1_M is 74.5 GB. | [unsloth Qwen3.8 docs](https://unsloth.ai/docs/de/modelle/qwen3.8-next.md), [OrcaRouter](https://www.orcarouter.ai/blog/qwen-3-8-flash-next) |
| DeepSeek-V4-Flash, Q2_K | ~98.6 GB | Conflicting: a July blog lists 102 GB for a 2-bit quant; Unsloth's V4-Flash guide gives 103 GB for 3-bit. No official V4-Flash GGUF was confirmed. | [blog](https://blog.asadfaizee.is-a.dev/posts/deepseek-v4-flash-0731-gguf-run-284b-ai-model-locally/), [unsloth V4 guide](https://unsloth.ai/docs/de/modelle/deepseek-v4) |
| DeepSeek-V4.1-Flash, UD-Q2_K_XL | ~335.4 GB | Unsloth lists **334.8 GB**; consistent. | [unsloth V4.1 guide](https://unsloth.ai/docs/models/deepseek-v4.1-flash) |
| MiniMax-H3 GGUF set | denoiser Q4_K ~10.6 GiB; text encoder 12.2 to 17 GiB | Unsloth repo exists; a ComfyUI article gives Q2_K 6.3 GB to Q8_0 20 GB. | [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF), [ComfyUI wiki](https://comfyui-wiki.com/fr/news/2026-08-10-unsloth-minimax-h3-gguf) |

## 3. Licences

| Model | Plan or review says | Web finding | Source |
|---|---|---|---|
| Qwen1.5-MoE-A2.7B-Chat | Tongyi Qianwen, not Apache or MIT | Confirmed: tag `tongyi-qianwen-research`, licence file `LICENSE`. A third-party summary says free commercial use; that conflicts with the research tag, so read the LICENSE file. | [HF commit LICENSE](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat/commits/222e0bec2bbfbd70d9ba5d1d7a777a81aad10859/LICENSE) |
| Hunyuan-A13B-Instruct | not stated in the plans | Tencent Hunyuan Community License. Aggregators say commercial use is allowed with conditions, and that some Hunyuan licences exclude the EU, UK and South Korea. Not confirmed for A13B. | [tencent/Hunyuan-A13B-Instruct](https://huggingface.co/tencent/Hunyuan-A13B-Instruct/commit/7ae536b754e17728409ee1fab30ff9dba8a36fa6) |
| Trinity-Mini (afmoe) | not stated | Apache 2.0 on the base, GGUF and FP8 cards. | [Trinity-Mini-GGUF](https://huggingface.co/arcee-ai/Trinity-Mini-GGUF) |
| Phi-3.5-MoE-instruct | not stated | MIT on the Hub licence tag and model card. | [HF commit LICENSE](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct/commits/ae6cb90aceffd86d1e3fba55c59ec62dfc88d4a1/LICENSE) |
| GLM-4.5-Air | "MIT" in the review | Secondary sources say MIT; the GLM-4.5 family including Air stays MIT (one source says GLM-5.3 differs). | [Simon Willison, 2025-07-28](https://simonwillison.net/2025/Jul/28/) |
| Muse-Glimmer-30B | not stated | Secondary sources say Apache 2.0; the Hub card was not fetched. | [aiweekly](https://aiweekly.co/node/9642) |
| MiniMax-H3 | "territorial restrictions" | Confirmed: the MiniMax H3 Community License defines an Applicable Territory that excludes some jurisdictions. Licence text not retrieved. | [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF) |

## 4. Llama 4 attention details

From the Transformers and vLLM configs (secondary: library source, not Meta's card):

- `attention_chunk_size` default **8192**.
- NoPE layers every `no_rope_layer_interval` = **4**; RoPE layers use chunked local attention, NoPE layers use full attention.
- Temperature tuning on the NoPE layers only: `floor_scale` default **8192**, `attn_scale` default **0.1**, enabled by default.
- `use_qk_norm` L2-normalises queries and keys on the RoPE layers.

Sources: [vLLM llama4 module](https://docs.vllm.ai/en/v0.11.1/api/vllm/model_executor/models/llama4/), [Transformers llama4 config](https://generationalwealth.duckdns.org/venv/lib/python3.13/site-packages/transformers/models/llama4/configuration_llama4.py) (a mirror; check upstream).

## 5. llama-server logprob fields (golden capture)

From the llama.cpp server history ([PR #10783, "server : fix logprobs, make it OAI-compatible"](https://cdn04132025.gitlink.org.cn/replica/llama.cpp/commit/57bb2c40cd94c5a09f5210ed8264cc93b21c4b7e), a mirror):

- By default the per-token fields are `logprob` and `top_logprobs`.
- With `post_sampling_probs` true, the same fields are renamed to `prob` and `top_probs`, and they are probabilities, not log-probabilities.
- The golden capture must keep `post_sampling_probs` false. The golden plan's margins depend on that.
