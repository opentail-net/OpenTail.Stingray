# External facts behind the 2-coverage research (2026-10-31)

Companion to the research sections appended to the plans in [docs/2-coverage](..). Each row is a claim the plans make, the value found on the web, and where it was checked. Nothing here changes code.

## 1. DiffusionGemma (`google/diffusiongemma-26B-A4B-it`)

Config values, fetched 2026-10-31 from [config.json] (pinned to repo revision f7f5b7f5, unchanged since 2026-07-15)(https://huggingface.co/google/diffusiongemma-26B-A4B-it/blob/f7f5b7f5fa82ffc52addd066915886d497f5517b/config.json):

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
| GLM-5.3, UD-Q2_K_XL | ~236.4 GiB in 7 shards | **Confirmed 2026-11-08:** the `UD-Q2_K_XL` folder of `unsloth/GLM-5.3-GGUF` is 7 files, 253,878,406,789 bytes = 236.44 GiB (HF tree listing). The earlier "not found" came from guides; the listing is the source. Unsloth's 239 GB is for UD-IQ2_M, a different quant. | [HF tree listing](https://huggingface.co/api/models/unsloth/GLM-5.3-GGUF/tree/main?recursive=true), [unsloth GLM-5.3 guide](https://unsloth.ai/docs/zh/mo-xing/glm-5.3.md) |
| Qwen3.8-Flash-Next, UD-IQ1_S | ~72.5 GB | Confirmed: 72.5 GB in Unsloth's quant table, and in two third-party blogs. UD-IQ1_M is 74.5 GB. | [unsloth Qwen3.8 docs](https://unsloth.ai/docs/de/modelle/qwen3.8-next.md), [OrcaRouter](https://www.orcarouter.ai/blog/qwen-3-8-flash-next) |
| DeepSeek-V4-Flash, Q2_K | ~98.6 GB | **Resolved 2026-11-08:** `unsloth/DeepSeek-V4-Flash-GGUF` has no plain `Q2_K` folder. Its closest file set is `UD-Q2_K_XL`: 3 files, 96,832,507,552 bytes (90.18 GiB, 96.8 GB, HF tree listing). The 98.6, 102 and 103 GB figures in the blog and guide do not match that folder, so use the listing. | [HF tree listing](https://huggingface.co/api/models/unsloth/DeepSeek-V4-Flash-GGUF/tree/main?recursive=true) |
| DeepSeek-V4.1-Flash, UD-Q2_K_XL | ~335.4 GB | Unsloth lists **334.8 GB**; consistent. | [unsloth V4.1 guide](https://unsloth.ai/docs/models/deepseek-v4.1-flash) |
| MiniMax-H3 GGUF set | denoiser Q4_K ~10.6 GiB; text encoder 12.2 to 17 GiB | Unsloth repo exists; a ComfyUI article gives Q2_K 6.3 GB to Q8_0 20 GB. | [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF), [ComfyUI wiki](https://comfyui-wiki.com/fr/news/2026-08-10-unsloth-minimax-h3-gguf) |

## 3. Licences

| Model | Plan or review says | Web finding | Source |
|---|---|---|---|
| Qwen1.5-MoE-A2.7B-Chat | Tongyi Qianwen, not Apache or MIT | Confirmed: tag `tongyi-qianwen-research`, licence file `LICENSE`. A third-party summary says free commercial use; that conflicts with the research tag, so read the LICENSE file. | [HF commit LICENSE](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat/commits/222e0bec2bbfbd70d9ba5d1d7a777a81aad10859/LICENSE) |
| Hunyuan-A13B-Instruct | not stated in the plans | **Confirmed 2026-11-08 from the LICENSE file:** the licence "does not apply in the European Union, United Kingdom and South Korea". Territory is worldwide minus those three. Use outside the Territory is unlicensed. | [tencent/Hunyuan-A13B-Instruct LICENSE](https://huggingface.co/tencent/Hunyuan-A13B-Instruct/resolve/main/LICENSE) |
| Trinity-Mini (afmoe) | not stated | Apache 2.0 on the base, GGUF and FP8 cards. | [Trinity-Mini-GGUF](https://huggingface.co/arcee-ai/Trinity-Mini-GGUF) |
| Phi-3.5-MoE-instruct | not stated | MIT on the Hub licence tag and model card. | [HF commit LICENSE](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct/commits/ae6cb90aceffd86d1e3fba55c59ec62dfc88d4a1/LICENSE) |
| GLM-4.5-Air | "MIT" in the review | **Confirmed 2026-11-08:** `zai-org/GLM-4.5-Air` README front matter says `license: mit`. The repo has no LICENSE file. | [zai-org/GLM-4.5-Air](https://huggingface.co/zai-org/GLM-4.5-Air) |
| GLM-5.3 | "differs per one source" | **Confirmed 2026-11-08 from `zai-org/GLM-5.3` LICENSE:** a custom "GLM-5.3 License" with an MIT-style grant, not plain MIT. A Model-as-a-Service business whose affiliates' aggregate revenue exceeds US$10B over any 12 consecutive months must pass Z.AI's security review before commercial use. `unsloth/GLM-5.3-GGUF` carries the same LICENSE. | [zai-org/GLM-5.3 LICENSE](https://huggingface.co/zai-org/GLM-5.3/resolve/main/LICENSE) |
| Muse-Glimmer-30B | not stated | **Confirmed 2026-11-08 for `meta-models/Muse-Glimmer-30B`:** licence `apache-2.0`, with a LICENSE file. The GGUF repo `meta-models/Muse-Glimmer-30B-GGUF` was not checked, and the port plan does not name a repo. | [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) |
| MiniMax-H3 | "territorial restrictions" | **Confirmed 2026-11-08 from the LICENSE text:** Applicable Territory is worldwide excluding the European Union, the United Kingdom, the Republic of Korea **and the United States**. Use outside it is not authorised. Commercial use above US$20M revenue needs prior written authorisation from MiniMax. The UI must display "MiniMax H3". | [unsloth/MiniMax-H3-GGUF LICENSE](https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/LICENSE) |

## 4. Llama 4 attention details

From the Transformers and vLLM configs (secondary: library source, not Meta's card):

- `attention_chunk_size` default **8192**.
- NoPE layers every `no_rope_layer_interval` = **4**; RoPE layers use chunked local attention, NoPE layers use full attention.
- Temperature tuning on the NoPE layers only: `floor_scale` default **8192**, `attn_scale` default **0.1**, enabled by default.
- `use_qk_norm` L2-normalises queries and keys on the RoPE layers.

Sources: [vLLM llama4 module](https://docs.vllm.ai/en/v0.11.1/api/vllm/model_executor/models/llama4/), [Transformers llama4 config](https://generationalwealth.duckdns.org/venv/lib/python3.13/site-packages/transformers/models/llama4/configuration_llama4.py) (a mirror; check upstream).

**Checked 2026-11-08** against `unsloth/Llama-4-Scout-17B-16E-Instruct` `config.json` (an ungated mirror of Meta's config, not Meta's card): `attention_chunk_size` 8192, `floor_scale` 8192, `attn_scale` 0.1, `attn_temperature_tuning` true, `use_qk_norm` true, `rope_theta` 500000, `no_rope_layers` with 0 (NoPE) at every fourth layer (indices 3, 7, 11, ...). These match the bullets above. Meta's own `meta-llama/Llama-4-Scout-17B-16E` is gated and could not be read anonymously, so Meta's card is still unread.

## 5. llama-server logprob fields (golden capture)

From the llama.cpp server history ([PR #10783, "server : fix logprobs, make it OAI-compatible"](https://cdn04132025.gitlink.org.cn/replica/llama.cpp/commit/57bb2c40cd94c5a09f5210ed8264cc93b21c4b7e), a mirror):

- By default the per-token fields are `logprob` and `top_logprobs`.
- With `post_sampling_probs` true, the same fields are renamed to `prob` and `top_probs`, and they are probabilities, not log-probabilities.
- The golden capture must keep `post_sampling_probs` false. The golden plan's margins depend on that.

**Checked 2026-11-08, not closed:** the local `tools/llama.cpp` is build `b10306-cpu`, not the pinned `bed0a8566`, and its `llama-server.exe` is a 9 KB stub. The field names cannot be read from it. This stays open until the pinned source or a real build is checked.

## Open checks from this research (status 2026-11-08)

- [x] GLM-5.3 UD-Q2_K_XL size: 236.44 GiB from the HF tree listing (section 2).
- [x] DeepSeek-V4-Flash Q2_K size: no plain `Q2_K` folder; `UD-Q2_K_XL` is 96.8 GB (section 2).
- [x] Hunyuan-A13B-Instruct licence: EU, UK and South Korea are excluded (section 3).
- [x] MiniMax-H3 licence text: retrieved. The Applicable Territory excludes the EU, UK, Korea and the USA (section 3).
- [x] Muse-Glimmer-30B licence: `apache-2.0` for `meta-models/Muse-Glimmer-30B` (section 3). The GGUF repo was not checked.
- [x] GLM-4.5-Air licence: HF tag `mit`. GLM-5.3 licence: custom, not MIT, with a US$10B Model-as-a-Service condition (section 3).
- [x] Llama 4 attention defaults: match the unsloth mirror of the Scout config (section 4). Meta's gated repo was not read.
- [ ] llama-server logprob field names: NOT CHECKED. Needs the pinned `bed0a8566` source or a real build. The local `llama-server.exe` is a stub.
- [x] File lists recorded: `unsloth/GLM-5.3-GGUF` has LICENSE (custom licence, section 3). `unsloth/DeepSeek-V4-Flash-GGUF` has no LICENSE file, and its HF tag is `mit`. `unsloth/Qwen3.8-Flash-Next-GGUF` has no LICENSE file. Its licence is the Qwen Community License 1.0, from the base repo `Qwen/Qwen3.8-Flash-Next` LICENSE (read 2026-11-08).
