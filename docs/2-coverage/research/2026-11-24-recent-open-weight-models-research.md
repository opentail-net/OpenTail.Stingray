# Recent open-weight models (2026): support gap, popularity, host fit (2026-11-24)

**Question.** Which models released in 2026 are not supported by OpenTail.Stingray yet? Keep only the popular ones (owner: "if the recent models are not popular, they are of no interest"). Which of those could run on this host?

**Companion files.** Plan: [2026-11-24-recent-models-support-plan.md](../2026-11-24-recent-models-support-plan.md). Earlier coverage facts: [2026-10-31-coverage-facts.md](2026-10-31-coverage-facts.md).

## 1. Summary

- **Popular, not supported, fits this host, llama.cpp implements it** (candidates for porting): Inkling-2-Mini (`inkling2_loop`), Spark-X2.5/2.6-4B (`spark2_5`), LFM3-Omni-2B (`lfm3`), K2-Horizon (`k2-horizon`, dense sizes), Nemotron-3.5-Lightning-30B-A3B (`nemotron_h_moe`), North-Mini-Code-1.0 (`cohere2moe`), Mellum2.1-12B (`mellum`), Ling-3.0-tiny (`bailingmoe3`), Laguna-XS-2.1 (`laguna`), Qwen3-Coder-Next (`qwen3next`, 79.7B; fits only at Q4/Q5 quants).
- **Already supported, no new code:** Nanbeige4.3-33B (`llama` architecture, admitted); Qwen 3.5/3.6/3.8 (`qwen35`, `qwen35moe`), Gemma 4 (`gemma4`), LFM2.5 (`lfm2moe`). Ternary-Bonsai-2 runs as `qwen35`, but its Q2_0 quant depends on an experimental gate that is already on the backlog.
- **Popular but too big for this host:** Kimi-K3 (2.78 T params; smallest file 434 GiB), Mistral-Large-4 (1.05 T; smallest 217 GiB), Hy4 family (770 B; smallest 214 GiB), MiniMax-M3.1-Flash (426 B; smallest 120 GiB), Step-3.5-Flash (197 B; smallest 111 GiB), Ling-3.0-flash (124 B; smallest 67 GiB).
- **Popular enough, but on the edge:** Mistral-Small-4-119B (`mistral4`): family sum 88k below the 100k threshold; Q3_K_M is 50.6 GiB.
- **Not text-LLM architectures** (audio, diffusion, drafters): see section 5.
- **Discrepancies found** (not fixed here): see section 8. The main one: the code admits `qwen4exp` (Qwen 3.8 Flash Next) on 2026-11-09, but two planning docs still say it is not admitted.

## 2. Method, sources, assumptions

**Data (all fetched 2026-11-24, local time about 22:10).**
- Hugging Face API `https://huggingface.co/api/models` with `filter=gguf`, `expand[]=downloads, downloadsAllTime, createdAt, gguf`: top 500 by `sort=downloads`, top 500 by `sort=createdAt`, top 300 by `sort=trendingScore`, plus 14 targeted `search=` queries for families named in web results. 1,158 unique GGUF repos. `downloads` is the last-30-day count; `downloadsAllTime` is all-time; `gguf.architecture` is `general.architecture` from the repo's GGUF header.
- llama.cpp `master` (raw `src/llama-arch.cpp`: 181 `LLM_ARCH_*` names; GitHub contents of `src/models/`: 188 files). This is master at fetch time, not the pinned revision (see section 8).
- Engine registry: `src/OpenTail.Stingray.Engine/Architectures/*.cs` (descriptor `Id`, `Status`, `Aliases`) and `src/OpenTail.Stingray.Engine/ModelCompatibility.cs`.
- Host-fit: `stingray scout` 1.0.9 (Release build at `src/OpenTail.Stingray.Cli/bin/Release/net10.0/stingray.exe`) with `-r <repo> --quants --budget 63G -c 4096 --max-quants 16`. This reads the GGUF index only; no weights are downloaded. 23 remote repos were scouted (21 in a batch, plus Inkling-2-Mini and K2-Horizon 3.7B). The per-repo JSON is in the session scratchpad, not in the repo.
- Web search (2026-11-24): used only to find candidates. Its results are secondary (blogs, DEV posts, a Weekly Project News digest). It mentions llama.cpp merges for Poolside Laguna (2026-07-22), Nanbeige 4.2 and Mimo v2.5 (PR #22493). No figure in this note comes from the search results.

**Assumptions (mine; the owner should confirm or change them, see the plan's owner decisions):**
- **A1 Popularity.** A family is "popular" if its GGUF repos in the sample sum to at least 100,000 downloads in the last 30 days. The sum is over sampled repos only, so it is a lower bound.
- **A2 Host fit.** A weights file of at most **55 GiB** is "fits resident" (63 GiB total RAM minus scout's default 8 GiB reserve). 55 to 110 GiB is "paged candidate" (not planned here). Above 110 GiB is excluded. KV cache and runtime overhead are not included: scout gives no memory estimate for unsupported architectures, so "fits" means a file-size test, not a measured run.
- **A3 Main files only.** Files prefixed `mtp-`, `dflash-`, `dspark-` (MTP heads or speculative drafters), `mmproj` and `imatrix` are excluded from the size ranges below.

## 3. Host

- CPU: AMD Ryzen 7 5700G, 8 cores / 16 threads (Windows reports this).
- RAM: 63.3 GiB visible; 36.3 GiB free at check time. Free RAM will vary with other processes.
- GPU: integrated AMD Radeon (driver 31.0.21925.1001, 512 MiB reported `AdapterRAM`). Shares system RAM. No discrete GPU (CLAUDE.md rule 13).
- Free disk at check (GB): C 39.2, E 121.8, F 3.2, G 27.1, H 67.4, I 86.9, J 107.4, K 3.9, L 11.5, M 17.8. Project policy puts checkpoints in `E:\_models` (or `F:\_models`), one candidate at a time (memory: disk constraint). C has the least room among the big drives.

## 4. Engine support today (baseline)

**Admitted text-generation IDs** (descriptors with `Status = Admitted`, plus aliases): afmoe, apertus, codeshell, cohere2, deepseek2, deepseek2-ocr, ernie4_5, exaone, exaone4, falcon, gemma, gemma2, gemma3, gemma3n, gemma4, glm4, glm4moe, gpt-oss, gpt2, gptneox, granite, granitehybrid, granitemoe, hunyuan-dense, hunyuan-moe, internlm2, jais, jais2, lfm2, lfm2moe, llama, llama4, maincoder, mimo, mimo2, minicpm, ministral, mistral3, muse-glimmer (alias muse_glimmer), nemotron_h, olmo, olmo2, olmoe, orion, paddleocr, phi2, phi3, phimoe, qwen, qwen2, qwen2moe, qwen2vl, qwen3, qwen35, qwen35moe, qwen3moe, qwen3vl, qwen4exp, rwkv6, rwkv7, smollm3, stablelm, starcoder, starcoder2, xverse.

**NotAdmitted descriptors:** deepseek32, deepseek4, deepseek41, diffusion-gemma, glm-dsa, glm5next (alias glm5-next).

**Not in the registry, but present elsewhere in the engine:** `ModelGraph.cs:713` has a RoPE-pairing table that lists `qwen3next`, `step35`, `bailingmoe2`, `minimax-m2`, `exaone-moe` and others. This is groundwork for GGUF architectures without a descriptor. It does not make them supported.

## 5. Popularity: 2026 GGUF families in the sample

Family = `general.architecture` string. "Sum 30d" is across all sampled repos of that architecture, regardless of creation date. The primary repo is the most-downloaded repo of that family in the sample; "created" is that repo's GGUF creation date, not the base model's release date (not checked).

| Architecture (family) | Sum 30d | Repos | Primary repo, 30d / all-time (created) | llama.cpp master | Engine | Verdict |
|---|---:|---:|---|---|---|---|
| `qwen35` (Qwen 3.5/3.6/3.8 dense, Ornith, Bonsai-2) | 82,611,766 | 248 | unsloth/Qwen3.8-27B-GGUF 8,019,028 / 29,192,084 | yes | Admitted | no action |
| `qwen35moe` | 28,777,457 | 54 | huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated-MTP-GGUF 4,457,126 | yes | Admitted | no action |
| `qwen4exp` (Qwen 3.8 Flash Next) | 27,367,261 | 38 | ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF 7,662,070 / 15,259,723 | yes | Admitted 2026-11-09 (see §8) | no action here |
| `gemma4` | 25,673,786 | 84 | unsloth/gemma-4-26B-A4B-it-GGUF 4,172,902 | yes | Admitted | no action |
| `llama` (includes Nanbeige4.3-33B) | 5,549,213 | 87 | bartowski/Nanbeige_Nanbeige4.3-33B-GGUF 158,031 / 172,785 (2026-10-25) | yes | Admitted | Nanbeige runs now (§6) |
| `inkling2_loop` (Inkling-2-Mini) | 3,912,873 | 47 | unsloth/Inkling-2-Mini-GGUF 1,913,407 (created 2026-11-20) | yes | **not in registry** | **port candidate, Tier 2** |
| `spark2_5` (Spark-X2.5 / X2.6 4B) | 3,648,070 | 27 | XHToken/Spark-X2.6-4B-GGUF 1,004,919 / 1,074,059 (2026-10-19) | yes | **not in registry** | **port candidate, Tier 1** |
| `dflash` | 4,686,500 | 14 | drafter files of Ternary-Bonsai-2 and others | yes | not in registry | not a standalone model (excluded) |
| `lfm3` (LFM3-Omni-2B) | 1,953,548 | 19 | LiquidAI/LFM3-Omni-2B-GGUF 566,642 (created 2026-10-27) | yes | **not in registry** | **port candidate, Tier 1** (text path) |
| `k2-horizon` (IFM K2-Horizon 0.9B / 3.7B / 7B / 36B-A4B MoVA) | 1,429,658 | 30 | IFM/K2-Horizon-3.7B 32,238 / 60,902; 7B 24,749 / 75,771; 0.9B 20,511 / 33,696; MoVA 14,597 / 68,292 (all created 2026-09-03) | yes | **not in registry** | **port candidate, Tier 1 (dense) / Tier 2 (MoVA)** |
| `qwen3next` (Qwen3-Coder-Next, 79.7B) | 849,213 | 3 | unsloth/Qwen3-Coder-Next-GGUF 217,217 / 5,429,651 (created 2026-02-03) | yes | not in registry | **port candidate, Tier 1, tight fit** |
| `bailingmoe3` (Ling-3.0) | 847,975 | 37 | bloomer010/Ling-3.0-tiny 243,858 / 780,093 (2026-08-11); AtomicChat/Ling-3.0-flash 242,425 / 986,287 (2026-08-04) | yes | not in registry | **port candidate, Tier 2 (tiny only fits)** |
| `nemotron_h_moe` | 652,095 | 6 | ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF 207,662 / 834,465 (2026-08-11) | yes | not in registry (`nemotron_h` admitted) | **port candidate, Tier 1** |
| `laguna` (Poolside Laguna XS-2.1 / S-2.1) | 141,593 | 25 | unsloth/Laguna-S-2.1-GGUF 21,576 / 275,124; poolside/Laguna-XS-2.1-GGUF 12,886 / 665,924 | yes (merged 2026-07-22 per web search, not checked) | not in registry (`afmoe` admitted) | **port candidate, Tier 1 (XS first)** |
| `cohere2moe` (North-Mini-Code-1.0) | 124,324 | 1 | unsloth/North-Mini-Code-1.0-GGUF 124,324 / 896,578 (2026-06-10) | yes | not in registry (`cohere2` admitted) | **port candidate, Tier 1** |
| `mellum` (JetBrains Mellum2.1-12B-A2.5B-Thinking) | 242,427 | 2 | JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF 242,427 / 501,411 (2026-10-07) | yes | not in registry | **port candidate, Tier 2 (base not read)** |
| `minimax-m3` | 115,124 | 21 | unsloth/MiniMax-M3.1-Flash-GGUF 35,077 / 39,426 (2026-10-23) | yes | not in registry | popular, **too big** (§6) |
| `hyv4` + `hy_v4_flash` + `hy_v4_next_dfly` (Hy4 family) | 958,380 | 17 | AngelSlim/Hy4-preview-GGUF 303,422 / 865,728 (2026-08-28) | **arch names not in master** (`hy_v4` is; §8) | not in registry | popular, **too big** (§6) |
| `kimi-k3` | 214,453 | 12 | unsloth/Kimi-K3-GGUF 183,142 / 1,541,459 (2026-07-27) | yes | not in registry | popular, **too big** (§6) |
| `mistral4` (Mistral-Small-4-119B) | 87,901 | 10 | unsloth/Mistral-Small-4-119B-2603-GGUF 20,442 / 179,117 (2026-03-17) | yes (`mistral4.cpp` is a 6-line stub; not read) | not in registry (`mistral3` admitted) | **below threshold** (§6: edge fit) |
| `mistral-large4` | 52,674 | 9 | unsloth/Mistral-Large-4-1T-A52B-GGUF 24,609 (2026-10-29) | yes | not in registry | below threshold and too big |
| `step35` (Step-3.5-Flash) | 39,349 | 22 | ggml-org/Step-3.5-Flash-GGUF 25,832 / 72,116 (2026-02-03) | yes | not in registry | below threshold and too big |
| `kimi-k31` | 28,136 | 4 | KarLexAI/Kimi-K3.1-Pro-GGUF 23,179 | yes | not in registry | below threshold |

**Not classified here (not text-LLM architectures).** These are audio, speech, music, image or video pipelines, or drafters, and belong to the other pipelines (CLAUDE.md lists the audio pipelines that already cover Parakeet, Qwen3-ASR, Qwen3-TTS, Whisper and Silero VAD). Popular examples by 30-day family sum: `audiocpp` (13,953,063), `parakeet` (7,872,735), `mert2` (5,490,084), `qwen_image` (4,888,833), `qwen3_asr` (3,634,000), `acestep-lm` (2,329,181). Also `wan`, `ltxv`, `flux`, `qwen3-tts`, `whisper`, `higgs_tts`, `breeze-tts-2`, `mm3-cond` (MiniMax-Music3). Whether any of them needs a pipeline is an owner decision, outside this plan.

## 6. Host fit (scout, index only; main files, A2 budget)

| Candidate (params) | Main-file sizes (GiB) | Fits 55 GiB | Notes |
|---|---|---|---|
| Inkling-2-Mini (26.1 B) | IQ1_S 7.2 … Q8_K_XL 21.3 | 10 of 10 | `dspark-` drafter excluded |
| Spark-X2.6-4B (4.4 B) | Q4_K_M 3.3, Q8_0 4.1, full 7.7 | 3 of 3 | Spark-X2.5 (4.1 B) not scouted |
| LFM3-Omni-2B (1.7 B) | Q4_0 1.0 … BF16 3.2 | 5 of 5 | text path only |
| K2-Horizon 0.9B (1.1 B) | Q4_K_M 0.6 … BF16 2.0 | 6 of 6 | dense |
| K2-Horizon 3.7B (5.1 B; files named "4B") | Q4_K_M 2.9 … BF16 9.4 | 6 of 6 | dense |
| K2-Horizon 7B (9.0 B) | Q4_K_M 5.2 … BF16 16.8 | 6 of 6 | dense |
| K2-Horizon MoVA 36B-A4B (37.4 B) | Q4_K_M 20.8 … Q8_0 37.1; BF16 69.8 | 5 of 6 | value experts (§7) |
| Nemotron-3.5-Lightning 30B-A3B (31.6 B) | Q4_0 17.6, NVFP4 18.4, Q8_0 31.3; BF16 58.8 | 3 of 4 | |
| Ling-3.0-tiny (7.9 B) | Q3_K_M 3.6 … F16 14.7 | 16 of 16 | |
| Ling-3.0-flash (124.4 B) | NVFP4 67.4 (smallest) | 0 | too big |
| North-Mini-Code-1.0 (30.5 B) | Q3_K_M 13.2 … BF16 56.8 | 15 of 16 | |
| Mellum2.1-12B-A2.5B (12.1 B) | MXFP4 6.6 … BF16 22.6 | all | |
| Laguna-XS-2.1 (33.4 B) | Q4_K_M 18.9; BF16 62.3 | 1 of 2 | |
| Laguna-S-2.1 (117.6 B) | Q3_K_M 50.3, Q3_K_XL 50.4, IQ4_XS 53.6, IQ4_NL 54.7; Q4_K_S 63.9 and up | 4 | edge |
| Qwen3-Coder-Next (79.7 B) | MXFP4 44.7, Q4_K_M 45.2, UD-Q4_K_M 45.9, UD-Q4_K_XL 46.2, Q4_1 46.6, Q5_K_S 51.3, UD-Q5_K_S 52.0, Q5_K_M 52.9 | 8 | UD-Q5_K_M (55.2) and above do not fit |
| Nanbeige4.3-33B (33.1 B, `llama`) | Q3_K_L 16.1 … Q8_0 32.8; BF16 61.7 | 15 of 16 | scout verdict "Fits" |
| Mistral-Small-4-119B (119 B) | Q3_K_M 50.6, Q3_K_XL 51.1, IQ4_XS 54.1; IQ4_NL 55.1 and up | 3 | edge; below popularity |
| Step-3.5-Flash (197 B) | Q4_K 110.6 (only listed file) | 0 | too big |
| MiniMax-M3.1-Flash (426 B) | IQ1_M 120.5 … Q8_K_XL 241.3 | 0 | too big |
| Hy4-preview (770 B) | IQ1_M 219.2, STQ1_0 213.7, Q4_K_M 435.2 | 0 | too big |
| Kimi-K3 (2,780 B) | Q1_0 434.3 … IQ1_S 553.2 | 0 | too big |
| Mistral-Large-4 (1,049 B) | IQ1_S 217.0 … Q2_K_XL 352.3 | 0 | too big |
| Ternary-Bonsai-2-9B (`qwen35`) | Q2_0 2.4 (verdict "Fits"); PTQ1_0 1.9 and PQ2_0 2.3 ("Unsupported") | 1 | Q2_0 quant gate already on backlog |

**Not checked:** the sizes above are file sizes. Whether a model runs at a useful speed on this CPU, and how much RAM the KV cache takes at long contexts, is unmeasured.

## 7. llama.cpp implementations (master, 2026-11-24): size and reuse signals

Line counts are the size of llama.cpp's own implementation, used only as a rough signal of port effort. Measured porting effort is not known.

| Architecture | llama.cpp file (bytes) | Reuse signal |
|---|---|---|
| `spark2_5` | `spark2-5.cpp` (5.6 KB, 137 lines) | standard attention with a gate (`wqkv_gate`), dense FFN; uses generic `build_qkv`, `build_rope`, `build_attn` |
| `k2-horizon` | `k2-horizon.cpp` (16.5 KB, 356 lines) | standard attention with QK-norm and dense FFN; MoVA adds `attn_v_gate` / `attn_v_exps` (value experts), a router, and an optional shared expert |
| `lfm3` | `lfm3.cpp` (9.3 KB) | LFM2-family hybrid (`lfm2` admitted); Omni inputs are separate |
| `nemotron_h_moe` | `nemotron-h-moe.cpp` (5.7 KB, 154 lines) | builds on the `nemotron_h` hybrid (admitted) |
| `cohere2moe` | `cohere2moe.cpp` (17.7 KB, 413 lines) | `cohere2` admitted; adds MoE |
| `laguna` | `laguna.cpp` (16.2 KB, 328 lines) | comment says it shares MoE/gate structure with `afmoe` (admitted) |
| `qwen3next` | `qwen3next.cpp` (35.7 KB) | hybrid Gated-DeltaNet family (same family as admitted `qwen35`/`qwen35moe`); RoPE entry already in `ModelGraph.cs` |
| `mellum` | `mellum.cpp` (13.5 KB, 373 lines) | base architecture not read |
| `bailingmoe3` | `bailingmoe3.cpp` (28.2 KB, 511 lines) | MoE; `bailingmoe2` (also not admitted) is the nearest relative |
| `inkling2_loop` | `inkling2-loop.cpp` (5.8 KB, 123 lines) on `inkling2.cpp` (22.3 KB, 458 lines) | the loop wrapper reuses the base graph and adds control tensors. The base has many mechanisms not in the admitted families (below) |

**Inkling-2 mechanisms** (from `inkling2.cpp`, master 2026-11-24): relative attention (`attn_rel`, `attn_rel_bias`, `attn_rel_clip`, `n_relative_extent`), Q/K/V clipping (`attn_q_clip`, `f_qk_clip`, `f_relative_clip`), short convolutions on K and V (`attn_k_conv`, `attn_v_conv`, `attn_post_conv`, `ffn_post_conv`), attention-residual queries (`attn_res_query`, block size `attn_res_block_size`), output and FFN residual queries (`output_res_query`, `ffn_res_query`), attention sink projection (`attn_sink_proj`), SWA and full layers in one model, a logit-softcap key, f32 router with expert bias, and an input-dependent K/V convolution tap (`attn_k_conv_proj`, `attn_v_conv_proj`). The loop wrapper adds `LOOP_MSG_QUERY`, `ctrl_down`, `ctrl_gain`, `ctrl_forget` per Q/K/V/FFN, and shares physical layer weights across passes. This is why Inkling-2 is Tier 2, not Tier 1.

## 8. Discrepancies, gaps and things not checked

- **`qwen4exp` status conflict.** `src/OpenTail.Stingray.Engine/Architectures/Qwen4ExpArchitecture.cs` sets `Status = Admitted` and notes "Admitted 2026-11-09 for text generation and tool calls on the UD-IQ1_S checkpoint". But [ported-families-todo.md](../ported-families-todo.md) (section 1, "Qwen 3.8 Flash Next") and [docs/00-current-work.md](../../00-current-work.md) §2 still describe it as not admitted and ported only. One of them is stale. Not resolved here.
- **Missing independent reference binary.** `tools/llama.cpp/llama-server.exe` and `llama-cli.exe` are 9,216-byte stubs (checked 2026-11-24). The admission path needs a real llama-server at a recorded revision (see the plan's P1). The local source checkout `examples/llama.cpp` is pinned to `bed0a8566` per ported-families-todo.md. Not built here.
- **Pinned vs master.** The llama.cpp architecture lists above come from master on 2026-11-24. They are not checked against `bed0a8566` or the vendored build b10306. Some architectures may be newer than either pin.
- **Hy4 arch names.** The GGUFs say `hyv4`, `hy_v4_flash` and `hy_v4_next_dfly`. Those strings are not in llama.cpp master's `LLM_ARCH` list. Master has `hy_v3` and `hy_v4`, and a file `hy-v4.cpp`. The mismatch is unexplained and was not investigated (the family is too big for this host anyway).
- **Glm5-next alias.** `Glm5NextArchitecture.cs` has `Id = "glm5next"` (NotAdmitted) with alias `glm5-next`. The alias inherits NotAdmitted.
- **Popularity sample.** 1,158 repos from four HF listings is not an exhaustive census. Family sums are lower bounds. Fine-tunes (abliterated, uncensored, MTP and quant re-uploads) inflate some sums. Primary-repo numbers are given next to family sums for that reason.
- **Base-model release dates not checked.** "created" in §5 is the GGUF repo's creation date, not the base model's release date. The owner's "2026" window is applied to the GGUF repo dates, and qwen3next's base may be older (its GGUF repo is 2026-02-03).
- **`mistral4.cpp` is a 6-line stub** that defers to a graph elsewhere. Not read; no reuse claim is made.
- **Kimi-K3 and Mistral-Large-4 scouts** returned only very large quants (see §6). Their parameter counts already rule them out.

## 9. Sources

- Hugging Face model API (fetched 2026-11-24): `https://huggingface.co/api/models?filter=gguf&sort=downloads|createdAt|trendingScore&direction=-1&limit=300|500&expand[]=downloads&expand[]=downloadsAllTime&expand[]=createdAt&expand[]=gguf`, and `…?search=<family>&filter=gguf&sort=downloads…` for laguna, nanbeige, mistral-4, mistral-large-4, minimax-m3, step-3.5, ling-3, k2-horizon, kimi-k3, inkling, hy4, exaone-moe, ernie-5, spark-x.
- Repos cited in §5–§6 (primary repos): `unsloth/Inkling-2-Mini-GGUF`, `XHToken/Spark-X2.6-4B-GGUF`, `LiquidAI/LFM3-Omni-2B-GGUF`, `IFM/K2-Horizon-{0.9B,3.7B,7B,MoVA-36B-A4B}-GGUF`, `ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF`, `bloomer010/Ling-3.0-tiny-GGUF`, `unsloth/North-Mini-Code-1.0-GGUF`, `JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF`, `poolside/Laguna-XS-2.1-GGUF`, `unsloth/Laguna-S-2.1-GGUF`, `unsloth/Qwen3-Coder-Next-GGUF`, `bartowski/Nanbeige_Nanbeige4.3-33B-GGUF`, `unsloth/Mistral-Small-4-119B-2603-GGUF`, `ggml-org/Step-3.5-Flash-GGUF`, `unsloth/MiniMax-M3.1-Flash-GGUF`, `AngelSlim/Hy4-preview-GGUF`, `unsloth/Kimi-K3-GGUF`, `unsloth/Mistral-Large-4-1T-A52B-GGUF`, `prism-ml/Ternary-Bonsai-2-9B-gguf`.
- llama.cpp master (fetched 2026-11-24): `https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-arch.cpp`; `https://api.github.com/repos/ggml-org/llama.cpp/contents/src/models`; `https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/models/{inkling2-loop,inkling2,spark2-5,k2-horizon,mistral4,nemotron-h-moe,cohere2moe,mellum,lfm3,laguna,bailingmoe3,minimax-m3,step35}.cpp`.
- Engine code: `src/OpenTail.Stingray.Engine/Architectures/` (descriptors), `src/OpenTail.Stingray.Engine/ModelCompatibility.cs`, `src/OpenTail.Stingray.Core/ModelGraph.cs` (RoPE table, line 713).
- Web search (2026-11-24, secondary, candidate discovery only): Poolside Laguna merge (2026-07-22), Nanbeige 4.2 PR, Mimo v2.5 PR #22493, K2 Horizon feature request (`ggml-org/llama.cpp` search results).
