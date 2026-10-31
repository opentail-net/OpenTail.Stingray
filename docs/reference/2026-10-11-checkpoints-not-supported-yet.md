# Checkpoints we do not support yet (2026-10-11)

Source: `scripts/bench-league-recheck.ps1 -Set all` over every text-generation GGUF found on every drive (inventory: `scripts/list-checkpoints.ps1`). Each checkpoint was run once on the CPU path (`-g 0`) and once on the Vulkan full-offload path (`-g -1`) with a ~500-token prompt. A row below means the engine **could not produce a result**, which is different from "slow": slow checkpoints are in `PerformanceLeague.md`.

Counts at generation: 103 models attempted, 15 with at least one unsupported backend, 0 timeouts, 3 oversized checkpoints not re-run because of size (section 6).

## 1. Fails on both CPU and Vulkan

| Checkpoint | GB | Architecture | Why |
|---|---:|---|---|
| GLM-5.3-UD-Q2_K_XL-00001-of-00007 | 0.01 | glm-dsa | architecture is ported but deliberately not admitted (see the exception text) |

## 2. Runs on CPU, fails on Vulkan full offload (`-g -1`)

These checkpoints work today with the default CPU path; only `-g -1` fails, so the user-visible limit is "CPU only". CPU figure is prefill / decode tok/s at ~500 tokens.

| Checkpoint | GB | Architecture | CPU prefill / decode | Why Vulkan fails |
|---|---:|---|---|---|
| granite-4.0-h-350m-Q8_0 | 0.34 | granitehybrid | 285.8 / 61.3 | fails with "Missing tensor: blk.0.attn_q.weight": the Vulkan full-offload pass assumes standard attention in every layer (inferred: this architecture has a hybrid, recurrent or MLA layer 0; not yet traced) |
| LFM2-1.2B-Q8_0 | 1.16 | lfm2 | 108.1 / 24.3 | fails with "Missing tensor: blk.0.attn_q.weight": the Vulkan full-offload pass assumes standard attention in every layer (inferred: this architecture has a hybrid, recurrent or MLA layer 0; not yet traced) |
| OLMo-1B.Q8_0 | 1.17 | olmo | 27.5 / 24.7 | fails with "Missing tensor: blk.0.attn_norm.weight" (inferred: this architecture has no attn_norm weight, e.g. non-parametric or post-norm layout; not yet traced) |
| granite-4.0-h-1b-Q8_0 | 1.45 | granitehybrid | 67.3 / 17.4 | fails with "Missing tensor: blk.0.attn_q.weight": the Vulkan full-offload pass assumes standard attention in every layer (inferred: this architecture has a hybrid, recurrent or MLA layer 0; not yet traced) |
| OLMo-2-0425-1B-Q8_0 | 1.47 | olmo2 | 101.9 / 17.3 | fails with "Missing tensor: blk.0.attn_norm.weight" (inferred: this architecture has no attn_norm weight, e.g. non-parametric or post-norm layout; not yet traced) |
| nemotron-nano-12b-v2-vl-Q2_K | 4.39 | nemotron_h | 11.7 / 6.7 | fails with "Missing tensor: blk.0.attn_q.weight": the Vulkan full-offload pass assumes standard attention in every layer (inferred: this architecture has a hybrid, recurrent or MLA layer 0; not yet traced) |
| LFM2-8B-A1B-Q4_K_M | 4.7 | lfm2moe | 118.3 / 24.0 | fails with "Missing tensor: blk.0.attn_q.weight": the Vulkan full-offload pass assumes standard attention in every layer (inferred: this architecture has a hybrid, recurrent or MLA layer 0; not yet traced) |
| youtu-vl-4b-Q8_0 | 4.85 | deepseek2 | 25.7 / 5.3 | fails with "Missing tensor: blk.0.attn_q.weight": the Vulkan full-offload pass assumes standard attention in every layer (inferred: this architecture has a hybrid, recurrent or MLA layer 0; not yet traced) |
| Ternary-Bonsai-2-27B-PTQ1_0 | 5.54 | qwen35 | 7.7 / 0.5 | PRISM (PQ2_0 / PTQ1_0 + Hadamard) weights are implemented only on the CPU hybrid-GDN pass |
| kimi-vl-a3b-thinking-Q2_K | 6.13 | deepseek2 | 49.9 / 17.1 | fails with "Missing tensor: blk.0.attn_k.weight" (inferred: MLA attention stores K differently; not yet traced) |
| GLM-4.7-Flash-Q2_K | 10.57 | deepseek2 | 31.7 / 9.3 | fails with "Missing tensor: blk.0.attn_q.weight": the Vulkan full-offload pass assumes standard attention in every layer (inferred: this architecture has a hybrid, recurrent or MLA layer 0; not yet traced) |
| granite-4.0-h-small-Q2_K | 10.97 | granitehybrid | 13.9 / 8.2 | fails with "Missing tensor: blk.0.attn_q.weight": the Vulkan full-offload pass assumes standard attention in every layer (inferred: this architecture has a hybrid, recurrent or MLA layer 0; not yet traced) |
| DeepSeek-V2-Lite-Chat.Q8_0 | 15.56 | deepseek2 | 45.3 / 11.7 | System.NotSupportedException: No row-offset matvec for Q8_0. |
| EXAONE-4.5-33B-Q4_K_M | 18.67 | exaone4 | 11.6 / 1.9 | fails with "Missing tensor: blk.0.attn_norm.weight" (inferred: this architecture has no attn_norm weight, e.g. non-parametric or post-norm layout; not yet traced) |

## 3. Runs on Vulkan, fails on CPU

None in this run.

## 4. Timeouts

None in this run.

## 5. Not tested by this harness (not "unsupported")

Embeddings / rerank, ASR, TTS, vocoders, diffusion, vision projectors (`mmproj`) and non-GGUF checkpoints are not text-generation models, so the text CLI cannot time them. They have their own harnesses (`scripts/bench-audio.ps1`, `scripts/bench-image.ps1`, the encoder benchmark tests) and their own rows in [STATUS.md](../STATUS.md). Their inventory is at the end of the results file this document was generated from; it is not repeated here because "not covered by this run" says nothing about whether they work.

## 6. Oversized checkpoints: NOT re-run because of their size (not "unsupported")

These checkpoints are 30 GB or larger. Their first attempt used the harness defaults and failed for size reasons; **they have deliberately not been re-run with the workarounds because of their size** (very long runs and memory pressure on this 63 GB machine). The engine supports oversized checkpoints, so these rows say nothing about architecture support. The two workarounds are the CLI flag `--ignore-preflight` for CPU runs and a partial offload (`-g N`) instead of full offload (`-g -1`) for Vulkan; `scripts/bench-league-recheck.ps1` now applies both automatically, so a future `-Set all -Resume` will measure them once these rows are deleted from the results file.

| Checkpoint | GB | Architecture | Backend | First attempt | Manual result |
|---|---:|---|---|---|---|
| cerebras_GLM-4.5-Air-REAP-82B-A12B-Q2_K | 33.28 | glm4moe | vulkan | full Vulkan offload (`-g -1`) ran out of host memory while uploading layers | not re-run |
| GLM-4.5-Air-Q2_K | 42.21 | glm4moe | vulkan | full Vulkan offload (`-g -1`) ran out of host memory while uploading layers | not re-run |
| Llama-4-Scout-17B-16E-Instruct-Q3_K_M-00001-of-00002 | 46.28 | llama4 | cpu | the CLI memory preflight refused the run (estimated about 57 GiB against 63 GiB with 8 GiB kept free) | ran once by hand on 2026-10-11 with `--ignore-preflight`: **prefill 4.2 tok/s, decode 4.4 tok/s** at 468 tokens |
| Llama-4-Scout-17B-16E-Instruct-Q3_K_M-00001-of-00002 | 46.28 | llama4 | vulkan | full Vulkan offload (`-g -1`) ran out of host memory while uploading layers | not re-run |

## What to do with this list

- Section 2 is the cheapest to shrink: each cause above is one tensor-naming or layer-layout assumption in the Vulkan full-offload pass (`GpuForwardPass`) or a missing hybrid GPU path, not a missing architecture. `stingray scout` (rule 15) is the first step for any of them.
- Section 1 entries are known architecture ports that are deliberately not admitted; promotion follows the normal admission path (a real checkpoint, an independent reference, timed runs, then the STATUS row).
- Re-run `scripts/bench-league-recheck.ps1 -Set all -Resume` and `scripts/league-unsupported-doc.ps1` after any such fix; a row that now produces numbers moves into `PerformanceLeague.md` automatically on the next `scripts/league-update-section.ps1`.

