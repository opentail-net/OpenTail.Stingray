# Which MoE families get the CPU-prefill handoff (2026-10-04)

Companion to [the batched MoE prefill plan](2026-10-03-batched-moe-prefill-plan.md). The handoff (CPU batched prefill, then copy K/V into the hybrid or GPU cache) turned OLMoE's 631-token prefill from about 24 into 130-140 tok/s. This page is about how far that spreads and how a family earns it.

## How others decide a MoE family is usable (read 2026-10-04)

| Project | Mechanism | What we take |
|---|---|---|
| llama.cpp | One generic `build_moe_ffn` used by 58 model builders; families differ by parameters (gating function, weight normalisation, scale, shared expert). Mixtral is `arch = llama` with `n_expert == 8`. | Do not key on the architecture name alone: a `llama` header with experts is its own family. |
| FreeToken | `docs/models.md` lists named known-good checkpoints; "other checkpoints of the same architecture work too". The MoE *strategy* (fused / offload / cpu / hybrid) is chosen per machine from a measured bandwidth profile, never per model. A family-wide test matrix checks correctness. | Strategy is a measurement (done: `stingray calibrate`); admission is per family and evidence-backed. |
| TensorSharp | Publishes per-checkpoint benchmark tables against llama.cpp. | Receipts name a checkpoint and numbers. |

Our bar (hybrid-vs-CPU logit parity on a real checkpoint, near-tie argmax rule) is stricter than any of these. It stays.

## The gate (code: `Engine/PrefillHandoffFamilies.cs`)

Two layers instead of a name list.

1. **Structural** (from the GGUF header). The handoff moves per-head K and V rows, so these are `Incompatible` and no setting lifts that: MLA (`*.attention.kv_lora_rank > 0`), recurrent or convolution state (`*.ssm.*`, `*.shortconv.*`, `*.full_attention_interval`), a forward pass of its own (`gpt-oss`), LongRoPE (`rope_factors_short.weight`; the hybrid cannot decode it).
2. **Evidence**. `Admitted` only with a receipt (a named parity test) in `PrefillHandoffFamilies.Receipts`. A structurally fine family without one is `Unverified`; `STINGRAY_HYBRID_CPU_PREFILL=all` runs it anyway, for experiments.

Refusals now say which of the two applies. Tests: `PrefillHandoffFamiliesTests` (synthetic headers, plus real headers of the checkpoints that exist on this machine: OLMoE and Qwen3-Coder classify Admitted; DeepSeek-V2-Lite, LFM2-8B-A1B, gpt-oss classify Incompatible; the Granite-H-small header was not found by the test on this run and skipped).

## Coverage matrix

"Header-checked" means the classifier was run on a real header on this machine. Other rows are from the code and from knowledge of the model architectures, not from a checkpoint.

| Family (GGUF arch) | Class | Status today | Basis | Next step |
|---|---|---|---|---|
| OLMoE (`olmoe`) | ordinary KV | **Admitted** | parity receipt, header-checked | none |
| Qwen3-MoE (`qwen3moe`) | ordinary KV | **Admitted** | parity receipt on Qwen3-Coder-30B, header-checked | none |
| Qwen3 dense (`qwen3`) | ordinary KV | **Admitted** | parity receipt (0.6B) | none |
| Qwen2-MoE (`qwen2moe`) | ordinary KV, shared expert | **Admitted on Vulkan hybrid and full-GPU** (2026-10-04); CUDA unverified | receipts on both Vulkan paths; found and fixed 3 bugs (expert width, shared-expert gate, hybrid scratch) | CUDA only |
| Mixtral (`llama` + experts) | ordinary KV | **Admitted on Vulkan hybrid** (2026-10-04); full-GPU and CUDA unverified | receipt on the Vulkan hybrid (Nous-Hermes-2-Mixtral); top-k renormalisation fixed | full-GPU, CUDA |
| Phi-3.5-MoE (`phimoe`) | ordinary KV + LongRoPE | **Admitted on Vulkan hybrid and full-GPU (short factors tested)** (2026-10-04); CUDA hybrid Incompatible (no LongRoPE) | Vulkan hybrid at both RoPE regimes; full-GPU only at the short-factor regime (ctx 1024) | full-GPU long factors (ctx > 4096), CUDA LongRoPE |
| Llama 4 (`llama4`) | ordinary KV, MoE top-1 + shared expert, NoPE every 4th layer, L2 QK-norm, chunked attention | **Admitted on Vulkan hybrid**, including past the 8192 chunk boundary (2026-10-04); CUDA unverified | Scout Q3_K_M: CPU parity vs llama-server, hybrid handoff at 4 and 1 GPU layers (40 steps); chunked attention + temperature tuning implemented; CPU 9430-token run matches llama-server (24 positions, 16 confident, 0 near-tie differences); hybrid handoff at 9429 tokens into 4 GPU layers, 6 decode steps cosine 0.9994-0.99996 vs all-CPU (`Handoff_Llama4_PastTheChunkBoundary_DecodeAgreesWithCpu`, 3130 s) | CUDA; sequential-GPU-prefill comparison at this length (hours) not run |
| GLM-4.5 / Air (`glm4moe`) | ordinary KV, sigmoid gating + selection bias, leading dense layers | CPU path admitted, Vulkan layer split works (-g 4); **admitted on `VulkanLayerSplit`** (2026-10-04) | `LayerSplitCpuPrefillHandoffTests`: GLM-4.5-Air Q2_K, -g 4, 205 tokens, GPU K/V equal to the CPU pass, prefill logits equal, 40 decode steps cosine >= 0.9946, 0 flips | -g above 4 (needs the expert slot cache), CUDA |
| Hunyuan-A13B (`hunyuan-moe`) | ordinary KV, 64 experts top-8 + shared expert, QK-norm after RoPE | CPU path admitted (2026-10-04, Q3_K_S); **no handoff receipt** | `HunyuanMoeGreedyParityTests`: 39 confident positions match llama-server, PPL 219.8 vs 201.6 +/- 53.7 | GPU paths (layer split candidate), receipt |
| Arcee Trinity Mini (`afmoe`) | ordinary KV, 128 experts top-8 + shared, sigmoid + bias routing, leading dense layers, attention output gate, sliding/global layers | CPU path admitted (2026-10-04, Q4_K_M, < 2048 tokens); **no handoff receipt** | `AfmoeGreedyParityTests`: 31 confident positions match llama-server, PPL 6.84 vs 7.11 +/- 1.21 | output gate in the Vulkan passes, receipt, > 2048-token window |
| Hunyuan-MoE | ordinary KV | not admitted at all | recognised in `ModelGraph` only | out of scope until admitted |
| DeepSeek-V2/V3/R1 (`deepseek2`) | MLA | **Incompatible** | header-checked on DeepSeek-V2-Lite | separate latent-cache handoff, not this one |
| LFM2-MoE (`lfm2moe`) | short-conv state | **Incompatible** | header-checked | separate class |
| Granite-4-H (`granitehybrid`) | Mamba-2 state | **Incompatible** | header-checked | separate class |
| Qwen3.5/3.6 (`qwen35*`) | gated delta-net hybrid | **Incompatible** | header-checked on Qwen3.6-27B | separate class |
| gpt-oss | own forward pass | **Incompatible** | by rule | separate class |

Count that matters: **7 admitted families (6 MoE: OLMoE, Qwen3-MoE, Qwen2-MoE, Mixtral, Phi-MoE, Llama 4; plus dense Qwen3) of 13 rows, all on Vulkan only (10 path-specific receipts)**; GLM-4.5 has a receipt on the Vulkan layer split path only (`-g 4`); 5 are a different class entirely, so the handoff should never claim them.

Per-family closure plan, in order: [2026-10-04-moe-handoff-closure-plan.md](2026-10-04-moe-handoff-closure-plan.md).

## What it takes to admit one more family

1. Obtain one checkpoint (disk budget: one at a time; for `qwen2moe`, the smallest published checkpoint is the sensible first proof).
2. Run the existing hybrid-vs-CPU parity test on it (`VulkanHybridOlmoeParityTests` pattern: cosine > 0.99, near-tie rule, byte-exact K/V).
3. Add the receipt line, name the test, and update this table. Never add a line without a passing run.

Nothing here has been downloaded: the two ordinary-KV MoE checkpoints on disk are already admitted, so no further family can be proven without a download.
