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
| Qwen2-MoE (`qwen2moe`) | ordinary KV, shared expert | **Admitted on Vulkan hybrid and full-GPU** (2026-10-04); CUDA unverified | in `ModelCompatibility`; shared-expert width fix already in the queue; no checkpoint on disk | small checkpoint, run the existing parity test, add a receipt |
| Mixtral (`llama` + experts) | ordinary KV | **Admitted on Vulkan hybrid** (2026-10-04); full-GPU and CUDA unverified | handled by the generic MoE path in code (`ForwardPass.Moe.cs`); no checkpoint, no receipt; treated as its own family key | one checkpoint, parity test |
| Phi-3.5-MoE (`phimoe`) | ordinary KV + LongRoPE | **Admitted on Vulkan hybrid and full-GPU (short factors tested)** (2026-10-04); CUDA hybrid Incompatible (no LongRoPE) | CPU receipt exists, not hybrid; `HybridForwardPass.cs:187` refuses | LongRoPE in the Vulkan and CUDA hybrid first |
| Llama 4 (`llama4`) | ordinary KV, interleaved MoE, shared expert, special RoPE/QK | Unverified | in the allowlist; large checkpoints | not a first proof; later |
| GLM-4.5 / Air (`glm4moe`) | ordinary KV, sigmoid gating, leading dense layers | Unverified | gating handled in `ModelGraph`; real-weight admission still open | after the small families |
| Hunyuan-MoE | ordinary KV | not admitted at all | recognised in `ModelGraph` only | out of scope until admitted |
| DeepSeek-V2/V3/R1 (`deepseek2`) | MLA | **Incompatible** | header-checked on DeepSeek-V2-Lite | separate latent-cache handoff, not this one |
| LFM2-MoE (`lfm2moe`) | short-conv state | **Incompatible** | header-checked | separate class |
| Granite-4-H (`granitehybrid`) | Mamba-2 state | **Incompatible** | header-checked | separate class |
| Qwen3.5/3.6 (`qwen35*`) | gated delta-net hybrid | **Incompatible** | header-checked on Qwen3.6-27B | separate class |
| gpt-oss | own forward pass | **Incompatible** | by rule | separate class |

Count that matters: **3 admitted families (two MoE) of 13 rows**; 5 more are structurally eligible and waiting for a receipt; 5 are a different class entirely, so the handoff should never claim them.

Per-family closure plan, in order: [2026-10-04-moe-handoff-closure-plan.md](2026-10-04-moe-handoff-closure-plan.md).

## What it takes to admit one more family

1. Obtain one checkpoint (disk budget: one at a time; for `qwen2moe`, the smallest published checkpoint is the sensible first proof).
2. Run the existing hybrid-vs-CPU parity test on it (`VulkanHybridOlmoeParityTests` pattern: cosine > 0.99, near-tie rule, byte-exact K/V).
3. Add the receipt line, name the test, and update this table. Never add a line without a passing run.

Nothing here has been downloaded: the two ordinary-KV MoE checkpoints on disk are already admitted, so no further family can be proven without a download.
