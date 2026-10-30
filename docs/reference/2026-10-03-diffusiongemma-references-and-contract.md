# DiffusionGemma: implementation references, verified GGUF contract and code map

Moved from [../2-coverage/2026-10-03-diffusiongemma-port-plan.md](../2-coverage/2026-10-03-diffusiongemma-port-plan.md) on 2026-10-30. These sections are facts about the reference implementations and the checkpoint. The plan that uses them keeps the phases and status.

# Implementation References

Use the following source priority.

### 1. Real checkpoint

The verified `unsloth/diffusiongemma-26B-A4B-it-GGUF` checkpoint is authoritative for:

* tensor names
* tensor shapes
* layer inventory
* quantization
* vocabulary size
* layer schedule
* presence/absence of `attn_v`
* fused expert tensor layout
* scale tensors

The checkpoint facts in this document must not be changed merely to match an implementation.

### 2. llama.cpp Gemma-4

Reference:

`llama.cpp/src/models/gemma4.cpp`

Reference commit:

`bed0a8566`

Use this as the independent oracle for:

* Gemma-4 block ordering
* Q/K normalization
* full-layer V derivation
* RoPE ordering
* dense FFN
* MoE routing
* expert execution
* post-normalization
* layer scaling

### 3. TensorSharp

Local checkout:

`C:\Git-Public\OpenTail.Stingray\examples\TensorSharp\TensorSharp`

Relevant implementation:

```text
TensorSharp.Models/Models/DiffusionGemma/
    DiffusionGemmaModel.cs
    DiffusionGemmaModel.Cpu.cs
    DiffusionGemmaModel.Multimodal.cs
    DiffusionGemmaModel.Structured.cs
    DiffusionGemmaSampler.cs
    DiffusionGemmaArchitecture.cs
```

TensorSharp provides concrete implementation evidence for:

* prompt prefill
* prompt-KV caching
* canvas-only decode
* local/global attention masks
* heterogeneous attention geometry
* fused MoE execution
* self-conditioning lifecycle
* exact soft-embedding path
* EntropyBound sampling
* argmax history
* block commitment
* multi-block generation
* CPU execution
* GPU/device execution

TensorSharp is **not** allowed to override the real checkpoint or llama.cpp when the implementations disagree on architectural semantics.

### 4. vLLM

Use current vLLM sources for:

```text
vllm/model_executor/models/diffusion_gemma.py
vllm/model_executor/models/diffusion_gemma_sampler.py
vllm/transformers_utils/configs/diffusion_gemma.py
```

Use these primarily to cross-check:

* self-conditioning
* model input regions
* scheduler semantics
* diffusion lifecycle
* temperature handling

### 5. Hugging Face / Google

Use official Diffusers / Transformers / Google sources for:

* public model configuration
* scheduler semantics
* model-level generation behaviour
* official model naming and configuration

---

# Verified Real GGUF Contract

## Global Architecture

* `diffusion-gemma.block_count` = 30
* `diffusion-gemma.embedding_length` = 2816
* `diffusion-gemma.vocab_size` = 262144
* `diffusion-gemma.feed_forward_length` = 2112
* `diffusion-gemma.expert_count` = 128
* `diffusion-gemma.expert_used_count` = 8
* `diffusion-gemma.expert_feed_forward_length` = 704
* `diffusion-gemma.canvas_length` = 256
* `diffusion-gemma.final_logit_softcapping` = 30.0
* embedding scale = `sqrt(2816)` = approximately `53.0659966`

---

# Heterogeneous Layer Schedule

## Sliding-Window Attention

Layers:

```text
0,1,2,3,4,
6,7,8,9,10,
12,13,14,15,16,
18,19,20,21,22,
24,25,26,27,28
```

Geometry:

* Q heads = 16
* KV heads = 8
* head dimension = 256
* sliding window = 1024
* RoPE theta = 10000
* RoPE dimension = 256
* dedicated `attn_v.weight` = required

## Full Attention

Layers:

```text
5, 11, 17, 23, 29
```

Geometry:

* Q heads = 16
* KV heads = 2
* head dimension = 512
* global context
* RoPE theta = 1,000,000
* RoPE dimension = 512
* `rope_freqs.weight` = required
* `attn_v.weight` = **absent**

The full-attention V path is:

```text
raw K projection
    ↓
per-head K RMSNorm
    ↓
copy K
    ↓
weightless RMSNorm
    ↓
V
```

RoPE is then applied to Q and K.

**V must never be derived from K after RoPE.**

---

# Tensor Inventory

Outside the blocks:

```text
token_embd.weight
output_norm.weight
rope_freqs.weight

self_cond_pre_norm.weight
self_cond_gate.weight
self_cond_up.weight
self_cond_down.weight
```

Per block:

```text
attn_norm.weight
attn_q.weight
attn_k.weight
attn_v.weight              // SWA only
attn_q_norm.weight
attn_k_norm.weight
attn_output.weight
post_attention_norm.weight

ffn_norm.weight
ffn_gate.weight
ffn_up.weight
ffn_down.weight
post_ffw_norm_1.weight

pre_ffw_norm_2.weight
ffn_gate_inp.weight
ffn_gate_inp.scale
ffn_gate_up_exps.weight
ffn_down_exps.weight
ffn_down_exps.scale
post_ffw_norm_2.weight

post_ffw_norm.weight
layer_output_scale
enc_layer_output_scale
```

Required dimensions:

```text
token_embd                  [262144,2816]

dense gate/up               [2816,2112]
dense down                  [2112,2816]

router                      [2816,128]
router scale                [2816]

fused expert gate/up       [2816,1408,128]
expert down                 [704,2816,128]
expert down scale           [128]
```

---


# Existing OpenTail Infrastructure To Reuse

Use existing Stingray primitives wherever they match the required semantics:

```text
SimdKernels.RmsNorm
SimdKernels.PureRmsNorm
SimdKernels.MatVecF32
SimdKernels.ApplyRoPECachedNeoxPartial
ForwardPass
ForwardPass.Moe
ForwardPass.PrefillCore
ForwardPass.Decode
GpuForwardPass
CudaForwardPass
```

Reuse existing:

* GGUF tensor resolution
* quantized matrix operations
* GQA attention
* KV cache structures
* RoPE infrastructure
* MoE routing
* expert dispatch
* CPU SIMD kernels
* Vulkan kernels
* CUDA kernels
* tied embedding / LM-head infrastructure

Do not introduce a second unrelated tensor/runtime abstraction merely to reproduce TensorSharp.

---

# TensorSharp Porting Map

The coder must inspect these TensorSharp files directly before implementing the corresponding Stingray subsystem.

| TensorSharp                         | Stingray target                                                  |
| :---------------------------------- | :--------------------------------------------------------------- |
| `DiffusionGemmaModel.cs`            | `DiffusionGemmaForwardPass` / new shared DiffusionGemma backbone |
| `DiffusionGemmaModel.Cpu.cs`        | CPU prompt-KV + canvas decode implementation                     |
| `DiffusionGemmaModel.Multimodal.cs` | multimodal / structured input handling where applicable          |
| `DiffusionGemmaModel.Structured.cs` | structured/labelled forward paths where useful                   |
| `DiffusionGemmaSampler.cs`          | `DiffusionGemmaSampler`                                          |
| `DiffusionGemmaArchitecture.cs`     | Stingray model architecture registration                         |
| TensorSharp GGML diffusion ops      | optional future fused GPU kernels                                |

The port should be **class/behaviour mapped**, not copied line-for-line.

---

