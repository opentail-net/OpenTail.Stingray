# DiffusionGemma port plan (`diffusion-gemma` / `diffusion_gemma`)

**Status (revised 2026-10-04): IN PROGRESS / IMPLEMENTATION REFERENCE NOW AVAILABLE.**
The repository now contains a complete local checkout of TensorSharp at:

`C:\Git-Public\OpenTail.Stingray\examples\TensorSharp\TensorSharp`

TensorSharp contains a substantial, executable DiffusionGemma implementation covering the Gemma-4 backbone, heterogeneous attention geometry, prompt-KV caching, canvas decoding, self-conditioning, EntropyBound sampling, block-autoregressive generation, CPU execution, and GPU/native execution paths.

This changes the implementation strategy materially.

The remaining work is **not** primarily specification discovery. The real GGUF and llama.cpp remain the architectural authorities, but TensorSharp now provides a concrete C# implementation reference for the previously missing integration and lifecycle details.

**Policy:** port now, prove later; not admitted, not advertised (CLAUDE.md rule 14; [ported-families-todo](ported-families-todo.md)). Real checkpoints remain explicitly refused by `DiffusionGemmaRealCheckpointGuardTests` until the full verification ladder passes.

---

## Strategic Architecture

DiffusionGemma is a **Gemma-4 MoE backbone plus a block-diffusion generation loop**.

The implementation should therefore be divided into four independently testable layers:

```text
Real GGUF tensor contract
        │
        ▼
Gemma-4 heterogeneous MoE backbone
        │
        ├── EncoderPrefill
        │      causal prompt → persistent KV
        │
        └── DecoderCanvas
               bidirectional canvas + prompt KV
        │
        ▼
Self-conditioning
        │
        ▼
EntropyBound denoising sampler
        │
        ▼
Block-autoregressive lifecycle
```

TensorSharp should be used as the **primary C# implementation reference** for the complete pipeline, while llama.cpp remains the independent numerical oracle for the Gemma-4 transformer block.

Do not copy TensorSharp's architecture wholesale. Port the **semantics and proven execution strategy** into Stingray's existing `ForwardPass`, `SimdKernels`, GGUF tensor, CPU, Vulkan, and CUDA infrastructure.

---

# High-Level Status & Roadmap Matrix

### Status Legend
- 🟢 **Verified Complete** — implementation complete and verified against real weights with independent parity established
- 🟡 **Impl-Complete / Synthetic-Covered** — implementation complete and covered by synthetic test suite; real-checkpoint parity pending
- 🔵 **Infrastructure Available** — generic infrastructure exists and is battle-tested in other recently-admitted MoE families; Gemma-4 specific delta implemented
- 🔴 **Pending** — awaiting real weights or verification ladder

| Phase                    | Subsystem                            | Current State | Infrastructure Available in Stingray | Gemma-4 Specific Delta Status |
| :----------------------- | :----------------------------------- | :-----------: | :----------------------------------- | :---------------------------- |
| **0. Contract**          | GGUF + Gemma-4 + Diffusion semantics |  🟢 Verified  | GGUF v3 parsing, tensor descriptors  | Locked to real 692-tensor GGUF contract |
| **1. Tensor Loader**     | `DiffusionGemmaTensorSet`            |  🟡 Impl-Covered | ModelBase, GgufModel, DType decoders | Strict typed loader with shape validation & SWA/Full separation |
| **2. Gemma-4 Backbone**  | Shared transformer block             |  🟡 Impl-Covered | 🔵 Top-k routing, expert dispatch, SIMD | Full/SWA RoPE, V-from-K, fused MoE, scales (synthetic covered) |
| **3. Prompt Prefill**    | Causal encoder path                  |  🟡 Impl-Covered | 🔵 Causal prefill, KV caches         | Multi-block causal prefill attending persistent prefix |
| **4. Canvas Forward**    | Bidirectional decoder path           |  🟡 Impl-Covered | 🔵 Multi-head attention, SimdKernels | Bidirectional canvas cross-attending prompt KV |
| **5. Self-Conditioning** | Soft embeddings + MLP                |  🟡 Impl-Covered | 🔵 Embedding table lookup, MatVec    | Pre-norm + GEGLU MLP + weightless post-norm (disabled step 0) |
| **6. Sampler**           | EntropyBound                         |  🟡 Impl-Covered | 🔵 Softmax, categorical sampling     | Deterministic inverse CDF, Shannon entropy, cumAccepted budget |
| **7. Block Lifecycle**   | Multi-block generation               |  🟡 Impl-Covered | 🔵 Generation loop, token management | Multi-block autoregressive, causal committed block prefill |
| **8. Tests**             | Synthetic unit suite (32 tests)      |  🟡 Impl-Covered | 🔵 xUnit v3 test framework           | 32 synthetic tests passing; real-weight parity pending |
| **9. Real Verification** | Q4_K_M (16.8 GB)                     |  🔴 Pending   | 🔵 Large model mmap, quantized ops   | Awaiting checkpoint download to F:\_models |
| **10. Admission**        | Registry + CLI                       |  🔴 Guarded   | 🔵 ModelCompatibility registration   | Guarded by RealCheckpointGuardTests per Rule 14 |

---

## Infrastructure Available From Recently-Admitted MoE Families

A critical development in Stingray is the substantial maturation of generic MoE infrastructure through recently admitted families:
- **Qwen MoE, Mixtral, Phi-MoE, GLM-4.5-Air, Hunyuan-MoE, Llama 4, and AFMoE**

These recent admissions have proven and hardened the core components required for high-parameter MoE inference:
1. **Router & Gating Execution**: Top-$k$ softmax gating, numerically stable logit handling, and routing dispatch (`SimdKernels`, `ForwardPass.Moe`).
2. **Quantized Expert GEMM / MatVec**: Efficient multi-threaded quantized expert multiplication across CPU and GPU backends.
3. **Multi-Device & KV Cache Handoff**: Vulkan hybrid / layer-split handoff, CPU/GPU KV synchronization, and long-context caching.
4. **SIMD Primitives**: `PureRmsNorm`, `RmsNorm`, `ApplyRoPECachedNeoxPartial`, and in-place softmax.

Consequently, implementing DiffusionGemma does **not** require building an MoE engine from scratch. The work focuses squarely on the **Gemma-4-specific architectural delta**:

```text
Stingray Proven Generic MoE Core
   (Top-k softmax router, quantized expert matmul, KV buffers, SIMD norms)
                                ┼
              Gemma-4 / Diffusion Specific Delta:
    ┌────────────────────────────────────────────────────────┐
    │ 1. Heterogeneous SWA (25 layers) / Full (5 layers)     │
    │ 2. Full-layer V derived from raw K before RoPE         │
    │ 3. Parallel Dense FFN (2112) + 128-expert MoE (704)    │
    │ 4. Fused 128×1408 expert weights (ffn_gate_up_exps)    │
    │ 5. Elementwise router scale & expert down scales       │
    │ 6. Dual output scales (enc_layer_output_scale vs dec)  │
    │ 7. Dual execution modes (EncoderPrefill vs Canvas)     │
    │ 8. Bidirectional canvas cross-attention                │
    │ 9. Soft-embedding self-conditioning MLP pipeline       │
    │ 10. EntropyBound sampler with deterministic SplitMix64 │
    └────────────────────────────────────────────────────────┘
```

---

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

# Phase 0 — Contract Lock

**Objective: eliminate all avoidable semantic uncertainty before implementation.**

* [x] Real GGUF tensor inventory verified.
* [x] 30-layer architecture verified.
* [x] 25 SWA / 5 full layer schedule verified.
* [x] 128 experts / top-8 verified.
* [x] 256-token canvas verified.
* [x] Full-layer V-from-K behaviour verified.
* [x] Gemma-4 block ordering verified against llama.cpp.
* [x] TensorSharp implementation acquired locally.
* [x] TensorSharp prompt-KV implementation identified.
* [x] TensorSharp EntropyBound sampler implementation identified.
* [x] TensorSharp step-0 self-conditioning behaviour identified.
* [x] TensorSharp temperature schedule identified.

Remaining contract questions must be limited to cases where authoritative sources genuinely disagree.

Do **not** leave previously resolved questions marked as open simply because Stingray has not implemented them yet.

---

# Phase 1 — Strict Tensor Loader

**Objective: replace the existing guessed tensor inventory with a real typed representation of the 692-tensor GGUF.**

Create:

```text
DiffusionGemmaTensorSet
DiffusionGemmaSlidingLayerTensors
DiffusionGemmaFullLayerTensors
```

Every required tensor must use:

```csharp
Required(GgufModel model, string name)
```

and fail immediately when missing.

## Required invariants

For SWA layers:

```text
attn_v != null
q_heads = 16
kv_heads = 8
head_dim = 256
rope_dim = 256
```

For full layers:

```text
attn_v == null
q_heads = 16
kv_heads = 2
head_dim = 512
rope_dim = 512
rope_freqs != null
```

Required fused MoE tensors:

```text
ffn_gate_up_exps.weight
ffn_down_exps.weight
ffn_gate_inp.weight
ffn_gate_inp.scale
ffn_down_exps.scale
```

Required layer scales:

```text
layer_output_scale
enc_layer_output_scale
```

Required self-conditioning tensors:

```text
self_cond_pre_norm.weight
self_cond_gate.weight
self_cond_up.weight
self_cond_down.weight
```

No real-model path may silently substitute optional or guessed tensor names.

---

# Phase 2 — Shared Gemma-4 Backbone

**Objective: implement one exact Gemma-4 transformer block usable by both prompt prefill and canvas decoding.**

Create an explicit pass mode:

```csharp
public enum DiffusionGemmaPassMode
{
    EncoderPrefill,
    DecoderCanvas
}
```

The shared layer routine must implement:

```text
hAttn = RMSNorm(x, attn_norm)

q = Q projection
q = per-head RMSNorm(q, attn_q_norm)

k = K projection
k = per-head RMSNorm(k, attn_k_norm)
```

### Full layer

```text
v = weightless RMSNorm(raw-normalized-K)
q = global RoPE(q)
k = global RoPE(k)
```

### SWA layer

```text
v = weightless RMSNorm(hAttn @ Wv)
q = SWA RoPE(q)
k = SWA RoPE(k)
```

Then:

```text
attention
    ↓
attn_output projection
    ↓
post_attention_norm
    ↓
residual
```

Follow with parallel dense and MoE branches.

---

# Phase 3 — Dense FFN + Routed MoE

**Objective: reproduce the Gemma-4 FFN structure exactly.**

## Dense branch

```text
denseNorm = RMSNorm(attnRes, ffn_norm)

gate = denseNorm @ W_gate
up   = denseNorm @ W_up

denseAct = GELU(gate) * up

denseOut =
    RMSNorm(
        denseAct @ W_down,
        post_ffw_norm_1)
```

## Router

The router input is derived from `attnRes`.

It is **not** the dense FFN input and is **not** the MoE expert input.

```text
routerNorm = RMSNorm(attnRes, router norm)
routerNorm *= 1 / sqrt(2816)
routerNorm *= ffn_gate_inp.scale

routerLogits = routerNorm @ ffn_gate_inp.weight
```

Then:

```text
softmax over 128 experts
top-8 selection
renormalize selected weights
```

## Expert branch

For each selected expert:

```text
moeInput = RMSNorm(attnRes, pre_ffw_norm_2)

gate/up = moeInput @ fused gate_up expert weights

gate = first 704 values
up   = second 704 values

act = GELU(gate) * up

expertOut =
    act @ expert down weights
    * expert down scale

moeAccum += routingWeight * expertOut
```

Then:

```text
moeOut = RMSNorm(moeAccum, post_ffw_norm_2)

combined =
    RMSNorm(
        denseOut + moeOut,
        post_ffw_norm)

output =
    (attnRes + combined)
    * layerOutputScale
```

During encoder prefill:

```text
layerOutputScale = enc_layer_output_scale
```

During canvas decoding:

```text
layerOutputScale = layer_output_scale
```

---

# Phase 4 — Prompt Prefill

**Objective: perform a genuine causal Gemma-4 forward over the prompt and retain its K/V state.**

TensorSharp demonstrates the required execution model in:

```text
DiffusionGemmaModel.cs
DiffusionGemmaModel.Cpu.cs
```

Implement equivalent semantics using Stingray primitives.

For each prompt token:

```text
token embedding
    × sqrt(hidden_dim)
    ↓
30 Gemma-4 layers
    ↓
persistent per-layer K/V
```

Prompt attention must be causal.

Each layer must retain K and V using that layer's own geometry.

Do not force all layers into one KV shape.

The persistent cache must support:

```text
SWA:
    KV heads = 8
    head dim = 256

FULL:
    KV heads = 2
    head dim = 512
```

The cache must survive all denoising steps for the current block.

---

# Phase 5 — Canvas Decode

**Objective: execute only the 256-token canvas during repeated denoising steps while reusing prompt K/V.**

TensorSharp provides a concrete reference implementation through:

```text
PrefillPrompt
DecodeCanvas
PrefillPromptInto
DecodeCanvasHidden
```

The Stingray implementation should follow the same high-level separation.

Each canvas step must:

1. receive the current 256-token canvas;
2. receive the previous step's self-conditioning signal;
3. reuse the persistent prompt K/V;
4. run all 30 Gemma-4 layers;
5. allow canvas-to-canvas bidirectional attention;
6. allow canvas-to-prompt attention according to layer geometry;
7. produce logits for the 256 canvas positions.

The canvas must **not** be treated as a causal autoregressive sequence.

---

# Phase 6 — Attention Masking

**Objective: reproduce the actual two-region attention structure.**

The model consists of:

```text
[prompt | canvas]
```

Prompt queries:

```text
causal
```

Canvas queries:

```text
bidirectional over canvas
+
allowed prompt context
```

For full-attention layers:

```text
canvas → all prompt positions
canvas → all canvas positions
```

For SWA layers:

```text
canvas → prompt positions permitted by the 1024-token sliding window
canvas → all canvas positions within the active decode sequence
```

Do not accidentally make canvas attention causal.

Do not accidentally allow prompt queries to see the canvas.

Do not rebuild the prompt forward during every denoising step when persistent prompt KV is available.

---

# Phase 7 — Self-Conditioning

**Objective: implement the TensorSharp/vLLM self-conditioning lifecycle exactly.**

TensorSharp establishes the concrete lifecycle:

```text
step 0:
    no previous prediction
    self-conditioning disabled

step > 0:
    previous step prediction
    ↓
    soft embedding
    ↓
    self-conditioning MLP
    ↓
    current canvas
```

## Soft embedding

Use the complete vocabulary:

```text
P = softmax(processed_logits)

softEmb =
    P @ token_embd.weight

softEmb *= sqrt(2816)
```

Do not prune low-probability vocabulary entries.

Do not use a top-k approximation on the CPU reference path.

## Self-conditioning MLP

```text
normed = RMSNorm(
    softEmb,
    self_cond_pre_norm)

gate = normed @ self_cond_gate
up   = normed @ self_cond_up

act = GELU(gate) * up

signal = act @ self_cond_down
```

Then:

```text
combined = canvasEmbedding + signal

combined =
    weightless RMSNorm(combined)
```

The weightless post-normalization is mandatory.

## Step lifecycle

```text
step 0:
    scUse = 0

step > 0:
    scUse = 1
    use previous step prediction
```

The previous prediction must not accidentally refer to the re-noised working canvas.

---

# Phase 8 — Logit Processing

**Objective: centralize logit semantics so sampler and self-conditioning cannot diverge.**

The final logits must first receive the model's final logit softcap:

```csharp
x = MathF.Tanh(x / 30.0f) * 30.0f;
```

Temperature is then applied for sampling:

```csharp
processed = x / temperature;
```

Keep the raw model logits separate from temperature-scaled probabilities where necessary.

The same processed probability distribution must drive:

```text
argmax
entropy
multinomial sampling
self-conditioning
```

Do not maintain separate subtly different implementations.

---

# Phase 9 — EntropyBound Sampler

**Objective: replace the current approximate sampler with the verified reference behaviour.**

TensorSharp's `DiffusionGemmaSampler` establishes the concrete reference behaviour.

Defaults:

```text
MaxDenoisingSteps = 48
TMax = 0.8
TMin = 0.4
EntropyBound = 0.1
StabilityThreshold = 1
ConfidenceThreshold = 0.005
```

## Temperature schedule

TensorSharp computes:

```text
step = S ... 1

T =
    TMin
    + (TMax - TMin) * (step / S)
```

For the default 48-step schedule this gives:

```text
step 0 ≈ 0.8
final step ≈ 0.408333
```

Do not invent a separate schedule.

Implement the schedule through the sampler/scheduler abstraction so it is testable.

## Per-position distribution

For each canvas position:

```text
scaled logits = processed logits / temperature

P = softmax(scaled logits)

argmax = argmax(scaled logits)

entropy = -Σ P log P
```

## Candidate sampling

The verified TensorSharp host reference uses **multinomial categorical sampling via inverse CDF**, not Gumbel-max.

Therefore the Stingray reference implementation must not describe or implement Gumbel-max as the canonical sampler unless an authoritative upstream source proves that TensorSharp's implementation is incorrect.

For each position:

```text
u = deterministic random number in [0,1)

sample =
    first vocabulary token whose cumulative probability
    >= u
```

## EntropyBound acceptance

Sort positions by ascending entropy.

Accept the lowest-entropy positions while:

```text
sum of strictly earlier accepted entropies <= 0.1
```

Accepted positions receive their sampled token.

Rejected positions receive a fresh random token.

The accepted token is the **multinomial sample**, not necessarily the argmax.

The emitted block remains the deterministic argmax canvas.

## Argmax history

Maintain:

```text
previousArgmaxCanvas
currentArgmaxCanvas
```

Do not compare the re-noised working canvas when determining convergence.

## Stopping condition

Stop when:

```text
argmax canvas stable
AND
mean canvas entropy < 0.005
```

The stability threshold is configurable and defaults to one stable transition according to the TensorSharp reference.

---

# Phase 10 — Block-Autoregressive Lifecycle

**Objective: implement the complete multi-block generation lifecycle.**

Generation:

```text
Prompt
  ↓
Causal Prompt Prefill
  ↓
Persistent Prompt KV
  ↓
Initialize 256-token random canvas
  ↓
Denoising step 0
  ↓
Denoising step 1
  ↓
...
  ↓
Convergence / 48-step limit
  ↓
Commit argmax canvas
  ↓
Causally prefill committed block
  ↓
Extend persistent prefix KV
  ↓
Initialize next 256-token canvas
  ↓
...
```

The committed canvas is the **argmax prediction**, not the re-noised working canvas.

After commitment, the block must be causally processed into the persistent prefix state before the next canvas is generated.

The prefix therefore grows:

```text
initial prompt
        ↓
prompt + block 0
        ↓
prompt + block 0 + block 1
        ↓
...
```

Stop on:

* EOS
* maximum requested tokens
* maximum block count
* cancellation
* other explicitly supported termination conditions

---

# Phase 11 — TensorSharp-Informed Execution Optimizations

**Objective: port useful proven execution strategies without importing TensorSharp's architecture wholesale.**

The following should be considered implementation targets after numerical correctness is established.

## Prompt-KV caching

Port the core idea from:

```text
DiffusionGemmaModel.cs
DiffusionGemmaModel.Cpu.cs
```

Prompt K/V must be calculated once per block and reused across denoising steps.

## Fused expert execution

TensorSharp uses stacked expert representations:

```text
_stackedGateUp
_stackedDown
_perExpertScale
```

and performs grouped/fused expert work rather than naïvely executing 128 independent projections.

Stingray should use its existing MoE dispatch infrastructure to achieve equivalent semantics.

Do not require one C# loop containing 128 ordinary matrix multiplications.

## Batched Q/K/V projections

TensorSharp batches projections that share the same input.

Stingray should reuse its existing optimized multi-projection infrastructure where it produces identical numerical results.

## Device-resident decode

TensorSharp's GPU paths demonstrate that repeatedly moving canvas hidden states and logits between host and device is unnecessarily expensive.

For Vulkan/CUDA implementations:

```text
canvas hidden
    ↓
attention
    ↓
FFN
    ↓
MoE
    ↓
next layer
```

should remain device-resident where existing Stingray infrastructure permits it.

This is an optimization phase, not a prerequisite for correctness.

---

# Phase 12 — Synthetic Numerical Test Suite

**Objective: prove every architectural component independently before loading the real 16.8 GB checkpoint.**

## Tensor layout

Required tests:

```text
RealCheckpoint_Layer5_IsFullAttentionAndHasNoV
RealCheckpoint_Layer4_IsSwaAndHasV
RealCheckpoint_AllRequiredTensorsPresent
RealCheckpoint_FusedExpertShapeMatches128x1408
```

## Gemma-4 block

Required tests:

```text
Gemma4_FullLayer_DerivesVFromRawKBeforeRoPE
Gemma4_SwaLayer_UsesDedicatedVProjection
Gemma4_RouterUsesAttnResidualNotFfnNorm
Gemma4_RouterScaleIsAppliedElementwiseBeforeRouterProjection
Gemma4_FusedGateUpSplits704And704
Gemma4_ExpertDownScaleIsAppliedPerExpert
Gemma4_PostNormAndLayerScaleOrderMatchesReference
```

## Attention

Required tests:

```text
CanvasAttention_IsBidirectional
PromptPrefill_IsCausal
Canvas_DoesNotWritePersistentPrefixKv
FullLayer_UsesGlobalAttention
SwaLayer_UsesWindow
```

## Self-conditioning

Required tests:

```text
SelfConditioning_Step0IsDisabled
SelfConditioning_UsesPreNorm
SelfConditioning_UsesWeightlessPostNorm
SelfConditioning_Uses2112Intermediate
SelfConditioning_UsesExactFullVocabularySoftEmbedding
SelfConditioning_UsesPreviousStepPrediction
```

## Sampler

Required tests:

```text
Sampler_TemperatureContract
Sampler_MultinomialInverseCdf
Sampler_EntropyCalculation
Sampler_EntropyBoundSelection
Sampler_ArgmaxHistoryIgnoresRenoisedCanvas
Sampler_FinalCommitUsesArgmaxCanvas
Sampler_RenoiseRejectedPositions
```

## Lifecycle

Required tests:

```text
Pipeline_PromptKvPersistsAcrossDenoisingSteps
Pipeline_CommittedBlockIsPrefilledCausally
Pipeline_NextCanvasSeesCommittedPrefix
Pipeline_SeedIsReproducible
Pipeline_MultiBlockPrefixContinuity
```

---

# Phase 13 — Cross-Implementation Parity

**Objective: use three independent references to isolate errors.**

For each major stage compare:

```text
Stingray
   ↕
TensorSharp
   ↕
llama.cpp / vLLM / HF
```

Comparison stages:

1. tensor loading
2. embedding
3. Q/K/V projections
4. Q/K RMSNorm
5. full-layer V derivation
6. RoPE
7. attention
8. dense FFN
9. MoE routing
10. expert output
11. post-norm
12. layer scale
13. prompt KV
14. canvas attention
15. self-conditioning
16. final logits
17. entropy
18. multinomial sampling
19. accepted positions
20. re-noised canvas
21. argmax canvas
22. block commit

Where TensorSharp and llama.cpp disagree:

```text
real GGUF
    >
llama.cpp / authoritative upstream
    >
TensorSharp implementation
    >
Stingray
```

The disagreement must be resolved explicitly rather than copied blindly.

---

# Phase 14 — Real 16.8 GB Q4_K_M Verification

**Objective: execute the actual DiffusionGemma checkpoint using Stingray.**

Checkpoint:

```text
unsloth/diffusiongemma-26B-A4B-it-GGUF
```

Expected Q4_K_M size:

```text
~16.8 GB
```

Host:

```text
64 GB RAM
```

The checkpoint fits directly in host memory.

## Verification sequence

### 14.1 Tensor inventory

Load the real checkpoint and prove all required tensors resolve.

### 14.2 Prompt prefill

Run a real prompt through all 30 layers.

Capture:

```text
layer output
K
V
logits
```

### 14.3 Causal comparison

Compare prompt/backbone traces against llama.cpp Gemma-4.

### 14.4 Canvas forward

Run one real 256-token canvas through the decoder path.

### 14.5 Self-conditioning

Compare at least one complete self-conditioning trace against TensorSharp/vLLM.

### 14.6 Sampler

Compare:

```text
temperature
entropy
argmax
sampled token
accepted positions
re-noised positions
```

### 14.7 Single-block generation

Run:

```text
1 prompt
1 canvas
48 maximum denoising steps
```

### 14.8 Multi-block generation

Run at least two blocks and verify prefix continuity.

### 14.9 Determinism

Repeat with the same seed and verify identical results.

### 14.10 Admission

Only after all real-weight verification succeeds:

```text
ModelCompatibility.cs
```

may remove:

```text
diffusion-gemma - NOT admitted
```

---

# Verification Ladder

The required order is:

```text
1. Real GGUF tensor inventory
        ↓
2. Single-layer synthetic numerical oracle
        ↓
3. Full synthetic Gemma-4 backbone
        ↓
4. Real-weight tensor loading
        ↓
5. Real causal prompt prefill
        ↓
6. llama.cpp causal parity
        ↓
7. Real canvas forward
        ↓
8. TensorSharp canvas parity
        ↓
9. Self-conditioning trace parity
        ↓
10. Sampler trace parity
        ↓
11. Single-block generation
        ↓
12. Multi-block generation
        ↓
13. Seed reproducibility
        ↓
14. Performance optimization
        ↓
15. Admission
```

Do not skip directly from synthetic tests to "the output looks reasonable".

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

# Explicit Things NOT To Copy From The Existing Stingray Implementation

The current implementation must not retain:

* guessed DiffusionGemma tensor names;
* mandatory `attn_v` on every layer;
* 256,000 vocabulary assumptions;
* 2816 self-conditioning intermediate assumptions;
* simplified MoE routing;
* missing router input scale;
* missing expert down scales;
* V derived after RoPE;
* causal canvas attention;
* repeated prompt recomputation when prompt-KV caching is available;
* top-k-only soft embeddings on the reference path;
* greedy-only sampling;
* Gumbel-max as an assumed canonical sampler;
* convergence based on the re-noised canvas;
* arbitrary `Random(42)` construction inside generation;
* approximate self-conditioning placement;
* approximate layer scaling.

---

# Performance Strategy

Correctness comes first.

Once the real CPU path is numerically correct:

## Priority 1

Prompt-KV caching.

## Priority 2

Fused/batched Q/K/V projections.

## Priority 3

Batched/fused MoE expert execution.

## Priority 4

Keep hidden states device-resident.

## Priority 5

Fuse LM-head/output-normalization where safe.

## Priority 6

Device-side sampling where the numerical contract can be demonstrated identical to the reference.

Do not sacrifice parity to achieve an early benchmark result.

---

# Definition of Done

DiffusionGemma is **DONE** only when all of the following are true:

```text
REAL Q4_K_M CHECKPOINT LOADS
        AND
ALL 692 TENSORS RESOLVE CORRECTLY
        AND
30-LAYER GEMMA-4 BACKBONE MATCHES REFERENCE
        AND
FULL/SWA ATTENTION GEOMETRY MATCHES REFERENCE
        AND
FULL-LAYER V IS DERIVED FROM RAW K BEFORE ROPE
        AND
DENSE + MOE BRANCHES MATCH REFERENCE
        AND
PROMPT PREFILL IS CAUSAL
        AND
PROMPT KV PERSISTS ACROSS DENOISING STEPS
        AND
CANVAS ATTENTION IS BIDIRECTIONAL
        AND
SELF-CONDITIONING MATCHES REFERENCE
        AND
ENTROPYBOUND SAMPLER MATCHES REFERENCE
        AND
ARGMAX HISTORY / RENOISING SEMANTICS MATCH
        AND
COMMITTED BLOCKS ARE CAUSALLY PREFILLED
        AND
MULTI-BLOCK CACHE LIFECYCLE WORKS
        AND
SEED REPRODUCIBILITY IS PROVEN
        AND
REAL GENERATION IS NON-DEGRADED
        AND
REFERENCE COMPARISON HAS BEEN COMPLETED
```

Only then:

```text
ModelCompatibility.cs
```

may admit:

```text
diffusion-gemma
```

and the family may be described as implemented.

---

# Strategic Outcome

The objective is no longer merely to "finish DiffusionGemma".

The implementation should produce a reusable **Gemma-4 MoE backbone abstraction** inside Stingray that can subsequently support other Gemma-4-derived architectures.

DiffusionGemma then becomes the first demanding consumer of:

```text
heterogeneous attention geometry
+
full/SWA RoPE
+
V-from-K global attention
+
128-expert top-8 MoE
+
fused expert tensors
+
persistent KV caching
+
dual encoder/decoder execution modes
```

This is the architectural work worth retaining after DiffusionGemma itself is complete.
