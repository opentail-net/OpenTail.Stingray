# Proposed Documentation Addition for `docs/01-gguf-model-coverage-plan.md`

### 1h. `gptneox` — ADMITTED 2026-08-08, 24-token exact continuation parity receipt against llama.cpp

`EleutherAI/pythia-160m` (Apache-2.0), via `mradermacher/pythia-160m-GGUF` `pythia-160m-Q8_0.gguf` (164.82 MB).

**What GPT-NeoX / Pythia needed:**
1. **LayerNorm with Learned Bias**: Replaces RMSNorm. Evaluates mean-subtraction and variance scaling with learned weight and bias vectors:
   $$y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot w_i + b_i$$
   Implemented in `SimdKernels.LayerNorm` (AVX2 + scalar) and integrated into `ForwardPass.FastNorm`. `gptneox.attention.layer_norm_epsilon` (1e-5) parsed in `ModelGraph.FromGgufMetadata`.
2. **Non-Gated GELU FFN with Learned Biases**: Replaces gated SiLU. Evaluates `down(gelu(up(x) + bUp)) + bDown`. Implemented in `SimdKernels.GeluInPlace` (AVX2 + scalar) and `FfnActivation.Gelu = 2`.
3. **True Parallel Residual Connections**: Attention and FFN both receive the same pre-norm input `x` (inpL), and sublayer outputs are combined in a 3-way residual sum `x + attn_out + ffn_out`. Gated by `ModelHyperparams.UseParallelResidual` parsed from `gptneox.use_parallel_residual`.
4. **Partial RoPE (NEOX Rotation Layout)**: Rotates only the first `rope.dimension_count` (e.g., 16 of 64 headDim) channels per head using `SimdKernels.ApplyRoPECachedNeoxPartial`.
5. **Fused QKV Tensor Resolution**: Detects `blk.{i}.attn_qkv.weight` and `blk.{i}.attn_qkv.bias` in GGUF metadata and constructs row-offset `TensorRef`s for `_wq`, `_wk`, `_wv`, `_bq`, `_bk`, and `_bv`.

**Parity Receipt & Invariant Verification:**
- `GptNeox_GreedyContinuation_MatchesLlamaCpp`: 24/24 generated tokens match `llama.cpp b8585` exactly on prompt `"The capital of France is"`.
- `GptNeox_DecodeStepwise_AgreesWithSinglePassPrefill`: Single-pass batched `PrefillCore` and single-token `RunTrunk` agree on argmax with max diff < 1.0.
