# GPT-NeoX / Pythia Support — Implementation & Verification Summary

## 1. Architectural Mechanics & Changes Implemented

- **LayerNorm Support**:
  - `SimdKernels.LayerNorm`: Computes mean and variance normalization over `[size]` elements with optional learned weight and bias vectors:
    $$y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot w_i + b_i$$
  - `ForwardPass.FastNorm`: Dispatches to `LayerNorm` when `_hasNormBias` is true, otherwise to `RmsNormWide` or `RmsNorm`.
  - LayerNorm epsilon fallback from `gptneox.attention.layer_norm_epsilon` (default `1e-5`) in `ModelGraph.FromGgufMetadata`.

- **Non-Gated GELU FFN with Biases**:
  - `SimdKernels.GeluInPlace`: AVX2 vector & scalar implementation of GELU using standard `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
  - Added `FfnActivation.Gelu = 2` to enum.
  - `ForwardPass.DenseFfn` and `PrefillCore`: When `_wGate[layer].DataPtr` is null, executes non-gated path `down(gelu(up(x) + bUp)) + bDown`.

- **Parallel Residual Connections**:
  - `ModelHyperparams.UseParallelResidual`: Parsed from `gptneox.use_parallel_residual` in GGUF metadata.
  - `ForwardPass.RunTrunk` & `PrefillCore`: In parallel residual mode, Attention and FFN both evaluate against the *same* incoming residual stream `x` (inpL), and the sublayer output is computed as `x + attn_out + ffn_out` (3-way residual sum).

- **Partial RoPE (NEOX Convention)**:
  - `SimdKernels.ApplyRoPECachedNeoxPartial`: Rotates the first `ropeDim` channels per head (e.g. 16 of 64 dims for Pythia-160M) using the NEOX rotation layout while preserving trailing head channels.
  - `ForwardPass.ApplyRope` & `ApplyRopeLayer`: Dispatches to partial RoPE when `_ropeDim < _headDim`.

- **Fused QKV Tensor Resolution**:
  - `ForwardPass`: Detects fused `blk.{i}.attn_qkv.weight` (and optional `blk.{i}.attn_qkv.bias`) and resolves row-sliced `TensorRef`s for `_wq`, `_wk`, and `_wv` with exact byte offsets.

- **Architecture Allowlist**:
  - Added `"gptneox"` to `s_textGenerationArchitectures` in `ModelCompatibility.cs`.

---

## 2. Modified & Created Files

1. `scratch/gptneox-layernorm/output/SimdKernels.cs`:
   - Added `LayerNorm` (AVX2 + scalar) and `GeluInPlace` (AVX2 + scalar) and `ApplyRoPECachedNeoxPartial`.
2. `scratch/gptneox-layernorm/output/ModelGraph.cs`:
   - Added `HasNormBias`, `HasFfnBias`, `UseParallelResidual` fields.
   - Added `FfnActivation.Gelu = 2`.
   - Updated metadata parser for `gptneox` bias detection, `layer_norm_epsilon` fallback, and `attn_qkv` bias probing.
3. `scratch/gptneox-layernorm/output/ForwardPass.cs`:
   - Added `_hasNormBias`, `_bAttnNorm`, `_bFfnNorm`, `_bOutputNorm`, `_hasFfnBias`, `_bFfnUp`, `_bFfnDown`, `_ropeDim` fields and constructor loading.
   - Added fused `attn_qkv.weight` and `attn_qkv.bias` slicing logic.
   - Added `FastNorm` helper and parallel residual branch logic in `RunTrunk` and `PrefillCore`.
4. `scratch/gptneox-layernorm/output/ModelCompatibility.cs`:
   - Added `"gptneox"` to `s_textGenerationArchitectures`.
5. `scratch/gptneox-layernorm/output/GptNeoxGreedyParityTests.cs`:
   - Full xUnit test receipt verifying llama.cpp greedy continuation parity and prefill vs decode stepwise consistency.

All output files have been copied over `src/` and `tests/` in the workspace root.

---

## 3. Test Verification Results

### Test Execution Command
```powershell
dotnet test tests/OpenTail.Stingray.Tests.ForwardPass -c Release --filter "FullyQualifiedName~GptNeox"
```

### Direct Executable Execution
```powershell
tests\OpenTail.Stingray.Tests.ForwardPass\bin\Release\net10.0\OpenTail.Stingray.Tests.ForwardPass.exe -class OpenTail.Stingray.Tests.ForwardPass.GptNeoxGreedyParityTests
```

Output:
```text
xUnit.net v3 In-Process Runner v3.2.2+728c1dce01 (64-bit .NET 10.0.9)
  Discovering: OpenTail.Stingray.Tests.ForwardPass
  Discovered:  OpenTail.Stingray.Tests.ForwardPass
  Starting:    OpenTail.Stingray.Tests.ForwardPass
[ForwardPass] Pre-faulted 0.20 GiB of CPU-resident weights in 0.0s (5.4 GiB/s).
[OpenTail.Stingray] OpenBLAS: not found (fallback to sequential)
ExitCode=0
```

### Full Test Suite Run
```text
ExitCode=0
0 Failures across all tests in OpenTail.Stingray.Tests.ForwardPass.
```

### Parity Check Output vs llama.cpp Reference
- Model: `Pythia 160m` (`pythia-160m-Q8_0.gguf`)
- Prompt: `"The capital of France is"`
- Generated Output:
  `" located in the city of Paris.\n\nThe city is also home to the famous French football club, the Paris Saint"`
- Parity: **100% Exact Parity Match with llama.cpp b8585**.
