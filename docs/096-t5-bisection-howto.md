# T5 Encoder Bisection Diagnostic Guide

**Date**: 2026-09-21  
**Purpose**: Isolate where T5 encoder diverges from the C++ reference by comparing internal checkpoints

## Background

The SD3.5 composition bug investigation (see `docs/095-sd35-t5-encoder-bisection-plan.md`) has isolated the issue to the T5 encoder itself. Token IDs are byte-for-byte identical, and CLIP encoders are now verified correct, but the T5 context tensor cosine similarity is only 0.169 (should be ~0.99+).

## The 4-Point Bisection Strategy

Compare tensors at these 4 checkpoints:

1. **E0** - Token embedding output (before any encoder blocks)
2. **A0** - After block 0's self-attention + residual (before block 0's FFN)
3. **F0** - After block 0's FFN + residual (= input to block 1)
4. **final** - After all 24 blocks + final RMSNorm

The FIRST checkpoint whose cosine collapses tells you the exact subsystem:
- E0 collapses → embedding lookup bug
- A0 collapses → self-attention bug in block 0
- F0 collapses → FFN bug in block 0
- final collapses but F0 matches → later-layer accumulation bug

## Diagnostic Infrastructure Added

### C# Side (OpenTail.Stingray)

**Modified**: `src/OpenTail.Stingray.Diffusion/TextEncoders/T5Encoder.cs`

- Added `DiagnosticDump(checkpoint, data)` helper method
- Environment variable: `STINGRAY_T5_DUMP_PREFIX`
- Dumps 4 checkpoints: E0, A0, F0, final
- Output format: raw float32 binary (4 bytes per float, little-endian)
- Filename convention: `{prefix}_{checkpoint}_{callcount}.bin`
  - Call count distinguishes cond (0, 2, 4...) vs uncond (1, 3, 5...) passes

**New test**: `tests/OpenTail.Stingray.Tests.Diffusion/ZZ_ScratchT5BisectionDumpTest.cs`

- `Scratch_DumpT5Checkpoints` - generates C# dumps
- `Scratch_CompareT5CheckpointsAgainstReference` - compares against C++ dumps

### C++ Side (stable-diffusion.cpp)

**Modified**: `examples/stable-diffusion.cpp/src/model/te/t5.hpp`

- Added `diagnostic_dump_t5_tensor(checkpoint, tensor, backend)` helper function
- Environment variable: `SD_DUMP_T5_PREFIX`
- Dumps same 4 checkpoints: E0, A0, F0, final
- Output format: raw float32 binary (same as C#)
- Filename convention: `{prefix}_{checkpoint}_{cond|uncond}.bin`
  - Uses alternating cond/uncond naming based on call count

## How to Run the Bisection

### Step 1: Generate C# Dumps

```bash
cd tests/OpenTail.Stingray.Tests.Diffusion
dotnet test --filter "FullyQualifiedName~ZZ_ScratchT5BisectionDumpTest.Scratch_DumpT5Checkpoints"
```

**Output files** (in `docs/diffusion-samples/`):
- `zz_scratch_csharp_t5_E0_0.bin` (cond)
- `zz_scratch_csharp_t5_A0_0.bin` (cond)
- `zz_scratch_csharp_t5_F0_0.bin` (cond)
- `zz_scratch_csharp_t5_final_0.bin` (cond)
- `zz_scratch_csharp_t5_E0_1.bin` (uncond)
- `zz_scratch_csharp_t5_A0_1.bin` (uncond)
- `zz_scratch_csharp_t5_F0_1.bin` (uncond)
- `zz_scratch_csharp_t5_final_1.bin` (uncond)

### Step 2: Build C++ Reference with Diagnostics

```bash
cd examples/stable-diffusion.cpp
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### Step 3: Generate C++ Dumps

**Important**: Use the EXACT same prompt as the C# test: `"a red apple on a wooden table"`

```bash
cd examples/stable-diffusion.cpp/build/bin

# Set the dump prefix
$env:SD_DUMP_T5_PREFIX="..\..\..\..\docs\diffusion-samples\zz_scratch_cpp_t5"

# Run with T5 enabled (adjust paths to your actual model locations)
.\sd.exe `
  --model ..\..\..\..\models\sd3.5_medium-Q4_K_M.gguf `
  --clip_l ..\..\..\..\models\sd35-medium-aux\text_encoder\model.fp16.safetensors `
  --clip_g ..\..\..\..\models\sd35-medium-aux\text_encoder_2\model.fp16.safetensors `
  --vae ..\..\..\..\models\sd35-medium-aux\vae\diffusion_pytorch_model.safetensors `
  --t5xxl ..\..\..\..\models\flux1-schnell\t5xxl_fp8_e4m3fn.safetensors `
  --prompt "a red apple on a wooden table" `
  --negative-prompt "" `
  --output test_t5_dump.png `
  --cfg-scale 4.5 `
  --steps 20 `
  --sampling-method euler `
  --width 256 `
  --height 256 `
  --seed 42
```

**Output files** (in `docs/diffusion-samples/`):
- `zz_scratch_cpp_t5_E0_cond.bin`
- `zz_scratch_cpp_t5_A0_cond.bin`
- `zz_scratch_cpp_t5_F0_cond.bin`
- `zz_scratch_cpp_t5_final_cond.bin`
- `zz_scratch_cpp_t5_E0_uncond.bin`
- `zz_scratch_cpp_t5_A0_uncond.bin`
- `zz_scratch_cpp_t5_F0_uncond.bin`
- `zz_scratch_cpp_t5_final_uncond.bin`

### Step 4: Compare Checkpoints

```bash
cd tests/OpenTail.Stingray.Tests.Diffusion
dotnet test --filter "FullyQualifiedName~ZZ_ScratchT5BisectionDumpTest.Scratch_CompareT5CheckpointsAgainstReference"
```

**Look for the cosine similarity in the output**:
- E0 cosine ~0.99+ → embedding lookup matches, bug is later
- A0 cosine drops to ~0.1-0.2 → **self-attention bug** (most likely)
- F0 cosine drops → FFN bug
- final cosine drops but F0 matches → later-layer accumulation

## Interpreting Results

### If E0 matches (~0.99+ cosine)
✅ Token embedding lookup is correct. Bug is in the transformer layers.

### If A0 diverges (low cosine)
🔍 **Self-attention bug in block 0**. Check:
1. Q/K/V projections (weight shape, transposition)
2. Attention score computation (scale convention - T5 uses NO scaling!)
3. Relative position bias (already verified, but recheck if this fires)
4. Output projection
5. Residual add

### If F0 diverges but A0 matches
🔍 **FFN bug in block 0**. Check:
1. LayerNorm (RMSNorm, no mean-centering)
2. wi_0/wi_1 projections (gated GELU)
3. GELU activation (must be `gelu_new` tanh-approx, NOT erf-based)
4. wo projection
5. Residual add

### If final diverges but F0 matches
🔍 **Later-layer accumulation bug**. The bug is in blocks 1-23, not block 0.
- Extend bisection to check mid-layer (block 12) and 3/4 point (block 18)
- OR check if the divergence grows gradually (numerical stability) vs suddenly (discrete bug in a specific layer)

## Expected Tensor Shapes

All checkpoints should be:
- **Cond prompt**: `[77, 4096]` = 315,392 floats = 1,261,568 bytes
- **Uncond (empty)**: `[77, 4096]` = 315,392 floats = 1,261,568 bytes

Both use 77 tokens (SD3's T5MaxTokens), not 256.

## Important Notes

1. **Prompt must match exactly**: `"a red apple on a wooden table"` (no trailing punctuation)
2. **Empty uncond**: Use `""` for negative prompt, not a space or other variation
3. **Binary format**: Raw float32, little-endian, no header (same as `Buffer.BlockCopy` in C# and `ggml_backend_tensor_get` in C++)
4. **Call count alternation**: Both implementations dump cond first (call 0), then uncond (call 1)
5. **Temporary diagnostic code**: All `diagnostic_dump_*` code and `ZZ_Scratch*` tests are temporary and NOT for commit

## Known Good Values (from previous investigation)

- CLIP-L pooled: 0.9999 cosine (essentially exact)
- CLIP-G pooled: 0.902 cosine (separate bug, lower priority)
- CLIP context rows (0-76): 0.992 cosine (very good)
- **T5 context rows (77-153): 0.169 cosine** ← THIS IS THE BUG WE'RE ISOLATING

## Next Actions After Finding the Divergence Point

1. Read the relevant C++ code for that specific subsystem
2. Read the C# code for the same subsystem
3. Line-by-line comparison of the math
4. Check for:
   - Transposition differences (row-major vs column-major)
   - Scale factors (especially attention scale - T5 has NO scale!)
   - Activation functions (gelu_new vs gelu_erf)
   - Residual connection order
   - Normalization (RMSNorm eps=1e-6, no bias, no mean-centering)

## Cleanup After Investigation

Once the bug is found and fixed:

1. Remove `DiagnosticDump` calls from `T5Encoder.cs`
2. Remove `diagnostic_dump_t5_tensor` calls from `t5.hpp`
3. Delete `ZZ_ScratchT5BisectionDumpTest.cs`
4. Delete all `zz_scratch_*.bin` files from `docs/diffusion-samples/`
5. Update `docs/095-sd35-t5-encoder-bisection-plan.md` with findings
6. Commit the actual fix, not the diagnostic infrastructure
