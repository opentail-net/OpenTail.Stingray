# E0 Checkpoint Status - 2026-09-21 Evening

## Current Status

### ✅ C# E0 Checkpoint Ready
- **File**: `docs/diffusion-samples/zz_scratch_csharp_t5_E0_0.bin`
- **Shape**: [77, 4096] = 315,392 floats
- **Statistics**:
  - Mean: 0.019779
  - Std: 5.661551
  - Min: -104
  - Max: 224
- **First token (row 0) sample values**: [0.375, -0.0625, 0.140625, 0.4375, -9.0, ...]

### ❌ C++ E0 Checkpoint Blocked
**Issue**: C++ `sd-cli.exe` execution crashes before T5 encoding completes.

**Error pattern**:
```
[INFO ] stable-diffusion.cpp:905  - Version: SD3.x 
[INFO ] stable-diffusion.cpp:4371 - sampling using Euler method
[crashes during inference, before T5 diagnostic dumps are written]
```

**Attempted solutions**:
1. ✗ Using `--diffusion-model` parameter
2. ✗ Adding `--vae-format sd3`
3. ✗ Using `--rng cpu`
4. ✗ Various parameter combinations

The diagnostic hooks are correctly implemented in `t5.hpp`, but the execution crashes before reaching the T5 encoding stage.

## Why E0 is Critical

E0 is the **simplest possible checkpoint** - it's just token embedding lookup with no transformer operations:

```csharp
// C# E0 generation (T5Encoder.cs, line ~210-217)
for (int t = 0; t < seq; t++)
{
    int off = tokens[t] * Dim;  // Dim = 4096
    tokEmb.AsSpan(off, Dim).CopyTo(x.AsSpan(t * Dim, Dim));
}
```

**If E0 matches between C# and C++, it clears**:
- ✓ `shared.weight` tensor loading
- ✓ Token → row indexing (offset calculation)
- ✓ Embedding stride/layout
- ✓ Float conversion (FP8/FP16/FP32)
- ✓ Row-major vs column-major orientation

**If E0 diverges**, it immediately points to the weight-loading infrastructure before we analyze any attention or FFN operations.

## Proper E0 Comparison Methodology

Once C++ E0 is obtained, compare:

### 1. Shape and Size
- Both should be [77, 4096]
- Total: 315,392 floats = 1,261,568 bytes

### 2. Overall Statistics
- Mean (should be similar, near 0)
- Std deviation
- Min/Max values
- Range distribution

### 3. Row-by-Row Analysis
For each of the 77 tokens:
- Cosine similarity (expect ~1.0 for identical embeddings)
- Max absolute difference
- RMS difference

**Critical**: Compare actual token positions:
- Token 0 (first token)
- Token 1 
- Token 9 or 10 (end of actual prompt before padding)
- Token 76 (last position, likely EOS or padding)

### 4. Specific Value Checks
Compare first 10-20 values of specific rows to catch:
- Transposition (would swap within-row values)
- Stride errors (would skip values)
- Type conversion issues (would show systematic scaling)

## Alternative Approaches to Get C++ E0

###Option 1: Fix C++ Execution (preferred)
- Debug why `sd-cli.exe` crashes during inference
- The model loads successfully, T5 encoder initializes, but crashes before encoding
- Possibly try with a simpler model or minimal parameters

### Option 2: Direct Weight Extraction (backup)
Since E0 is just embedding lookup, we could:
1. Load `t5xxl_fp8_e4m3fn.safetensors` directly
2. Extract `shared.weight` tensor
3. Get token IDs from C# tokenizer
4. Manually construct E0 by copying the corresponding rows
5. Compare against C# E0

This bypasses the C++ execution entirely for E0 validation.

### Option 3: Simpler C++ Test
Create a minimal T5-only test program that:
- Loads just the T5 encoder
- Encodes a single prompt
- Dumps E0 and exits (no diffusion, no VAE, no image generation)

## Token IDs Needed

For "a red apple on a wooden table" padded to 77 tokens:
- **Actual tokens**: Need to dump from C# tokenizer (appears to be ~10 real tokens + EOS)
- **Padding**: Remaining positions filled with token ID 0

Command to get token IDs:
```csharp
var tokenizer = T5Tokenizer.FromFile("tokenizer.json", maxLen: 77);
var tokens = tokenizer.Tokenize("a red apple on a wooden table");
// Pad to 77 with token 0
```

## Expected Outcome

**If E0 matches** (cosine ~0.99+ per row):
→ Move to full 4-point bisection tomorrow (E0, A0, F0, final)
→ Focus shifts to attention/FFN implementation

**If E0 diverges**:
→ Stop and investigate weight loading
→ Check tensor shapes, layouts, conversions
→ Don't proceed to attention analysis until embeddings are correct

## Files Created Tonight

- C# dumps: `zz_scratch_csharp_t5_E0_0.bin` (and A0, F0, final)
- This status doc: `docs/099-e0-checkpoint-status.md`
- Diagnostic infrastructure: Already in place in both C# and C++ code

## Recommendation for Tomorrow

1. **Priority 1**: Get C++ E0 dump working (try alternative approaches if needed)
2. **Priority 2**: Perform detailed E0 comparison using methodology above
3. **If E0 matches**: Proceed with full 4-point bisection (get A0, F0, final from C++)
4. **If E0 diverges**: Focus on weight loading before any attention analysis

The C++ execution issue is the current blocker. Once resolved, E0 comparison should take < 5 minutes and provide immediate, unambiguous diagnostic value.
