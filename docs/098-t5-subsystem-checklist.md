# T5 Encoder Subsystem Bug Checklist

**Date**: 2026-09-21  
**Purpose**: Quick reference for what to check in each subsystem based on bisection results

## If E0 Diverges (Token Embedding)

**Unlikely** - Token IDs are already verified byte-for-byte identical, but check:

### C# Side
- `T5Encoder.Encode()` line ~210-217
- Token embedding lookup: `tokEmb.AsSpan(off, Dim).CopyTo(x.AsSpan(t * Dim, Dim))`
- Offset calculation: `off = tokens[t] * Dim`

### C++ Side
- `T5` struct (likely has an embedding layer)
- Check if there's any normalization or scaling applied after embedding lookup
- Verify the embedding matrix shape and layout (row-major vs column-major)

### Common Bugs
- ❌ Transposed embedding matrix
- ❌ Offset calculation error (should be `token_id * hidden_dim`)
- ❌ Unexpected normalization or scaling
- ❌ Data type conversion issue (fp16 vs fp32)

---

## If A0 Diverges (Self-Attention)

**Most likely candidate** - Self-attention is complex and has many moving parts.

### C# Side: `T5Encoder.SelfAttention()` (lines ~256-310)

Check in order:
1. **Q/K/V Projections** (lines ~268-270)
   - `DiffusionOps.Linear(x, qW, null, seq, Dim, Dim)`
   - Weight shapes: `[Dim, Dim]` = `[4096, 4096]`
   - Output shapes: `[seq, Dim]`
   - ⚠️ Check if weights need transposition

2. **Attention Score Computation** (lines ~279-287)
   - `float dot = TensorPrimitives.Dot(qSpan, k.AsSpan(kOff, HeadDim))`
   - **NO SCALE FACTOR** - T5 intentionally omits `1/sqrt(d_head)` scaling
   - Relative bias added: `scores[i * seq + j] = dot + relBias[relRowOff + j]`

3. **Relative Position Bias** (lines ~281-282)
   - Index: `relRowOff = (h * seq + i) * seq`
   - Shape: `[Heads, seq, seq]` = `[64, 77, 77]`
   - Already verified correct, but double-check indexing

4. **Softmax** (line ~288)
   - Applied per-row: `DiffusionOps.Softmax(scores, seq)`
   - Should be: `exp(x[i]) / sum(exp(x))` for each row

5. **Value Aggregation** (lines ~290-297)
   - Weighted sum: `attnOut[outOff + d] += w * v[vOff + d]`
   - Check loop bounds and offsets

6. **Output Projection** (line ~303)
   - `DiffusionOps.Linear(attnOut, oW, null, seq, Dim, Dim)`

### C++ Side: `T5Attention::forward()` (t5.hpp, around line 220-250)

Check equivalent operations:
1. Q/K/V projections via Linear blocks
2. K scaling by `sqrt(d_head)` THEN division by `sqrt(d_head)` in attention → net effect is NO scaling
3. Relative position bias computation and application
4. Attention computation via `ggml_ext_attention_ext`
5. Output projection

### Common Bugs
- ❌ **Weight transposition** - MOST COMMON (row-major vs column-major)
- ❌ Incorrect attention scale (should be NO scale for T5)
- ❌ Relative bias indexing (head, query, key order)
- ❌ Softmax dimension (should be over keys, not queries)
- ❌ Head dimension calculation (`inner_dim / num_heads`)
- ❌ Output projection weight shape

### How to Isolate Further

If A0 diverges, add intermediate dumps:
- After Q/K/V projections (before attention)
- After attention scores (before softmax)
- After softmax
- After value aggregation (before output projection)

---

## If F0 Diverges (Feed-Forward)

**Second most likely** - FFN has gated activation and specific GELU variant.

### C# Side: `T5Encoder.FeedForward()` (lines ~311-329)

Check in order:
1. **Up-projections** (lines ~318-319)
   - `gate = DiffusionOps.Linear(x, wi0W, null, seq, Dim, FfDim)`
   - `val = DiffusionOps.Linear(x, wi1W, null, seq, Dim, FfDim)`
   - Input shape: `[seq, 4096]`
   - Output shape: `[seq, 10240]`

2. **Gated GELU** (lines ~323-324)
   - `DiffusionOps.GeluInPlace(gate)` - must be `gelu_new` (tanh-approx), NOT erf-based
   - `TensorPrimitives.Multiply(gate, val, gate)`
   - Formula: `h = gelu_new(gate) * val`

3. **Down-projection** (line ~326)
   - `DiffusionOps.Linear(gate, woW, null, seq, FfDim, Dim)`
   - Input shape: `[seq, 10240]`
   - Output shape: `[seq, 4096]`

### C++ Side: `T5DenseGatedActDense::forward()` (t5.hpp, around line 150-160)

Check equivalent operations:
1. Two up-projections: `wi_0` and `wi_1`
2. GELU activation: `ggml_ext_gelu(..., true)` - the `true` flag means `gelu_new` variant
3. Element-wise multiply: `ggml_mul_inplace`
4. Down-projection: `wo`

### Common Bugs
- ❌ **Wrong GELU variant** - must be `gelu_new` (tanh-approx), not erf-based
- ❌ Weight transposition (especially for up/down projections)
- ❌ Gate/value order (should be `gelu(wi_0) * wi_1`, not reversed)
- ❌ Dimension mismatch (FfDim should be 10240, not something else)

### GELU Variants (CRITICAL)

**Correct** (gelu_new, tanh-approx):
```
0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
= 0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x³)))
```

**Wrong** (gelu_erf, exact):
```
0.5 * x * (1 + erf(x / sqrt(2)))
```

Verify: `DiffusionOps.Gelu` uses tanh formula, NOT `DiffusionOps.GeluExact`.

---

## If final Diverges but F0 Matches (Later Layers)

**Accumulation bug** - Error compounds across 24 layers.

### Possible Causes

1. **Layer Normalization** (RMSNorm)
   - Check eps value (should be `1e-6`)
   - No mean-centering (RMSNorm, not LayerNorm)
   - No bias term
   - Formula: `x * rsqrt(mean(x²) + eps) * weight`

2. **Residual Connections**
   - Order: `x = x + sublayer_output` (not `x = sublayer_output + x`)
   - Applied after both self-attention and FFN

3. **Numerical Stability**
   - Check for NaN/Inf propagation
   - Verify weight scaling (some models apply scale factors)

4. **Block-Specific Bugs**
   - Only block 0 has relative attention bias
   - Blocks 1-23 reuse the bias from block 0
   - Check if bias is properly passed/reused

### How to Isolate

Add dumps at:
- Block 6 output (1/4 point)
- Block 12 output (1/2 point)
- Block 18 output (3/4 point)

This narrows down which range of layers contains the bug.

---

## Verification Checklist

After finding and fixing the bug:

- [ ] Re-run full SD3.5 comparison test
- [ ] T5 context cosine similarity > 0.99
- [ ] CLIP-L pooled cosine > 0.999 (already verified)
- [ ] CLIP-G pooled cosine improves (separate bug, but check)
- [ ] Generated image matches reference (apple full-frame, centered)
- [ ] CPU and GPU backends produce identical results
- [ ] Remove all diagnostic dump infrastructure
- [ ] Update docs/095 with findings
- [ ] Commit only the actual fix

---

## Common Pitfalls

1. **Don't trust visual inspection** - Images can look similar with wrong math
2. **Use tensor dumps, not intuition** - Cosine similarity is the ground truth
3. **Check weight shapes carefully** - Row-major vs column-major is subtle
4. **Verify conventions** - Some frameworks use different dimension orderings
5. **Test both cond and uncond** - Empty prompt exercises different code paths

---

## Reference Values (Known Good)

From previous investigation (docs/095):

| Component | Cosine Similarity | Status |
|-----------|------------------|---------|
| CLIP-L pooled (cond) | 0.9999 | ✅ Excellent |
| CLIP-G pooled (cond) | 0.902 | ⚠️ Separate bug |
| CLIP context rows | 0.992 | ✅ Very good |
| **T5 context rows** | **0.169** | ❌ **BUG** |

Goal: Get T5 context cosine to ~0.99+, matching CLIP rows.

---

## Quick Reference: File Locations

**C# T5 Implementation**:
- `src/OpenTail.Stingray.Diffusion/TextEncoders/T5Encoder.cs`

**C++ T5 Implementation**:
- `examples/stable-diffusion.cpp/src/model/te/t5.hpp`

**Test Infrastructure**:
- `tests/OpenTail.Stingray.Tests.Diffusion/ZZ_ScratchT5BisectionDumpTest.cs`

**Dump Files** (after running bisection):
- `docs/diffusion-samples/zz_scratch_csharp_t5_*.bin`
- `docs/diffusion-samples/zz_scratch_cpp_t5_*.bin`
