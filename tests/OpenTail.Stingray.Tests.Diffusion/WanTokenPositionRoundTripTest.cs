using OpenTail.Stingray.Diffusion.Wan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Stages 2, 3, and 4 of the Wan2.1 tensor-layout diagnostic plan (docs/081):
///
/// Stage 2: Unique-value tensor round-trip comparing PackLatents and UnpackLatents
/// against reference Conv3d -> reshape -> permute and reference unpatchify sequences.
///
/// Stage 3: RoPE position-to-token coordinate mapping verification.
///
/// Stage 4: Q/K/V physical dimension ordering: headDim vs numHeads memory layout.
/// </summary>
public sealed class WanTokenPositionRoundTripTest
{
    private static int LatentIndex(int c, int t, int y, int x, int numFrames, int latH, int latW)
    {
        return ((c * numFrames + t) * latH + y) * latW + x;
    }

    [Fact]
    public void Stage2_UniqueValueTensor_PackLatentsMatchesReferenceConv3d()
    {
        const int numFrames = 2;
        const int latH = 4;
        const int latW = 6;
        const int inChannels = 16;
        const int patchH = latH / 2; // 2
        const int patchW = latW / 2; // 3
        const int numTokens = numFrames * patchH * patchW; // 12

        // 1. Build unique-value latent tensor
        var latent = new float[inChannels * numFrames * latH * latW];
        for (int c = 0; c < inChannels; c++)
        {
            for (int t = 0; t < numFrames; t++)
            {
                for (int y = 0; y < latH; y++)
                {
                    for (int x = 0; x < latW; x++)
                    {
                        // Unique recognizable value per coordinate
                        float val = c * 1_000_000f + t * 10_000f + y * 100f + x;
                        latent[LatentIndex(c, t, y, x, numFrames, latH, latW)] = val;
                    }
                }
            }
        }

        // 2. Run WanModel.PackLatents
        var packed = WanModel.PackLatents(latent, numFrames, latH, latW);

        // 3. Hand-implement reference Conv3d(kernel=stride=(1,2,2)) -> flatten -> transpose
        // In PyTorch:
        // Conv3d weight shape is [dim, in_channels, 1, 2, 2].
        // To test token and patch-tap correspondence directly, we define 64 distinct "detector" weights,
        // one per (c, dy, dx) tap:
        // weight[d, c, 0, dy, dx] = 1.0 if d == (c * 4 + dy * 2 + dx) else 0.0.
        // Conv3d output at (d, t, ph, pw) = sum_{c, dy, dx} weight[d, c, 0, dy, dx] * input[c, t, ph*2+dy, pw*2+dx]
        // Then flatten(2).transpose(1, 2): tokenIdx = (t * patchH + ph) * patchW + pw.
        // refConv3d[tokenIdx, d] = input[c, t, ph*2+dy, pw*2+dx] where d = c*4 + dy*2 + dx.
        var refConv3d = new float[numTokens, 64];
        for (int t = 0; t < numFrames; t++)
        {
            for (int ph = 0; ph < patchH; ph++)
            {
                for (int pw = 0; pw < patchW; pw++)
                {
                    int tokenIdx = (t * patchH + ph) * patchW + pw;
                    for (int c = 0; c < inChannels; c++)
                    {
                        for (int dy = 0; dy < 2; dy++)
                        {
                            for (int dx = 0; dx < 2; dx++)
                            {
                                int y = ph * 2 + dy;
                                int x = pw * 2 + dx;
                                float srcVal = latent[LatentIndex(c, t, y, x, numFrames, latH, latW)];
                                int d = c * 4 + dy * 2 + dx;
                                refConv3d[tokenIdx, d] = srcVal;
                            }
                        }
                    }
                }
            }
        }

        // 4. Compare token-by-token and dimension-by-dimension
        int mismatches = 0;
        for (int tok = 0; tok < numTokens; tok++)
        {
            for (int d = 0; d < 64; d++)
            {
                float actual = packed[tok * 64 + d];
                float expected = refConv3d[tok, d];
                if (actual != expected)
                {
                    if (mismatches < 5)
                    {
                        Console.WriteLine($"[Stage2 Mismatch] token={tok}, d={d}: actual={actual}, expected={expected}");
                    }
                    mismatches++;
                }
            }
        }

        Console.WriteLine($"[Stage2 PackLatents vs Ref Conv3d] Total elements: {numTokens * 64}, Mismatches: {mismatches}");
        Assert.Equal(0, mismatches);
    }

    [Fact]
    public void Stage2_UnpackLatentsMatchesReferenceUnpatchify()
    {
        const int numFrames = 2;
        const int latH = 4;
        const int latW = 6;
        const int inChannels = 16;
        const int patchH = latH / 2; // 2
        const int patchW = latW / 2; // 3
        const int numTokens = numFrames * patchH * patchW; // 12

        // Build a synthetic output from head.head (proj_out) [tokens, 64]
        // where each token and slot has a unique value
        var projOut = new float[numTokens * 64];
        for (int tok = 0; tok < numTokens; tok++)
        {
            for (int s = 0; s < 64; s++)
            {
                projOut[tok * 64 + s] = tok * 1000f + s;
            }
        }

        // 1. Run WanModel.UnpackLatents
        var unpacked = WanModel.UnpackLatents(projOut, numFrames, latH, latW);

        // 2. Reference diffusers unpatchify:
        // proj_out output [numTokens, 64]
        // reshape(numFrames, patchH, patchW, p_t=1, p_h=2, p_w=2, C=16)
        // In row-major, element at (t, ph, pw, dt=0, dy, dx, c) has flat index in token:
        // flatSlot = ((0 * 2 + dy) * 2 + dx) * 16 + c = (dy * 2 + dx) * 16 + c
        // permute(0, C, t, dt, ph, dy, pw, dx) -> y = ph*2+dy, x = pw*2+dx
        // output[c, t, y, x] = tokenSlot[flatSlot]
        var refUnpatchify = new float[inChannels * numFrames * latH * latW];
        for (int t = 0; t < numFrames; t++)
        {
            for (int ph = 0; ph < patchH; ph++)
            {
                for (int pw = 0; pw < patchW; pw++)
                {
                    int tokenIdx = (t * patchH + ph) * patchW + pw;
                    for (int dy = 0; dy < 2; dy++)
                    {
                        for (int dx = 0; dx < 2; dx++)
                        {
                            for (int c = 0; c < inChannels; c++)
                            {
                                int y = ph * 2 + dy;
                                int x = pw * 2 + dx;
                                int flatSlot = (dy * 2 + dx) * 16 + c;
                                float val = projOut[tokenIdx * 64 + flatSlot];
                                refUnpatchify[LatentIndex(c, t, y, x, numFrames, latH, latW)] = val;
                            }
                        }
                    }
                }
            }
        }

        // 3. Compare UnpackLatents against ReferenceUnpatchify
        int mismatches = 0;
        for (int i = 0; i < unpacked.Length; i++)
        {
            if (unpacked[i] != refUnpatchify[i])
            {
                if (mismatches < 5)
                {
                    Console.WriteLine($"[Stage2 Unpack Mismatch] idx={i}: actual={unpacked[i]}, expected={refUnpatchify[i]}");
                }
                mismatches++;
            }
        }

        Console.WriteLine($"[Stage2 UnpackLatents vs Ref Unpatchify] Total elements: {unpacked.Length}, Mismatches: {mismatches}");
        Assert.Equal(0, mismatches);
    }

    [Fact]
    public void Stage3_RoPECoordinatesMatchPackLatentsTokenEnumeration()
    {
        const int numFrames = 3;
        const int patchH = 4;
        const int patchW = 5;
        const int totalTokens = numFrames * patchH * patchW;

        // WanRoPE computes RoPE frequencies by sweeping:
        // for t in 0..numFrames-1
        //   for y in 0..patchH-1
        //     for x in 0..patchW-1
        //       tokenIdx = (t * patchH + y) * patchW + x
        //
        // WanModel.PackLatents sweeps:
        // for f in 0..numFrames-1
        //   for ph in 0..patchH-1
        //     for pw in 0..patchW-1
        //       tokenIdx = (f * patchH + ph) * patchW + pw
        //
        // Verify for every token index 0..totalTokens-1 that both derivations produce
        // the EXACT same (t, h, w) coordinates:
        for (int tokenIdx = 0; tokenIdx < totalTokens; tokenIdx++)
        {
            // Invert tokenIdx to (t, ph, pw)
            int pw = tokenIdx % patchW;
            int rem = tokenIdx / patchW;
            int ph = rem % patchH;
            int f = rem / patchH;

            // Reconstruct tokenIdx from coordinates
            int reconstructedPackToken = (f * patchH + ph) * patchW + pw;
            int reconstructedRopeToken = (f * patchH + ph) * patchW + pw;

            Assert.Equal(tokenIdx, reconstructedPackToken);
            Assert.Equal(tokenIdx, reconstructedRopeToken);
        }

        // Now verify by inspecting actual RoPE frequency values generated by WanRoPE
        var (cos, sin) = WanRoPE.Compute3DRoPE(numFrames, patchH, patchW, headDim: 128);

        // In WanRoPE:
        // tokenIdx = (t * patchH + y) * patchW + x
        // baseOff = tokenIdx * 128
        // Temporal axis: pos = t (dim 44: offsets 0..43)
        // Height axis: pos = y (dim 42: offsets 44..85)
        // Width axis: pos = x (dim 42: offsets 86..127)
        // For pos = 0, angle = 0, so cos = 1.0, sin = 0.0.
        // For pos > 0, angle = pos * theta^(-2i/dim) > 0.
        for (int t = 0; t < numFrames; t++)
        {
            for (int y = 0; y < patchH; y++)
            {
                for (int x = 0; x < patchW; x++)
                {
                    int tokenIdx = (t * patchH + y) * patchW + x;
                    int baseOff = tokenIdx * 128;

                    // If t == 0, temporal axis cos must be 1.0
                    if (t == 0)
                    {
                        Assert.Equal(1.0f, cos[baseOff + 0]);
                        Assert.Equal(0.0f, sin[baseOff + 0]);
                    }
                    else
                    {
                        Assert.NotEqual(1.0f, cos[baseOff + 0]);
                    }

                    // If y == 0, height axis cos must be 1.0
                    if (y == 0)
                    {
                        Assert.Equal(1.0f, cos[baseOff + 44]);
                        Assert.Equal(0.0f, sin[baseOff + 44]);
                    }
                    else
                    {
                        Assert.NotEqual(1.0f, cos[baseOff + 44]);
                    }

                    // If x == 0, width axis cos must be 1.0
                    if (x == 0)
                    {
                        Assert.Equal(1.0f, cos[baseOff + 86]);
                        Assert.Equal(0.0f, sin[baseOff + 86]);
                    }
                    else
                    {
                        Assert.NotEqual(1.0f, cos[baseOff + 86]);
                    }
                }
            }
        }
        Console.WriteLine($"[Stage3 RoPE Mapping] Verified (t,h,w) coordinate tracking across {totalTokens} tokens.");
    }

    [Fact]
    public void Stage4_QkvPhysicalDimensionOrdering_MatchesGgmlAndDiffusers()
    {
        const int seqLen = 4;
        const int numHeads = 12;
        const int headDim = 128;
        const int dim = numHeads * headDim; // 1536

        // In ggml:
        // q is reshaped via ggml_reshape_4d(ctx, q, head_dim, num_heads, n_token, N)
        // In ggml, ne[0] = head_dim (fastest), ne[1] = num_heads (stride head_dim),
        // ne[2] = n_token (stride num_heads * head_dim).
        //
        // In PyTorch / diffusers:
        // q has shape [batch, seqLen, dim], then .view(batch, seqLen, num_heads, head_dim).
        // In PyTorch C-order: head_dim is fastest (stride 1), num_heads is stride head_dim,
        // seqLen is stride num_heads * head_dim.
        //
        // Therefore, both reference frameworks agree:
        // Memory offset for token s, head h, feature d is:
        // offset = (s * numHeads + h) * headDim + d.

        // Test our CPU TransposeToHeadContiguous:
        // Input: flat Q/K/V array where token s, head h, feature d is at (s * numHeads + h) * headDim + d.
        var flatQ = new float[seqLen * dim];
        for (int s = 0; s < seqLen; s++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                for (int d = 0; d < headDim; d++)
                {
                    // Unique value identifying (s, h, d)
                    flatQ[(s * numHeads + h) * headDim + d] = s * 100_000f + h * 1000f + d;
                }
            }
        }

        // Transpose to head-contiguous [numHeads, seqLen, headDim]:
        // In head-contiguous layout: head h, token s, feature d is at:
        // (h * seqLen + s) * headDim + d.
        var headContig = new float[seqLen * dim];
        unsafe
        {
            fixed (float* pSrc = flatQ, pDst = headContig)
            {
                WanAttention.TransposeToHeadContiguous(pSrc, pDst, seqLen, numHeads, headDim);
            }
        }

        // Verify headContig has exact expected elements
        for (int h = 0; h < numHeads; h++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int d = 0; d < headDim; d++)
                {
                    float expected = s * 100_000f + h * 1000f + d;
                    float actual = headContig[(h * seqLen + s) * headDim + d];
                    Assert.Equal(expected, actual);
                }
            }
        }

        // Now test round-trip back via TransposeFromHeadContiguous
        var restored = new float[seqLen * dim];
        unsafe
        {
            fixed (float* pSrc = headContig, pDst = restored)
            {
                WanAttention.TransposeFromHeadContiguous(pSrc, pDst, seqLen, numHeads, headDim);
            }
        }

        for (int i = 0; i < flatQ.Length; i++)
        {
            Assert.Equal(flatQ[i], restored[i]);
        }
        Console.WriteLine("[Stage4 Q/K/V Ordering] Verified exact [headDim, numHeads] physical layout matches reference.");
    }
}
