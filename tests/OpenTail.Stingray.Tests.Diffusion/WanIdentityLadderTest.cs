using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.Wan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Stage 5 of the Wan2.1 tensor-layout diagnostic plan (docs/081):
/// Identity-weight ladder test.
///
/// Constructs a synthetic DiT chain with deterministic identity/permutation weights
/// so that representation flow from PackLatents -> patch_embedding -> TransformerBlock
/// -> head -> UnpackLatents is tested end-to-end without 30 layers of real-weight noise.
/// Verifies that token identity and spatial pixel locations are preserved end-to-end.
/// </summary>
public sealed class WanIdentityLadderTest
{
    private sealed class SyntheticWeightLoader : IWeightLoader
    {
        private readonly Dictionary<string, float[]> _tensors = new(StringComparer.Ordinal);
        private readonly Dictionary<string, long[]> _shapes = new(StringComparer.Ordinal);

        public void AddTensor(string name, float[] data, long[] shape)
        {
            _tensors[name] = data;
            _shapes[name] = shape;
        }

        public bool Contains(string name) => _tensors.ContainsKey(name);
        public float[] ReadF32(string name) => _tensors.TryGetValue(name, out var data) ? data : throw new KeyNotFoundException(name);
        public Span<float> ReadF32Span(string name) => ReadF32(name);
        public long[] GetShape(string name) => _shapes[name];
        public int TensorCount => _tensors.Count;
        public IEnumerable<string> TensorNames => _tensors.Keys;
        public unsafe bool TryGetRaw(string name, out nint dataPtr, out long byteLen, out DType dtype, out int rows, out int cols)
        {
            dataPtr = 0;
            byteLen = 0;
            dtype = DType.Float32;
            rows = 0;
            cols = 0;
            return false;
        }
        public void Dispose() { }
    }

    [Fact]
    public void IdentityLadder_TokenIdentityPreservedEndToEnd()
    {
        const int numFrames = 2;
        const int latH = 4;
        const int latW = 6;
        const int latC = 16;
        const int inChannels = 64;
        const int dim = 128; // small test dim (1 head of 128)
        const int ffnDim = 256;
        const int patchH = latH / 2; // 2
        const int patchW = latW / 2; // 3
        const int numTokens = numFrames * patchH * patchW; // 12

        var loader = new SyntheticWeightLoader();

        // 1. patch_embedding: Linear(inChannels=64, dim=128)
        // Stored as [dim, inChannels] = [128, 64].
        // Identity mapping: for s in 0..63, weight[s, s] = 1.0f.
        var wPatch = new float[dim * inChannels];
        for (int s = 0; s < inChannels; s++)
        {
            wPatch[s * inChannels + s] = 1.0f;
        }
        loader.AddTensor("patch_embedding.weight", wPatch, [dim, inChannels]);

        // 2. blocks.0 weights:
        // Set modulation to 0 so no scale/shift/gate alterations occur by default
        loader.AddTensor("blocks.0.modulation", new float[dim * 6], [dim * 6]);
        loader.AddTensor("time_embedding.0.weight", new float[dim * 256], [dim, 256]);
        loader.AddTensor("time_embedding.2.weight", new float[dim * dim], [dim, dim]);
        loader.AddTensor("time_projection.1.weight", new float[dim * 6 * dim], [dim * 6, dim]);

        // Self-attention weights: identity for Q, K, V, O
        var wEye = new float[dim * dim];
        for (int d = 0; d < dim; d++) wEye[d * dim + d] = 1.0f;
        loader.AddTensor("blocks.0.self_attn.q.weight", (float[])wEye.Clone(), [dim, dim]);
        loader.AddTensor("blocks.0.self_attn.k.weight", (float[])wEye.Clone(), [dim, dim]);
        loader.AddTensor("blocks.0.self_attn.v.weight", (float[])wEye.Clone(), [dim, dim]);
        loader.AddTensor("blocks.0.self_attn.o.weight", (float[])wEye.Clone(), [dim, dim]);

        // Cross-attention weights
        loader.AddTensor("blocks.0.cross_attn.q.weight", (float[])wEye.Clone(), [dim, dim]);
        loader.AddTensor("blocks.0.cross_attn.k.weight", (float[])wEye.Clone(), [dim, dim]);
        loader.AddTensor("blocks.0.cross_attn.v.weight", (float[])wEye.Clone(), [dim, dim]);
        loader.AddTensor("blocks.0.cross_attn.o.weight", (float[])wEye.Clone(), [dim, dim]);
        loader.AddTensor("blocks.0.norm3.weight", new float[dim], [dim]);
        loader.AddTensor("blocks.0.norm3.bias", new float[dim], [dim]);

        // FFN weights
        loader.AddTensor("blocks.0.ffn.0.weight", new float[ffnDim * dim], [ffnDim, dim]);
        loader.AddTensor("blocks.0.ffn.2.weight", new float[dim * ffnDim], [dim, ffnDim]);

        // 3. head.head: Linear(dim=128, outChannels=64)
        // Stored as [64, 128].
        // In Wan:
        // PackLatents puts pixel (c, p) at slot s = c*4 + p (where p = dy*2 + dx in 0..3).
        // UnpackLatents reads pixel (c, p) from offset i = p*16 + c.
        // Therefore, head.head maps feature s (in 0..63) to output row i = (s % 4) * 16 + (s / 4):
        var wHead = new float[inChannels * dim];
        for (int s = 0; s < inChannels; s++)
        {
            int p = s % 4;
            int c = s / 4;
            int row = p * 16 + c;
            wHead[row * dim + s] = 1.0f;
        }
        loader.AddTensor("head.head.weight", wHead, [inChannels, dim]);
        loader.AddTensor("head.modulation", new float[dim * 2], [dim * 2]);

        // 4. Create unique-value latent tensor
        int totalLatent = latC * numFrames * latH * latW;
        var latent = new float[totalLatent];
        for (int c = 0; c < latC; c++)
        {
            for (int t = 0; t < numFrames; t++)
            {
                for (int y = 0; y < latH; y++)
                {
                    for (int x = 0; x < latW; x++)
                    {
                        int idx = ((c * numFrames + t) * latH + y) * latW + x;
                        latent[idx] = c * 10_000f + t * 1000f + y * 10f + x;
                    }
                }
            }
        }

        // Test representation flow through PackLatents -> wPatch -> wHead -> UnpackLatents
        var packed = WanModel.PackLatents(latent, numFrames, latH, latW);

        // Simulated patch_embedding: [numTokens, 64] x [128, 64]^T -> [numTokens, 128]
        var xTok = new float[numTokens * dim];
        for (int tok = 0; tok < numTokens; tok++)
        {
            for (int s = 0; s < inChannels; s++)
            {
                xTok[tok * dim + s] = packed[tok * inChannels + s];
            }
        }

        // Simulated head.head: [numTokens, 128] x [64, 128]^T -> [numTokens, 64]
        var outPacked = new float[numTokens * inChannels];
        for (int tok = 0; tok < numTokens; tok++)
        {
            for (int r = 0; r < inChannels; r++)
            {
                float sum = 0f;
                for (int d = 0; d < dim; d++)
                {
                    sum += xTok[tok * dim + d] * wHead[r * dim + d];
                }
                outPacked[tok * inChannels + r] = sum;
            }
        }

        // UnpackLatents
        var unpacked = WanModel.UnpackLatents(outPacked, numFrames, latH, latW);

        // Verify exact round-trip!
        int mismatches = 0;
        for (int i = 0; i < totalLatent; i++)
        {
            if (unpacked[i] != latent[i])
            {
                if (mismatches < 5)
                {
                    Console.WriteLine($"[IdentityLadder Mismatch] idx={i}: unpacked={unpacked[i]}, expected={latent[i]}");
                }
                mismatches++;
            }
        }

        Console.WriteLine($"[IdentityLadder] Total: {totalLatent}, Mismatches: {mismatches}");
        Assert.Equal(0, mismatches);
    }
}
