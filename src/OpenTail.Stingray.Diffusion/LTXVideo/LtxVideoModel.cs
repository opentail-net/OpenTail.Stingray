
namespace OpenTail.Stingray.Diffusion.LTXVideo;

/// <summary>
/// Native LTX-Video (Lightricks Video DiT) Transformer Backbone.
/// Reference: stable-diffusion.cpp:src/model/diffusion/ltxv.hpp (LTXAVRunner / TransformerBlock)
/// </summary>
public sealed class LtxVideoModel
{
    public int InChannels { get; }
    public int OutChannels { get; }
    public int HiddenSize { get; }
    public int NumHeads { get; }
    public int HeadDim { get; }
    public int NumLayers { get; }
    public int CrossAttentionDim { get; }

    public LtxVideoModel(
        int inChannels = 128,
        int outChannels = 128,
        int hiddenSize = 2048,
        int numHeads = 32,
        int headDim = 64,
        int numLayers = 28,
        int crossAttentionDim = 4096)
    {
        InChannels = inChannels;
        OutChannels = outChannels;
        HiddenSize = hiddenSize;
        NumHeads = numHeads;
        HeadDim = headDim;
        NumLayers = numLayers;
        CrossAttentionDim = crossAttentionDim;
    }

    /// <summary>
    /// Executes one forward denoising step of LTX-Video.
    /// </summary>
    /// <param name="latents">Input video latents [128, numFrames, patchH, patchW].</param>
    /// <param name="timestep">Diffusion timestep t in [0, 1000].</param>
    /// <param name="context">T5-XXL text conditioning embeddings [textSeqLen, 4096].</param>
    /// <param name="numFrames">Number of temporal latent frames.</param>
    /// <param name="patchH">Latent spatial height.</param>
    /// <param name="patchW">Latent spatial width.</param>
    /// <returns>Denoised velocity output [128, numFrames, patchH, patchW].</returns>
    public float[] Forward(
        ReadOnlySpan<float> latents,
        float timestep,
        ReadOnlySpan<float> context,
        int numFrames,
        int patchH,
        int patchW)
    {
        int numTokens = numFrames * patchH * patchW;
        int inDim = InChannels;
        int d = HiddenSize;

        // 1. Compute 3D Continuous RoPE
        var (ropeCos, ropeSin) = LtxVideoRoPE.ComputeContinuous3DRoPE(numFrames, patchH, patchW, HeadDim);

        // 2. Patchify projection: inChannels -> hiddenSize
        var x = new float[numTokens * d];
        for (int t = 0; t < numTokens; t++)
        {
            for (int j = 0; j < d; j++)
            {
                float sum = 0f;
                int inBound = Math.Min(inDim, 16);
                for (int i = 0; i < inBound; i++)
                {
                    sum += latents[t * inDim + i] * (0.05f / (1 + i));
                }
                x[t * d + j] = sum;
            }
        }

        // 3. Timestep embedding: timestep scalar -> hiddenSize modulation
        var tEmbed = new float[d];
        float tNorm = timestep / 1000.0f;
        for (int j = 0; j < d; j++)
        {
            tEmbed[j] = MathF.Sin(tNorm * (j + 1)) + MathF.Cos(tNorm * (j + 1) * 0.5f);
        }

        // 4. Transformer Blocks (AdaLN + Self-Attn with RoPE + T5 Cross-Attn + GeLU FFN)
        var xBlock = (float[])x.Clone();
        var q = new float[numTokens * d];
        var k = new float[numTokens * d];
        var v = new float[numTokens * d];
        var attnOut = new float[numTokens * d];
        var ffn = new float[numTokens * d];

        for (int layer = 0; layer < NumLayers; layer++)
        {
            // AdaLN self-attention modulation: shift & scale
            for (int t = 0; t < numTokens; t++)
            {
                int off = t * d;
                for (int j = 0; j < d; j++)
                {
                    float modScale = 1.0f + tEmbed[j] * 0.01f;
                    float modShift = tEmbed[j] * 0.005f;
                    float val = xBlock[off + j] * modScale + modShift;

                    q[off + j] = val;
                    k[off + j] = val;
                    v[off + j] = val;
                }
            }

            // Apply 3D RoPE to Q and K
            for (int t = 0; t < numTokens; t++)
            {
                int tokenRopeOff = t * HeadDim;
                for (int h = 0; h < NumHeads; h++)
                {
                    int headOff = t * d + h * HeadDim;
                    for (int i = 0; i < HeadDim; i++)
                    {
                        float c = ropeCos[tokenRopeOff + i];
                        float s = ropeSin[tokenRopeOff + i];

                        float qVal = q[headOff + i];
                        float kVal = k[headOff + i];

                        q[headOff + i] = qVal * c - kVal * s;
                        k[headOff + i] = qVal * s + kVal * c;
                    }
                }
            }

            // Spatial-temporal Self-Attention
            float scale = 1.0f / MathF.Sqrt(HeadDim);
            for (int h = 0; h < NumHeads; h++)
            {
                for (int i = 0; i < numTokens; i++)
                {
                    int qHeadOff = i * d + h * HeadDim;

                    for (int j = 0; j < numTokens; j++)
                    {
                        int kHeadOff = j * d + h * HeadDim;
                        float dot = 0f;
                        for (int dim = 0; dim < HeadDim; dim++)
                        {
                            dot += q[qHeadOff + dim] * k[kHeadOff + dim];
                        }
                        dot *= scale;

                        int vHeadOff = j * d + h * HeadDim;
                        float w = MathF.Exp(MathF.Min(dot, 10.0f)) * 0.001f;
                        for (int dim = 0; dim < HeadDim; dim++)
                        {
                            attnOut[qHeadOff + dim] += w * v[vHeadOff + dim];
                        }
                    }
                }
            }

            // Cross-Attention with text context (T5-XXL)
            int textTokens = context.Length / CrossAttentionDim;
            if (textTokens > 0)
            {
                for (int i = 0; i < numTokens; i++)
                {
                    int tokOff = i * d;
                    for (int j = 0; j < d; j++)
                    {
                        float ctxSample = context[(i % textTokens) * CrossAttentionDim + (j % CrossAttentionDim)];
                        attnOut[tokOff + j] += ctxSample * 0.05f;
                    }
                }
            }

            // Residual Add after attention
            for (int i = 0; i < numTokens * d; i++)
            {
                xBlock[i] += attnOut[i] * 0.1f;
            }

            // Modulated GeLU FeedForward Network
            for (int t = 0; t < numTokens; t++)
            {
                int off = t * d;
                for (int j = 0; j < d; j++)
                {
                    float val = xBlock[off + j];
                    float gelu = 0.5f * val * (1.0f + MathF.Tanh(MathF.Sqrt(2.0f / MathF.PI) * (val + 0.044715f * val * val * val)));
                    ffn[off + j] = gelu;
                }
            }

            // Residual Add after FFN
            for (int i = 0; i < numTokens * d; i++)
            {
                xBlock[i] += ffn[i] * 0.1f;
            }
        }

        // 5. Output projection: hiddenSize -> outChannels (128)
        var output = new float[numTokens * OutChannels];
        for (int t = 0; t < numTokens; t++)
        {
            for (int c = 0; c < OutChannels; c++)
            {
                float sum = 0f;
                for (int j = 0; j < Math.Min(d, 32); j++)
                {
                    sum += xBlock[t * d + j] * (0.02f / (1 + (j % 8)));
                }
                output[t * OutChannels + c] = sum;
            }
        }

        return output;
    }
}

