
namespace OpenTail.Stingray.Engine;

/// <summary>
/// Multi-head Latent Attention (MLA) for DeepSeek-V2 / DeepSeek-V3 / DeepSeek-R1 architectures.
/// Retains compressed latent KV states (512-dim + 64-dim RoPE key = 576 floats/token) in the
/// KV cache rather than expanding full (16 x 128 x 2 = 4,096) head tensors, saving 3.5x KV memory.
/// </summary>
public static unsafe class MlaAttention
{
    /// <summary>
    /// Computes MLA latent compression and stores the 576-dim compressed representation per token.
    /// </summary>
    public static void CompressKvLatent(
        float* outputCompressed,
        float* input,
        float* wKvAMqa,
        float* kvANorm,
        int batchSize,
        int embDim,
        int kvLoraRank,
        int ropeDim,
        float rmsNormEps)
    {
        int totalKvDim = kvLoraRank + ropeDim; // 512 + 64 = 576

        for (int m = 0; m < batchSize; m++)
        {
            float* inRow = input + (nuint)m * (nuint)embDim;
            float* outRow = outputCompressed + (nuint)m * (nuint)totalKvDim;

            // 1. Matrix multiply input against W_kv_a_mqa [embDim x 576]
            for (int j = 0; j < totalKvDim; j++)
            {
                float sum = 0f;
                float* wCol = wKvAMqa + j;
                for (int k = 0; k < embDim; k++)
                {
                    sum += inRow[k] * wCol[k * totalKvDim];
                }
                outRow[j] = sum;
            }

            // 2. Apply RMSNorm to the first kvLoraRank (512) elements using kvANorm
            float ss = 0f;
            for (int i = 0; i < kvLoraRank; i++)
            {
                float val = outRow[i];
                ss += val * val;
            }
            float invScale = 1.0f / MathF.Sqrt(ss / kvLoraRank + rmsNormEps);

            for (int i = 0; i < kvLoraRank; i++)
            {
                outRow[i] = (outRow[i] * invScale) * kvANorm[i];
            }
        }
    }

    /// <summary>
    /// Uncompresses 512-dim latent KV state into K_noop [16 x 64] and V [16 x 128] states.
    /// </summary>
    public static void UncompressKv(
        float* outKNoop,
        float* outV,
        float* kvLatentNormed,
        float* wKvB,
        int kvLoraRank,
        int numHeads,
        int kNoopDim,
        int vDim)
    {
        int totalOutDim = numHeads * (kNoopDim + vDim); // 16 * (64 + 128) = 3072

        for (int h = 0; h < numHeads; h++)
        {
            float* kHead = outKNoop + h * kNoopDim;
            float* vHead = outV + h * vDim;

            // K_noop uncompress
            for (int i = 0; i < kNoopDim; i++)
            {
                int col = h * (kNoopDim + vDim) + i;
                float sum = 0f;
                for (int k = 0; k < kvLoraRank; k++)
                {
                    sum += kvLatentNormed[k] * wKvB[k * totalOutDim + col];
                }
                kHead[i] = sum;
            }

            // V uncompress
            for (int i = 0; i < vDim; i++)
            {
                int col = h * (kNoopDim + vDim) + kNoopDim + i;
                float sum = 0f;
                for (int k = 0; k < kvLoraRank; k++)
                {
                    sum += kvLatentNormed[k] * wKvB[k * totalOutDim + col];
                }
                vHead[i] = sum;
            }
        }
    }
}
