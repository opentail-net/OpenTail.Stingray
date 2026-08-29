
namespace OpenTail.Stingray.Core.Embeddings;

/// <summary>
/// Vector normalization, Matryoshka dimension truncation, pooling, and similarity operations for embeddings.
/// </summary>
public static class EmbeddingNormalizer
{
    /// <summary>
    /// Applies in-place Euclidean L2 unit normalization: v = v / ||v||_2.
    /// </summary>
    public static void NormalizeL2(Span<float> vector)
    {
        if (vector.IsEmpty) return;

        float normSq = Dot(vector, vector);
        if (normSq <= 1e-12f) return;

        float invNorm = 1.0f / MathF.Sqrt(normSq);
        MultiplyScalar(vector, invNorm);
    }

    /// <summary>
    /// Truncates embedding to target Matryoshka dimensions and applies L2 re-normalization.
    /// </summary>
    public static float[] TruncateAndNormalize(ReadOnlySpan<float> vector, int targetDim)
    {
        int clampedDim = Math.Clamp(targetDim, 1, vector.Length);
        float[] result = new float[clampedDim];
        vector[..clampedDim].CopyTo(result);
        NormalizeL2(result);
        return result;
    }

    /// <summary>
    /// Computes cosine similarity between two embedding vectors.
    /// Returns value in range [-1.0, 1.0].
    /// </summary>
    public static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int len = Math.Min(a.Length, b.Length);
        if (len == 0) return 0f;

        float dot = Dot(a[..len], b[..len]);
        float normA = Dot(a[..len], a[..len]);
        float normB = Dot(b[..len], b[..len]);

        float denom = MathF.Sqrt(normA * normB);
        if (denom <= 1e-12f) return 0f;

        return Math.Clamp(dot / denom, -1.0f, 1.0f);
    }

    /// <summary>
    /// Applies sequence-level pooling to token hidden states [seqLen, dModel].
    /// </summary>
    public static float[] ApplyPooling(
        ReadOnlySpan<float> hiddenStates,
        int seqLen,
        int dModel,
        PoolingType pooling)
    {
        if (seqLen <= 0 || dModel <= 0) return [];

        float[] output = new float[dModel];

        switch (pooling)
        {
            case PoolingType.Cls:
                // First token (index 0)
                hiddenStates[..dModel].CopyTo(output);
                break;

            case PoolingType.LastToken:
                // Last token (index seqLen - 1)
                int lastOffset = (seqLen - 1) * dModel;
                hiddenStates.Slice(lastOffset, dModel).CopyTo(output);
                break;

            case PoolingType.Mean:
            default:
                // Average all token representations
                for (int t = 0; t < seqLen; t++)
                {
                    int offset = t * dModel;
                    var tokenSpan = hiddenStates.Slice(offset, dModel);
                    AddInPlace(output, tokenSpan);
                }
                float invLen = 1.0f / seqLen;
                MultiplyScalar(output, invLen);
                break;
        }

        return output;
    }

    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int len = Math.Min(a.Length, b.Length);
        float sum = 0f;
        int i = 0;

        if (Vector.IsHardwareAccelerated && len >= Vector<float>.Count)
        {
            var vSum = Vector<float>.Zero;
            int vecLen = len - (len % Vector<float>.Count);

            for (; i < vecLen; i += Vector<float>.Count)
            {
                var va = new Vector<float>(a.Slice(i, Vector<float>.Count));
                var vb = new Vector<float>(b.Slice(i, Vector<float>.Count));
                vSum += va * vb;
            }

            sum += Vector.Sum(vSum);
        }

        for (; i < len; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }

    private static void MultiplyScalar(Span<float> target, float scalar)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated && target.Length >= Vector<float>.Count)
        {
            var vScalar = new Vector<float>(scalar);
            int vecLen = target.Length - (target.Length % Vector<float>.Count);

            for (; i < vecLen; i += Vector<float>.Count)
            {
                var v = new Vector<float>(target.Slice(i, Vector<float>.Count));
                (v * vScalar).CopyTo(target.Slice(i, Vector<float>.Count));
            }
        }

        for (; i < target.Length; i++)
        {
            target[i] *= scalar;
        }
    }

    private static void AddInPlace(Span<float> dest, ReadOnlySpan<float> src)
    {
        int len = Math.Min(dest.Length, src.Length);
        int i = 0;

        if (Vector.IsHardwareAccelerated && len >= Vector<float>.Count)
        {
            int vecLen = len - (len % Vector<float>.Count);
            for (; i < vecLen; i += Vector<float>.Count)
            {
                var vd = new Vector<float>(dest.Slice(i, Vector<float>.Count));
                var vs = new Vector<float>(src.Slice(i, Vector<float>.Count));
                (vd + vs).CopyTo(dest.Slice(i, Vector<float>.Count));
            }
        }

        for (; i < len; i++)
        {
            dest[i] += src[i];
        }
    }
}
