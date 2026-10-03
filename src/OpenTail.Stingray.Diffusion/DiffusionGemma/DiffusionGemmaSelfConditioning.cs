using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Diffusion.DiffusionGemma;

/// <summary>
/// Self-conditioning implementation for DiffusionGemma.
/// Conditions step t > 1 on soft probability-weighted token embeddings from step t - 1
/// passed through a learned self-conditioning transformation.
/// </summary>
public static unsafe class DiffusionGemmaSelfConditioning
{
    /// <summary>
    /// Computes soft probabilities P = softmax(logits / temperature).
    /// </summary>
    public static void ComputeSoftProbabilities(
        ReadOnlySpan<float> logits, float temperature, Span<float> probsOut)
    {
        int vocabSize = logits.Length;
        float invT = 1.0f / MathF.Max(1e-4f, temperature);

        float maxLogit = float.NegativeInfinity;
        for (int i = 0; i < vocabSize; i++)
        {
            float scaled = logits[i] * invT;
            probsOut[i] = scaled;
            if (scaled > maxLogit) maxLogit = scaled;
        }

        float sumExp = 0f;
        for (int i = 0; i < vocabSize; i++)
        {
            float e = MathF.Exp(probsOut[i] - maxLogit);
            probsOut[i] = e;
            sumExp += e;
        }

        float invSum = 1.0f / sumExp;
        for (int i = 0; i < vocabSize; i++)
        {
            probsOut[i] *= invSum;
        }
    }

    /// <summary>
    /// Projects soft probabilities P against the embedding matrix W_embed and applies the sqrt(D) scale factor.
    /// E_soft = (P * W_embed) * sqrt(hiddenDim).
    /// </summary>
    public static void ComputeSoftEmbedding(
        ReadOnlySpan<float> probs, float* embedWeights, int vocabSize, int hiddenDim,
        float embedScale, Span<float> softEmbedOut)
    {
        softEmbedOut.Clear();

        // embedWeights is [vocabSize, hiddenDim] row-major
        for (int v = 0; v < vocabSize; v++)
        {
            float p = probs[v];
            if (p < 1e-6f) continue; // skip insignificant tail mass for CPU efficiency

            float* row = embedWeights + (long)v * hiddenDim;
            for (int d = 0; d < hiddenDim; d++)
            {
                softEmbedOut[d] += p * row[d];
            }
        }

        for (int d = 0; d < hiddenDim; d++)
        {
            softEmbedOut[d] *= embedScale;
        }
    }

    /// <summary>
    /// Applies the self-conditioning transformation (MLP projection) to E_soft:
    /// E_sc = GeLU(E_soft * W_gate) * (E_soft * W_up) * W_down.
    /// </summary>
    public static void ApplySelfCondMlp(
        ReadOnlySpan<float> softEmbed,
        float* wGate, float* wUp, float* wDown,
        int hiddenDim, int intermediateDim,
        Span<float> scOut)
    {
        Span<float> gate = intermediateDim <= 4096 ? stackalloc float[intermediateDim] : new float[intermediateDim];
        Span<float> up = intermediateDim <= 4096 ? stackalloc float[intermediateDim] : new float[intermediateDim];

        fixed (float* inPtr = softEmbed, gPtr = gate, uPtr = up)
        {
            SimdKernels.MatVecF32(gPtr, wGate, null, inPtr, intermediateDim, hiddenDim);
            SimdKernels.MatVecF32(uPtr, wUp, null, inPtr, intermediateDim, hiddenDim);
        }

        // GELU(gate) * up
        for (int i = 0; i < intermediateDim; i++)
        {
            float x = gate[i];
            // Standard GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            float gelu = 0.5f * x * (1.0f + MathF.Tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
            gate[i] = gelu * up[i];
        }

        fixed (float* inPtr = gate, outPtr = scOut)
        {
            SimdKernels.MatVecF32(outPtr, wDown, null, inPtr, hiddenDim, intermediateDim);
        }
    }

    /// <summary>
    /// Injects self-conditioning vector into the canvas representations:
    /// canvas[pos] += E_sc.
    /// </summary>
    public static void InjectSelfConditioning(Span<float> canvasPos, ReadOnlySpan<float> selfCond)
    {
        for (int i = 0; i < canvasPos.Length; i++)
        {
            canvasPos[i] += selfCond[i];
        }
    }
}
