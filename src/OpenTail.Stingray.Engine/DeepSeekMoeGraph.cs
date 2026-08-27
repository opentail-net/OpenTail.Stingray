using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// DeepSeekMoE executor for DeepSeek-V2 / DeepSeek-V3 architecture layers.
/// Combines 2 always-active shared experts with top-k routed expert gating.
/// </summary>
public static unsafe class DeepSeekMoeGraph
{
    /// <summary>
    /// Computes top-k expert routing scores for DeepSeekMoE router.
    /// </summary>
    public static void SelectTopKExperts(
        float* inputToken,
        float* wGateInp,
        int embDim,
        int numExperts,
        int topK,
        int* outExpertIndices,
        float* outExpertWeights)
    {
        // 1. Router logits: S = inputToken * wGateInp [embDim x numExperts]
        Span<float> logits = stackalloc float[numExperts];
        float maxLogit = -float.MaxValue;

        for (int e = 0; e < numExperts; e++)
        {
            float sum = 0f;
            // wGateInp is row-major [numExperts, embDim] (the standard GGUF Linear-weight layout
            // -- see the identical fix and rationale on OpenTail.Stingray.Cpu.DeepSeekMoeGraph,
            // this class's twin; this one is also unreachable dead code, kept consistent anyway).
            float* row = wGateInp + (long)e * embDim;
            for (int k = 0; k < embDim; k++)
            {
                sum += inputToken[k] * row[k];
            }
            logits[e] = sum;
            if (sum > maxLogit) maxLogit = sum;
        }

        // 2. Softmax
        Span<float> probs = stackalloc float[numExperts];
        float sumExp = 0f;
        for (int e = 0; e < numExperts; e++)
        {
            float p = MathF.Exp(logits[e] - maxLogit);
            probs[e] = p;
            sumExp += p;
        }
        float invSumExp = 1.0f / sumExp;
        for (int e = 0; e < numExperts; e++)
        {
            probs[e] *= invSumExp;
        }

        // 3. Select Top K experts
        Span<bool> chosen = stackalloc bool[numExperts];
        float weightSum = 0f;

        for (int k = 0; k < topK; k++)
        {
            int bestIdx = -1;
            float bestProb = -1f;

            for (int e = 0; e < numExperts; e++)
            {
                if (!chosen[e] && probs[e] > bestProb)
                {
                    bestProb = probs[e];
                    bestIdx = e;
                }
            }

            if (bestIdx >= 0)
            {
                chosen[bestIdx] = true;
                outExpertIndices[k] = bestIdx;
                outExpertWeights[k] = bestProb;
                weightSum += bestProb;
            }
        }

        // 4. Renormalize top-k weights to sum to 1.0
        if (weightSum > 0f)
        {
            float normFactor = 1.0f / weightSum;
            for (int k = 0; k < topK; k++)
            {
                outExpertWeights[k] *= normFactor;
            }
        }
    }
}
