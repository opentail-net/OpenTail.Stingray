using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Cpu;

/// <summary>
/// DeepSeekMoE executor for DeepSeek-V2 / DeepSeek-V3 architecture layers.
/// Combines 2 always-active shared experts with top-k routed expert gating across single or multi-sequence batches.
/// </summary>
public static unsafe class DeepSeekMoeGraph
{
    /// <summary>
    /// Computes top-k expert routing scores for DeepSeekMoE router on a single token.
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
            float* col = wGateInp + e;
            for (int k = 0; k < embDim; k++)
            {
                sum += inputToken[k] * col[k * numExperts];
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

    /// <summary>
    /// Routes a batch of N tokens to top-k experts and populates expert assignment buckets in MoEWorkspace.
    /// </summary>
    public static void BatchRouteTokens(
        float* inputTokens,
        float* wGateInp,
        int numTokens,
        int embDim,
        int numExperts,
        int topK,
        MoEWorkspace workspace)
    {
        workspace.EnsureCapacity(numTokens, numExperts, topK, embDim, workspace.ExpertIntermediateDim);
        workspace.Reset();

        // 1. Route each token and populate expert counts
        for (int t = 0; t < numTokens; t++)
        {
            float* tokenPtr = inputTokens + (nuint)t * (nuint)embDim;
            int offset = t * topK;

            fixed (int* pIdx = &workspace.TokenIndices[offset])
            fixed (float* pWt = &workspace.RoutingWeights[offset])
            {
                SelectTopKExperts(tokenPtr, wGateInp, embDim, numExperts, topK, pIdx, pWt);
            }

            for (int k = 0; k < topK; k++)
            {
                int expertId = workspace.TokenIndices[offset + k];
                workspace.ExpertCounts[expertId]++;
            }
        }

        // 2. Compute prefix sum offsets
        int runningOffset = 0;
        for (int e = 0; e < numExperts; e++)
        {
            workspace.ExpertOffsets[e] = runningOffset;
            runningOffset += workspace.ExpertCounts[e];
        }
        workspace.ExpertOffsets[numExperts] = runningOffset;

        // 3. Build ExpertTokenAssignment array
        Span<int> currentOffsets = stackalloc int[numExperts];
        for (int e = 0; e < numExperts; e++)
        {
            currentOffsets[e] = workspace.ExpertOffsets[e];
        }

        for (int t = 0; t < numTokens; t++)
        {
            int offset = t * topK;
            for (int k = 0; k < topK; k++)
            {
                int expertId = workspace.TokenIndices[offset + k];
                float weight = workspace.RoutingWeights[offset + k];
                int slot = currentOffsets[expertId]++;

                workspace.Assignments[slot] = new ExpertTokenAssignment(
                    SequenceId: t,
                    TokenIndex: t,
                    ExpertId: expertId,
                    Weight: weight);
            }
        }
    }

    /// <summary>
    /// Gathers input tokens into contiguous expert matrices.
    /// </summary>
    public static void GatherInputs(
        float* inputTokens,
        int numTokens,
        int embDim,
        MoEWorkspace workspace)
    {
        int totalAssignments = workspace.ExpertOffsets[workspace.MaxExperts];
        for (int i = 0; i < totalAssignments; i++)
        {
            ref readonly var assignment = ref workspace.Assignments[i];
            float* srcToken = inputTokens + (nuint)assignment.TokenIndex * (nuint)embDim;

            fixed (float* pDst = &workspace.GatheredInputs[i * embDim])
            {
                Buffer.MemoryCopy(srcToken, pDst, embDim * sizeof(float), embDim * sizeof(float));
            }
        }
    }

    /// <summary>
    /// Scatters expert output vectors back into their originating token output slots with weighted accumulation.
    /// </summary>
    public static void ScatterOutputs(
        float* expertOutputs,
        float* outputTokens,
        int numTokens,
        int embDim,
        MoEWorkspace workspace)
    {
        int totalAssignments = workspace.ExpertOffsets[workspace.MaxExperts];
        for (int i = 0; i < totalAssignments; i++)
        {
            ref readonly var assignment = ref workspace.Assignments[i];
            float* expOutPtr = expertOutputs + (nuint)i * (nuint)embDim;
            float* tokenOutPtr = outputTokens + (nuint)assignment.TokenIndex * (nuint)embDim;
            float weight = assignment.Weight;

            for (int d = 0; d < embDim; d++)
            {
                tokenOutPtr[d] += expOutPtr[d] * weight;
            }
        }
    }
}
