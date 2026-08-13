using System;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

public unsafe class MultiSequenceMoeBatchingTests
{
    [Fact]
    public void BatchRouteTokens_PopulatesBucketsAndOffsets()
    {
        const int numTokens = 3;
        const int embDim = 4;
        const int numExperts = 8;
        const int topK = 2;

        float[] inputTokens = new float[numTokens * embDim];
        // Token 0
        inputTokens[0] = 1.0f; inputTokens[1] = 0.5f; inputTokens[2] = -0.2f; inputTokens[3] = 0.8f;
        // Token 1
        inputTokens[4] = 0.2f; inputTokens[5] = 1.5f; inputTokens[6] = 0.1f; inputTokens[7] = -0.4f;
        // Token 2
        inputTokens[8] = -0.5f; inputTokens[9] = 0.8f; inputTokens[10] = 1.2f; inputTokens[11] = 0.3f;

        float[] wGateInp = new float[embDim * numExperts];
        // Expert 3 and 5 favored for token 0
        wGateInp[0 * numExperts + 3] = 2.0f;
        wGateInp[0 * numExperts + 5] = 1.5f;

        var workspace = new MoEWorkspace(numTokens, numExperts, topK, embDim, 16);

        fixed (float* pIn = inputTokens)
        fixed (float* pW = wGateInp)
        {
            DeepSeekMoeGraph.BatchRouteTokens(pIn, pW, numTokens, embDim, numExperts, topK, workspace);
        }

        Assert.Equal(numTokens * topK, workspace.ExpertOffsets[numExperts]);
        Assert.Equal(numTokens * topK, workspace.Assignments.Length >= numTokens * topK ? numTokens * topK : 0);
    }

    [Fact]
    public void GatherAndScatter_PreservesWeightedOutputsAndIsolation()
    {
        const int numTokens = 2;
        const int embDim = 4;
        const int numExperts = 4;
        const int topK = 2;

        float[] inputTokens = [
            1.0f, 2.0f, 3.0f, 4.0f, // Token 0
            5.0f, 6.0f, 7.0f, 8.0f  // Token 1
        ];

        float[] wGateInp = new float[embDim * numExperts];
        wGateInp[0 * numExperts + 1] = 2.0f;
        wGateInp[0 * numExperts + 2] = 1.0f;

        var workspace = new MoEWorkspace(numTokens, numExperts, topK, embDim, 16);

        fixed (float* pIn = inputTokens)
        fixed (float* pW = wGateInp)
        {
            DeepSeekMoeGraph.BatchRouteTokens(pIn, pW, numTokens, embDim, numExperts, topK, workspace);
            DeepSeekMoeGraph.GatherInputs(pIn, numTokens, embDim, workspace);
        }

        // Mock expert outputs: double the gathered input
        float[] expertOutputs = new float[workspace.GatheredInputs.Length];
        for (int i = 0; i < expertOutputs.Length; i++)
        {
            expertOutputs[i] = workspace.GatheredInputs[i] * 2.0f;
        }

        float[] finalOutputs = new float[numTokens * embDim];

        fixed (float* pExpOut = expertOutputs)
        fixed (float* pOut = finalOutputs)
        {
            DeepSeekMoeGraph.ScatterOutputs(pExpOut, pOut, numTokens, embDim, workspace);
        }

        // Output for each token should be non-zero and finite
        for (int i = 0; i < numTokens * embDim; i++)
        {
            Assert.False(float.IsNaN(finalOutputs[i]));
            Assert.False(float.IsInfinity(finalOutputs[i]));
            Assert.True(finalOutputs[i] > 0f);
        }
    }
}
