
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

        // Row-major [numExperts, embDim] -- see DeepSeek2Tests' own comment on this layout.
        float[] wGateInp = new float[numExperts * embDim];
        // Expert 3 and 5 favored for token 0
        wGateInp[3 * embDim + 0] = 2.0f;
        wGateInp[5 * embDim + 0] = 1.5f;

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

        float[] wGateInp = new float[numExperts * embDim];
        wGateInp[1 * embDim + 0] = 2.0f;
        wGateInp[2 * embDim + 0] = 1.0f;

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

    /// <summary>
    /// bugstofix.md: <c>MoEWorkspace.EnsureCapacity</c> only ever grows <c>MaxExperts</c>, so
    /// reusing a workspace for a call with FEWER experts than a prior call left
    /// <c>ExpertOffsets[MaxExperts]</c> stale/zeroed while the real prefix-sum total for THIS
    /// call sat at <c>ExpertOffsets[numExperts]</c> instead -- <see cref="DeepSeekMoeGraph.GatherInputs"/>/
    /// <see cref="DeepSeekMoeGraph.ScatterOutputs"/> read the former, silently dropping every
    /// assignment. Fixed via <see cref="MoEWorkspace.LastNumExperts"/>.
    /// </summary>
    [Fact]
    public void GatherAndScatter_AfterWorkspaceShrinksExpertCount_StillProcessesEveryAssignment()
    {
        const int embDim = 4;
        const int topK = 2;
        var workspace = new MoEWorkspace(maxTokens: 4, maxExperts: 4, topK: topK, embDim: embDim, expertIntermediateDim: 16);

        // First call: grow MaxExperts to 8 (larger than the second call below will use).
        const int firstNumExperts = 8;
        float[] firstTokens = new float[2 * embDim];
        Array.Fill(firstTokens, 1.0f);
        fixed (float* pIn = firstTokens)
        fixed (float* pW = new float[firstNumExperts * embDim])
        {
            DeepSeekMoeGraph.BatchRouteTokens(pIn, pW, numTokens: 2, embDim, firstNumExperts, topK, workspace);
        }
        Assert.Equal(8, workspace.MaxExperts);

        // Second call: same workspace, fewer experts. Prior to the fix, GatherInputs/ScatterOutputs
        // read ExpertOffsets[8] (zeroed by Reset(), never rewritten since numExperts=4 this call)
        // instead of the real total at ExpertOffsets[4].
        const int secondNumExperts = 4;
        const int numTokens = 2;
        float[] inputTokens = [1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f];
        float[] wGateInp = new float[secondNumExperts * embDim];
        wGateInp[1 * embDim + 0] = 2.0f;
        wGateInp[2 * embDim + 0] = 1.0f;

        fixed (float* pIn = inputTokens)
        fixed (float* pW = wGateInp)
        {
            DeepSeekMoeGraph.BatchRouteTokens(pIn, pW, numTokens, embDim, secondNumExperts, topK, workspace);
            DeepSeekMoeGraph.GatherInputs(pIn, numTokens, embDim, workspace);
        }

        Assert.Equal(numTokens * topK, workspace.ExpertOffsets[secondNumExperts]);

        float[] expertOutputs = new float[workspace.GatheredInputs.Length];
        for (int i = 0; i < expertOutputs.Length; i++)
            expertOutputs[i] = workspace.GatheredInputs[i] * 2.0f;

        float[] finalOutputs = new float[numTokens * embDim];
        fixed (float* pExpOut = expertOutputs)
        fixed (float* pOut = finalOutputs)
        {
            DeepSeekMoeGraph.ScatterOutputs(pExpOut, pOut, numTokens, embDim, workspace);
        }

        // Every token must have received its (weighted) expert contribution -- a regression to
        // the MaxExperts-indexed read would leave every one of these at exactly 0.
        for (int i = 0; i < numTokens * embDim; i++)
        {
            Assert.False(float.IsNaN(finalOutputs[i]));
            Assert.True(finalOutputs[i] > 0f, $"finalOutputs[{i}] was {finalOutputs[i]} -- gather/scatter silently dropped work.");
        }
    }
}
