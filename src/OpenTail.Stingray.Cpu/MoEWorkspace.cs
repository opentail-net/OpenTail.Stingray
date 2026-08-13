using System;
using System.Collections.Generic;

namespace OpenTail.Stingray.Cpu;

/// <summary>
/// Immutable record struct capturing the assignment of a token to a specific expert.
/// </summary>
public readonly record struct ExpertTokenAssignment(
    int SequenceId,
    int TokenIndex,
    int ExpertId,
    float Weight);

/// <summary>
/// Preallocated, zero-allocation workspace for multi-sequence MoE routing, expert bucketing, gather, and scatter.
/// Reused across generation steps to eliminate GC allocations in the hot inference path.
/// </summary>
public sealed class MoEWorkspace
{
    public int MaxTokens { get; private set; }
    public int MaxExperts { get; private set; }
    public int TopK { get; private set; }
    public int EmbDim { get; private set; }
    public int ExpertIntermediateDim { get; private set; }

    public int[] ExpertCounts { get; private set; } = Array.Empty<int>();
    public int[] ExpertOffsets { get; private set; } = Array.Empty<int>();
    public int[] TokenIndices { get; private set; } = Array.Empty<int>();
    public float[] RoutingWeights { get; private set; } = Array.Empty<float>();
    public ExpertTokenAssignment[] Assignments { get; private set; } = Array.Empty<ExpertTokenAssignment>();

    public float[] GatheredInputs { get; private set; } = Array.Empty<float>();
    public float[] ScatteredOutputs { get; private set; } = Array.Empty<float>();

    public MoEWorkspace(int maxTokens = 64, int maxExperts = 64, int topK = 6, int embDim = 2048, int expertIntermediateDim = 1408)
    {
        EnsureCapacity(maxTokens, maxExperts, topK, embDim, expertIntermediateDim);
    }

    /// <summary>
    /// Ensures workspace buffers are sufficiently sized for the current batch token count and expert dimensions.
    /// </summary>
    public void EnsureCapacity(int numTokens, int numExperts, int topK, int embDim, int expertIntermediateDim)
    {
        if (numTokens <= MaxTokens && numExperts <= MaxExperts && topK <= TopK && embDim <= EmbDim && expertIntermediateDim <= ExpertIntermediateDim && ExpertCounts.Length > 0)
        {
            return;
        }

        MaxTokens = Math.Max(MaxTokens, numTokens);
        MaxExperts = Math.Max(MaxExperts, numExperts);
        TopK = Math.Max(TopK, topK);
        EmbDim = Math.Max(EmbDim, embDim);
        ExpertIntermediateDim = Math.Max(ExpertIntermediateDim, expertIntermediateDim);

        int totalAssignments = MaxTokens * TopK;

        ExpertCounts = new int[MaxExperts];
        ExpertOffsets = new int[MaxExperts + 1];
        TokenIndices = new int[totalAssignments];
        RoutingWeights = new float[totalAssignments];
        Assignments = new ExpertTokenAssignment[totalAssignments];

        GatheredInputs = new float[totalAssignments * EmbDim];
        ScatteredOutputs = new float[MaxTokens * EmbDim];
    }

    /// <summary>
    /// Resets workspace count and offset arrays for a new generation step.
    /// </summary>
    public void Reset()
    {
        Array.Clear(ExpertCounts, 0, ExpertCounts.Length);
        Array.Clear(ExpertOffsets, 0, ExpertOffsets.Length);
        Array.Clear(ScatteredOutputs, 0, ScatteredOutputs.Length);
    }
}
