using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace OpenTail.Stingray.Core.Lora;

/// <summary>
/// Represents a single fine-tuned Low-Rank Adaptation (LoRA) module weight pair (Down/A and Up/B).
/// Computes low-rank delta: Delta Y = Scaling * (X * A) * B
/// </summary>
public sealed class LoraLayer
{
    public string TargetName { get; }
    public int LayerIndex { get; }
    public float[] DownWeight { get; } // A matrix: [Rank, InDim]
    public float[] UpWeight { get; }   // B matrix: [OutDim, Rank]
    public int InDim { get; }
    public int OutDim { get; }
    public int Rank { get; }
    public float Alpha { get; }
    public float Scaling { get; }

    public LoraLayer(
        string targetName,
        int layerIndex,
        float[] downWeight,
        float[] upWeight,
        int inDim,
        int outDim,
        int rank,
        float alpha)
    {
        TargetName = targetName;
        LayerIndex = layerIndex;
        DownWeight = downWeight;
        UpWeight = upWeight;
        InDim = inDim;
        OutDim = outDim;
        Rank = rank;
        Alpha = alpha;
        Scaling = rank > 0 ? alpha / rank : 1.0f;
    }

    /// <summary>
    /// Computes and adds low-rank delta in-place: output[o] += Scaling * (input * A * B)[o]
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
    public void ApplyDelta(ReadOnlySpan<float> input, Span<float> output)
    {
        if (Rank <= 0 || input.Length < InDim || output.Length < OutDim)
            return;

        // Intermediate vector: rankVec = input * A^T  [Rank]
        Span<float> rankVec = stackalloc float[Math.Min(Rank, 256)];
        float[]? rented = null;
        if (Rank > 256)
        {
            rented = System.Buffers.ArrayPool<float>.Shared.Rent(Rank);
            rankVec = rented.AsSpan(0, Rank);
        }

        try
        {
            // 1. rankVec = A * input  where A is [Rank, InDim]
            for (int r = 0; r < Rank; r++)
            {
                int aOff = r * InDim;
                float sum = 0f;
                int i = 0;

                if (Vector.IsHardwareAccelerated && InDim >= Vector<float>.Count)
                {
                    var vSum = Vector<float>.Zero;
                    int vecEnd = InDim - Vector<float>.Count;
                    while (i <= vecEnd)
                    {
                        var vIn = new Vector<float>(input.Slice(i));
                        var vA = new Vector<float>(DownWeight, aOff + i);
                        vSum += vIn * vA;
                        i += Vector<float>.Count;
                    }
                    sum = Vector.Dot(vSum, Vector<float>.One);
                }

                for (; i < InDim; i++)
                    sum += input[i] * DownWeight[aOff + i];

                rankVec[r] = sum;
            }

            // 2. output += Scaling * (B * rankVec) where B is [OutDim, Rank]
            float scale = Scaling;
            for (int o = 0; o < OutDim; o++)
            {
                int bOff = o * Rank;
                float sum = 0f;
                int r = 0;

                if (Vector.IsHardwareAccelerated && Rank >= Vector<float>.Count)
                {
                    var vSum = Vector<float>.Zero;
                    int vecEnd = Rank - Vector<float>.Count;
                    while (r <= vecEnd)
                    {
                        var vR = new Vector<float>(rankVec.Slice(r));
                        var vB = new Vector<float>(UpWeight, bOff + r);
                        vSum += vR * vB;
                        r += Vector<float>.Count;
                    }
                    sum = Vector.Dot(vSum, Vector<float>.One);
                }

                for (; r < Rank; r++)
                    sum += rankVec[r] * UpWeight[bOff + r];

                output[o] += sum * scale;
            }
        }
        finally
        {
            if (rented is not null)
                System.Buffers.ArrayPool<float>.Shared.Return(rented);
        }
    }
}
