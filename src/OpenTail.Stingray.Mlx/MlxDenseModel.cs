namespace OpenTail.Stingray.Mlx;

/// <summary>
/// A full dense-llama model: embedding + N <see cref="MlxDenseLayer"/>s + output norm + output head.
/// The embedding lookup and logits projection are plain host-side array operations, like
/// <see cref="MlxKvCache"/> -- no mlx-c call, nothing to fail predictably, so (like the cache) these
/// are fully testable for real rather than just "fails predictably." Never run on real hardware; see
/// docs/4-performance/gpu/2026-11-06-mlx-metal-backend-plan.md.
/// </summary>
public sealed class MlxDenseModel
{
    /// <summary>Row-major [vocabSize, hiddenDim].</summary>
    public required float[] TokenEmbedding { get; init; }
    public required MlxDenseLayer[] Layers { get; init; }
    public required float[] OutputNormWeight { get; init; }
    /// <summary>Row-major [hiddenDim, vocabSize].</summary>
    public required float[] OutputWeight { get; init; }
    public required int HiddenDim { get; init; }
    public required int VocabSize { get; init; }
    public float Eps { get; init; } = 1e-5f;

    private float[] Embed(ReadOnlySpan<int> tokenIds)
    {
        var hidden = new float[tokenIds.Length * HiddenDim];
        for (int i = 0; i < tokenIds.Length; i++)
        {
            int token = tokenIds[i];
            if ((uint)token >= (uint)VocabSize) throw new ArgumentOutOfRangeException(nameof(tokenIds), token, $"Token id must be in [0, {VocabSize}).");
            TokenEmbedding.AsSpan(token * HiddenDim, HiddenDim).CopyTo(hidden.AsSpan(i * HiddenDim, HiddenDim));
        }
        return hidden;
    }

    /// <summary>
    /// Runs every layer's <see cref="MlxDenseLayer.Forward"/> over a fresh prompt and returns logits
    /// for every position, row-major [seqLen, VocabSize] (not just the last position -- a caller that
    /// only wants the next-token distribution reads the last <see cref="VocabSize"/> floats).
    /// </summary>
    public float[] Prefill(ReadOnlySpan<int> tokenIds)
    {
        int seqLen = tokenIds.Length;
        float[] hidden = Embed(tokenIds);
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) positions[i] = i;

        foreach (var layer in Layers)
            hidden = layer.Forward(hidden, seqLen, positions);

        return FinalLogits(hidden, seqLen);
    }

    /// <summary>
    /// One incremental decode step via <see cref="MlxDenseLayer.ForwardDecode"/>. <paramref name="caches"/>
    /// has one entry per layer (same order as <see cref="Layers"/>), each already containing every
    /// earlier position's K/V.
    /// </summary>
    public float[] Decode(int token, int position, MlxKvCache[] caches)
    {
        if (caches.Length != Layers.Length)
            throw new ArgumentException($"caches has {caches.Length} entries, expected one per layer ({Layers.Length}).", nameof(caches));

        float[] hidden = Embed([token]);
        for (int i = 0; i < Layers.Length; i++)
            hidden = Layers[i].ForwardDecode(hidden, position, caches[i]);

        return FinalLogits(hidden, 1);
    }

    private float[] FinalLogits(ReadOnlySpan<float> hidden, int seqLen)
    {
        float[] normed = MlxOps.RmsNorm(hidden, seqLen, HiddenDim, OutputNormWeight, Eps);
        return MlxOps.MatMul(normed, seqLen, HiddenDim, OutputWeight, VocabSize);
    }
}
