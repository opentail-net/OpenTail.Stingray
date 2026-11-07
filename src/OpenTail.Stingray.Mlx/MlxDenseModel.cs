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

    internal Func<int, float[], int, int[], MlxKvCache?, float[]>? LayerForwardForTesting { get; set; }
    internal Func<int, float[], int, MlxKvCache, float[]>? LayerDecodeForTesting { get; set; }

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
    /// Runs every layer's <see cref="MlxDenseLayer.Forward"/> over a prompt and returns logits for
    /// every position, row-major [seqLen, VocabSize] (not just the last position -- a caller that
    /// only wants the next-token distribution reads the last <see cref="VocabSize"/> floats).
    /// <paramref name="caches"/> is optional: one entry per layer (same order as <see cref="Layers"/>).
    /// When given, each layer also appends this prompt's K/V into its cache (<see cref="MlxDenseLayer.Forward"/>'s
    /// own <c>cache</c> parameter), so a later <see cref="Decode"/> -- or another <see cref="Prefill"/>
    /// call -- on the same caches continues this sequence. Every cache must already hold the same
    /// length (a fresh prefill: all empty; a continuation: all already advanced together) -- this
    /// model does not support per-layer caches at different lengths. Positions start at that shared
    /// length, not always at 0, so a continuation's RoPE/attention offsets are correct.
    /// </summary>
    public float[] Prefill(ReadOnlySpan<int> tokenIds, MlxKvCache[]? caches = null)
    {
        int startPos = 0;
        if (caches is not null)
        {
            ValidateCaches(caches);
            if (tokenIds.Length == 0)
                throw new ArgumentException("Prefill requires at least one token when updating caches.", nameof(tokenIds));
            startPos = caches[0].Length;
            if (tokenIds.Length > caches[0].MaxSeqLen - startPos)
                throw new ArgumentException("Prefill would exceed a KV cache's maximum sequence length.", nameof(tokenIds));
        }

        int seqLen = tokenIds.Length;
        float[] hidden = Embed(tokenIds);
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) positions[i] = startPos + i;

        try
        {
            for (int i = 0; i < Layers.Length; i++)
            {
                hidden = LayerForwardForTesting is { } testForward
                    ? testForward(i, hidden, seqLen, positions, caches?[i])
                    : Layers[i].Forward(hidden, seqLen, positions, caches?[i]);
            }

            return FinalLogits(hidden, seqLen);
        }
        catch
        {
            ResetCaches(caches);
            throw;
        }
    }

    /// <summary>
    /// One incremental decode step via <see cref="MlxDenseLayer.ForwardDecode"/>. <paramref name="caches"/>
    /// has one entry per layer (same order as <see cref="Layers"/>), each already containing every
    /// earlier position's K/V.
    /// </summary>
    public float[] Decode(int token, int position, MlxKvCache[] caches)
    {
        ValidateCaches(caches);
        if (position != caches[0].Length)
            throw new ArgumentException($"Decode position ({position}) must equal the current KV cache length ({caches[0].Length}).", nameof(position));
        if (caches.Any(cache => cache.Length >= cache.MaxSeqLen))
            throw new InvalidOperationException("A KV cache is already at maximum sequence length.");

        try
        {
            float[] hidden = Embed([token]);
            for (int i = 0; i < Layers.Length; i++)
            {
                hidden = LayerDecodeForTesting is { } testDecode
                    ? testDecode(i, hidden, position, caches[i])
                    : Layers[i].ForwardDecode(hidden, position, caches[i]);
            }

            return FinalLogits(hidden, 1);
        }
        catch
        {
            ResetCaches(caches);
            throw;
        }
    }

    private void ValidateCaches(MlxKvCache[] caches)
    {
        ArgumentNullException.ThrowIfNull(caches);
        if (caches.Length != Layers.Length)
            throw new ArgumentException($"caches has {caches.Length} entries, expected one per layer ({Layers.Length}).", nameof(caches));

        for (int i = 0; i < Layers.Length; i++)
        {
            MlxKvCache cache = caches[i] ?? throw new ArgumentException($"caches[{i}] is null.", nameof(caches));
            MlxDenseLayer layer = Layers[i];
            if (cache.NumHeads != layer.NumKvHeads || cache.HeadDim != layer.HeadDim)
                throw new ArgumentException($"caches[{i}] dimensions [{cache.NumHeads}, {cache.HeadDim}] do not match layer dimensions [{layer.NumKvHeads}, {layer.HeadDim}].", nameof(caches));
            if (i > 0 && cache.Length != caches[0].Length)
                throw new ArgumentException($"caches[{i}].Length ({cache.Length}) != caches[0].Length ({caches[0].Length}) -- every layer's cache must be at the same length.", nameof(caches));
        }
    }

    private static void ResetCaches(MlxKvCache[]? caches)
    {
        if (caches is null) return;
        foreach (MlxKvCache? cache in caches)
            cache?.Reset();
    }

    private float[] FinalLogits(ReadOnlySpan<float> hidden, int seqLen)
    {
        float[] normed = MlxOps.RmsNorm(hidden, seqLen, HiddenDim, OutputNormWeight, Eps);
        return MlxOps.MatMul(normed, seqLen, HiddenDim, OutputWeight, VocabSize);
    }
}
