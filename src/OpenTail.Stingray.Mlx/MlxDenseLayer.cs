namespace OpenTail.Stingray.Mlx;

/// <summary>
/// One dense (non-MoE, non-hybrid) transformer block's weights. <c>CudaForwardPass</c>'s per-layer
/// loop (QKV -> attention -> output projection -> residual -> FFN -> residual) is the structural
/// reference; nothing here has been run (no Mac on this development machine), so this is reviewed for
/// structural correctness only. See docs/4-performance/gpu/2026-11-06-mlx-metal-backend-plan.md.
/// </summary>
public sealed class MlxDenseLayer
{
    /// <summary>Row-major [hiddenDim] RMSNorm scale before attention.</summary>
    public required float[] AttnNormWeight { get; init; }
    /// <summary>Row-major [hiddenDim, numHeads*headDim].</summary>
    public required float[] Wq { get; init; }
    /// <summary>Row-major [hiddenDim, numKvHeads*headDim].</summary>
    public required float[] Wk { get; init; }
    public required float[] Wv { get; init; }
    /// <summary>Row-major [numHeads*headDim, hiddenDim] attention output projection.</summary>
    public required float[] Wo { get; init; }
    /// <summary>Row-major [hiddenDim] RMSNorm scale before the FFN.</summary>
    public required float[] FfnNormWeight { get; init; }
    /// <summary>Row-major [hiddenDim, intermDim] gate/up projections.</summary>
    public required float[] WGate { get; init; }
    public required float[] WUp { get; init; }
    /// <summary>Row-major [intermDim, hiddenDim] down projection.</summary>
    public required float[] WDown { get; init; }

    public required int HiddenDim { get; init; }
    public required int NumHeads { get; init; }
    /// <summary>
    /// Key/value head count. <b>No GQA/MQA broadcast is implemented yet</b>: must equal
    /// <see cref="NumHeads"/> for now, checked in <see cref="Forward"/> -- a family whose KV head
    /// count differs would need K/V repeated across query-head groups before
    /// <see cref="MlxOps.Attention"/>, not built here.
    /// </summary>
    public required int NumKvHeads { get; init; }
    public required int HeadDim { get; init; }
    public required int IntermDim { get; init; }
    public float Eps { get; init; } = 1e-5f;
    public float RopeBase { get; init; } = 10000f;

    /// <summary>
    /// One transformer block over a whole sequence: RMSNorm -> QKV -> split into heads -> RoPE on
    /// each Q/K head -> causal multi-head self-attention -> merge heads -> output projection ->
    /// residual add -> RMSNorm -> SwiGLU FFN (<c>down(silu(gate(x)) * up(x))</c>) -> residual add.
    /// <paramref name="hidden"/> is row-major [seqLen, HiddenDim]; returns the same shape.
    /// <paramref name="positions"/> has length seqLen (RoPE position ids).
    ///
    /// <para>Head split/merge (<see cref="MlxOps.SplitHeads"/>/<see cref="MlxOps.MergeHeads"/>) is
    /// composed from <c>mlx_reshape</c> + <c>mlx_transpose_axes</c>, not TensorSharp's fused
    /// <c>FlatToHeadFirst</c> Metal kernel -- faithful to the reshape, not to that specific kernel.
    /// RoPE runs per head on a [seqLen, headDim] slice, matching TensorSharp's own call site
    /// (<c>MlxBasicOps.cs</c>'s RoPE reshapes to isolate one head-sized feature dimension before
    /// calling <c>FastRopeDynamic</c>).</para>
    ///
    /// <para><b>Still simplified, named explicitly:</b> no GQA broadcast (<see cref="NumKvHeads"/>
    /// must equal <see cref="NumHeads"/>); no bias terms (<see cref="MlxOps.AddMm"/> exists for when a
    /// family needs one); F32 weights only (a real GGUF checkpoint's quantized weights would need
    /// dequantizing to F32 first -- not done here).</para>
    ///
    /// <para><paramref name="cache"/> is optional: when provided, this call also appends the
    /// prompt's post-RoPE K/V (head-first, the same layout <see cref="MlxKvCache.Append"/> already
    /// accepts for any <c>seqLen</c>, not just one) so a later <see cref="ForwardDecode"/> on the same
    /// cache continues this sequence rather than starting a fresh, cache-less one. Passing a
    /// non-empty cache (<see cref="MlxKvCache.Length"/> &gt; 0) is refused: that would be continuing a
    /// multi-token prefill into an already-populated cache, which needs an offset causal mask this
    /// port does not build (see <see cref="ForwardDecode"/>'s own doc comment for the same limitation
    /// from the other direction).</para>
    /// </summary>
    public float[] Forward(ReadOnlySpan<float> hidden, int seqLen, ReadOnlySpan<int> positions, MlxKvCache? cache = null)
    {
        if (hidden.Length != seqLen * HiddenDim)
            throw new ArgumentException($"hidden has {hidden.Length} elements, expected seqLen*HiddenDim = {seqLen * HiddenDim}.", nameof(hidden));
        if (positions.Length != seqLen)
            throw new ArgumentException($"positions has {positions.Length} elements, expected seqLen = {seqLen}.", nameof(positions));
        if (NumKvHeads != NumHeads)
            throw new NotSupportedException($"NumKvHeads ({NumKvHeads}) != NumHeads ({NumHeads}): GQA broadcast is not implemented.");
        if (cache is { Length: > 0 })
            throw new NotSupportedException($"cache already has {cache.Length} positions: continuing a multi-token prefill into an already-populated cache is not implemented.");

        float[] normed = MlxOps.RmsNorm(hidden, seqLen, HiddenDim, AttnNormWeight, Eps);

        float[] qFlat = MlxOps.MatMul(normed, seqLen, HiddenDim, Wq, NumHeads * HeadDim);
        float[] kFlat = MlxOps.MatMul(normed, seqLen, HiddenDim, Wk, NumKvHeads * HeadDim);
        float[] vFlat = MlxOps.MatMul(normed, seqLen, HiddenDim, Wv, NumKvHeads * HeadDim);

        float[] q = ApplyRopePerHead(MlxOps.SplitHeads(qFlat, seqLen, NumHeads, HeadDim), NumHeads, seqLen, positions);
        float[] k = ApplyRopePerHead(MlxOps.SplitHeads(kFlat, seqLen, NumKvHeads, HeadDim), NumKvHeads, seqLen, positions);
        float[] v = MlxOps.SplitHeads(vFlat, seqLen, NumKvHeads, HeadDim);

        cache?.Append(k, v, seqLen);

        float scale = 1f / MathF.Sqrt(HeadDim);
        float[] attnOut = MlxOps.Attention(q, k, v, NumHeads, querySeqLen: seqLen, kvSeqLen: seqLen, HeadDim, scale, causal: true);
        float[] merged = MlxOps.MergeHeads(attnOut, NumHeads, seqLen, HeadDim);

        float[] projected = MlxOps.MatMul(merged, seqLen, NumHeads * HeadDim, Wo, HiddenDim);
        float[] afterAttn = Add(hidden, projected);

        float[] normed2 = MlxOps.RmsNorm(afterAttn, seqLen, HiddenDim, FfnNormWeight, Eps);
        float[] gate = MlxOps.MatMul(normed2, seqLen, HiddenDim, WGate, IntermDim);
        float[] up = MlxOps.MatMul(normed2, seqLen, HiddenDim, WUp, IntermDim);
        float[] ffnHidden = MlxOps.Multiply(MlxOps.Silu(gate), up);
        float[] down = MlxOps.MatMul(ffnHidden, seqLen, IntermDim, WDown, HiddenDim);

        return Add(afterAttn, down);
    }

    /// <summary>
    /// One incremental decode step: a single new token's hidden state attends to everything already
    /// in <paramref name="cache"/> plus itself. Matches TensorSharp's own convention for why no mask
    /// is needed here (<c>FastScaledDotProductAttention</c> call sites pass <c>string.Empty</c>
    /// specifically "for a single-token decode step" -- a lone query has no future token relative to
    /// itself or to the cached past, so <see cref="Forward"/>'s causal mask has nothing to do).
    /// <paramref name="hidden"/> is [HiddenDim] (one token); <paramref name="position"/> is its
    /// absolute position (for RoPE). <paramref name="cache"/> is appended to in place.
    ///
    /// <para><b>Only the single-new-token case is handled.</b> Continuing a multi-token prefill into
    /// an already-populated cache (more than one new token at once, non-empty cache) is refused
    /// rather than silently mismasked -- TensorSharp's "causal" mask string assumes the query and key
    /// axes start causally aligned, which an offset multi-token continuation would need its own mask
    /// array for, not built here.</para>
    /// </summary>
    public float[] ForwardDecode(ReadOnlySpan<float> hidden, int position, MlxKvCache cache)
    {
        if (hidden.Length != HiddenDim)
            throw new ArgumentException($"hidden has {hidden.Length} elements, expected HiddenDim = {HiddenDim}.", nameof(hidden));
        if (NumKvHeads != NumHeads)
            throw new NotSupportedException($"NumKvHeads ({NumKvHeads}) != NumHeads ({NumHeads}): GQA broadcast is not implemented.");
        if (cache.NumHeads != NumKvHeads || cache.HeadDim != HeadDim)
            throw new ArgumentException($"cache is [{cache.NumHeads} heads, {cache.HeadDim} headDim], expected [{NumKvHeads}, {HeadDim}].", nameof(cache));

        float[] normed = MlxOps.RmsNorm(hidden, 1, HiddenDim, AttnNormWeight, Eps);

        float[] qFlat = MlxOps.MatMul(normed, 1, HiddenDim, Wq, NumHeads * HeadDim);
        float[] kFlat = MlxOps.MatMul(normed, 1, HiddenDim, Wk, NumKvHeads * HeadDim);
        float[] vFlat = MlxOps.MatMul(normed, 1, HiddenDim, Wv, NumKvHeads * HeadDim);

        ReadOnlySpan<int> positions = [position];
        float[] q = ApplyRopePerHead(MlxOps.SplitHeads(qFlat, 1, NumHeads, HeadDim), NumHeads, 1, positions);
        float[] kNew = ApplyRopePerHead(MlxOps.SplitHeads(kFlat, 1, NumKvHeads, HeadDim), NumKvHeads, 1, positions);
        float[] vNew = MlxOps.SplitHeads(vFlat, 1, NumKvHeads, HeadDim);

        cache.Append(kNew, vNew, seqLen: 1);
        float[] kAll = cache.Keys();
        float[] vAll = cache.Values();

        float scale = 1f / MathF.Sqrt(HeadDim);
        float[] attnOut = MlxOps.Attention(q, kAll, vAll, NumHeads, querySeqLen: 1, kvSeqLen: cache.Length, HeadDim, scale, causal: false);
        float[] merged = MlxOps.MergeHeads(attnOut, NumHeads, 1, HeadDim);

        float[] projected = MlxOps.MatMul(merged, 1, NumHeads * HeadDim, Wo, HiddenDim);
        float[] afterAttn = Add(hidden, projected);

        float[] normed2 = MlxOps.RmsNorm(afterAttn, 1, HiddenDim, FfnNormWeight, Eps);
        float[] gate = MlxOps.MatMul(normed2, 1, HiddenDim, WGate, IntermDim);
        float[] up = MlxOps.MatMul(normed2, 1, HiddenDim, WUp, IntermDim);
        float[] ffnHidden = MlxOps.Multiply(MlxOps.Silu(gate), up);
        float[] down = MlxOps.MatMul(ffnHidden, 1, IntermDim, WDown, HiddenDim);

        return Add(afterAttn, down);
    }

    /// <summary>Applies <see cref="MlxOps.Rope"/> independently to each head's [seqLen, headDim] slice of a head-first [numHeads, seqLen, headDim] array.</summary>
    private float[] ApplyRopePerHead(float[] headFirst, int numHeads, int seqLen, ReadOnlySpan<int> positions)
    {
        int headDim = HeadDim;
        var result = new float[headFirst.Length];
        for (int h = 0; h < numHeads; h++)
        {
            var slice = new ReadOnlySpan<float>(headFirst, h * seqLen * headDim, seqLen * headDim);
            float[] roped = MlxOps.Rope(slice, seqLen, headDim, headDim, traditional: false, RopeBase, scale: 1f, positions);
            roped.CopyTo(result.AsSpan(h * seqLen * headDim, seqLen * headDim));
        }
        return result;
    }

    private static float[] Add(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        // The residual add has no dedicated port yet (TensorSharp's mlx_add exists but isn't bound
        // here); this is plain host-side addition, correct regardless of MLX, so it needs no native
        // call or test beyond the shape check below.
        if (a.Length != b.Length) throw new ArgumentException($"a has {a.Length} elements, b has {b.Length}; they must match.");
        var result = new float[a.Length];
        for (int i = 0; i < result.Length; i++) result[i] = a[i] + b[i];
        return result;
    }
}
