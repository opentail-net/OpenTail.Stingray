namespace OpenTail.Stingray.Mlx;

/// <summary>
/// One dense (non-MoE, non-hybrid) transformer block's weights -- single-head attention only in this
/// first wiring (see <see cref="MlxDenseLayer.Forward"/>). <c>CudaForwardPass</c>'s per-layer loop
/// (QKV -> attention -> output projection -> residual -> FFN -> residual) is the structural reference;
/// nothing here has been run (no Mac on this development machine), so this is reviewed for structural
/// correctness only. See docs/4-performance/gpu/2026-11-06-mlx-metal-backend-plan.md.
/// </summary>
public sealed class MlxDenseLayer
{
    /// <summary>Row-major [hiddenDim] RMSNorm scale before attention.</summary>
    public required float[] AttnNormWeight { get; init; }
    /// <summary>Row-major [hiddenDim, headDim] (single head: <see cref="HeadDim"/> == attention width).</summary>
    public required float[] Wq { get; init; }
    public required float[] Wk { get; init; }
    public required float[] Wv { get; init; }
    /// <summary>Row-major [headDim, hiddenDim] attention output projection.</summary>
    public required float[] Wo { get; init; }
    /// <summary>Row-major [hiddenDim] RMSNorm scale before the FFN.</summary>
    public required float[] FfnNormWeight { get; init; }
    /// <summary>Row-major [hiddenDim, intermDim] gate/up projections.</summary>
    public required float[] WGate { get; init; }
    public required float[] WUp { get; init; }
    /// <summary>Row-major [intermDim, hiddenDim] down projection.</summary>
    public required float[] WDown { get; init; }

    public required int HiddenDim { get; init; }
    public required int HeadDim { get; init; }
    public required int IntermDim { get; init; }
    public float Eps { get; init; } = 1e-5f;
    public float RopeBase { get; init; } = 10000f;

    /// <summary>
    /// One transformer block over a whole sequence: RMSNorm -> single-head QKV (<see cref="Wq"/>/
    /// <see cref="Wk"/>/<see cref="Wv"/>) -> RoPE on Q and K -> causal self-attention -> output
    /// projection -> residual add -> RMSNorm -> SwiGLU FFN (<c>down(silu(gate(x)) * up(x))</c>) ->
    /// residual add. <paramref name="hidden"/> is row-major [seqLen, HiddenDim]; returns the same
    /// shape. <paramref name="positions"/> has length seqLen (RoPE position ids).
    ///
    /// <para><b>Simplified relative to a real dense-llama layer, named explicitly:</b> single
    /// attention head only (<see cref="Wq"/>/<see cref="Wk"/>/<see cref="Wv"/> project straight to
    /// <see cref="HeadDim"/>, no multi-head split/concat); no GQA (one KV "head" only); no bias terms
    /// (<see cref="MlxOps.AddMm"/> exists for when a family needs one); F32 weights only (a real GGUF
    /// checkpoint's quantized weights would need dequantizing to F32 first -- not done here); no KV
    /// cache (every call recomputes attention over the whole sequence, there is no incremental decode
    /// step yet).</para>
    /// </summary>
    public float[] Forward(ReadOnlySpan<float> hidden, int seqLen, ReadOnlySpan<int> positions)
    {
        if (hidden.Length != seqLen * HiddenDim)
            throw new ArgumentException($"hidden has {hidden.Length} elements, expected seqLen*HiddenDim = {seqLen * HiddenDim}.", nameof(hidden));
        if (positions.Length != seqLen)
            throw new ArgumentException($"positions has {positions.Length} elements, expected seqLen = {seqLen}.", nameof(positions));

        float[] normed = MlxOps.RmsNorm(hidden, seqLen, HiddenDim, AttnNormWeight, Eps);

        float[] q = MlxOps.MatMul(normed, seqLen, HiddenDim, Wq, HeadDim);
        float[] k = MlxOps.MatMul(normed, seqLen, HiddenDim, Wk, HeadDim);
        float[] v = MlxOps.MatMul(normed, seqLen, HiddenDim, Wv, HeadDim);

        q = MlxOps.Rope(q, seqLen, HeadDim, HeadDim, traditional: false, RopeBase, scale: 1f, positions);
        k = MlxOps.Rope(k, seqLen, HeadDim, HeadDim, traditional: false, RopeBase, scale: 1f, positions);

        float scale = 1f / MathF.Sqrt(HeadDim);
        float[] attnOut = MlxOps.Attention(q, k, v, numHeads: 1, seqLen, HeadDim, scale, causal: true);

        float[] projected = MlxOps.MatMul(attnOut, seqLen, HeadDim, Wo, HiddenDim);
        float[] afterAttn = Add(hidden, projected);

        float[] normed2 = MlxOps.RmsNorm(afterAttn, seqLen, HiddenDim, FfnNormWeight, Eps);
        float[] gate = MlxOps.MatMul(normed2, seqLen, HiddenDim, WGate, IntermDim);
        float[] up = MlxOps.MatMul(normed2, seqLen, HiddenDim, WUp, IntermDim);
        float[] ffnHidden = MlxOps.Multiply(MlxOps.Silu(gate), up);
        float[] down = MlxOps.MatMul(ffnHidden, seqLen, IntermDim, WDown, HiddenDim);

        return Add(afterAttn, down);
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
