namespace OpenTail.Stingray.Audio.VoxCpm2;

/// <summary>
/// Native C# port of VoxCPM2's local encoder forward pass, from `generator.cpp`'s
/// `VoxCPM2LocalEncoderRuntime::Impl::build`/`encode_patch` and `minicpm_blocks.h`'s
/// `minicpm_layer`/`minicpm_transformer`/`apply_minicpm_rope` (not guessed). Real per-call
/// shape: one "patch" of `patch_size` (real: 4) rows of `feat_dim` (real: 64) continuous
/// features -&gt; `Linear(feat_dim-&gt;encoderHiddenDim)` per row -&gt; a real learned "special token"
/// row PREPENDED (CLS-token pattern) -&gt; the shared <see cref="VoxCpm2MiniCpmBidirectionalStack"/>
/// (real BIDIRECTIONAL 12-layer MiniCPM transformer, RoPE positions `0..patchSize`) -&gt; only the
/// special-token row (index 0) is kept -&gt; `Linear(encoderHiddenDim-&gt;lmHiddenDim)` projects into
/// the LM's own hidden space.
///
/// <para><b>Real RoPE, confirmed non-trivial and NOT plain unscaled RoPE</b>: NEOX (split-half)
/// rotation, real `longrope` per-dimension frequency-correction factors (`rope_scaling.
/// short_factor`, an array of 64 real learned/derived values, reused directly from this
/// checkpoint's `lm_config` since `local_transformer_config` inherits the base LM's rope
/// settings verbatim except width/depth/head-count) -- `long_factor` is never selected here
/// because `apply_minicpm_rope`'s real config carries `max_position_embeddings=32768 &lt;=
/// original_max_position_embeddings=32768` (equal, not strictly greater), so
/// `active_rope_factors` always picks `short_factor`. Also confirmed real (not guessed):
/// `rope_attn_factor`'s real formula evaluates to exactly `1.0` for this checkpoint (same
/// `&lt;=` condition), and `ext_factor=0.0`/`freq_scale=1.0` always for this call site, so the
/// reference's more general YaRN ramp-mixing logic never activates here.</para>
///
/// <para>The transformer stack itself (per-layer math, RoPE table, NEOX rotation) is shared with
/// VoxCPM2's DiT estimator decoder via <see cref="VoxCpm2MiniCpmBidirectionalStack"/> -- both use
/// the exact same real config shape (`hidden_dim=1024`, `num_heads=16`, `num_key_value_heads=2`,
/// `head_dim=128`, `ffn_dim=4096`, 12 layers), confirmed identical in the real checkpoint's
/// `config.json`.</para>
/// </summary>
public static class VoxCpm2LocalEncoder
{
    public const int PatchSize = 4;
    public const int FeatDim = 64;
    public const int EncoderHiddenDim = VoxCpm2MiniCpmBidirectionalStack.HiddenDim;

    // Real `lm_config.rope_scaling.short_factor` (64 values, one per RoPE pair -- headDim/2),
    // extracted directly from the checkpoint's embedded config.json (not guessed). Shared with
    // VoxCpm2MiniCpmBidirectionalStack since the DiT decoder uses the identical rope config.
    public static readonly float[] RopeShortFactor =
    [
        0.9977997200264581f, 1.014658295992452f, 1.0349680404997148f, 1.059429246056193f,
        1.0888815016813513f, 1.1243301355211495f, 1.166977103606075f, 1.2182568066927284f,
        1.2798772354275727f, 1.3538666751582975f, 1.4426259039919596f, 1.5489853358570191f,
        1.6762658237220625f, 1.8283407612492941f, 2.0096956085876183f, 2.225478927469756f,
        2.481536379650452f, 2.784415934557119f, 3.1413289096347365f, 3.560047844772632f,
        4.048719380066383f, 4.615569542115128f, 5.2684819496549835f, 6.014438591970396f,
        6.858830049237097f, 7.804668263503327f, 8.851768731513417f, 9.99600492938444f,
        11.228766118181639f, 12.536757560834843f, 13.902257701387796f, 15.303885189125953f,
        16.717837610115794f, 18.119465097853947f, 19.484965238406907f, 20.792956681060105f,
        22.02571786985731f, 23.16995406772833f, 24.217054535738416f, 25.16289275000465f,
        26.007284207271347f, 26.753240849586767f, 27.40615325712662f, 27.973003419175363f,
        28.461674954469114f, 28.880393889607006f, 29.237306864684626f, 29.540186419591297f,
        29.79624387177199f, 30.01202719065413f, 30.193382037992453f, 30.34545697551969f,
        30.47273746338473f, 30.579096895249787f, 30.66785612408345f, 30.741845563814174f,
        30.80346599254902f, 30.85474569563567f, 30.897392663720595f, 30.932841297560394f,
        30.962293553185553f, 30.986754758742034f, 31.007064503249293f, 31.02392307921529f,
    ];

    /// <summary>Encodes one real patch of `[PatchSize][FeatDim]` continuous features into the LM's
    /// `lmHiddenDim`-wide embedding space.</summary>
    public static float[] EncodePatch(VoxCpm2LocalEncoderWeights w, float[][] patchFeatures, int lmHiddenDim)
    {
        if (patchFeatures.Length != PatchSize) throw new ArgumentException($"Expected {PatchSize} rows.", nameof(patchFeatures));

        int seqLen = PatchSize + 1;
        var hidden = new float[seqLen][];
        hidden[0] = (float[])w.SpecialToken.Clone();
        for (int i = 0; i < PatchSize; i++)
            hidden[i + 1] = VoxCpm2MiniCpmBidirectionalStack.Linear(patchFeatures[i], w.InProjWeight, w.InProjBias, FeatDim, EncoderHiddenDim);

        hidden = VoxCpm2MiniCpmBidirectionalStack.Run(hidden, w.Layers, w.FinalNorm, seqLen);

        return VoxCpm2MiniCpmBidirectionalStack.Linear(hidden[0], w.EncToLmProjWeight, w.EncToLmProjBias, EncoderHiddenDim, lmHiddenDim);
    }
}
