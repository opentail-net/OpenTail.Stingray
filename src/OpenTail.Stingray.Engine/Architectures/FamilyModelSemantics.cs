namespace OpenTail.Stingray.Engine;

using OpenTail.Stingray.Core;

/// <summary>
/// Model semantics for families whose hyperparameter interpretation is more than a one-line override.
/// Each is a pure function of the generic baseline plus metadata; see ArchitectureDescriptor.ApplyModelSemantics.
/// </summary>
internal static class FamilyModelSemantics
{
    /// <summary>
    /// Gemma 3 (gemma3.cpp): sqrt(d) embedding scale, GELU-tanh FFN, scalar sliding window + period (default 6,
    /// last layer of each block global), separate local RoPE base (default 10000, same rope dim), optional softcap.
    /// </summary>
    public static ModelHyperparams Gemma3(ModelArchitectureSemanticsContext ctx)
    {
        var hp = ctx.Baseline;
        int numLayers = hp.NumLayers;
        if (numLayers <= 0) return hp;

        int sliding = ctx.Int("attention.sliding_window");
        var result = hp with
        {
            EmbeddingScale = MathF.Sqrt(hp.EmbeddingDim),
            FfnActivation = FfnActivation.GeluApprox,
            SlidingWindowSize = sliding,
            FinalLogitSoftcap = ctx.Float("final_logit_softcapping"),
        };
        if (sliding > 0)
        {
            int period = ctx.Int("attention.sliding_window_pattern", 6);
            var swa = new bool[numLayers];
            for (int i = 0; i < numLayers; i++)
                swa[i] = period == 0 || (i % period < period - 1);
            result = result with
            {
                IsSwaLayer = swa,
                RopeThetaSwa = ctx.Float("rope.freq_base_swa", 10_000f),
                // ForwardPass builds the separate SWA RoPE table only when LayerRopeDim is set; a uniform array suffices.
                LayerRopeDim = Enumerable.Repeat(hp.RopeDim, numLayers).ToArray(),
            };
        }
        return result;
    }

    /// <summary>EXAONE 4: only the 64-layer models use SWA (window 4096, 3 SWA : 1 global, RoPE only on SWA layers).</summary>
    public static ModelHyperparams Exaone4(ModelArchitectureSemanticsContext ctx)
    {
        var hp = ctx.Baseline;
        int numLayers = hp.NumLayers;
        if (numLayers != 64) return hp;
        var pattern = ctx.BoolArray("attention.sliding_window_pattern");
        var swa = new bool[numLayers];
        for (int i = 0; i < numLayers; i++)
            swa[i] = pattern is { Count: > 0 } ? pattern[i % pattern.Count] : i % 4 < 3;
        return hp with
        {
            SlidingWindowSize = ctx.Int("attention.sliding_window", 4096),
            IsSwaLayer = swa,
            RopeOnlySwaLayers = true,
        };
    }

    /// <summary>
    /// Muse-Glimmer and AFMoE share the attention shape: per-layer sigmoid output gate, SWA pattern from a bool array or
    /// a scalar period (default 4), RoPE only on SWA layers. Muse adds an input-embedding RMS norm and a direct-multiply
    /// logit scale; AFMoE adds muP embedding scaling.
    /// </summary>
    public static ModelHyperparams GatedSwa(ModelArchitectureSemanticsContext ctx, bool muse)
    {
        var hp = ctx.Baseline;
        int numLayers = hp.NumLayers;
        var result = hp with
        {
            AttentionOutputGate = true,
            RopeOnlySwaLayers = true,
            PostNormEps = muse ? 1e-8f : hp.PostNormEps,
            InputEmbeddingRmsNorm = muse || hp.InputEmbeddingRmsNorm,
            EmbeddingScale = !muse && hp.EmbeddingScale == 1f ? MathF.Sqrt(hp.EmbeddingDim) : hp.EmbeddingScale,
        };
        if (muse)
        {
            float raw = ctx.Float("logit_scale");
            if (raw != 0f) result = result with { LogitScale = raw };
        }
        if (numLayers <= 0) return result;

        var pattern = ctx.BoolArray("attention.sliding_window_pattern");
        int period = pattern is { Count: > 0 } ? 0 : ctx.Int("attention.sliding_window_pattern", 4);
        var swa = new bool[numLayers];
        for (int i = 0; i < numLayers; i++)
            swa[i] = pattern is { Count: > 0 } ? pattern[i % pattern.Count] : period == 0 || (i % period < period - 1);
        result = result with
        {
            SlidingWindowSize = ctx.Int("attention.sliding_window"),
            IsSwaLayer = swa,
            FinalLogitSoftcap = ctx.Float("final_logit_softcapping"),
        };
        if (ctx.HasKey("rope.freq_base_swa"))
            result = result with
            {
                RopeThetaSwa = ctx.Float("rope.freq_base_swa"),
                LayerRopeDim = Enumerable.Repeat(hp.RopeDim, numLayers).ToArray(),
            };
        return result;
    }

    /// <summary>
    /// Granite (dense/MoE/hybrid) and MiniCPM share one graph builder in llama.cpp: the scale trio with different
    /// constants. GGUF "0 / absent = off" maps onto multiplicative identities. MiniCPM hardcodes formula defaults before
    /// checking overrides and never reads attention.scale; Granite's logit_scale is DIVIDED (reciprocal baked in here).
    /// </summary>
    public static ModelHyperparams Granite(ModelArchitectureSemanticsContext ctx, bool miniCpm, bool hybrid = false)
    {
        var hp = ctx.Baseline;
        float embeddingScale = hp.EmbeddingScale;
        float residualScale = hp.ResidualScale;
        float logitScale = hp.LogitScale;
        float attentionScale = hp.AttentionScaleOverride;

        if (miniCpm)
        {
            embeddingScale = 12.0f;
            residualScale = hp.NumLayers > 0 ? 1.4f / MathF.Sqrt(hp.NumLayers) : 1f;
            logitScale = hp.EmbeddingDim > 0 ? 256.0f / hp.EmbeddingDim : 1f;
        }
        float rawEmbedding = ctx.Float("embedding_scale");
        if (rawEmbedding != 0f) embeddingScale = rawEmbedding;
        float rawResidual = ctx.Float("residual_scale");
        if (rawResidual != 0f) residualScale = rawResidual;
        if (!miniCpm) attentionScale = ctx.Float("attention.scale"); // 0 = no override
        float rawLogit = ctx.Float("logit_scale");
        if (rawLogit != 0f) logitScale = 1f / rawLogit;

        var result = hp with
        {
            EmbeddingScale = embeddingScale,
            ResidualScale = residualScale,
            LogitScale = logitScale,
            AttentionScaleOverride = attentionScale,
            ScaleRawEmbeddings = !ctx.HasKey("deepstack_mapping"),
        };
        // granite-hybrid.cpp: rope.scaling.finetuned is a RoPE on/off switch (default on); 4.0-H ships false -> NoPE everywhere.
        if (hybrid && !ctx.Bool("rope.scaling.finetuned", true))
            result = result with { NoRopeLayerStep = 1 };
        return result;
    }
}
