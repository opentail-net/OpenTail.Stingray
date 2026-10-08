namespace OpenTail.Stingray.Engine;

using OpenTail.Stingray.Core;

/// <summary>
/// Gemma 4 model semantics: sliding-window/global layer pattern, per-layer head/RoPE dims and KV heads, shared-KV
/// source layers, sqrt(d) embedding scale, GELU-tanh FFN, softcap, PLE and K==V detection. Pure function of the
/// generic baseline plus the file's metadata and tensor inventory.
/// </summary>
internal static class Gemma4ModelSemantics
{
    public static ModelHyperparams Apply(ModelArchitectureSemanticsContext ctx)
    {
        var hp = ctx.Baseline;
        var metadata = ctx.Metadata;
        var tensors = ctx.TensorSource;
        int numLayers = hp.NumLayers;
        int headDim = hp.HeadDim;
        int ropeDim = hp.RopeDim;

        int keyLengthSwa = ctx.Int("attention.key_length_swa", headDim);
        int ropeDimSwa = ctx.Int("rope.dimension_count_swa", keyLengthSwa);
        int sharedKvLayers = ctx.Int("attention.shared_kv_layers");

        IReadOnlyList<bool>? isSwaLayer = null;
        IReadOnlyList<int>? layerHeadDim = null, layerRopeDim = null, layerKvHeads = null, kvSourceLayer = null;

        if (numLayers > 0)
        {
            var pattern = ctx.BoolArray("attention.sliding_window_pattern");
            var swa = new bool[numLayers];
            if (pattern is not null && pattern.Count > 0)
                for (int i = 0; i < numLayers; i++)
                    swa[i] = pattern[i % pattern.Count];
            isSwaLayer = swa;

            var hdArr = new int[numLayers];
            var rdArr = new int[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                hdArr[i] = swa[i] ? keyLengthSwa : headDim;
                rdArr[i] = swa[i] ? ropeDimSwa : ropeDim;
            }
            layerHeadDim = hdArr;
            layerRopeDim = rdArr;

            // Per-layer KV head count (12B: 8 on SWA, 1 on global); a corrupt 0 would divide by zero downstream.
            var kvArr = ctx.IntArray("attention.head_count_kv");
            if (kvArr is not null && kvArr.Count > 0)
            {
                var lkv = new int[numLayers];
                for (int i = 0; i < numLayers; i++)
                {
                    int val = kvArr[i % kvArr.Count];
                    lkv[i] = val > 0 ? val : 1;
                }
                layerKvHeads = lkv;
            }

            if (sharedKvLayers > 0)
            {
                int firstSharedLayer = numLayers - sharedKvLayers;
                var src = new int[numLayers];
                for (int i = 0; i < numLayers; i++)
                {
                    if (i < firstSharedLayer) { src[i] = -1; continue; }
                    int found = -1;
                    for (int j = firstSharedLayer - 1; j >= 0; j--)
                        if (swa[j] == swa[i]) { found = j; break; }
                    src[i] = found;
                }
                kvSourceLayer = src;
            }
        }

        return hp with
        {
            SlidingWindowSize = ctx.Int("attention.sliding_window"),
            PerLayerEmbeddingWidth = ctx.Int("embedding_length_per_layer_input"),
            FinalLogitSoftcap = ctx.Float("final_logit_softcapping"),
            RopeThetaSwa = ctx.Float("rope.freq_base_swa", 10_000f),
            EmbeddingScale = MathF.Sqrt(hp.EmbeddingDim),
            FfnActivation = FfnActivation.GeluApprox,
            HasPerLayerTokenEmbd = metadata.ContainsKey("_opentailllm.has_ple")
                || tensors.FindTensor("per_layer_token_embd.weight") is not null,
            HasLayerOutputScale = metadata.ContainsKey("_opentailllm.has_layer_output_scale")
                || tensors.FindTensor("blk.0.layer_output_scale.weight") is not null,
            // 12B dense global layers omit attn_v and reuse K as V (not a GGUF key; flagged by the loader).
            AttentionKEqV = metadata.ContainsKey("_opentailllm.attention_k_eq_v"),
            IsSwaLayer = isSwaLayer,
            LayerHeadDim = layerHeadDim,
            LayerRopeDim = layerRopeDim,
            LayerKvHeads = layerKvHeads,
            KvSourceLayer = kvSourceLayer,
        };
    }
}
