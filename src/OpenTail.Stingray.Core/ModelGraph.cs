namespace OpenTail.Stingray.Core;

/// <summary>
/// Represents the full computation graph of a loaded model.
/// Layers are stored in execution order; weights are resolved lazily.
/// </summary>
public sealed class ModelGraph
{
    public string Architecture { get; init; } = string.Empty;
    public ModelHyperparams Hyperparams { get; init; } = new();
    public IReadOnlyList<ModelLayer> Layers { get; init; } = [];
    public IReadOnlyDictionary<string, GgufTensorInfo> WeightIndex { get; init; } =
        new Dictionary<string, GgufTensorInfo>();
}

public sealed record ModelHyperparams
{
    public int VocabSize { get; init; }
    public int ContextLength { get; init; }
    public int EmbeddingDim { get; init; }
    public int NumLayers { get; init; }
    public int NumHeads { get; init; }
    public int NumKvHeads { get; init; }
    public int IntermediateDim { get; init; }

    /// <summary>
    /// Attention head dimension. For most models this equals EmbeddingDim / NumHeads,
    /// but some architectures (e.g. Qwen3-MoE) use a larger head dim stored in
    /// {arch}.attention.key_length metadata.
    /// </summary>
    public int HeadDim { get; init; }

    public float RmsNormEps { get; init; } = 1e-5f;
    public float RopeTheta { get; init; } = 10_000f;

    /// <summary>
    /// Scalar multiplier applied to the token embeddings before they enter the
    /// transformer trunk. Gemma 4 multiplies by <c>sqrt(EmbeddingDim)</c>; Granite and
    /// MiniCPM read an explicit <c>{arch}.embedding_scale</c> metadata value (Granite
    /// 3.3 2B ships 12.0). Defaults to 1 (no scaling) for every other architecture.
    /// </summary>
    public float EmbeddingScale { get; init; } = 1f;

    /// <summary>
    /// Cap value for the final-logits softcap (<c>x = tanh(x/cap) * cap</c>).
    /// 0 disables softcapping (default for non-Gemma architectures). Gemma 4 = 30.0.
    /// </summary>
    public float FinalLogitSoftcap { get; init; }

    /// <summary>
    /// Scalar multiplier applied to a sublayer's output (attention output and FFN
    /// output, independently) immediately before it is added to the residual stream.
    /// Granite-family models (<c>models/granite.cpp</c>'s <c>build_layer_ffn</c>, shared
    /// verbatim by MiniCPM) apply this at both residual adds. Defaults to 1 (no scaling).
    /// </summary>
    public float ResidualScale { get; init; } = 1f;

    /// <summary>
    /// Explicit attention softmax scale (<c>kq_scale</c>), overriding the usual
    /// <c>1/sqrt(HeadDim)</c>. 0 means "no override — use the default". Granite reads
    /// <c>{arch}.attention.scale</c> (3.3 2B ships 0.015625 = 1/64, not 1/sqrt(64)=0.125).
    /// </summary>
    public float AttentionScaleOverride { get; init; }

    /// <summary>
    /// Scalar multiplier applied to the final logits after the output projection.
    /// Granite/MiniCPM declare a GGUF <c>{arch}.logit_scale</c> that llama.cpp DIVIDES
    /// by (<c>ggml_scale(cur, 1/f_logit_scale)</c>), so this field already carries the
    /// reciprocal — <see cref="ModelHyperparams"/> callers just multiply by it. Command-R
    /// uses the opposite convention (multiplies by the raw value directly); it is not
    /// wired here since no small Command-R checkpoint has been validated yet. Defaults to
    /// 1 (no scaling).
    /// </summary>
    public float LogitScale { get; init; } = 1f;

    /// <summary>
    /// RoPE base frequency used by sliding-window-attention layers. Gemma 4 mixes
    /// two RoPE bases: <see cref="RopeTheta"/> (1e6 for global layers) and this
    /// value (1e4 for SWA layers). 0 when the model has only one RoPE base.
    /// </summary>
    public float RopeThetaSwa { get; init; }

    /// <summary>
    /// Whether the model has bias terms on the Q/K/V attention projections (e.g. Qwen models).
    /// Detected at load time by probing for "blk.0.attn_q.bias" in the GGUF tensor index.
    /// </summary>
    public bool HasAttnBias { get; init; }

    /// <summary>
    /// Whether the model also carries a bias on the attention <em>output</em> projection
    /// (<c>blk.*.attn_output.bias</c>). Qwen2 has Q/K/V bias but no output-projection bias,
    /// so this is probed independently of <see cref="HasAttnBias"/> (and is only ever true
    /// when <see cref="HasAttnBias"/> is). Mirrors llama.cpp treating <c>bo</c> as optional.
    /// </summary>
    public bool HasAttnOutputBias { get; init; }

    /// <summary>
    /// Whether the model carries learned biases on its attention-norm, FFN-norm, and
    /// output-norm tensors — i.e. uses LayerNorm (mean/variance-normalize + learned
    /// scale/bias) rather than the RMSNorm path. GPT-NeoX/Pythia is the first user.
    /// Detected at load time by probing for "blk.0.attn_norm.bias".
    /// </summary>
    public bool HasNormBias { get; init; }

    /// <summary>
    /// Whether the model's norm uses true LayerNorm math (mean-subtract + variance-normalize)
    /// rather than RMSNorm (no mean-subtraction), independent of whether a learned bias exists.
    /// Always true when <see cref="HasNormBias"/> is (a bias tensor implies LayerNorm), but also
    /// true for architectures like Command-R (cohere2) that use LayerNorm with a learned scale
    /// but NO bias tensor at all — a fact llama.cpp's graph hardcodes per-architecture
    /// (<c>LLM_NORM</c> vs <c>LLM_NORM_RMS</c> in <c>build_norm</c>), not something a GGUF's
    /// tensor inventory can distinguish on its own (a weight-only norm tensor looks identical
    /// whether the architecture intends RMSNorm or bias-less LayerNorm), hence the arch-string
    /// check rather than tensor-presence detection.
    /// </summary>
    public bool UsesLayerNorm { get; init; }

    /// <summary>
    /// Whether the norm has NEITHER a learned scale NOR a bias — pure mean-subtract +
    /// variance-normalize with no parameters at all. OLMo v1 (<c>src/models/olmo.cpp</c>:
    /// <c>build_norm(inpL, NULL, NULL, LLM_NORM, il)</c>) is the only known user: its GGUF ships
    /// no <c>attn_norm</c>/<c>ffn_norm</c>/<c>output_norm</c> tensor at all. Distinct from
    /// <see cref="UsesLayerNorm"/> (which still implies a learned weight tensor exists) — when
    /// this is true, <see cref="UsesLayerNorm"/> is irrelevant, since there is no weight to decide
    /// LayerNorm-vs-RMSNorm math for; the norm is unconditionally
    /// <c>SimdKernels.PureLayerNorm</c>. Not detectable from tensor presence alone, since a
    /// missing norm tensor is the SAME signal OLMo2 uses for "no pre-norm at all, sandwich-normed
    /// on the output instead" — needs the arch-string check to disambiguate.
    /// </summary>
    public bool UsesUnweightedNorm { get; init; }

    /// <summary>
    /// Whether the dense FFN carries biases on both the up and down projections
    /// (<c>blk.*.ffn_up.bias</c> / <c>blk.*.ffn_down.bias</c>). GPT-NeoX pairs this with a
    /// non-gated GELU FFN: <c>down(gelu(up(x) + bUp)) + bDown</c> — the up bias goes inside
    /// the activation, not after it.
    /// </summary>
    public bool HasFfnBias { get; init; }

    /// <summary>
    /// Whether attention and FFN both read the SAME incoming residual and their outputs are
    /// summed with it three ways (<c>x + attn(ln1(x)) + ffn(ln2(x))</c>), rather than the
    /// ordinary sequential residual (<c>x1 = x + attn(ln1(x)); out = x1 + ffn(ln2(x1))</c>).
    /// GPT-NeoX-family models read this from GGUF metadata (<c>gptneox.use_parallel_residual</c>,
    /// llama.cpp's <c>LLM_KV_USE_PARALLEL_RESIDUAL</c>); Pythia checkpoints always set it true.
    /// </summary>
    public bool UseParallelResidual { get; init; }

    /// <summary>
    /// Whether the model has per-head Q/K RMSNorm (e.g. Qwen3).
    /// Detected at load time by probing for "blk.0.attn_q_norm.weight" in the GGUF tensor index.
    /// </summary>
    public bool HasQkNorm { get; init; }

    /// <summary>
    /// Whether QK-norm uses a per-channel learned weight of size <c>numHeads * headDim</c>
    /// (OLMoE) rather than a single <c>headDim</c> vector shared across heads (Qwen3).
    /// Detected at load time from <c>blk.0.attn_q_norm.weight</c>'s element count.
    /// Only meaningful when <see cref="HasQkNorm"/> and <see cref="UseL2QkNorm"/> is false.
    /// </summary>
    public bool IsPerChannelQkNorm { get; init; }

    // ── MoE (Mixture of Experts) ──

    /// <summary>Whether this model uses Mixture of Experts architecture.</summary>
    public bool IsMoE { get; init; }

    /// <summary>Total number of experts per layer (e.g. 16 for Llama 4 Scout).</summary>
    public int NumExperts { get; init; }

    /// <summary>Number of experts activated per token (e.g. 1 for Llama 4 Scout, 2 for Mixtral).</summary>
    public int NumActiveExperts { get; init; }

    /// <summary>FFN dimension per expert (may differ from IntermediateDim which is the shared FFN dim).</summary>
    public int ExpertIntermediateDim { get; init; }

    /// <summary>Whether the model has a shared expert that runs on every token (e.g. Llama 4, DeepSeek-V2).</summary>
    public bool HasSharedExpert { get; init; }

    /// <summary>Number of shared experts per MoE layer (e.g. 2 for DeepSeek-V2).</summary>
    public int NumSharedExperts { get; init; }

    /// <summary>Number of leading dense blocks before MoE layers begin (e.g. 1 for DeepSeek-V2).</summary>
    public int LeadingDenseBlockCount { get; init; }

    /// <summary>MLA (Multi-head Latent Attention) compressed KV rank (e.g. 512 for DeepSeek-V2).</summary>
    public int KvLoraRank { get; init; }

    /// <summary>
    /// MLA's per-head VALUE width ({arch}.attention.value_length), read separately from
    /// <see cref="HeadDim"/> (which is the Q/K width, {arch}.attention.key_length) because MLA's
    /// decompressed nope+rope Q/K width and its decompressed V width are independent GGUF keys --
    /// DeepSeek-V2-Lite ships key_length=192 (128 nope + 64 rope) and value_length=128, not equal
    /// by any structural necessity. 0 when absent (every non-MLA architecture).
    /// </summary>
    public int MlaVHeadDim { get; init; }

    /// <summary>
    /// YaRN RoPE scaling factor ({arch}.rope.scaling.factor, e.g. 40 for DeepSeek-V2-Lite).
    /// 1 (the default) means no scaling -- RopeYarnOrigCtxLen/RopeYarnLogMul are then unused.
    /// See ggml-cpu/ops.cpp rope_yarn() / rope_yarn_corr_dims() for the reference formula this
    /// mirrors (llama.cpp), and MlaAttention.cs / ForwardPass.cs's YaRN table builder for where
    /// it's applied. Only meaningful when {arch}.rope.scaling.type == "yarn".
    /// </summary>
    public float RopeYarnFactor { get; init; } = 1f;

    /// <summary>Original (pre-extension) training context length ({arch}.rope.scaling.original_context_length),
    /// used by the YaRN correction-range formula. Only meaningful when <see cref="RopeYarnFactor"/> &gt; 1.</summary>
    public int RopeYarnOrigCtxLen { get; init; }

    /// <summary>
    /// DeepSeek2-specific YaRN attention-score correction ({arch}.rope.scaling.yarn_log_multiplier).
    /// Unlike the rest of the YaRN parameters (which feed the RoPE cos/sin table itself), this one
    /// ONLY scales the attention softmax's kq_scale (see deepseek2.cpp's mscale/attn_factor_org
    /// derivation) -- 0 (the default) leaves kq_scale at the standard YaRN-adjusted value with no
    /// further correction.
    /// </summary>
    public float RopeYarnLogMul { get; init; }

    /// <summary>
    /// Whether MoE router top-k weights should be renormalized to sum to 1 after
    /// selecting the top-k experts. Most architectures (Qwen3-MoE, Mixtral) do.
    /// OLMoE was trained with <c>norm_topk_prob=false</c> and uses the raw
    /// post-softmax probabilities directly — renormalizing produces wrong outputs.
    /// </summary>
    public bool NormalizeMoeTopKWeights { get; init; } = true;

    /// <summary>
    /// Multiplier applied to every routed expert's weight after top-k selection (and after
    /// renormalization, if <see cref="NormalizeMoeTopKWeights"/> applies) — llama.cpp's
    /// <c>{arch}.expert_weights_scale</c> ("routed_scaling_factor" in the source HF config).
    /// 1 (no-op) for architectures that don't declare it. DeepSeek-V2/V3 declare a non-1 value
    /// on the larger checkpoints; applying only <see cref="NormalizeMoeTopKWeights"/> without
    /// this scale still produces a wrong overall MoE magnitude for those.
    /// </summary>
    public float ExpertWeightsScale { get; init; } = 1f;

    // ── NoPE (No Positional Encoding) ──

    /// <summary>
    /// Every Nth layer skips RoPE (NoPE). 0 = all layers use RoPE.
    /// Llama-4: step=4 → layers 3,7,11,... (0-indexed where (layer+1)%4==0) use NoPE.
    /// </summary>
    public int NoRopeLayerStep { get; init; }

    /// <summary>
    /// Whether the MoE router uses sigmoid gating instead of softmax (e.g. Llama-4).
    /// </summary>
    public bool UseSigmoidGating { get; init; }

    /// <summary>
    /// Whether QK-norm uses pure RMS norm (L2 normalize) without learned weights.
    /// Llama-4 uses Llama4TextL2Norm (pure RMS norm); Qwen3 uses weighted RMS norm.
    /// </summary>
    public bool UseL2QkNorm { get; init; }

    /// <summary>
    /// Whether WEIGHTED QK-norm (a learned RMSNorm, not <see cref="UseL2QkNorm"/>'s pure L2) is
    /// applied AFTER RoPE rather than before. Qwen3's convention (the default this engine assumes
    /// when <see cref="HasQkNorm"/> is set) norms then rotates; Hunyuan-Dense's graph
    /// (<c>hunyuan-vl.cpp</c>, shared by the dense variant) rotates Q/K first and applies its
    /// learned <c>attn_q_norm</c>/<c>attn_k_norm</c> to the ALREADY-ROTATED vectors — a timing
    /// this engine had no combination for until Hunyuan-Dense: Llama-4's after-RoPE case
    /// (<see cref="UseL2QkNorm"/>) is unweighted, and every weighted case before this one normed
    /// before RoPE.
    /// </summary>
    public bool QkNormAfterRope { get; init; }

    /// <summary>
    /// True for NEOX-style RoPE (rotates dim pairs (i, i + headDim/2)).
    /// False for LLaMA-style "normal" RoPE (rotates consecutive pairs (2i, 2i+1)).
    /// Qwen2/Qwen3, Phi, Gemma, Falcon, and most non-LLaMA architectures use NEOX.
    /// LLaMA, Mistral, SmolLM, Granite, and DeepSeek use the interleaved convention.
    /// </summary>
    public bool IsNeoxRope { get; init; }

    /// <summary>
    /// Number of head dims that receive RoPE rotation. Default equals HeadDim (full RoPE).
    /// Some architectures (notably qwen35moe) use partial RoPE where only the first
    /// <see cref="RopeDim"/> dims of each head are rotated and the rest pass through.
    /// </summary>
    public int RopeDim { get; init; }

    // ── Hybrid Gated DeltaNet + Attention (qwen35moe) ──

    /// <summary>
    /// True for hybrid models whose trunk interleaves recurrent (Gated DeltaNet / SSM-named)
    /// blocks with full softmax-attention blocks. Drives layer-by-layer dispatch.
    /// </summary>
    public bool IsHybridSsm { get; init; }

    /// <summary>
    /// Per-layer block type. <c>null</c> for non-hybrid models (every layer is Attention).
    /// Indexed by absolute layer number (0..NumLayers-1).
    /// </summary>
    public IReadOnlyList<LayerType>? LayerTypes { get; init; }

    /// <summary>
    /// Gated DeltaNet configuration. Non-null iff <see cref="IsHybridSsm"/> is true.
    /// Holds per-head dims, group count, conv kernel, and rank — the parameters of the
    /// recurrent block. Despite the GGUF prefix <c>ssm.*</c>, the math is delta-rule
    /// linear attention with a 2D matrix state per head, NOT Mamba selective scan.
    /// </summary>
    public GdnConfig? Gdn { get; init; }

    /// <summary>
    /// Number of Multi-Token Prediction (MTP) head layers stored at the end of the GGUF
    /// block stack. Read from <c>{arch}.nextn_predict_layers</c> (default 0 when absent).
    /// On disk these live at block indices <c>NumLayers..NumLayers+NumMtpLayers-1</c> —
    /// <see cref="NumLayers"/> already excludes them so the main forward loop stays clean.
    /// Used by MTP self-speculative decoding to draft N-ahead tokens per main forward.
    /// </summary>
    public int NumMtpLayers { get; init; }

    // ── Gemma 4 (sliding-window + per-layer head-dim + PLE) ──

    /// <summary>
    /// Sliding window size (in tokens) used by SWA layers. 0 when the model has
    /// no sliding-window attention. Gemma 4 = 512.
    /// </summary>
    public int SlidingWindowSize { get; init; }

    /// <summary>
    /// Whether RoPE is applied ONLY on SWA (local) layers, with global layers getting NO
    /// rotary embedding at all — Command-R's (cohere2) convention, confirmed against
    /// <c>src/models/cohere2.cpp</c>: its attention block calls <c>ggml_rope_ext</c> only inside
    /// <c>if (is_swa) { ... }</c>, with no <c>else</c> branch for global layers. This is the
    /// opposite selection rule from Llama-4/SmolLM3's period-based <see cref="NoRopeLayerStep"/>
    /// (which is unconditional on layer index, not on SWA-ness), so it's a separate flag rather
    /// than a variant of that mechanism. Hardcoded per architecture, like
    /// <see cref="UseParallelResidual"/>'s Falcon/cohere2 case — llama.cpp bakes this into the
    /// graph rather than reading it from GGUF metadata.
    /// </summary>
    public bool RopeOnlySwaLayers { get; init; }

    /// <summary>
    /// Per-Layer-Embedding (PLE) projection width. Gemma 4 E4B = 256. 0 when the
    /// model has no PLE table.
    /// </summary>
    public int PerLayerEmbeddingWidth { get; init; }

    /// <summary>Whether each layer has a post-attention RMSNorm before residual add (Gemma 4).</summary>
    public bool HasPostAttnNorm { get; init; }

    /// <summary>Whether each layer has a post-FFN RMSNorm before residual add (Gemma 4).</summary>
    public bool HasPostFfwNorm { get; init; }

    /// <summary>Whether the model carries a <c>per_layer_token_embd.weight</c> table (Gemma 4 PLE).</summary>
    public bool HasPerLayerTokenEmbd { get; init; }

    /// <summary>
    /// Whether each layer has a learned <c>layer_output_scale.weight</c> scalar applied
    /// to the layer output (Gemma 4).
    /// </summary>
    public bool HasLayerOutputScale { get; init; }

    /// <summary>
    /// FFN activation function. <see cref="FfnActivation.Silu"/> for the vast majority of
    /// architectures (LLaMA/Mistral/Qwen/etc); <see cref="FfnActivation.GeluApprox"/> for Gemma 4.
    /// </summary>
    public FfnActivation FfnActivation { get; init; } = FfnActivation.Silu;

    /// <summary>
    /// Per-layer xIELU parameters (Apertus). Non-null iff the model's FFN is non-gated with xIELU
    /// activation — detected at load time from tensor inventory (no <c>ffn_gate</c> weight) rather
    /// than from architecture string, matching the pattern <c>HasAttnBias</c> etc. use. All four
    /// arrays are always non-null together (Apertus always declares all four GGUF keys) or all
    /// null. See <c>SimdKernels.XieluInPlace</c> for the formula and the llama.cpp reference.
    /// </summary>
    public IReadOnlyList<float>? XieluAlphaN { get; init; }
    public IReadOnlyList<float>? XieluAlphaP { get; init; }
    public IReadOnlyList<float>? XieluBeta { get; init; }
    public IReadOnlyList<float>? XieluEps { get; init; }

    /// <summary>
    /// Whether the non-gated FFN (no <c>ffn_gate</c> tensor) uses ReLU-squared
    /// (<c>max(0,x)^2</c>, llama.cpp's <c>LLM_FFN_RELU_SQR</c>) instead of GELU. JAIS-2 is the
    /// only known user (confirmed against <c>src/models/jais2.cpp</c>: <c>build_ffn(..., NULL,
    /// ..., LLM_FFN_RELU_SQR, LLM_FFN_SEQ, il)</c>, biased up/down like GPT-NeoX's GELU shape) —
    /// not detectable from tensor inventory alone, since a biased non-gated FFN looks identical on
    /// disk whether it means "GELU" or "ReLU-squared"; needs the arch-string check the same way
    /// <see cref="XieluAlphaN"/>'s absence-vs-GELU default already relies on <em>xielu.*</em>
    /// metadata presence rather than a tensor-shape signal.
    /// </summary>
    public bool UsesReluSquared { get; init; }

    /// <summary>
    /// Per-layer flag: <c>true</c> when the layer uses sliding-window attention,
    /// <c>false</c> for global (full-context) attention. <c>null</c> when every
    /// layer is global. Gemma 4 follows a 5-SWA : 1-global repeating pattern.
    /// </summary>
    public IReadOnlyList<bool>? IsSwaLayer { get; init; }

    /// <summary>
    /// Per-layer KV-cache source. <c>-1</c> means the layer owns its own K/V;
    /// otherwise the layer aliases another layer's KV pages (Gemma 4
    /// <c>shared_kv_layers</c> tail). <c>null</c> when no layer shares KV.
    /// </summary>
    public IReadOnlyList<int>? KvSourceLayer { get; init; }

    /// <summary>
    /// Per-layer attention head dimension (Gemma 4 mixes 256 for SWA and 512 for
    /// global). <c>null</c> when every layer uses <see cref="HeadDim"/>.
    /// </summary>
    public IReadOnlyList<int>? LayerHeadDim { get; init; }

    /// <summary>
    /// Per-layer RoPE rotation dimension (Gemma 4 mixes 256 for SWA and 512 for
    /// global). <c>null</c> when every layer uses <see cref="RopeDim"/>.
    /// </summary>
    public IReadOnlyList<int>? LayerRopeDim { get; init; }

    /// <summary>
    /// Per-layer KV head count. Gemma 4 12B (dense) mixes 8 (GQA) on SWA layers
    /// and 1 (MQA) on global layers; stored in the GGUF as a per-layer
    /// <c>attention.head_count_kv</c> array. <c>null</c> when every layer uses the
    /// scalar <see cref="NumKvHeads"/>.
    /// </summary>
    public IReadOnlyList<int>? LayerKvHeads { get; init; }

    /// <summary>
    /// When <c>true</c>, attention reuses the K projection as V on the layers that
    /// omit a <c>attn_v.weight</c> tensor (Gemma 4 12B global layers,
    /// <c>attention_k_eq_v=true</c> in the HF config). Such layers carry no V
    /// projection; the K output doubles as the value stream. <c>false</c> for the
    /// usual separate-K/V layout.
    /// </summary>
    public bool AttentionKEqV { get; init; }

    /// <summary>
    /// Extract hyperparameters from GGUF metadata using the model's architecture prefix.
    /// Supports llama-family models (llama, mistral, qwen, smollm, etc.) and MoE variants.
    /// </summary>
    public static ModelHyperparams FromGgufMetadata(IReadOnlyDictionary<string, object> metadata)
        => FromGgufMetadata(metadata, null);

    public static ModelHyperparams FromGgufMetadata(IReadOnlyDictionary<string, object> metadata,
        IModelTensorSource? tensorSource)
    {
        var arch = metadata.TryGetValue("general.architecture", out var a) ? (string)a : "llama";

        int numExperts = GetInt(metadata, $"{arch}.expert_count");
        int numActiveExperts = GetInt(metadata, $"{arch}.expert_used_count");
        bool isMoE = numExperts > 0;

        // Detect features by probing tensor names
        bool hasAttnBias = metadata.ContainsKey("_opentailllm.has_attn_bias")
            || (tensorSource?.FindTensor("blk.0.attn_q.bias") is not null)
            // GPT-NeoX/Pythia ships a fused blk.N.attn_qkv.bias rather than separate
            // attn_q/attn_k/attn_v biases — see ForwardPass's fused-QKV constructor path.
            || (tensorSource?.FindTensor("blk.0.attn_qkv.bias") is not null);
        // The output-projection bias is optional even when Q/K/V bias is present (Qwen2 omits it).
        bool hasAttnOutputBias = hasAttnBias
            && (metadata.ContainsKey("_opentailllm.has_attn_output_bias")
                || (tensorSource?.FindTensor("blk.0.attn_output.bias") is not null));
        bool hasNormBias = metadata.ContainsKey("_opentailllm.has_norm_bias")
            || (tensorSource?.FindTensor("blk.0.attn_norm.bias") is not null);
        bool hasFfnBias = metadata.ContainsKey("_opentailllm.has_ffn_bias")
            || (tensorSource?.FindTensor("blk.0.ffn_up.bias") is not null);
        bool hasQkNorm = metadata.ContainsKey("_opentailllm.has_qk_norm")
            || (tensorSource?.FindTensor("blk.0.attn_q_norm.weight") is not null);
        bool perChannelQkNorm = false;
        if (hasQkNorm && tensorSource is not null)
        {
            var qNormInfo = tensorSource.FindTensor("blk.0.attn_q_norm.weight");
            int numHeadsTmp = GetInt(metadata, $"{arch}.attention.head_count");
            int embDimTmp = GetInt(metadata, $"{arch}.embedding_length");
            int headDimMetaTmp = GetInt(metadata, $"{arch}.attention.key_length");
            int headDimTmp = headDimMetaTmp > 0 ? headDimMetaTmp
                : (numHeadsTmp > 0 ? embDimTmp / numHeadsTmp : embDimTmp);
            if (qNormInfo is not null && headDimTmp > 0 && numHeadsTmp > 0)
                perChannelQkNorm = qNormInfo.Value.ElementCount >= (long)numHeadsTmp * headDimTmp;
        }
        bool hasSharedExpert = isMoE
            && (GetInt(metadata, $"{arch}.expert_shared_count") > 0
                || (tensorSource?.FindTensor("blk.0.ffn_gate_shexp.weight") is not null)
                || (tensorSource?.FindTensor("blk.1.ffn_gate_shexp.weight") is not null));

        // NoPE: every Nth layer skips RoPE entirely. Hardcoded in llama.cpp rather than stored in
        // GGUF metadata, for both architectures that use it — Llama-4 (`llama.cpp` sets
        // n_no_rope_layer_step = 4) and SmolLM3 (`models/smollm3.cpp` does the same). The gate
        // below, `(layer + 1) % step != 0`, is the same expression llama.cpp applies.
        bool isLlama4 = arch.Equals("llama4", StringComparison.OrdinalIgnoreCase);
        bool isSmolLm3 = arch.Equals("smollm3", StringComparison.OrdinalIgnoreCase);
        bool usesReluSquared = arch == "jais2";
        // GPT-2 has no RoPE at all — position is encoded once via a learned absolute
        // position-embedding table (ModelHyperparams consumers detect this from
        // `position_embd.weight`'s tensor presence directly, not a hyperparam flag) added to the
        // token embedding before the trunk starts. Step=1 makes `(layer+1) % step != 0` false for
        // every layer, so useRoPE computes to false unconditionally via the SAME formula
        // Llama-4/SmolLM3 already use for their periodic skip — no new dispatch code needed.
        // StarCoder (v1) shares GPT-2's exact absolute-position-embedding shape (confirmed against
        // src/models/starcoder.cpp: same ggml_get_rows(pos_embd, inp_pos) pattern, no RoPE call
        // anywhere in the graph).
        bool isGpt2Family = arch is "gpt2" or "starcoder";
        int noRopeStep = isLlama4 || isSmolLm3 ? 4 : isGpt2Family ? 1 : 0;
        // Llama-4 uses sigmoid gating with weight-before-FFN per Meta's reference impl.
        bool useSigmoidGating = isLlama4;
        // Llama-4 uses Llama4TextL2Norm for QK-norm: pure RMS norm without learned weights.
        // No attn_q_norm.weight tensor exists, so force hasQkNorm for Llama-4.
        bool useL2QkNorm = isLlama4;
        if (isLlama4) hasQkNorm = true;

        // Hunyuan-Dense's graph (src/models/hunyuan-vl.cpp, shared by the dense variant) applies
        // its WEIGHTED attn_q_norm/attn_k_norm AFTER ggml_rope_ext, not before like every other
        // weighted-QK-norm architecture this engine has (Qwen3, OLMoE). Distinct from Llama-4's
        // UseL2QkNorm (also after-RoPE, but unweighted).
        bool qkNormAfterRope = arch == "hunyuan-dense";

        // RoPE convention: NEOX (pairs offset by headDim/2) vs NORM/interleaved (consecutive pairs).
        // Mirrors llama.cpp's llama_model_rope_type() in src/llama-model.cpp (NEOX block).
        // Architectures NOT listed here default to NORM (LLaMA-style interleaved).
        // Special rope types (MROPE for QWEN2VL/PADDLEOCR, IMROPE for QWEN3VL family, conditional
        // for GLM4/GLM4_MOE) are not currently supported and would need their own dispatch.
        bool isNeoxRope = arch switch
        {
            "falcon" or "falcon-h1" or "grok" or "dbrx" or
            "bert" or "jina-bert-v3" or "modern-bert" or "nomic-bert" or "nomic-bert-moe" or "eurobert" or
            "stablelm" or "bitnet" or
            "qwen" or "qwen2" or "dream" or "qwen2moe" or "qwen3" or "qwen3moe" or "qwen3-tts" or
            "llada-moe" or "rnd1" or
            "olmo2" or "olmoe" or
            "phi2" or "phi3" or "phimoe" or
            "plamo" or "plamo2" or "plamo3" or
            "gemma" or "gemma2" or "gemma3" or "gemma3n" or "gemma4" or "gemma-embedding" or
            "starcoder2" or "openelm" or "gptneox" or "codeshell" or "orion" or
            "nemotron" or "exaone" or "exaone4" or "exaone-moe" or
            "minicpm3" or "bailingmoe2" or "dots1" or
            "hunyuan-moe" or "hunyuan-dense" or
            "jais2" or "gpt-oss" or
            "lfm2" or "lfm2moe" or "smallthinker" or "seed_oss" or "grovemoe" or
            "apertus" or "minimax-m2" or "cogvlm" or "pangu-embedded" or "afmoe" or
            "qwen3next" or "qwen35moe" or "qwen35" or "mimo2" or "step35" => true,
            _ => false,
        };

        int embDim = GetInt(metadata, $"{arch}.embedding_length");
        int numHeads = GetInt(metadata, $"{arch}.attention.head_count");
        // Some models (e.g. Qwen3-MoE) use a head dim that differs from embDim/numHeads.
        // Read from metadata if available; fall back to computed value.
        int headDimFromMeta = GetInt(metadata, $"{arch}.attention.key_length");
        int headDim = headDimFromMeta > 0 ? headDimFromMeta : (numHeads > 0 ? embDim / numHeads : embDim);

        // Partial RoPE: rope.dimension_count, when present and smaller than headDim,
        // rotates only the first ropeDim dims of each head. qwen35moe rotates 64 of 256.
        int ropeDimFromMeta = GetInt(metadata, $"{arch}.rope.dimension_count");
        int ropeDim = ropeDimFromMeta > 0 ? ropeDimFromMeta : headDim;

        // Hybrid Gated-DeltaNet detection. qwen35moe (and similar future architectures)
        // interleave recurrent and attention blocks. We rely on metadata exclusively here;
        // the synthetic-metadata probe in GgufModel.Open injects _opentailllm.is_hybrid_ssm
        // when GDN tensors are observed.
        bool isHybridSsm = metadata.ContainsKey("_opentailllm.is_hybrid_ssm")
                        || arch == "qwen35moe";

        // {arch}.block_count is the total block count in the file, which on MTP-enabled
        // models (qwen35 27B-MTP, qwen35moe-MTP) includes the MTP head blocks appended
        // after the main layers. Strip them so NumLayers reflects only the main model;
        // MTP blocks are loaded separately by the MTP head logic.
        int totalBlocks = GetInt(metadata, $"{arch}.block_count");
        int numMtpLayers = GetInt(metadata, $"{arch}.nextn_predict_layers", 0);
        int numLayers = totalBlocks - numMtpLayers;

        IReadOnlyList<LayerType>? layerTypes = null;
        GdnConfig? gdn = null;
        if (isHybridSsm && numLayers > 0)
        {
            int fullAttnInterval = GetInt(metadata, $"{arch}.full_attention_interval", 4);
            var types = new LayerType[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                // qwen35moe: full attention when (i+1) % full_attention_interval == 0.
                // i.e. the LAST layer of each group of full_attention_interval is full attn.
                bool isFullAttn = fullAttnInterval > 0 && ((i + 1) % fullAttnInterval) == 0;
                types[i] = isFullAttn ? LayerType.Attention : LayerType.GatedDeltaNet;
            }
            layerTypes = types;

            gdn = new GdnConfig(
                NumKHeads:    GetInt(metadata, $"{arch}.ssm.group_count"),
                NumVHeads:    GetInt(metadata, $"{arch}.ssm.time_step_rank"),
                HeadDim:      GetInt(metadata, $"{arch}.ssm.state_size"),
                InnerSize:    GetInt(metadata, $"{arch}.ssm.inner_size"),
                ConvKernel:   GetInt(metadata, $"{arch}.ssm.conv_kernel"),
                FullAttentionInterval: fullAttnInterval);
        }

        bool isGemma4 = arch.Equals("gemma4", StringComparison.OrdinalIgnoreCase);

        int slidingWindow = 0;
        int perLayerEmbedWidth = 0;
        float finalLogitSoftcap = 0f;
        float ropeThetaSwa = 0f;
        float embeddingScale = 1f;
        float residualScale = 1f;
        float attentionScaleOverride = 0f;
        float logitScale = 1f;
        bool hasPerLayerTokenEmbd = false;
        bool hasLayerOutputScale = false;
        FfnActivation ffnActivation = FfnActivation.Silu;
        IReadOnlyList<bool>? isSwaLayer = null;
        IReadOnlyList<int>? kvSourceLayer = null;
        IReadOnlyList<int>? layerHeadDim = null;
        IReadOnlyList<int>? layerRopeDim = null;
        IReadOnlyList<int>? layerKvHeads = null;
        bool attentionKEqV = false;

        // Post-sublayer norm (weight applied to the sublayer's OUTPUT, immediately before the
        // residual add) — tensor-presence detected generically rather than gated to any one
        // architecture. Gemma 4 combines this with an ordinary pre-norm (sandwich norm). OLMo2
        // has NO pre-norm tensor at all (attn_norm/ffn_norm are absent from its GGUF) and uses
        // ONLY this post-norm, reusing the exact same llama.cpp tensor names/roles
        // (LLM_TENSOR_ATTN_POST_NORM/FFN_POST_NORM both map to blk.%d.post_attention_norm /
        // blk.%d.post_ffw_norm for both architectures) — see src/models/olmo2.cpp vs gemma4.cpp.
        bool hasPostAttnNorm = metadata.ContainsKey("_opentailllm.has_post_attn_norm")
            || (tensorSource?.FindTensor("blk.0.post_attention_norm.weight") is not null);
        bool hasPostFfwNorm = metadata.ContainsKey("_opentailllm.has_post_ffw_norm")
            || (tensorSource?.FindTensor("blk.0.post_ffw_norm.weight") is not null);

        if (isGemma4)
        {
            slidingWindow         = GetInt(metadata, $"{arch}.attention.sliding_window");
            perLayerEmbedWidth    = GetInt(metadata, $"{arch}.embedding_length_per_layer_input");
            finalLogitSoftcap     = GetFloat(metadata, $"{arch}.final_logit_softcapping");
            ropeThetaSwa          = GetFloat(metadata, $"{arch}.rope.freq_base_swa", 10_000f);
            int sharedKvLayers    = GetInt(metadata, $"{arch}.attention.shared_kv_layers");
            int keyLengthSwa      = GetInt(metadata, $"{arch}.attention.key_length_swa", headDim);
            int ropeDimSwa        = GetInt(metadata, $"{arch}.rope.dimension_count_swa", keyLengthSwa);

            embeddingScale = MathF.Sqrt(embDim);
            ffnActivation  = FfnActivation.GeluApprox;

            hasPerLayerTokenEmbd = metadata.ContainsKey("_opentailllm.has_ple")
                || (tensorSource?.FindTensor("per_layer_token_embd.weight") is not null);
            hasLayerOutputScale  = metadata.ContainsKey("_opentailllm.has_layer_output_scale")
                || (tensorSource?.FindTensor("blk.0.layer_output_scale.weight") is not null);

            // Gemma 4 12B (dense) global layers omit attn_v and reuse K as V
            // (attention_k_eq_v=true in the HF config; not a GGUF metadata key, so it
            // is detected from the tensor inventory via a GgufModel.Open probe).
            attentionKEqV = metadata.ContainsKey("_opentailllm.attention_k_eq_v");

            if (numLayers > 0)
            {
                var pattern = GetBoolArray(metadata, $"{arch}.attention.sliding_window_pattern");
                var swa = new bool[numLayers];
                if (pattern is not null && pattern.Count > 0)
                {
                    for (int i = 0; i < numLayers; i++)
                        swa[i] = pattern[i % pattern.Count];
                }
                isSwaLayer = swa;

                var hdArr = new int[numLayers];
                var rdArr = new int[numLayers];
                for (int i = 0; i < numLayers; i++)
                {
                    bool sw = swa[i];
                    hdArr[i] = sw ? keyLengthSwa : headDim;
                    rdArr[i] = sw ? ropeDimSwa : ropeDim;
                }
                layerHeadDim = hdArr;
                layerRopeDim = rdArr;

                // Per-layer KV head count (Gemma 4 12B: 8 on SWA, 1 on global).
                // Stored as a per-layer array in the GGUF; build the full vector so
                // forward passes can size each layer's KV independently. Falls back
                // to the scalar head_count_kv (broadcast) when stored as a scalar.
                var kvArr = GetIntArray(metadata, $"{arch}.attention.head_count_kv");
                if (kvArr is not null && kvArr.Count > 0)
                {
                    var lkv = new int[numLayers];
                    for (int i = 0; i < numLayers; i++)
                    {
                        // Guard a corrupt/0 KV head count → it would divide-by-zero in the
                        // attention group-size calc (_numHeads / kvHeads) downstream.
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
                        if (i < firstSharedLayer)
                        {
                            src[i] = -1;
                        }
                        else
                        {
                            int found = -1;
                            for (int j = firstSharedLayer - 1; j >= 0; j--)
                            {
                                if (swa[j] == swa[i]) { found = j; break; }
                            }
                            src[i] = found;
                        }
                    }
                    kvSourceLayer = src;
                }
            }
        }
        else if (arch == "cohere2" && numLayers > 0)
        {
            // Command-R: sliding-window attention alternates every swaPeriod layers, the LAST
            // layer of each block being global — llama.cpp's set_swa_pattern(period) with its
            // default dense_first=false: is_swa[il] = period==0 || (il % period < period-1).
            // Confirmed against llama-hparams.cpp directly rather than assumed. The period is a
            // plain scalar in GGUF metadata (cohere2.cpp: ml.get_key_or_arr(..., swa_period=4,
            // false) — 4 even when the key is absent), NOT a literal per-layer bool array the
            // way Gemma 4's own sliding_window_pattern key is, so this does not reuse Gemma 4's
            // GetBoolArray-based cycling above.
            slidingWindow = GetInt(metadata, $"{arch}.attention.sliding_window");
            int swaPeriod = GetInt(metadata, $"{arch}.attention.sliding_window_pattern", 4);
            var swa = new bool[numLayers];
            for (int i = 0; i < numLayers; i++)
                swa[i] = swaPeriod == 0 || (i % swaPeriod < swaPeriod - 1);
            isSwaLayer = swa;
            // No separate SWA rope table is needed: cohere2 doesn't use a different frequency
            // for SWA vs global layers, it skips RoPE on global layers ENTIRELY (see
            // ModelHyperparams.RopeOnlySwaLayers) — ropeThetaSwa stays 0f (unset), so the
            // SWA-rope-table construction path (gated on RopeThetaSwa > 0f) never fires.
        }

        // Granite (dense/MoE/hybrid) and MiniCPM share ONE graph builder in llama.cpp
        // (models.h: "using graph = llama_model_granite::graph") — MiniCPM is Granite's
        // scale trio with different constants, not a different structure. All four keys
        // are read generically per-arch rather than globally: llama.cpp itself only calls
        // ml.get_key for these on this specific family (grok and command-r read
        // logit_scale independently, with a DIFFERENT sign convention — see
        // ModelHyperparams.LogitScale — and are not wired here).
        //
        // Deliberately EXCLUDES "minicpm3": despite the name it is a different
        // architecture, not a MiniCPM variant — models/minicpm3.cpp declares
        // ATTENTION_Q_LORA_RANK / ATTENTION_KV_LORA_RANK and builds Multi-head Latent
        // Attention, the same mechanism as deepseek2, not Granite's dense/GQA attention.
        // Routing it through this branch would silently misapply the scale trio to an
        // architecture that needs MLA kernels first — new-kernel work, not this.
        //
        // GGUF's convention is "0 / absent = off"; ModelHyperparams' fields are
        // multiplicative identities ("1 = off"), so an absent or exactly-zero key must
        // fall through to 1, not 0 — multiplying embeddings or a residual by a literal 0
        // would zero the trunk instead of leaving it alone.
        bool isMiniCpm = arch.Equals("minicpm", StringComparison.OrdinalIgnoreCase);
        bool isGraniteFamily = arch.Equals("granite", StringComparison.OrdinalIgnoreCase)
            || arch.Equals("granitemoe", StringComparison.OrdinalIgnoreCase)
            || arch.Equals("granitehybrid", StringComparison.OrdinalIgnoreCase)
            || isMiniCpm;
        if (isGraniteFamily)
        {
            // MiniCPM ships older GGUFs that omit these keys entirely and rely on
            // llama.cpp hardcoding formula-based defaults (models/minicpm.cpp) BEFORE
            // checking for an explicit override — unlike Granite, which has no
            // architecture-level default and simply leaves scaling off when absent.
            // Applying these unconditionally for non-MiniCPM archs would be wrong (they
            // have no such fallback), so they are gated on isMiniCpm specifically.
            if (isMiniCpm)
            {
                embeddingScale = 12.0f;
                residualScale = numLayers > 0 ? 1.4f / MathF.Sqrt(numLayers) : 1f;
                logitScale = embDim > 0 ? 256.0f / embDim : 1f;
            }

            float rawEmbeddingScale = GetFloat(metadata, $"{arch}.embedding_scale");
            if (rawEmbeddingScale != 0f) embeddingScale = rawEmbeddingScale;

            float rawResidualScale = GetFloat(metadata, $"{arch}.residual_scale");
            if (rawResidualScale != 0f) residualScale = rawResidualScale;

            // MiniCPM's load_arch_hparams never calls ml.get_key for attention.scale at
            // all (only Granite does) — so even if a MiniCPM GGUF happened to carry that
            // key, llama.cpp would ignore it. Mirror that exactly rather than reading it
            // generically for the whole family.
            if (!isMiniCpm)
            {
                // kq_scale override: 0 sentinel means "no override — caller falls back to
                // 1/sqrt(HeadDim)". Granite 3.3 2B declares 0.015625 (1/64), NOT
                // 1/sqrt(64) (0.125) — a real per-architecture override, not a rounding
                // of the usual formula.
                attentionScaleOverride = GetFloat(metadata, $"{arch}.attention.scale");
            }

            // llama.cpp's granite.cpp DIVIDES by the raw metadata value
            // (ggml_scale(cur, 1.0f / f_logit_scale)); LogitScale is documented as
            // already carrying that reciprocal, so bake the division in here rather
            // than at every call site.
            float rawLogitScale = GetFloat(metadata, $"{arch}.logit_scale");
            if (rawLogitScale != 0f) logitScale = 1f / rawLogitScale;
        }

        // DeepSeek2 (MLA) YaRN attention-score correction (src/models/deepseek2.cpp): YaRN's
        // own magnitude scaling (baked into the RoPE cos/sin table itself, see the YaRN table
        // builder ForwardPass.cs constructs) uses the ambient attn_factor. This is a SEPARATE,
        // architecture-specific second correction applied only to the attention softmax scale
        // (kq_scale):
        //   attn_factor_org = attn_factor * (1 + 0.1*log(factor))
        //   mscale = attn_factor_org * (1 + 0.1*RopeYarnLogMul*log(factor))
        //   kq_scale = mscale^2 / sqrt(n_embd_head_k_mla)
        // where factor = RopeYarnFactor and "attn_factor" is NOT a constant 1 -- it's
        // llama-context.cpp's cparams.yarn_attn_factor, which that file deliberately PRE-DIVIDES
        // by (1 + 0.1*log(factor)) ("cancel this factor" -- discussions/7416, PR #17945)
        // specifically so deepseek2.cpp's attn_factor_org multiplication cancels it back out.
        // The pre-division and re-multiplication always cancel exactly, so attn_factor_org
        // reduces to the PRE-cancellation value: 1 when RopeYarnLogMul != 0 (llama-context.cpp's
        // DEEPSEEK2 special case forces get_mscale(factor,mscale)/get_mscale(factor,mscale)'s
        // ratio to 1 regardless of the value), else get_mscale(factor, 1) = 1 + 0.1*log(factor).
        // With RopeYarnFactor==1 (no YaRN) this all collapses to the ordinary 1/sqrt(HeadDim)
        // every other architecture gets from AttentionScaleOverride's 0 sentinel -- so it's safe
        // to compute unconditionally for any KvLoraRank>0 (MLA) model, not just YaRN ones.
        bool isMla = GetInt(metadata, $"{arch}.attention.kv_lora_rank", 0) > 0;
        if (isMla && headDim > 0)
        {
            float freqScale = GetFloat(metadata, $"{arch}.rope.scaling.factor", 1f) is var yf && yf > 0f ? 1f / yf : 1f;
            float factor = 1f / freqScale;
            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX] llama.cpp's load_arch_hparams divides the raw
            // GGUF value by 0.1f right after reading it ("cancel the factor from the convert
            // script") -- every downstream consumer (this formula and llama-context.cpp's
            // ambient yarn_attn_factor) uses that ADJUSTED value, not the raw metadata one.
            float logMul = GetFloat(metadata, $"{arch}.rope.scaling.yarn_log_multiplier", 0f) / 0.1f;
            float attnFactorOrg = logMul != 0f ? 1f : (factor <= 1f ? 1f : 0.1f * MathF.Log(factor) + 1f);
            float mscale = attnFactorOrg * (1f + 0.1f * logMul * MathF.Log(factor));
            attentionScaleOverride = mscale * mscale / MathF.Sqrt(headDim);
        }

        // xIELU non-gated FFN (Apertus): detected from tensor inventory (no ffn_gate weight),
        // same style as HasAttnBias/HasQkNorm — not gated on architecture string, since the
        // xielu.* GGUF keys are declared without an {arch}. prefix in llama.cpp's own key table
        // (llama-arch.cpp: "xielu.alpha_n" etc, unlike almost every other hparam key).
        IReadOnlyList<float>? xieluAlphaN = null, xieluAlphaP = null, xieluBeta = null, xieluEps = null;
        bool hasFfnGate = tensorSource?.FindTensor("blk.0.ffn_gate.weight") is not null;
        if (!hasFfnGate && numLayers > 0 && metadata.ContainsKey("xielu.alpha_n"))
        {
            var rawAlphaN = GetFloatArray(metadata, "xielu.alpha_n", numLayers)!;
            var rawAlphaP = GetFloatArray(metadata, "xielu.alpha_p", numLayers)!;
            xieluBeta = GetFloatArray(metadata, "xielu.beta", numLayers);
            xieluEps  = GetFloatArray(metadata, "xielu.eps", numLayers);

            // GGUF stores the RAW (pre-softplus) parameters — the reparametrization that keeps
            // them positive during training. ggml's actual compute kernel (op_xielu) takes them
            // already-transformed; the transform lives one layer up, in the ggml_xielu() graph
            // wrapper (ggml.c), NOT in load_arch_hparams or the kernel itself — easy to miss by
            // reading only the kernel, which is exactly how this was first gotten wrong (produced
            // fluent-looking garbage, not an error, on the first attempt at this receipt).
            // softplus(x) = x>20 ? x : log(1+exp(x)); effective_alpha_n = beta + softplus(raw_n),
            // effective_alpha_p = softplus(raw_p).
            static float Softplus(float x) => x > 20f ? x : MathF.Log(1f + MathF.Exp(x));
            var alphaN = new float[numLayers];
            var alphaP = new float[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                float beta = xieluBeta is { } b ? b[i] : 0f;
                alphaN[i] = beta + Softplus(rawAlphaN[i]);
                alphaP[i] = Softplus(rawAlphaP[i]);
            }
            xieluAlphaN = alphaN;
            xieluAlphaP = alphaP;
        }

        if (xieluAlphaN is not null && hasFfnBias)
        {
            // Both are non-gated FFN activations, but wire to mutually exclusive tensor sets
            // (xIELU: unbiased up/down + xielu.* params; GPT-NeoX GELU: biased up/down, no
            // xielu.* params). A GGUF advertising both would mean the tensor inventory doesn't
            // match either known non-gated architecture — refuse rather than guess.
            throw new NotSupportedException(
                "A non-gated FFN cannot be both xIELU and biased-GELU; refusing an ambiguous tensor layout.");
        }

        // GPT-NeoX/Pythia parallel-residual flag (llama.cpp LLM_KV_USE_PARALLEL_RESIDUAL).
        // The converter always writes this key for gptneox (defaulting the HF config's own
        // absence to true), so the GetBool fallback below only matters for a hand-edited GGUF.
        // Falcon has no such metadata key at all — its graph (src/models/falcon.cpp) hardcodes
        // the same 3-way residual sum unconditionally, so it's hardcoded here to match rather
        // than read from a key that was never written.
        //
        // StableLM is the opposite trap: its GGUF carries a `stablelm.use_parallel_residual` key
        // (the converter copies it straight from the HF config), but src/models/stablelm.cpp's
        // graph builder never reads that hparam at all — it branches on whether the per-layer
        // `ffn_norm` TENSOR exists (`if (layer.ffn_norm) sequential; else cur = inpSA` — reuse the
        // attn-norm output as the FFN input). StableLM-2-1.6B ships `use_parallel_residual: true`
        // in metadata while every layer still carries a real `ffn_norm` weight+bias, so trusting
        // the metadata key here would wrongly take the parallel path; only the 12B variant (no
        // `ffn_norm` tensor) is genuinely parallel-residual.
        bool stablelmHasFfnNorm = tensorSource?.FindTensor("blk.0.ffn_norm.weight") is not null;
        bool useParallelResidual = arch is "falcon" or "cohere2"
            || (arch == "stablelm" ? !stablelmHasFfnNorm : GetBool(metadata, $"{arch}.use_parallel_residual"));

        // Command-R (cohere2) uses true LayerNorm (build_norm's LLM_NORM, not LLM_NORM_RMS) but
        // ships no bias tensor at all — hasNormBias (tensor-presence) stays false, so this needs
        // an explicit arch-string check the same way useParallelResidual does above; a GGUF's
        // tensor inventory alone can't distinguish "RMSNorm with a weight tensor" from
        // "bias-less LayerNorm with a weight tensor", since both look identical on disk.
        bool usesLayerNorm = hasNormBias || arch == "cohere2";
        bool ropeOnlySwaLayers = arch == "cohere2";

        // OLMo v1 ships no attn_norm/ffn_norm/output_norm tensor at all (confirmed against
        // src/models/olmo.cpp: build_norm's weight AND bias arguments are both NULL) — the SAME
        // tensor-absence signal OLMo2 uses for "no pre-norm, sandwich-normed instead", so this
        // needs its own arch-string check to mean "normalize anyway, just with no parameters" as
        // opposed to OLMo2's "skip normalizing here entirely".
        bool usesUnweightedNorm = arch == "olmo";

        // Command-R (cohere2) reads logit_scale independently of the Granite-family block above
        // (isGraniteFamily deliberately doesn't cover it) and applies it with the OPPOSITE
        // convention: cohere2.cpp does `ggml_scale(cur, f_logit_scale)` — a direct multiply —
        // while granite.cpp does `ggml_scale(cur, 1.0f / f_logit_scale)`. LogitScale is documented
        // as already carrying whatever reciprocal is needed so every call site can just multiply,
        // so this reads the raw value straight through, not inverted.
        if (arch == "cohere2")
        {
            float rawLogitScaleC2 = GetFloat(metadata, $"{arch}.logit_scale");
            if (rawLogitScaleC2 != 0f) logitScale = rawLogitScaleC2;
        }

        return new ModelHyperparams
        {
            VocabSize = GetInt(metadata, $"{arch}.vocab_size"),
            ContextLength = GetInt(metadata, $"{arch}.context_length"),
            EmbeddingDim = embDim,
            NumLayers = numLayers,
            NumMtpLayers = numMtpLayers,
            NumHeads = numHeads,
            NumKvHeads = GetInt(metadata, $"{arch}.attention.head_count_kv",
                            GetInt(metadata, $"{arch}.attention.head_count")),
            IntermediateDim = GetInt(metadata, $"{arch}.feed_forward_length"),
            HeadDim = headDim,
            // GPT-NeoX stores LayerNorm epsilon under attention.layer_norm_epsilon (the LayerNorm
            // key), not attention.layer_norm_rms_epsilon (the RMSNorm key other architectures use)
            // — the first GetFloat returns its 0f default and falls through for gptneox.
            RmsNormEps = GetFloat(metadata, $"{arch}.attention.layer_norm_rms_epsilon",
                GetFloat(metadata, $"{arch}.attention.layer_norm_epsilon", 1e-5f)),
            RopeTheta = GetFloat(metadata, $"{arch}.rope.freq_base", 10_000f),
            HasAttnBias = hasAttnBias,
            HasAttnOutputBias = hasAttnOutputBias,
            HasNormBias = hasNormBias,
            UsesLayerNorm = usesLayerNorm,
            UsesUnweightedNorm = usesUnweightedNorm,
            RopeOnlySwaLayers = ropeOnlySwaLayers,
            HasFfnBias = hasFfnBias,
            UseParallelResidual = useParallelResidual,
            HasQkNorm = hasQkNorm,
            IsPerChannelQkNorm = perChannelQkNorm,
            IsMoE = isMoE,
            NumExperts = numExperts,
            NumActiveExperts = numActiveExperts,
            ExpertIntermediateDim = GetInt(metadata, $"{arch}.expert_feed_forward_length",
                                       GetInt(metadata, $"{arch}.feed_forward_length")),
            HasSharedExpert = hasSharedExpert,
            NumSharedExperts = GetInt(metadata, $"{arch}.expert_shared_count", hasSharedExpert ? 1 : 0),
            LeadingDenseBlockCount = GetInt(metadata, $"{arch}.leading_dense_block_count", 0),
            KvLoraRank = GetInt(metadata, $"{arch}.attention.kv_lora_rank", 0),
            MlaVHeadDim = GetInt(metadata, $"{arch}.attention.value_length", 0),
            RopeYarnFactor = GetFloat(metadata, $"{arch}.rope.scaling.factor", 1f),
            RopeYarnOrigCtxLen = GetInt(metadata, $"{arch}.rope.scaling.original_context_length", 0),
            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX] see the matching comment above -- same /0.1f
            // correction llama.cpp applies at load time, not just in the local kq_scale calc.
            RopeYarnLogMul = GetFloat(metadata, $"{arch}.rope.scaling.yarn_log_multiplier", 0f) / 0.1f,
            // llama.cpp's per-architecture MoE loaders diverge on where this comes from:
            // qwen3moe.cpp and olmoe.cpp pass a LITERAL true/false to build_moe_ffn (never
            // read any GGUF key), while every other MoE arch that supports it (deepseek2,
            // bailingmoe, glm4-moe, cohere2moe, dots1, minimax-m3, nemotron-h, step35, ...)
            // calls ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM, hparams.expert_weights_norm, false)
            // -- an explicit FALSE default (see llama-hparams.h's own field default), not true.
            // Ground-truth-verified 2026-08-21 via llama-eval-callback against
            // DeepSeek-V2-Lite-Chat: the real graph has NO renormalization node between the
            // top-k softmax probs and the weighted expert combine, confirming false is correct
            // for this checkpoint (its GGUF has no expert_weights_norm key at all). The
            // previous version of this line defaulted to TRUE for every non-olmoe architecture
            // (including deepseek2) when the key was absent -- backwards from llama.cpp's own
            // default, and the reason this fix didn't unblock deepseek2 on first attempt.
            NormalizeMoeTopKWeights = arch.Equals("qwen3moe", StringComparison.OrdinalIgnoreCase) ? true
                : arch.Equals("olmoe", StringComparison.OrdinalIgnoreCase) ? false
                : GetBool(metadata, $"{arch}.expert_weights_norm", false),
            // llama.cpp's LLM_KV_EXPERT_WEIGHTS_SCALE ("routed_scaling_factor" in DeepSeek-V2/V3's
            // HF config) -- multiplies every routed expert's weight after top-k selection/
            // renormalization. 1 (no-op) when the GGUF doesn't declare it.
            ExpertWeightsScale = GetFloat(metadata, $"{arch}.expert_weights_scale", 1f),
            NoRopeLayerStep = noRopeStep,
            UseSigmoidGating = useSigmoidGating,
            UseL2QkNorm = useL2QkNorm,
            QkNormAfterRope = qkNormAfterRope,
            IsNeoxRope = isNeoxRope,
            RopeDim = ropeDim,
            IsHybridSsm = isHybridSsm,
            LayerTypes = layerTypes,
            Gdn = gdn,
            EmbeddingScale = embeddingScale,
            ResidualScale = residualScale,
            AttentionScaleOverride = attentionScaleOverride,
            LogitScale = logitScale,
            XieluAlphaN = xieluAlphaN,
            XieluAlphaP = xieluAlphaP,
            XieluBeta = xieluBeta,
            XieluEps = xieluEps,
            UsesReluSquared = usesReluSquared,
            FinalLogitSoftcap = finalLogitSoftcap,
            RopeThetaSwa = ropeThetaSwa,
            SlidingWindowSize = slidingWindow,
            PerLayerEmbeddingWidth = perLayerEmbedWidth,
            HasPostAttnNorm = hasPostAttnNorm,
            HasPostFfwNorm = hasPostFfwNorm,
            HasPerLayerTokenEmbd = hasPerLayerTokenEmbd,
            HasLayerOutputScale = hasLayerOutputScale,
            FfnActivation = ffnActivation,
            IsSwaLayer = isSwaLayer,
            KvSourceLayer = kvSourceLayer,
            LayerHeadDim = layerHeadDim,
            LayerRopeDim = layerRopeDim,
            LayerKvHeads = layerKvHeads,
            AttentionKEqV = attentionKEqV,
        };
    }

    private static int GetInt(IReadOnlyDictionary<string, object> m, string key, int fallback = 0)
    {
        if (!m.TryGetValue(key, out var v)) return fallback;
        // Some keys are stored per-layer as an array (e.g. Gemma 4 12B's
        // gemma4.attention.head_count_kv = [8,8,8,8,8,1,…]). A plain Convert.ToInt32
        // throws on an array; collapse to the first element so the scalar reader
        // doesn't crash. IList covers the reader's object[] plus any typed array
        // (int[]/long[]). Per-layer consumers use GetIntArray instead.
        if (v is System.Collections.IList list) return list.Count > 0 ? Convert.ToInt32(list[0]) : fallback;
        return Convert.ToInt32(v);
    }

    private static float GetFloat(IReadOnlyDictionary<string, object> m, string key, float fallback = 0f) =>
        m.TryGetValue(key, out var v) ? Convert.ToSingle(v) : fallback;

    private static bool GetBool(IReadOnlyDictionary<string, object> m, string key, bool fallback = false) =>
        m.TryGetValue(key, out var v) ? Convert.ToBoolean(v) : fallback;

    /// <summary>
    /// Reads a per-layer integer array (e.g. Gemma 4's per-layer
    /// <c>attention.head_count_kv</c>). Returns <c>null</c> when the key is absent
    /// or stored as a scalar (the caller then falls back to the scalar field).
    /// </summary>
    private static IReadOnlyList<int>? GetIntArray(IReadOnlyDictionary<string, object> m, string key)
    {
        if (!m.TryGetValue(key, out var v)) return null;
        switch (v)
        {
            case IReadOnlyList<int> rl: return rl;          // int[]/List<int> — zero-copy
            case System.Collections.IList list:             // object[] (the reader's form), long[], … — convert
            {
                var result = new int[list.Count];
                for (int i = 0; i < list.Count; i++)
                    result[i] = Convert.ToInt32(list[i]);
                return result;
            }
            default: return null;
        }
    }

    private static IReadOnlyList<bool>? GetBoolArray(IReadOnlyDictionary<string, object> m, string key)
    {
        if (!m.TryGetValue(key, out var v)) return null;
        switch (v)
        {
            case bool[] ba: return ba;
            case IReadOnlyList<bool> rl: return rl;
            case object[] oa:
            {
                var result = new bool[oa.Length];
                for (int i = 0; i < oa.Length; i++)
                    result[i] = Convert.ToBoolean(oa[i]);
                return result;
            }
            default: return null;
        }
    }

    /// <summary>Per-layer float array (e.g. Apertus's <c>xielu.alpha_n</c>). Same shape as
    /// <see cref="GetIntArray"/>; also accepts a scalar (broadcast to every layer), since
    /// <c>get_key_or_arr</c> on the llama.cpp side accepts either.</summary>
    private static IReadOnlyList<float>? GetFloatArray(IReadOnlyDictionary<string, object> m, string key, int numLayers)
    {
        if (!m.TryGetValue(key, out var v)) return null;
        switch (v)
        {
            case IReadOnlyList<float> rl: return rl;
            case System.Collections.IList list:
            {
                var result = new float[list.Count];
                for (int i = 0; i < list.Count; i++)
                    result[i] = Convert.ToSingle(list[i]);
                return result;
            }
            default:
            {
                float scalar = Convert.ToSingle(v);
                var result = new float[numLayers];
                Array.Fill(result, scalar);
                return result;
            }
        }
    }
}

public abstract class ModelLayer
{
    public string Name { get; init; } = string.Empty;
}

public sealed class AttentionLayer : ModelLayer { }
public sealed class FeedForwardLayer : ModelLayer { }
public sealed class EmbeddingLayer : ModelLayer { }
public sealed class NormLayer : ModelLayer { }
public sealed class OutputLayer : ModelLayer { }

/// <summary>
/// Block type for one trunk layer. Hybrid models (qwen35moe) interleave the two
/// types according to a fixed interval; pure transformer models are all-Attention.
/// </summary>
public enum LayerType
{
    Attention = 0,
    GatedDeltaNet = 1,
}

/// <summary>
/// FFN inner activation. Most LLaMA-family models use SiLU/Swish gating; Gemma 4 uses
/// the tanh-approximation of GELU on the gate projection.
/// </summary>
public enum FfnActivation
{
    Silu = 0,
    GeluApprox = 1,
}

/// <summary>
/// Hyperparameters for a Gated DeltaNet recurrent block (linear attention with
/// delta-rule rank-1 state update). Despite the GGUF prefix <c>ssm.*</c>, this
/// is NOT Mamba selective scan — there is no per-state-dim A vector and the
/// recurrent state is a per-head matrix.
/// </summary>
/// <param name="NumKHeads">Number of key heads (= <c>ssm.group_count</c>). Each K head is shared by
/// <c>NumVHeads / NumKHeads</c> value heads (GQA-style for the GDN block).</param>
/// <param name="NumVHeads">Number of value heads (= <c>ssm.time_step_rank</c>). The per-head decay
/// (alpha/A) and write rate (beta) are scalars indexed by v-head.</param>
/// <param name="HeadDim">Head dimension shared by Q, K, V, and the per-head matrix state
/// (= <c>ssm.state_size</c>). Each head's recurrent state is a <c>[HeadDim, HeadDim]</c> matrix.</param>
/// <param name="InnerSize">Total value channels = <c>NumVHeads * HeadDim</c> (= <c>ssm.inner_size</c>).</param>
/// <param name="ConvKernel">Depthwise causal conv1d kernel size, applied to the joint Q‖K‖V stream
/// (= <c>ssm.conv_kernel</c>; typically 4).</param>
/// <param name="FullAttentionInterval">Stride between full-attention layers. With value 4,
/// layers where <c>(i+1) % 4 == 0</c> are full attention and the rest are GDN.</param>
public sealed record GdnConfig(
    int NumKHeads,
    int NumVHeads,
    int HeadDim,
    int InnerSize,
    int ConvKernel,
    int FullAttentionInterval)
{
    /// <summary>Total key channels = <c>NumKHeads * HeadDim</c>.</summary>
    public int KeyDim => NumKHeads * HeadDim;

    /// <summary>Total value channels = <c>NumVHeads * HeadDim</c>; equals <see cref="InnerSize"/>.</summary>
    public int ValueDim => NumVHeads * HeadDim;

    /// <summary>
    /// Channels in the joint QKV stream that the depthwise conv1d operates on:
    /// <c>KeyDim*2 + ValueDim</c> (Q and K share KeyDim each, V is ValueDim).
    /// </summary>
    public int ConvChannels => KeyDim * 2 + ValueDim;
}
