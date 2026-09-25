using System.Globalization;

namespace OpenTail.Stingray.Core;

/// <summary>
/// Describes the first SafeTensors text-model interchange lane: a Hugging Face model package
/// containing weights plus the configuration and tokenizer assets that GGUF normally embeds.
/// </summary>
/// <remarks>
/// This is intentionally package discovery and validation, not a claim that every SafeTensors
/// model can execute. The first inference lane will be dense Llama-family weights in F32/F16/BF16;
/// quantized GGUF remains the preferred local deployment format.
/// </remarks>
public sealed record SafetensorsTextModelPackage(
    string RootDirectory,
    string WeightsPath,
    string ConfigPath,
    string TokenizerPath,
    string ModelType,
    int HiddenSize,
    int NumHiddenLayers,
    int NumAttentionHeads,
    int NumKeyValueHeads,
    int IntermediateSize,
    int VocabSize,
    int ContextLength,
    float RopeTheta,
    float RmsNormEps,
    IReadOnlyList<string> WeightDtypes,
    int HeadDim = 0)
{
    private static readonly HashSet<string> SupportedDtypes = new(StringComparer.Ordinal)
    {
        "F32", "F16", "BF16"
    };

    /// <summary>
    /// Opens a local Hugging Face-style SafeTensors package and validates the assets and tensor
    /// names needed by the dense Llama-family first lane. Multi-shard directories are supported.
    /// </summary>
    public static SafetensorsTextModelPackage Open(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        string root = Directory.Exists(path)
            ? Path.GetFullPath(path)
            : Path.GetDirectoryName(Path.GetFullPath(path))
                ?? throw new ArgumentException("SafeTensors path has no parent directory.", nameof(path));

        string weights = ResolveWeights(path, root);
        string config = Path.Combine(root, "config.json");
        if (!File.Exists(config))
            throw new FileNotFoundException("SafeTensors text-model packages require sibling config.json.", config);
        string tokenizer = ResolveTokenizer(root);

        using var document = JsonDocument.Parse(File.ReadAllBytes(config));
        var json = document.RootElement;
        string modelType = RequiredString(json, "model_type", config);
        if (modelType == "gpt2")
            return OpenGpt2(json, root, weights, config, tokenizer);
        if (modelType is not ("llama" or "mistral" or "qwen2" or "qwen3"))
            throw new NotSupportedException(
                $"SafeTensors text-model support currently covers dense Llama-family packages (llama, mistral, qwen2, qwen3) only; model_type '{modelType}' is not supported.");

        int hiddenSize = RequiredInt(json, "hidden_size", config);
        int layerCount = RequiredInt(json, "num_hidden_layers", config);
        int attentionHeads = RequiredInt(json, "num_attention_heads", config);
        int kvHeads = OptionalInt(json, "num_key_value_heads", attentionHeads);
        int intermediateSize = RequiredInt(json, "intermediate_size", config);
        int vocabSize = RequiredInt(json, "vocab_size", config);
        int contextLength = OptionalInt(json, "max_position_embeddings", 0);
        float ropeTheta = OptionalFloat(json, "rope_theta", 10_000f);
        float rmsNormEps = OptionalFloat(json, "rms_norm_eps", 1e-5f);
        if (hiddenSize <= 0 || layerCount <= 0 || attentionHeads <= 0 || kvHeads <= 0
            || intermediateSize <= 0 || vocabSize <= 0 || contextLength <= 0
            || ropeTheta <= 0 || rmsNormEps <= 0 || (!json.TryGetProperty("head_dim", out _) && hiddenSize % attentionHeads != 0))
            throw new InvalidDataException($"SafeTensors config contains invalid dense Llama dimensions: {config}");
        if (json.TryGetProperty("hidden_act", out var activation)
            && activation.ValueKind == JsonValueKind.String
            && !string.Equals(activation.GetString(), "silu", StringComparison.OrdinalIgnoreCase))
            throw new NotSupportedException(
                $"SafeTensors text-model support currently requires the Llama SiLU activation; found '{activation.GetString()}'.");
        bool tiedEmbeddings = OptionalBool(json, "tie_word_embeddings", false);
        // Qwen3 decouples head_dim from hidden/heads (0.6B: 16 heads x 128 at hidden 1024).
        int headDim = OptionalInt(json, "head_dim", hiddenSize / attentionHeads);

        using var tensors = OpenWeights(root, weights);
        ValidateRequiredTensors(tensors, hiddenSize, layerCount, attentionHeads, kvHeads,
            intermediateSize, vocabSize, tiedEmbeddings, modelType, headDim);
        var dtypes = tensors.TensorNames.Select(tensors.GetDtype).Distinct(StringComparer.Ordinal).Order().ToArray();
        string[] unsupported = dtypes.Where(dtype => !SupportedDtypes.Contains(dtype)).ToArray();
        if (unsupported.Length > 0)
            throw new NotSupportedException(
                $"SafeTensors text-model support currently accepts only F32, F16, or BF16 weights; found {string.Join(", ", unsupported)}.");

        return new SafetensorsTextModelPackage(root, weights, config, tokenizer, modelType,
            hiddenSize, layerCount, attentionHeads, kvHeads, intermediateSize, vocabSize,
            contextLength, ropeTheta, rmsNormEps, dtypes, headDim);
    }

    /// <summary>
    /// GPT-2 (<c>GPT2LMHeadModel</c>): LayerNorm with bias, learned absolute positions, fused <c>c_attn</c>,
    /// tanh-GELU (<c>gelu_new</c>) MLP, tied output. HF stores the four projection weights as Conv1D
    /// <c>[in, out]</c>; <see cref="SafetensorsTensorSource"/> transposes them on load, as llama.cpp's
    /// <c>conversion/gpt2.py</c> does, so the GGUF <c>gpt2</c> graph sees <c>[out, in]</c>.
    /// </summary>
    private static SafetensorsTextModelPackage OpenGpt2(JsonElement json, string root, string weights, string config, string tokenizer)
    {
        int hidden = RequiredInt(json, "n_embd", config);
        int layers = RequiredInt(json, "n_layer", config);
        int heads = RequiredInt(json, "n_head", config);
        int vocab = RequiredInt(json, "vocab_size", config);
        int context = OptionalInt(json, "n_positions", OptionalInt(json, "n_ctx", 0));
        int inner = json.TryGetProperty("n_inner", out var ni) && ni.ValueKind == JsonValueKind.Number ? ni.GetInt32() : 4 * hidden;
        float eps = OptionalFloat(json, "layer_norm_epsilon", 1e-5f);
        string activation = json.TryGetProperty("activation_function", out var af) && af.ValueKind == JsonValueKind.String ? af.GetString()! : "gelu_new";
        if (activation != "gelu_new")
            throw new NotSupportedException($"GPT-2 SafeTensors support requires activation_function 'gelu_new'; found '{activation}'.");
        if (hidden <= 0 || layers <= 0 || heads <= 0 || vocab <= 0 || context <= 0 || hidden % heads != 0)
            throw new InvalidDataException($"SafeTensors config contains invalid GPT-2 dimensions: {config}");

        using var tensors = OpenWeights(root, weights);
        ValidateShape(tensors, "wte.weight", vocab, hidden);
        ValidateShape(tensors, "wpe.weight", context, hidden);
        ValidateShape(tensors, "ln_f.weight", hidden);
        for (int l = 0; l < layers; l++)
        {
            string p = $"h.{l}.";
            ValidateShape(tensors, p + "ln_1.weight", hidden);
            ValidateShape(tensors, p + "ln_1.bias", hidden);
            ValidateShape(tensors, p + "attn.c_attn.weight", hidden, 3 * hidden);
            ValidateShape(tensors, p + "attn.c_attn.bias", 3 * hidden);
            ValidateShape(tensors, p + "attn.c_proj.weight", hidden, hidden);
            ValidateShape(tensors, p + "ln_2.weight", hidden);
            ValidateShape(tensors, p + "mlp.c_fc.weight", hidden, inner);
            ValidateShape(tensors, p + "mlp.c_proj.weight", inner, hidden);
        }
        var dtypes = tensors.TensorNames.Select(tensors.GetDtype).Distinct(StringComparer.Ordinal).Order().ToArray();
        if (dtypes.Any(d => d != "F32"))
            throw new NotSupportedException($"GPT-2 SafeTensors support currently accepts only F32 weights; found {string.Join(", ", dtypes)}.");
        return new SafetensorsTextModelPackage(root, weights, config, tokenizer, "gpt2",
            hidden, layers, heads, heads, inner, vocab, context, RopeTheta: 0f, RmsNormEps: eps, dtypes, hidden / heads);
    }

    /// <summary>True for GPT-2 Conv1D weights stored <c>[in, out]</c> that must be transposed to <c>[out, in]</c>.</summary>
    public static bool IsConv1DWeight(string safetensorsName) =>
        safetensorsName.StartsWith("h.", StringComparison.Ordinal)
        && (safetensorsName.EndsWith(".attn.c_attn.weight", StringComparison.Ordinal)
            || safetensorsName.EndsWith(".attn.c_proj.weight", StringComparison.Ordinal)
            || safetensorsName.EndsWith(".mlp.c_fc.weight", StringComparison.Ordinal)
            || safetensorsName.EndsWith(".mlp.c_proj.weight", StringComparison.Ordinal));

    internal static SafetensorsLoader OpenWeights(SafetensorsTextModelPackage package) =>
        OpenWeights(package.RootDirectory, package.WeightsPath);

    private static SafetensorsLoader OpenWeights(string root, string weights) =>
        weights.EndsWith(".index.json", StringComparison.OrdinalIgnoreCase)
            || Directory.Exists(weights)
            ? SafetensorsLoader.OpenDirectory(root)
            : SafetensorsLoader.Open(weights);

    /// <summary>
    /// Produces the canonical metadata consumed by OpenTail's Llama-family model graph.
    /// The returned dictionary is deliberately separate from GGUF parsing: it lets a future
    /// SafeTensors weight adapter use the same graph contract without inventing another
    /// hyperparameter representation.
    /// </summary>
    public IReadOnlyDictionary<string, object> ToOpenTailMetadata()
    {
        if (ModelType == "gpt2")
            return new Dictionary<string, object>(StringComparer.Ordinal)
            {
                ["general.architecture"] = "gpt2",
                ["gpt2.vocab_size"] = VocabSize,
                ["gpt2.context_length"] = ContextLength,
                ["gpt2.embedding_length"] = HiddenSize,
                ["gpt2.feed_forward_length"] = IntermediateSize,
                ["gpt2.block_count"] = NumHiddenLayers,
                ["gpt2.attention.head_count"] = NumAttentionHeads,
                ["gpt2.attention.layer_norm_epsilon"] = RmsNormEps,
            };

        // qwen2/qwen3 keep their own GGUF architecture (NeoX rope, biases / QK-norm), keyed the way
        // llama.cpp's converter writes them; llama and mistral share the llama graph.
        string arch = ModelType is "qwen2" or "qwen3" ? ModelType : "llama";
        int headDim = HeadDim > 0 ? HeadDim : HiddenSize / NumAttentionHeads;
        var metadata = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            ["general.architecture"] = arch,
            [$"{arch}.vocab_size"] = VocabSize,
            [$"{arch}.context_length"] = ContextLength,
            [$"{arch}.embedding_length"] = HiddenSize,
            [$"{arch}.block_count"] = NumHiddenLayers,
            [$"{arch}.attention.head_count"] = NumAttentionHeads,
            [$"{arch}.attention.head_count_kv"] = NumKeyValueHeads,
            [$"{arch}.attention.key_length"] = headDim,
            [$"{arch}.feed_forward_length"] = IntermediateSize,
            [$"{arch}.attention.layer_norm_rms_epsilon"] = RmsNormEps,
            [$"{arch}.rope.freq_base"] = RopeTheta,
        };
        if (arch == "llama") metadata["llama.rope.dimension_count"] = headDim;
        else metadata[$"{arch}.attention.value_length"] = headDim;
        return metadata;
    }

    /// <summary>
    /// Maps the dense Hugging Face Llama/Mistral tensor naming scheme to the canonical
    /// OpenTail naming scheme. Returns <c>null</c> for tensors outside the first lane.
    /// </summary>
    public static string? TryMapToOpenTailTensorName(string safetensorsName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(safetensorsName);
        return safetensorsName switch
        {
            "model.embed_tokens.weight" => "token_embd.weight",
            "model.norm.weight" => "output_norm.weight",
            "lm_head.weight" => "output.weight",
            "wte.weight" => "token_embd.weight",
            "wpe.weight" => "position_embd.weight",
            "ln_f.weight" => "output_norm.weight",
            "ln_f.bias" => "output_norm.bias",
            _ => TryMapLayerTensorName(safetensorsName) ?? TryMapGpt2LayerTensorName(safetensorsName),
        };
    }


    private static string? TryMapGpt2LayerTensorName(string name)
    {
        const string prefix = "h.";
        if (!name.StartsWith(prefix, StringComparison.Ordinal)) return null;
        int separator = name.IndexOf('.', prefix.Length);
        if (separator < 0 || !int.TryParse(name.AsSpan(prefix.Length, separator - prefix.Length),
                NumberStyles.None, CultureInfo.InvariantCulture, out int layer) || layer < 0)
            return null;
        string targetSuffix = name[(separator + 1)..] switch
        {
            "ln_1.weight" => "attn_norm.weight",
            "ln_1.bias" => "attn_norm.bias",
            "attn.c_attn.weight" => "attn_qkv.weight",
            "attn.c_attn.bias" => "attn_qkv.bias",
            "attn.c_proj.weight" => "attn_output.weight",
            "attn.c_proj.bias" => "attn_output.bias",
            "ln_2.weight" => "ffn_norm.weight",
            "ln_2.bias" => "ffn_norm.bias",
            "mlp.c_fc.weight" => "ffn_up.weight",
            "mlp.c_fc.bias" => "ffn_up.bias",
            "mlp.c_proj.weight" => "ffn_down.weight",
            "mlp.c_proj.bias" => "ffn_down.bias",
            _ => string.Empty, // attn.bias / attn.masked_bias are causal-mask buffers, not weights
        };
        return targetSuffix.Length == 0 ? null : $"blk.{layer}.{targetSuffix}";
    }
    private static string? TryMapLayerTensorName(string name)
    {
        const string prefix = "model.layers.";
        if (!name.StartsWith(prefix, StringComparison.Ordinal)) return null;
        int separator = name.IndexOf('.', prefix.Length);
        if (separator < 0 || !int.TryParse(name.AsSpan(prefix.Length, separator - prefix.Length),
                NumberStyles.None, CultureInfo.InvariantCulture, out int layer) || layer < 0)
            return null;

        string targetSuffix = name[(separator + 1)..] switch
        {
            "input_layernorm.weight" => "attn_norm.weight",
            "self_attn.q_proj.weight" => "attn_q.weight",
            "self_attn.k_proj.weight" => "attn_k.weight",
            "self_attn.v_proj.weight" => "attn_v.weight",
            "self_attn.o_proj.weight" => "attn_output.weight",
            "post_attention_layernorm.weight" => "ffn_norm.weight",
            "mlp.gate_proj.weight" => "ffn_gate.weight",
            "mlp.up_proj.weight" => "ffn_up.weight",
            "mlp.down_proj.weight" => "ffn_down.weight",
            "self_attn.q_proj.bias" => "attn_q.bias",
            "self_attn.k_proj.bias" => "attn_k.bias",
            "self_attn.v_proj.bias" => "attn_v.bias",
            "self_attn.q_norm.weight" => "attn_q_norm.weight",
            "self_attn.k_norm.weight" => "attn_k_norm.weight",
            _ => string.Empty,
        };
        return targetSuffix.Length == 0 ? null : $"blk.{layer}.{targetSuffix}";
    }

    private static string ResolveWeights(string path, string root)
    {
        if (File.Exists(path))
        {
            if (!path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException("Expected a .safetensors file or a model directory.", nameof(path));
            return Path.GetFullPath(path);
        }
        string single = Path.Combine(root, "model.safetensors");
        if (File.Exists(single)) return single;
        string index = Path.Combine(root, "model.safetensors.index.json");
        if (File.Exists(index)) return index;
        if (Directory.EnumerateFiles(root, "model*.safetensors").Any())
            return root;
        throw new FileNotFoundException("No model.safetensors or model.safetensors.index.json found.", root);
    }

    private static string ResolveTokenizer(string root)
    {
        foreach (string candidate in new[] { "tokenizer.json", "tokenizer.model", "spiece.model" })
        {
            string path = Path.Combine(root, candidate);
            if (File.Exists(path)) return path;
        }
        throw new FileNotFoundException(
            "SafeTensors text-model packages require tokenizer.json, tokenizer.model, or spiece.model beside the weights.", root);
    }

    private static void ValidateRequiredTensors(SafetensorsLoader tensors, int hiddenSize, int layerCount,
        int attentionHeads, int kvHeads, int intermediateSize, int vocabSize, bool tiedEmbeddings, string modelType, int headDim)
    {
        string[] shared = tiedEmbeddings || !tensors.Contains("lm_head.weight")
            ? ["model.embed_tokens.weight", "model.norm.weight"]
            : ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"];

        foreach (string name in shared)
            if (!tensors.Contains(name))
                throw new InvalidDataException($"SafeTensors Llama package is missing required tensor '{name}'.");

        ValidateShape(tensors, "model.embed_tokens.weight", vocabSize, hiddenSize);
        ValidateShape(tensors, "model.norm.weight", hiddenSize);
        if (tensors.Contains("lm_head.weight"))
            ValidateShape(tensors, "lm_head.weight", vocabSize, hiddenSize);
        for (int layer = 0; layer < layerCount; layer++)
        {
            string prefix = $"model.layers.{layer}.";
            foreach (string suffix in new[]
            {
                "input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                "self_attn.v_proj.weight", "self_attn.o_proj.weight", "post_attention_layernorm.weight",
                "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"
            })
                if (!tensors.Contains(prefix + suffix))
                    throw new InvalidDataException($"SafeTensors Llama package is missing required tensor '{prefix + suffix}'.");

            ValidateShape(tensors, prefix + "input_layernorm.weight", hiddenSize);
            ValidateShape(tensors, prefix + "self_attn.q_proj.weight", attentionHeads * headDim, hiddenSize);
            ValidateShape(tensors, prefix + "self_attn.k_proj.weight", kvHeads * headDim, hiddenSize);
            ValidateShape(tensors, prefix + "self_attn.v_proj.weight", kvHeads * headDim, hiddenSize);
            ValidateShape(tensors, prefix + "self_attn.o_proj.weight", hiddenSize, attentionHeads * headDim);
            ValidateShape(tensors, prefix + "post_attention_layernorm.weight", hiddenSize);
            ValidateShape(tensors, prefix + "mlp.gate_proj.weight", intermediateSize, hiddenSize);
            ValidateShape(tensors, prefix + "mlp.up_proj.weight", intermediateSize, hiddenSize);
            ValidateShape(tensors, prefix + "mlp.down_proj.weight", hiddenSize, intermediateSize);
            if (modelType == "qwen2")
            {
                ValidateShape(tensors, prefix + "self_attn.q_proj.bias", attentionHeads * headDim);
                ValidateShape(tensors, prefix + "self_attn.k_proj.bias", kvHeads * headDim);
                ValidateShape(tensors, prefix + "self_attn.v_proj.bias", kvHeads * headDim);
            }
            if (modelType == "qwen3")
            {
                ValidateShape(tensors, prefix + "self_attn.q_norm.weight", headDim);
                ValidateShape(tensors, prefix + "self_attn.k_norm.weight", headDim);
            }
        }
    }

    private static void ValidateShape(SafetensorsLoader tensors, string name, params int[] expected)
    {
        int[] actual = tensors.GetShape(name);
        if (!actual.AsSpan().SequenceEqual(expected))
            throw new InvalidDataException(
                $"SafeTensors Llama tensor '{name}' has shape [{string.Join(", ", actual)}], expected [{string.Join(", ", expected)}].");
    }

    private static string RequiredString(JsonElement json, string name, string config) =>
        json.TryGetProperty(name, out var property) && property.ValueKind == JsonValueKind.String
            ? property.GetString() ?? throw new InvalidDataException($"config.json property '{name}' is empty: {config}")
            : throw new InvalidDataException($"config.json property '{name}' is required: {config}");

    private static int RequiredInt(JsonElement json, string name, string config) =>
        json.TryGetProperty(name, out var property) && property.TryGetInt32(out int value)
            ? value
            : throw new InvalidDataException($"config.json integer property '{name}' is required: {config}");

    private static int OptionalInt(JsonElement json, string name, int fallback) =>
        json.TryGetProperty(name, out var property) && property.TryGetInt32(out int value) ? value : fallback;

    private static float OptionalFloat(JsonElement json, string name, float fallback) =>
        json.TryGetProperty(name, out var property) && property.TryGetSingle(out float value) ? value : fallback;

    private static bool OptionalBool(JsonElement json, string name, bool fallback) =>
        json.TryGetProperty(name, out var property) && property.ValueKind is JsonValueKind.True or JsonValueKind.False
            ? property.GetBoolean()
            : fallback;
}
