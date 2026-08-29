
namespace OpenTail.Stingray.Core;

/// <summary>
/// The generation defaults a package ships alongside its configuration.
/// </summary>
public sealed record ModelGenerationDefaults(
    int? BosTokenId,
    int? EosTokenId,
    int? PadTokenId,
    float? Temperature,
    float? TopP,
    int? TopK);

/// <summary>Outcome of reading a package configuration strictly.</summary>
public sealed record SafetensorsConfigReadResult(
    ModelHyperparams? Hyperparams,
    ModelGenerationDefaults Generation,
    IReadOnlyList<ModelPackageRejection> Rejections)
{
    /// <summary>True when the configuration was fully understood and mapped.</summary>
    public bool IsUsable => Hyperparams is not null && Rejections.Count == 0;
}

/// <summary>
/// Reads a Hugging Face <c>config.json</c> into canonical <see cref="ModelHyperparams"/> for the
/// dense Llama/Mistral profile, refusing anything it does not fully understand.
/// </summary>
/// <remarks>
/// <para><b>Why this is separate from <c>ModelHyperparams.FromGgufMetadata</c>.</b> That
/// mapper is deliberately permissive: absent GGUF keys fall back to defaults, because a GGUF file is
/// produced by a converter that already made those decisions. A Hugging Face configuration is
/// author-written and carries settings that change the model's arithmetic — RoPE scaling, tied
/// embeddings, biases, unusual activations. Applying GGUF's tolerant defaults to it would silently
/// run a different model than the one on disk.</para>
///
/// <para><b>Unknown keys are refused, not ignored.</b> The plan's review rules say so directly: "do
/// not ignore unknown config fields simply because tensor names look familiar". A curated set of keys
/// is known to be either mapped or provably irrelevant to execution (provenance, dtype hints,
/// tooling versions); everything else is a refusal. That will occasionally reject a package that
/// would have run correctly, which is the intended direction of error — the alternative is executing
/// a model whose configuration we did not read.</para>
/// </remarks>
public static class SafetensorsConfigReader
{
    /// <summary>Keys this reader maps into <see cref="ModelHyperparams"/>.</summary>
    private static readonly HashSet<string> MappedKeys = new(StringComparer.Ordinal)
    {
        "model_type", "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "intermediate_size", "vocab_size", "max_position_embeddings",
        "rope_theta", "rms_norm_eps", "hidden_act", "attention_bias", "mlp_bias",
        "tie_word_embeddings", "rope_scaling", "head_dim", "rope_interleaved",
        "bos_token_id", "eos_token_id", "pad_token_id",
    };

    /// <summary>
    /// Keys that carry no execution semantics for this profile: provenance, tooling and storage
    /// hints. Ignoring these is safe; ignoring anything else is not.
    /// </summary>
    private static readonly HashSet<string> BenignKeys = new(StringComparer.Ordinal)
    {
        "architectures", "torch_dtype", "dtype", "transformers_version", "_name_or_path",
        "use_cache", "initializer_range", "model_max_length", "unk_token_id",
        "attention_dropout", "output_attentions", "output_hidden_states", "return_dict",
        "pretraining_tp", "_attn_implementation_autoset",
        // Markers and downstream-runtime hints observed on real packages (SmolLM2-135M-Instruct).
        "is_llama_config", "transformers.js_config",
    };

    /// <summary>Reads and validates <paramref name="configPath"/> for the dense Llama/Mistral profile.</summary>
    public static SafetensorsConfigReadResult Read(string configPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(configPath);
        var rejections = new List<ModelPackageRejection>();

        if (!File.Exists(configPath))
        {
            rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.MissingConfig,
                "config.json", $"No configuration at '{configPath}'."));
            return new SafetensorsConfigReadResult(null, Empty, rejections);
        }

        JsonDocument document;
        try
        {
            document = JsonDocument.Parse(File.ReadAllBytes(configPath));
        }
        catch (JsonException ex)
        {
            rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.MalformedPackage,
                "config.json", $"Not valid JSON: {ex.Message}"));
            return new SafetensorsConfigReadResult(null, Empty, rejections);
        }

        using (document)
        {
            var json = document.RootElement;
            if (json.ValueKind != JsonValueKind.Object)
            {
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.MalformedPackage,
                    "config.json", "Root element is not a JSON object."));
                return new SafetensorsConfigReadResult(null, Empty, rejections);
            }

            foreach (var property in json.EnumerateObject())
            {
                if (MappedKeys.Contains(property.Name) || BenignKeys.Contains(property.Name)) continue;
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.UnsupportedConfig,
                    property.Name,
                    "Unrecognised configuration key. It is refused rather than ignored because an " +
                    "unread setting may change the model's arithmetic."));
            }

            string? modelType = OptionalString(json, "model_type");
            if (modelType is null)
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.UnsupportedConfig,
                    "model_type", "Missing from config.json."));
            else if (modelType is not ("llama" or "mistral"))
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.UnsupportedArchitecture,
                    modelType, "Profile 'dense-llama-cpu' covers llama/mistral."));

            int hidden = RequiredInt(json, "hidden_size", rejections);
            int layers = RequiredInt(json, "num_hidden_layers", rejections);
            int heads = RequiredInt(json, "num_attention_heads", rejections);
            int intermediate = RequiredInt(json, "intermediate_size", rejections);
            int vocab = RequiredInt(json, "vocab_size", rejections);
            int kvHeads = OptionalInt(json, "num_key_value_heads") ?? heads;
            int context = OptionalInt(json, "max_position_embeddings") ?? 0;
            float ropeTheta = OptionalFloat(json, "rope_theta") ?? 10_000f;
            float rmsEps = OptionalFloat(json, "rms_norm_eps") ?? 1e-5f;

            // head_dim is explicit in newer configs and must be honoured rather than derived: a model
            // whose head_dim is not hidden_size/heads is a different model, and deriving it silently
            // would produce plausible-looking output.
            int? explicitHeadDim = OptionalInt(json, "head_dim");
            int headDim = explicitHeadDim ?? (heads > 0 ? hidden / heads : 0);

            if (hidden > 0 && heads > 0 && explicitHeadDim is null && hidden % heads != 0)
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.UnsupportedConfig,
                    "hidden_size", $"hidden_size {hidden} is not divisible by num_attention_heads {heads} " +
                    "and the configuration does not state head_dim."));
            if (kvHeads > 0 && heads > 0 && heads % kvHeads != 0)
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.UnsupportedConfig,
                    "num_key_value_heads", $"num_attention_heads {heads} is not a multiple of " +
                    $"num_key_value_heads {kvHeads}; grouped-query grouping would be ambiguous."));
            if (context <= 0)
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.UnsupportedConfig,
                    "max_position_embeddings", "Missing or non-positive; the context length is not stated."));
            if (ropeTheta <= 0)
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.UnsupportedConfig,
                    "rope_theta", $"Must be positive; found {ropeTheta}."));
            if (rmsEps <= 0)
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.UnsupportedConfig,
                    "rms_norm_eps", $"Must be positive; found {rmsEps}."));

            string? activation = OptionalString(json, "hidden_act");
            if (activation is not null && !string.Equals(activation, "silu", StringComparison.OrdinalIgnoreCase))
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.UnsupportedConfig,
                    "hidden_act", $"Profile requires SiLU; found '{activation}'."));

            if (OptionalBool(json, "attention_bias") == true)
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.UnsupportedConfig,
                    "attention_bias", "Profile requires bias-free attention projections."));
            if (OptionalBool(json, "mlp_bias") == true)
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.UnsupportedConfig,
                    "mlp_bias", "Profile requires bias-free MLP projections."));
            // rope_interleaved selects a different RoPE pairing. `false` is the default this profile
            // implements, so it is accepted; `true` is a different rotation and is refused. Listing it
            // as simply "benign" would have silently mis-rotated any model that sets it.
            if (OptionalBool(json, "rope_interleaved") == true)
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.UnsupportedConfig,
                    "rope_interleaved", "Interleaved RoPE pairing is not implemented by this profile."));

            if (json.TryGetProperty("rope_scaling", out var scaling) && scaling.ValueKind is not JsonValueKind.Null)
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.UnsupportedConfig,
                    "rope_scaling",
                    $"RoPE scaling is not part of this profile (found {scaling.ValueKind})."));

            var generation = new ModelGenerationDefaults(
                OptionalInt(json, "bos_token_id"),
                OptionalInt(json, "eos_token_id"),
                OptionalInt(json, "pad_token_id"),
                Temperature: null, TopP: null, TopK: null);

            if (rejections.Count > 0)
                return new SafetensorsConfigReadResult(null, generation, rejections);

            var hp = new ModelHyperparams
            {
                VocabSize = vocab,
                ContextLength = context,
                EmbeddingDim = hidden,
                NumLayers = layers,
                NumHeads = heads,
                NumKvHeads = kvHeads,
                IntermediateDim = intermediate,
                HeadDim = headDim,
                RopeDim = headDim,
                RmsNormEps = rmsEps,
                RopeTheta = ropeTheta,
                IsNeoxRope = true,
            };
            return new SafetensorsConfigReadResult(hp, generation, rejections);
        }
    }

    /// <summary>
    /// Merges <c>generation_config.json</c> sampling defaults over the values already read from
    /// <c>config.json</c>. Absent file is not an error — generation defaults are optional.
    /// </summary>
    public static ModelGenerationDefaults ReadGenerationDefaults(string packageRoot, ModelGenerationDefaults fromConfig)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(packageRoot);
        string path = Path.Combine(packageRoot, "generation_config.json");
        if (!File.Exists(path)) return fromConfig;

        try
        {
            using var document = JsonDocument.Parse(File.ReadAllBytes(path));
            var json = document.RootElement;
            if (json.ValueKind != JsonValueKind.Object) return fromConfig;
            return new ModelGenerationDefaults(
                OptionalInt(json, "bos_token_id") ?? fromConfig.BosTokenId,
                OptionalInt(json, "eos_token_id") ?? fromConfig.EosTokenId,
                OptionalInt(json, "pad_token_id") ?? fromConfig.PadTokenId,
                OptionalFloat(json, "temperature") ?? fromConfig.Temperature,
                OptionalFloat(json, "top_p") ?? fromConfig.TopP,
                OptionalInt(json, "top_k") ?? fromConfig.TopK);
        }
        catch (JsonException)
        {
            // A malformed generation_config.json must not block loading: it carries defaults, not
            // semantics. The caller still gets whatever config.json stated.
            return fromConfig;
        }
    }

    private static readonly ModelGenerationDefaults Empty = new(null, null, null, null, null, null);

    private static string? OptionalString(JsonElement json, string name) =>
        json.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;

    private static int? OptionalInt(JsonElement json, string name) =>
        json.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number && v.TryGetInt32(out int i)
            ? i : null;

    private static float? OptionalFloat(JsonElement json, string name) =>
        json.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number && v.TryGetDouble(out double d)
            ? (float)d : null;

    private static bool? OptionalBool(JsonElement json, string name) =>
        json.TryGetProperty(name, out var v) && v.ValueKind is JsonValueKind.True or JsonValueKind.False
            ? v.GetBoolean() : null;

    private static int RequiredInt(JsonElement json, string name, List<ModelPackageRejection> rejections)
    {
        int? value = OptionalInt(json, name);
        if (value is null or <= 0)
        {
            rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.UnsupportedConfig,
                name, value is null ? "Missing or non-numeric in config.json." : $"Must be positive; found {value}."));
            return 0;
        }
        return value.Value;
    }
}
