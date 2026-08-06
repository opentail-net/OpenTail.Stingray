using System.Globalization;
using System.Text.Json;

namespace OpenTail.Stingray.Core;

/// <summary>
/// Answers "can OpenTail run this package, and by which profile?" from the files on disk alone.
/// </summary>
/// <remarks>
/// <para>Phase 0's exit gate is that support can be determined <b>without constructing a forward
/// pass</b>, so this reads <c>config.json</c>, the tokenizer assets and the SafeTensors headers, and
/// nothing else. It never allocates tensor data and never loads a model.</para>
///
/// <para>It also never throws for an unsupported package. Refusal is data — a list of
/// <see cref="ModelPackageRejection"/> naming the offending asset or setting — because callers such as
/// doctor and the CLI need to print every reason at once rather than the first one that threw.
/// Exceptions remain for genuinely exceptional conditions like an unreadable directory.</para>
/// </remarks>
public static class ModelPackageInspector
{
    /// <summary>Inspects <paramref name="packagePath"/> against the dense Llama/Mistral CPU profile.</summary>
    public static ModelPackageCapabilityReport Inspect(string packagePath) =>
        Inspect(packagePath, ModelPackageCapability.DenseLlamaCpu);

    /// <summary>Inspects <paramref name="packagePath"/> against one profile.</summary>
    public static ModelPackageCapabilityReport Inspect(string packagePath, ModelPackageCapability profile)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(packagePath);
        ArgumentNullException.ThrowIfNull(profile);

        var rejections = new List<ModelPackageRejection>();
        string? architecture = null;
        string[] dtypes = [];
        var tokenizerFamily = ModelPackageTokenizerFamily.Unknown;
        long? weightBytes = null;
        long? workingSetBytes = null;

        string root = Directory.Exists(packagePath)
            ? Path.GetFullPath(packagePath)
            : Path.GetDirectoryName(Path.GetFullPath(packagePath)) ?? string.Empty;

        if (root.Length == 0 || !Directory.Exists(root))
        {
            rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.MalformedPackage,
                packagePath, "Path is not a model directory and has no parent directory."));
            return Report(packagePath, profile, false, null, dtypes, tokenizerFamily, null, null, rejections);
        }

        // ── config.json ──────────────────────────────────────────────────────
        string configPath = Path.Combine(root, "config.json");
        if (!File.Exists(configPath))
        {
            rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.MissingConfig,
                "config.json", $"No config.json in '{root}'. A SafeTensors package must carry the configuration GGUF embeds."));
        }
        else
        {
            architecture = InspectConfig(configPath, profile, rejections);
        }

        // ── tokenizer assets ─────────────────────────────────────────────────
        tokenizerFamily = DetectTokenizerFamily(root);
        if (tokenizerFamily == ModelPackageTokenizerFamily.Unknown)
        {
            rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.MissingTokenizer,
                "tokenizer.json", $"No tokenizer.json, tokenizer.model or spiece.model in '{root}'."));
        }
        else if (tokenizerFamily != profile.TokenizerFamily)
        {
            rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.MissingTokenizer,
                tokenizerFamily.ToString(),
                $"Profile '{profile.ProfileId}' requires a {profile.TokenizerFamily} tokenizer; found {tokenizerFamily}."));
        }

        // ── weights ──────────────────────────────────────────────────────────
        InspectWeights(root, profile, rejections, ref dtypes, ref weightBytes, ref workingSetBytes);

        bool supported = rejections.Count == 0;
        return Report(packagePath, profile, supported, architecture, dtypes, tokenizerFamily, weightBytes, workingSetBytes, rejections);
    }

    private static ModelPackageCapabilityReport Report(
        string path, ModelPackageCapability profile, bool supported, string? architecture,
        IReadOnlyList<string> dtypes, ModelPackageTokenizerFamily tokenizer, long? weightBytes,
        long? workingSetBytes, IReadOnlyList<ModelPackageRejection> rejections) =>
        new(path, profile.ProfileId, supported, architecture, dtypes, tokenizer,
            supported ? profile.Backends : ModelPackageBackends.None, weightBytes, workingSetBytes, rejections);

    private static string? InspectConfig(
        string configPath, ModelPackageCapability profile, List<ModelPackageRejection> rejections)
    {
        JsonElement json;
        JsonDocument document;
        try
        {
            document = JsonDocument.Parse(File.ReadAllBytes(configPath));
        }
        catch (JsonException ex)
        {
            rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.MalformedPackage,
                "config.json", $"Not valid JSON: {ex.Message}"));
            return null;
        }

        using (document)
        {
            json = document.RootElement;
            if (json.ValueKind != JsonValueKind.Object)
            {
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.MalformedPackage,
                    "config.json", "Configuration root must be a JSON object."));
                return null;
            }
            string? architecture = json.TryGetProperty("model_type", out var mt) && mt.ValueKind == JsonValueKind.String
                ? mt.GetString()
                : null;

            if (architecture is null)
            {
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.UnsupportedConfig,
                    "model_type", "config.json does not declare model_type."));
            }
            else if (!profile.ArchitectureIds.Contains(architecture, StringComparer.Ordinal))
            {
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.UnsupportedArchitecture,
                    architecture,
                    $"Profile '{profile.ProfileId}' covers {string.Join("/", profile.ArchitectureIds)}."));
            }

            // Unknown semantics are refused, not ignored: familiar tensor names are not evidence that
            // the configuration means what this profile assumes.
            if (json.TryGetProperty("hidden_act", out var act) && act.ValueKind == JsonValueKind.String
                && !string.Equals(act.GetString(), "silu", StringComparison.OrdinalIgnoreCase))
            {
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.UnsupportedConfig,
                    "hidden_act", $"Profile requires SiLU; found '{act.GetString()}'."));
            }

            if (TryGetBool(json, "attention_bias"))
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.UnsupportedConfig,
                    "attention_bias", "Profile requires bias-free attention projections."));
            if (TryGetBool(json, "mlp_bias"))
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.UnsupportedConfig,
                    "mlp_bias", "Profile requires bias-free MLP projections."));

            if (json.TryGetProperty("rope_scaling", out var scaling) && scaling.ValueKind is not JsonValueKind.Null)
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.UnsupportedConfig,
                    "rope_scaling", "RoPE scaling is not part of this profile."));

            foreach (string required in (string[])["hidden_size", "num_hidden_layers", "num_attention_heads",
                                                   "intermediate_size", "vocab_size"])
            {
                if (!json.TryGetProperty(required, out var value) || value.ValueKind != JsonValueKind.Number)
                    rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.UnsupportedConfig,
                        required, "Missing or non-numeric in config.json."));
            }

            return architecture;
        }
    }

    private static bool TryGetBool(JsonElement json, string name) =>
        json.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.True;

    private static ModelPackageTokenizerFamily DetectTokenizerFamily(string root)
    {
        if (File.Exists(Path.Combine(root, "tokenizer.json"))) return ModelPackageTokenizerFamily.HuggingFaceJson;
        if (File.Exists(Path.Combine(root, "tokenizer.model")) || File.Exists(Path.Combine(root, "spiece.model")))
            return ModelPackageTokenizerFamily.SentencePiece;
        return ModelPackageTokenizerFamily.Unknown;
    }

    private static void InspectWeights(
        string root, ModelPackageCapability profile, List<ModelPackageRejection> rejections,
        ref string[] dtypes, ref long? weightBytes, ref long? workingSetBytes)
    {
        string single = Path.Combine(root, "model.safetensors");
        string index = Path.Combine(root, "model.safetensors.index.json");
        bool hasSingle = File.Exists(single);
        bool hasIndex = File.Exists(index);

        if (!hasSingle && !hasIndex)
        {
            if (!Directory.EnumerateFiles(root, "model*.safetensors").Any())
            {
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.MissingWeights,
                    "*.safetensors", $"No SafeTensors weights in '{root}'."));
                return;
            }
        }

        if (hasIndex && !VerifyShards(root, index, rejections)) return;

        SafetensorsLoader? loader = null;
        try
        {
            loader = hasSingle && !hasIndex
                ? SafetensorsLoader.Open(single)
                : SafetensorsLoader.OpenDirectory(root);

            dtypes = loader.TensorNames.Select(loader.GetDtype)
                .Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToArray();

            string[] unsupported = dtypes
                .Where(d => !profile.SourceDtypes.Contains(d, StringComparer.Ordinal)).ToArray();
            foreach (string d in unsupported)
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.UnsupportedDtype,
                    d, $"Profile accepts {string.Join("/", profile.SourceDtypes)}."));

            // Reported before load so a caller can refuse on memory grounds rather than discovering it
            // during allocation. This is header arithmetic, not a read.
            long totalDisk = 0;
            long totalWorking = 0;
            foreach (string name in loader.TensorNames)
            {
                int elements = 1;
                foreach (int dim in loader.GetShape(name)) elements = checked(elements * dim);
                string dt = loader.GetDtype(name);
                long onDisk = (long)elements * BytesPerElement(dt);
                totalDisk = checked(totalDisk + onDisk);
                long working = dt == "BF16" ? (long)elements * 4 : onDisk;
                totalWorking = checked(totalWorking + working);
            }
            weightBytes = totalDisk;
            workingSetBytes = totalWorking;
        }
        catch (Exception ex) when (ex is InvalidDataException or JsonException or IOException or OverflowException)
        {
            rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.MalformedPackage,
                "safetensors", ex.Message));
        }
        finally
        {
            loader?.Dispose();
        }
    }

    private static bool VerifyShards(string root, string indexPath, List<ModelPackageRejection> rejections)
    {
        try
        {
            using var document = JsonDocument.Parse(File.ReadAllBytes(indexPath));
            if (!document.RootElement.TryGetProperty("weight_map", out var map)
                || map.ValueKind != JsonValueKind.Object)
            {
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.MalformedPackage,
                    "model.safetensors.index.json", "Index has no weight_map object."));
                return false;
            }

            var missing = new SortedSet<string>(StringComparer.Ordinal);
            foreach (var entry in map.EnumerateObject())
            {
                string? shard = entry.Value.GetString();
                if (string.IsNullOrEmpty(shard)) continue;

                // Shard-index paths must not escape the package root. Metadata is untrusted input.
                string full = Path.GetFullPath(Path.Combine(root, shard));
                if (!full.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.Ordinal))
                {
                    rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.MalformedPackage,
                        shard, "Shard index references a path outside the package root."));
                    return false;
                }
                if (!File.Exists(full)) missing.Add(shard);
            }

            foreach (string shard in missing)
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.MissingShard,
                    shard, "Referenced by model.safetensors.index.json but not present."));
            return missing.Count == 0;
        }
        catch (JsonException ex)
        {
            rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.MalformedPackage,
                "model.safetensors.index.json", $"Not valid JSON: {ex.Message}"));
            return false;
        }
    }

    private static int BytesPerElement(string dtype) => dtype switch
    {
        "F64" or "I64" or "U64" => 8,
        "F32" or "I32" or "U32" => 4,
        "F16" or "BF16" or "I16" or "U16" => 2,
        _ => 1,
    };

    /// <summary>Renders the published capability rows for CLI help and README generation.</summary>
    public static string RenderCapabilityTable()
    {
        var lines = new List<string>
        {
            "| Profile | Architectures | Source dtypes | Tokenizer | Backends | Batching | Sessions |",
            "|---|---|---|---|---|---|---|",
        };
        foreach (var c in ModelPackageCapability.All)
        {
            lines.Add(string.Create(CultureInfo.InvariantCulture,
                $"| {c.ProfileId} | {string.Join("/", c.ArchitectureIds)} | {string.Join("/", c.SourceDtypes)} " +
                $"| {c.TokenizerFamily} | {c.Backends} | {(c.SupportsBatching ? "yes" : "no")} " +
                $"| {(c.SupportsSessions ? "yes" : "no")} |"));
        }
        return string.Join(Environment.NewLine, lines);
    }
}
