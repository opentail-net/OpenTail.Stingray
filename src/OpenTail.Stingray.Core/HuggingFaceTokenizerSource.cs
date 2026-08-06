using System.Text.Json;

namespace OpenTail.Stingray.Core;

/// <summary>
/// Builds a <see cref="TokenizerSource"/> from a Hugging Face package's tokenizer assets.
/// </summary>
/// <remarks>
/// <para>Reads <c>tokenizer.json</c> for the vocabulary and merges, then overlays
/// <c>tokenizer_config.json</c> (special-token names, chat template, add_bos_token) and
/// <c>special_tokens_map.json</c>. The sidecars are optional; the vocabulary is not.</para>
///
/// <para><b>Only the BPE model is accepted.</b> A <c>tokenizer.json</c> can declare Unigram, WordPiece
/// or WordLevel, and each segments text differently. Reading the vocabulary out of one and feeding it
/// to a BPE constructor would produce a tokenizer that encodes without error and disagrees with the
/// model's training — silently wrong output rather than a failure. Anything but BPE is refused.</para>
/// </remarks>
public static class HuggingFaceTokenizerSource
{
    /// <summary>Result of loading a package's tokenizer assets.</summary>
    public sealed record Result(TokenizerSource? Source, IReadOnlyList<ModelPackageRejection> Rejections)
    {
        public bool IsUsable => Source is not null && Rejections.Count == 0;
    }

    /// <summary>Loads the tokenizer assets from <paramref name="packageRoot"/>.</summary>
    public static Result Load(string packageRoot)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(packageRoot);
        var rejections = new List<ModelPackageRejection>();

        string tokenizerPath = Path.Combine(packageRoot, "tokenizer.json");
        if (!File.Exists(tokenizerPath))
        {
            // A SentencePiece asset is detected and named, but loading one is NOT implemented: the
            // vocabulary lives in a protobuf whose pieces and scores would have to be parsed and
            // normalised before they could reach TokenizerSource. Reporting it distinctly is the
            // honest middle ground — the user learns their package has a tokenizer OpenTail can see
            // but not yet read, rather than being told no tokenizer exists. The published capability
            // profile advertises HuggingFaceJson only, so this refusal is consistent with the claim.
            string spPath = Path.Combine(packageRoot, "tokenizer.model");
            if (!File.Exists(spPath)) spPath = Path.Combine(packageRoot, "spiece.model");

            if (File.Exists(spPath))
            {
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.MissingTokenizer,
                    Path.GetFileName(spPath),
                    "SentencePiece tokenizer assets are recognised but cannot yet be loaded; only a " +
                    "Hugging Face tokenizer.json (BPE) is supported."));
                return new Result(null, rejections);
            }

            rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.MissingTokenizer,
                "tokenizer.json", $"No tokenizer.json, tokenizer.model or spiece.model in '{packageRoot}'."));
            return new Result(null, rejections);
        }

        JsonDocument document;
        try
        {
            document = JsonDocument.Parse(File.ReadAllBytes(tokenizerPath));
        }
        catch (JsonException ex)
        {
            rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.MalformedPackage,
                "tokenizer.json", $"Not valid JSON: {ex.Message}"));
            return new Result(null, rejections);
        }

        using (document)
        {
            var root = document.RootElement;
            if (root.ValueKind != JsonValueKind.Object
                || !root.TryGetProperty("model", out var model)
                || model.ValueKind != JsonValueKind.Object)
            {
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.MalformedPackage,
                    "tokenizer.json", "Missing the 'model' object."));
                return new Result(null, rejections);
            }

            string? modelType = model.TryGetProperty("type", out var t) && t.ValueKind == JsonValueKind.String
                ? t.GetString() : null;
            if (!string.Equals(modelType, "BPE", StringComparison.Ordinal))
            {
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.MissingTokenizer,
                    modelType ?? "unknown",
                    "Only the BPE tokenizer model is supported. Other models segment text differently, " +
                    "and reusing their vocabulary through a BPE constructor would encode without error " +
                    "while disagreeing with the model's training."));
                return new Result(null, rejections);
            }

            if (!model.TryGetProperty("vocab", out var vocab) || vocab.ValueKind != JsonValueKind.Object)
            {
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.MalformedPackage,
                    "tokenizer.json", "model.vocab is missing or not an object."));
                return new Result(null, rejections);
            }

            // vocab is {token: id}; rebuild the id-indexed array the construction path expects.
            var byId = new SortedDictionary<int, string>();
            foreach (var entry in vocab.EnumerateObject())
            {
                if (entry.Value.ValueKind != JsonValueKind.Number || !entry.Value.TryGetInt32(out int id)) continue;
                byId[id] = entry.Name;
            }
            if (byId.Count == 0)
            {
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.MalformedPackage,
                    "tokenizer.json", "model.vocab is empty."));
                return new Result(null, rejections);
            }

            // A gap means an id no token claims. Leaving a null there would surface much later as a
            // decode fault, so it is refused here where the cause is still visible.
            int maxId = 0;
            foreach (int id in byId.Keys) if (id > maxId) maxId = id;
            var tokens = new string[maxId + 1];
            foreach (var pair in byId) tokens[pair.Key] = pair.Value;
            var gaps = new List<int>();
            for (int i = 0; i < tokens.Length; i++) if (tokens[i] is null) gaps.Add(i);
            if (gaps.Count > 0)
            {
                rejections.Add(new ModelPackageRejection(ModelPackageRejectionKind.MalformedPackage,
                    "tokenizer.json",
                    $"Vocabulary has {gaps.Count} unassigned id(s) below the maximum (first: {gaps[0]})."));
                return new Result(null, rejections);
            }

            string[] merges = ReadMerges(model);

            var addedSpecial = new Dictionary<string, int>(StringComparer.Ordinal);
            var addedTypes = new Dictionary<int, int>();
            if (root.TryGetProperty("added_tokens", out var added) && added.ValueKind == JsonValueKind.Array)
            {
                foreach (var token in added.EnumerateArray())
                {
                    if (token.ValueKind != JsonValueKind.Object) continue;
                    if (!token.TryGetProperty("id", out var idElement) || !idElement.TryGetInt32(out int id)) continue;
                    string? content = token.TryGetProperty("content", out var c) && c.ValueKind == JsonValueKind.String
                        ? c.GetString() : null;
                    if (content is null) continue;
                    bool special = token.TryGetProperty("special", out var sp) && sp.ValueKind == JsonValueKind.True;
                    if (special)
                    {
                        addedSpecial[content] = id;
                        addedTypes[id] = TokenizerSource.ControlTokenType;
                    }
                    else
                    {
                        addedTypes[id] = TokenizerSource.UserDefinedTokenType;
                    }
                }
            }

            int[]? tokenTypes = null;
            if (addedTypes.Count > 0)
            {
                tokenTypes = new int[tokens.Length];
                foreach (var pair in addedTypes)
                    if (pair.Key >= 0 && pair.Key < tokenTypes.Length) tokenTypes[pair.Key] = pair.Value;
            }

            var config = ReadTokenizerConfig(packageRoot, tokens);
            return new Result(new TokenizerSource
            {
                Tokens = tokens,
                Merges = merges,
                TokenTypes = tokenTypes,
                AdditionalSpecialTokens = addedSpecial,
                BosTokenId = config.Bos ?? 1,
                EosTokenId = config.Eos ?? 2,
                UnknownTokenId = config.Unk ?? 0,
                PadTokenId = config.Pad ?? config.Eos ?? 2,
                AddBosToken = config.AddBos,
                ModelFamily = "hf-bpe",
                ChatTemplate = config.ChatTemplate,
            }, rejections);
        }
    }

    /// <summary>
    /// Merges appear either as "a b" strings or as ["a","b"] pairs depending on the tokenizers
    /// version that wrote the file. Both normalise to the space-separated form.
    /// </summary>
    private static string[] ReadMerges(JsonElement model)
    {
        if (!model.TryGetProperty("merges", out var merges) || merges.ValueKind != JsonValueKind.Array)
            return [];

        var result = new List<string>();
        foreach (var merge in merges.EnumerateArray())
        {
            if (merge.ValueKind == JsonValueKind.String)
            {
                string? value = merge.GetString();
                if (!string.IsNullOrEmpty(value)) result.Add(value);
            }
            else if (merge.ValueKind == JsonValueKind.Array)
            {
                var parts = new List<string>(2);
                foreach (var part in merge.EnumerateArray())
                    if (part.ValueKind == JsonValueKind.String && part.GetString() is { } s) parts.Add(s);
                if (parts.Count == 2) result.Add(parts[0] + " " + parts[1]);
            }
        }
        return result.ToArray();
    }

    private readonly record struct TokenizerConfig(
        int? Bos, int? Eos, int? Unk, int? Pad, bool AddBos, string? ChatTemplate);

    private static TokenizerConfig ReadTokenizerConfig(string packageRoot, string[] tokens)
    {
        var byToken = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < tokens.Length; i++) byToken.TryAdd(tokens[i], i);

        int? bos = null, eos = null, unk = null, pad = null;
        bool addBos = false;
        string? chatTemplate = null;

        foreach (string file in (string[])["tokenizer_config.json", "special_tokens_map.json"])
        {
            string path = Path.Combine(packageRoot, file);
            if (!File.Exists(path)) continue;
            try
            {
                using var document = JsonDocument.Parse(File.ReadAllBytes(path));
                var json = document.RootElement;
                if (json.ValueKind != JsonValueKind.Object) continue;

                bos ??= ResolveTokenId(json, "bos_token", byToken);
                eos ??= ResolveTokenId(json, "eos_token", byToken);
                unk ??= ResolveTokenId(json, "unk_token", byToken);
                pad ??= ResolveTokenId(json, "pad_token", byToken);

                if (json.TryGetProperty("add_bos_token", out var abt) && abt.ValueKind == JsonValueKind.True)
                    addBos = true;
                if (chatTemplate is null
                    && json.TryGetProperty("chat_template", out var ct) && ct.ValueKind == JsonValueKind.String)
                    chatTemplate = ct.GetString();
            }
            catch (JsonException)
            {
                // Sidecars carry names and defaults, not semantics. A malformed one falls back to the
                // TokenizerSource defaults rather than blocking the package.
            }
        }
        return new TokenizerConfig(bos, eos, unk, pad, addBos, chatTemplate);
    }

    /// <summary>
    /// A special token is either a bare string or an object with a <c>content</c> field, depending on
    /// which file and which tokenizers version wrote it.
    /// </summary>
    private static int? ResolveTokenId(JsonElement json, string name, Dictionary<string, int> byToken)
    {
        if (!json.TryGetProperty(name, out var element)) return null;
        string? content = element.ValueKind switch
        {
            JsonValueKind.String => element.GetString(),
            JsonValueKind.Object when element.TryGetProperty("content", out var c)
                && c.ValueKind == JsonValueKind.String => c.GetString(),
            _ => null,
        };
        return content is not null && byToken.TryGetValue(content, out int id) ? id : null;
    }
}
