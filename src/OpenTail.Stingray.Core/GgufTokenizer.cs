using Microsoft.ML.Tokenizers;

namespace OpenTail.Stingray.Core;

/// <summary>
/// Tokenizer that loads BPE vocab and merges from GGUF metadata
/// and delegates to Microsoft.ML.Tokenizers.CodeGenTokenizer (GPT-2 style byte-level BPE).
/// </summary>
public sealed partial class GgufTokenizer : ITokenizer
{
    private readonly Tokenizer _inner;
    private readonly Dictionary<string, int> _specialTokens;
    private readonly Dictionary<int, string> _specialTokensById;
    private readonly Dictionary<string, int> _vocab;
    // Vocab strings indexed by token ID — for byte-level BPE these are the raw
    // GPT-2 byte-level encoded forms, used by DecodeBytes to recover exact bytes
    // without the inner tokenizer's lossy U+FFFD substitution on partial sequences.
    private readonly string[] _idToToken;
    private readonly bool _needsByteEncoding;
    // SentencePiece-style BPE (Gemma/Llama): the vocab encodes spaces as U+2581 (▁).
    // Pre-encode by replacing ' ' with ▁ on input and ▁ back to ' ' on output.
    private readonly bool _isSpmBpe;
    // For SPM mode, merges as (left, right) → priority (lower = higher priority).
    // Microsoft.ML.Tokenizers' BPE applies a built-in pre-tokenizer that splits on
    // ▁ boundaries so merges like "▁ capital → ▁capital" never fire. We do the
    // merge loop ourselves directly against this map.
    private readonly Dictionary<(string, string), int>? _spmMerges;
    // Per-token scores (tokenizer.ggml.scores), aligned by vocab id -- the REAL SentencePiece
    // BPE merge-priority signal (see TokenizerSource.Scores's doc comment). Null when the source
    // has none; EncodeSpm falls back to treating every score as 0.0f in that case, matching
    // llama.cpp's own fallback (still correct, since the real algorithm is gated on vocabulary
    // membership, not on this array being present).
    private readonly float[]? _spmScores;
    private readonly bool _addSpacePrefix;
    // Real SentencePiece Unigram-LM (tokenizer.ggml.model=t5) -- a genuinely different
    // segmentation algorithm (Viterbi lattice over per-token scores), not merges-based at all.
    // Null unless the model declares this vocab type. Uses the SAME Metaspace ('▁') convention
    // as SPM for decode, so Decode/DecodeBytes treat it identically to _isSpmBpe there.
    private readonly UnigramTokenizer? _unigram;

    // Byte-level BPE pre-tokenizer split cascade, applied BEFORE byte encoding — see
    // EncodeByteLevelBpe for why that order matters, and PreTokenizerPatterns for why it is a
    // cascade rather than one pattern.
    private readonly Regex[]? _preTokenSplit;
    // Byte-level BPE merge ranks, used with _preTokenSplit. Same shape as _spmMerges but keyed on
    // GPT-2 byte-level pieces rather than the SentencePiece U+2581-prefixed ones.
    private readonly Dictionary<(string, string), int>? _byteBpeMerges;

    /// <summary>
    /// Mistral's Tekken pre-tokenizer split pattern. The pattern itself now lives in
    /// <see cref="PreTokenizerPatterns"/> alongside every other <c>tokenizer.ggml.pre</c> value;
    /// this accessor is retained because the split-pattern tests assert against it directly, and
    /// those tests need no model file.
    /// </summary>
    internal static Regex TekkenPreTokenizer()
    {
        PreTokenizerPatterns.TryResolve("tekken", out var patterns);
        return patterns[0];
    }

    public int VocabSize { get; }
    public int BosTokenId { get; }
    public int EosTokenId { get; }
    public int UnknownTokenId { get; }
    public int PadTokenId { get; }
    public bool AddBosToken { get; }

    /// <summary>
    /// All end-of-generation token IDs: the configured EOS plus any well-known EOG marker
    /// present in this vocab (e.g. Gemma 4's <c>&lt;eos&gt;</c> at id 1, which is distinct from
    /// its configured EOS <c>&lt;turn|&gt;</c> at id 106). Generation should stop on ANY of these;
    /// otherwise a model that emits an alternate end token decodes it as literal text and runs on.
    /// Always contains at least <see cref="EosTokenId"/>. See <see cref="BuildEogTokenIds"/> for
    /// the resolution rules. Immutable, so the published stop set can't be tampered with.
    /// </summary>
    public ImmutableArray<int> EogTokenIds { get; }

    /// <summary>All special (control) tokens keyed by their string representation.</summary>
    public IReadOnlyDictionary<string, int> SpecialTokens => _specialTokens;

    /// <inheritdoc/>
    public (int Open, int Close) ReasoningTokens { get; }

    /// <summary>The type name of the inner tokenizer (for diagnostics).</summary>
    public string InnerTokenizerType => _inner.GetType().Name;

    /// <summary>
    /// The model's declared <c>tokenizer.ggml.pre</c>, empty when it declares none.
    /// </summary>
    public string DeclaredPreTokenizer { get; private set; } = "";

    /// <summary>
    /// Whether <see cref="DeclaredPreTokenizer"/> is one <see cref="PreTokenizerPatterns"/>
    /// implements. When false, the model is being tokenized with the GPT-2 fallback, which may not
    /// be what it was trained with — and that failure is silent, so it is worth surfacing. Always
    /// true for SentencePiece models, which do not use this mechanism.
    /// </summary>
    public bool PreTokenizerIsKnown { get; private set; } = true;

    private readonly Lazy<JinjaChatTemplate?> _chatTemplate;

    /// <summary>
    /// Jinja2 chat template parsed from GGUF tokenizer.chat_template metadata, if present.
    /// Use this to format messages into a prompt string for any model without hardcoding templates.
    ///
    /// <para>Parsed lazily on first access, not at tokenizer construction — see the constructor's
    /// remarks on <c>chatTemplateSource</c> for why. A malformed template still resolves to
    /// <c>null</c> (caught internally); a pathologically slow one is the caller's problem the first
    /// time they touch this property, same as before, just no longer everyone's problem on load.</para>
    /// </summary>
    public JinjaChatTemplate? ChatTemplate => _chatTemplate.Value;

    private GgufTokenizer(
        Tokenizer inner,
        Dictionary<string, int> specialTokens,
        Dictionary<int, string> specialTokensById,
        Dictionary<string, int> vocab,
        string[] idToToken,
        int vocabSize,
        int bosTokenId,
        int eosTokenId,
        int unknownTokenId,
        int padTokenId,
        bool addBosToken,
        bool needsByteEncoding,
        bool isSpmBpe,
        Dictionary<(string, string), int>? spmMerges,
        float[]? spmScores,
        bool addSpacePrefix,
        UnigramTokenizer? unigram,
        ImmutableArray<int> eogTokenIds,
        Regex[]? preTokenSplit = null,
        Dictionary<(string, string), int>? byteBpeMerges = null,
        string? chatTemplateSource = null,
        string? chatTemplateBos = null,
        string? chatTemplateEos = null)
    {
        _inner = inner;
        _specialTokens = specialTokens;
        _specialTokensById = specialTokensById;
        _vocab = vocab;
        _idToToken = idToToken;
        _needsByteEncoding = needsByteEncoding;
        _isSpmBpe = isSpmBpe;
        _spmMerges = spmMerges;
        _spmScores = spmScores;
        _addSpacePrefix = addSpacePrefix;
        _unigram = unigram;
        _preTokenSplit = preTokenSplit;
        _byteBpeMerges = byteBpeMerges;
        VocabSize = vocabSize;
        BosTokenId = bosTokenId;
        EosTokenId = eosTokenId;
        UnknownTokenId = unknownTokenId;
        PadTokenId = padTokenId;
        AddBosToken = addBosToken;
        EogTokenIds = eogTokenIds;
        ReasoningTokens = ResolveReasoningTokens(specialTokens);
        _chatTemplate = new Lazy<JinjaChatTemplate?>(() =>
        {
            if (chatTemplateSource is null) return null;
            try
            {
                return new JinjaChatTemplate(chatTemplateSource)
                {
                    BosToken = chatTemplateBos,
                    EosToken = chatTemplateEos,
                };
            }
            catch
            {
                return null;
            }
        });
    }

    /// <summary>
    /// Resolves the open/close special-token IDs that bracket a model's reasoning stream so an
    /// engine can split it into a separate thinking channel. Tries the ChatML
    /// <c>&lt;think&gt;</c>/<c>&lt;/think&gt;</c> convention first, then Gemma 4's
    /// <c>&lt;|channel&gt;</c>/<c>&lt;channel|&gt;</c> "thought" channel (single special tokens —
    /// the template strips every channel block from history, so treating the whole block as
    /// reasoning matches the model's own content/thought split). Both IDs must be positive — id 0
    /// is usually <c>&lt;pad&gt;</c>/<c>&lt;unk&gt;</c> and would mis-trigger — and both must be
    /// present. No match returns <c>(-1, -1)</c>, leaving reasoning-stream splitting disabled.
    /// </summary>
    internal static (int Open, int Close) ResolveReasoningTokens(IReadOnlyDictionary<string, int> specialTokens)
    {
        if (specialTokens.TryGetValue("<think>", out int tid)
            && specialTokens.TryGetValue("</think>", out int eid)
            && tid > 0 && eid > 0)
            return (tid, eid);
        if (specialTokens.TryGetValue("<|channel>", out int cid)
            && specialTokens.TryGetValue("<channel|>", out int ceid)
            && cid > 0 && ceid > 0)
            return (cid, ceid);
        return (-1, -1);
    }

    /// <summary>
    /// Creates a tokenizer from GGUF model metadata.
    /// Expects tokenizer.ggml.tokens, tokenizer.ggml.merges, and special token IDs.
    /// </summary>
    public static GgufTokenizer FromGgufModel(GgufModel model)
    {
        var tokensArray = model.Metadata.TryGetValue("tokenizer.ggml.tokens", out var tokensObj)
            ? (object[])tokensObj
            : throw new InvalidDataException("GGUF metadata missing 'tokenizer.ggml.tokens'");
        var mergesArray = model.Metadata.TryGetValue("tokenizer.ggml.merges", out var mergesObj)
            ? (object[])mergesObj
            : [];

        var tokens = new string[tokensArray.Length];
        for (int i = 0; i < tokensArray.Length; i++) tokens[i] = (string)tokensArray[i];
        var merges = new string[mergesArray.Length];
        for (int i = 0; i < mergesArray.Length; i++) merges[i] = (string)mergesArray[i];

        int[]? tokenTypes = null;
        if (model.Metadata.TryGetValue("tokenizer.ggml.token_type", out var tokenTypeObj))
        {
            var raw = (object[])tokenTypeObj;
            tokenTypes = new int[raw.Length];
            for (int i = 0; i < raw.Length; i++) tokenTypes[i] = Convert.ToInt32(raw[i]);
        }

        float[]? scores = null;
        if (model.Metadata.TryGetValue("tokenizer.ggml.scores", out var scoresObj))
        {
            var raw = (object[])scoresObj;
            scores = new float[raw.Length];
            for (int i = 0; i < raw.Length; i++) scores[i] = Convert.ToSingle(raw[i]);
        }

        int eos = GetMetadataInt(model, "tokenizer.ggml.eos_token_id", 2);
        string modelFamily = model.Metadata.TryGetValue("tokenizer.ggml.model", out var tmObj) ? (string)tmObj : "";
        // Real llama.cpp default for LLAMA_VOCAB_TYPE_SPM (tokenizer.ggml.model=llama) is
        // add_space_prefix=true -- a leading space is prepended before tokenizing, so "The" at
        // the very start of a prompt encodes as the SAME piece as a mid-sentence " The" (both
        // "▁The"), not the bare "The" piece (a different vocab entry, if one exists at all).
        // tokenizer.ggml.add_space_prefix, when present, overrides this. Found missing entirely
        // while re-admitting `xverse` (2026-09-02): first-token divergence only, every subsequent
        // token matched exactly, isolating this from the separate SpmMergePiecesByScore fix.
        bool addSpacePrefix = GetMetadataBool(model, "tokenizer.ggml.add_space_prefix", modelFamily == "llama");
        // Real llama.cpp default UNK id differs by vocab type: 0 for LLAMA_VOCAB_TYPE_SPM
        // (tokenizer.ggml.model=llama), 2 for LLAMA_VOCAB_TYPE_UGM (tokenizer.ggml.model=t5 --
        // real SentencePiece Unigram-LM, confirmed via tokenizer_model=="t5" in llama-vocab.cpp).
        // tokenizer.ggml.unknown_token_id, when present, overrides this either way.
        int defaultUnkId = modelFamily == "t5" ? 2 : 0;

        var source = new TokenizerSource
        {
            Tokens = tokens,
            Merges = merges,
            Scores = scores,
            TokenTypes = tokenTypes,
            BosTokenId = GetMetadataInt(model, "tokenizer.ggml.bos_token_id", 1),
            EosTokenId = eos,
            UnknownTokenId = GetMetadataInt(model, "tokenizer.ggml.unknown_token_id", defaultUnkId),
            PadTokenId = GetMetadataInt(model, "tokenizer.ggml.padding_token_id", eos),
            AddBosToken = GetMetadataBool(model, "tokenizer.ggml.add_bos_token", false),
            AddSpacePrefix = addSpacePrefix,
            ModelFamily = modelFamily,
            TokenizerPre = model.Metadata.TryGetValue("tokenizer.ggml.pre", out var tpObj) ? (string)tpObj : "",
            ChatTemplate = model.Metadata.TryGetValue("tokenizer.chat_template", out var tmpl) && tmpl is string t ? t : null,
        };
        return FromSource(source);
    }

    /// <summary>
    /// Builds a tokenizer from normalized metadata, whatever produced it.
    /// </summary>
    /// <remarks>
    /// This is the single construction path. <see cref="FromGgufModel"/> and the Hugging Face
    /// <c>tokenizer.json</c> adapter both normalise into <see cref="TokenizerSource"/> and call here,
    /// so a SafeTensors package and a GGUF file carrying the same vocabulary produce the same
    /// tokenizer — which is what makes a differential parity test between the two formats meaningful.
    /// </remarks>
    public static GgufTokenizer FromSource(TokenizerSource source)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (source.Tokens.Length == 0)
            throw new InvalidDataException("Tokenizer source has an empty vocabulary.");

        var bosTokenId = source.BosTokenId;
        var eosTokenId = source.EosTokenId;
        var unknownTokenId = source.UnknownTokenId;
        var padTokenId = source.PadTokenId;
        var addBosToken = source.AddBosToken;

        // Identify special tokens (control tokens type 3, and user-defined type 4 like <think>)
        var specialTokens = new Dictionary<string, int>();
        if (source.TokenTypes is { } tokenTypes)
        {
            for (int i = 0; i < tokenTypes.Length && i < source.Tokens.Length; i++)
                if (tokenTypes[i] is TokenizerSource.ControlTokenType or TokenizerSource.UserDefinedTokenType)
                    specialTokens[source.Tokens[i]] = i;
        }
        foreach (var pair in source.AdditionalSpecialTokens) specialTokens[pair.Key] = pair.Value;

        // Build vocab and merges as byte arrays (tokenizer constructors may dispose streams)
        byte[] vocabBytes;
        {
            using var vocabStream = new MemoryStream();
            using (var writer = new Utf8JsonWriter(vocabStream))
            {
                writer.WriteStartObject();
                for (int i = 0; i < source.Tokens.Length; i++)
                    writer.WriteNumber(source.Tokens[i], i);
                writer.WriteEndObject();
            }
            vocabBytes = vocabStream.ToArray();
        }

        byte[] mergesBytes;
        {
            using var mergesStream = new MemoryStream();
            using (var sw = new StreamWriter(mergesStream, Encoding.UTF8, leaveOpen: true))
            {
                for (int i = 0; i < source.Merges.Length; i++)
                    sw.WriteLine(source.Merges[i]);
            }
            mergesBytes = mergesStream.ToArray();
        }

        // Get token strings for special tokens.
        // If the unknown token is a control/special token (type 3), don't pass it to
        // CodeGenTokenizer as it won't be in the BPE vocab and will throw.
        string? unknownToken = null;
        if (unknownTokenId >= 0 && unknownTokenId < source.Tokens.Length)
        {
            bool isControl = source.TokenTypes is { } unkTypes
                && unknownTokenId < unkTypes.Length
                && unkTypes[unknownTokenId] == TokenizerSource.ControlTokenType;
            if (!isControl)
                unknownToken = source.Tokens[unknownTokenId];
        }
        string? bosToken = bosTokenId >= 0 && bosTokenId < source.Tokens.Length
            ? source.Tokens[bosTokenId]
            : null;
        string? eosToken = eosTokenId >= 0 && eosTokenId < source.Tokens.Length
            ? source.Tokens[eosTokenId]
            : null;

        IReadOnlyDictionary<string, int>? specialTokensDict =
            specialTokens.Count > 0 ? specialTokens : null;

        // SPM-family tokenizers (Gemma, Llama SentencePiece) encode spaces as U+2581 (▁)
        // and may have merges containing literal newlines/whitespace. The stream-based
        // BpeTokenizer.Create reads merges via StreamReader, so embedded newlines split
        // a merge across multiple lines and the parser throws ("Invalid merger file
        // format"). Use the in-memory BpeOptions API for these models — no file parsing.
        bool isSpmModel = source.ModelFamily is "gemma" or "gemma2" or "gemma3" or "gemma4" or "llama";

        // Real SentencePiece Unigram-LM (LLAMA_VOCAB_TYPE_UGM, tokenizer.ggml.model=t5 --
        // confirmed via llama-vocab.cpp's `tokenizer_model == "t5"` check). A genuinely different
        // segmentation algorithm from both SPM-BPE and byte-BPE: Viterbi max-additive-log-score
        // lattice search over per-token scores, not a merge-priority table. Blocks minicpm,
        // internlm2, ernie4_5, baichuan, orion, nanbeige until now -- these ship
        // tokenizer.ggml.scores with NO merges array, which this engine used to fall through to
        // byte-BPE for (near-total fragmentation, same symptom class as the `xverse` SPM bug, but
        // this one really is a different algorithm entirely, not a missing-data edge case in an
        // algorithm this engine already implements).
        bool isUnigramModel = source.ModelFamily == "t5";
        UnigramTokenizer? unigram = isUnigramModel && source.Scores is not null
            ? UnigramTokenizer.FromGgufVocab(source.Tokens, source.Scores, source.UnknownTokenId, source.TokenTypes)
            : null;

        Tokenizer? inner = null;
        bool needsByteEncoding = false;
        bool isSpmBpe = false;

        // Byte-level BPE: the model's declared pre-tokenizer decides where text is cut before BPE
        // merges apply, and getting it wrong is silent (the pieces still reassemble, so a
        // Decode(Encode(s)) round-trip cannot detect it — only the token IDs differ). This used to
        // apply only to Tekken; every other model fell through to CodeGenTokenizer, which applies
        // GPT-2's split regardless of what tokenizer.ggml.pre declares. That was wrong for Qwen
        // and the StarCoder/SmolLM family — measured, see PreTokenizerParityTests.
        Regex[]? preTokenSplit = null;
        bool knownPre = true;
        if (!isSpmModel && !isUnigramModel && source.Merges.Length > 0)
        {
            knownPre = PreTokenizerPatterns.TryResolve(source.TokenizerPre, out var resolved);
            preTokenSplit = resolved;
        }

        // Try CodeGenTokenizer first (better decode quality for GPT-2 style models).
        // CodeGenTokenizer handles GPT-2 byte-level BPE encoding internally.
        // Skip for SPM and Unigram-LM models — SPM's merges contain whitespace that breaks the
        // parser; Unigram has no merges array at all (a real vocab-only, score-based model), so
        // there's nothing meaningful for CodeGenTokenizer to build from here.
        if (!isSpmModel && !isUnigramModel)
        {
            try
            {
                using var vs1 = new MemoryStream(vocabBytes);
                using var ms1 = new MemoryStream(mergesBytes);
                inner = CodeGenTokenizer.Create(vs1, ms1,
                    addPrefixSpace: false,
                    addBeginOfSentence: false,
                    addEndOfSentence: false);
            }
            catch
            {
                inner = null;
            }

            if (inner is null)
            {
                try
                {
                    using var vs2 = new MemoryStream(vocabBytes);
                    using var ms2 = new MemoryStream(mergesBytes);
                    inner = BpeTokenizer.Create(vs2, ms2,
                        specialTokens: specialTokensDict,
                        unknownToken: unknownToken);
                    needsByteEncoding = true;
                }
                catch
                {
                    inner = null;
                }
            }
        }

        Dictionary<(string, string), int>? spmMerges = null;
        Dictionary<(string, string), int>? byteBpeMerges = null;
        if (inner is null)
        {
            var vocabDict = new Dictionary<string, int>(source.Tokens.Length, StringComparer.Ordinal);
            for (int i = 0; i < source.Tokens.Length; i++)
                vocabDict.TryAdd(source.Tokens[i], i);

            var mergesList = new List<string>(source.Merges.Length);
            for (int i = 0; i < source.Merges.Length; i++)
                mergesList.Add(source.Merges[i]);

            var bpeOptions = new BpeOptions(vocabDict)
            {
                Merges = mergesList,
                SpecialTokens = specialTokensDict,
                UnknownToken = unknownToken,
            };
            inner = BpeTokenizer.Create(bpeOptions);
            needsByteEncoding = false;
            isSpmBpe = isSpmModel;
        }

        // SPM mode: build a (left,right)→priority map for our manual BPE.
        // The first space-separated pair per merge line is the merge rule.
        // This is built outside the `inner is null` branch above because the byte-level BPE path
        // needs it regardless of which inner tokenizer happened to construct: `inner` is what
        // decodes, but for a model with a declared pre-tokenizer we must do the encode ourselves,
        // since CodeGenTokenizer would silently apply GPT-2's split instead of the declared one.
        if (isSpmBpe || preTokenSplit is not null)
        {
            var rankTable = new Dictionary<(string, string), int>(source.Merges.Length);
            for (int i = 0; i < source.Merges.Length; i++)
            {
                var line = source.Merges[i];
                int sp = line.IndexOf(' ');
                if (sp <= 0 || sp >= line.Length - 1) continue;
                var left = line[..sp];
                var right = line[(sp + 1)..];
                rankTable.TryAdd((left, right), i);
            }
            if (isSpmBpe) spmMerges = rankTable; else byteBpeMerges = rankTable;
        }

        var idToToken = new string[source.Tokens.Length];
        for (int i = 0; i < source.Tokens.Length; i++)
            idToToken[i] = source.Tokens[i];

        var vocabLookup = BuildVocabLookup(source.Tokens);
        var eogIds = BuildEogTokenIds(vocabLookup, new HashSet<int>(specialTokens.Values), eosTokenId);

        // Chat-template parsing is deferred to first access (see the ChatTemplate property) rather
        // than done here. Some real-world templates are large enough (Granite 3.3: nested loops,
        // tool-call/citation/hallucination-risk sections, a strftime_now() call) to take unbounded
        // time in JinjaChatTemplate's parser — measured hanging past 45s with zero progress on one
        // such template. Every model load called this constructor, so that hang blocked plain
        // completion use that never touches a chat template at all. Deferring means a pathological
        // template only costs the caller that actually renders one — see docs/01-gguf-model-coverage-plan.md
        // §1d for the Granite investigation that found this, and the follow-up to fix the parser itself.
        string? chatTemplateSource = source.ChatTemplate is { Length: > 0 } tmplStr ? tmplStr : null;
        // Seed the BOS string so the template's `{{- bos_token -}}` (Gemma, Llama, …) renders it
        // instead of an empty string — otherwise the prompt ships with no BOS token, which Gemma is
        // sensitive to (the model degenerates). Only when the model actually prepends BOS
        // (add_bos_token); add_bos_token=false models (e.g. Qwen) keep bos_token empty.
        string? chatTemplateBos = addBosToken ? bosToken : null;
        // Unlike BOS, EOS is seeded unconditionally: it is emitted by the template body to close
        // assistant turns, not prepended to the prompt, so add_bos_token has no bearing on it.
        // Mistral and Llama templates close every assistant turn with
        // `{{ message["content"] + eos_token }}`; with no value the variable renders empty and
        // multi-turn history arrives with no turn boundaries at all.
        string? chatTemplateEos = eosToken;

        var tokenizer = new GgufTokenizer(
            inner,
            specialTokens,
            specialTokens.ToDictionary(kv => kv.Value, kv => kv.Key),
            vocabLookup,
            idToToken,
            source.Tokens.Length,
            bosTokenId,
            eosTokenId,
            unknownTokenId,
            padTokenId,
            addBosToken,
            needsByteEncoding,
            isSpmBpe,
            spmMerges,
            source.Scores,
            source.AddSpacePrefix,
            unigram,
            eogIds,
            preTokenSplit,
            byteBpeMerges,
            chatTemplateSource,
            chatTemplateBos,
            chatTemplateEos);
        tokenizer.PreTokenizerIsKnown = knownPre;
        tokenizer.DeclaredPreTokenizer = source.TokenizerPre;

        return tokenizer;
    }

    private static Dictionary<string, int> BuildVocabLookup(string[] tokens)
    {
        var vocab = new Dictionary<string, int>(tokens.Length, StringComparer.Ordinal);
        for (int i = 0; i < tokens.Length; i++)
            vocab.TryAdd(tokens[i], i);
        return vocab;
    }

    // Canonical end-of-sequence names accepted even when typed NORMAL rather than control —
    // Gemma 4 ships <eos> (id 1) as a normal token, distinct from its end-of-turn token,
    // which these GGUFs name <turn|> (id 106) — NOT the HF-canonical <end_of_turn> (which
    // doesn't exist as a single vocab token here; it splits into pieces). E4B declares
    // eos_token_id=106 so it stops anyway, but the 12B QAT model declares eos_token_id=1,
    // so without <turn|> in this list its turn-end (106) never enters the EOG set and
    // generation runs past the turn (leaking <turn|>…<channel|> garbage). Including all
    // three names makes the turn-end a stop regardless of the (model-varying) eos metadata.
    // A normal token literally named <eos>/<end_of_turn>/<turn|> appearing in generated
    // content is not a real-world case, so treating it as a stop is safe.
    private static readonly string[] s_canonicalEosNames = ["<eos>", "<end_of_turn>", "<turn|>"];

    // Bracket-style end-of-turn markers (Llama, Mistral, Phi, ChatML, ...). These are only
    // accepted as stops when the vocab types them as control/user-defined tokens, so a model
    // that legitimately uses one of these strings as ordinary text isn't silently truncated.
    private static readonly string[] s_controlEogNames =
        ["<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<|eot|>", "<|eom|>", "<|end|>", "<|endoftext|>"];

    /// <summary>
    /// Builds the end-of-generation token set: the configured EOS plus any well-known EOG marker
    /// present in <paramref name="vocabLookup"/>. llama.cpp stops on any EOG; mirroring that
    /// prevents a model from decoding an alternate end token as literal text and running on past
    /// its turn. Canonical EOS-family names (<see cref="s_canonicalEosNames"/>) are accepted even
    /// when typed NORMAL (Gemma 4's <c>&lt;eos&gt;</c> is id 1, NORMAL-typed); the bracket-style
    /// markers are accepted only when the vocab types them as control (id present in
    /// <paramref name="specialIds"/>) so an ordinary token that happens to share the string isn't
    /// silently turned into a stop. Always contains <paramref name="eosTokenId"/>.
    /// </summary>
    internal static ImmutableArray<int> BuildEogTokenIds(
        IReadOnlyDictionary<string, int> vocabLookup, IReadOnlySet<int> specialIds, int eosTokenId)
    {
        var eogIds = new List<int> { eosTokenId };

        void TryAdd(string name, bool requireControl)
        {
            if (vocabLookup.TryGetValue(name, out int id) && id > 0 && !eogIds.Contains(id)
                && (!requireControl || specialIds.Contains(id)))
                eogIds.Add(id);
        }

        foreach (var name in s_canonicalEosNames) TryAdd(name, requireControl: false);
        foreach (var name in s_controlEogNames)   TryAdd(name, requireControl: true);
        return [.. eogIds];
    }

    private static readonly bool s_profileTokenize =
        Environment.GetEnvironmentVariable("STINGRAY_PROFILE_TOKENIZE") == "1";

    public IReadOnlyList<int> Encode(string text)
    {
        if (!s_profileTokenize)
            return EncodeCore(text);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var result = EncodeCore(text);
        sw.Stop();
        double ms = sw.Elapsed.TotalMilliseconds;
        Console.Error.WriteLine(
            $"[TokenizeProfile] {text.Length} chars -> {result.Count} tokens in {ms:F3}ms " +
            $"({(ms > 0 ? text.Length / ms : 0):F0} chars/ms, {(ms > 0 ? result.Count / ms : 0):F0} tok/ms)");
        return result;
    }

    private IReadOnlyList<int> EncodeCore(string text)
    {
        if (_specialTokens.Count == 0)
            return EncodeTextSegment(text);

        // Split text on special token boundaries and encode each segment,
        // inserting special token IDs directly.
        var result = new List<int>();
        int pos = 0;
        while (pos < text.Length)
        {
            // Find the earliest special token match from current position
            int bestStart = text.Length;
            string? bestToken = null;
            foreach (var st in _specialTokens.Keys)
            {
                int idx = text.IndexOf(st, pos, StringComparison.Ordinal);
                if (idx >= 0 && idx < bestStart)
                {
                    bestStart = idx;
                    bestToken = st;
                }
            }

            if (bestToken is null)
            {
                // No more special tokens — encode the rest
                if (pos < text.Length)
                    result.AddRange(EncodeTextSegment(text[pos..]));
                break;
            }

            // Encode text before the special token
            if (bestStart > pos)
                result.AddRange(EncodeTextSegment(text[pos..bestStart]));

            // Insert the special token ID
            result.Add(_specialTokens[bestToken]);
            pos = bestStart + bestToken.Length;
        }
        return result;
    }

    private IReadOnlyList<int> EncodeTextSegment(string text)
    {
        if (text.Length == 0) return [];

        // Real Unigram-LM (Viterbi lattice, not merge-based) -- takes priority over every other
        // path; UnigramTokenizer.Encode does its own Metaspace ('▁') preprocessing internally, so
        // pass raw text, not pre-replaced.
        if (_unigram is not null)
            return _unigram.Encode(text);

        // BpeTokenizer doesn't do GPT-2 byte-level encoding internally —
        // we must convert raw bytes to GPT-2 Unicode before BPE lookup.
        // CodeGenTokenizer handles this automatically.
        // Tekken: split FIRST, then byte-encode each piece. Byte-encoding first would let the
        // split see GPT-2 replacement characters instead of the real text.
        if (_preTokenSplit is not null && _byteBpeMerges is not null)
            return EncodeByteLevelBpe(text, _preTokenSplit, _byteBpeMerges);

        if (_needsByteEncoding)
            text = EncodeToGpt2Bytes(text);
        else if (_isSpmBpe)
        {
            if (_addSpacePrefix) text = " " + text;
            text = text.Replace(' ', '▁');
            if (_spmMerges is not null)
                return EncodeSpm(text);
        }

        var ids = _inner.EncodeToIds(text);

        if (ids.Count > 0) return RemapOutOfVocab(ids);

        var result = new List<int>(text.Length);
        foreach (char c in text)
        {
            // Text is already in GPT-2 / SPM encoding if a transform was applied above.
            char bpe = _needsByteEncoding ? c : EncodeByteToGpt2(c);
            if (_vocab.TryGetValue(bpe.ToString(), out int id))
                result.Add(id);
        }
        return result;
    }

    /// <summary>
    /// Guarantees every emitted id is &lt; <see cref="VocabSize"/>, i.e. addressable in the
    /// model's embedding table. <see cref="CodeGenTokenizer"/> injects model-independent
    /// consecutive-whitespace tokens (ids beyond the supplied vocab — e.g. 2–8-space runs at
    /// ids 49152+) that this GGUF has no embedding row for; feeding one to the GPU embedding
    /// gather reads out of bounds and aborts the CUDA context with error 700 (issue #267), while
    /// the CPU silently reads an adjacent tensor. Any such id is decomposed into its constituent
    /// single-byte tokens (always present in a byte-level BPE vocab), so the whitespace is
    /// preserved as in-vocab tokens rather than dropped. The fast path returns the input
    /// unchanged when every id is already in range, so normal models pay only one scan.
    /// </summary>
    private IReadOnlyList<int> RemapOutOfVocab(IReadOnlyList<int> ids)
    {
        int oob = -1;
        for (int i = 0; i < ids.Count; i++)
            if ((uint)ids[i] >= (uint)VocabSize) { oob = i; break; }
        if (oob < 0) return ids; // common case: nothing to remap

        // Last-resort id for a byte not found in the vocab. UnknownTokenId is normally 0 and
        // in-vocab, but guard so a pathological model (unk = -1 or ≥ VocabSize) can't make the
        // remap itself emit an out-of-range id — that would defeat the whole point.
        int unk = (uint)UnknownTokenId < (uint)VocabSize ? UnknownTokenId : 0;

        var result = new List<int>(ids.Count + 4);
        for (int i = 0; i < ids.Count; i++)
        {
            int id = ids[i];
            if ((uint)id < (uint)VocabSize) { result.Add(id); continue; }

            // Decode the offending token and map each GPT-2 byte-level char to its single-byte
            // vocab token (the base alphabet of a byte-level BPE — always in-vocab). The two inner
            // tokenizers decode differently (see Decode): CodeGenTokenizer returns clean UTF-8, so
            // re-encode each byte to its GPT-2 char; the BpeTokenizer fallback already returns the
            // GPT-2 byte-level form, so use it as-is (re-encoding would double-encode it).
            string piece = _inner.Decode(new[] { id }) ?? string.Empty;
            string gpt2 = _needsByteEncoding ? piece : EncodeToGpt2Bytes(piece);
            foreach (char ch in gpt2)
            {
                result.Add(_vocab.TryGetValue(ch.ToString(), out int byteId) && byteId < VocabSize
                    ? byteId
                    : unk);
            }
        }
        return result;
    }

    /// <summary>
    /// SentencePiece-style BPE: split into unicode code points, then iteratively apply the
    /// highest-SCORE adjacent merge whose concatenated text exists in the vocabulary (leftmost on
    /// a score tie), until no merge applies. Matches llama.cpp's real <c>llm_tokenizer_spm</c>
    /// algorithm exactly -- see <see cref="SpmMergePiecesByScore"/>'s doc comment for why this is
    /// NOT the same as the merges-rank-table algorithm the byte-BPE path uses.
    /// </summary>
    private IReadOnlyList<int> EncodeSpm(string text)
    {
        // Split into Unicode text elements (code points / graphemes), each a start symbol.
        var pieces = new List<string>(text.Length);
        var en = System.Globalization.StringInfo.GetTextElementEnumerator(text);
        while (en.MoveNext()) pieces.Add((string)en.Current);

        var merged = SpmMergePiecesByScore(pieces, _vocab, _spmScores);

        var ids = new List<int>(merged.Count);
        foreach (var piece in merged)
        {
            if (_vocab.TryGetValue(piece, out int id))
                ids.Add(id);
            else
                AppendSpmByteFallback(piece, ids);
        }
        return ids;
    }

    /// <summary>
    /// Real llama.cpp SPM fallback for a merged piece with no direct vocab entry
    /// (<c>llm_tokenizer_spm_session::resegment</c>'s "output any symbols that did not form
    /// tokens as bytes" branch): emit one token per UTF-8 BYTE of the piece, not one UnknownTokenId
    /// for the whole piece -- each byte is looked up as its SentencePiece byte-fallback token
    /// (<c>&lt;0xXX&gt;</c>, uppercase hex, real format confirmed via <c>llama_vocab::byte_to_token</c>),
    /// falling back to the raw single-byte string entry if a model has that form instead, and only
    /// falling all the way to UnknownTokenId if neither exists. Found missing while re-checking
    /// `ernie4_5` (2026-09-01): a literal newline mid-prompt (no direct "\n" vocab entry, but a real
    /// "&lt;0x0A&gt;" byte-fallback entry at a different id) was mapped to UNK instead of that byte
    /// token -- the one divergence in an otherwise full greedy match, isolating this as a real,
    /// general SPM gap distinct from the two `xverse`-motivated fixes (SpmMergePiecesByScore /
    /// AddSpacePrefix): those made merging work at all; this is the STILL-unmatched-after-merging
    /// tail case neither of them touched.
    /// </summary>
    private void AppendSpmByteFallback(string piece, List<int> ids)
    {
        var utf8 = Encoding.UTF8.GetBytes(piece);
        foreach (byte b in utf8)
        {
            string hex = $"<0x{b:X2}>";
            if (_vocab.TryGetValue(hex, out int hexId))
                ids.Add(hexId);
            else if (_vocab.TryGetValue(((char)b).ToString(), out int rawId))
                ids.Add(rawId);
            else if (UnknownTokenId >= 0)
                ids.Add(UnknownTokenId);
        }
    }

    /// <summary>
    /// Real SentencePiece BPE merge, matching llama.cpp's <c>llm_tokenizer_spm_session::tokenize</c>
    /// exactly: unlike <see cref="SpmMergePieces"/> (a merges-RANK-table algorithm reused for
    /// byte-level BPE), classic SentencePiece (<c>tokenizer.ggml.model=llama</c>) has no merges
    /// list in the real algorithm at all -- a candidate adjacent pair is mergeable purely because
    /// its CONCATENATED TEXT already exists as a vocabulary entry, and among mergeable candidates
    /// the one with the HIGHEST SCORE (that entry's own <c>tokenizer.ggml.scores</c> value) is
    /// applied first, leftmost breaking ties (confirmed against <c>llm_bigram_spm::comparator</c>:
    /// <c>(l.score &lt; r.score) || (l.score == r.score &amp;&amp; l.left &gt; r.left)</c> under a
    /// max-heap). <c>tokenizer.ggml.merges</c> is a GGUF export convenience some converters also
    /// emit that happens to encode a compatible order for models that ship it, which is why this
    /// bug (this engine using the merges-rank algorithm for real SPM too) went unnoticed for every
    /// architecture that ships both arrays -- it silently fragmented tokenization down to
    /// near-individual-codepoints for any checkpoint shipping neither (found investigating
    /// `xverse`, 2026-09-02). <paramref name="scores"/> may be null (checkpoint has no scores
    /// array); every candidate is then treated as score 0.0f, matching llama.cpp's own fallback --
    /// still correct, since mergeability is gated on vocabulary membership, not on this array.
    /// </summary>
    internal static List<string> SpmMergePiecesByScore(
        List<string> pieces, IReadOnlyDictionary<string, int> vocab, float[]? scores)
    {
        int n = pieces.Count;
        if (n == 0) return pieces;

        var sym = new string?[n];
        var prev = new int[n];
        var next = new int[n];
        for (int i = 0; i < n; i++)
        {
            sym[i] = pieces[i];
            prev[i] = i - 1;
            next[i] = i + 1 < n ? i + 1 : -1;
        }

        float Score(int id) => scores is not null && (uint)id < (uint)scores.Length ? scores[id] : 0f;

        // Max-heap on score (highest first), leftmost slot breaking ties -- matches
        // llm_bigram_spm::comparator under std::priority_queue exactly. PriorityQueue<> is a
        // min-heap, so negate the score and keep the left index as the tie-break key: two
        // candidates with the same negated score compare by left index ascending, i.e. leftmost
        // wins the tie, same as the reference.
        var pq = new PriorityQueue<(int l, int r, int llen, int rlen), (float negScore, int l)>();

        void TryAdd(int l, int r)
        {
            if (l < 0 || r < 0) return;
            if (sym[l] is not { } ls || sym[r] is not { } rs) return;
            string merged = ls + rs;
            if (!vocab.TryGetValue(merged, out int id)) return;
            pq.Enqueue((l, r, ls.Length, rs.Length), (-Score(id), l));
        }

        for (int i = 0; i + 1 < n; i++) TryAdd(i, i + 1);

        while (pq.Count > 0)
        {
            var (l, r, llen, rlen) = pq.Dequeue();
            // Skip a stale candidate: either operand already consumed (null) or grown (len
            // changed), so this queued pair no longer reflects the current adjacency.
            if (sym[l] is not { } ls || sym[r] is not { } rs || ls.Length != llen || rs.Length != rlen)
                continue;

            sym[l] = ls + rs;
            sym[r] = null;
            int nn = next[r];
            next[l] = nn;
            if (nn >= 0) prev[nn] = l;

            TryAdd(prev[l], l);
            TryAdd(l, nn);
        }

        var result = new List<string>();
        for (int s = 0; s >= 0; s = next[s])
            if (sym[s] is { } piece) result.Add(piece);
        return result;
    }

    /// <summary>
    /// Core SentencePiece merge: given starting symbols (one per code point) and a
    /// bigram→rank merge table, repeatedly apply the lowest-rank adjacent merge (leftmost on
    /// a rank tie) until none apply, returning the surviving pieces in order. Matches
    /// llama.cpp's <c>llm_tokenizer_spm</c>.
    ///
    /// <para>O(n log n) via a min-heap of merge candidates over a symbol doubly-linked list.
    /// The previous implementation rescanned every adjacent pair to find the single best merge
    /// and applied one merge per pass — O(n²) — which made a 40k-token prompt take minutes to
    /// tokenize (the dominant server-request latency). Slots are never reused and a symbol's
    /// text only ever grows via merges, so a slot's current Length uniquely identifies its
    /// state — used as an O(1) staleness check to discard queued candidates whose operands
    /// have since merged away. <c>internal</c> so the parity tests can exercise it directly
    /// with a synthetic merge table (no model file needed).</para>
    /// </summary>
    internal static List<string> SpmMergePieces(
        List<string> pieces, IReadOnlyDictionary<(string, string), int> merges)
    {
        int n = pieces.Count;
        if (n == 0) return pieces;

        var sym = new string?[n];
        var prev = new int[n];
        var next = new int[n];
        for (int i = 0; i < n; i++)
        {
            sym[i] = pieces[i];
            prev[i] = i - 1;
            next[i] = i + 1 < n ? i + 1 : -1;
        }

        // Priority packs merge rank (primary, lower wins — the merges-file order) with the
        // left slot index (secondary, so ties resolve leftmost, matching the old linear scan).
        var pq = new PriorityQueue<(int l, int r, int llen, int rlen), long>();

        void TryAdd(int l, int r)
        {
            if (l < 0 || r < 0) return;
            if (sym[l] is not { } ls || sym[r] is not { } rs) return;
            if (merges.TryGetValue((ls, rs), out int rank))
                pq.Enqueue((l, r, ls.Length, rs.Length), ((long)rank << 32) | (uint)l);
        }

        for (int i = 0; i + 1 < n; i++) TryAdd(i, i + 1);

        while (pq.Count > 0)
        {
            var (l, r, llen, rlen) = pq.Dequeue();
            // Skip a stale candidate: either operand already consumed (null) or grown (len
            // changed), so this queued pair no longer reflects the current adjacency.
            if (sym[l] is not { } ls || sym[r] is not { } rs || ls.Length != llen || rs.Length != rlen)
                continue;

            // Merge r into l and unlink r from the chain.
            sym[l] = ls + rs;
            sym[r] = null;
            int nn = next[r];
            next[l] = nn;
            if (nn >= 0) prev[nn] = l;

            // Re-evaluate the two adjacencies created around the merged symbol.
            TryAdd(prev[l], l);
            TryAdd(l, nn);
        }

        // Walk the surviving symbols in order (slot 0 is always the live head — nothing
        // precedes it, so it is never removed as a right operand).
        var result = new List<string>();
        for (int s = 0; s >= 0; s = next[s])
            if (sym[s] is { } piece) result.Add(piece);
        return result;
    }

    /// <summary>
    /// Converts a UTF-8 string to GPT-2 byte-level BPE Unicode representation.
    /// Each byte in the UTF-8 encoding is mapped to its GPT-2 Unicode codepoint.
    /// </summary>
    /// <summary>
    /// Byte-level BPE encode for vocabs with a declared pre-tokenizer split.
    ///
    /// <para>Order is load-bearing: split the RAW text first, then byte-encode each piece. Byte
    /// encoding first would hand the split GPT-2 replacement characters instead of real letters and
    /// digits, so its Unicode categories would match the wrong things.</para>
    /// </summary>
    // EncodeByteToGpt2's output range is fixed and small (printable ASCII, extended printable,
    // and U+0100-U+0142 -- see its doc comment), so every single-char string EmitPiece needs is
    // one of at most 0x143 possibilities. Cached once instead of allocating fresh per character
    // per call -- see docs/perf-loop-project-review-progress.md for the measurement.
    private static readonly string[] s_singleCharCache = BuildSingleCharCache();
    private static string[] BuildSingleCharCache()
    {
        var arr = new string[0x143];
        for (int i = 0; i < arr.Length; i++) arr[i] = ((char)i).ToString();
        return arr;
    }

    private List<int> EncodeByteLevelBpe(
        string text, Regex[] split, IReadOnlyDictionary<(string, string), int> merges)
    {
        var ids = new List<int>(text.Length);
        var pieces = new List<string>();

        void EmitPiece(string raw)
        {
            string encoded = EncodeToGpt2Bytes(raw);
            pieces.Clear();
            foreach (char ch in encoded)
                pieces.Add(ch < s_singleCharCache.Length ? s_singleCharCache[ch] : ch.ToString());

            foreach (var merged in SpmMergePieces(pieces, merges))
            {
                if (_vocab.TryGetValue(merged, out int id) && (uint)id < (uint)VocabSize)
                {
                    ids.Add(id);
                    continue;
                }
                // A merge produced a string with no vocab entry — impossible for a well-formed
                // merges table, but fall back to single-byte tokens rather than dropping text.
                foreach (char ch in merged)
                {
                    if (_vocab.TryGetValue(ch.ToString(), out int byteId) && (uint)byteId < (uint)VocabSize)
                        ids.Add(byteId);
                    else if ((uint)UnknownTokenId < (uint)VocabSize)
                        ids.Add(UnknownTokenId);
                }
            }
        }

        // PreTokenizerPatterns.Split applies the cascade and already accounts for unmatched gaps,
        // which would otherwise silently drop input.
        foreach (var piece in PreTokenizerPatterns.Split(text, split))
            EmitPiece(piece);

        return ids;
    }

    private static string EncodeToGpt2Bytes(string text)
    {
        byte[] utf8 = Encoding.UTF8.GetBytes(text);
        var sb = new StringBuilder(utf8.Length);
        foreach (byte b in utf8)
            sb.Append(EncodeByteToGpt2((char)b));
        return sb.ToString();
    }

    /// <summary>
    /// Maps a single character to its GPT-2 byte-level BPE Unicode representation.
    /// Printable ASCII (0x21–0x7E) and extended printable (0xA1–0xFF) are unchanged.
    /// Control and non-printable bytes map to U+0100–U+0142.
    /// </summary>
    private static char EncodeByteToGpt2(char c)
    {
        if (c is >= '!' and <= '~') return c;   // printable ASCII: unchanged
        if (c >= '\u00A1') return c;             // extended printable: unchanged
        if (c <= '\u0020') return (char)(c + 0x100); // 0x00–0x20 → U+0100–U+0120
        return (char)(c - 0x7F + 0x121);        // 0x7F–0xA0 → U+0121–U+0142
    }

    public string Decode(IEnumerable<int> tokens)
    {
        // For single-token decode of a special token, bypass the inner tokenizer
        // which doesn't know about special token IDs (type 3/4).
        if (tokens is IReadOnlyList<int> list && list.Count == 1 &&
            _specialTokensById.TryGetValue(list[0], out var specialStr))
            return specialStr;

        var text = _inner.Decode(tokens) ?? string.Empty;

        if (_isSpmBpe || _unigram is not null)
            return text.Replace('▁', ' ');

        // BpeTokenizer may output GPT-2 byte-level BPE artifacts:
        // Ġ (U+0120) = space, Ċ (U+010A) = newline, etc.
        // Convert them back to actual bytes if present.
        if (text.Contains('\u0120') || text.Contains('\u010A'))
            text = Encoding.UTF8.GetString(Gpt2CharsToBytes(text));

        return text;
    }

    /// <inheritdoc/>
    public byte[] DecodeBytes(int token)
    {
        // Special tokens (control / user-defined) decode to their literal UTF-8 bytes.
        if (_specialTokensById.TryGetValue(token, out var specialStr))
            return Encoding.UTF8.GetBytes(specialStr);

        // Look up the raw vocab string by ID rather than going through _inner.Decode,
        // which silently substitutes U+FFFD for incomplete UTF-8 sequences and so
        // loses the exact bytes a single token contributes to the stream.
        if ((uint)token >= (uint)_idToToken.Length)
            return [];

        if (_isSpmBpe || _unigram is not null)
            return Encoding.UTF8.GetBytes(_idToToken[token].Replace('▁', ' '));

        return Gpt2CharsToBytes(_idToToken[token]);
    }

    /// <summary>
    /// Convert GPT-2 byte-level BPE Unicode characters back to raw bytes.
    /// Bytes 0x21-0x7E and 0xA1-0xFF stay as-is; 0x00-0x20 and 0x7F-0xA0 are
    /// remapped to U+0100-U+0142 to keep them printable.
    ///
    /// Returns the raw byte sequence — caller is responsible for UTF-8 decoding,
    /// which may need to span multiple tokens to assemble multi-byte characters.
    /// </summary>
    private static byte[] Gpt2CharsToBytes(string text)
    {
        // Inverse of EncodeByteToGpt2 — must match exactly:
        //   0x21-0x7E       → identity
        //   0xA1-0xFF       → identity
        //   U+0100-U+0120   → 0x00-0x20  (subtract 0x100)
        //   U+0121-U+0142   → 0x7F-0xA0  (subtract 0xA2 = 0x121 - 0x7F)
        var bytes = new List<byte>(text.Length);
        foreach (char c in text)
        {
            if (c is >= '!' and <= '~')
                bytes.Add((byte)c);
            else if (c is >= '¡' and <= 'ÿ')
                bytes.Add((byte)c);
            else if (c is >= 'Ā' and <= 'Ġ')
                bytes.Add((byte)(c - 0x100));
            else if (c is >= 'ġ' and <= 'ł')
                bytes.Add((byte)(c - 0xA2));
            else
            {
                // Not a byte token — encode as UTF-8 (rare fallback)
                foreach (byte b in Encoding.UTF8.GetBytes(new[] { c }))
                    bytes.Add(b);
            }
        }
        return bytes.ToArray();
    }

    private static int GetMetadataInt(GgufModel model, string key, int defaultValue)
    {
        if (!model.Metadata.TryGetValue(key, out var value))
            return defaultValue;
        return Convert.ToInt32(value);
    }

    private static bool GetMetadataBool(GgufModel model, string key, bool defaultValue)
    {
        if (!model.Metadata.TryGetValue(key, out var value))
            return defaultValue;
        return Convert.ToBoolean(value);
    }
}
