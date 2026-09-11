namespace OpenTail.Stingray.Audio.ParaformerOnnx;

/// <summary>
/// Real Paraformer symbol table (`sherpa_onnx::SymbolTable` in
/// `examples/sherpa-onnx/sherpa-onnx/csrc/symbol-table.h/.cc`) plus the real token-id-sequence to
/// text conversion, ported verbatim from `offline-recognizer-paraformer-impl.h`'s free function
/// Convert(decoder result, symbol table).
///
/// File format: one "symbol id" pair per line (`tokens.txt`, sherpa-onnx's standard format).
/// This project's copy, for the local `paraformer-zh-small.int8.onnx` checkpoint specifically, is
/// downloaded from the real matching sherpa-onnx release
/// (`csukuangfj/sherpa-onnx-paraformer-zh-small-2024-03-09`, HuggingFace) to
/// `F:\_models\paraformer-zh-small-tokens.txt` -- confirmed the real match for THIS checkpoint by
/// line count (8359 lines == this checkpoint's own `vocab_size` custom-metadata value, exactly)
/// and by `&lt;/s&gt;` sitting at id 2 in both.
/// </summary>
public sealed class ParaformerOnnxVocab
{
    private readonly string[] _idToSymbol;

    public int VocabSize => _idToSymbol.Length;

    /// <summary>Real `eos_id`: NOT read from ONNX custom metadata (the exported graph carries no
    /// such key -- confirmed by reading `offline-paraformer-model.cc`'s `Init()`, which only reads
    /// `vocab_size`/`lfr_window_size`/`lfr_window_shift`/`neg_mean`/`inv_stddev`). The real source
    /// is `offline-recognizer-paraformer-impl.h` line 95: `symbol_table_["&lt;/s&gt;"]` -- i.e. the
    /// id of the literal `"&lt;/s&gt;"` symbol inside `tokens.txt` itself.</summary>
    public int EosId { get; }

    private ParaformerOnnxVocab(string[] idToSymbol, int eosId)
    {
        _idToSymbol = idToSymbol;
        EosId = eosId;
    }

    public static ParaformerOnnxVocab Load(string tokensPath)
    {
        var lines = File.ReadAllLines(tokensPath);
        var map = new Dictionary<int, string>(lines.Length);
        int maxId = -1;
        foreach (var line in lines)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            int sep = line.LastIndexOf(' ');
            if (sep < 0) continue;
            string symbol = line[..sep];
            if (!int.TryParse(line[(sep + 1)..], out int id)) continue;
            map[id] = symbol;
            if (id > maxId) maxId = id;
        }

        var arr = new string[maxId + 1];
        foreach (var (id, symbol) in map) arr[id] = symbol;
        for (int i = 0; i < arr.Length; i++) arr[i] ??= "<unk>";

        int eosId = -1;
        foreach (var (id, symbol) in map)
        {
            if (symbol == "</s>") { eosId = id; break; }
        }
        if (eosId < 0)
            throw new InvalidDataException($"Paraformer tokens file '{tokensPath}' has no '</s>' entry -- cannot determine eos_id.");

        return new ParaformerOnnxVocab(arr, eosId);
    }

    public string Symbol(int id) => (uint)id < (uint)_idToSymbol.Length ? _idToSymbol[id] : "<unk>";

    /// <summary>
    /// Real text assembly, ported verbatim from `offline-recognizer-paraformer-impl.h`'s
    /// `Convert()`: BPE `@@`-suffixed pieces get merged without a space; ASCII pieces get a
    /// leading space unless continuing a mergeable BPE run; non-ASCII (CJK) pieces get no
    /// separating space from a preceding non-ASCII piece, but DO get one after an ASCII piece.
    /// </summary>
    public string Convert(IReadOnlyList<int> tokenIds)
    {
        var text = new System.Text.StringBuilder();
        bool mergeable = false;

        for (int i = 0; i < tokenIds.Count; i++)
        {
            string sym = Symbol(tokenIds[i]);
            bool endsWithDoubleAt = sym.Length >= 2 && sym[^1] == '@' && sym[^2] == '@';

            if (!endsWithDoubleAt)
            {
                bool isAscii = sym.Length > 0 && sym[0] < 0x80;
                if (isAscii)
                {
                    if (mergeable)
                    {
                        mergeable = false;
                        text.Append(sym);
                    }
                    else
                    {
                        text.Append(' ');
                        text.Append(sym);
                    }
                }
                else
                {
                    mergeable = false;
                    if (i > 0)
                    {
                        string prevSym = Symbol(tokenIds[i - 1]);
                        if (prevSym.Length > 0 && prevSym[0] < 0x80)
                        {
                            text.Append(' ');
                        }
                    }
                    text.Append(sym);
                }
            }
            else
            {
                // ends with "@@": strip it, this piece continues into the next.
                string trimmed = sym[..^2];
                if (mergeable)
                {
                    text.Append(trimmed);
                }
                else
                {
                    text.Append(' ');
                    text.Append(trimmed);
                    mergeable = true;
                }
            }
        }

        return text.ToString();
    }
}
