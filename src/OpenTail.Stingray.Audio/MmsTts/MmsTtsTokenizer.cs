using System.Text.Json;

namespace OpenTail.Stingray.Audio.MmsTts;

/// <summary>
/// Real MMS-TTS tokenizer, ported directly from the real HuggingFace `transformers.VitsTokenizer`
/// source (`transformers/models/vits/tokenization_vits.py`) with `phonemize=False` (this
/// checkpoint's own `tokenizer_config.json` sets `"phonemize": false` -- English MMS-TTS uses
/// plain grapheme/character tokens, not phonemes, unlike Piper's espeak-ng IPA pipeline).
///
/// Confirmed against the real Python tokenizer's own output (not guessed): `VitsTokenizer.
/// from_pretrained(..., phonemize=False)("Hello, world!")` produces
/// `[0,6,0,7,0,21,0,21,0,22,0,19,0,9,0,22,0,25,0,21,0,5,0]` -- lowercase + strip any character not
/// in the vocab (drops the comma and exclamation mark), then intersperse blank token id 0 between
/// every character AND at both ends: length = 2*len(normalized_text) + 1. The blank id (0) is
/// whatever vocab entry happens to map to numeric id 0 (for `facebook/mms-tts-eng`, that's the
/// character "k" -- an artifact of training, not semantically meaningful; NEVER hardcode the
/// blank character itself, always use vocab id 0 directly).
/// </summary>
public sealed class MmsTtsTokenizer
{
    private readonly Dictionary<char, int> _charToId;

    public MmsTtsTokenizer(string vocabJsonPath)
    {
        using var stream = File.OpenRead(vocabJsonPath);
        using var doc = JsonDocument.Parse(stream);
        _charToId = new Dictionary<char, int>();
        foreach (var prop in doc.RootElement.EnumerateObject())
        {
            if (prop.Name.Length == 1)
                _charToId[prop.Name[0]] = prop.Value.GetInt32();
        }
    }

    /// <summary>
    /// Normalizes (lowercase, strip any character not in the vocab), then tokenizes with blank
    /// (id 0) interspersed between every character and at both ends -- real `add_blank=True`
    /// behavior. Returns real vocab ids, ready to feed straight into the text encoder's embedding
    /// lookup.
    /// </summary>
    public int[] Encode(string text)
    {
        var lowered = text.ToLowerInvariant();
        var kept = new List<char>(lowered.Length);
        foreach (var c in lowered)
            if (_charToId.ContainsKey(c)) kept.Add(c);

        var ids = new int[kept.Count * 2 + 1];
        for (int i = 0; i < kept.Count; i++)
            ids[2 * i + 1] = _charToId[kept[i]];
        // ids[0], ids[2], ids[4], ... stay 0 (the blank token id) -- default array value.
        return ids;
    }
}
