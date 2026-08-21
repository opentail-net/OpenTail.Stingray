using System;
using System.Collections.Generic;
using System.IO;

namespace OpenTail.Stingray.Audio.F5TTS;

/// <summary>
/// Real F5-TTS character vocabulary lookup, loaded from the checkpoint's own `vocab.txt`
/// (`examples/f5-tts-py/f5_tts/infer/examples/vocab.txt`, copied to `models/f5tts_vocab.txt` --
/// 2545 lines, matching `F5TtsWeights.VocabSize` exactly). Ported from `f5_tts/model/utils.py`'s
/// `get_tokenizer`/`list_str_to_idx`: line index = token id, unknown characters map to id 0
/// (space -- confirmed by the reference's own assert that `vocab_char_map[" "] == 0`).
///
/// This is a literal character-level tokenizer, NOT the full reference pipeline's `rjieba`+pinyin
/// conversion for Chinese text (`convert_char_to_pinyin`) -- same scope boundary as this whole
/// rebuild's other pipelines (Piper/MeloTTS's phonemizers are similarly simplified placeholders):
/// the neural DiT math is real and verified, but full text normalization/g2p for every supported
/// language is a separate, larger undertaking. For plain Latin-script text this literal per-char
/// lookup against the real vocab IS what the reference does for non-Chinese segments.
/// </summary>
public sealed class F5Tokenizer
{
    private readonly Dictionary<char, int> _vocab = [];

    public F5Tokenizer(string vocabPath)
    {
        if (!File.Exists(vocabPath))
            throw new FileNotFoundException($"F5-TTS vocab file not found: {vocabPath}");

        int i = 0;
        foreach (var line in File.ReadLines(vocabPath))
        {
            if (line.Length == 1) _vocab[line[0]] = i;
            i++;
        }
    }

    public int[] Encode(string text)
    {
        var ids = new int[text.Length];
        for (int i = 0; i < text.Length; i++)
            ids[i] = _vocab.TryGetValue(text[i], out int id) ? id : 0;
        return ids;
    }
}
