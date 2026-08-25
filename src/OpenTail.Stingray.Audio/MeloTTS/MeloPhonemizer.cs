using System.Text.RegularExpressions;
using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Audio.MeloTTS;

/// <summary>
/// Multilingual G2P phonemizer for MeloTTS (ZH/EN checkpoint): maps English text to ARPAbet
/// phoneme tokens via <see cref="CmuDictG2P"/> and punctuation to their exact token IDs in
/// <c>melotts-zh_en-tokens.txt</c>, with VITS pad token interspersing.
/// </summary>
public sealed class MeloPhonemizer
{
    private static readonly Dictionary<string, int> TokenMap = new(StringComparer.OrdinalIgnoreCase)
    {
        ["_"] = 0,
        ["AA"] = 1, ["E"] = 2, ["EE"] = 3, ["En"] = 4, ["N"] = 5, ["OO"] = 6, ["V"] = 7,
        ["a"] = 8, ["a:"] = 9, ["aa"] = 10, ["ae"] = 11, ["ah"] = 12, ["ai"] = 13, ["an"] = 14,
        ["ang"] = 15, ["ao"] = 16, ["aw"] = 17, ["ay"] = 18, ["b"] = 19, ["by"] = 20, ["c"] = 21,
        ["ch"] = 22, ["d"] = 23, ["dh"] = 24, ["dy"] = 25, ["e"] = 26, ["e:"] = 27, ["eh"] = 28,
        ["ei"] = 29, ["en"] = 30, ["eng"] = 31, ["er"] = 32, ["ey"] = 33, ["f"] = 34, ["g"] = 35,
        ["gy"] = 36, ["h"] = 37, ["hh"] = 38, ["hy"] = 39, ["i"] = 40, ["i0"] = 41, ["i:"] = 42,
        ["ia"] = 43, ["ian"] = 44, ["iang"] = 45, ["iao"] = 46, ["ie"] = 47, ["ih"] = 48, ["in"] = 49,
        ["ing"] = 50, ["iong"] = 51, ["ir"] = 52, ["iu"] = 53, ["iy"] = 54, ["j"] = 55, ["jh"] = 56,
        ["k"] = 57, ["ky"] = 58, ["l"] = 59, ["m"] = 60, ["my"] = 61, ["n"] = 62, ["ng"] = 63,
        ["ny"] = 64, ["o"] = 65, ["o:"] = 66, ["ong"] = 67, ["ou"] = 68, ["ow"] = 69, ["oy"] = 70,
        ["p"] = 71, ["py"] = 72, ["q"] = 73, ["r"] = 74, ["ry"] = 75, ["s"] = 76, ["sh"] = 77,
        ["t"] = 78, ["th"] = 79, ["ts"] = 80, ["ty"] = 81, ["u"] = 82, ["u:"] = 83, ["ua"] = 84,
        ["uai"] = 85, ["uan"] = 86, ["uang"] = 87, ["uh"] = 88, ["ui"] = 89, ["un"] = 90, ["uo"] = 91,
        ["uw"] = 92, ["v"] = 93, ["van"] = 94, ["ve"] = 95, ["vn"] = 96, ["w"] = 97, ["x"] = 98,
        ["y"] = 99, ["z"] = 100, ["zh"] = 101, ["zy"] = 102,
        ["!"] = 103, ["?"] = 104, ["…"] = 105, [","] = 106, ["."] = 107, ["'"] = 108, ["-"] = 109,
        ["SP"] = 110, ["UNK"] = 111
    };

    /// <summary>
    /// Phonemizes input text and returns phone IDs, tone IDs, and language IDs.
    /// </summary>
    public MeloPhonemeResult Phonemize(string text, string language = "EN-US")
    {
        int langId = language.ToUpperInvariant() switch
        {
            "ZH" => 1,
            "ES" => 2,
            "FR" => 3,
            "JP" => 4,
            "KR" => 5,
            _ => 0 // English (EN-US, EN-BR, etc.)
        };

        if (string.IsNullOrWhiteSpace(text))
        {
            return new MeloPhonemeResult([0, 110, 0], [0, 0, 0], [langId, langId, langId]);
        }

        var rawPhones = new List<int>();
        var tokens = Regex.Split(text.Trim(), @"(\s+|[.,!?;:\-'])");

        foreach (var tok in tokens)
        {
            if (string.IsNullOrEmpty(tok)) continue;

            if (char.IsWhiteSpace(tok[0]))
            {
                rawPhones.Add(110); // SP (space pause)
            }
            else if (tok is "." or "," or "!" or "?" or "-" or "'" or ";" or ":")
            {
                string puncKey = tok switch
                {
                    ";" or ":" => ",",
                    _ => tok
                };
                if (TokenMap.TryGetValue(puncKey, out int puncId))
                    rawPhones.Add(puncId);
                else
                    rawPhones.Add(110); // fallback to space pause
            }
            else if (CmuDictG2P.TryLookupArpabet(tok, out var phones))
            {
                foreach (var p in phones)
                {
                    if (TokenMap.TryGetValue(p, out int pid))
                        rawPhones.Add(pid);
                }
            }
            else
            {
                // Fallback: character by character
                foreach (char c in tok)
                {
                    string chStr = c.ToString();
                    if (TokenMap.TryGetValue(chStr, out int cid))
                        rawPhones.Add(cid);
                }
            }
        }

        if (rawPhones.Count == 0)
        {
            rawPhones.Add(110);
        }

        // Intersperse pad token (0) between every token (VITS commons.intersperse)
        int total = rawPhones.Count * 2 + 1;
        var outPhones = new int[total];
        var outTones = new int[total];
        var outLangIds = new int[total];

        for (int i = 0; i < rawPhones.Count; i++)
        {
            int p = rawPhones[i];
            outPhones[i * 2] = 0;
            outTones[i * 2] = 0;
            outLangIds[i * 2] = langId;

            outPhones[i * 2 + 1] = p;
            // Tone 7 is General American English in MeloTTS tone embedding (tones 0..6 are Chinese)
            outTones[i * 2 + 1] = (p == 110 || (p >= 103 && p <= 109)) ? 0 : 7;
            outLangIds[i * 2 + 1] = langId;
        }

        outPhones[^1] = 0;
        outTones[^1] = 0;
        outLangIds[^1] = langId;

        return new MeloPhonemeResult(outPhones, outTones, outLangIds);
    }
}

public sealed record MeloPhonemeResult(int[] Phones, int[] Tones, int[] LangIds);
