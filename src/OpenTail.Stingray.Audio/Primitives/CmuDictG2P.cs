using System.IO.Compression;
using System.Reflection;

namespace OpenTail.Stingray.Audio.Primitives;

/// <summary>
/// Real Grapheme-to-Phoneme lookup backed by the CMU Pronouncing Dictionary (cmudict), the
/// standard, freely-licensed (BSD-style) English pronunciation dictionary maintained by Carnegie
/// Mellon University's Speech Group (<c>github.com/cmusphinx/cmudict</c>) -- the same resource
/// espeak-ng, CMU Sphinx, Festival, and most production English G2P pipelines are built on or
/// influenced by. ~135,000 words, each mapped to its real ARPABET pronunciation, converted here to
/// IPA to match this codebase's existing native TTS phoneme vocabularies (Kokoro, Piper).
///
/// <para>Embedded gzip-compressed (~900KB) as <c>Primitives/cmudict.dict.gz</c> so the engine stays
/// a true single NativeAOT binary with no external data-file dependency, matching every other
/// weight/vocab asset's "load once at startup" convention in this codebase (just via
/// <see cref="Assembly.GetManifestResourceStream"/> instead of a filesystem path, since this data
/// is fixed and small enough to ship inside the assembly rather than alongside a model checkpoint).
/// </para>
///
/// <para><b>Not a substitute for a real letter-to-sound engine on out-of-dictionary words</b>
/// (names, neologisms, typos): cmudict is a closed list. <see cref="TryLookup"/> returns false for
/// anything not in it; callers should keep a rule-based fallback for those words specifically,
/// exactly as before this class existed -- this only replaces the previous ~16-word hardcoded
/// pronunciation table, which covered a vanishing fraction of real English text.</para>
/// </summary>
public static class CmuDictG2P
{
    // Lazy, not a plain field initializer: Load() reaches into VowelIpa/ConsonantIpaTable below,
    // and C# runs field initializers in textual declaration order within the static constructor --
    // a plain `= Load()` here (declared first) would run before those tables' own initializers,
    // reading them as null. Lazy<T> defers Load() until first real use, after the whole type (and
    // therefore every field) has finished initializing.
    private static readonly Lazy<(Dictionary<string, string> Ipa, Dictionary<string, string[]> Arpabet)> DictLazy = new(Load);

    /// <summary>Real IPA lookup: dictionary word -> already-stress-marked IPA string (e.g. "hello" -> "h AH0 L OW1" -> "həlˈoʊ"). Returns false for words not in cmudict.</summary>
    public static bool TryLookup(string word, out string ipa)
    {
        return DictLazy.Value.Ipa.TryGetValue(word.ToLowerInvariant(), out ipa!);
    }

    /// <summary>Real ARPABET lookup: dictionary word -> array of lowercased ARPAbet phonemes without stress digits (e.g. "hello" -> ["hh", "ah", "l", "ow"]). Returns false for words not in cmudict.</summary>
    public static bool TryLookupArpabet(string word, out string[] phones)
    {
        return DictLazy.Value.Arpabet.TryGetValue(word.ToLowerInvariant(), out phones!);
    }

    private static (Dictionary<string, string> Ipa, Dictionary<string, string[]> Arpabet) Load()
    {
        var ipaDict = new Dictionary<string, string>(140_000, StringComparer.Ordinal);
        var arpabetDict = new Dictionary<string, string[]>(140_000, StringComparer.Ordinal);

        var asm = typeof(CmuDictG2P).Assembly;
        using var raw = asm.GetManifestResourceStream("OpenTail.Stingray.Audio.Primitives.cmudict.dict.gz")
            ?? throw new InvalidOperationException("Embedded cmudict.dict.gz resource not found.");
        using var gz = new GZipStream(raw, CompressionMode.Decompress);
        using var reader = new StreamReader(gz, Encoding.UTF8);

        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            if (line.Length == 0 || line[0] == ';') continue; // ";;;" comment header lines

            int sp = line.IndexOf(' ');
            if (sp < 0) continue;
            string word = line[..sp];

            // Alternate pronunciations are keyed "word(2)", "word(3)", ... -- keep only the first
            // (unsuffixed) entry per base word; it is cmudict's own primary/most-common form.
            int paren = word.IndexOf('(');
            if (paren >= 0) continue;

            string arpabet = line[(sp + 1)..];
            ipaDict[word] = ArpabetToIpa(arpabet);

            var rawTokens = arpabet.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            var cleanTokens = new string[rawTokens.Length];
            for (int i = 0; i < rawTokens.Length; i++)
            {
                string p = rawTokens[i];
                if (p.Length > 0 && char.IsDigit(p[^1])) p = p[..^1];
                cleanTokens[i] = p.ToLowerInvariant();
            }
            arpabetDict[word] = cleanTokens;
        }

        return (ipaDict, arpabetDict);
    }

    /// <summary>
    /// Real ARPABET -> IPA phoneme mapping, General American English. Stress digits (0 = none,
    /// 1 = primary, 2 = secondary) on vowel phonemes become a preceding ˈ/ˌ mark on that vowel's
    /// IPA symbol -- an informal-but-standard simplification (not full syllable-boundary
    /// placement), matching the convention already used by this codebase's prior hand-written
    /// examples (e.g. "hello" -> "həlˈoʊ": the stress mark sits directly before the stressed
    /// vowel). Unstressed AH0/ER0 map to the real reduced-vowel forms (ə / ɚ), not their stressed
    /// counterparts (ʌ / ɜɹ) -- a common and easy-to-get-wrong G2P mistake.
    /// </summary>
    private static string ArpabetToIpa(string arpabet)
    {
        var sb = new StringBuilder(arpabet.Length);
        var phones = arpabet.Split(' ', StringSplitOptions.RemoveEmptyEntries);

        foreach (var phone in phones)
        {
            char stress = '\0';
            string p = phone;
            if (p.Length > 0 && char.IsDigit(p[^1]))
            {
                stress = p[^1];
                p = p[..^1];
            }

            string ipa = stress switch
            {
                '1' => StressedVowelIpa(p, primary: true),
                '2' => StressedVowelIpa(p, primary: false),
                '0' => UnstressedVowelIpa(p),
                _ => ConsonantIpa(p),
            };
            sb.Append(ipa);
        }

        return sb.ToString();
    }

    private static string StressedVowelIpa(string p, bool primary)
    {
        string mark = primary ? "ˈ" : "ˌ";
        return mark + (VowelIpa.TryGetValue(p, out var v) ? v : p.ToLowerInvariant());
    }

    private static string UnstressedVowelIpa(string p) => p switch
    {
        "AH" => "ə",
        "ER" => "ɚ",
        _ => VowelIpa.TryGetValue(p, out var v) ? v : p.ToLowerInvariant(),
    };

    private static string ConsonantIpa(string p) =>
        ConsonantIpaTable.TryGetValue(p, out var v) ? v : p.ToLowerInvariant();

    // Real General American English ARPABET vowel -> IPA table.
    private static readonly Dictionary<string, string> VowelIpa = new(StringComparer.Ordinal)
    {
        ["AA"] = "ɑ",
        ["AE"] = "æ",
        ["AH"] = "ʌ",
        ["AO"] = "ɔ",
        ["AW"] = "aʊ",
        ["AY"] = "aɪ",
        ["EH"] = "ɛ",
        ["ER"] = "ɜɹ",
        ["EY"] = "eɪ",
        ["IH"] = "ɪ",
        ["IY"] = "i",
        ["OW"] = "oʊ",
        ["OY"] = "ɔɪ",
        ["UH"] = "ʊ",
        ["UW"] = "u",
    };

    // Real ARPABET consonant -> IPA table (no stress digit on these).
    private static readonly Dictionary<string, string> ConsonantIpaTable = new(StringComparer.Ordinal)
    {
        ["B"] = "b",
        ["CH"] = "ʧ",
        ["D"] = "d",
        ["DH"] = "ð",
        ["F"] = "f",
        ["G"] = "ɡ",
        ["HH"] = "h",
        ["JH"] = "ʤ",
        ["K"] = "k",
        ["L"] = "l",
        ["M"] = "m",
        ["N"] = "n",
        ["NG"] = "ŋ",
        ["P"] = "p",
        ["R"] = "ɹ",
        ["S"] = "s",
        ["SH"] = "ʃ",
        ["T"] = "t",
        ["TH"] = "θ",
        ["V"] = "v",
        ["W"] = "w",
        ["Y"] = "j",
        ["Z"] = "z",
        ["ZH"] = "ʒ",
    };
}
