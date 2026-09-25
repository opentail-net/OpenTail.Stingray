using System.Collections.Frozen;
using System.Globalization;
using System.Text;

namespace OpenTail.Stingray.Core;

/// <summary>
/// Culture-free equivalent of Python's `unicodedata.normalize("NFD", s)` followed by dropping every
/// NonSpacingMark (Mn) char, i.e. BERT's `_run_strip_accents`. `string.Normalize(FormD)` can't be used:
/// with this repo's InvariantGlobalization it does not decompose non-ASCII, so "Äpfel" stayed "äpfel"
/// and became [UNK] instead of "apfel" (found by the llama.cpp BGE vocab golden test, 2026-09-25).
/// The canonical decompositions come from a generated table (<c>UnicodeAccentStrip.g.cs</c>); Hangul
/// syllables use the Unicode algorithmic decomposition into jamo, as Python's NFD does.
/// </summary>
internal static partial class UnicodeAccentStrip
{
    private static readonly FrozenDictionary<char, string> s_map = BuildMap();

    private static FrozenDictionary<char, string> BuildMap()
    {
        var parts = Values.Split('\0');
        var map = new Dictionary<char, string>(Keys.Length);
        for (int i = 0; i < Keys.Length; i++) map[Keys[i]] = parts[i];
        return map.ToFrozenDictionary();
    }

    // Unicode Hangul syllable decomposition constants (Unicode §3.12).
    private const int SBase = 0xAC00, LBase = 0x1100, VBase = 0x1161, TBase = 0x11A7;
    private const int VCount = 21, TCount = 28, NCount = VCount * TCount, SCount = 11172;

    public static string StripAccents(string text)
    {
        StringBuilder? sb = null;
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            string? replacement = null;
            bool drop = false;
            int sIndex = c - SBase;
            if (sIndex >= 0 && sIndex < SCount)
            {
                var jamo = new StringBuilder(3);
                jamo.Append((char)(LBase + sIndex / NCount)).Append((char)(VBase + sIndex % NCount / TCount));
                int t = sIndex % TCount;
                if (t != 0) jamo.Append((char)(TBase + t));
                replacement = jamo.ToString();
            }
            else if (c >= 0x80 && s_map.TryGetValue(c, out var mapped)) replacement = mapped;
            else if (c >= 0x80 && CharUnicodeInfo.GetUnicodeCategory(c) == UnicodeCategory.NonSpacingMark) drop = true;

            if (replacement is null && !drop)
            {
                sb?.Append(c);
                continue;
            }
            sb ??= new StringBuilder(text.Length).Append(text, 0, i);
            if (replacement is not null) sb.Append(replacement);
        }
        return sb?.ToString() ?? text;
    }
}
