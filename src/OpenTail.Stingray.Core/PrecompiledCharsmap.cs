using System.Buffers.Binary;
using System.Text;

namespace OpenTail.Stingray.Core;

/// <summary>
/// SentencePiece's <c>precompiled_charsmap</c> normalizer (the <c>nmt_nfkc</c>-family rules baked into
/// every SentencePiece model and into HF <c>tokenizer.json</c> as the <c>Precompiled</c> normalizer).
/// Ported from llama.cpp's UGM tokenizer (<c>examples/llama.cpp/llama.cpp/src/llama-vocab.cpp</c>,
/// <c>xcda_array_view</c>/<c>normalize_prefix</c>), which mirrors SentencePiece's own
/// <c>normalizer.cc</c>. Blob layout: a uint32 XCDA byte size, then the XOR-compressed double-array
/// trie entries, then null-terminated UTF-8 replacement strings. At each input position the longest
/// matching byte prefix is replaced; otherwise one valid UTF-8 scalar is copied, and an invalid byte
/// becomes U+FFFD. Whitespace handling is left to the caller.
/// </summary>
internal sealed class PrecompiledCharsmap
{
    private readonly uint[] _xcda;
    private readonly byte[] _replacements;

    public PrecompiledCharsmap(byte[] blob)
    {
        if (blob.Length < 4) throw new InvalidDataException("precompiled_charsmap is too short.");
        int xcdaSize = (int)BinaryPrimitives.ReadUInt32LittleEndian(blob);
        if (xcdaSize < 0 || 4 + xcdaSize > blob.Length || xcdaSize % 4 != 0)
            throw new InvalidDataException("precompiled_charsmap has an invalid XCDA size.");
        _xcda = new uint[xcdaSize / 4];
        for (int i = 0; i < _xcda.Length; i++)
            _xcda[i] = BinaryPrimitives.ReadUInt32LittleEndian(blob.AsSpan(4 + i * 4));
        _replacements = blob[(4 + xcdaSize)..];
    }

    private uint Node(uint index) => index < (uint)_xcda.Length
        ? _xcda[index]
        : throw new InvalidDataException("Index out of array bounds in XCDA array.");

    private uint Base(uint index) { uint n = Node(index); return (n >> 10) << (int)((n & (1u << 9)) >> 6); }
    private uint LCheck(uint index) => Node(index) & ((1u << 31) | 0xff);
    private bool Leaf(uint index) => ((Node(index) >> 8) & 1) != 0;
    private uint Value(uint index) => Node(index) & ((1u << 31) - 1);

    public string Normalize(string text)
    {
        var input = Encoding.UTF8.GetBytes(text);
        var output = new List<byte>(input.Length + 8);
        int offset = 0;
        while (offset < input.Length)
        {
            int matchLen = 0;
            uint matchOffset = 0;
            if (_xcda.Length > 0)
            {
                uint node = Base(0);
                for (int p = offset; p < input.Length; p++)
                {
                    byte c = input[p];
                    if (c == 0) break;
                    node ^= c;
                    if (LCheck(node) != c) break;
                    bool leaf = Leaf(node);
                    node ^= Base(node);
                    if (leaf)
                    {
                        matchLen = p - offset + 1;
                        matchOffset = Value(node);
                    }
                }
            }

            if (matchLen > 0)
            {
                int end = (int)matchOffset;
                while (end < _replacements.Length && _replacements[end] != 0) end++;
                if (end == _replacements.Length) throw new InvalidDataException("Unterminated string in precompiled charsmap.");
                for (int i = (int)matchOffset; i < end; i++) output.Add(_replacements[i]);
                offset += matchLen;
                continue;
            }

            int scalarLen = ValidUtf8ScalarLength(input, offset);
            if (scalarLen > 0)
            {
                for (int i = 0; i < scalarLen; i++) output.Add(input[offset + i]);
                offset += scalarLen;
            }
            else
            {
                output.Add(0xEF); output.Add(0xBF); output.Add(0xBD);
                offset++;
            }
        }
        return Encoding.UTF8.GetString(output.ToArray());
    }

    private static int ValidUtf8ScalarLength(byte[] s, int i)
    {
        byte b = s[i];
        int len = b < 0x80 ? 1 : (b & 0xE0) == 0xC0 ? 2 : (b & 0xF0) == 0xE0 ? 3 : (b & 0xF8) == 0xF0 ? 4 : 0;
        if (len == 0 || i + len > s.Length) return 0;
        for (int k = 1; k < len; k++)
            if ((s[i + k] & 0xC0) != 0x80) return 0;
        return len;
    }
}
