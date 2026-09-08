using System.Text;

namespace OpenTail.Stingray.Audio.Citrinet;

/// <summary>
/// Minimal, DECODE-ONLY SentencePiece vocabulary reader for Citrinet ASR's CTC output ids -&gt;
/// text. Citrinet's real decode is a plain id-to-piece lookup (NeMo's CTC tokenizer never
/// re-encodes), so this deliberately does NOT reuse the full BPE-merge
/// <see cref="OpenTail.Stingray.Core.SentencePieceBpeTokenizer"/> (encode-only, no id-&gt;piece
/// exposed) -- just a lightweight parallel reader over the same real `ModelProto` wire format
/// (field 1 = repeated `SentencePiece{text, score, type}`, piece INDEX IN FILE ORDER == the real
/// vocab id, confirmed by `citrinet_asr/assets.cpp`'s own `tokenizer_pieces.size() ==
/// config.vocab_size` check). Real SentencePiece text convention: `▁` (U+2581) marks a
/// word-start/space, replaced with an ASCII space and the whole result trimmed.
/// </summary>
public static class CitrinetSentencePieceVocab
{
    public static string[] LoadPieces(byte[] data)
    {
        var pieces = new List<string>();
        int i = 0;
        while (i < data.Length)
        {
            long tag = ReadVarint(data, ref i);
            int fieldNo = (int)(tag >> 3);
            int wire = (int)(tag & 7);

            if (wire == 0) { ReadVarint(data, ref i); }
            else if (wire == 5) { i += 4; }
            else if (wire == 1) { i += 8; }
            else if (wire == 2)
            {
                long len = ReadVarint(data, ref i);
                int start = i;
                i += (int)len;
                if (fieldNo == 1)
                    pieces.Add(ParsePieceText(data, start, (int)len));
            }
            else
            {
                throw new InvalidDataException($"Unexpected protobuf wire type {wire} at offset {i}.");
            }
        }
        return [.. pieces];
    }

    private static string ParsePieceText(byte[] data, int start, int len)
    {
        string text = "";
        int j = start;
        while (j < start + len)
        {
            long subTag = ReadVarint(data, ref j);
            int subField = (int)(subTag >> 3);
            int subWire = (int)(subTag & 7);
            if (subWire == 5) j += 4;
            else if (subWire == 0) ReadVarint(data, ref j);
            else if (subWire == 2)
            {
                long subLen = ReadVarint(data, ref j);
                if (subField == 1) text = Encoding.UTF8.GetString(data, j, (int)subLen);
                j += (int)subLen;
            }
        }
        return text;
    }

    private static long ReadVarint(byte[] data, ref int i)
    {
        long result = 0;
        int shift = 0;
        while (true)
        {
            byte b = data[i++];
            result |= (long)(b & 0x7F) << shift;
            if ((b & 0x80) == 0) break;
            shift += 7;
        }
        return result;
    }

    /// <summary>Real detokenize: join pieces for the given ids, replace `▁` (U+2581) with a
    /// space, trim.</summary>
    public static string Decode(string[] pieces, IEnumerable<int> ids)
    {
        var sb = new StringBuilder();
        foreach (int id in ids)
        {
            if (id < 0 || id >= pieces.Length) continue;
            sb.Append(pieces[id]);
        }
        return sb.ToString().Replace('▁', ' ').Trim();
    }
}
