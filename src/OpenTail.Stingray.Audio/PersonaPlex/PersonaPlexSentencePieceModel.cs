using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.PersonaPlex;

/// <summary>
/// Minimal, real (not guessed) hand-rolled protobuf wire-format reader for PersonaPlex's real
/// embedded `tokenizer_spm_32k_3.model` (a raw SentencePiece `ModelProto`, not a `tokenizer.json`
/// -- so this codebase's existing <see cref="UnigramTokenizer.FromTokenizerJson"/>/
/// <see cref="SentencePieceBpeTokenizer.FromModelBytes"/> loaders don't apply directly). Real
/// field numbers confirmed from the vendored `sentencepiece_model.proto` (not guessed):
/// `ModelProto.pieces` = field 1 (repeated LEN submessage), `ModelProto.trainer_spec` = field 2
/// (LEN), `TrainerSpec.model_type` = field 3 (VARINT, `UNIGRAM=1`/`BPE=2`/`WORD=3`/`CHAR=4`),
/// `SentencePiece.piece` = field 1 (LEN string), `SentencePiece.score` = field 2 (FIXED32 float),
/// `SentencePiece.type` = field 3 (VARINT enum, default `NORMAL=1` when absent -- and this proto
/// enum's numeric values are confirmed IDENTICAL to llama.cpp's own `llama_token_type` convention
/// `UnigramTokenizer.FromGgufVocab` already expects, so no remapping is needed).
///
/// <para>Real, confirmed via <c>PersonaPlexTokenizerModelTypeDebugTest</c>: this checkpoint's own
/// `trainer_spec.model_type` is `UNIGRAM` (not BPE) -- `Load` throws if a future checkpoint turns
/// out to declare a different real model type, rather than silently mis-tokenizing with the wrong
/// algorithm.</para>
/// </summary>
public static class PersonaPlexSentencePieceModel
{
    public static UnigramTokenizer Load(byte[] modelBytes)
    {
        var pieces = new List<string>();
        var scores = new List<float>();
        var types = new List<int>();
        int? modelType = null;

        foreach (var (field, wireType, value, start, len) in WalkFields(modelBytes, 0, modelBytes.Length))
        {
            if (field == 1 && wireType == 2) // pieces (repeated SentencePiece submessage)
            {
                string piece = "";
                float score = 0f;
                int type = 1; // real default NORMAL
                foreach (var (f2, wt2, v2, start2, len2) in WalkFields(modelBytes, start, start + len))
                {
                    if (f2 == 1 && wt2 == 2) piece = System.Text.Encoding.UTF8.GetString(modelBytes, start2, len2);
                    else if (f2 == 2 && wt2 == 5) score = BitConverter.ToSingle(modelBytes, start2);
                    else if (f2 == 3 && wt2 == 0) type = (int)v2;
                }
                pieces.Add(piece);
                scores.Add(score);
                types.Add(type);
            }
            else if (field == 2 && wireType == 2) // trainer_spec
            {
                foreach (var (f2, wt2, v2, _, _) in WalkFields(modelBytes, start, start + len))
                    if (f2 == 3 && wt2 == 0) modelType = (int)v2;
            }
        }

        if (modelType != 1)
            throw new NotSupportedException($"PersonaPlex tokenizer real trainer_spec.model_type={modelType} is not UNIGRAM(1) -- this loader assumes Unigram, confirmed for the currently-downloaded checkpoint; a different real model type needs a different tokenizer algorithm, not a silent mis-tokenization.");

        int unkId = types.IndexOf(2); // real SentencePiece.Type.UNKNOWN=2
        if (unkId < 0) throw new InvalidDataException("PersonaPlex tokenizer has no real UNKNOWN piece.");

        return UnigramTokenizer.FromGgufVocab([.. pieces], [.. scores], unkId, [.. types]);
    }

    // Minimal protobuf wire-format walk: yields (fieldNumber, wireType, varintValueOrLen, byteStart, byteLen)
    // for each top-level field in [start,end). wireType 2 (LEN) yields the submessage/string's own byte range.
    private static IEnumerable<(int Field, int WireType, long Value, int Start, int Len)> WalkFields(byte[] data, int start, int end)
    {
        int i = start;
        while (i < end)
        {
            long tag = ReadVarint(data, ref i);
            int field = (int)(tag >> 3);
            int wireType = (int)(tag & 0x7);
            switch (wireType)
            {
                case 0:
                {
                    int valStart = i;
                    long v = ReadVarint(data, ref i);
                    yield return (field, wireType, v, valStart, i - valStart);
                    break;
                }
                case 2:
                {
                    long len = ReadVarint(data, ref i);
                    yield return (field, wireType, len, i, (int)len);
                    i += (int)len;
                    break;
                }
                case 5: yield return (field, wireType, 0, i, 4); i += 4; break;
                case 1: yield return (field, wireType, 0, i, 8); i += 8; break;
                default: yield break;
            }
        }
    }

    private static long ReadVarint(byte[] data, ref int i)
    {
        long result = 0; int shift = 0;
        while (true)
        {
            byte b = data[i++];
            result |= (long)(b & 0x7F) << shift;
            if ((b & 0x80) == 0) break;
            shift += 7;
        }
        return result;
    }
}
