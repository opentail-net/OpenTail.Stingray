
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>ONE-OFF DEBUG: minimal protobuf wire-format scanner for MOSS-TTS-Nano's real
/// tokenizer.model (SentencePiece ModelProto) -- dumps top-level field numbers/wire-types plus
/// TrainerSpec.model_type (field 3 of TrainerSpec, varint: 1=UNIGRAM,2=BPE,3=WORD,4=CHAR) and
/// NormalizerSpec presence/precompiled_charsmap length, to confirm the real algorithm before
/// implementing a tokenizer port (never guessed).</summary>
public sealed class MossTtsSpmProtoDumpDebugTest : HeavyTestBase
{
    private static string? FindRepoFile(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
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

    [Fact]
    public void DumpModelProtoStructure()
    {
        string? path = FindRepoFile("moss-tts-tokenizer.model");
        Assert.SkipUnless(path != null, "moss-tts-tokenizer.model not found (run MossTtsMetaDumpDebugTest first)");
        var data = File.ReadAllBytes(path!);

        int i = 0;
        var pieceCount = 0;
        var topLevelCounts = new Dictionary<int, int>();
        int trainerSpecModelType = -1;
        int normalizerCharsmapLen = -1;
        string normalizerName = "";
        var normalizerBoolFields = new Dictionary<int, long>();
        string firstPieceText = "";
        float firstPieceScore = 0;
        var pieceTypeCounts = new Dictionary<int, int>();
        var sampleLines = new List<string>();
        long trainerByteFallback = -1;

        while (i < data.Length)
        {
            long tag = ReadVarint(data, ref i);
            int fieldNo = (int)(tag >> 3);
            int wire = (int)(tag & 7);
            topLevelCounts[fieldNo] = topLevelCounts.GetValueOrDefault(fieldNo) + 1;

            if (wire == 0) { ReadVarint(data, ref i); }
            else if (wire == 5) { i += 4; }
            else if (wire == 1) { i += 8; }
            else if (wire == 2)
            {
                long len = ReadVarint(data, ref i);
                int start = i;
                i += (int)len;

                if (fieldNo == 1) // SentencePiece (repeated)
                {
                    pieceCount++;
                    int j = start;
                    int pieceType = 1; // default NORMAL
                    string pieceText = "";
                    float pieceScore = 0;
                    while (j < start + len)
                    {
                        long subTag = ReadVarint(data, ref j);
                        int subField = (int)(subTag >> 3);
                        int subWire = (int)(subTag & 7);
                        if (subWire == 5) // fixed32 = score (float)
                        {
                            if (subField == 2) pieceScore = BitConverter.ToSingle(data, j);
                            j += 4;
                        }
                        else if (subWire == 0)
                        {
                            long v = ReadVarint(data, ref j);
                            if (subField == 3) pieceType = (int)v;
                        }
                        else if (subWire == 2)
                        {
                            long subLen = ReadVarint(data, ref j);
                            if (subField == 1) pieceText = System.Text.Encoding.UTF8.GetString(data, j, (int)subLen);
                            j += (int)subLen;
                        }
                    }
                    pieceTypeCounts[pieceType] = pieceTypeCounts.GetValueOrDefault(pieceType) + 1;
                    if (pieceCount <= 5 || pieceCount == 6600) sampleLines.Add($"[{pieceCount - 1}] '{pieceText}' score={pieceScore} type={pieceType}");
                    if (pieceCount == 1) { firstPieceText = pieceText; firstPieceScore = pieceScore; }
                }
                else if (fieldNo == 2) // TrainerSpec
                {
                    int j = start;
                    while (j < start + len)
                    {
                        long subTag = ReadVarint(data, ref j);
                        int subField = (int)(subTag >> 3);
                        int subWire = (int)(subTag & 7);
                        if (subWire == 0)
                        {
                            long v = ReadVarint(data, ref j);
                            if (subField == 3) trainerSpecModelType = (int)v; // model_type enum
                            if (subField == 35) trainerByteFallback = v; // byte_fallback
                        }
                        else if (subWire == 5) j += 4;
                        else if (subWire == 1) j += 8;
                        else if (subWire == 2)
                        {
                            long subLen = ReadVarint(data, ref j);
                            j += (int)subLen;
                        }
                    }
                }
                else if (fieldNo == 3) // NormalizerSpec
                {
                    int j = start;
                    while (j < start + len)
                    {
                        long subTag = ReadVarint(data, ref j);
                        int subField = (int)(subTag >> 3);
                        int subWire = (int)(subTag & 7);
                        if (subWire == 0)
                        {
                            long v = ReadVarint(data, ref j);
                            normalizerBoolFields[subField] = v;
                        }
                        else if (subWire == 5) j += 4;
                        else if (subWire == 1) j += 8;
                        else if (subWire == 2)
                        {
                            long subLen = ReadVarint(data, ref j);
                            if (subField == 2) normalizerCharsmapLen = (int)subLen; // precompiled_charsmap
                            if (subField == 1) normalizerName = System.Text.Encoding.UTF8.GetString(data, j, (int)subLen);
                            j += (int)subLen;
                        }
                    }
                }
            }
            else throw new InvalidDataException($"unexpected wire type {wire} at field {fieldNo}, offset {i}");
        }

        string? outDir = FindRepoFile("docs/audio-review-progress.md");
        string outPath = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(outDir!)!, "..", "moss-tts-spm-proto-dump.txt"));
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"top-level field counts: {string.Join(", ", topLevelCounts.Select(kv => $"{kv.Key}={kv.Value}"))}");
        sb.AppendLine($"piece count: {pieceCount}");
        sb.AppendLine($"first piece: '{firstPieceText}' score={firstPieceScore}");
        sb.AppendLine($"trainer_spec.model_type: {trainerSpecModelType} (1=UNIGRAM,2=BPE,3=WORD,4=CHAR)");
        sb.AppendLine($"trainer_spec.byte_fallback: {trainerByteFallback}");
        sb.AppendLine($"piece type histogram (1=NORMAL,2=UNK,3=CONTROL,4=USER_DEFINED,5=UNUSED,6=BYTE): {string.Join(", ", pieceTypeCounts.Select(kv => $"{kv.Key}={kv.Value}"))}");
        sb.AppendLine("sample pieces:");
        foreach (var line in sampleLines) sb.AppendLine("  " + line);
        sb.AppendLine($"normalizer_spec.precompiled_charsmap length: {normalizerCharsmapLen}");
        sb.AppendLine($"normalizer_spec.name: '{normalizerName}'");
        sb.AppendLine($"normalizer_spec bool/varint fields: {string.Join(", ", normalizerBoolFields.Select(kv => $"{kv.Key}={kv.Value}"))}");
        File.WriteAllText(outPath, sb.ToString());
        Console.Error.WriteLine($"[MossTtsSpmProto] wrote {outPath}");
        Console.Error.WriteLine(sb.ToString());
    }
}
