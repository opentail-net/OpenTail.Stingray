namespace OpenTail.Stingray.Tests.Audio;

/// <summary>ONE-OFF DEBUG: reads PersonaPlex's real embedded `tokenizer_spm_32k_3.model`
/// SentencePiece protobuf and extracts `trainer_spec.model_type` (real field numbers confirmed
/// from the vendored `sentencepiece_model.proto`: `ModelProto.trainer_spec` is field 2 (LEN),
/// `TrainerSpec.model_type` is field 3 (VARINT), `UNIGRAM=1`/`BPE=2`/`WORD=3`/`CHAR=4`) -- resolves
/// the real scoped gap from docs/audio-review-progress.md's "system-prompt SentencePiece
/// model-type gap" entry: this codebase's SentencePieceBpeTokenizer/UnigramTokenizer classes each
/// hard-code one algorithm and don't read this field themselves, so picking the wrong one would
/// silently mis-tokenize.</summary>
public sealed class PersonaPlexTokenizerModelTypeDebugTest : HeavyTestBase
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

    private static byte[] ExtractEmbeddedFile(GgufModel model, string fileName)
    {
        var namesObj = model.Metadata["audiocpp.embedded_files.names"];
        var names = (object[])namesObj;
        var offsets = (object[])model.Metadata["audiocpp.embedded_files.offsets"];
        var data = (object[])model.Metadata["audiocpp.embedded_files.data"];
        var bytes = data.Select(o => (byte)Convert.ToInt64(o)).ToArray();
        for (int i = 0; i < names.Length; i++)
        {
            if ((string)names[i] != fileName) continue;
            long start = Convert.ToInt64(offsets[i]);
            long end = i + 1 < offsets.Length ? Convert.ToInt64(offsets[i + 1]) : bytes.Length;
            return bytes[(int)start..(int)end];
        }
        throw new InvalidOperationException($"Embedded file '{fileName}' not found.");
    }

    // Minimal protobuf wire-format walk: returns (fieldNumber, wireType, valuePosition) enumerated
    // top-level, or recurses into a LEN submessage's byte range.
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
                case 0: // varint
                {
                    int valStart = i;
                    long v = ReadVarint(data, ref i);
                    yield return (field, wireType, v, valStart, i - valStart);
                    break;
                }
                case 2: // LEN
                {
                    long len = ReadVarint(data, ref i);
                    yield return (field, wireType, len, i, (int)len);
                    i += (int)len;
                    break;
                }
                case 5: i += 4; break; // fixed32
                case 1: i += 8; break; // fixed64
                default: yield break; // unsupported wire type, stop (shouldn't happen for this proto)
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

    [Fact]
    public void ReadTrainerSpecModelType()
    {
        string? path = FindRepoFile("models/_models/personaplex/PersonaPlex-GGUF/personaplex-7b-v1-q8_0.gguf");
        Assert.SkipUnless(path != null, "personaplex-7b-v1-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var bytes = ExtractEmbeddedFile(model, "tokenizer_spm_32k_3.model");
        Console.WriteLine($"tokenizer_spm_32k_3.model: {bytes.Length} bytes");

        int? modelType = null;
        foreach (var (field, wireType, value, start, len) in WalkFields(bytes, 0, bytes.Length))
        {
            if (field == 2 && wireType == 2) // trainer_spec (LEN submessage)
            {
                foreach (var (f2, wt2, v2, _, _) in WalkFields(bytes, start, start + len))
                {
                    if (f2 == 3 && wt2 == 0) // model_type (varint)
                    {
                        modelType = (int)v2;
                    }
                }
            }
        }

        Assert.NotNull(modelType);
        string name = modelType switch { 1 => "UNIGRAM", 2 => "BPE", 3 => "WORD", 4 => "CHAR", _ => "UNKNOWN" };
        Console.WriteLine($"PersonaPlex real trainer_spec.model_type = {modelType} ({name})");
    }
}
