namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real-weight smoke test for MOSS-TTS-Nano's SentencePiece BPE text tokenizer: extracts the real
/// embedded `tokenizer.model` from the packed GGUF and runs it through
/// <see cref="SentencePieceBpeTokenizer"/>. Not a numeric golden-parity check against real Python
/// `sentencepiece` output (no such reference available in this environment) -- see
/// docs/audio-review-progress.md for that gap. Instead verifies real, checkable invariants: a
/// known real vocabulary piece round-trips to its own real vocab id, and byte-fallback produces
/// in-range ids for input outside the vocab.
/// </summary>
public sealed class MossTtsTextTokenizerRealWeightsTests : HeavyTestBase
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

    private static byte[]? ExtractEmbeddedFile(GgufModel model, string fileName)
    {
        if (!model.Metadata.TryGetValue("audiocpp.embedded_files.names", out var namesObj) || namesObj is not object[] names) return null;
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
        return null;
    }

    [Fact]
    public void Encode_OnRealCheckpoint_RoundTripsAKnownVocabPieceAndByteFallsBackCleanly()
    {
        string? path = FindRepoFile("models/_models/moss-tts-nano/MOSS-TTS-Nano-100M-GGUF/moss-tts-nano-100m-q8_0.gguf");
        Assert.SkipUnless(path != null, "moss-tts-nano checkpoint not found");

        using var model = GgufModel.Open(path!);
        var tokenizerBytes = ExtractEmbeddedFile(model, "tokenizer.model");
        Assert.NotNull(tokenizerBytes);

        var tok = SentencePieceBpeTokenizer.FromModelBytes(tokenizerBytes!);

        // Real piece confirmed via MossTtsSpmProtoDumpDebugTest's dump: vocab id 6599 is the real
        // piece "▁regular" (score -6328, NORMAL type). Encoding the plain word " regular" (leading
        // space -> real SentencePiece metaspace escaping) should therefore produce EXACTLY that one
        // token id, since the entire word is itself a single vocabulary piece.
        var ids = tok.Encode(" regular");
        Assert.Equal([6599], ids);

        // A rare CJK character unlikely to be a direct single-token vocab entry in a 16K English/
        // multilingual-BPE vocab still must produce a finite, in-range token sequence via real
        // UTF-8 byte fallback (checkpoint confirmed byte_fallback=true) -- never throw, never
        // produce an out-of-range id.
        var rareIds = tok.Encode("𓀀"); // U+13000, Egyptian hieroglyph, 4-byte UTF-8
        Assert.NotEmpty(rareIds);
        Assert.All(rareIds, id => Assert.InRange(id, 0, 16383));
    }
}
