using OpenTail.Stingray.Diffusion.TextEncoders;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Numeric parity check of the existing (Flux-built) <see cref="T5Encoder"/>/<see cref="T5Tokenizer"/>
/// against a golden encode dumped from HuggingFace `transformers`' real `T5EncoderModel`, loaded
/// with the real `google/t5-v1_1-xxl`-architecture text encoder LTX-Video ships
/// (`Lightricks/LTX-Video`'s own `text_encoder/` + `tokenizer/` subfolders -- downloaded locally to
/// `models/ltx-t5/`, config-confirmed: 24 layers, d_model=4096, 64 heads, matching
/// <see cref="T5Encoder"/>'s existing hardcoded constants exactly). This is the same encoder
/// architecture family this project already built for FLUX -- LTX just needed the real weights
/// wired up, per docs/055-ltx-video-implementation-plan.md step 6.
/// </summary>
public sealed class LtxT5EncoderGoldenParityTests
{
    private const string TextEncoderDirRelative = "models/ltx-t5/text_encoder";
    private const string TokenizerJsonRelative = "models/ltx-t5/tokenizer/tokenizer.json";

    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath.Replace('/', Path.DirectorySeparatorChar));
            if (File.Exists(p) || Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static string? FindGoldenDir()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "tests", "OpenTail.Stingray.Tests.Diffusion", "TestData", "LtxT5Golden");
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        return (float)(dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12));
    }

    /// <summary>
    /// KNOWN, PRE-EXISTING GAP (not introduced by the LTX work, found while verifying it): real
    /// T5/SentencePiece Unigram tokenization is Viterbi-OPTIMAL (picks the globally
    /// highest-log-probability segmentation), but <see cref="T5Tokenizer.Tokenize"/> does a GREEDY
    /// longest-match instead -- confirmed to diverge on this real prompt at "fox" (real:
    /// `[..., 9, 1131, 3, 20400, 1180, ...]`; greedy: `[..., 9, 1131, 5575, 226, 1180, ...]`, a
    /// different, wrong split). This matches `docs/00-current-work.md`'s tracked "Unigram-tokenizer"
    /// backlog item -- out of scope for the LTX-Video port itself (the T5 ENCODER math is verified
    /// correct against real token ids in the test below; only the tokenizer's own segmentation
    /// algorithm needs the real Viterbi fix, tracked separately). Documents the gap with a real
    /// example rather than silently passing or being deleted.
    /// </summary>
    [Fact]
    public void T5Tokenizer_GreedyLongestMatch_DivergesFromRealViterbiUnigram_KnownGap()
    {
        string? tokenizerJson = FindRepoFile(TokenizerJsonRelative);
        string? goldenDir = FindGoldenDir();
        if (tokenizerJson is null || goldenDir is null) return; // skip: needs local T5 tokenizer + fixtures

        var idsBytes = File.ReadAllBytes(Path.Combine(goldenDir, "ids.bin"));
        var goldenIds = new int[idsBytes.Length / 4];
        Buffer.BlockCopy(idsBytes, 0, goldenIds, 0, idsBytes.Length);

        var tokenizer = T5Tokenizer.FromFile(tokenizerJson, maxLen: 256);
        var ids = tokenizer.Tokenize("A cinematic shot of a red fox running through snow");

        // Documents the CURRENT (imperfect) behavior -- flip to Assert.Equal once the tokenizer's
        // greedy algorithm is replaced with real Viterbi Unigram decoding.
        Assert.NotEqual(goldenIds, ids);
    }

    [Fact]
    public void T5Encoder_MatchesRealTransformersReference_OnRealTokenIds()
    {
        string? textEncoderDir = FindRepoFile(TextEncoderDirRelative);
        string? goldenDir = FindGoldenDir();
        if (textEncoderDir is null || goldenDir is null) return; // skip: needs local T5 weights + fixtures

        // T5Encoder's sharded-safetensors loading goes through SafetensorsLoader.OpenDirectory,
        // which reads the real `model.safetensors.index.json` HF shards this checkpoint uses.
        var indexPath = Path.Combine(textEncoderDir, "model.safetensors.index.json");
        Assert.True(File.Exists(indexPath), $"expected sharded T5 checkpoint at {indexPath}");

        using var st = SafetensorsLoader.OpenDirectory(textEncoderDir);
        using var encoder = T5Encoder.FromLoader(st);

        var idsBytes = File.ReadAllBytes(Path.Combine(goldenDir, "ids.bin"));
        var ids = new int[idsBytes.Length / 4];
        Buffer.BlockCopy(idsBytes, 0, ids, 0, idsBytes.Length);

        var outBytes = File.ReadAllBytes(Path.Combine(goldenDir, "output.bin"));
        var goldenOutput = new float[outBytes.Length / 4];
        Buffer.BlockCopy(outBytes, 0, goldenOutput, 0, outBytes.Length);

        var output = encoder.Encode(ids);

        Assert.Equal(goldenOutput.Length, output.Length);
        float cos = CosineSimilarity(output, goldenOutput);
        Assert.True(cos > 0.999f, $"T5 encoder cosine-sim too low: {cos}");
    }
}
