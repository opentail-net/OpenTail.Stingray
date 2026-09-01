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
    /// Real Viterbi-optimal SentencePiece Unigram segmentation (fixed 2026-09-01, replacing a
    /// greedy longest-match approximation that was confirmed to diverge from HuggingFace's real
    /// `T5TokenizerFast` on this exact prompt at "fox": real `[..., 9, 1131, 3, 20400, 1180, ...]`
    /// vs. the old greedy algorithm's wrong `[..., 9, 1131, 5575, 226, 1180, ...]`). Matches
    /// `docs/00-current-work.md`'s formerly-tracked "Unigram-tokenizer" backlog item -- closed by
    /// this fix.
    /// </summary>
    [Fact]
    public void T5Tokenizer_MatchesRealT5TokenizerFast_ViaViterbiUnigram()
    {
        string? tokenizerJson = FindRepoFile(TokenizerJsonRelative);
        string? goldenDir = FindGoldenDir();
        if (tokenizerJson is null || goldenDir is null) return; // skip: needs local T5 tokenizer + fixtures

        var idsBytes = File.ReadAllBytes(Path.Combine(goldenDir, "ids.bin"));
        var goldenIds = new int[idsBytes.Length / 4];
        Buffer.BlockCopy(idsBytes, 0, goldenIds, 0, idsBytes.Length);

        var tokenizer = T5Tokenizer.FromFile(tokenizerJson, maxLen: 256);
        var ids = tokenizer.Tokenize("A cinematic shot of a red fox running through snow");

        Assert.Equal(goldenIds, ids);
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
