using OpenTail.Stingray.Audio.MusicGen;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric golden verification for <see cref="MusicGenTransformer"/> -- compares against
/// `scratch-llamacpp-ref/musicgen_decoder_golden.py`, which uses the real, already-local
/// `models/musicgen-small/musicgen-small.safetensors` and computes the real 24-layer decoder
/// math directly in numpy, transcribed from the real `transformers` `modeling_musicgen.py`.
/// Closes the gap noted in docs/audio-review-progress.md: MusicGen's decoder forward pass was
/// previously confirmed correct BY EAR only, never against a real numeric reference.
///
/// <para><see cref="MusicGenTransformer.Step"/> only exposes per-codebook logits publicly (not
/// the pre-lm_head hidden state), so this test compares codebook-0 logits against the oracle's
/// `logits[0]` rather than the raw hidden state -- both sides run through the identical real
/// `decoder.lm_heads.0.weight` on top of the identical hidden state, so a logits-space cosine
/// match is just as strong a parity signal as a hidden-space one.</para>
/// </summary>
public sealed class MusicGenDecoderGoldenParityTests : HeavyTestBase
{
    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Forward_RealWeights_MatchesGoldenOutput()
    {
        string? modelPath = FindRepoFile("models/musicgen-small/musicgen-small.safetensors");
        Assert.SkipUnless(modelPath != null, "models/musicgen-small/musicgen-small.safetensors not found");

        string? idsPath = FindRepoFile("scratch-llamacpp-ref/musicgen_decoder_golden_codebook_ids.txt");
        string? logitsPath = FindRepoFile("scratch-llamacpp-ref/musicgen_decoder_golden_logits0.txt");
        Assert.SkipUnless(idsPath != null && logitsPath != null,
            "golden MusicGen decoder files not found (re-run scratch-llamacpp-ref/musicgen_decoder_golden.py)");

        var idLines = File.ReadAllText(idsPath!).Trim().Split('\n');
        int t = idLines.Length;
        var codebookIds = new int[t][];
        for (int i = 0; i < t; i++) codebookIds[i] = Array.ConvertAll(idLines[i].Split(','), int.Parse);

        var logitsLines = File.ReadAllText(logitsPath!).Split('\n');
        var dims = logitsLines[0].Trim().Split(',');
        int goldenT = int.Parse(dims[0]);
        int goldenVocab = int.Parse(dims[1]);
        var goldenParts = logitsLines[1].Trim().Split(',');
        Assert.Equal(goldenT * goldenVocab, goldenParts.Length);
        var golden = new float[goldenT * goldenVocab];
        for (int i = 0; i < golden.Length; i++) golden[i] = float.Parse(goldenParts[i]);

        Assert.Equal(goldenT, t);
        Assert.Equal(MusicGenConfig.CodebookSize, goldenVocab);

        using var loader = SafetensorsLoader.Open(modelPath!);
        var weights = new MusicGenTransformerWeights(loader);

        // Fake "encoder" hidden matching the oracle's deterministic 5-position, all-0.05 T5
        // stand-in -- PrepareCrossAttention runs it through the real enc_to_dec_proj internally.
        var encoderHidden = new float[5][];
        for (int i = 0; i < 5; i++)
        {
            var row = new float[MusicGenConfig.TextDModel];
            for (int d = 0; d < row.Length; d++) row[d] = 0.05f;
            encoderHidden[i] = row;
        }

        var cache = new MusicGenTransformer.KvCache();
        MusicGenTransformer.PrepareCrossAttention(weights, encoderHidden, cache);

        var output = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var logits = MusicGenTransformer.Step(weights, codebookIds[i], cache);
            output[i] = logits[0];
        }

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < goldenT; i++)
        {
            for (int d = 0; d < goldenVocab; d++)
            {
                float a = output[i][d];
                float b = golden[i * goldenVocab + d];
                dot += a * b;
                normA += a * a;
                normB += b * b;
            }
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.True(cosine > 0.999, $"cosine similarity {cosine} too low vs golden MusicGen decoder output");
    }
}
