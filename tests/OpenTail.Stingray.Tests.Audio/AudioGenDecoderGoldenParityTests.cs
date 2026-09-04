using OpenTail.Stingray.Audio.AudioGen;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric golden verification for <see cref="AudioGenTransformer"/>'s decoder forward pass
/// -- compares against `scratch-llamacpp-ref/audiogen_decoder_golden.py`, which uses the real,
/// already-local `models/audiogen-medium/audiogen-medium-lm.safetensors` checkpoint and computes
/// the real decoder math directly in numpy, transcribed from the real `audiocraft.modules
/// .transformer`/`audiocraft.models.lm` source. Closes the gap noted in
/// docs/063-audiogen-implementation-plan.md: AudioGen end-to-end generation was previously
/// confirmed correct by ear only (no numeric reference existed), unlike Parler-TTS/F5-TTS/Fish
/// Speech which already have this same style of golden-parity test.
///
/// Calls the real production <see cref="AudioGenTransformer.PrepareCrossAttention"/> and
/// <see cref="AudioGenTransformer.Step"/> methods directly (not a reimplementation) with the same
/// fixed codebook token ids and fake encoder hidden state the oracle used.
/// </summary>
public sealed class AudioGenDecoderGoldenParityTests : HeavyTestBase
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
    public void Step_RealWeights_MatchesGoldenOutput()
    {
        string? modelPath = FindRepoFile("models/audiogen-medium/audiogen-medium-lm.safetensors");
        Assert.SkipUnless(modelPath != null, "models/audiogen-medium/audiogen-medium-lm.safetensors not found");

        string? idsPath = FindRepoFile("scratch-llamacpp-ref/audiogen_decoder_golden_codebook_ids.txt");
        string? hiddenPath = FindRepoFile("scratch-llamacpp-ref/audiogen_decoder_golden_hidden.txt");
        Assert.SkipUnless(idsPath != null && hiddenPath != null,
            "golden AudioGen decoder files not found (re-run scratch-llamacpp-ref/audiogen_decoder_golden.py)");

        var idLines = File.ReadAllText(idsPath!).Trim().Split('\n');
        int t = idLines.Length;
        var codebookIds = new int[t][];
        for (int i = 0; i < t; i++) codebookIds[i] = Array.ConvertAll(idLines[i].Split(','), int.Parse);

        var hiddenLines = File.ReadAllText(hiddenPath!).Split('\n');
        var dims = hiddenLines[0].Trim().Split(',');
        int goldenT = int.Parse(dims[0]);
        int goldenCodebooks = int.Parse(dims[1]);
        int goldenCodebookSize = int.Parse(dims[2]);
        var goldenParts = hiddenLines[1].Trim().Split(',');
        Assert.Equal(goldenT * goldenCodebooks * goldenCodebookSize, goldenParts.Length);
        var golden = new float[goldenT * goldenCodebooks * goldenCodebookSize];
        for (int i = 0; i < golden.Length; i++) golden[i] = float.Parse(goldenParts[i]);

        var sw = System.Diagnostics.Stopwatch.StartNew();

        using var loader = SafetensorsLoader.Open(modelPath!);
        var weights = new AudioGenTransformerWeights(loader);

        // Fake "encoder" (T5-large description conditioner) hidden state stand-in, matching the
        // oracle's deterministic 4-position, all-0.05 raw T5-dim (1024) input -- PrepareCrossAttention
        // itself performs the real output_proj (1024 -> 1536, WITH bias).
        var encoderHiddenRaw = new float[4][];
        for (int i = 0; i < 4; i++)
        {
            var row = new float[AudioGenConfig.TextDModel];
            for (int d = 0; d < row.Length; d++) row[d] = 0.05f;
            encoderHiddenRaw[i] = row;
        }

        var cache = new AudioGenTransformer.KvCache();
        AudioGenTransformer.PrepareCrossAttention(weights, encoderHiddenRaw, cache);

        // Step() returns real per-codebook logits [codebook][CodebookSize] -- compared directly
        // against the oracle's logits fixture (the oracle runs the same real out_norm + lm_heads
        // stage, so logits are the natural, fully-real comparison point without needing to expose
        // an internal hidden state from production code).
        var output = new float[t][][];
        for (int i = 0; i < t; i++)
            output[i] = AudioGenTransformer.Step(weights, codebookIds[i], cache);

        sw.Stop();
        // A real 48-layer, 1536-dim decoder forward over multiple timesteps with real weight
        // loading should take a real, non-trivial amount of wall-clock time -- a sub-something
        // run here would indicate a silent no-op rather than a real run (see CLAUDE.md rule 12).
        Assert.True(sw.ElapsedMilliseconds > 50, $"suspiciously fast run ({sw.ElapsedMilliseconds}ms) -- did this actually execute against real weights?");

        Assert.Equal(goldenT, output.Length);
        Assert.Equal(AudioGenConfig.NumCodebooks, goldenCodebooks);
        Assert.Equal(AudioGenConfig.CodebookSize, goldenCodebookSize);

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < goldenT; i++)
        {
            for (int q = 0; q < goldenCodebooks; q++)
            {
                for (int c = 0; c < goldenCodebookSize; c++)
                {
                    float a = output[i][q][c];
                    float b = golden[(i * goldenCodebooks + q) * goldenCodebookSize + c];
                    dot += a * b;
                    normA += a * a;
                    normB += b * b;
                }
            }
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.True(cosine > 0.999, $"cosine similarity {cosine} too low vs golden AudioGen decoder logits");
    }
}
