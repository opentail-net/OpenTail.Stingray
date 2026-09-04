using OpenTail.Stingray.Audio.AudioGen;
using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric golden verification for AudioGen's text conditioner -- the shared, non-gated
/// <see cref="T5EncoderKernels"/> forward pass run with AudioGen's real, stock, external
/// `t5-large` checkpoint (models/audiogen-medium/t5-large.safetensors). Compares against
/// `scratch-llamacpp-ref/audiogen_t5_large_golden.py`, a pure-numpy oracle transcribed from the
/// real HuggingFace `transformers/models/t5/modeling_t5.py` T5 encoder math (relative-position-bias
/// self-attention with no 1/sqrt(d) scaling, non-mean-centering RMSNorm `T5LayerNorm`, plain-ReLU
/// non-gated `T5DenseActDense` FFN).
///
/// This is a DIFFERENT real checkpoint/dims than <see cref="T5EncoderTests"/> (which golden-checks
/// Parler-TTS's fine-tuned, GATED flan-t5-large text encoder) and than MusicGen's bundled
/// `t5-base` -- this test is what actually closes numeric verification for
/// <see cref="AudioGenTextEncoderWeights"/>/the shared kernel's non-gated code path with AudioGen's
/// real dims (d_model=1024, d_ff=4096, num_layers=24, num_heads=16).
/// </summary>
public sealed class AudioGenTextEncoderGoldenParityTests : HeavyTestBase
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
        string? modelPath = FindRepoFile("models/audiogen-medium/t5-large.safetensors");
        Assert.SkipUnless(modelPath != null, "models/audiogen-medium/t5-large.safetensors not found");

        string? idsPath = FindRepoFile("scratch-llamacpp-ref/audiogen_t5_large_golden_input_ids.txt");
        string? outPath = FindRepoFile("scratch-llamacpp-ref/audiogen_t5_large_golden_output.txt");
        Assert.SkipUnless(idsPath != null && outPath != null,
            "golden AudioGen T5 files not found (re-run scratch-llamacpp-ref/audiogen_t5_large_golden.py)");

        var idsCsv = File.ReadAllText(idsPath!).Trim().Split(',');
        var tokenIds = new int[idsCsv.Length];
        for (int i = 0; i < idsCsv.Length; i++) tokenIds[i] = int.Parse(idsCsv[i]);

        var lines = File.ReadAllText(outPath!).Split('\n');
        var dims = lines[0].Trim().Split(',');
        int goldenT = int.Parse(dims[0]);
        int goldenDim = int.Parse(dims[1]);
        var goldenParts = lines[1].Trim().Split(',');
        Assert.Equal(goldenT * goldenDim, goldenParts.Length);
        var golden = new float[goldenT * goldenDim];
        for (int i = 0; i < golden.Length; i++) golden[i] = float.Parse(goldenParts[i]);

        var sw = System.Diagnostics.Stopwatch.StartNew();

        using var loader = SafetensorsLoader.Open(modelPath!);
        var weights = AudioGenTextEncoderWeights.Load(loader);
        var output = T5EncoderKernels.Forward(AudioGenTextEncoderWeights.Dims, weights, tokenIds);

        sw.Stop();
        // A real 24-layer, 1024-dim T5-large encoder forward with real weight loading should take
        // a real, non-trivial amount of wall-clock time -- see CLAUDE.md rule 12.
        Assert.True(sw.ElapsedMilliseconds > 50, $"suspiciously fast run ({sw.ElapsedMilliseconds}ms) -- did this actually execute against real weights?");

        Assert.Equal(goldenT, output.Length);
        Assert.Equal(goldenDim, output[0].Length);

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < goldenT; i++)
        {
            for (int d = 0; d < goldenDim; d++)
            {
                float a = output[i][d];
                float b = golden[i * goldenDim + d];
                dot += a * b;
                normA += a * a;
                normB += b * b;
            }
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.True(cosine > 0.999, $"cosine similarity {cosine} too low vs golden AudioGen T5-large encoder output");
    }
}
