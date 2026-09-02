using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.AceStep.Conditioning;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion.AceStep;

/// <summary>
/// Real numeric golden-parity check for <see cref="AceStepConditionEncoder.EncodeLyrics"/> against
/// the real `diffusers.pipelines.ace_step.modeling_ace_step.AceStepLyricEncoder` reference, loaded
/// with the SAME real `turbo.safetensors` `encoder.lyric_encoder.*` weights (remapped via a real,
/// mechanical key-rename verified with `load_state_dict(strict=False)` reporting zero missing/
/// unexpected keys) and run on a fixed-seed synthetic input. See
/// docs/064-acestep-implementation-plan.md's "Golden-parity" section for how the reference dump
/// (`golden_lyric_*.bin`) was produced -- not checked into the repo, regenerate via the Python
/// script referenced there if needed.
/// </summary>
public sealed class AceStepLyricEncoderGoldenParityTests
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

    private static float[] ReadBin(string path, int count)
    {
        var bytes = File.ReadAllBytes(path);
        Assert.Equal(count * 4, bytes.Length);
        var result = new float[count];
        Buffer.BlockCopy(bytes, 0, result, 0, bytes.Length);
        return result;
    }

    private static float[][] ToRows(float[] flat, int rows, int cols)
    {
        var result = new float[rows][];
        for (int r = 0; r < rows; r++)
        {
            result[r] = new float[cols];
            Array.Copy(flat, r * cols, result[r], 0, cols);
        }
        return result;
    }

    [Fact]
    public unsafe void EncodeLyrics_RealWeights_MatchesRealDiffusersReference()
    {
        string? turboPath = FindRepoFile("models/acestep-v15/turbo.safetensors");
        Assert.SkipUnless(turboPath != null, "models/acestep-v15/turbo.safetensors not found");

        string scratchDir = @"C:\Users\Dmitri\AppData\Local\Temp\claude\C--Git-Public-OpenTail-Stingray\6cb31b57-ce45-49d6-9926-8736cdcfcfa9\scratchpad\acestep";
        string inputPath = Path.Combine(scratchDir, "golden_lyric_input.bin");
        Assert.SkipUnless(File.Exists(inputPath), "golden_lyric_*.bin reference dump not found -- regenerate via golden_lyric.py (see docs/064)");

        using var loader = SafetensorsLoader.Open(turboPath!);
        var weights = AceStepConditionEncoderWeights.Load(loader);

        int seqLen = 20;
        var inputFlat = ReadBin(inputPath, seqLen * 1024);
        var expectedFlat = ReadBin(Path.Combine(scratchDir, "golden_lyric_output.bin"), seqLen * 2048);
        var rawEmbeds = ToRows(inputFlat, seqLen, 1024);
        var expected = ToRows(expectedFlat, seqLen, 2048);

        // Real pipeline: raw text_hidden_dim(1024) input -> embed_tokens (Linear WITH bias,
        // 1024->2048) -> the 8-layer encoder. Matches AceStepConditionEncoder.Forward's own
        // sequencing for the lyric path (see its doc comment).
        var projected = new float[seqLen][];
        for (int i = 0; i < seqLen; i++)
        {
            var row = new float[2048];
            fixed (float* rp = rawEmbeds[i], pp = row, bp = weights.LyricEmbedBias)
                weights.LyricEmbedWeight.MatMul(rp, 1, pp, bp);
            projected[i] = row;
        }

        var actual = AceStepConditionEncoder.EncodeLyrics(weights, projected);

        Assert.Equal(seqLen, actual.Length);

        double sumAbsDiff = 0, sumAbsExpected = 0, maxAbsDiff = 0;
        int count = 0;
        for (int t = 0; t < seqLen; t++)
        {
            for (int c = 0; c < 2048; c++)
            {
                double diff = Math.Abs(actual[t][c] - expected[t][c]);
                maxAbsDiff = Math.Max(maxAbsDiff, diff);
                sumAbsDiff += diff;
                sumAbsExpected += Math.Abs(expected[t][c]);
                count++;
            }
        }
        double relError = sumAbsDiff / sumAbsExpected;

        Assert.True(relError < 0.001, $"relative error {relError:F6} exceeds tolerance -- real numeric mismatch against diffusers reference (maxAbsDiff={maxAbsDiff:F6})");
    }
}
