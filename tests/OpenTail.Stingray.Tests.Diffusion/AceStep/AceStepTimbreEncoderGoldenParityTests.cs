using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.AceStep.Conditioning;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion.AceStep;

/// <summary>
/// Real numeric golden-parity check for <see cref="AceStepTimbreEncoder"/> against the real
/// `diffusers.pipelines.ace_step.modeling_ace_step.AceStepTimbreEncoder` reference, loaded with the
/// SAME real `turbo.safetensors` `encoder.timbre_encoder.*` weights (zero missing/unexpected keys)
/// and run on a fixed-seed synthetic input. See docs/064-acestep-implementation-plan.md's
/// "Golden-parity check" section for how the reference dump (`golden_timbre_*.bin`) was produced.
/// </summary>
public sealed class AceStepTimbreEncoderGoldenParityTests
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
    public void Forward_RealWeights_MatchesRealDiffusersReference()
    {
        string? turboPath = FindRepoFile("models/acestep-v15/turbo.safetensors");
        Assert.SkipUnless(turboPath != null, "models/acestep-v15/turbo.safetensors not found");

        string scratchDir = @"C:\Users\Dmitri\AppData\Local\Temp\claude\C--Git-Public-OpenTail-Stingray\6cb31b57-ce45-49d6-9926-8736cdcfcfa9\scratchpad\acestep";
        string inputPath = Path.Combine(scratchDir, "golden_timbre_input.bin");
        Assert.SkipUnless(File.Exists(inputPath), "golden_timbre_*.bin reference dump not found -- regenerate via golden_timbre.py (see docs/064)");

        using var loader = SafetensorsLoader.Open(turboPath!);
        var weights = AceStepTimbreEncoderWeights.Load(loader);

        int seqLen = 30;
        var inputFlat = ReadBin(inputPath, seqLen * 64);
        var expected = ReadBin(Path.Combine(scratchDir, "golden_timbre_output.bin"), 2048);
        var acousticLatent = ToRows(inputFlat, seqLen, 64);

        var actual = AceStepTimbreEncoder.Forward(weights, acousticLatent);

        Assert.Equal(2048, actual.Length);

        double sumAbsDiff = 0, sumAbsExpected = 0, maxAbsDiff = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            double diff = Math.Abs(actual[i] - expected[i]);
            maxAbsDiff = Math.Max(maxAbsDiff, diff);
            sumAbsDiff += diff;
            sumAbsExpected += Math.Abs(expected[i]);
        }
        double relError = sumAbsDiff / sumAbsExpected;

        Assert.True(relError < 0.001, $"relative error {relError:F6} exceeds tolerance -- real numeric mismatch against diffusers reference (maxAbsDiff={maxAbsDiff:F6})");
    }
}
