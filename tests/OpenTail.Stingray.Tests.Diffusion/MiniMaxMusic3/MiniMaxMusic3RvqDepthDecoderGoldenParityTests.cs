using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.MiniMaxMusic3;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion.MiniMaxMusic3;

/// <summary>
/// Real numeric golden-parity check for <see cref="MiniMaxMusic3RvqDepthDecoder"/> against the real
/// `diffusers.MiniMaxMusic3RVQDepthDecoder` reference, loaded with the SAME real
/// `rvq_depth_decoder/diffusion_pytorch_model.safetensors` weights (zero missing/unexpected keys)
/// and run on a fixed-seed synthetic input. See docs/066-minimax-music3-future-plan.md.
/// </summary>
public sealed class MiniMaxMusic3RvqDepthDecoderGoldenParityTests
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
        string? weightsPath = FindRepoFile("models/minimax-music3/rvq_depth_decoder.safetensors");
        Assert.SkipUnless(weightsPath != null, "models/minimax-music3/rvq_depth_decoder.safetensors not found");

        string scratchDir = @"C:\Users\Dmitri\AppData\Local\Temp\claude\C--Git-Public-OpenTail-Stingray\6cb31b57-ce45-49d6-9926-8736cdcfcfa9\scratchpad";
        string inputPath = Path.Combine(scratchDir, "minimax_rvqdepth_input.bin");
        Assert.SkipUnless(File.Exists(inputPath), "minimax_rvqdepth_*.bin reference dump not found");

        using var loader = SafetensorsLoader.Open(weightsPath!);
        var weights = MiniMaxMusic3RvqDepthDecoderWeights.Load(loader);

        int steps = 8;
        int hidden = MiniMaxMusic3Config.RvqDepthDecoderHiddenSize;
        var inputFlat = ReadBin(inputPath, steps * hidden);
        var expectedFlat = ReadBin(Path.Combine(scratchDir, "minimax_rvqdepth_output.bin"), steps * hidden);
        var inputsEmbeds = ToRows(inputFlat, steps, hidden);
        var expected = ToRows(expectedFlat, steps, hidden);

        var actual = MiniMaxMusic3RvqDepthDecoder.Forward(weights, inputsEmbeds);

        Assert.Equal(steps, actual.Length);

        double sumAbsDiff = 0, sumAbsExpected = 0, maxAbsDiff = 0;
        for (int t = 0; t < steps; t++)
        {
            for (int c = 0; c < hidden; c++)
            {
                double diff = Math.Abs(actual[t][c] - expected[t][c]);
                maxAbsDiff = Math.Max(maxAbsDiff, diff);
                sumAbsDiff += diff;
                sumAbsExpected += Math.Abs(expected[t][c]);
            }
        }
        double relError = sumAbsDiff / sumAbsExpected;

        Assert.True(relError < 0.001, $"relative error {relError:F6} exceeds tolerance -- real numeric mismatch against diffusers reference (maxAbsDiff={maxAbsDiff:F6})");
    }
}
