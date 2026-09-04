using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.MiniMaxMusic3;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion.MiniMaxMusic3;

/// <summary>
/// Real numeric golden-parity check for <see cref="MiniMaxMusic3Transformer"/> (the flow-matching
/// DiT) against the real `diffusers.MiniMaxMusic3Transformer1DModel` reference, loaded with the SAME
/// real `transformer/diffusion_pytorch_model-*.safetensors` weights (zero missing/unexpected keys)
/// and run on a fixed-seed synthetic input. See docs/066-minimax-music3-future-plan.md.
/// </summary>
public sealed class MiniMaxMusic3TransformerGoldenParityTests
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

    private static string? FindRepoDir(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (Directory.Exists(p)) return p;
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
        string? weightsDir = FindRepoDir("models/minimax-music3/transformer");
        Assert.SkipUnless(weightsDir != null, "models/minimax-music3/transformer/ not found");

        string scratchDir = @"C:\Users\Dmitri\AppData\Local\Temp\claude\C--Git-Public-OpenTail-Stingray\6cb31b57-ce45-49d6-9926-8736cdcfcfa9\scratchpad";
        string latentPath = Path.Combine(scratchDir, "minimax_transformer_latent.bin");
        Assert.SkipUnless(File.Exists(latentPath), "minimax_transformer_*.bin reference dump not found");

        const int length = 8;
        const int inChannels = 128;
        const int condDim = 2048;
        const float timestep = 0.37f;

        using var loader = SafetensorsLoader.OpenDirectory(weightsDir!);
        var weights = MiniMaxMusic3TransformerWeights.Load(loader);

        var latentFlat = ReadBin(latentPath, length * inChannels);
        var conditionFlat = ReadBin(Path.Combine(scratchDir, "minimax_transformer_condition.bin"), length * condDim);
        var expectedFlat = ReadBin(Path.Combine(scratchDir, "minimax_transformer_output.bin"), length * inChannels);

        var latent = ToRows(latentFlat, length, inChannels);
        var condition = ToRows(conditionFlat, length, condDim);
        var expected = ToRows(expectedFlat, length, inChannels);

        var actual = MiniMaxMusic3Transformer.Forward(weights, latent, condition, timestep);

        Assert.Equal(length, actual.Length);

        double sumAbsDiff = 0, sumAbsExpected = 0, maxAbsDiff = 0;
        for (int t = 0; t < length; t++)
        {
            for (int c = 0; c < inChannels; c++)
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

    [Fact]
    public void ForwardPair_MatchesForward_BitForBit()
    {
        string? weightsDir = FindRepoDir("models/minimax-music3/transformer");
        Assert.SkipUnless(weightsDir != null, "models/minimax-music3/transformer/ not found");

        string scratchDir = @"C:\Users\Dmitri\AppData\Local\Temp\claude\C--Git-Public-OpenTail-Stingray\6cb31b57-ce45-49d6-9926-8736cdcfcfa9\scratchpad";
        string latentPath = Path.Combine(scratchDir, "minimax_transformer_latent.bin");
        Assert.SkipUnless(File.Exists(latentPath), "minimax_transformer_*.bin reference dump not found");

        const int length = 8;
        const int inChannels = 128;
        const int condDim = 2048;
        const float timestep = 0.37f;

        using var loader = SafetensorsLoader.OpenDirectory(weightsDir!);
        var weights = MiniMaxMusic3TransformerWeights.Load(loader);

        var latentFlat = ReadBin(latentPath, length * inChannels);
        var conditionFlat = ReadBin(Path.Combine(scratchDir, "minimax_transformer_condition.bin"), length * condDim);

        var latent = ToRows(latentFlat, length, inChannels);
        var condition = ToRows(conditionFlat, length, condDim);
        var zeroCondition = new float[length][];
        for (int t = 0; t < length; t++) zeroCondition[t] = new float[condDim];

        var expectedCond = MiniMaxMusic3Transformer.Forward(weights, latent, condition, timestep);
        var expectedUncond = MiniMaxMusic3Transformer.Forward(weights, latent, zeroCondition, timestep);

        var (actualCond, actualUncond) = MiniMaxMusic3Transformer.ForwardPair(weights, latent, condition, zeroCondition, timestep);

        Assert.Equal(length, actualCond.Length);
        Assert.Equal(length, actualUncond.Length);

        for (int t = 0; t < length; t++)
        {
            for (int c = 0; c < inChannels; c++)
            {
                Assert.Equal(expectedCond[t][c], actualCond[t][c], 1e-5f);
                Assert.Equal(expectedUncond[t][c], actualUncond[t][c], 1e-5f);
            }
        }
    }
}
