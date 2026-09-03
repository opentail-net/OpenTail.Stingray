using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.MiniMaxMusic3;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion.MiniMaxMusic3;

/// <summary>
/// Real numeric golden-parity check for <see cref="MiniMaxMusic3ConditionEncoder"/> against the
/// real `diffusers.MiniMaxMusic3ConditionEncoder` reference, loaded with the SAME real
/// `condition_encoder/diffusion_pytorch_model.safetensors` weights (zero missing/unexpected keys)
/// and run on a fixed-seed synthetic input. See docs/066-minimax-music3-future-plan.md.
/// </summary>
public sealed class MiniMaxMusic3ConditionEncoderGoldenParityTests
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

    [Fact]
    public void Forward_RealWeights_MatchesRealDiffusersReference()
    {
        string? weightsPath = FindRepoFile("models/minimax-music3/condition_encoder.safetensors");
        Assert.SkipUnless(weightsPath != null, "models/minimax-music3/condition_encoder.safetensors not found");

        string scratchDir = @"C:\Users\Dmitri\AppData\Local\Temp\claude\C--Git-Public-OpenTail-Stingray\6cb31b57-ce45-49d6-9926-8736cdcfcfa9\scratchpad";
        string inputPath = Path.Combine(scratchDir, "minimax_condenc_input.bin");
        Assert.SkipUnless(File.Exists(inputPath), "minimax_condenc_*.bin reference dump not found");

        using var loader = SafetensorsLoader.Open(weightsPath!);
        var weights = MiniMaxMusic3ConditionEncoderWeights.Load(loader);

        int numFrames = 15;
        int numLayers = MiniMaxMusic3Config.ConditionEncoderNumLayers;
        int condHiddenDim = MiniMaxMusic3Config.ConditionEncoderHiddenDim;
        int outDim = MiniMaxMusic3Config.ConditionEncoderOutDim;
        int expectedLatentLength = 51; // computed from the real reference run, num_frames=15

        // Real flat input layout: [frame, layer*condHiddenDim + c] (real `reshape(batch,
        // num_layers, condHiddenDim, frames)` after transpose -- layer-major within each frame).
        var inputFlat = ReadBin(inputPath, numFrames * numLayers * condHiddenDim);
        var hiddenStates = new float[numFrames][][];
        for (int f = 0; f < numFrames; f++)
        {
            hiddenStates[f] = new float[numLayers][];
            for (int l = 0; l < numLayers; l++)
            {
                var row = new float[condHiddenDim];
                Array.Copy(inputFlat, (f * numLayers + l) * condHiddenDim, row, 0, condHiddenDim);
                hiddenStates[f][l] = row;
            }
        }

        var expectedFlat = ReadBin(Path.Combine(scratchDir, "minimax_condenc_output.bin"), expectedLatentLength * outDim);

        var actual = MiniMaxMusic3ConditionEncoder.Forward(weights, hiddenStates);

        Assert.Equal(expectedLatentLength, actual.Length);

        double sumAbsDiff = 0, sumAbsExpected = 0, maxAbsDiff = 0;
        for (int t = 0; t < expectedLatentLength; t++)
        {
            for (int c = 0; c < outDim; c++)
            {
                double diff = Math.Abs(actual[t][c] - expectedFlat[t * outDim + c]);
                maxAbsDiff = Math.Max(maxAbsDiff, diff);
                sumAbsDiff += diff;
                sumAbsExpected += Math.Abs(expectedFlat[t * outDim + c]);
            }
        }
        double relError = sumAbsDiff / sumAbsExpected;

        Assert.True(relError < 0.001, $"relative error {relError:F6} exceeds tolerance -- real numeric mismatch against diffusers reference (maxAbsDiff={maxAbsDiff:F6})");
    }
}
