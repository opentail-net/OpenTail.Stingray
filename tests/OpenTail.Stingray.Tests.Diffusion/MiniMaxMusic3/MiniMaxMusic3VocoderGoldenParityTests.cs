using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.MiniMaxMusic3;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion.MiniMaxMusic3;

/// <summary>
/// Real numeric golden-parity check for <see cref="MiniMaxMusic3Vocoder"/> against the real
/// `diffusers.MiniMaxMusic3Vocoder` reference (already installed, `diffusers==0.40.0`), loaded with
/// the SAME real `vocoder/diffusion_pytorch_model.safetensors` weights (zero missing/unexpected
/// keys) and run on a fixed-seed synthetic latent. See docs/066-minimax-music3-future-plan.md for
/// the archaeology; reference dump (`minimax_vocoder_*.bin`) not checked into the repo.
/// </summary>
public sealed class MiniMaxMusic3VocoderGoldenParityTests
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
    public void Decode_RealWeights_MatchesRealDiffusersReference()
    {
        string? weightsPath = FindRepoFile("models/minimax-music3/vocoder.safetensors");
        Assert.SkipUnless(weightsPath != null, "models/minimax-music3/vocoder.safetensors not found");

        string scratchDir = @"C:\Users\Dmitri\AppData\Local\Temp\claude\C--Git-Public-OpenTail-Stingray\6cb31b57-ce45-49d6-9926-8736cdcfcfa9\scratchpad";
        string inputPath = Path.Combine(scratchDir, "minimax_vocoder_input.bin");
        Assert.SkipUnless(File.Exists(inputPath), "minimax_vocoder_*.bin reference dump not found");

        using var loader = SafetensorsLoader.Open(weightsPath!);
        var weights = MiniMaxMusic3VocoderWeights.Load(loader);

        int latentLen = 10;
        var latent = ReadBin(inputPath, MiniMaxMusic3Config.VocoderLatentChannels * latentLen);
        int hop = MiniMaxMusic3Config.VocoderUpsamplingRatios.Aggregate(1, (a, b) => a * b);
        var expected = ReadBin(Path.Combine(scratchDir, "minimax_vocoder_output.bin"), 2 * latentLen * hop);

        var actual = MiniMaxMusic3Vocoder.Decode(weights, latent, latentLen);

        Assert.Equal(expected.Length, actual.Length);

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
