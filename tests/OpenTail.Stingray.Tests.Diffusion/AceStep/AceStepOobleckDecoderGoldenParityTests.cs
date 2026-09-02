using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.AceStep;
using OpenTail.Stingray.Diffusion.AceStep.Vae;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion.AceStep;

/// <summary>
/// Real numeric golden-parity check for <see cref="AceStepOobleckDecoder"/> against the real
/// `diffusers.models.autoencoders.autoencoder_oobleck.AutoencoderOobleck.decoder` reference, loaded
/// with the SAME real `vae.safetensors` weights (`load_state_dict(strict=False)` reported zero
/// missing/unexpected keys against the real checkpoint) and run on a fixed-seed synthetic latent.
/// See docs/064-acestep-implementation-plan.md's "Golden-parity check" section for how the
/// reference dump (`golden_vae_*.bin`) was produced -- not checked into the repo, regenerate via
/// the Python script referenced there if needed.
/// </summary>
public sealed class AceStepOobleckDecoderGoldenParityTests
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
        string? vaePath = FindRepoFile("models/acestep-v15/vae.safetensors");
        Assert.SkipUnless(vaePath != null, "models/acestep-v15/vae.safetensors not found");

        string scratchDir = @"C:\Users\Dmitri\AppData\Local\Temp\claude\C--Git-Public-OpenTail-Stingray\6cb31b57-ce45-49d6-9926-8736cdcfcfa9\scratchpad\acestep";
        string latentPath = Path.Combine(scratchDir, "golden_vae_latent.bin");
        Assert.SkipUnless(File.Exists(latentPath), "golden_vae_*.bin reference dump not found -- regenerate via golden_vae.py (see docs/064)");

        using var loader = SafetensorsLoader.Open(vaePath!);
        var weights = AceStepOobleckDecoderWeights.Load(loader);

        int latentFrames = 50;
        int hopLength = AceStepConfig.VaeDownsamplingRatios.Aggregate(1, (a, b) => a * b);
        int expectedSamples = latentFrames * hopLength;

        // Real PyTorch latent layout is channel-major [1,64,T] -- our C# Decode wants the same
        // flat [channel,T] layout (see AceStepOobleckDecoder.Decode's own doc comment).
        var latent = ReadBin(latentPath, 64 * latentFrames);
        var expected = ReadBin(Path.Combine(scratchDir, "golden_vae_output.bin"), 2 * expectedSamples);

        var actual = AceStepOobleckDecoder.Decode(weights, latent, latentFrames);

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

        // Measured relative error against the real reference is ~4e-6 (F32-rounding-level
        // agreement over the full 5-block decoder) -- generous headroom above that while still
        // catching a real bug loudly rather than letting it slip through.
        Assert.True(relError < 0.001, $"relative error {relError:F6} exceeds tolerance -- real numeric mismatch against diffusers reference (maxAbsDiff={maxAbsDiff:F6})");
    }
}
