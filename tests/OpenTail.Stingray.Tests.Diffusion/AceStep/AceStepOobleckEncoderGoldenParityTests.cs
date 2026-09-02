using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.AceStep;
using OpenTail.Stingray.Diffusion.AceStep.Vae;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion.AceStep;

/// <summary>
/// Real numeric golden-parity check for <see cref="AceStepOobleckEncoder"/> against the real
/// `diffusers.models.autoencoders.autoencoder_oobleck.AutoencoderOobleck.encoder` reference, loaded
/// with the SAME real `vae.safetensors` weights (zero missing/unexpected keys) and run on a
/// fixed-seed synthetic PCM input. See docs/064-acestep-implementation-plan.md's "Golden-parity
/// check" section for how the reference dump (`golden_vae_enc_*.bin`) was produced.
///
/// <para>This encoder was ported specifically to derive a real `silence_latent` self-sufficiently
/// (encode true silence through it) rather than needing an external asset -- see
/// <see cref="AceStepOobleckEncoder"/>'s doc comment.</para>
/// </summary>
public sealed class AceStepOobleckEncoderGoldenParityTests
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
    public void EncodeMode_RealWeights_MatchesRealDiffusersReference()
    {
        string? vaePath = FindRepoFile("models/acestep-v15/vae.safetensors");
        Assert.SkipUnless(vaePath != null, "models/acestep-v15/vae.safetensors not found");

        string scratchDir = @"C:\Users\Dmitri\AppData\Local\Temp\claude\C--Git-Public-OpenTail-Stingray\6cb31b57-ce45-49d6-9926-8736cdcfcfa9\scratchpad\acestep";
        string inputPath = Path.Combine(scratchDir, "golden_vae_enc_input.bin");
        Assert.SkipUnless(File.Exists(inputPath), "golden_vae_enc_*.bin reference dump not found -- regenerate via golden_vae_encoder.py (see docs/064)");

        using var loader = SafetensorsLoader.Open(vaePath!);
        var weights = AceStepOobleckEncoderWeights.Load(loader);

        int latentFrames = 50;
        int hopLength = AceStepConfig.VaeDownsamplingRatios.Aggregate(1, (a, b) => a * b);
        int sampleCount = latentFrames * hopLength;

        var pcm = ReadBin(inputPath, AceStepConfig.VaeAudioChannels * sampleCount);
        var expected = ReadBin(Path.Combine(scratchDir, "golden_vae_enc_output.bin"), AceStepConfig.VaeDecoderInputChannels * latentFrames);

        var actual = AceStepOobleckEncoder.EncodeMode(weights, pcm, AceStepConfig.VaeAudioChannels, sampleCount);

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
