
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Verifies ChatterboxCfmDecoder (S3Gen stage 2: the CFM flow-matching UNet + meanflow 2-step
/// Euler solver) chained after ChatterboxFlowEncoder (stage 1), against real GGUF weights. Same
/// honest limitation as the other Chatterbox structural tests: no local PyTorch checkpoint to
/// build a golden-mel oracle from, so this checks shape correctness, finiteness, and that the
/// output isn't degenerate (constant/zero) -- not numeric ground truth.
/// </summary>
public sealed class ChatterboxCfmDecoderTests : HeavyTestBase
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
    public void ChatterboxCfmDecoder_RealWeights_ProducesValidMelShapeAndFiniteValues()
    {
        string? t3Path = FindRepoFile("models/chatterbox-turbo-t3-q4_k.gguf");
        string? s3GenPath = FindRepoFile("models/chatterbox-turbo-s3gen-q4_k.gguf");
        if (t3Path is null || s3GenPath is null) return;

        using var t3Weights = new ChatterboxWeights(t3Path);
        using var s3Weights = new ChatterboxS3GenWeights(s3GenPath);

        int[] promptTokens = t3Weights.GenPromptToken!;
        int[] speechTokens = [10, 20, 30, 40, 50, 60, 70, 80];

        var (mu, totalFrames) = ChatterboxFlowEncoder.Forward(s3Weights, promptTokens, speechTokens);
        var spkEmbed = ChatterboxFlowEncoder.ProjectSpeakerEmbedding(s3Weights, t3Weights.GenEmbedding!);

        // cond: prompt_feat in the first mel_len1 frames (the reference clip's real mel), zero
        // elsewhere -- flow.py's `conds[:, :mel_len1] = prompt_feat; conds = conds.transpose(1,2)`.
        int mel = s3Weights.MelChannels;
        float[] promptFeat = t3Weights.GenPromptFeat!; // channel-first [80, 500] already, per GGUF dump
        int mel1 = promptFeat.Length / mel;
        Assert.True(mel1 <= totalFrames, "prompt_feat must not be longer than the encoder output.");

        var cond = new float[mel * totalFrames];
        for (int c = 0; c < mel; c++)
            Array.Copy(promptFeat, c * mel1, cond, c * totalFrames, mel1);

        var rng = new Random(7);
        var melOut = ChatterboxCfmDecoder.Generate(s3Weights, mu, cond, spkEmbed, totalFrames, rng, nSteps: 2);

        Assert.Equal(mel * totalFrames, melOut.Length);
        foreach (float v in melOut)
        {
            Assert.False(float.IsNaN(v), "generated mel must not contain NaN");
            Assert.False(float.IsInfinity(v), "generated mel must not contain Infinity");
        }

        double mean = melOut.Average(v => (double)v);
        double variance = melOut.Average(v => (v - mean) * (v - mean));
        Assert.True(variance > 1e-6, $"generated mel variance {variance} is suspiciously low for a real forward pass.");

        // The truly "new" (non-prompt) region is what the caller keeps (flow.py slices off the
        // first mel1 frames) -- check that region specifically too, since a bug could plausibly
        // leave the prompt-conditioned region looking fine while the generated tail is garbage.
        int mel2 = totalFrames - mel1;
        Assert.True(mel2 > 0, "must have generated at least one new mel frame beyond the prompt.");
        double tailMean = 0, tailCount = 0;
        for (int c = 0; c < mel; c++)
            for (int ti = mel1; ti < totalFrames; ti++)
            {
                tailMean += melOut[c * totalFrames + ti];
                tailCount++;
            }
        tailMean /= tailCount;
        double tailVar = 0;
        for (int c = 0; c < mel; c++)
            for (int ti = mel1; ti < totalFrames; ti++)
            {
                double d = melOut[c * totalFrames + ti] - tailMean;
                tailVar += d * d;
            }
        tailVar /= tailCount;
        Assert.True(tailVar > 1e-6, $"generated (non-prompt) mel tail variance {tailVar} is suspiciously low.");
    }
}
