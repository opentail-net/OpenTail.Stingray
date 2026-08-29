
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Verifies ChatterboxFlowEncoder (S3Gen stage 1: UpsampleConformerEncoder) against real GGUF
/// weights. As with ChatterboxT3Tests, there is no local PyTorch/safetensors checkpoint to build
/// a golden-output oracle from, so this is structural verification (real weights load, correct
/// output shapes given the 2x upsample, finite non-degenerate values) rather than a cosine-
/// similarity-against-ground-truth check.
/// </summary>
public sealed class ChatterboxFlowEncoderTests : HeavyTestBase
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
    public void ChatterboxFlowEncoder_RealWeights_ProducesValidMuShapeAndFiniteValues()
    {
        string? t3Path = FindRepoFile("models/chatterbox-turbo-t3-q4_k.gguf");
        string? s3GenPath = FindRepoFile("models/chatterbox-turbo-s3gen-q4_k.gguf");
        if (t3Path is null || s3GenPath is null) return;

        using var t3Weights = new ChatterboxWeights(t3Path);
        using var s3Weights = new ChatterboxS3GenWeights(s3GenPath);

        Assert.NotNull(t3Weights.GenPromptToken);
        Assert.NotNull(t3Weights.GenEmbedding);
        Assert.NotNull(t3Weights.GenPromptFeat);

        int[] promptTokens = t3Weights.GenPromptToken!;
        int[] speechTokens = [10, 20, 30, 40, 50, 60, 70, 80]; // arbitrary in-vocab placeholder tokens for a structural check

        var (mu, totalFrames) = ChatterboxFlowEncoder.Forward(s3Weights, promptTokens, speechTokens);

        int expectedT = promptTokens.Length + speechTokens.Length;
        Assert.Equal(expectedT * 2, totalFrames); // Upsample1D doubles the token-rate sequence length
        Assert.Equal(s3Weights.MelChannels * totalFrames, mu.Length);

        foreach (float v in mu)
        {
            Assert.False(float.IsNaN(v), "mu must not contain NaN");
            Assert.False(float.IsInfinity(v), "mu must not contain Infinity");
        }

        // A collapsed/broken forward pass (e.g. reading garbage weights) tends to produce a
        // constant or near-zero output; a real conditioning tensor should have real variance.
        double mean = mu.Average(v => (double)v);
        double variance = mu.Average(v => (v - mean) * (v - mean));
        Assert.True(variance > 1e-6, $"mu variance {variance} is suspiciously low for a real forward pass.");

        var spkEmbed = ChatterboxFlowEncoder.ProjectSpeakerEmbedding(s3Weights, t3Weights.GenEmbedding!);
        Assert.Equal(s3Weights.MelChannels, spkEmbed.Length);
        foreach (float v in spkEmbed)
        {
            Assert.False(float.IsNaN(v));
            Assert.False(float.IsInfinity(v));
        }
    }
}
