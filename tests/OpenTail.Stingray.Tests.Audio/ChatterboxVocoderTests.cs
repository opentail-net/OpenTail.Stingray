
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Verifies ChatterboxVocoder (S3Gen stage 3: HiFTGenerator) chained after the flow encoder and
/// CFM decoder, against real GGUF weights -- the full S3Gen token-to-waveform pipeline. Same
/// structural-verification limitation as the other Chatterbox tests (no local golden-waveform
/// oracle): checks shape, finiteness, audio-limit clamping, and non-degenerate energy.
/// </summary>
public sealed class ChatterboxVocoderTests : HeavyTestBase
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
    public void ChatterboxVocoder_RealWeights_ProducesValidWaveform()
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

        int mel = s3Weights.MelChannels;
        float[] promptFeat = t3Weights.GenPromptFeat!;
        int mel1 = promptFeat.Length / mel;
        var cond = new float[mel * totalFrames];
        for (int c = 0; c < mel; c++)
            Array.Copy(promptFeat, c * mel1, cond, c * totalFrames, mel1);

        var rng = new Random(7);
        var melOut = ChatterboxCfmDecoder.Generate(s3Weights, mu, cond, spkEmbed, totalFrames, rng, nSteps: 2);

        // Keep only the truly generated (non-prompt) mel region, matching flow.py's
        // `feat[:, :, mel_len1:]` -- this is what actually feeds the vocoder in the real pipeline.
        int mel2 = totalFrames - mel1;
        var melTail = new float[mel * mel2];
        for (int c = 0; c < mel; c++)
            Array.Copy(melOut, c * totalFrames + mel1, melTail, c * mel2, mel2);

        var waveform = ChatterboxVocoder.Generate(s3Weights, melTail, mel2, new Random(11));

        Assert.NotEmpty(waveform);

        int totalUp = s3Weights.VocIstftHopLen;
        foreach (int r in s3Weights.VocUpsampleRates) totalUp *= r;
        // The waveform length should be in the right ballpark for mel2 frames at this upsample
        // factor (exact length depends on the reflection-pad/iSTFT framing, so allow slack).
        int expectedApprox = mel2 * totalUp;
        Assert.InRange(waveform.Length, (int)(expectedApprox * 0.5), (int)(expectedApprox * 1.5));

        float energy = 0f;
        foreach (float s in waveform)
        {
            Assert.False(float.IsNaN(s), "waveform must not contain NaN");
            Assert.False(float.IsInfinity(s), "waveform must not contain Infinity");
            Assert.InRange(s, -1.0f, 1.0f); // audio_limit clamp is 0.99
            energy += s * s;
        }
        Assert.True(energy > 0.01f, $"waveform energy {energy} is suspiciously low (near-silent) for a real forward pass.");
    }
}
