using OpenTail.Stingray.Diffusion.StableAudio;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Unit tests for <see cref="StableAudioScheduleKernels.ApplyApg"/> verifying that padded tokens
/// are masked out of the APG norm and dot-product, and that orthogonal guidance is zeroed on padded tokens.
/// </summary>
public sealed class StableAudioApgTests
{
    [Fact]
    public void ApplyApg_WhenValidTokensEqualsTotalTokens_MatchesExpectedGuidance()
    {
        int tokens = 8;
        int channels = 16;
        int totalElements = tokens * channels;

        float[] condX0 = new float[totalElements];
        float[] diff = new float[totalElements];
        for (int i = 0; i < totalElements; i++)
        {
            condX0[i] = (float)Math.Sin(i * 0.1);
            diff[i] = (float)Math.Cos(i * 0.2);
        }

        float cfgScale = 6.0f;
        float apgScale = 0.75f;

        float[] output = new float[totalElements];
        StableAudioScheduleKernels.ApplyApg(condX0, diff, cfgScale, apgScale, tokens, tokens, channels, output);

        // Compute expected mathematically: projection of diff onto condX0
        double normSq = 0.0;
        for (int i = 0; i < totalElements; i++)
        {
            normSq += (double)condX0[i] * condX0[i];
        }

        double invNorm = 1.0 / Math.Sqrt(normSq + 1e-16);
        double dot = 0.0;
        for (int i = 0; i < totalElements; i++)
        {
            dot += (double)diff[i] * ((double)condX0[i] * invNorm);
        }

        for (int i = 0; i < totalElements; i++)
        {
            double v1Normalized = (double)condX0[i] * invNorm;
            double parallel = dot * v1Normalized;
            double orthogonal = (double)diff[i] - parallel;
            double guided = apgScale * orthogonal + (1.0f - apgScale) * diff[i];
            double expected = (double)condX0[i] + (cfgScale - 1.0f) * guided;
            Assert.Equal((float)expected, output[i], tolerance: 1e-4f);
        }
    }

    [Fact]
    public void ApplyApg_PaddedTokensDoNotAffectValidTokens_AndReceiveConditionedGuidance()
    {
        int validTokens = 6;
        int totalTokens = 10;
        int channels = 32;
        int validElements = validTokens * channels;
        int totalElements = totalTokens * channels;

        float[] condX0 = new float[totalElements];
        float[] diffA = new float[totalElements];
        float[] diffB = new float[totalElements];

        for (int i = 0; i < totalElements; i++)
        {
            condX0[i] = 1.0f + i * 0.01f;
            diffA[i] = 0.5f + i * 0.005f;
            diffB[i] = 0.5f + i * 0.005f;
        }

        // Corrupt padded tokens in diffB with massive values
        for (int i = validElements; i < totalElements; i++)
        {
            diffB[i] = 9999.0f;
        }

        float cfgScale = 7.0f;
        float apgScale = 1.0f;

        float[] outA = new float[totalElements];
        float[] outB = new float[totalElements];

        StableAudioScheduleKernels.ApplyApg(condX0, diffA, cfgScale, apgScale, validTokens, totalTokens, channels, outA);
        StableAudioScheduleKernels.ApplyApg(condX0, diffB, cfgScale, apgScale, validTokens, totalTokens, channels, outB);

        // 1. Valid tokens in outA and outB must be identical despite padded diffB corruption
        for (int i = 0; i < validElements; i++)
        {
            Assert.Equal(outA[i], outB[i]);
        }

        // 2. On padded tokens with apg = 1.0f, orthogonal guidance is masked to 0,
        // so out = condX0 + (cfgScale - 1) * (1 - apg) * diff = condX0
        for (int i = validElements; i < totalElements; i++)
        {
            Assert.Equal(condX0[i], outA[i]);
            Assert.Equal(condX0[i], outB[i]);
        }
    }

    [Fact]
    public void ApplyApg_CfgScaleOne_ReturnsCondX0Everywhere()
    {
        int validTokens = 5;
        int totalTokens = 8;
        int channels = 16;
        int totalElements = totalTokens * channels;

        float[] condX0 = new float[totalElements];
        float[] diff = new float[totalElements];
        for (int i = 0; i < totalElements; i++)
        {
            condX0[i] = (float)Math.Sin(i);
            diff[i] = (float)Math.Cos(i);
        }

        float[] output = new float[totalElements];
        StableAudioScheduleKernels.ApplyApg(condX0, diff, cfgScale: 1.0f, apgScale: 1.0f, validTokens, totalTokens, channels, output);

        for (int i = 0; i < totalElements; i++)
        {
            Assert.Equal(condX0[i], output[i]);
        }
    }
}
