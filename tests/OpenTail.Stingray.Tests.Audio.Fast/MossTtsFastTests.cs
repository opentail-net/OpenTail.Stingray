using System;
using OpenTail.Stingray.Audio.MossTts;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio.Fast;

public sealed class MossTtsFastTests
{
    [Fact]
    public void PatchUpsample_TransposesAndInterleavesChannelsCorrectly()
    {
        // 2 frames, 6 packed values per frame, patch factor 3 -> channels = 2
        const int frames = 2;
        const int packed = 6;
        const int patch = 3;
        const int expectedChannels = packed / patch; // 2

        var input = new float[frames][];
        // Frame 0: [c0p0, c0p1, c0p2, c1p0, c1p1, c1p2]
        input[0] = [10f, 11f, 12f, 20f, 21f, 22f];
        // Frame 1:
        input[1] = [30f, 31f, 32f, 40f, 41f, 42f];

        var output = MossTtsAudioCodecDecoder.PatchUpsample(input, patch);

        Assert.Equal(frames * patch, output.Length); // 6 rows
        for (int i = 0; i < output.Length; i++)
        {
            Assert.Equal(expectedChannels, output[i].Length);
        }

        // Verify mathematical contract: output[frame * patch + p][c] == input[frame][c * patch + p]
        for (int f = 0; f < frames; f++)
        {
            for (int p = 0; p < patch; p++)
            {
                int outRow = f * patch + p;
                for (int c = 0; c < expectedChannels; c++)
                {
                    float expected = input[f][c * patch + p];
                    Assert.Equal(expected, output[outRow][c]);
                }
            }
        }

        // Explicit check of row contents:
        // Frame 0:
        Assert.Equal([10f, 20f], output[0]); // p=0
        Assert.Equal([11f, 21f], output[1]); // p=1
        Assert.Equal([12f, 22f], output[2]); // p=2
        // Frame 1:
        Assert.Equal([30f, 40f], output[3]); // p=0
        Assert.Equal([31f, 41f], output[4]); // p=1
        Assert.Equal([32f, 42f], output[5]); // p=2
    }

    [Fact]
    public void PatchUpsample_StereoInterleaveFinalStage()
    {
        // Final stage upsamples by 2: packed = 2, patch = 2 -> channels = 1 (monaural stream before stereo de-interleave)
        var input = new float[2][];
        input[0] = [0.5f, -0.5f]; // left0, right0
        input[1] = [0.8f, -0.8f]; // left1, right1

        var output = MossTtsAudioCodecDecoder.PatchUpsample(input, patch: 2);

        Assert.Equal(4, output.Length);
        Assert.Equal([0.5f], output[0]);
        Assert.Equal([-0.5f], output[1]);
        Assert.Equal([0.8f], output[2]);
        Assert.Equal([-0.8f], output[3]);
    }

    [Fact]
    public void Decoder_InvalidCodebookCount_ThrowsArgumentException()
    {
        // Moss expects 16 codebooks
        var invalidCodes = new int[8][];
        for (int i = 0; i < 8; i++) invalidCodes[i] = [0, 1, 2];

        Assert.Throws<ArgumentException>(() =>
            MossTtsAudioCodecDecoder.Decode(null!, null!, invalidCodes));
    }

    [Fact]
    public void Decoder_EmptyCodes_ThrowsArgumentException()
    {
        var emptyCodes = new int[MossTtsAudioCodecQuantizerWeights.NumQuantizers][];
        for (int i = 0; i < emptyCodes.Length; i++) emptyCodes[i] = [];

        Assert.Throws<ArgumentException>(() =>
            MossTtsAudioCodecDecoder.Decode(null!, null!, emptyCodes));
    }

    [Fact]
    public void BuildVoiceClonePrompt_EmptyReferenceCodes_ThrowsArgumentException()
    {
        var emptyCodes = new int[MossTtsGlobalTransformerWeights.NumCodebooks][];
        for (int i = 0; i < emptyCodes.Length; i++) emptyCodes[i] = [];

        Assert.Throws<ArgumentException>(() =>
            MossTtsPromptBuilder.BuildVoiceClonePrompt(null!, "target text", emptyCodes));

        var invalidCodebooks = new int[4][];
        for (int i = 0; i < 4; i++) invalidCodebooks[i] = [1, 2, 3];

        Assert.Throws<ArgumentException>(() =>
            MossTtsPromptBuilder.BuildVoiceClonePrompt(null!, "target text", invalidCodebooks));
    }

    [Fact]
    public void BuildZeroShotPrompt_NullOrEmptyText_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            MossTtsPromptBuilder.BuildZeroShotPrompt(null!, ""));
    }
}
