using System;
using OpenTail.Stingray.Audio.PersonaPlex;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio.Fast;

public sealed class PersonaPlexFastTests
{
    [Fact]
    public void DelayState_Bootstrap_StepZeroReturnsNullAndAdvancesOffset()
    {
        var delay = new PersonaPlexDelayState();
        Assert.Equal(0, delay.Offset);

        // Step 0: Bootstrap round. Prepare must return null, advancing Offset to 1.
        var step0 = delay.Prepare(userTokens: null, moshiTokens: null, textToken: null);
        Assert.Null(step0);
        Assert.Equal(1, delay.Offset);
    }

    [Fact]
    public void DelayState_StepOneAndTwo_ProducesDelayedOutputCorrectly()
    {
        var delay = new PersonaPlexDelayState();

        // Step 0: bootstrap
        var step0 = delay.Prepare(null, null, null);
        Assert.Null(step0);

        // Step 1: first real model step
        var step1 = delay.Prepare(userTokens: null, moshiTokens: null, textToken: 123);
        Assert.NotNull(step1);
        Assert.Equal(PersonaPlexDelayState.NumStreams, step1!.Value.Tokens.Length);
        Assert.Equal(17, step1.Value.Tokens.Length);

        // Provide 16 audio tokens from Depformer (8 moshi + 8 user)
        int[] sampledAudio1 = [10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27];
        var outAudio1 = delay.FinishWithSampling(sampledTextToken: 55, sampledAudio1);
        // Step 1 is warming up (offset <= maxDelay of 1), output is null
        Assert.Null(outAudio1);
        Assert.Equal(2, delay.Offset);

        // Step 2: second model step
        var step2 = delay.Prepare(userTokens: null, moshiTokens: null, textToken: null);
        Assert.NotNull(step2);

        int[] sampledAudio2 = [30, 31, 32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44, 45, 46, 47];
        var outAudio2 = delay.FinishWithSampling(sampledTextToken: 66, sampledAudio2);
        // Step 2: offset > 1, produces real 8-codebook moshi audio output
        Assert.NotNull(outAudio2);
        Assert.Equal(PersonaPlexDelayState.MimiFrameCodebooks, outAudio2!.Length);
        Assert.Equal(8, outAudio2.Length);
        Assert.Equal(3, delay.Offset);
    }

    [Fact]
    public void DelayState_ImportCache_RequiresExactLength()
    {
        var delay = new PersonaPlexDelayState();
        int expectedLen = PersonaPlexDelayState.NumStreams * PersonaPlexDelayState.DelayCacheSteps; // 17 * 4 = 68

        // Wrong size: throws ArgumentException
        Assert.Throws<ArgumentException>(() => delay.ImportCache(new long[10]));
        Assert.Throws<ArgumentException>(() => delay.ImportCache(new long[69]));

        // Correct size: succeeds
        var validSnapshot = new long[expectedLen];
        Array.Fill(validSnapshot, 42);
        delay.ImportCache(validSnapshot);
    }

    [Fact]
    public void DelayState_FinishWithSampling_RequiresSixteenAudioStreams()
    {
        var delay = new PersonaPlexDelayState();
        delay.Prepare(null, null, null); // step 0 bootstrap
        delay.Prepare(null, null, null); // step 1

        // Passing 8 instead of 16 audio streams throws ArgumentException
        Assert.Throws<ArgumentException>(() =>
            delay.FinishWithSampling(sampledTextToken: 0, sampledAudioTokens: new int[8]));

        // Passing 16 audio streams succeeds
        var outTokens = delay.FinishWithSampling(sampledTextToken: 0, sampledAudioTokens: new int[16]);
        Assert.Null(outTokens); // step 1 is warmup
    }

    [Fact]
    public void SentencePieceModel_NonUnigramModelType_ThrowsNotSupportedException()
    {
        // Minimal protobuf: ModelProto with trainer_spec (field 2) containing model_type (field 3) = 2 (BPE)
        // wire format:
        // tag for field 2 (LEN): (2 << 3) | 2 = 18 = 0x12
        // submessage: tag for field 3 (VARINT): (3 << 3) | 0 = 24 = 0x18, value = 2
        byte[] bpeModelProto = [0x12, 0x02, 0x18, 0x02];

        Assert.Throws<NotSupportedException>(() =>
            PersonaPlexSentencePieceModel.Load(bpeModelProto));
    }
}
