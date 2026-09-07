using OpenTail.Stingray.Audio.MossTts;

namespace OpenTail.Stingray.Tests.Audio.Fast;

/// <summary>Structural tests for <see cref="MossTtsAudioCodecDecoder"/>'s reshape primitives.</summary>
public sealed class MossTtsAudioCodecDecoderTests
{
    [Fact]
    public void PatchUpsample_MatchesHandDerivedReshapeTransposeReshape()
    {
        // input: 2 frames, packed=4 (patch=2 -> channels=2). Values chosen so each element's
        // (frame, packed-index) is recoverable from its value: value = frame*100 + packedIndex.
        float[][] input =
        [
            [0, 1, 2, 3],
            [100, 101, 102, 103],
        ];

        var output = MossTtsAudioCodecDecoder.PatchUpsample(input, patch: 2);

        // output[frame*patch + p][c] = input[frame][c*patch + p]
        Assert.Equal(4, output.Length); // frames(2) * patch(2)
        Assert.Equal(2, output[0].Length); // packed(4) / patch(2)

        // frame=0,p=0: c=0-> input[0][0*2+0]=0; c=1-> input[0][1*2+0]=2
        Assert.Equal([0, 2], output[0]);
        // frame=0,p=1: c=0-> input[0][0*2+1]=1; c=1-> input[0][1*2+1]=3
        Assert.Equal([1, 3], output[1]);
        // frame=1,p=0: c=0-> input[1][0]=100; c=1-> input[1][2]=102
        Assert.Equal([100, 102], output[2]);
        // frame=1,p=1: c=0-> input[1][1]=101; c=1-> input[1][3]=103
        Assert.Equal([101, 103], output[3]);
    }

    [Fact]
    public void PatchUpsample_IdentityPatchOne_LeavesDataUnchanged()
    {
        float[][] input = [[1, 2, 3], [4, 5, 6]];
        var output = MossTtsAudioCodecDecoder.PatchUpsample(input, patch: 1);
        Assert.Equal(input, output);
    }
}
