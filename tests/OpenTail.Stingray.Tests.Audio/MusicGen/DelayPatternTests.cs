using OpenTail.Stingray.Audio.MusicGen;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio.MusicGen;

public class DelayPatternTests
{
    [Fact]
    public void BuildInput_StaggersEachCodebookByItsIndex()
    {
        int[][] tokens =
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
            [40, 41, 42, 43],
        ];

        var delayed = DelayPattern.BuildInput(tokens, padToken: -1);

        Assert.Equal(4, delayed.Length);
        Assert.Equal(7, delayed[0].Length); // frames(4) + codebooks(4) - 1

        Assert.Equal([10, 11, 12, 13, -1, -1, -1], delayed[0]);
        Assert.Equal([-1, 20, 21, 22, 23, -1, -1], delayed[1]);
        Assert.Equal([-1, -1, 30, 31, 32, 33, -1], delayed[2]);
        Assert.Equal([-1, -1, -1, 40, 41, 42, 43], delayed[3]);
    }

    [Fact]
    public void BuildInput_RemoveDelay_RoundTrips()
    {
        int[][] tokens =
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
            [40, 41, 42, 43],
        ];

        var delayed = DelayPattern.BuildInput(tokens, padToken: -1);
        var restored = DelayPattern.RemoveDelay(delayed, frames: 4);

        Assert.Equal(tokens, restored);
    }

    [Fact]
    public void InputColumnForStep_IsThePreviousTargetColumn_ShiftedByOne()
    {
        // Simulates the real generation loop: at each step, the INPUT fed to predict target
        // column `step` must be the target column `step - 1` (already generated), not the
        // value being predicted at `step` itself -- a causal LM predicts the NEXT position from
        // the current one, it cannot be fed its own not-yet-sampled output.
        int[][] tokens =
        [
            [10, 11, 12],
            [20, 21, 22],
            [30, 31, 32],
            [40, 41, 42],
        ];
        var targetGrid = DelayPattern.BuildInput(tokens, padToken: -1);
        int seqLen = targetGrid[0].Length;

        var generated = new int[4][] { [], [], [], [] };
        for (int step = 0; step < seqLen; step++)
        {
            var inputColumn = DelayPattern.InputColumnForStep(4, step, generated, bosOrPadToken: -1);

            var expectedInput = step == 0 ? [-1, -1, -1, -1] : targetGrid.Select(row => row[step - 1]).ToArray();
            Assert.Equal(expectedInput, inputColumn);

            // Sample this step's target column and append real tokens to their codebook streams,
            // matching what the real generation loop does after calling the transformer.
            for (int q = 0; q < 4; q++)
            {
                int localIndex = step - q;
                if (localIndex >= 0 && localIndex < tokens[q].Length)
                    generated[q] = [.. generated[q], tokens[q][localIndex]];
            }
        }
    }

    [Fact]
    public void InputColumnForStep_ReturnsBosForFirstStep()
    {
        var generated = new int[4][] { [], [], [], [] };
        var column = DelayPattern.InputColumnForStep(4, step: 0, generated, bosOrPadToken: 2048);

        Assert.Equal([2048, 2048, 2048, 2048], column);
    }
}
