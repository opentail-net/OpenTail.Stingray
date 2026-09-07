using OpenTail.Stingray.Audio.VibeVoice;

namespace OpenTail.Stingray.Tests.Audio.Fast;

/// <summary>Structural tests for <see cref="VibeVoiceConnector"/>.</summary>
public sealed class VibeVoiceConnectorTests
{
    [Fact]
    public void Project_ProducesFiniteOutput_WithExpectedShape()
    {
        var rng = new Random(1);
        float[] Rand(int n) => Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 0.1 - 0.05)).ToArray();

        const int inputDim = 4, hiddenSize = 8, frames = 5;
        var tensors = new Dictionary<string, float[]>
        {
            ["c.fc1.weight"] = Rand(hiddenSize * inputDim),
            ["c.fc1.bias"] = Rand(hiddenSize),
            ["c.norm.weight"] = Enumerable.Repeat(1f, hiddenSize).ToArray(),
            ["c.fc2.weight"] = Rand(hiddenSize * hiddenSize),
            ["c.fc2.bias"] = Rand(hiddenSize),
        };
        var w = VibeVoiceConnectorWeights.Load("c", inputDim, hiddenSize, name => tensors[name]);

        var input = new float[inputDim][];
        for (int c = 0; c < inputDim; c++)
        {
            input[c] = new float[frames];
            for (int t = 0; t < frames; t++) input[c][t] = (float)(rng.NextDouble() * 2 - 1);
        }

        var output = VibeVoiceConnector.Project(w, input);

        Assert.Equal(hiddenSize, output.Length);
        Assert.Equal(frames, output[0].Length);
        foreach (var row in output)
            Assert.All(row, v => Assert.True(float.IsFinite(v)));
    }
}
