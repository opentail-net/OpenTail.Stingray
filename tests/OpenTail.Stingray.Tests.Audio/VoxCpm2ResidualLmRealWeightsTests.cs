using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Audio.VoxCpm2;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real-weight smoke test for <see cref="VoxCpm2ResidualLm"/>.</summary>
public sealed class VoxCpm2ResidualLmRealWeightsTests : HeavyTestBase
{
    private static string? FindRepoFile(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Step_OnRealCheckpoint_ProducesFiniteHiddenStates_AcrossMultipleSteps()
    {
        string? path = FindRepoFile("models/_models/voxcpm2/VoxCPM2-GGUF/voxcpm2-q8_0.gguf");
        Assert.SkipUnless(path != null, "voxcpm2-q8_0.gguf not found");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        var lm = VoxCpm2ResidualLm.Load(source.GetTensor);

        var rng = new Random(11);
        float[] RandEmbedding() => Enumerable.Range(0, VoxCpm2ResidualLm.HiddenDim)
            .Select(_ => (float)(rng.NextDouble() * 0.2 - 0.1)).ToArray();

        for (int step = 0; step < 3; step++)
        {
            var hidden = lm.Step(RandEmbedding());
            Assert.Equal(VoxCpm2ResidualLm.HiddenDim, hidden.Length);
            Assert.All(hidden, v => Assert.True(float.IsFinite(v)));
        }
    }
}
