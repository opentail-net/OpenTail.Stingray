using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Audio.VoxCpm2;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real-weight smoke tests for <see cref="VoxCpm2DiTEstimator"/> and
/// <see cref="VoxCpm2CfmSolver"/>.</summary>
public sealed class VoxCpm2DiTEstimatorRealWeightsTests : HeavyTestBase
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

    private static (VoxCpm2DiTEstimatorWeights, RvcPackedTensorSource, GgufModel)? LoadReal()
    {
        string? path = FindRepoFile("models/_models/voxcpm2/VoxCPM2-GGUF/voxcpm2-q8_0.gguf");
        if (path == null) return null;
        var model = GgufModel.Open(path);
        var source = new RvcPackedTensorSource(model);
        var w = VoxCpm2DiTEstimatorWeights.Load(numLayers: 12, source.GetTensor);
        return (w, source, model);
    }

    [Fact]
    public void Run_OnRealCheckpoint_ProducesFiniteOutput()
    {
        var loaded = LoadReal();
        Assert.SkipUnless(loaded != null, "voxcpm2-q8_0.gguf not found");
        var (w, _, model) = loaded!.Value;
        using var _ = model;

        const int hiddenDim = 1024;
        var rng = new Random(6);
        float[] RandRow(int n) => Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 0.2 - 0.1)).ToArray();

        var x = Enumerable.Range(0, VoxCpm2DiTEstimator.PatchSize).Select(_ => RandRow(VoxCpm2DiTEstimator.FeatDim)).ToArray();
        var cond = Enumerable.Range(0, VoxCpm2DiTEstimator.PatchSize).Select(_ => RandRow(VoxCpm2DiTEstimator.FeatDim)).ToArray();
        var mu = new[] { RandRow(hiddenDim), RandRow(hiddenDim) };

        var output = VoxCpm2DiTEstimator.Run(w, x, mu, cond, timestep: 0.5f, deltaTimestep: 0f);

        Assert.Equal(VoxCpm2DiTEstimator.PatchSize, output.Length);
        foreach (var row in output)
        {
            Assert.Equal(VoxCpm2DiTEstimator.FeatDim, row.Length);
            Assert.All(row, v => Assert.True(float.IsFinite(v)));
        }
    }

    [Fact]
    public void GeneratePatch_OnRealCheckpoint_ProducesFiniteOutput()
    {
        var loaded = LoadReal();
        Assert.SkipUnless(loaded != null, "voxcpm2-q8_0.gguf not found");
        var (w, _, model) = loaded!.Value;
        using var _ = model;

        const int hiddenDim = 1024;
        var rng = new Random(8);
        float[] RandRow(int n) => Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 0.2 - 0.1)).ToArray();

        var mu = new[] { RandRow(hiddenDim), RandRow(hiddenDim) };
        var cond = Enumerable.Range(0, VoxCpm2DiTEstimator.PatchSize).Select(_ => RandRow(VoxCpm2DiTEstimator.FeatDim)).ToArray();

        var output = VoxCpm2CfmSolver.GeneratePatch(w, mu, cond, timesteps: 4, cfgValue: 2.0f, meanMode: false, new Random(3));

        Assert.Equal(VoxCpm2DiTEstimator.PatchSize, output.Length);
        foreach (var row in output)
        {
            Assert.Equal(VoxCpm2DiTEstimator.FeatDim, row.Length);
            Assert.All(row, v => Assert.True(float.IsFinite(v)));
        }
    }
}
