using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Audio.VoxCpm2;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real-weight smoke test for <see cref="VoxCpm2LocalEncoder"/>.</summary>
public sealed class VoxCpm2LocalEncoderRealWeightsTests : HeavyTestBase
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
    public void EncodePatch_OnRealCheckpoint_ProducesFiniteOutput()
    {
        string? path = FindRepoFile("models/_models/voxcpm2/VoxCPM2-GGUF/voxcpm2-q8_0.gguf");
        Assert.SkipUnless(path != null, "voxcpm2-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        var w = VoxCpm2LocalEncoderWeights.Load(numLayers: 12, source.GetTensor);

        var rng = new Random(4);
        var patch = new float[VoxCpm2LocalEncoder.PatchSize][];
        for (int i = 0; i < patch.Length; i++)
        {
            patch[i] = new float[VoxCpm2LocalEncoder.FeatDim];
            for (int j = 0; j < patch[i].Length; j++) patch[i][j] = (float)(rng.NextDouble() * 2 - 1);
        }

        const int lmHiddenDim = 2048;
        var output = VoxCpm2LocalEncoder.EncodePatch(w, patch, lmHiddenDim);

        Assert.Equal(lmHiddenDim, output.Length);
        Assert.All(output, v => Assert.True(float.IsFinite(v)));
    }
}
