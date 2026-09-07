using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Audio.VoxCpm2;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real-weight smoke test for <see cref="VoxCpm2StepProjection"/>.</summary>
public sealed class VoxCpm2StepProjectionRealWeightsTests : HeavyTestBase
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

    // Real config: lm.hidden_size=2048, dit hidden_dim (see dit_config, not yet fully dumped --
    // use a conservative real value confirmed via encoder_config's shared hidden_dim=256 pattern;
    // fall back to hiddenDim itself if unknown -- shape mismatches would throw and fail loudly).
    private const int HiddenDim = 2048;
    private const int ScalarQuantizationLatentDim = 8; // real: config.scalar_quantization_latent_dim (from earlier dump, patch_size=4/feat_dim=64 area)
    private const int ScalarQuantizationScale = 8; // placeholder if the real value differs, the round-trip is still finite/deterministic

    [Fact]
    public void Run_OnRealCheckpoint_ProducesFiniteOutput()
    {
        string? path = FindRepoFile("models/_models/voxcpm2/VoxCPM2-GGUF/voxcpm2-q8_0.gguf");
        Assert.SkipUnless(path != null, "voxcpm2-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        var w = VoxCpm2StepProjectionWeights.Load(source.GetTensor);

        // Real dit.hidden_dim inferred from the loaded lm_to_dit_proj weight's own shape (avoids
        // hardcoding a possibly-wrong number for a config section not dumped this session).
        int ditHiddenDim = w.LmToDitProjWeight.Length / HiddenDim;
        int latentDim = w.FsqInProjWeight.Length / HiddenDim;

        var rng = new Random(9);
        float[] Rand() => Enumerable.Range(0, HiddenDim).Select(_ => (float)(rng.NextDouble() * 0.2 - 0.1)).ToArray();

        var output = VoxCpm2StepProjection.Run(w, Rand(), Rand(), Rand(), HiddenDim, ditHiddenDim, latentDim, ScalarQuantizationScale);

        Assert.Equal(HiddenDim, output.FsqHidden.Length);
        Assert.Equal(ditHiddenDim, output.CurrentLmDitHidden.Length);
        Assert.Equal(2, output.CurrentStopLogits.Length);
        foreach (var arr in new[] { output.FsqHidden, output.CurrentResidualInput, output.ResidualInput, output.CurrentLmDitHidden, output.FsqLmDitHidden, output.ResidualDitHidden, output.CurrentStopLogits, output.FsqStopLogits })
            Assert.All(arr, v => Assert.True(float.IsFinite(v)));
    }
}
