using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.AceStep;
using OpenTail.Stingray.Diffusion.AceStep.Transformer;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion.AceStep;

/// <summary>
/// First real-weight smoke test for <see cref="AceStepFlowScheduler"/>: the real 8-step Turbo
/// Euler-ODE loop against the real 4.79GB `turbo.safetensors` DiT weights, with a synthetic
/// (real-shaped) condition sequence (no condition encoder involved -- purely validates the
/// scheduler's own denoising loop, timestep-schedule selection, and ProjIn/Forward/ProjOut wiring
/// across real repeated steps). Non-degeneracy receipt, not yet a numeric golden-parity test.
/// </summary>
public sealed class AceStepFlowSchedulerTests
{
    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Generate_RealWeights_ProducesFiniteNonDegenerateLatent()
    {
        string? turboPath = FindRepoFile("models/acestep-v15/turbo.safetensors");
        Assert.SkipUnless(turboPath != null, "models/acestep-v15/turbo.safetensors not found");

        using var loader = SafetensorsLoader.Open(turboPath!);
        var weights = AceStepDiTWeights.Load(loader);

        int condLen = 32;
        var rng = new Random(0);
        var condition = new float[condLen][];
        for (int i = 0; i < condLen; i++)
        {
            var row = new float[AceStepConfig.HiddenSize];
            for (int d = 0; d < row.Length; d++) row[d] = (float)(rng.NextDouble() * 0.2 - 0.1);
            condition[i] = row;
        }

        int latentFrames = 50; // 2 real seconds @ 25Hz
        var result = AceStepFlowScheduler.Generate(weights, condition, latentFrames, shift: 3.0f, seed: 1234);

        Assert.Equal(latentFrames, result.Length);
        foreach (var row in result)
        {
            Assert.Equal(AceStepConfig.AudioAcousticHiddenDim, row.Length);
            foreach (var v in row)
                Assert.True(float.IsFinite(v), "flow scheduler output contains NaN/Inf -- degenerate");
        }

        double sumSq = 0;
        int count = 0;
        foreach (var row in result)
            foreach (var v in row) { sumSq += (double)v * v; count++; }
        double rms = Math.Sqrt(sumSq / count);
        Assert.True(rms > 1e-4, $"flow scheduler output RMS ({rms}) is near-zero -- likely a wiring bug");

        // Real sensitivity check: a different seed's initial noise should reach a measurably
        // different final latent (a wiring bug that ignored `xt`/noise entirely, e.g. always
        // returning the DiT's raw velocity prediction, could still pass the checks above).
        var result2 = AceStepFlowScheduler.Generate(weights, condition, latentFrames, shift: 3.0f, seed: 5678);
        double diff = 0;
        for (int t = 0; t < latentFrames; t++)
            for (int d = 0; d < AceStepConfig.AudioAcousticHiddenDim; d++)
                diff += Math.Abs(result[t][d] - result2[t][d]);
        Assert.True(diff > 1e-2, "different seeds produced (near-)identical final latents -- likely a wiring bug");
    }
}
