using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.AceStep;
using OpenTail.Stingray.Diffusion.AceStep.Transformer;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion.AceStep;

/// <summary>
/// First real-weight smoke test for ACE-Step Turbo's 24-layer DiT. Non-degeneracy receipt (finite,
/// non-trivial, shape-correct, sensitive to real inputs), not yet a numeric golden-parity test
/// against a real `diffusers` `AceStepTransformer1DModel` reference run. No real condition
/// encoder exists yet (see docs/064-acestep-implementation-plan.md), so this test drives the DiT
/// with a synthetic (but real-shaped) condition sequence -- sufficient to validate that the real
/// weight loading and forward-pass math run correctly end to end without NaN/crashes.
/// </summary>
public sealed class AceStepDiTTests
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
    public void Forward_RealWeights_ProducesNonDegenerateOutput()
    {
        string? turboPath = FindRepoFile("models/acestep-v15/turbo.safetensors");
        Assert.SkipUnless(turboPath != null, "models/acestep-v15/turbo.safetensors not found");

        using var loader = SafetensorsLoader.Open(turboPath!);
        var weights = AceStepDiTWeights.Load(loader);

        int frames = 20; // 0.8s @ 25Hz, small enough to run quickly
        var rng = new Random(0);

        float[][] RandomRows(int n, int dim, float scale)
        {
            var rows = new float[n][];
            for (int i = 0; i < n; i++)
            {
                rows[i] = new float[dim];
                for (int d = 0; d < dim; d++) rows[i][d] = (float)(rng.NextDouble() * 2 - 1) * scale;
            }
            return rows;
        }

        var noisyLatent = RandomRows(frames, AceStepConfig.AudioAcousticHiddenDim, 1.0f);
        var contextLatents = RandomRows(frames, 2 * AceStepConfig.AudioAcousticHiddenDim, 0.5f); // src_latents(64) + chunk_masks(64)
        var condition = RandomRows(30, AceStepConfig.HiddenSize, 0.3f); // synthetic packed text+lyric+timbre condition

        var (patches, originalSeqLen) = AceStepDiT.ProjIn(weights, contextLatents, noisyLatent);
        Assert.Equal(frames, originalSeqLen);
        Assert.Equal((frames + 1) / 2, patches.Length); // patch_size=2, frames=20 is already even

        var ctx = AceStepDiT.PrepareCrossAttention(weights, condition, patches.Length);
        var ditOut = AceStepDiT.Forward(weights, patches, timestep: 1.0f, timestepR: 1.0f, ctx);
        Assert.Equal(patches.Length, ditOut.Length);
        foreach (var row in ditOut)
        {
            Assert.Equal(AceStepConfig.HiddenSize, row.Length);
            foreach (var v in row)
                Assert.True(float.IsFinite(v), "DiT hidden output contains NaN/Inf -- degenerate");
        }

        var velocity = AceStepDiT.ProjOut(weights, ditOut, originalSeqLen);
        Assert.Equal(frames, velocity.Length);
        foreach (var row in velocity)
        {
            Assert.Equal(AceStepConfig.AudioAcousticHiddenDim, row.Length);
            foreach (var v in row)
                Assert.True(float.IsFinite(v), "DiT velocity output contains NaN/Inf -- degenerate");
        }

        double sumSq = 0;
        int count = 0;
        foreach (var row in velocity) foreach (var v in row) { sumSq += (double)v * v; count++; }
        double rms = Math.Sqrt(sumSq / count);
        Assert.True(rms > 1e-6, $"velocity RMS ({rms}) is near-zero -- likely a wiring bug");

        // Different timesteps must produce different output (a real, non-degenerate diffusion
        // model is timestep-conditioned) -- a common wiring bug is dropping the timestep
        // embedding entirely, which would make this fail.
        var ctx2 = AceStepDiT.PrepareCrossAttention(weights, condition, patches.Length);
        var ditOut2 = AceStepDiT.Forward(weights, patches, timestep: 0.2f, timestepR: 0.2f, ctx2);
        double diff = 0;
        for (int i = 0; i < ditOut.Length; i++)
            for (int d = 0; d < AceStepConfig.HiddenSize; d++)
                diff += Math.Abs(ditOut[i][d] - ditOut2[i][d]);
        Assert.True(diff > 1e-3, "DiT output identical across different timesteps -- timestep conditioning likely broken");
    }
}
