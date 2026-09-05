using System.Diagnostics;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.MiniMaxMusic3;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion.MiniMaxMusic3;

/// <summary>
/// Scratch: measures a single real ForwardPair call at the REAL scale used by the 200-frame/8s
/// generation (T=689 latent frames, matching MiniMaxMusic3ConditionEncoder's real resample of
/// 200 25Hz frames -> ~86.13Hz), to isolate the DiT stage's real per-step cost independent of the
/// rest of the pipeline (AR loop, condition encoder, vocoder). Compares against the real
/// minimaxmusic.cpp C++ reference's own measured 34.3s/step at the identical T=689 scale
/// (docs/066, 2026-09-05). NOT a golden-parity check. Delete once superseded.
/// </summary>
public sealed class ZZ_ScratchDitForwardPairFullScaleBenchTests
{
    private static string? FindRepoDir(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void ForwardPair_RealWeights_T689_MeasuresRealPerStepCost()
    {
        string? weightsDir = FindRepoDir("models/minimax-music3/transformer");
        Assert.SkipUnless(weightsDir != null, "models/minimax-music3/transformer/ not found");

        const int length = 689;
        const int condDim = 2048;
        const int inChannels = 128;
        const float timestep = 0.37f;

        using var loader = SafetensorsLoader.OpenDirectory(weightsDir!);
        var weights = MiniMaxMusic3TransformerWeights.Load(loader);

        var random = new Random(42);
        var latent = new float[length][];
        var condition = new float[length][];
        var zeroCondition = new float[length][];
        for (int t = 0; t < length; t++)
        {
            latent[t] = new float[inChannels];
            for (int c = 0; c < inChannels; c++) latent[t][c] = (float)random.NextDouble() - 0.5f;

            condition[t] = new float[condDim];
            for (int c = 0; c < condDim; c++) condition[t][c] = (float)random.NextDouble() - 0.5f;

            zeroCondition[t] = new float[condDim];
        }

        // Warmup (JIT, weight paging).
        _ = MiniMaxMusic3Transformer.ForwardPair(weights, latent, condition, zeroCondition, timestep);

        var sw = Stopwatch.StartNew();
        var (cond, uncond) = MiniMaxMusic3Transformer.ForwardPair(weights, latent, condition, zeroCondition, timestep);
        sw.Stop();

        Console.WriteLine($"[bench] ForwardPair T={length} (batch=2): {sw.Elapsed.TotalSeconds:F2}s, ProcessorCount={Environment.ProcessorCount}");
        Assert.Equal(length, cond.Length);
        Assert.Equal(length, uncond.Length);
    }
}
