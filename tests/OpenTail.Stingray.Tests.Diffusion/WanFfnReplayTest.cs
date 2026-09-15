using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Decisive isolation test (docs/081, 2026-09-14): replays the ACTUAL captured C# "ffnin" dump
/// (real values from a live Forward() call) through the ACTUAL loaded blocks.0.ffn.0 weight/bias
/// (both independently verified correct in isolation -- WanFfnMatMulIsolationTest,
/// WanFfnWeightLoadCheckTest) via the real SimdKernels.MatMulBatchedF32 kernel. If this reproduces
/// the same ~2x-too-large, wrong-direction output seen in the live run's own
/// wan_cs_dump_wandbg_block0_ffn0.bin, the bug is data-dependent in kernel+weights after all. If it
/// does NOT reproduce it, the live run's actual FeedForward call used different data than what was
/// captured as "ffnin" -- an aliasing/ordering bug elsewhere in the real code path.
/// </summary>
public sealed class WanFfnReplayTest
{
    private static float[] ReadBin(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var arr = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, arr, 0, bytes.Length);
        return arr;
    }

    [Fact]
    public unsafe void ReplayCapturedFfnin_ThroughRealWeights_CompareAgainstCapturedOutputs()
    {
        string dumpDir = "examples/stable-diffusion.cpp";
        for (int i = 0; i < 6; i++)
        {
            if (Directory.Exists(Path.Combine(dumpDir))) break;
            dumpDir = Path.Combine("..", dumpDir);
        }
        string ffninPath = Path.Combine(dumpDir, "wan_cs_dump_wandbg_block0_ffnin.bin");
        string ffn0Path = Path.Combine(dumpDir, "wan_cs_dump_wandbg_block0_ffn0.bin");
        string modelPath = @"C:\Git-Public\OpenTail.Stingray\models\wan2.1\wan2.1-t2v-1.3b-dit.safetensors";
        if (!File.Exists(ffninPath) || !File.Exists(ffn0Path) || !File.Exists(modelPath)) return;

        var ffnin = ReadBin(ffninPath);
        var capturedFfn0 = ReadBin(ffn0Path);

        using var loader = SafetensorsLoader.Open(modelPath);
        var w = loader.ReadF32("blocks.0.ffn.0.weight");
        var b = loader.ReadF32("blocks.0.ffn.0.bias");

        const int dim = 1536, ffnDim = 8960;
        int numTokens = ffnin.Length / dim;
        Console.WriteLine($"[WanFfnReplay] numTokens={numTokens} ffnin.Length={ffnin.Length} capturedFfn0.Length={capturedFfn0.Length}");

        var replayed = new float[numTokens * ffnDim];
        fixed (float* pOut = replayed, pW = w, pIn = ffnin, pB = b)
        {
            SimdKernels.MatMulBatchedF32(pOut, pW, pIn, numTokens, ffnDim, dim, pB);
        }

        double dot = 0, na = 0, nb = 0, maxDiff = 0;
        for (int i = 0; i < replayed.Length; i++)
        {
            dot += (double)replayed[i] * capturedFfn0[i];
            na += (double)replayed[i] * replayed[i];
            nb += (double)capturedFfn0[i] * capturedFfn0[i];
            double d = Math.Abs(replayed[i] - capturedFfn0[i]);
            if (d > maxDiff) maxDiff = d;
        }
        double cos = dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
        Console.WriteLine($"[WanFfnReplay] cosine(replayed, capturedLiveRun)={cos:F9} replayedNorm={Math.Sqrt(na):F4} capturedNorm={Math.Sqrt(nb):F4} maxDiff={maxDiff:F6}");
        Console.WriteLine($"[WanFfnReplay] replayed[0..4]={replayed[0]:F6},{replayed[1]:F6},{replayed[2]:F6},{replayed[3]:F6}");
        Console.WriteLine($"[WanFfnReplay] captured[0..4]={capturedFfn0[0]:F6},{capturedFfn0[1]:F6},{capturedFfn0[2]:F6},{capturedFfn0[3]:F6}");
    }
}
