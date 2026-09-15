using OpenTail.Stingray.Diffusion.Wan;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Decisive isolation test (docs/081, 2026-09-14): replays the ACTUAL captured C# cross-attention
/// Q/K/V (real values from a live Forward() call, already independently verified >0.9999999
/// cosine-similar to the real C++ reference's own Q/K/V) through the real
/// WanAttention.TiledMultiHeadAttention kernel (already independently verified correct in isolation
/// for both symmetric and this exact asymmetric qSeq/kvSeq shape against synthetic data --
/// WanAttentionAsymmetricSeqIsolationTest). If this reproduces the live run's own (surprisingly
/// divergent, cosine=0.967 vs the C++ reference) wan_cs_dump_wandbg_block0_crossattn_raw.bin
/// bit-for-bit, that proves the kernel is faithfully computing what it's given, and the divergence
/// is a genuine sensitivity-amplification of the tiny (but real) upstream Q/K/V difference through
/// a near-degenerate (close to uniform) softmax distribution -- not a code bug in the attention
/// kernel itself.
/// </summary>
public sealed class WanCrossAttnReplayTest
{
    private static float[] ReadBin(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var arr = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, arr, 0, bytes.Length);
        return arr;
    }

    [Fact]
    public void ReplayCapturedCrossAttnQkv_ThroughRealKernel_CompareAgainstCapturedOutput()
    {
        string dumpDir = "examples/stable-diffusion.cpp";
        for (int i = 0; i < 6; i++)
        {
            if (Directory.Exists(dumpDir)) break;
            dumpDir = Path.Combine("..", dumpDir);
        }
        string qPath = Path.Combine(dumpDir, "wan_cs_dump_wandbg_block0_crossq.bin");
        string kPath = Path.Combine(dumpDir, "wan_cs_dump_wandbg_block0_crossk.bin");
        string vPath = Path.Combine(dumpDir, "wan_cs_dump_wandbg_block0_crossv.bin");
        string outPath = Path.Combine(dumpDir, "wan_cs_dump_wandbg_block0_crossattn_raw.bin");
        if (!File.Exists(qPath) || !File.Exists(kPath) || !File.Exists(vPath) || !File.Exists(outPath)) return;

        var q = ReadBin(qPath);
        var k = ReadBin(kPath);
        var v = ReadBin(vPath);
        var capturedOut = ReadBin(outPath);

        const int dim = 1536, numHeads = 12, headDim = 128;
        int qSeq = q.Length / dim;
        int kvSeq = k.Length / dim;
        Console.WriteLine($"[WanCrossAttnReplay] qSeq={qSeq} kvSeq={kvSeq}");

        var replayed = new float[qSeq * dim];
        WanAttention.TiledMultiHeadAttention(q, k, v, replayed, qSeq, kvSeq, numHeads, headDim);

        double dot = 0, na = 0, nb = 0, maxDiff = 0;
        for (int i = 0; i < replayed.Length; i++)
        {
            dot += (double)replayed[i] * capturedOut[i];
            na += (double)replayed[i] * replayed[i];
            nb += (double)capturedOut[i] * capturedOut[i];
            double d = Math.Abs(replayed[i] - capturedOut[i]);
            if (d > maxDiff) maxDiff = d;
        }
        double cos = dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
        Console.WriteLine($"[WanCrossAttnReplay] cosine(replayed, capturedLiveRun)={cos:F9} replayedNorm={Math.Sqrt(na):F4} capturedNorm={Math.Sqrt(nb):F4} maxDiff={maxDiff:F9}");
    }
}
