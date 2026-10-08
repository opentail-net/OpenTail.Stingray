using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.ForwardPass;

// Scratch (untracked): 11.d isolation. Env STINGRAY_HYBRID_GDN_MODEL.
public sealed class ZzGdnChunkDiagTmp
{
    [Fact]
    public void Run()
    {
        var path = Environment.GetEnvironmentVariable("STINGRAY_HYBRID_GDN_MODEL");
        if (path is null) return;
        using var model = GgufModel.Open(path);
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        var tok = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        using var fwd = new HybridGdnForwardPass(model, backend, hp);
        var tokens = tok.Encode(
            "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. " +
            "How razorback-jumping frogs can level six piqued gymnasts! Sphinx of black quartz, judge my vow. " +
            "Five wizards quickly jinxed the gnome before it vaporized. Crazy Fredrick bought many very exquisite opal jewels. " +
            "The jay, pig, fox, zebra, and my wolves quack! We promptly judged antique ivory buckles for the next prize.");
        foreach (var (q8, moe) in new[] { (true, true), (false, true), (true, false), (false, false) })
        {
            SimdKernels.Q8PrefillEnabled = q8;
            HybridGdnForwardPass.MoeBatchedPrefillEnabled = moe;
            HybridGdnForwardPass.GdnChunkedPrefillEnabled = false;
            fwd.ResetCache();
            var seq = fwd.Prefill(tokens).ToArray();
            HybridGdnForwardPass.GdnChunkedPrefillEnabled = true;
            fwd.ResetCache();
            var chunk = fwd.Prefill(tokens).ToArray();
            double maxd = 0, d = 0, a = 0, b = 0; int mi = -1;
            for (int i = 0; i < seq.Length; i++)
            {
                double df = Math.Abs(seq[i] - chunk[i]); if (df > maxd) { maxd = df; mi = i; }
                d += (double)seq[i] * chunk[i]; a += (double)seq[i] * seq[i]; b += (double)chunk[i] * chunk[i];
            }
            Console.WriteLine($"[ZzGdn] q8={q8} moeBatched={moe} n={tokens.Count} maxAbs={maxd:F4} at {mi} (seq {seq[mi]:F4} chunk {chunk[mi]:F4}) cos={d / Math.Sqrt(a * b):F7} argmax {Sampler.Greedy(seq)} vs {Sampler.Greedy(chunk)}");
        }
    }
}
