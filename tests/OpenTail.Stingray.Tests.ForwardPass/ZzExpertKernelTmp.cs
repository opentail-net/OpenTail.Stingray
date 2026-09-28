using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using Xunit;

namespace OpenTail.Stingray.Tests.ForwardPass;

// Scratch (untracked): batched int8 matmul vs per-row MatVec on real LFM2-MoE expert weights.
public sealed unsafe class ZzExpertKernelTmp
{
    [Fact]
    public void Run()
    {
        var path = Environment.GetEnvironmentVariable("ZZ_EXPERT_MODEL");
        if (path is null) return;
        using var model = GgufModel.Open(path);
        foreach (var (name, rows, cols) in new[] { ("blk.2.ffn_gate_exps.weight", 1792, 2048), ("blk.2.ffn_down_exps.weight", 2048, 1792) })
        {
            var info = model.FindTensor(name)!.Value;
            byte* w = (byte*)model.GetTensorDataPtr(info); // expert 0
            var rng = new Random(1);
            foreach (int n in new[] { 1, 3, 4, 7, 8, 16, 40, 64 })
            {
                var x = new float[n * cols];
                for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
                var yb = new float[n * rows];
                var yr = new float[n * rows];
                fixed (float* xp = x, bp = yb, rp = yr)
                {
                    SimdKernels.MatMulBatched(bp, w, xp, n, rows, cols, info.DType, allowQ8: true);
                    for (int t = 0; t < n; t++) SimdKernels.MatVec(rp + (long)t * rows, w, xp + (long)t * cols, rows, cols, info.DType);
                }
                double d = 0, a = 0;
                for (int i = 0; i < yb.Length; i++) { d += (yb[i] - yr[i]) * (double)(yb[i] - yr[i]); a += (double)yr[i] * yr[i]; }
                Console.WriteLine($"[ZzEK] {name} {info.DType} n={n} relErr={Math.Sqrt(d / a):E3}");
            }
        }
    }
}
