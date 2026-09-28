using System.Numerics.Tensors;
namespace OpenTail.Stingray.Tests.Diffusion;
public sealed class ZzConv1x1BenchTmp
{
    [Fact]
    public void Bench()
    {
        foreach (var (inC, outC, hw) in new[] { (512, 512, 4096), (128, 256, 65536), (256, 256, 16384), (64, 64, 65536), (128, 128, 16384), (32, 64, 65536), (512, 16, 4096) })
        {
            var rng = new Random(1);
            var x = new float[inC * hw]; var k = new float[outC * inC]; var b = new float[outC];
            for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < k.Length; i++) k[i] = (float)(rng.NextDouble() * 2 - 1) * 0.05f;
            for (int i = 0; i < b.Length; i++) b[i] = (float)rng.NextDouble();
            float[] got = Array.Empty<float>();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            got = DiffusionOps.Conv2D(x, k, b, 1, inC, 1, hw, outC, 1, 1, 1, 0); sw.Restart(); for (int r = 0; r < 5; r++) got = DiffusionOps.Conv2D(x, k, b, 1, inC, 1, hw, outC, 1, 1, 1, 0);
            double tNew = sw.Elapsed.TotalMilliseconds / 5;
            // old formulation: per-output Dot over token-major input
            sw.Restart();
            var want = new float[outC * hw];
            for (int r = 0; r < 6; r++)
            {
                if (r == 1) sw.Restart();
                var packedIn = new float[hw * inC];
                Parallel.For(0, hw, p => { for (int ic = 0; ic < inC; ic++) packedIn[p * inC + ic] = x[ic * hw + p]; });
                Parallel.For(0, hw, p =>
                {
                    for (int oc = 0; oc < outC; oc++)
                        want[oc * hw + p] = TensorPrimitives.Dot(packedIn.AsSpan(p * inC, inC), k.AsSpan(oc * inC, inC)) + b[oc];
                });
            }
            double tOld = sw.Elapsed.TotalMilliseconds / 5;
            double worst = 0; for (int i = 0; i < want.Length; i++) worst = Math.Max(worst, Math.Abs(got[i] - want[i]));
            Console.WriteLine($"1x1 inC={inC} outC={outC} hw={hw}: dot {tOld:F1} ms -> packed {tNew:F1} ms, max abs diff {worst:G3}");
            Assert.True(worst < 1e-3);
        }
    }
}
