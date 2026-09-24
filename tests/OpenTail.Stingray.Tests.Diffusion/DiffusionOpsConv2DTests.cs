using OpenTail.Stingray.Diffusion;

namespace OpenTail.Stingray.Tests.Diffusion;

// Guards the im2col+GEMM CPU Conv2D against the original scalar direct convolution.
public sealed class DiffusionOpsConv2DTests
{
    static float[] Ref(float[] input, float[] kernel, float[]? bias, int n, int inC, int inH, int inW, int outC, int kH, int kW, int stride, int padding)
    {
        int outH = (inH + 2 * padding - kH) / stride + 1, outW = (inW + 2 * padding - kW) / stride + 1;
        var output = new float[n * outC * outH * outW];
        int hw = outH * outW, inHW = inH * inW;
        Parallel.For(0, n * outC, bo =>
        {
            int b = bo / outC, oc = bo % outC;
            for (int oh = 0; oh < outH; oh++)
                for (int ow = 0; ow < outW; ow++)
                {
                    float sum = bias?[oc] ?? 0f;
                    for (int ic = 0; ic < inC; ic++)
                        for (int kh = 0; kh < kH; kh++)
                        {
                            int ih = oh * stride - padding + kh; if ((uint)ih >= (uint)inH) continue;
                            for (int kw = 0; kw < kW; kw++)
                            {
                                int iw = ow * stride - padding + kw; if ((uint)iw >= (uint)inW) continue;
                                sum += input[b * inC * inHW + ic * inHW + ih * inW + iw] * kernel[((oc * inC + ic) * kH + kh) * kW + kw];
                            }
                        }
                    output[b * outC * hw + oc * hw + oh * outW + ow] = sum;
                }
        });
        return output;
    }

    static float[] R(int len, Random r) { var a = new float[len]; for (int i = 0; i < len; i++) a[i] = (float)(r.NextDouble() * 2 - 1); return a; }

    [Fact]
    public void Conv2D_Im2ColGemm_MatchesScalarReference()
    {
        var rng = new Random(3);
        foreach (var (n, inC, h, w, outC, k, s, pad, useBias) in new[] {
            (1, 3, 7, 5, 4, 3, 1, 1, true), (2, 5, 9, 11, 6, 3, 2, 1, true), (1, 8, 6, 6, 3, 3, 2, 0, false),
            (1, 16, 13, 10, 32, 3, 1, 1, true), (2, 4, 8, 8, 5, 5, 1, 2, false), (1, 64, 16, 16, 64, 3, 1, 1, true) })
        {
            var x = R(n * inC * h * w, rng); var kk = R(outC * inC * k * k, rng); var b = useBias ? R(outC, rng) : null;
            var a1 = Ref(x, kk, b, n, inC, h, w, outC, k, k, s, pad);
            var a2 = DiffusionOps.Conv2D(x, kk, b, n, inC, h, w, outC, k, k, s, pad);
            Assert.Equal(a1.Length, a2.Length);
            float md = 0, mref = 0; for (int i = 0; i < a1.Length; i++) { md = MathF.Max(md, MathF.Abs(a1[i] - a2[i])); mref = MathF.Max(mref, MathF.Abs(a1[i])); }
            Console.WriteLine($"[ConvCheck] n={n} inC={inC} {h}x{w} outC={outC} k={k} s={s} p={pad} maxDiff={md:E2} (max|ref|={mref:F2})");
            Assert.True(md <= 1e-4f * MathF.Max(1f, mref), "mismatch");
        }
    }
}
