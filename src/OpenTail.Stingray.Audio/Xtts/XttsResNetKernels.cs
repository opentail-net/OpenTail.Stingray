
namespace OpenTail.Stingray.Audio.Xtts;

/// <summary>
/// Generic real 2D-conv/BatchNorm2d/SE-block math for XTTS-v2's `ResNetSpeakerEncoder`
/// (`TTS/encoder/models/resnet.py`) -- this codebase's other pipelines are all 1D channel-first
/// audio convs; this is the first CPU 2D-conv usage in Audio (Diffusion's own `Conv2d` is a
/// GPU-only Vulkan compute path, not reusable here). Correctness-first, not perf-optimized (no
/// im2col/SIMD) -- the speaker encoder runs once per reference-audio clip, not per generation
/// step, so this is not a hot path; revisit in a perf pass only if profiling shows otherwise
/// (CLAUDE.md rule 7).
/// </summary>
public static class XttsResNetKernels
{
    /// <summary>Real Conv2d, channel-first [inCh,H,W], "same"-ish padding=1 (matches every real conv in this network: kernel=3,pad=1 or kernel=1,pad=0), configurable stride. `bias` is optional -- most convs in this network are `bias=False` (pass null), but the real stem `conv1` (unlike every `SEBasicBlock` conv) DOES have a bias term, easy to miss since it's the one exception.</summary>
    public static float[] Conv2d(float[] input, int inCh, int h, int w, float[] weight, int outCh, int kernel, int stride, int pad, out int outH, out int outW, float[]? bias = null)
    {
        outH = (h + 2 * pad - kernel) / stride + 1;
        outW = (w + 2 * pad - kernel) / stride + 1;
        int outHLocal = outH, outWLocal = outW;
        var output = new float[outCh * outHLocal * outWLocal];

        System.Threading.Tasks.Parallel.For(0, outCh, oc =>
        {
            int wOcBase = oc * inCh * kernel * kernel;
            float b = bias?[oc] ?? 0f;
            for (int oy = 0; oy < outHLocal; oy++)
            {
                for (int ox = 0; ox < outWLocal; ox++)
                {
                    float sum = b;
                    for (int ic = 0; ic < inCh; ic++)
                    {
                        int inBase = ic * h * w;
                        int wIcBase = wOcBase + ic * kernel * kernel;
                        for (int ky = 0; ky < kernel; ky++)
                        {
                            int iy = oy * stride - pad + ky;
                            if ((uint)iy >= (uint)h) continue;
                            int rowBase = inBase + iy * w;
                            int wRowBase = wIcBase + ky * kernel;
                            for (int kx = 0; kx < kernel; kx++)
                            {
                                int ix = ox * stride - pad + kx;
                                if ((uint)ix >= (uint)w) continue;
                                sum += input[rowBase + ix] * weight[wRowBase + kx];
                            }
                        }
                    }
                    output[(oc * outHLocal + oy) * outWLocal + ox] = sum;
                }
            }
        });

        return output;
    }

    /// <summary>Real BatchNorm2d, inference mode: `(x-runningMean)/sqrt(runningVar+eps)*weight+bias`, per channel, broadcast over H,W.</summary>
    public static void BatchNorm2dInPlace(float[] x, int ch, int h, int w, float[] weight, float[] bias, float[] runningMean, float[] runningVar, float eps = 1e-5f)
    {
        int hw = h * w;
        for (int c = 0; c < ch; c++)
        {
            float invStd = 1f / MathF.Sqrt(runningVar[c] + eps);
            float scale = weight[c] * invStd;
            float shift = bias[c] - runningMean[c] * scale;
            int baseIdx = c * hw;
            for (int i = 0; i < hw; i++) x[baseIdx + i] = x[baseIdx + i] * scale + shift;
        }
    }

    /// <summary>Real BatchNorm1d, inference mode, channel-first [ch,T].</summary>
    public static void BatchNorm1dInPlace(float[] x, int ch, int t, float[] weight, float[] bias, float[] runningMean, float[] runningVar, float eps = 1e-5f)
    {
        for (int c = 0; c < ch; c++)
        {
            float invStd = 1f / MathF.Sqrt(runningVar[c] + eps);
            float scale = weight[c] * invStd;
            float shift = bias[c] - runningMean[c] * scale;
            int baseIdx = c * t;
            for (int i = 0; i < t; i++) x[baseIdx + i] = x[baseIdx + i] * scale + shift;
        }
    }

    public static void ReluInPlace(float[] x)
    {
        for (int i = 0; i < x.Length; i++) if (x[i] < 0f) x[i] = 0f;
    }

    /// <summary>Real `SELayer`: global-average-pool (over H,W) -> FC(ch->ch/r) -> ReLU -> FC(ch/r->ch) -> Sigmoid -> channel-wise rescale.</summary>
    public static float[] SqueezeExcite(float[] x, int ch, int h, int w, float[] fc0Weight, float[] fc0Bias, float[] fc2Weight, float[] fc2Bias, int reducedCh)
    {
        int hw = h * w;
        var pooled = new float[ch];
        for (int c = 0; c < ch; c++)
        {
            float sum = 0f;
            int baseIdx = c * hw;
            for (int i = 0; i < hw; i++) sum += x[baseIdx + i];
            pooled[c] = sum / hw;
        }

        var hidden = new float[reducedCh];
        for (int o = 0; o < reducedCh; o++)
        {
            float sum = fc0Bias[o];
            int wBase = o * ch;
            for (int i = 0; i < ch; i++) sum += fc0Weight[wBase + i] * pooled[i];
            hidden[o] = MathF.Max(0f, sum); // ReLU
        }

        var gate = new float[ch];
        for (int o = 0; o < ch; o++)
        {
            float sum = fc2Bias[o];
            int wBase = o * reducedCh;
            for (int i = 0; i < reducedCh; i++) sum += fc2Weight[wBase + i] * hidden[i];
            gate[o] = 1f / (1f + MathF.Exp(-sum)); // Sigmoid
        }

        var output = new float[x.Length];
        for (int c = 0; c < ch; c++)
        {
            float g = gate[c];
            int baseIdx = c * hw;
            for (int i = 0; i < hw; i++) output[baseIdx + i] = x[baseIdx + i] * g;
        }
        return output;
    }
}
