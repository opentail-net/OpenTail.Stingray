using System.Buffers;
using System.Numerics.Tensors;
using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// Universal VAE decoder: decodes latent tensors to RGB image [1, 3, H*8, W*8].
/// Supports:
///   - 16-channel latents (FLUX.1, Z-Image-Turbo)
///   - 4-channel latents (Stable Diffusion 1.5, SDXL)
///
/// Handles standalone VAE checkpoints (ae.safetensors) and unified model weights
/// with both CompVis (first_stage_model.decoder.mid.block_1...) and Diffusers (decoder.mid_block.resnets.0...) schemas.
/// </summary>
public sealed class VaeDecoder : IDisposable, IVaeDecoder
{
    private readonly IWeightLoader _st;
    private readonly IComputeBackend? _backend;

    private readonly Dictionary<string, float[]> _cpuWeights = new(StringComparer.Ordinal);
    private readonly Dictionary<string, CoreTensor>? _gpuWeights;

    private const float VaeShift = 0.1159f;
    private const int MaxColChunkFloats = 32 * 1024 * 1024;

    public VaeDecoder(string path) => _st = SafetensorsLoader.Open(path);

    public VaeDecoder(IWeightLoader st) => _st = st;

    public VaeDecoder(IWeightLoader st, IComputeBackend? backend = null)
    {
        _st      = st;
        _backend = backend;
        if (backend is not null)
            _gpuWeights = new Dictionary<string, CoreTensor>(StringComparer.Ordinal);
    }

    private string Resolve(string name)
    {
        if (_st.Contains(name)) return name;
        if (_st.Contains("first_stage_model." + name)) return "first_stage_model." + name;
        if (_st.Contains("vae." + name)) return "vae." + name;
        return name;
    }

    /// <summary>
    /// Decode latent [C, H, W] → RGB float [3, H*8, W*8], values in [0,1].
    /// </summary>
    public float[] Decode(float[] latent, int latH, int latW)
    {
        int latentCh = latent.Length / (latH * latW);
        float scale = latentCh == 4 ? (1f / 0.18215f) : (1f / 0.3611f);
        float shift = latentCh == 4 ? 0f : VaeShift;

        var z = new float[latent.Length];
        for (int i = 0; i < z.Length; i++)
            z[i] = latent[i] * scale + shift;

        // post_quant_conv: Conv2D(C→C, 1×1)
        string pqKey = Resolve("post_quant_conv");
        if (_st.Contains($"{pqKey}.weight"))
            z = ConvBlock(pqKey, z, 1, latentCh, latH, latW, latentCh, 1, padding: 0);

        // conv_in: Conv2D(C→512, 3×3)
        string convInKey = Resolve("decoder.conv_in");
        z = ConvBlock(convInKey, z, 1, latentCh, latH, latW, 512, 3);
        int ch = 512, h = latH, w = latW;

        bool isCompVis = _st.Contains(Resolve("decoder.mid.block_1.conv1.weight"));

        if (isCompVis)
        {
            // CompVis SD1.5 schema
            string dec = Resolve("decoder");
            z = ResBlock($"{dec}.mid.block_1", z, 1, ch, h, w);
            z = MidAttnCompVis($"{dec}.mid.attn_1", z, 1, ch, h, w);
            z = ResBlock($"{dec}.mid.block_2", z, 1, ch, h, w);

            // Up blocks: up.3 (512, upsample), up.2 (512, upsample), up.1 (256, upsample), up.0 (128, no upsample)
            (z, ch, h, w) = UpBlockCompVis(z, 1, ch, h, w, $"{dec}.up.3", outCh: 512, upsample: true);
            (z, ch, h, w) = UpBlockCompVis(z, 1, ch, h, w, $"{dec}.up.2", outCh: 512, upsample: true);
            (z, ch, h, w) = UpBlockCompVis(z, 1, ch, h, w, $"{dec}.up.1", outCh: 256, upsample: true);
            (z, ch, h, w) = UpBlockCompVis(z, 1, ch, h, w, $"{dec}.up.0", outCh: 128, upsample: false);
        }
        else
        {
            // Diffusers / FLUX schema
            string dec = Resolve("decoder");
            z = ResBlock($"{dec}.mid_block.resnets.0", z, 1, ch, h, w);
            z = MidAttnDiffusers($"{dec}.mid_block.attentions.0", z, 1, ch, h, w);
            z = ResBlock($"{dec}.mid_block.resnets.1", z, 1, ch, h, w);

            (z, ch, h, w) = UpBlockDiffusers(z, 1, ch, h, w, $"{dec}.up_blocks.0", outCh: 512, upsample: true);
            (z, ch, h, w) = UpBlockDiffusers(z, 1, ch, h, w, $"{dec}.up_blocks.1", outCh: 512, upsample: true);
            (z, ch, h, w) = UpBlockDiffusers(z, 1, ch, h, w, $"{dec}.up_blocks.2", outCh: 256, upsample: true);
            (z, ch, h, w) = UpBlockDiffusers(z, 1, ch, h, w, $"{dec}.up_blocks.3", outCh: 128, upsample: false);
        }

        // norm_out
        string normName = _st.Contains(Resolve("decoder.conv_norm_out.weight")) ? Resolve("decoder.conv_norm_out")
                                                                                : Resolve("decoder.norm_out");
        var gnW = Wt($"{normName}.weight");
        var gnB = Wt($"{normName}.bias");
        DiffusionOps.GroupNorm(z, gnW, gnB, 1, ch, h, w, groups: 32);
        DiffusionOps.SiluInPlace(z);

        // conv_out: Conv2D(128→3)
        string convOutKey = Resolve("decoder.conv_out");
        z = ConvBlock(convOutKey, z, 1, ch, h, w, 3, 3);

        // Clamp to [0, 1]
        for (int i = 0; i < z.Length; i++)
            z[i] = Math.Clamp((z[i] + 1f) * 0.5f, 0f, 1f);

        return z;
    }

    // ── Building blocks ───────────────────────────────────────────────────

    private (float[] z, int ch, int h, int w) UpBlockDiffusers(
        float[] z, int n, int inCh, int h, int w,
        string prefix, int outCh, bool upsample)
    {
        for (int r = 0; r < 3; r++)
        {
            string resPrefix = $"{prefix}.resnets.{r}";
            z = ResBlock(resPrefix, z, n, r == 0 ? inCh : outCh, h, w, outCh);
        }
        int ch = outCh;

        if (upsample)
        {
            z = DiffusionOps.Upsample2x(z, n, ch, h, w);
            h *= 2; w *= 2;
            string convKey = $"{prefix}.upsamplers.0.conv";
            z = ConvBlock(convKey, z, n, ch, h, w, ch, 3);
        }
        return (z, ch, h, w);
    }

    private (float[] z, int ch, int h, int w) UpBlockCompVis(
        float[] z, int n, int inCh, int h, int w,
        string prefix, int outCh, bool upsample)
    {
        for (int r = 0; r < 3; r++)
        {
            string resPrefix = $"{prefix}.block.{r}";
            z = ResBlock(resPrefix, z, n, r == 0 ? inCh : outCh, h, w, outCh);
        }
        int ch = outCh;

        if (upsample)
        {
            z = DiffusionOps.Upsample2x(z, n, ch, h, w);
            h *= 2; w *= 2;
            string convKey = $"{prefix}.upsample.conv";
            z = ConvBlock(convKey, z, n, ch, h, w, ch, 3);
        }
        return (z, ch, h, w);
    }

    private float[] ResBlock(string prefix, float[] x, int n, int inCh, int h, int w, int outCh = -1)
    {
        if (outCh < 0) outCh = inCh;

        // norm1 + silu + conv1
        var gnW1 = Wt($"{prefix}.norm1.weight");
        var gnB1 = Wt($"{prefix}.norm1.bias");
        var h1 = (float[])x.Clone();
        DiffusionOps.GroupNorm(h1, gnW1, gnB1, n, inCh, h, w, groups: 32);
        DiffusionOps.SiluInPlace(h1);
        h1 = ConvBlock($"{prefix}.conv1", h1, n, inCh, h, w, outCh, 3);

        // norm2 + silu + conv2
        var gnW2 = Wt($"{prefix}.norm2.weight");
        var gnB2 = Wt($"{prefix}.norm2.bias");
        DiffusionOps.GroupNorm(h1, gnW2, gnB2, n, outCh, h, w, groups: 32);
        DiffusionOps.SiluInPlace(h1);
        h1 = ConvBlock($"{prefix}.conv2", h1, n, outCh, h, w, outCh, 3);

        // Skip connection: project input if channels differ
        float[] skip = x;
        if (inCh != outCh)
        {
            string shortcutKey = _st.Contains($"{prefix}.nin_shortcut.weight") ? $"{prefix}.nin_shortcut" : $"{prefix}.conv_shortcut";
            skip = ConvBlock(shortcutKey, x, n, inCh, h, w, outCh, 1, padding: 0);
        }

        TensorPrimitives.Add(h1.AsSpan(), skip.AsSpan(), h1.AsSpan());
        return h1;
    }

    private float[] MidAttnCompVis(string prefix, float[] x, int n, int ch, int h, int w)
    {
        var normed = (float[])x.Clone();
        var gnW = Wt($"{prefix}.norm.weight");
        var gnB = Wt($"{prefix}.norm.bias");
        DiffusionOps.GroupNorm(normed, gnW, gnB, n, ch, h, w, groups: 32);

        int hw = h * w;

        var wQ   = Wt($"{prefix}.q.weight");
        var wK   = Wt($"{prefix}.k.weight");
        var wV   = Wt($"{prefix}.v.weight");
        var wOut = Wt($"{prefix}.proj_out.weight");
        var bOut = _st.Contains($"{prefix}.proj_out.bias") ? Wt($"{prefix}.proj_out.bias") : null;

        return ComputeSpatialSelfAttn(x, normed, wQ, wK, wV, wOut, bOut, n, ch, h, w);
    }

    private float[] MidAttnDiffusers(string prefix, float[] x, int n, int ch, int h, int w)
    {
        var normed = (float[])x.Clone();
        var gnW = Wt($"{prefix}.group_norm.weight");
        var gnB = Wt($"{prefix}.group_norm.bias");
        DiffusionOps.GroupNorm(normed, gnW, gnB, n, ch, h, w, groups: 32);

        var wQ   = Wt($"{prefix}.to_q.weight");
        var wK   = Wt($"{prefix}.to_k.weight");
        var wV   = Wt($"{prefix}.to_v.weight");
        var wOut = Wt($"{prefix}.to_out.0.weight");
        var bOut = _st.Contains($"{prefix}.to_out.0.bias") ? Wt($"{prefix}.to_out.0.bias") : null;

        return ComputeSpatialSelfAttn(x, normed, wQ, wK, wV, wOut, bOut, n, ch, h, w);
    }

    private float[] ComputeSpatialSelfAttn(float[] x, float[] normed, float[] wQ, float[] wK, float[] wV, float[] wOut, float[]? bOut, int n, int ch, int h, int w)
    {
        int hw = h * w;
        float scale = 1f / MathF.Sqrt(ch);
        var result = (float[])x.Clone();

        for (int b = 0; b < n; b++)
        {
            var tokens = new float[hw * ch];
            for (int pos = 0; pos < hw; pos++)
            for (int c2 = 0; c2 < ch; c2++)
                tokens[pos * ch + c2] = normed[b * ch * hw + c2 * hw + pos];

            var q = LinearHW(tokens, wQ, hw, ch);
            var k = LinearHW(tokens, wK, hw, ch);
            var v = LinearHW(tokens, wV, hw, ch);

            var attnOut = new float[hw * ch];
            Parallel.For(0, hw, i =>
            {
                var qi = q.AsSpan(i * ch, ch);
                var localScores = new float[hw];
                for (int j = 0; j < hw; j++)
                    localScores[j] = TensorPrimitives.Dot(qi, k.AsSpan(j * ch, ch)) * scale;

                float maxS = TensorPrimitives.Max(localScores.AsSpan());
                TensorPrimitives.Subtract(localScores.AsSpan(), maxS, localScores.AsSpan());
                TensorPrimitives.Exp(localScores.AsSpan(), localScores.AsSpan());
                TensorPrimitives.Divide(localScores.AsSpan(),
                    TensorPrimitives.Sum(localScores.AsSpan()), localScores.AsSpan());

                var outRow = attnOut.AsSpan(i * ch, ch);
                outRow.Clear();
                for (int j = 0; j < hw; j++)
                    TensorPrimitives.MultiplyAdd<float>(v.AsSpan(j * ch, ch), localScores[j], outRow, outRow);
            });

            var proj = LinearHW(attnOut, wOut, hw, ch);
            for (int pos = 0; pos < hw; pos++)
            for (int c2 = 0; c2 < ch; c2++)
            {
                int nchwOff = b * ch * hw + c2 * hw + pos;
                result[nchwOff] += proj[pos * ch + c2] + (bOut is not null ? bOut[c2] : 0f);
            }
        }

        return result;
    }

    private static float[] LinearHW(float[] tokens, float[] w, int hw, int ch)
    {
        var result = new float[hw * ch];
        Parallel.For(0, hw, pos =>
        {
            var rowIn  = tokens.AsSpan(pos * ch, ch);
            var rowOut = result.AsSpan(pos * ch, ch);
            for (int oc = 0; oc < ch; oc++)
                rowOut[oc] = TensorPrimitives.Dot(rowIn, w.AsSpan(oc * ch, ch));
        });
        return result;
    }

    private float[] ConvBlock(string name, float[] x, int n, int inCh, int h, int w, int outCh, int k, int padding = -1)
    {
        if (_backend is not null)
            return ConvGpu(name, x, inCh, h, w, outCh, k, padding);
        var weight = Wt($"{name}.weight");
        var bias   = _st.Contains($"{name}.bias") ? Wt($"{name}.bias") : null;
        return DiffusionOps.Conv2D(x, weight, bias, n, inCh, h, w, outCh, k, k, stride: 1, padding: padding);
    }

    private float[] ConvGpu(string name, float[] x, int inCh, int h, int w, int outCh, int ksize, int padding)
    {
        if (padding < 0) padding = (ksize - 1) / 2;
        int outH = h, outW = w;
        int hw   = outH * outW;
        int kPts = inCh * ksize * ksize;

        string wKey = $"{name}.weight";
        if (!_gpuWeights!.TryGetValue(wKey, out var wGpu))
        {
            var wf = Wt(wKey);
            wGpu = _backend!.Upload(wf.AsSpan(), TensorShape.D1(wf.Length));
            _gpuWeights[wKey] = wGpu;
        }

        var biasArr = _st.Contains($"{name}.bias") ? Wt($"{name}.bias") : null;

        int chunkRows = kPts > 0 ? Math.Max(1, Math.Min(outH, MaxColChunkFloats / (outW * kPts))) : outH;
        var output = new float[outCh * hw];

        var colBuf    = ArrayPool<float>.Shared.Rent(chunkRows * outW * kPts);
        var resultBuf = ArrayPool<float>.Shared.Rent(chunkRows * outW * outCh);
        try
        {
            for (int rowStart = 0; rowStart < outH; rowStart += chunkRows)
            {
                int rowEnd  = Math.Min(rowStart + chunkRows, outH);
                int chunkH  = rowEnd - rowStart;
                int chunkHW = chunkH * outW;
                int colSize = chunkHW * kPts;

                Im2ColChunk(x, inCh, h, w, ksize, padding, rowStart, rowEnd, colBuf);

                var colGpu = _backend!.Upload(colBuf.AsSpan(0, colSize), TensorShape.D1(colSize));
                var cGpu   = _backend.Allocate(TensorShape.D1(chunkHW * outCh));
                try
                {
                    _backend.Sgemm(cGpu, colGpu, wGpu, chunkHW, kPts, outCh);
                    _backend.Synchronize();
                    _backend.Download(cGpu, resultBuf.AsSpan(0, chunkHW * outCh));
                }
                finally
                {
                    _backend.Free(colGpu);
                    _backend.Free(cGpu);
                }

                int basePos = rowStart * outW;
                for (int pos = 0; pos < chunkHW; pos++)
                {
                    int absPos = basePos + pos;
                    for (int oc = 0; oc < outCh; oc++)
                        output[oc * hw + absPos] = resultBuf[pos * outCh + oc] + (biasArr?[oc] ?? 0f);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(colBuf);
            ArrayPool<float>.Shared.Return(resultBuf);
        }
        return output;
    }

    private static void Im2ColChunk(float[] x, int inCh, int h, int w,
                                    int ksize, int padding,
                                    int rowStart, int rowEnd, float[] col)
    {
        int outW = w;
        int idx  = 0;
        for (int oh = rowStart; oh < rowEnd; oh++)
        for (int ow = 0; ow < outW; ow++)
        {
            for (int ic = 0; ic < inCh; ic++)
            for (int kh = 0; kh < ksize; kh++)
            for (int kw = 0; kw < ksize; kw++)
            {
                int ih = oh + kh - padding;
                int iw = ow + kw - padding;
                col[idx++] = ((uint)ih < (uint)h && (uint)iw < (uint)w)
                    ? x[ic * h * w + ih * w + iw]
                    : 0f;
            }
        }
    }

    private float[] Wt(string name)
    {
        if (_cpuWeights.TryGetValue(name, out var c)) return c;
        var w = _st.ReadF32(name);
        _cpuWeights[name] = w;
        return w;
    }

    public void Dispose()
    {
        if (_gpuWeights is not null)
        {
            foreach (var t in _gpuWeights.Values) _backend!.Free(t);
            _gpuWeights.Clear();
        }
        _cpuWeights.Clear();
        _st.Dispose();
    }
}
