using System.Numerics.Tensors;

namespace OpenTail.Stingray.Diffusion.HunyuanVideo;

/// <summary>
/// Causal 3D VAE Decoder for HunyuanVideo (base T2V/720p checkpoint).
///
/// Architecture ported against the real `diffusers` reference
/// (`examples/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_hunyuan_video.py`:
/// `HunyuanVideoDecoder3D`/`HunyuanVideoResnetBlockCausal3D`/`HunyuanVideoMidBlock3D`/
/// `HunyuanVideoUpBlock3D`/`HunyuanVideoUpsampleCausal3D`) -- NOT `examples/stable-diffusion.cpp/
/// src/model/vae/hunyuan_vae.hpp`, which ports a different, newer Hunyuan VAE variant (RMSNorm +
/// pixel-shuffle upsample, the same shape as Wan's VAE) that does not match this codebase's real
/// checkpoints at all.
///
/// <para><b>Weight KEY NAMES are the original CompVis/Tencent naming, not diffusers' renamed
/// ones</b> -- confirmed by direct inspection of the real downloaded
/// `models/hunyuanvideo/hunyuan_video_vae_bf16.safetensors` (Comfy-Org/HunyuanVideo_repackaged):
/// `decoder.mid.block_1`/`decoder.mid.attn_1` (separate `.q`/`.k`/`.v`/`.norm`/`.proj_out`, not
/// diffusers' `to_q`/`to_k`/`to_v`/`group_norm`/`to_out.0`)/`decoder.mid.block_2`, then
/// `decoder.up.{level}.block.{i}` (`.nin_shortcut.conv`, not `conv_shortcut.conv`) with
/// `decoder.up.{level}.upsample.conv.conv` where present, `decoder.norm_out` (not
/// `conv_norm_out`), `decoder.conv_out.conv`. Real per-level shapes confirmed directly against
/// this checkpoint: `up.3`/`up.2` are 512-&gt;512 (no shortcut), `up.1` is 512-&gt;256 (has
/// `nin_shortcut`), `up.0` is 256-&gt;128 (has `nin_shortcut`, no `upsample` key at all) --
/// **levels are stored and must be processed in REVERSE index order (3,2,1,0)**, matching the
/// classic CompVis `for i_level in reversed(range(num_resolutions))` decoder loop (confirmed
/// against which levels actually carry an `upsample` key -- 1,2,3, not 0 -- rather than assumed).
/// `scaling_factor=0.476986`, `block_out_channels=(128,256,512,512)`, `layers_per_block=2`,
/// `spatial_compression_ratio=8`, `temporal_compression_ratio=4` (real diffusers config; channel
/// counts independently confirmed against this checkpoint's own tensor shapes).</para>
///
/// <para><b>Causal padding is REPLICATE, not zero</b> (real `HunyuanVideoCausalConv3d`'s
/// `pad_mode="replicate"` default): the temporal left-pad (`kt-1` frames) replicates frame 0, and
/// spatial padding replicates the nearest edge pixel on all four sides -- implemented below via
/// index clamping rather than a materialized padded buffer, which is exactly equivalent for
/// straightforward direct convolution and avoids the WAN VAE port's zero-padding assumption
/// (real for Wan's own VAE, but NOT real here).</para>
///
/// <para><b>Known scope limit</b>: like `WanVaeDecoder3D`, this processes the whole latent volume
/// as one call (no chunked/cached cross-frame streaming) -- correct math for any `t`, just not the
/// real reference's memory-bounded frame-chunking strategy for very long videos.</para>
/// </summary>
public sealed class HunyuanVaeDecoder3D : IDisposable
{
    private readonly IWeightLoader? _weights;
    private readonly Dictionary<string, float[]> _weightCache = new(StringComparer.Ordinal);
    private bool _disposed;

    public const int LatentChannels = 16;
    public const int SpatialScale = 8;
    public const int TemporalScale = 4;
    public const float ScalingFactor = 0.476986f;

    private const int NormGroups = 32;
    private const int LayersPerBlock = 2;

    // Real per-level (processing order 3,2,1,0) output channels, confirmed directly against the
    // checkpoint's own conv1/conv2 shapes: level3=512, level2=512, level1=256, level0=128.
    private static readonly int[] LevelOutChannels = [512, 512, 256, 128];

    // Per real diffusers Decoder __init__ logic (spatial_compression_ratio=8 -> 3 spatial-upsample
    // stages, temporal_compression_ratio=4 -> 2 temporal-upsample stages), remapped to REAL
    // checkpoint processing order (level 3,2,1,0 == diffusers stage 0,1,2,3): level3=spatial-only,
    // level2/1=spatial+temporal, level0=no upsample at all (confirmed: level0 has no upsample key).
    private static readonly bool[] AddSpatialUpsample = [true, true, true, false];
    private static readonly bool[] AddTemporalUpsample = [false, true, true, false];

    public HunyuanVaeDecoder3D(IWeightLoader? weights = null)
    {
        _weights = weights;
    }

    /// <summary>
    /// Decodes a HunyuanVideo latent [16, T, latH, latW] into RGB video frames, one per output
    /// temporal position (post temporal-upsample: `t_out = 1 + (t-1)*TemporalScale` in the real
    /// multi-frame case, `t_out = t` for the single-frame image case this pipeline currently
    /// exercises end-to-end).
    /// </summary>
    public List<float[]> Decode(float[] latent, int t, int latH, int latW)
    {
        const int c = LatentChannels;
        if (latent.Length != c * t * latH * latW)
            throw new ArgumentException($"Latent size mismatch: expected {c * t * latH * latW}, got {latent.Length}");

        // Real: latents = latents / scaling_factor (no per-channel mean/std here, unlike Wan).
        var z = new float[latent.Length];
        float invScale = 1.0f / ScalingFactor;
        for (int i = 0; i < z.Length; i++) z[i] = latent[i] * invScale;

        int curC = LevelOutChannels[0]; // 512, matches decoder.conv_in.conv output channels.
        var x = CausalConv3D(z, "decoder.conv_in.conv", c, curC, t, latH, latW);
        int curT = t, curH = latH, curW = latW;

        // Mid block: resnet(block_1) -> attention(attn_1) -> resnet(block_2), all at 512 channels.
        x = ResnetBlock(x, "decoder.mid.block_1", curC, curC, curT, curH, curW);
        x = SelfAttention3D(x, "decoder.mid.attn_1", curC, curT, curH, curW);
        x = ResnetBlock(x, "decoder.mid.block_2", curC, curC, curT, curH, curW);

        // Up levels, REAL processing order 3,2,1,0 (see class doc comment).
        int prevOut = curC;
        for (int idx = 0; idx < LevelOutChannels.Length; idx++)
        {
            int level = 3 - idx;
            int outCh = LevelOutChannels[idx];
            string levelPrefix = $"decoder.up.{level}";

            for (int j = 0; j < LayersPerBlock + 1; j++)
            {
                int inCh = j == 0 ? prevOut : outCh;
                x = ResnetBlock(x, $"{levelPrefix}.block.{j}", inCh, outCh, curT, curH, curW);
            }
            curC = outCh;

            bool addSpatial = AddSpatialUpsample[idx];
            bool addTemporal = AddTemporalUpsample[idx];
            if (addSpatial || addTemporal)
            {
                (x, curT, curH, curW) = UpsampleCausal3D(x, $"{levelPrefix}.upsample.conv.conv", curC, curT, curH, curW, addSpatial, addTemporal);
            }

            prevOut = outCh;
        }

        // Post-process: GroupNorm -> SiLU -> conv_out (curC = 128).
        GroupNorm3D(x, "decoder.norm_out", curC, curT, curH, curW);
        DiffusionOps.SiluInPlace(x);
        x = CausalConv3D(x, "decoder.conv_out.conv", curC, 3, curT, curH, curW);

        int spatialOut = curH * curW;
        var frames = new List<float[]>(curT);
        for (int fi = 0; fi < curT; fi++)
        {
            var frame = new float[3 * spatialOut];
            for (int ch = 0; ch < 3; ch++)
            {
                int srcOff = (ch * curT + fi) * spatialOut;
                for (int s = 0; s < spatialOut; s++)
                    frame[ch * spatialOut + s] = Math.Clamp((x[srcOff + s] + 1.0f) * 0.5f, 0.0f, 1.0f);
            }
            frames.Add(frame);
        }
        return frames;
    }

    // ── Building blocks ─────────────────────────────────────────────────────

    /// <summary>Real resnet block (CompVis-style, real key suffixes `.norm1`/`.conv1.conv`/
    /// `.norm2`/`.conv2.conv`/`.nin_shortcut.conv`): GroupNorm->SiLU->conv1(3x3x3) ->
    /// GroupNorm->SiLU->conv2(3x3x3), plus a 1x1x1 `nin_shortcut` conv when channels change.
    /// </summary>
    private float[] ResnetBlock(float[] x, string prefix, int inCh, int outCh, int t, int h, int w)
    {
        var residual = inCh == outCh ? x : CausalConv3D(x, $"{prefix}.nin_shortcut.conv", inCh, outCh, t, h, w, kt: 1, kh: 1, kw: 1);

        var hState = (float[])x.Clone();
        GroupNorm3D(hState, $"{prefix}.norm1", inCh, t, h, w);
        DiffusionOps.SiluInPlace(hState);
        hState = CausalConv3D(hState, $"{prefix}.conv1.conv", inCh, outCh, t, h, w);

        GroupNorm3D(hState, $"{prefix}.norm2", outCh, t, h, w);
        DiffusionOps.SiluInPlace(hState);
        hState = CausalConv3D(hState, $"{prefix}.conv2.conv", outCh, outCh, t, h, w);

        TensorPrimitives.Add(hState, residual, hState);
        return hState;
    }

    /// <summary>Real CompVis-style `AttnBlock` (real key suffixes `.norm`/`.q`/`.k`/`.v`/
    /// `.proj_out`, each a 1x1x1 conv): single-head full spatio-temporal (per
    /// frame*height*width flattened) self-attention with a causal frame mask (a later frame may
    /// attend to an earlier one, never the reverse, matching the real `HunyuanVideoMidBlock3D`
    /// causal attention mask) -- GroupNorm -> q/k/v -> attention -> proj_out, residual add.
    /// </summary>
    private float[] SelfAttention3D(float[] x, string prefix, int ch, int t, int h, int w)
    {
        var identity = x;
        var normX = (float[])x.Clone();
        GroupNorm3D(normX, $"{prefix}.norm", ch, t, h, w);

        int spatial = h * w;
        int seq = t * spatial;

        var q = Linear1x1(normX, $"{prefix}.q", ch, ch, t, spatial);
        var k = Linear1x1(normX, $"{prefix}.k", ch, ch, t, spatial);
        var v = Linear1x1(normX, $"{prefix}.v", ch, ch, t, spatial);

        // Reorder [ch, t*spatial] (channel-major) -> [t*spatial, ch] (token-major) for attention.
        var qT = ChannelMajorToTokenMajor(q, ch, seq);
        var kT = ChannelMajorToTokenMajor(k, ch, seq);
        var vT = ChannelMajorToTokenMajor(v, ch, seq);

        var attnOut = new float[seq * ch];
        float scale = 1f / MathF.Sqrt(ch);
        Parallel.For(0, seq, si =>
        {
            int frameI = si / spatial;
            var scores = new float[seq];
            float max = float.NegativeInfinity;
            for (int sj = 0; sj < seq; sj++)
            {
                int frameJ = sj / spatial;
                if (frameJ > frameI) { scores[sj] = float.NegativeInfinity; continue; }
                float dot = 0f;
                int qOff = si * ch, kOff = sj * ch;
                for (int d = 0; d < ch; d++) dot += qT[qOff + d] * kT[kOff + d];
                dot *= scale;
                scores[sj] = dot;
                if (dot > max) max = dot;
            }
            float sum = 0f;
            for (int sj = 0; sj < seq; sj++)
            {
                if (scores[sj] == float.NegativeInfinity) { scores[sj] = 0f; continue; }
                scores[sj] = MathF.Exp(scores[sj] - max);
                sum += scores[sj];
            }
            float invSum = sum > 0f ? 1f / sum : 0f;
            int outOff = si * ch;
            for (int d = 0; d < ch; d++)
            {
                float acc = 0f;
                for (int sj = 0; sj < seq; sj++)
                    acc += scores[sj] * vT[sj * ch + d];
                attnOut[outOff + d] = acc * invSum;
            }
        });

        // Project back [seq, ch] -> [ch, seq] channel-major, then proj_out (1x1 linear).
        var attnChanMajor = TokenMajorToChannelMajor(attnOut, ch, seq);
        var projected = Linear1x1(attnChanMajor, $"{prefix}.proj_out", ch, ch, t, spatial);

        TensorPrimitives.Add(projected, identity, projected);
        return projected;
    }

    private static float[] ChannelMajorToTokenMajor(float[] x, int ch, int seq)
    {
        var res = new float[seq * ch];
        for (int s = 0; s < seq; s++)
            for (int d = 0; d < ch; d++)
                res[s * ch + d] = x[d * seq + s];
        return res;
    }

    private static float[] TokenMajorToChannelMajor(float[] x, int ch, int seq)
    {
        var res = new float[ch * seq];
        for (int s = 0; s < seq; s++)
            for (int d = 0; d < ch; d++)
                res[d * seq + s] = x[s * ch + d];
        return res;
    }

    /// <summary>1x1 pointwise "conv" == per-token linear, applied over the channel-major
    /// [ch, t*spatial] layout used throughout this decoder.</summary>
    private float[] Linear1x1(float[] x, string prefix, int inCh, int outCh, int t, int spatial)
    {
        var weight = GetWeight($"{prefix}.weight", outCh * inCh);
        var bias = GetWeight($"{prefix}.bias", outCh);
        int seq = t * spatial;
        var output = new float[outCh * seq];

        Parallel.For(0, outCh, oc =>
        {
            float b = bias[oc];
            int wOff = oc * inCh;
            var outSpan = output.AsSpan(oc * seq, seq);
            outSpan.Fill(b);
            for (int ic = 0; ic < inCh; ic++)
            {
                float wVal = weight[wOff + ic];
                if (wVal == 0f) continue;
                var inSpan = x.AsSpan(ic * seq, seq);
                TensorPrimitives.MultiplyAdd(inSpan, wVal, outSpan, outSpan);
            }
        });
        return output;
    }

    /// <summary>Real `HunyuanVideoUpsampleCausal3D`: nearest-neighbor upsample (frame 0 gets only
    /// the spatial factor -- never a temporal factor, matching the real `first_frame`/`other_frames`
    /// split -- remaining frames get the full temporal+spatial factor), then a causal conv.</summary>
    private (float[] output, int outT, int outH, int outW) UpsampleCausal3D(
        float[] x, string convPrefix, int ch, int t, int h, int w, bool spatial, bool temporal)
    {
        int factorS = spatial ? 2 : 1;
        int factorT = temporal ? 2 : 1;
        int outH = h * factorS, outW = w * factorS;
        int outT = temporal ? 1 + (t - 1) * factorT : t;
        int spatialOut = outH * outW;

        var upsampled = new float[ch * outT * spatialOut];

        // Frame 0: spatial-only nearest upsample.
        Parallel.For(0, ch, c =>
        {
            int inFrameOff = (c * t) * (h * w);
            int outFrameOff = (c * outT) * spatialOut;
            for (int oh = 0; oh < outH; oh++)
            {
                int inH = oh / factorS;
                for (int ow = 0; ow < outW; ow++)
                {
                    int inW = ow / factorS;
                    upsampled[outFrameOff + oh * outW + ow] = x[inFrameOff + inH * w + inW];
                }
            }
        });

        // Remaining frames: full temporal+spatial nearest upsample.
        if (t > 1)
        {
            Parallel.For(0, ch, c =>
            {
                for (int inT = 1; inT < t; inT++)
                {
                    int inFrameOff = (c * t + inT) * (h * w);
                    for (int dt = 0; dt < factorT; dt++)
                    {
                        int outTIdx = 1 + (inT - 1) * factorT + dt;
                        int outFrameOff = (c * outT + outTIdx) * spatialOut;
                        for (int oh = 0; oh < outH; oh++)
                        {
                            int inH = oh / factorS;
                            for (int ow = 0; ow < outW; ow++)
                            {
                                int inW = ow / factorS;
                                upsampled[outFrameOff + oh * outW + ow] = x[inFrameOff + inH * w + inW];
                            }
                        }
                    }
                }
            });
        }

        var conv = CausalConv3D(upsampled, convPrefix, ch, ch, outT, outH, outW);
        return (conv, outT, outH, outW);
    }

    /// <summary>3D causal convolution with REPLICATE padding (temporal: left-pad only, replicates
    /// frame 0; spatial: symmetric replicate on both sides) -- see class doc comment. Implemented
    /// via index clamping, exactly equivalent to materializing a replicate-padded input.</summary>
    private float[] CausalConv3D(float[] x, string weightPrefix, int inCh, int outCh, int t, int h, int w,
        int kt = 3, int kh = 3, int kw = 3)
    {
        var weight = GetWeight($"{weightPrefix}.weight", outCh * inCh * kt * kh * kw);
        var bias = GetWeight($"{weightPrefix}.bias", outCh);

        var output = new float[outCh * t * h * w];
        int padT = kt - 1;
        int padH = kh / 2;
        int padW = kw / 2;
        int spatial = h * w;

        if (kt == 1 && kh == 1 && kw == 1)
        {
            Parallel.For(0, outCh, oc =>
            {
                float b = bias[oc];
                int wBase = oc * inCh;
                for (int ti = 0; ti < t; ti++)
                {
                    int outOffset = (oc * t + ti) * spatial;
                    var outSpan = output.AsSpan(outOffset, spatial);
                    outSpan.Fill(b);
                    for (int ic = 0; ic < inCh; ic++)
                    {
                        float wVal = weight[wBase + ic];
                        if (wVal == 0f) continue;
                        int inOffset = (ic * t + ti) * spatial;
                        var inSpan = x.AsSpan(inOffset, spatial);
                        TensorPrimitives.MultiplyAdd(inSpan, wVal, outSpan, outSpan);
                    }
                }
            });
            return output;
        }

        Parallel.For(0, outCh, oc =>
        {
            float b = bias[oc];
            for (int outT = 0; outT < t; outT++)
            {
                int outOffset = (oc * t + outT) * spatial;
                var outSpan = output.AsSpan(outOffset, spatial);
                outSpan.Fill(b);

                for (int ic = 0; ic < inCh; ic++)
                for (int dt = 0; dt < kt; dt++)
                {
                    // Causal temporal replicate-pad: left only, clamp negative indices to frame 0.
                    int inT = outT - padT + dt;
                    if (inT < 0) inT = 0;
                    // Right side never exceeds range for a causal (left-only) pad by construction.

                    int inFrameOff = (ic * t + inT) * spatial;
                    int weightSliceOff = (((oc * inCh + ic) * kt + dt) * kh) * kw;

                    for (int dh = 0; dh < kh; dh++)
                    for (int dw = 0; dw < kw; dw++)
                    {
                        float wVal = weight[weightSliceOff + dh * kw + dw];
                        if (wVal == 0f) continue;

                        int hShift = dh - padH;
                        int wShift = dw - padW;

                        for (int oh = 0; oh < h; oh++)
                        {
                            int inH = oh + hShift;
                            if (inH < 0) inH = 0; else if (inH >= h) inH = h - 1;
                            int inRowOff = inFrameOff + inH * w;
                            int outRowOff = oh * w;

                            for (int ow = 0; ow < w; ow++)
                            {
                                int inW = ow + wShift;
                                if (inW < 0) inW = 0; else if (inW >= w) inW = w - 1;
                                outSpan[outRowOff + ow] += x[inRowOff + inW] * wVal;
                            }
                        }
                    }
                }
            }
        });

        return output;
    }

    /// <summary>GroupNorm(32 groups) over [C, T, H, W] treating T*H*W as combined spatial extent
    /// (matches real `nn.GroupNorm` applied to a [N,C,T,H,W] tensor -- PyTorch GroupNorm normalizes
    /// over all non-channel/non-batch dims together, so T*H*W flattened is exactly equivalent to
    /// passing H'=T*H, W'=W to the existing 2D `DiffusionOps.GroupNorm` helper).</summary>
    private void GroupNorm3D(float[] x, string prefix, int c, int t, int h, int w)
    {
        var weight = GetWeight($"{prefix}.weight", c);
        var bias = GetWeight($"{prefix}.bias", c);
        DiffusionOps.GroupNorm(x.AsSpan(), weight, bias, n: 1, c: c, h: t * h, w: w, groups: NormGroups, eps: 1e-6f);
    }

    private float[] GetWeight(string name, int expectedLength)
    {
        string resolved = Resolve(name);
        if (_weightCache.TryGetValue(resolved, out var cached)) return cached;

        if (_weights is null || !_weights.Contains(resolved))
            throw new InvalidOperationException($"HunyuanVideo VAE tensor not found: '{name}' (expected {expectedLength} elements).");

        var data = _weights.ReadF32(resolved);
        if (data.Length != expectedLength)
            throw new InvalidOperationException($"HunyuanVideo VAE tensor '{resolved}' has {data.Length} elements, expected {expectedLength}.");

        _weightCache[resolved] = data;
        return data;
    }

    private string Resolve(string name)
    {
        if (_weights is null) return name;
        if (_weights.Contains(name)) return name;
        if (_weights.Contains("vae." + name)) return "vae." + name;
        if (_weights.Contains("first_stage_model." + name)) return "first_stage_model." + name;
        return name;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _weightCache.Clear();
        }
    }
}
