namespace OpenTail.Stingray.Diffusion.LTXVideo;

/// <summary>
/// LTX-Video's real, timestep-conditioned causal-3D VAE decoder.
///
/// Ported directly from the OFFICIAL `ltx-video` PyPI package's
/// `ltx_video.models.autoencoders.causal_video_autoencoder.Decoder`/`UNetMidBlock3D`/
/// `ResnetBlock3D`/`DepthToSpaceUpsample`/`CausalConv3d` -- NOT HuggingFace `diffusers`' own
/// `AutoencoderKLLTXVideo`, which (unlike the DiT transformer, which matched diffusers with zero
/// missing/unexpected keys) does NOT match this checkpoint's real tensor structure: diffusers'
/// `LTXVideoDecoder3d` unconditionally builds a `mid_block` (none exists here), and combines each
/// stage's resnets+upsample into one class instance instead of the real, separate `res_x` /
/// `compress_all` blocks this checkpoint's own tensor names show (`up_blocks.{0,2,4,6}.res_blocks.*`
/// vs. `up_blocks.{1,3,5}.conv.*`) -- see docs/055-ltx-video-implementation-plan.md for the full
/// investigation. The real per-stage architecture below is read directly from THIS checkpoint's own
/// embedded `__metadata__["config"]["vae"]["decoder_blocks"]` JSON (confirmed, not inferred):
/// <c>[["res_x",{"num_layers":5,"inject_noise":true}], ["compress_all",{"residual":true,
/// "multiplier":2}], ["res_x",{"num_layers":6,"inject_noise":true}], ["compress_all",...],
/// ["res_x",{"num_layers":7,"inject_noise":true}], ["compress_all",...],
/// ["res_x",{"num_layers":8,"inject_noise":false}]]</c> -- the real `Decoder.__init__` iterates
/// this list REVERSED, which is why stage 0 below (nearest the latent) is the LARGEST resnet stack
/// (8 layers, 1024ch) and stage 6 (nearest the pixel output) is the smallest (5 layers, 128ch).
///
/// <para><b>Real per-block math</b> (`ResnetBlock3D.forward`, `norm_layer="pixel_norm"` for this
/// checkpoint, confirmed via the same metadata): `PixelNorm` (non-affine, per-pixel RMS over the
/// CHANNEL axis only, eps=1e-8) -&gt; optional 4-way timestep scale/shift (`scale_shift_table[4,C]
/// + sharedStageTimestepEmbed`, only shift1/scale1 applied here) -&gt; SiLU -&gt; conv1 (3x3x3,
/// symmetric edge-replicate temporal pad since `causal_decoder=false`, zero spatial pad) -&gt;
/// optional per-channel spatial noise injection -&gt; PixelNorm -&gt; optional shift2/scale2 -&gt;
/// SiLU -&gt; conv2 -&gt; optional noise -&gt; residual add (shortcut is Identity: every VAE resnet
/// here keeps in_channels==out_channels, confirmed by the checkpoint's own conv1/conv2 shapes all
/// being square `[C,C,3,3,3]`).</para>
///
/// <para><b>Decode-time timestep default</b> (`pipeline_ltx_video.py`'s real default, NOT the
/// "nominal 0.05" this project's own earlier planning-pass research guessed): `decode_timestep=0.0`,
/// `decode_noise_scale` defaults to the same value -- i.e. by default NO noise is mixed into the
/// latent before decode, but the timestep-conditioning path (with t=0) still runs for real, since
/// the decoder is unconditionally timestep-conditioned (`timestep_conditioning: true` in the real
/// config) with no code path to skip it.</para>
/// </summary>
public sealed class LtxVaeDecoder : IDisposable
{
    private readonly IWeightLoader _weights;
    private readonly string _prefix;
    private readonly Dictionary<string, float[]> _weightCache = new(StringComparer.Ordinal);
    private bool _disposed;

    public const int LatentChannels = 128;
    public const int OutChannels = 3;
    public const int PatchSize = 4;
    public const int SpatialScale = 32; // 4 (patch) * 2*2*2 (three compress_all stages)
    public const int TemporalScale = 8; // 2*2*2 (three compress_all stages)

    /// <summary>One decoder stage: either a resnet stack ("res_x") or a compress_all
    /// upsample-and-halve-channels stage. Real order (index 0 = nearest latent), read directly from
    /// this checkpoint's embedded config -- see class doc comment.</summary>
    private readonly record struct Stage(bool IsResnetStage, int Channels, int NumLayers, bool InjectNoise);

    private static readonly Stage[] Stages =
    {
        new(true, 1024, 8, false),
        new(false, 1024, 0, false), // compress_all: 1024 -> 512, 2x2x2 upsample
        new(true, 512, 7, true),
        new(false, 512, 0, false),  // -> 256
        new(true, 256, 6, true),
        new(false, 256, 0, false),  // -> 128
        new(true, 128, 5, true),
    };

    public LtxVaeDecoder(IWeightLoader weights, string prefix = "vae.decoder")
    {
        _weights = weights;
        _prefix = prefix.Length > 0 && !prefix.EndsWith('.') ? prefix + "." : prefix;
    }

    private float[] GetWeight(string name)
    {
        string full = _prefix + name;
        if (_weightCache.TryGetValue(full, out var cached)) return cached;
        var data = _weights.ReadF32(full);
        _weightCache[full] = data;
        return data;
    }

    /// <summary>
    /// Decodes a latent video tensor [128, F, H, W] into pixel-space RGB [3, F, H*32, W*32]
    /// (temporal frame count changes per the compress_all stages' first-frame-drop trimming -- see
    /// <see cref="UpsampleStage"/>).
    /// </summary>
    /// <param name="injectNoise">Real per-checkpoint behavior applies trained per-channel
    /// StyleGAN-style spatial noise (see <see cref="InjectSpatialNoise"/>) at 3 of the 7 stages --
    /// genuinely stochastic in the reference (fresh, unseeded `torch.randn` per call), so it cannot
    /// be numerically reproduced bit-for-bit against a golden dump. Set false only for golden-parity
    /// testing against a reference run that also had noise injection disabled; real inference should
    /// leave this true.</param>
    public float[] Decode(ReadOnlySpan<float> latents, float decodeTimestep, int f, int h, int w, bool injectNoise = true)
    {
        var x = CausalConv3D(latents, "conv_in", LatentChannels, Stages[0].Channels, f, h, w, causalTemporal: false);
        int curC = Stages[0].Channels;
        int curF = f, curH = h, curW = w;

        float timestepScaleMultiplier = GetWeight("timestep_scale_multiplier")[0];
        float scaledTimestep = decodeTimestep * timestepScaleMultiplier;

        foreach (var stage in Stages)
        {
            if (stage.IsResnetStage)
            {
                x = ResnetStage(x, stage, scaledTimestep, curF, curH, curW, injectNoise, out curC);
            }
            else
            {
                x = UpsampleStage(x, curC, scaledTimestep, ref curF, ref curH, ref curW, out curC);
            }
        }

        // conv_norm_out (PixelNorm, non-affine) -> last_time_embedder/last_scale_shift_table -> SiLU
        PixelNormInPlace(x, curC, curF * curH * curW);

        var lastEmbed = TimestepEmbedMlp("last_time_embedder.timestep_embedder", scaledTimestep, curC * 2);
        var lastTable = GetWeight("last_scale_shift_table"); // [2, C]
        var shift = new float[curC];
        var scale = new float[curC];
        for (int c = 0; c < curC; c++)
        {
            shift[c] = lastTable[c] + lastEmbed[c];
            scale[c] = lastTable[curC + c] + lastEmbed[curC + c];
        }
        ApplyChannelScaleShift(x, curC, curF * curH * curW, shift, scale);

        DiffusionOps.SiluInPlace(x);

        int patchedOutCh = OutChannels * PatchSize * PatchSize;
        x = CausalConv3D(x, "conv_out", curC, patchedOutCh, curF, curH, curW, causalTemporal: false);

        return Unpatchify(x, curF, curH, curW, patchedOutCh);
    }

    // ── Resnet stage (real "res_x" / UNetMidBlock3D) ────────────────────────────────────────

    private float[] ResnetStage(float[] x, Stage stage, float scaledTimestep, int f, int h, int w, bool injectNoise, out int outC)
    {
        outC = stage.Channels;
        int c = stage.Channels;
        int stageIndex = Array.IndexOf(Stages, stage);
        string stagePrefix = $"up_blocks.{stageIndex}";

        // Real: ONE shared time_embedder per stage (not per resnet); its output is combined with
        // each individual resnet's OWN learned scale_shift_table[4,C].
        var stageEmbed = TimestepEmbedMlp($"{stagePrefix}.time_embedder.timestep_embedder", scaledTimestep, c * 4);

        for (int layer = 0; layer < stage.NumLayers; layer++)
        {
            x = ResnetBlock(x, $"{stagePrefix}.res_blocks.{layer}", c, stage.InjectNoise && injectNoise, stageEmbed, f, h, w);
        }
        return x;
    }

    private float[] ResnetBlock(float[] x, string prefix, int c, bool injectNoise, float[] stageEmbed, int f, int h, int w)
    {
        int spatial = f * h * w;
        var table = GetWeight($"{prefix}.scale_shift_table"); // [4, C]: shift1,scale1,shift2,scale2
        var shift1 = new float[c]; var scale1 = new float[c];
        var shift2 = new float[c]; var scale2 = new float[c];
        for (int i = 0; i < c; i++)
        {
            shift1[i] = table[i] + stageEmbed[i];
            scale1[i] = table[c + i] + stageEmbed[c + i];
            shift2[i] = table[2 * c + i] + stageEmbed[2 * c + i];
            scale2[i] = table[3 * c + i] + stageEmbed[3 * c + i];
        }

        var hState = (float[])x.Clone();
        PixelNormInPlace(hState, c, spatial);
        ApplyChannelScaleShift(hState, c, spatial, shift1, scale1);
        DiffusionOps.SiluInPlace(hState);
        hState = CausalConv3D(hState, $"{prefix}.conv1", c, c, f, h, w, causalTemporal: false);

        if (injectNoise) InjectSpatialNoise(hState, $"{prefix}.per_channel_scale1", c, f, h, w);

        PixelNormInPlace(hState, c, spatial);
        ApplyChannelScaleShift(hState, c, spatial, shift2, scale2);
        DiffusionOps.SiluInPlace(hState);
        hState = CausalConv3D(hState, $"{prefix}.conv2", c, c, f, h, w, causalTemporal: false);

        if (injectNoise) InjectSpatialNoise(hState, $"{prefix}.per_channel_scale2", c, f, h, w);

        // Real shortcut is Identity here: every VAE resnet keeps in_channels==out_channels.
        TensorPrimitives.Add(hState, x, hState);
        return hState;
    }

    /// <summary>StyleGAN-style per-channel spatial noise injection (real `_feed_spatial_noise`):
    /// ONE random [H,W] noise map shared across all frames/channels, scaled per-channel by a
    /// learned constant, broadcast-added. Since this is genuinely stochastic in the reference
    /// (fresh `torch.randn` every call, no seed threading exposed), this port uses a
    /// zero-mean-preserving deterministic substitute (skips the add) when weights are all-zero
    /// (the real learned `per_channel_scale` initializes to `torch.zeros`, so a freshly-loaded
    /// checkpoint's noise contribution is a real, trained-away-from-zero value -- this is NOT
    /// optional in a trained checkpoint, but IS inherently non-reproducible noise, or a real
    /// project caller must decide how to seed it).</summary>
    private void InjectSpatialNoise(float[] x, string weightName, int c, int f, int h, int w)
    {
        var scale = GetWeight(weightName); // [C, 1, 1]
        var rng = Random.Shared;
        var noise = new float[h * w];
        // Box-Muller for approx-normal noise, matching torch.randn's distribution shape.
        for (int i = 0; i < noise.Length; i += 2)
        {
            double u1 = Math.Max(1e-9, rng.NextDouble());
            double u2 = rng.NextDouble();
            double mag = Math.Sqrt(-2.0 * Math.Log(u1));
            noise[i] = (float)(mag * Math.Cos(2 * Math.PI * u2));
            if (i + 1 < noise.Length) noise[i + 1] = (float)(mag * Math.Sin(2 * Math.PI * u2));
        }

        int spatial = h * w;
        for (int ch = 0; ch < c; ch++)
        {
            float s = scale[ch];
            for (int fr = 0; fr < f; fr++)
            {
                int off = (ch * f + fr) * spatial;
                for (int p = 0; p < spatial; p++) x[off + p] += noise[p] * s;
            }
        }
    }

    // ── Upsample stage (real "compress_all" / DepthToSpaceUpsample, residual=true, stride=(2,2,2),
    // out_channels_reduction_factor=2) ──────────────────────────────────────────────────────────

    private float[] UpsampleStage(float[] x, int inC, float scaledTimestep, ref int f, ref int h, ref int w, out int outC)
    {
        // Real conv: in_channels -> in_channels*8/2 (=in_channels*4), matching this checkpoint's
        // own conv weight shapes (e.g. [4096,1024,...], [2048,512,...], [1024,256,...]).
        int convOutC = inC * 4;
        string prefix = $"up_blocks.{FindUpBlockIndex(inC)}";

        var conv = CausalConv3D(x, $"{prefix}.conv", inC, convOutC, f, h, w, causalTemporal: false);

        int newF = f * 2, newH = h * 2, newW = w * 2;
        int shuffledC = convOutC / 8; // = inC / 2
        var shuffled = PixelShuffle3D(conv, convOutC, f, h, w, shuffledC, newF, newH, newW);

        // Residual path: pixel-shuffle the ORIGINAL (pre-conv) input directly, then repeat
        // channels by (8/reductionFactor=4) to match the conv path's channel count.
        int residualShuffledC = inC / 8;
        var residualShuffled = PixelShuffle3D(x, inC, f, h, w, residualShuffledC, newF, newH, newW);
        var residual = new float[shuffledC * newF * newH * newW];
        int repeats = 8 / 2; // prod(stride)/reduction_factor = 4
        int spatialNew = newH * newW;
        for (int rep = 0; rep < repeats; rep++)
        {
            int dstChBase = rep * residualShuffledC;
            for (int c = 0; c < residualShuffledC; c++)
            {
                Array.Copy(residualShuffled, c * newF * spatialNew, residual, (dstChBase + c) * newF * spatialNew, newF * spatialNew);
            }
        }

        // Both paths drop the first upsampled frame (real `x[:, :, 1:, :, :]`, stride[0]==2).
        int trimmedF = newF - 1;
        var outArr = new float[shuffledC * trimmedF * spatialNew];
        for (int c = 0; c < shuffledC; c++)
        {
            for (int fr = 0; fr < trimmedF; fr++)
            {
                int srcOffMain = (c * newF + (fr + 1)) * spatialNew;
                int srcOffRes = (c * newF + (fr + 1)) * spatialNew;
                int dstOff = (c * trimmedF + fr) * spatialNew;
                for (int p = 0; p < spatialNew; p++)
                    outArr[dstOff + p] = shuffled[srcOffMain + p] + residual[srcOffRes + p];
            }
        }

        f = trimmedF; h = newH; w = newW;
        outC = shuffledC;
        return outArr;
    }

    private static int FindUpBlockIndex(int inC) => inC switch
    {
        1024 => 1,
        512 => 3,
        256 => 5,
        _ => throw new ArgumentOutOfRangeException(nameof(inC)),
    };

    /// <summary>Real `PixelShuffleND(dims=3, upscale_factors=(2,2,2))`:
    /// `"b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)"` -- channel axis splits c-major with
    /// (p1,p2,p3) minor (p3 fastest), each spatial/temporal axis gets its own upscale index appended
    /// as the low-order bit of the expanded coordinate.</summary>
    private static float[] PixelShuffle3D(float[] src, int srcC, int f, int h, int w, int dstC, int dstF, int dstH, int dstW)
    {
        var dst = new float[dstC * dstF * dstH * dstW];
        int spatialSrc = h * w;
        int spatialDst = dstH * dstW;
        for (int c = 0; c < dstC; c++)
        {
            for (int p1 = 0; p1 < 2; p1++)
            for (int p2 = 0; p2 < 2; p2++)
            for (int p3 = 0; p3 < 2; p3++)
            {
                int srcCh = ((c * 2 + p1) * 2 + p2) * 2 + p3;
                if (srcCh >= srcC) continue;
                for (int fr = 0; fr < f; fr++)
                {
                    int dstFr = fr * 2 + p1;
                    for (int hh = 0; hh < h; hh++)
                    {
                        int dstHh = hh * 2 + p2;
                        int srcRowOff = (srcCh * f + fr) * spatialSrc + hh * w;
                        int dstRowBase = (c * dstF + dstFr) * spatialDst + dstHh * dstW;
                        for (int ww = 0; ww < w; ww++)
                        {
                            int dstWw = ww * 2 + p3;
                            dst[dstRowBase + dstWw] = src[srcRowOff + ww];
                        }
                    }
                }
            }
        }
        return dst;
    }

    // ── Shared primitives ────────────────────────────────────────────────────────────────────

    /// <summary>Real `PixelNorm(dim=1)`: `x / sqrt(mean(x^2, dim=channel) + eps)`, eps=1e-8, no
    /// learned affine -- normalizes each (frame,row,col) location's channel VECTOR to unit RMS,
    /// the exact opposite axis of a per-token RMSNorm over the channel-last convention this
    /// project's transformer code uses elsewhere (this tensor is channel-FIRST, [C,F,H,W]).</summary>
    private static void PixelNormInPlace(float[] x, int c, int spatial)
    {
        Parallel.For(0, spatial, p =>
        {
            float sumSq = 0f;
            for (int ch = 0; ch < c; ch++)
            {
                float v = x[ch * spatial + p];
                sumSq += v * v;
            }
            float invRms = 1f / MathF.Sqrt(sumSq / c + 1e-8f);
            for (int ch = 0; ch < c; ch++) x[ch * spatial + p] *= invRms;
        });
    }

    private static void ApplyChannelScaleShift(float[] x, int c, int spatial, float[] shift, float[] scale)
    {
        Parallel.For(0, c, ch =>
        {
            float sc = 1f + scale[ch];
            float sh = shift[ch];
            int off = ch * spatial;
            var span = x.AsSpan(off, spatial);
            TensorPrimitives.Multiply(span, sc, span);
            TensorPrimitives.Add(span, sh, span);
        });
    }

    /// <summary>Real `PixArtAlphaCombinedTimestepSizeEmbeddings(dim, 0)` with
    /// `use_additional_conditions=False` (the default, not overridden by this package): sinusoidal
    /// `Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)` -> `Linear(256,dim)` -> SiLU ->
    /// `Linear(dim,dim)`. NO extra outer SiLU/Linear afterward here (unlike the DiT transformer's
    /// `AdaLayerNormSingle`, which adds its own `.linear` 6x-projection on top) -- this value is used
    /// directly as the per-stage/final shared timestep embedding.</summary>
    private float[] TimestepEmbedMlp(string prefix, float scaledTimestep, int dim)
    {
        const int freqEmbedSize = 256;
        var emb = new float[freqEmbedSize];
        int half = freqEmbedSize / 2;
        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Exp(-MathF.Log(10000.0f) * i / half);
            float angle = scaledTimestep * freq;
            // flip_sin_to_cos=True: [cos, sin] order (real `get_timestep_embedding`).
            emb[i] = MathF.Cos(angle);
            emb[half + i] = MathF.Sin(angle);
        }

        var h1 = Linear($"{prefix}.linear_1", emb, freqEmbedSize, dim);
        DiffusionOps.SiluInPlace(h1);
        return Linear($"{prefix}.linear_2", h1, dim, dim);
    }

    private float[] Linear(string name, float[] x, int inDim, int outDim)
    {
        var w = GetWeight($"{name}.weight");
        var b = GetWeight($"{name}.bias");
        var outF = new float[outDim];
        Parallel.For(0, outDim, o =>
        {
            float sum = b[o] + TensorPrimitives.Dot(x.AsSpan(0, inDim), w.AsSpan(o * inDim, inDim));
            outF[o] = sum;
        });
        return outF;
    }

    /// <summary>Real `CausalConv3d.forward`: kernel=3, `causal=false` (this decoder's real
    /// `causal_decoder: false` config) -- SYMMETRIC edge-replicate temporal padding (1 frame
    /// repeated on each side), zero spatial padding (1px each side). Distinct from
    /// <c>Wan/WanVaeDecoder3D.CausalConv3D</c>'s zero-padded, one-sided-only convention -- LTX's
    /// decoder is explicitly non-causal.</summary>
    private float[] CausalConv3D(ReadOnlySpan<float> x, string name, int inCh, int outCh, int f, int h, int w, bool causalTemporal)
    {
        var weight = GetWeight($"{name}.conv.weight");
        var bias = GetWeight($"{name}.conv.bias");
        const int k = 3;
        int padT = causalTemporal ? k - 1 : (k - 1) / 2;
        int padH = k / 2, padW = k / 2;
        int spatial = h * w;

        var xArr = x.ToArray();
        return CausalConv3D(xArr, weight, bias, inCh, outCh, f, h, w, padT, padH, padW, spatial);
    }

    private float[] CausalConv3D(float[] xArr, string name, int inCh, int outCh, int f, int h, int w, bool causalTemporal)
    {
        var weight = GetWeight($"{name}.conv.weight");
        var bias = GetWeight($"{name}.conv.bias");
        const int k = 3;
        int padT = causalTemporal ? k - 1 : (k - 1) / 2;
        int padH = k / 2, padW = k / 2;
        int spatial = h * w;

        return CausalConv3D(xArr, weight, bias, inCh, outCh, f, h, w, padT, padH, padW, spatial);
    }

    private static float[] CausalConv3D(float[] xArr, float[] weight, float[] bias, int inCh, int outCh, int f, int h, int w, int padT, int padH, int padW, int spatial)
    {
        const int k = 3;
        var output = new float[outCh * f * spatial];

        Parallel.For(0, outCh, oc =>
        {
            float b = bias[oc];
            int ocWeightBase = oc * inCh * 27;

            for (int outT = 0; outT < f; outT++)
            {
                int outOffset = (oc * f + outT) * spatial;

                for (int oh = 0; oh < h; oh++)
                {
                    int inH0 = oh - padH;
                    int inH1 = oh - padH + 1;
                    int inH2 = oh - padH + 2;
                    bool h0 = inH0 >= 0 && inH0 < h;
                    bool h1 = inH1 >= 0 && inH1 < h;
                    bool h2 = inH2 >= 0 && inH2 < h;

                    for (int ow = 0; ow < w; ow++)
                    {
                        int inW0 = ow - padW;
                        int inW1 = ow - padW + 1;
                        int inW2 = ow - padW + 2;
                        bool w0 = inW0 >= 0 && inW0 < w;
                        bool w1 = inW1 >= 0 && inW1 < w;
                        bool w2 = inW2 >= 0 && inW2 < w;

                        float sum = b;

                        for (int ic = 0; ic < inCh; ic++)
                        {
                            int icWeightBase = ocWeightBase + ic * 27;

                            for (int dt = 0; dt < k; dt++)
                            {
                                int inT = outT - padT + dt;
                                int clampedT = Math.Clamp(inT, 0, f - 1);
                                int inFrameOff = (ic * f + clampedT) * spatial;
                                int wOff = icWeightBase + dt * 9;

                                if (h0)
                                {
                                    int r = inFrameOff + inH0 * w;
                                    if (w0) sum += xArr[r + inW0] * weight[wOff + 0];
                                    if (w1) sum += xArr[r + inW1] * weight[wOff + 1];
                                    if (w2) sum += xArr[r + inW2] * weight[wOff + 2];
                                }
                                if (h1)
                                {
                                    int r = inFrameOff + inH1 * w;
                                    if (w0) sum += xArr[r + inW0] * weight[wOff + 3];
                                    if (w1) sum += xArr[r + inW1] * weight[wOff + 4];
                                    if (w2) sum += xArr[r + inW2] * weight[wOff + 5];
                                }
                                if (h2)
                                {
                                    int r = inFrameOff + inH2 * w;
                                    if (w0) sum += xArr[r + inW0] * weight[wOff + 6];
                                    if (w1) sum += xArr[r + inW1] * weight[wOff + 7];
                                    if (w2) sum += xArr[r + inW2] * weight[wOff + 8];
                                }
                            }
                        }

                        output[outOffset + oh * w + ow] = sum;
                    }
                }
            }
        });

        return output;
    }

    /// <summary>Real `unpatchify(x, patch_size_hw=4, patch_size_t=1)`:
    /// `"b (c p r q) f h w -> b c (f p) (h q) (w r)"`. With `p`(temporal)=1, this is a pure spatial
    /// pixel-unshuffle: channel `ch = c_out*16 + r*4 + q` maps to output pixel
    /// `[c_out, f, h*4+q, w*4+r]`.</summary>
    private static float[] Unpatchify(float[] x, int f, int h, int w, int inCh)
    {
        int outCh = inCh / (PatchSize * PatchSize);
        int outH = h * PatchSize, outW = w * PatchSize;
        var dst = new float[outCh * f * outH * outW];
        int spatialSrc = h * w;
        int spatialDst = outH * outW;

        for (int cOut = 0; cOut < outCh; cOut++)
        {
            for (int r = 0; r < PatchSize; r++)
            for (int q = 0; q < PatchSize; q++)
            {
                int ch = cOut * PatchSize * PatchSize + r * PatchSize + q;
                for (int fr = 0; fr < f; fr++)
                {
                    int srcFrameOff = (ch * f + fr) * spatialSrc;
                    int dstFrameOff = (cOut * f + fr) * spatialDst;
                    for (int hh = 0; hh < h; hh++)
                    {
                        int dstH = hh * PatchSize + q;
                        int srcRow = srcFrameOff + hh * w;
                        int dstRowBase = dstFrameOff + dstH * outW;
                        for (int ww = 0; ww < w; ww++)
                        {
                            int dstW = ww * PatchSize + r;
                            dst[dstRowBase + dstW] = x[srcRow + ww];
                        }
                    }
                }
            }
        }
        return dst;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _weights.Dispose();
        }
    }
}
