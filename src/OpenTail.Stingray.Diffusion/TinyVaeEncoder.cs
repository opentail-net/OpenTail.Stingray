
namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// Tiny Autoencoder (TAESD / AutoencoderTiny) Encoder.
/// Fast, lightweight encoder mapping RGB images [3, H, W] to compact latent representations [C, H/8, W/8].
/// Supports 4-channel latents (Stable Diffusion 1.5, SDXL) and 16-channel latents (FLUX.1, SD3, Z-Image).
/// Reference: diffusers.models.autoencoders.autoencoder_tiny.EncoderTiny (Ollin Boer Bohan / HuggingFace)
/// </summary>
public sealed class TinyVaeEncoder : IDisposable
{
    private readonly IWeightLoader? _weights;
    private readonly IComputeBackend? _backend;
    private readonly Dictionary<string, float[]> _weightCache = new(StringComparer.Ordinal);
    private bool _disposed;

    public int InChannels { get; }
    public int LatentChannels { get; }
    public int BlockChannels { get; }
    public int[] NumBlocks { get; }

    public TinyVaeEncoder(
        IWeightLoader? weights = null,
        int latentChannels = 4,
        int inChannels = 3,
        int blockChannels = 64,
        int[]? numBlocks = null,
        IComputeBackend? backend = null)
    {
        _weights = weights;
        LatentChannels = latentChannels;
        InChannels = inChannels;
        BlockChannels = blockChannels;
        NumBlocks = numBlocks ?? [1, 3, 3, 3];
        _backend = backend;
    }

    public TinyVaeEncoder(string safetensorsPath, int latentChannels = 4)
        : this(SafetensorsLoader.Open(safetensorsPath), latentChannels)
    {
    }

    /// <summary>
    /// Encodes an RGB image [3, H, W] in range [0, 1] (or [-1, 1]) to latent [LatentChannels, H/8, W/8].
    /// </summary>
    public float[] Encode(float[] rgb, int height, int width, bool isMinusOneToOne = false)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (height % 8 != 0 || width % 8 != 0)
            throw new ArgumentException($"Image dimensions must be divisible by 8. Got {height}x{width}.", nameof(height));

        if (rgb.Length != InChannels * height * width)
            throw new ArgumentException($"RGB input length mismatch: expected {InChannels * height * width}, got {rgb.Length}", nameof(rgb));

        // 1. Pre-process to [0, 1] range expected by TAESD internal layers
        var z = new float[rgb.Length];
        if (isMinusOneToOne)
        {
            for (int i = 0; i < rgb.Length; i++)
            {
                z[i] = Math.Clamp((rgb[i] + 1.0f) * 0.5f, 0.0f, 1.0f);
            }
        }
        else
        {
            for (int i = 0; i < rgb.Length; i++)
            {
                z[i] = Math.Clamp(rgb[i], 0.0f, 1.0f);
            }
        }

        int h = height;
        int w = width;
        int ch = BlockChannels;
        int layerIdx = 0;

        // 2. Multi-stage encoding with stride-2 convolutions
        for (int stage = 0; stage < NumBlocks.Length; stage++)
        {
            int blocksInStage = NumBlocks[stage];

            if (stage == 0)
            {
                // Conv2D(3 -> 64, 3x3, stride=1, with bias)
                z = Conv2D(z, $"encoder.layers.{layerIdx}", InChannels, ch, h, w, 3, stride: 1, hasBias: true);
                layerIdx++;
            }
            else
            {
                // Downsample 2x: Conv2D(64 -> 64, 3x3, stride=2, no bias)
                int nextH = h / 2;
                int nextW = w / 2;
                z = Conv2D(z, $"encoder.layers.{layerIdx}", ch, ch, h, w, 3, stride: 2, hasBias: false);
                h = nextH;
                w = nextW;
                layerIdx++;
            }

            for (int b = 0; b < blocksInStage; b++)
            {
                z = AutoencoderTinyBlock(z, $"encoder.layers.{layerIdx}", ch, ch, h, w);
                layerIdx++;
            }
        }

        // 3. Final projection: Conv2D(64 -> LatentChannels, 3x3, with bias)
        z = Conv2D(z, $"encoder.layers.{layerIdx}", ch, LatentChannels, h, w, 3, stride: 1, hasBias: true);

        return z;
    }

    private float[] AutoencoderTinyBlock(float[] x, string prefix, int inCh, int outCh, int h, int w)
    {
        var hState = Conv2D(x, $"{prefix}.conv.0", inCh, outCh, h, w, 3, stride: 1, hasBias: true);
        ReluInPlace(hState);

        hState = Conv2D(hState, $"{prefix}.conv.2", outCh, outCh, h, w, 3, stride: 1, hasBias: true);
        ReluInPlace(hState);

        hState = Conv2D(hState, $"{prefix}.conv.4", outCh, outCh, h, w, 3, stride: 1, hasBias: true);

        float[] residual;
        if (inCh != outCh)
        {
            residual = Conv2D(x, $"{prefix}.skip", inCh, outCh, h, w, 1, stride: 1, hasBias: false);
        }
        else
        {
            residual = x;
        }

        TensorPrimitives.Add(hState, residual, hState);
        ReluInPlace(hState);
        return hState;
    }

    private static void ReluInPlace(Span<float> x)
    {
        for (int i = 0; i < x.Length; i++)
        {
            if (x[i] < 0.0f) x[i] = 0.0f;
        }
    }

    private float[] Conv2D(float[] x, string prefix, int inCh, int outCh, int inH, int inW, int ksize, int stride, bool hasBias)
    {
        var weights = GetWeight($"{prefix}.weight", outCh * inCh * ksize * ksize);
        var bias = hasBias ? GetWeight($"{prefix}.bias", outCh) : null;

        int outH = inH / stride;
        int outW = inW / stride;
        var output = new float[outCh * outH * outW];
        int pad = ksize / 2;

        Parallel.For(0, outCh, oc =>
        {
            float b = bias != null && oc < bias.Length ? bias[oc] : 0.0f;
            int outOff = oc * outH * outW;

            for (int oh = 0; oh < outH; oh++)
            {
                int outRowOff = outOff + oh * outW;
                int inCenterH = oh * stride;

                for (int ow = 0; ow < outW; ow++)
                {
                    int inCenterW = ow * stride;
                    float sum = b;

                    for (int ic = 0; ic < inCh; ic++)
                    {
                        int inOff = ic * inH * inW;
                        int wOff = (oc * inCh + ic) * ksize * ksize;

                        for (int kh = 0; kh < ksize; kh++)
                        {
                            int ih = inCenterH - pad + kh;
                            if ((uint)ih >= (uint)inH) continue;

                            int inRowOff = inOff + ih * inW;
                            int wRowOff = wOff + kh * ksize;

                            for (int kw = 0; kw < ksize; kw++)
                            {
                                int iw = inCenterW - pad + kw;
                                if ((uint)iw < (uint)inW)
                                {
                                    float inVal = x[inRowOff + iw];
                                    float wVal = weights != null ? weights[wRowOff + kw] : 0.0f;
                                    sum += inVal * wVal;
                                }
                            }
                        }
                    }
                    output[outRowOff + ow] = sum;
                }
            }
        });

        return output;
    }

    private float[]? GetWeight(string key, int expectedLength)
    {
        if (_weights == null)
        {
            return GetOrCreateSyntheticWeight(key, expectedLength);
        }

        lock (_weightCache)
        {
            if (_weightCache.TryGetValue(key, out var cached))
                return cached;

            string resolvedKey = ResolveKey(key);
            if (_weights.Contains(resolvedKey))
            {
                var w = _weights.ReadF32(resolvedKey);
                _weightCache[key] = w;
                return w;
            }

            return GetOrCreateSyntheticWeight(key, expectedLength);
        }
    }

    private string ResolveKey(string key)
    {
        if (_weights == null) return key;
        if (_weights.Contains(key)) return key;

        if (key.StartsWith("encoder.layers.", StringComparison.Ordinal))
        {
            string stripped = "encoder." + key["encoder.layers.".Length..];
            if (_weights.Contains(stripped)) return stripped;
        }

        string[] prefixes = ["first_stage_model.", "taesd.", "vae."];
        foreach (var p in prefixes)
        {
            if (_weights.Contains(p + key)) return p + key;
            if (key.StartsWith("encoder.layers.", StringComparison.Ordinal))
            {
                string alt = p + "encoder." + key["encoder.layers.".Length..];
                if (_weights.Contains(alt)) return alt;
            }
        }

        return key;
    }

    private float[] GetOrCreateSyntheticWeight(string key, int length)
    {
        lock (_weightCache)
        {
            if (_weightCache.TryGetValue(key, out var w))
                return w;

            var arr = new float[length];
            if (key.EndsWith(".bias", StringComparison.Ordinal))
            {
                Array.Fill(arr, 0.05f);
            }
            else
            {
                float scale = MathF.Sqrt(2.0f / MathF.Max(1.0f, length));
                for (int i = 0; i < length; i++)
                {
                    arr[i] = ((i % 17) + 1) * (scale * 0.05f);
                }
            }
            _weightCache[key] = arr;
            return arr;
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _weightCache.Clear();
            if (_weights is IDisposable d)
            {
                d.Dispose();
            }
        }
    }
}
