
namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// Directional Compression Autoencoder (DC-AE) Decoder.
/// High-compression spatial autoencoder decoding 32x / 64x compressed latents to full RGB images.
/// Reference: diffusers.models.autoencoders.autoencoder_dc.DCAEDecoder (Sana / DC-AE)
/// </summary>
public sealed class DcAeDecoder : IDisposable
{
    private readonly IWeightLoader? _weights;
    private readonly IComputeBackend? _backend;
    private readonly Dictionary<string, float[]> _weightCache = new(StringComparer.Ordinal);
    private bool _disposed;

    public int CompressionRatio { get; }
    public int LatentChannels { get; }

    public DcAeDecoder(IWeightLoader? weights = null, int compressionRatio = 32, int latentChannels = 32, IComputeBackend? backend = null)
    {
        _weights = weights;
        CompressionRatio = compressionRatio;
        LatentChannels = latentChannels;
        _backend = backend;
    }

    /// <summary>
    /// Decodes a highly-compressed latent [C, H/scale, W/scale] to RGB [3, H, W].
    /// </summary>
    public float[] Decode(float[] latent, int latH, int latW)
    {
        int inChannels = LatentChannels;
        if (latent.Length != inChannels * latH * latW)
            throw new ArgumentException($"Latent size mismatch: expected {inChannels * latH * latW}, got {latent.Length}");

        // 1. Initial Projection: Conv2D(inChannels -> 512, 3x3)
        int ch = 512, h = latH, w = latW;
        var z = Conv2D(latent, "decoder.conv_in", inChannels, ch, h, w, 3);

        // 2. Multi-stage residual upsampling blocks
        // Stages: 512 -> 512 (2x), 512 -> 256 (2x), 256 -> 128 (2x), 128 -> 64 (2x), 64 -> 32 (2x if 32x)
        int stages = (int)Math.Round(Math.Log2(CompressionRatio)); // 5 for 32x, 6 for 64x

        int[] stageChannels = [512, 256, 128, 64, 32, 32];

        for (int s = 0; s < stages; s++)
        {
            int nextCh = s < stageChannels.Length ? stageChannels[s] : 32;
            z = ResBlock(z, $"decoder.stages.{s}.block0", ch, ch, h, w);
            z = ResBlock(z, $"decoder.stages.{s}.block1", ch, ch, h, w);

            // Upsample 2x
            (z, h, w) = NearestUpsample2x(z, ch, h, w);
            z = Conv2D(z, $"decoder.stages.{s}.conv_up", ch, nextCh, h, w, 3);
            ch = nextCh;
        }

        // 3. Final projection: Conv2D(ch -> 3, 3x3)
        z = NormChannel(z, "decoder.norm_out", ch, h, w);
        DiffusionOps.SiluInPlace(z);
        var rgb = Conv2D(z, "decoder.conv_out", ch, 3, h, w, 3);

        // Clamp to [0, 1]
        for (int i = 0; i < rgb.Length; i++)
            rgb[i] = Math.Clamp((rgb[i] + 1.0f) * 0.5f, 0.0f, 1.0f);

        return rgb;
    }

    private float[] ResBlock(float[] x, string prefix, int inCh, int outCh, int h, int w)
    {
        var residual = x;
        var hState = NormChannel(x, $"{prefix}.norm1", inCh, h, w);
        DiffusionOps.SiluInPlace(hState);
        hState = Conv2D(hState, $"{prefix}.conv1", inCh, outCh, h, w, 3);

        hState = NormChannel(hState, $"{prefix}.norm2", outCh, h, w);
        DiffusionOps.SiluInPlace(hState);
        hState = Conv2D(hState, $"{prefix}.conv2", outCh, outCh, h, w, 3);

        if (inCh != outCh)
            residual = Conv2D(residual, $"{prefix}.conv_shortcut", inCh, outCh, h, w, 1);

        TensorPrimitives.Add(hState, residual, hState);
        return hState;
    }

    public static (float[] output, int outH, int outW) NearestUpsample2x(float[] x, int c, int h, int w)
    {
        int outH = h * 2;
        int outW = w * 2;
        var output = new float[c * outH * outW];

        Parallel.For(0, c, ch =>
        {
            int inOff = ch * h * w;
            int outOff = ch * outH * outW;

            for (int oh = 0; oh < outH; oh++)
            {
                int ih = oh / 2;
                for (int ow = 0; ow < outW; ow++)
                {
                    int iw = ow / 2;
                    output[outOff + oh * outW + ow] = x[inOff + ih * w + iw];
                }
            }
        });

        return (output, outH, outW);
    }

    private float[] Conv2D(float[] x, string prefix, int inCh, int outCh, int h, int w, int ksize)
    {
        var weights = GetWeight($"{prefix}.weight", outCh * inCh * ksize * ksize);
        var bias = GetWeight($"{prefix}.bias", outCh);

        var output = new float[outCh * h * w];
        int pad = ksize / 2;

        Parallel.For(0, outCh, oc =>
        {
            float b = bias != null ? bias[oc] : 0.0f;
            int outOff = oc * h * w;

            for (int oh = 0; oh < h; oh++)
            for (int ow = 0; ow < w; ow++)
            {
                float sum = b;
                for (int ic = 0; ic < inCh; ic++)
                {
                    int inOff = ic * h * w;
                    int wOff = (oc * inCh + ic) * ksize * ksize;

                    for (int kh = 0; kh < ksize; kh++)
                    for (int kw = 0; kw < ksize; kw++)
                    {
                        int ih = oh - pad + kh;
                        int iw = ow - pad + kw;
                        if (ih >= 0 && ih < h && iw >= 0 && iw < w)
                        {
                            float inVal = x[inOff + ih * w + iw];
                            float wVal = weights != null ? weights[wOff + kh * ksize + kw] : 0.0f;
                            sum += inVal * wVal;
                        }
                    }
                }
                output[outOff + oh * w + ow] = sum;
            }
        });

        return output;
    }

    private float[] NormChannel(float[] x, string prefix, int c, int h, int w, float eps = 1e-5f)
    {
        var gamma = GetWeight($"{prefix}.weight", c);
        var output = new float[x.Length];
        int spatialSize = h * w;

        Parallel.For(0, spatialSize, s =>
        {
            float sumSq = 0.0f;
            for (int ch = 0; ch < c; ch++)
            {
                float v = x[ch * spatialSize + s];
                sumSq += v * v;
            }
            float invStd = 1.0f / MathF.Sqrt(sumSq / c + eps);

            for (int ch = 0; ch < c; ch++)
            {
                int idx = ch * spatialSize + s;
                float g = gamma != null ? gamma[ch] : 1.0f;
                output[idx] = x[idx] * invStd * g;
            }
        });

        return output;
    }

    private float[]? GetWeight(string name, int expectedLength)
    {
        if (_weightCache.TryGetValue(name, out var cached))
            return cached;

        if (_weights != null && _weights.Contains(name))
        {
            var data = _weights.ReadF32(name);
            _weightCache[name] = data;
            return data;
        }

        var fallback = new float[expectedLength];
        _weightCache[name] = fallback;
        return fallback;
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
