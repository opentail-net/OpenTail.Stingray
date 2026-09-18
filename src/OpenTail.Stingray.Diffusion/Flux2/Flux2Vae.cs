namespace OpenTail.Stingray.Diffusion.Flux2;

/// <summary>
/// Real FLUX.2 VAE latent preprocessing (docs/087): un-normalizes the DiT's 128-channel patched
/// latent via the checkpoint's real per-channel BatchNorm running stats, then un-shuffles it back
/// to a 32-channel latent at 2x the spatial resolution -- confirmed against
/// `examples/flux2/src/flux2/autoencoder.py`'s real `AutoEncoder.inv_normalize`/`decode`:
/// <code>
/// s = sqrt(running_var + eps); m = running_mean
/// z = z * s + m                                      // per-channel affine, NOT a global scalar
/// z = rearrange(z, "(c pi pj) i j -> c (i pi) (j pj)", pi=2, pj=2)   // 2x2 pixel-unshuffle
/// </code>
/// The decoder blocks themselves (mid_block/up_blocks/conv_out) are the same real diffusers
/// `AutoencoderKL` schema this codebase's general-purpose <see cref="VaeDecoder"/> already parses
/// for FLUX.1/SD1.5/SDXL/SD3 -- only this input preprocessing step is FLUX.2-specific. After
/// calling <see cref="UnnormalizeAndUnshuffle"/>, pass the result to
/// <c>VaeDecoder.Decode(latent, h, w, scaleOverride: 1f, shiftOverride: 0f)</c> since the real
/// normalization already happened here.
/// </summary>
public static class Flux2Vae
{
    public const float BatchNormEps = 1e-4f;

    /// <summary>
    /// Real per-channel affine un-normalize (BatchNorm2d inverse) followed by the real 2x2
    /// pixel-unshuffle. <paramref name="normalizedLatent"/> is `[128, h, w]` (CHW, row-major);
    /// <paramref name="runningMean"/>/<paramref name="runningVar"/> are the checkpoint's real
    /// `bn.running_mean`/`bn.running_var` tensors (128 values each). Returns a `[32, 2h, 2w]`
    /// (CHW) latent ready for the standard VAE decoder blocks.
    /// </summary>
    public static (float[] latent, int h, int w) UnnormalizeAndUnshuffle(
        ReadOnlySpan<float> normalizedLatent, int inH, int inW,
        ReadOnlySpan<float> runningMean, ReadOnlySpan<float> runningVar)
    {
        int cIn = runningMean.Length;
        if (runningVar.Length != cIn)
            throw new ArgumentException("runningMean/runningVar length mismatch.", nameof(runningVar));
        if (normalizedLatent.Length != cIn * inH * inW)
            throw new ArgumentException($"normalizedLatent length {normalizedLatent.Length} != {cIn}*{inH}*{inW}.", nameof(normalizedLatent));
        if (cIn % 4 != 0)
            throw new ArgumentException($"Channel count {cIn} must be divisible by 4 (2x2 unshuffle).", nameof(runningMean));

        int cOut = cIn / 4;
        int outH = inH * 2, outW = inW * 2;
        int inPlane = inH * inW;
        var output = new float[cOut * outH * outW];

        for (int c = 0; c < cOut; c++)
        {
            for (int pi = 0; pi < 2; pi++)
            {
                for (int pj = 0; pj < 2; pj++)
                {
                    int cIndex = c * 4 + pi * 2 + pj;
                    float scale = MathF.Sqrt(runningVar[cIndex] + BatchNormEps);
                    float shift = runningMean[cIndex];
                    int inOff = cIndex * inPlane;
                    for (int i = 0; i < inH; i++)
                    {
                        int outRow = i * 2 + pi;
                        for (int j = 0; j < inW; j++)
                        {
                            int outCol = j * 2 + pj;
                            float v = normalizedLatent[inOff + i * inW + j] * scale + shift;
                            output[c * outH * outW + outRow * outW + outCol] = v;
                        }
                    }
                }
            }
        }

        return (output, outH, outW);
    }
}
