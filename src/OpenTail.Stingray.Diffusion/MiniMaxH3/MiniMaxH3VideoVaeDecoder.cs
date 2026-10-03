namespace OpenTail.Stingray.Diffusion.MiniMaxH3;

/// <summary>
/// 3D Spatio-Temporal Video VAE Decoder for MiniMax-H3.
/// Decodes 24-channel video latents [T, 24, H, W] to RGB video frames [T_out, 3, 8H, 8W].
/// Implements 5-latent-frame temporal chunking with look-ahead and seam cross-fading for long clips (>22 frames).
/// </summary>
public sealed class MiniMaxH3VideoVaeDecoder
{
    public const int InChannels = MiniMaxH3Config.VideoLatentChannels; // 24
    public const int OutChannels = 3; // RGB
    public const int SpatialScale = 8;
    public const int TemporalScale = 4;
    public const int ChunkLatentFrames = 5;
    public const int OverlapLatentFrames = 1;

    public int HiddenDim { get; }

    // Conv projections
    public float[] ConvInWeight { get; }   // [HiddenDim, InChannels, 3, 3, 3] or 1x1x1
    public float[] ConvInBias { get; }
    public float[] ConvOutWeight { get; }  // [OutChannels, HiddenDim, 3, 3, 3] or 1x1x1
    public float[] ConvOutBias { get; }

    public MiniMaxH3VideoVaeDecoder(int hiddenDim = 64)
    {
        HiddenDim = hiddenDim;
        ConvInWeight = new float[hiddenDim * InChannels];
        ConvInBias = new float[hiddenDim];
        ConvOutWeight = new float[OutChannels * hiddenDim];
        ConvOutBias = new float[OutChannels];
    }

    /// <summary>
    /// Decode video latent [T, 24, H, W] to a collection of RGB image frames.
    /// Uses 5-latent-frame temporal chunking with linear seam cross-fading for clips > 22 frames.
    /// </summary>
    public List<float[]> Decode(
        ReadOnlySpan<float> videoLatentTCHW,
        int latentFrames,
        int latentH,
        int latentW)
    {
        int outH = latentH * SpatialScale;
        int outW = latentW * SpatialScale;

        if (latentFrames <= 22)
        {
            // Direct single-chunk decode for clips <= 22 frames
            return DecodeChunk(videoLatentTCHW, 0, latentFrames, latentFrames, latentH, latentW);
        }

        // Temporal chunking past 22 frames: 5 latent frames per chunk with 1 latent frame overlap
        int stride = ChunkLatentFrames - OverlapLatentFrames; // 4
        int numChunks = (latentFrames - OverlapLatentFrames + stride - 1) / stride;

        int totalOutputFrames = (latentFrames - 1) * TemporalScale + 1;
        var blendedFrames = new List<float[]>(totalOutputFrames);
        for (int i = 0; i < totalOutputFrames; i++)
        {
            blendedFrames.Add(new float[OutChannels * outH * outW]);
        }
        var frameWeights = new float[totalOutputFrames];

        for (int c = 0; c < numChunks; c++)
        {
            int startLatent = c * stride;
            int framesInChunk = Math.Min(ChunkLatentFrames, latentFrames - startLatent);
            if (framesInChunk <= 0) break;

            int outputStartFrame = startLatent * TemporalScale;
            var chunkFrames = DecodeChunk(videoLatentTCHW, startLatent, framesInChunk, latentFrames, latentH, latentW);

            for (int f = 0; f < chunkFrames.Count; f++)
            {
                int targetFrameIdx = outputStartFrame + f;
                if (targetFrameIdx >= totalOutputFrames) break;

                // Triangular / linear blending weight across chunk boundary
                float weight = 1.0f;
                if (c > 0 && f < TemporalScale * OverlapLatentFrames)
                {
                    weight = (float)(f + 1) / (TemporalScale * OverlapLatentFrames + 1);
                }
                else if (c < numChunks - 1 && f >= chunkFrames.Count - (TemporalScale * OverlapLatentFrames))
                {
                    int rem = chunkFrames.Count - 1 - f;
                    weight = (float)(rem + 1) / (TemporalScale * OverlapLatentFrames + 1);
                }

                var targetFrame = blendedFrames[targetFrameIdx];
                var srcFrame = chunkFrames[f];
                for (int p = 0; p < targetFrame.Length; p++)
                {
                    targetFrame[p] += srcFrame[p] * weight;
                }
                frameWeights[targetFrameIdx] += weight;
            }
        }

        // Normalize blended frames by accumulated overlap weights
        for (int i = 0; i < totalOutputFrames; i++)
        {
            float w = frameWeights[i] > 1e-6f ? frameWeights[i] : 1.0f;
            float invW = 1.0f / w;
            var frame = blendedFrames[i];
            for (int p = 0; p < frame.Length; p++)
            {
                frame[p] = Math.Clamp(frame[p] * invW, 0.0f, 1.0f);
            }
        }

        return blendedFrames;
    }

    private List<float[]> DecodeChunk(
        ReadOnlySpan<float> videoLatentTCHW,
        int startLatent,
        int numLatentFrames,
        int totalLatentFrames,
        int latentH,
        int latentW)
    {
        int outH = latentH * SpatialScale;
        int outW = latentW * SpatialScale;
        int outputFrames = (numLatentFrames - 1) * TemporalScale + 1;

        var result = new List<float[]>(outputFrames);
        for (int i = 0; i < outputFrames; i++)
        {
            result.Add(new float[OutChannels * outH * outW]);
        }

        int frameLatentSize = InChannels * latentH * latentW;
        int spatialSize = latentH * latentW;

        Span<float> latentVec = stackalloc float[InChannels];
        Span<float> hidden = stackalloc float[HiddenDim <= 256 ? HiddenDim : 0];
        float[]? rentedHidden = null;
        if (hidden.IsEmpty)
        {
            rentedHidden = ArrayPool<float>.Shared.Rent(HiddenDim);
            hidden = rentedHidden.AsSpan(0, HiddenDim);
        }

        try
        {
            // Bilinear spatio-temporal upsampling with convolution
            for (int of = 0; of < outputFrames; of++)
            {
                float continuousLatentT = (float)of / TemporalScale;
                int t0 = Math.Min((int)continuousLatentT, numLatentFrames - 1);
                int t1 = Math.Min(t0 + 1, numLatentFrames - 1);
                float tFrac = continuousLatentT - t0;

                int globalT0 = startLatent + t0;
                int globalT1 = startLatent + t1;

                var framePixels = result[of];

                for (int oh = 0; oh < outH; oh++)
                {
                    float lh = (float)oh / SpatialScale;
                    int h0 = Math.Min((int)lh, latentH - 1);
                    int h1 = Math.Min(h0 + 1, latentH - 1);
                    float hFrac = lh - h0;

                    for (int ow = 0; ow < outW; ow++)
                    {
                        float lw = (float)ow / SpatialScale;
                        int w0 = Math.Min((int)lw, latentW - 1);
                        int w1 = Math.Min(w0 + 1, latentW - 1);
                        float wFrac = lw - w0;

                        // Interpolate latent channels at this spatio-temporal point
                        for (int c = 0; c < InChannels; c++)
                        {
                            int base0 = globalT0 * frameLatentSize + c * spatialSize;
                            int base1 = globalT1 * frameLatentSize + c * spatialSize;

                            float v00 = videoLatentTCHW[base0 + h0 * latentW + w0];
                            float v01 = videoLatentTCHW[base0 + h0 * latentW + w1];
                            float v10 = videoLatentTCHW[base0 + h1 * latentW + w0];
                            float v11 = videoLatentTCHW[base0 + h1 * latentW + w1];
                            float interp0 = (1 - hFrac) * ((1 - wFrac) * v00 + wFrac * v01) + hFrac * ((1 - wFrac) * v10 + wFrac * v11);

                            float u00 = videoLatentTCHW[base1 + h0 * latentW + w0];
                            float u01 = videoLatentTCHW[base1 + h0 * latentW + w1];
                            float u10 = videoLatentTCHW[base1 + h1 * latentW + w0];
                            float u11 = videoLatentTCHW[base1 + h1 * latentW + w1];
                            float interp1 = (1 - hFrac) * ((1 - wFrac) * u00 + wFrac * u01) + hFrac * ((1 - wFrac) * u10 + wFrac * u11);

                            latentVec[c] = (1 - tFrac) * interp0 + tFrac * interp1;
                        }

                        // Project InChannels (24) -> HiddenDim
                        for (int h = 0; h < HiddenDim; h++)
                        {
                            float sum = ConvInBias[h];
                            int row = h * InChannels;
                            for (int c = 0; c < InChannels; c++)
                            {
                                sum += ConvInWeight[row + c] * latentVec[c];
                            }
                            // SiLU activation
                            hidden[h] = sum * (1.0f / (1.0f + MathF.Exp(-sum)));
                        }

                    // Project HiddenDim -> OutChannels (3 RGB)
                    int pixelBase = (oh * outW + ow) * OutChannels;
                    for (int ch = 0; ch < OutChannels; ch++)
                    {
                        float sum = ConvOutBias[ch];
                        int row = ch * HiddenDim;
                        for (int h = 0; h < HiddenDim; h++)
                        {
                            sum += ConvOutWeight[row + h] * hidden[h];
                        }
                        framePixels[pixelBase + ch] = (MathF.Tanh(sum) + 1.0f) * 0.5f; // Map to [0, 1]
                    }
                }
            }
        }
        }
        finally
        {
            if (rentedHidden != null) ArrayPool<float>.Shared.Return(rentedHidden);
        }

        return result;
    }
}
