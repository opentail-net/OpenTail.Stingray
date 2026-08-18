namespace OpenTail.Stingray.Audio;

/// <summary>
/// Channel downmixing modes and standard broadcast coefficient matrices.
/// </summary>
public enum DownmixMatrix
{
    /// <summary>ITU-R BS.775 broadcast matrix: -3dB center, -3dB surrounds, drops LFE.</summary>
    ItuRbs775 = 0,

    /// <summary>ATSC A/85 matrix: -3dB center, -3dB surrounds, folds LFE at -10dB with headroom normalization.</summary>
    AtscA85 = 1
}

/// <summary>
/// High-performance audio channel downmixer ported from Transformations.Audio.
/// Converts multi-channel interleaved audio to Mono or Stereo with power-normalized folding.
/// </summary>
public static class AudioDownmixer
{
    private const float InvSqrt2 = 0.70710678f; // -3dB
    private const float LfeGain = 0.31622777f;  // -10dB

    /// <summary>
    /// Downmixes interleaved multi-channel audio to single-channel (Mono).
    /// </summary>
    public static float[] DownmixToMono(
        ReadOnlySpan<float> input,
        int sourceChannels,
        DownmixMatrix matrix = DownmixMatrix.AtscA85)
    {
        if (sourceChannels <= 0)
            throw new ArgumentException("Channel count must be positive.", nameof(sourceChannels));
        if (input.Length == 0)
            return [];
        if (input.Length % sourceChannels != 0)
            throw new ArgumentException("Input length must be a multiple of sourceChannels.", nameof(input));

        if (sourceChannels == 1)
            return input.ToArray();

        int frames = input.Length / sourceChannels;
        float[] mono = new float[frames];

        switch (sourceChannels)
        {
            case 2: // Stereo [L, R]
                for (int i = 0; i < frames; i++)
                {
                    int offset = i * 2;
                    mono[i] = (input[offset] + input[offset + 1]) * 0.5f;
                }
                break;

            case 6: // 5.1 Surround [L, R, C, LFE, Ls, Rs]
                float norm51 = matrix == DownmixMatrix.AtscA85 ? (1.0f / (1.0f + InvSqrt2 + InvSqrt2 + (matrix == DownmixMatrix.AtscA85 ? LfeGain : 0f))) : 0.5f;
                for (int i = 0; i < frames; i++)
                {
                    int off = i * 6;
                    float l = input[off];
                    float r = input[off + 1];
                    float c = input[off + 2];
                    float lfe = input[off + 3];
                    float ls = input[off + 4];
                    float rs = input[off + 5];

                    float sum = l + r + (c * InvSqrt2 * 2f) + (ls * InvSqrt2) + (rs * InvSqrt2);
                    if (matrix == DownmixMatrix.AtscA85) sum += lfe * LfeGain;
                    mono[i] = sum * norm51 * 0.5f;
                }
                break;

            default: // Generic N-channel average
                float invN = 1.0f / sourceChannels;
                for (int i = 0; i < frames; i++)
                {
                    int off = i * sourceChannels;
                    float sum = 0f;
                    for (int c = 0; c < sourceChannels; c++)
                        sum += input[off + c];
                    mono[i] = sum * invN;
                }
                break;
        }

        return mono;
    }

    /// <summary>
    /// Downmixes interleaved multi-channel audio to 2-channel Stereo [L, R].
    /// </summary>
    public static float[] DownmixToStereo(
        ReadOnlySpan<float> input,
        int sourceChannels,
        DownmixMatrix matrix = DownmixMatrix.AtscA85)
    {
        if (sourceChannels <= 0)
            throw new ArgumentException("Channel count must be positive.", nameof(sourceChannels));
        if (input.Length == 0)
            return [];
        if (input.Length % sourceChannels != 0)
            throw new ArgumentException("Input length must be a multiple of sourceChannels.", nameof(input));

        if (sourceChannels == 2)
            return input.ToArray();

        int frames = input.Length / sourceChannels;
        float[] stereo = new float[frames * 2];

        if (sourceChannels == 1) // Mono -> Dual Mono
        {
            for (int i = 0; i < frames; i++)
            {
                float val = input[i];
                stereo[i * 2] = val;
                stereo[i * 2 + 1] = val;
            }
            return stereo;
        }

        if (sourceChannels == 6) // 5.1 [L, R, C, LFE, Ls, Rs] -> Stereo [L', R']
        {
            float norm = matrix == DownmixMatrix.AtscA85 ? (1.0f / (1.0f + InvSqrt2 + InvSqrt2 + LfeGain * InvSqrt2)) : (1.0f / (1.0f + InvSqrt2 + InvSqrt2));
            for (int i = 0; i < frames; i++)
            {
                int off = i * 6;
                float l = input[off];
                float r = input[off + 1];
                float c = input[off + 2];
                float lfe = input[off + 3];
                float ls = input[off + 4];
                float rs = input[off + 5];

                float outL = l + (c * InvSqrt2) + (ls * InvSqrt2);
                float outR = r + (c * InvSqrt2) + (rs * InvSqrt2);
                if (matrix == DownmixMatrix.AtscA85)
                {
                    outL += lfe * LfeGain * InvSqrt2;
                    outR += lfe * LfeGain * InvSqrt2;
                }

                stereo[i * 2] = outL * norm;
                stereo[i * 2 + 1] = outR * norm;
            }
            return stereo;
        }

        // Generic fallback: first 2 channels or split evenly
        for (int i = 0; i < frames; i++)
        {
            int off = i * sourceChannels;
            float left = 0f, right = 0f;
            int half = sourceChannels / 2;
            for (int c = 0; c < half; c++) left += input[off + c];
            for (int c = half; c < sourceChannels; c++) right += input[off + c];
            stereo[i * 2] = left / half;
            stereo[i * 2 + 1] = right / (sourceChannels - half);
        }

        return stereo;
    }
}
