namespace OpenTail.Stingray.Diffusion.LTXVideo;

/// <summary>
/// 3D Continuous Positional Rotary Embedding (3D-RoPE) for LTX-Video.
/// Reference: stable-diffusion.cpp:src/model/diffusion/ltxv.hpp (generate_freq_grid and apply_rope)
/// </summary>
public static class LtxVideoRoPE
{
    public static (float[] cos, float[] sin) ComputeContinuous3DRoPE(
        int numFrames,
        int patchH,
        int patchW,
        int headDim = 64,
        float theta = 10000.0f,
        int fps = 24,
        int temporalScale = 8,
        int spatialScale = 32)
    {
        int positionalDims = 3; // (t, h, w)
        int nElem = 2 * positionalDims; // 6
        int freqCount = headDim / nElem;

        var freqs = new float[freqCount];
        float halfPi = MathF.PI / 2.0f;
        float logTheta = MathF.Log(theta);

        for (int i = 0; i < freqCount; i++)
        {
            float ratio = freqCount > 1 ? (float)i / (freqCount - 1) : 0f;
            freqs[i] = MathF.Exp(logTheta * ratio) * halfPi;
        }

        int totalTokens = numFrames * patchH * patchW;
        var cos = new float[totalTokens * headDim];
        var sin = new float[totalTokens * headDim];

        int token = 0;
        for (int t = 0; t < numFrames; t++)
        {
            float tCorner = t * temporalScale;
            float tStart = MathF.Max(0f, tCorner + 1f - temporalScale) / fps;
            float tEnd = MathF.Max(0f, (t + 1) * temporalScale + 1f - temporalScale) / fps;
            float tMid = (tStart + tEnd) * 0.5f;

            for (int h = 0; h < patchH; h++)
            {
                float hStart = h * spatialScale;
                float hEnd = (h + 1) * spatialScale;
                float hMid = (hStart + hEnd) * 0.5f;

                for (int w = 0; w < patchW; w++)
                {
                    float wStart = w * spatialScale;
                    float wEnd = (w + 1) * spatialScale;
                    float wMid = (wStart + wEnd) * 0.5f;

                    int baseOff = token * headDim;

                    for (int f = 0; f < freqCount; f++)
                    {
                        float freq = freqs[f];

                        // Temporal
                        float tAngle = tMid * freq;
                        cos[baseOff + f] = MathF.Cos(tAngle);
                        sin[baseOff + f] = MathF.Sin(tAngle);

                        // Height
                        float hAngle = hMid * freq;
                        cos[baseOff + freqCount + f] = MathF.Cos(hAngle);
                        sin[baseOff + freqCount + f] = MathF.Sin(hAngle);

                        // Width
                        float wAngle = wMid * freq;
                        cos[baseOff + 2 * freqCount + f] = MathF.Cos(wAngle);
                        sin[baseOff + 2 * freqCount + f] = MathF.Sin(wAngle);
                    }

                    // Remaining channels pass through with cos=1, sin=0
                    for (int rem = 3 * freqCount; rem < headDim; rem++)
                    {
                        cos[baseOff + rem] = 1.0f;
                        sin[baseOff + rem] = 0.0f;
                    }

                    token++;
                }
            }
        }

        return (cos, sin);
    }
}
