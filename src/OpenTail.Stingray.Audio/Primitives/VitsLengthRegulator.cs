using System;

namespace OpenTail.Stingray.Audio.Primitives;

/// <summary>
/// Bridges a duration predictor's per-token log-duration output to a normalizing flow's per-frame
/// latent input -- shared across the whole VITS family (Piper, MeloTTS). Ported from VITS's
/// `SynthesizerTrn.infer`: `w = exp(logw) * length_scale`, `w_ceil = ceil(w)`, each token `i` is
/// repeated `w_ceil[i]` consecutive frames (a monotonic, non-overlapping alignment -- the
/// reference builds this via a `generate_path` matmul against a [T_frames, T_tokens] mask, which
/// is mathematically a repeat-interleave since the mask has exactly one contiguous run of 1s per
/// token). `z_p[frame] = m_p_exp[frame] + noise[frame] * exp(logs_p_exp[frame]) * noise_scale`,
/// where `noise` is a fresh [dim, T_frames] N(0,1) draw (a second RandomNormalLike in the graph,
/// distinct from the duration predictor's own noise draw). Extracted from the original Piper-only
/// implementation so MeloTTS's port reuses the same math instead of a second hand-rolled copy.
/// </summary>
public static class VitsLengthRegulator
{
    /// <summary>
    /// mu/logs are the TextEncoder's per-token [dim, tTokens] outputs; logw is the duration
    /// predictor's [tTokens] output; noise is a [dim * tFrames] N(0,1) draw (caller-supplied,
    /// matching tFrames = sum(ceil(exp(logw) * lengthScale))). Returns (zp [dim*tFrames], tFrames,
    /// durations [tTokens]).
    /// </summary>
    public static (float[] Zp, int TFrames, int[] Durations) Expand(float[] mu, float[] logs, int dim, int tTokens, float[] logw, float lengthScale, float[] noise, float noiseScale)
    {
        var durations = new int[tTokens];
        for (int i = 0; i < tTokens; i++)
        {
            float w = MathF.Exp(logw[i]) * lengthScale;
            int d = (int)MathF.Ceiling(w);
            durations[i] = d < 0 ? 0 : d;
        }
        return ExpandWithDurations(mu, logs, dim, tTokens, durations, noise, noiseScale);
    }

    /// <summary>Same as <see cref="Expand"/> but takes pre-computed integer durations directly (for
    /// isolating length-regulator/flow correctness from the duration predictor's own correctness in tests).</summary>
    public static (float[] Zp, int TFrames, int[] Durations) ExpandWithDurations(float[] mu, float[] logs, int dim, int tTokens, int[] durations, float[] noise, float noiseScale)
    {
        int tFrames = 0;
        for (int i = 0; i < tTokens; i++) tFrames += durations[i];

        if (noise.Length != dim * tFrames)
            throw new ArgumentException($"noise length {noise.Length} does not match dim*tFrames = {dim * tFrames}.");

        var zp = new float[dim * tFrames];
        int frame = 0;
        for (int tok = 0; tok < tTokens; tok++)
        {
            for (int r = 0; r < durations[tok]; r++)
            {
                for (int c = 0; c < dim; c++)
                {
                    float m = mu[c * tTokens + tok];
                    float logStd = logs[c * tTokens + tok];
                    zp[c * tFrames + frame] = m + noise[c * tFrames + frame] * MathF.Exp(logStd) * noiseScale;
                }
                frame++;
            }
        }

        return (zp, tFrames, durations);
    }
}
