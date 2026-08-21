using System;

namespace OpenTail.Stingray.Audio.F5TTS;

/// <summary>
/// F5-TTS's Euler flow-matching ODE sampler with classifier-free guidance, ported from
/// `examples/f5-tts-py/f5_tts/model/cfm.py`'s `CFM.sample`. `condMel` is the reference-audio mel
/// (real frames in its prefix, zero-padded to `numFrames` -- fixed for the whole trajectory, per
/// `step_cond` in the reference: conditioning is NOT recomputed every step).
///
/// SIMPLIFICATION vs. the reference (documented, not silently dropped): the reference defaults to
/// "Empirically Pruned Step Sampling" (EPSS, `use_epss=True`), a precomputed non-uniform NFE-
/// dependent step schedule loaded from a lookup table. This port uses the older, simpler
/// `torch.linspace(0, 1, steps+1)` + optional `sway_sampling_coef` schedule instead (still a real,
/// correct flow-matching schedule the reference explicitly supports as its non-EPSS path, just
/// not the newer pruned-step optimization) -- reproducing EPSS's exact lookup table was judged
/// out of scope for this iteration; revisit if audio quality at low step counts is insufficient.
/// </summary>
public static class F5FlowMatchingOde
{
    public static float[] Solve(
        F5TtsWeights w,
        float[] condMel,
        ReadOnlySpan<int> text,
        int numFrames,
        int steps = 32,
        float cfgStrength = 2.0f,
        float swaySamplingCoef = -1.0f,
        int seed = 42)
    {
        int melDim = F5TtsWeights.MelDim;
        var rng = new Random(seed);
        var x = new float[numFrames * melDim];
        for (int i = 0; i < x.Length; i++)
        {
            // Box-Muller N(0,1), matching torch.randn_like semantics closely enough for inference
            // (production sampling noise, not a value under golden-exactness verification).
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            x[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }

        var nullCond = new float[condMel.Length];
        var tokenArray = text.ToArray();

        for (int step = 0; step < steps; step++)
        {
            float t0 = (float)step / steps;
            float t1 = (float)(step + 1) / steps;
            float t = Sway(t0, swaySamplingCoef);
            float dt = Sway(t1, swaySamplingCoef) - t;

            float[] v;
            if (cfgStrength < 1e-5f)
            {
                v = F5DiTModel.ForwardVelocity(w, x, condMel, tokenArray, t, numFrames, dropText: false);
            }
            else
            {
                var vCond = F5DiTModel.ForwardVelocity(w, x, condMel, tokenArray, t, numFrames, dropText: false);
                var vUncond = F5DiTModel.ForwardVelocity(w, x, nullCond, tokenArray, t, numFrames, dropText: true);
                v = new float[vCond.Length];
                for (int i = 0; i < v.Length; i++) v[i] = vCond[i] + (vCond[i] - vUncond[i]) * cfgStrength;
            }

            for (int i = 0; i < x.Length; i++) x[i] += dt * v[i];
        }

        return x;
    }

    private static float Sway(float t, float coef)
    {
        if (MathF.Abs(coef) < 1e-4f) return t;
        float swayed = t + coef * (MathF.Cos(MathF.PI / 2f * t) - 1f + t);
        return Math.Clamp(swayed, 0f, 1f);
    }
}
