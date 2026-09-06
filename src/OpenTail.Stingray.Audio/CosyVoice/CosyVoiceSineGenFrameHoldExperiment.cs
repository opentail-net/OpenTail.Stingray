
using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// EXPERIMENT (2026-09-05): CosyVoice2/3-only fork of the NSF harmonic sine-source generator,
/// implementing the frame-rate cumulative-phase-then-hold algorithm (with a half-sample-position
/// linear interpolation of the per-sample instantaneous frequency before accumulating), matching
/// `examples/audio.cpp/src/framework/modules/vocoders/hift_vocoder.cpp`'s
/// `make_causal_sinegen2_source` (the real CosyVoice3 `HiftVocoderSourceMode::CausalSineGen2` path)
/// -- and this project's own docs/audio-review-progress.md "CosyVoice3: status set to PARTIALLY
/// SUPPORTED" entry, which independently checked the SAME question against a DIFFERENT real
/// reference (`examples/cosyvoice.cpp`) and reached the same conclusion.
///
/// Deliberately NOT wired into the shared <see cref="HiFTVocoderKernels"/> used by Chatterbox
/// (`ChatterboxVocoder.cs`), which was confirmed to sound correct this session under the CURRENT
/// continuous per-sample phase accumulation in that shared file -- changing shared code on an
/// unproven hypothesis risks regressing an already-confirmed-good pipeline. This file exists to
/// test the frame-hold hypothesis against CosyVoice3's own "wobbling distortion" symptom in
/// isolation, using only the shared, unmodified F0Predictor/ISTFT-decode pieces (which are not
/// under suspicion here) via <see cref="HiFTVocoderKernels"/>'s internal test-support methods.
/// </summary>
public static class CosyVoiceSineGenFrameHoldExperiment
{
    private const float Pi = MathF.PI;
    private const float TwoPi = 2f * Pi;

    /// <summary>Same call shape as <see cref="CosyVoiceHiftVocoder.Generate"/>, but with the
    /// frame-hold SineGen instead of the shared continuous-phase one.</summary>
    public static float[] Generate(IHiFTVocoderWeights w, float[] mel, int t, Random rng, float pitchScale = 1.0f)
    {
        float[] f0 = HiFTVocoderKernels.PredictF0Full(w.F0Predictor, mel, t, 80, w.IsCausal);
        if (pitchScale != 1.0f && pitchScale > 0.05f)
        {
            for (int i = 0; i < f0.Length; i++) f0[i] *= pitchScale;
        }

        int totalUp = w.IstftHopLen;
        foreach (int r in w.UpsampleRates) totalUp *= r;
        int sampleLen = t * totalUp;

        float[] harSource = SineGenFrameHold(f0, t, totalUp, w.SampleRate, w.NbHarmonics, rng,
            sineAmp: 0.1f, noiseStd: 0.003f, voicedThreshold: 10f);
        float[] excitation = LinearTanhMerge(harSource, sampleLen, w.NbHarmonics + 1, w.MSourceLinearWeight, w.MSourceLinearBias);

        return HiFTVocoderKernels.DecodeForTest(w, mel, t, excitation, sampleLen, 80);
    }

    /// <summary>
    /// Real `make_causal_sinegen2_source` (audio.cpp reference): for each output sample, the
    /// per-harmonic instantaneous normalized frequency is linearly interpolated between the two
    /// FRAME-RATE sample positions straddling that sample's "center" (half-pixel convention,
    /// matching image-resize align_corners=False), then accumulated ONCE PER FRAME (not per
    /// sample) into a running phase; the resulting sine value is HELD CONSTANT across all
    /// `scaleFactor` samples in that frame (only the per-sample additive noise varies within a
    /// frame), not smoothly varied sample-to-sample like the continuous-phase implementation.
    /// </summary>
    private static float[] SineGenFrameHold(float[] f0, int t, int scaleFactor, int sampleRate, int harmonicNum, Random rng,
                                             float sineAmp, float noiseStd, float voicedThreshold)
    {
        int dim = harmonicNum + 1;
        int sampleLen = t * scaleFactor;

        var randIniRadians = new float[dim];
        randIniRadians[0] = 0f;
        for (int h = 1; h < dim; h++) randIniRadians[h] = (float)(rng.NextDouble() * TwoPi - Pi);

        // high_rad(sample, harmonic): instantaneous normalized frequency at the FRAME owning
        // `sample`, plus the random initial phase offset for sample 0 only.
        float HighRad(int sample, int h)
        {
            int frame = Math.Clamp(sample / scaleFactor, 0, t - 1);
            float rad = (f0[frame] * (h + 1) / sampleRate) % 1.0f;
            if (sample == 0) rad += randIniRadians[h];
            return rad;
        }

        var sineWaves = new float[dim * sampleLen];
        var cumulative = new float[dim];

        for (int frame = 0; frame < t; frame++)
        {
            float baseF0 = f0[frame];
            float uv = baseF0 > voicedThreshold ? 1f : 0f;

            // Half-pixel-center sample position this frame's phase-rate estimate is centered on.
            float sourcePos = (frame + 0.5f) * scaleFactor - 0.5f;
            int left = (int)MathF.Floor(sourcePos);
            int right = Math.Min(left + 1, sampleLen - 1);
            float lerp = sourcePos - left;
            int leftClamped = Math.Max(left, 0);

            for (int h = 0; h < dim; h++)
            {
                float rad = HighRad(leftClamped, h) * (1f - lerp) + HighRad(right, h) * lerp;
                cumulative[h] += rad;
                float phase = cumulative[h] * TwoPi;
                float sineVal = sineAmp * MathF.Sin(phase) * uv;

                int row = h * sampleLen;
                int frameStart = frame * scaleFactor;
                for (int s = 0; s < scaleFactor; s++)
                    sineWaves[row + frameStart + s] = sineVal;
            }
        }

        for (int ti = 0; ti < t; ti++)
        {
            bool voiced = f0[ti] > voicedThreshold;
            float uv = voiced ? 1f : 0f;
            float noiseAmp = uv * noiseStd + (1f - uv) * sineAmp / 3f;
            int frameStart = ti * scaleFactor;
            for (int s = 0; s < scaleFactor; s++)
            {
                int n = frameStart + s;
                for (int h = 0; h < dim; h++)
                {
                    int idx = h * sampleLen + n;
                    float u1 = MathF.Max(1e-7f, (float)rng.NextDouble());
                    float u2 = (float)rng.NextDouble();
                    float gNoise = MathF.Sqrt(-2.0f * MathF.Log(u1)) * MathF.Cos(2.0f * MathF.PI * u2);
                    sineWaves[idx] += noiseAmp * gNoise;
                }
            }
        }

        return sineWaves;
    }

    private static float[] LinearTanhMerge(float[] sines, int len, int dim, float[] weight, float[] bias)
    {
        var output = new float[len];
        float b = bias[0];
        for (int n = 0; n < len; n++)
        {
            float sum = b;
            for (int h = 0; h < dim; h++) sum += weight[h] * sines[h * len + n];
            output[n] = MathF.Tanh(sum);
        }
        return output;
    }
}
