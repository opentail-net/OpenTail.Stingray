using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Diffusion.MiniMaxMusic3;

/// <summary>
/// Real MiniMax Music 3 single-chunk flow-matching Euler solver (`MiniMaxMusic3ChunkDenoiseStep`'s
/// "denoise inner" loop, minus the multi-window overlap-blend machinery -- V1 scope is a song short
/// enough to fit one real `_CHUNK_FRAMES=200`-frame (~8s at 25Hz) window, so `chunk_starts=[0]` and
/// there is no previous-window state to blend). See docs/066-minimax-music3-future-plan.md, "Real
/// chunked flow-matching denoise + stitching".
///
/// <para><b>Real scheduler -- CORRECTED against the independent `minimaxmusic.cpp` C++ reference
/// (`ServeurpersoCom/minimaxmusic.cpp`, `src/pipeline.cpp`), which is the actual ground truth here
/// since it is a real, working executable port, not just a generic reading of `diffusers` scheduler
/// source</b>: the schedule is ASCENDING, `t` runs 0.0 (pure noise) -> ~(1-1/steps) -> a final
/// appended 1.0 (clean), NOT the standard-looking descending `linspace(1.0, 1/steps, steps)+[0.0]`
/// this file previously used. The reference builds it as `lin = linspace(1, 1/steps, steps)` then
/// INVERTS every entry (`sig[i] = 1 - lin[i]`) before appending a final `1.0`; verified consistent
/// with the reference's own windowing math, where `t=0` weights the initial noise fully and `t=1`
/// weights the fully-denoised latent fully. The raw `t` value (this ascending sigma, not a
/// descending one) is fed straight into the DiT's Fourier timestep embedding
/// (`MiniMaxMusic3Transformer.TimestepEmbed`/reference `dit_time_embed`, both use the identical
/// `angle = 2*pi*t*w` formula with no extra scaling) -- so getting the direction backwards means
/// every single denoise step queries the network with a timestep from the opposite phase of its
/// training distribution, which plausibly explains the "jitter, not music" output despite every
/// individual component passing its own golden-parity test. Real CFG: unconditional branch
/// conditions on `zeros_like(condition)` (not a re-encoded empty prompt), `guidance_scale=1.7`.</para>
/// </summary>
public static class MiniMaxMusic3FlowScheduler
{
    public const float RealGuidanceScale = 1.7f;

    /// <summary>Real single-chunk Euler denoise loop. `condition` is this window's real
    /// `MiniMaxMusic3ConditionEncoder` output (`[latentLength][conditionDim]`). Returns the
    /// denoised latent, `[latentLength][inChannels(128)]`.</summary>
    public static float[][] Denoise(
        MiniMaxMusic3TransformerWeights transformerWeights,
        float[][] condition,
        int numSteps,
        int? seed,
        IComputeBackend? backend = null)
    {
        int latentLength = condition.Length;
        int inChannels = MiniMaxMusic3Config.TransformerInChannels;
        int condDim = MiniMaxMusic3Config.TransformerConditionDim;

        var random = seed is int s ? new Random(s) : new Random();
        var latent = new float[latentLength][];
        for (int t = 0; t < latentLength; t++)
        {
            var row = new float[inChannels];
            for (int c = 0; c < inChannels; c++) row[c] = SampleStandardNormal(random);
            latent[t] = row;
        }

        var zeroCondition = new float[latentLength][];
        for (int t = 0; t < latentLength; t++) zeroCondition[t] = new float[condDim];

        var sigmas = BuildSigmaSchedule(numSteps);

        for (int step = 0; step < numSteps; step++)
        {
            float sigma = sigmas[step];
            float sigmaNext = sigmas[step + 1];

            var (vCond, vUncond) = MiniMaxMusic3Transformer.ForwardPair(
                transformerWeights, latent, condition, zeroCondition, sigma, backend);

            float dt = sigmaNext - sigma;
            for (int t = 0; t < latentLength; t++)
            {
                for (int c = 0; c < inChannels; c++)
                {
                    float guided = vUncond[t][c] + (vCond[t][c] - vUncond[t][c]) * RealGuidanceScale;
                    latent[t][c] += dt * guided;
                }
            }
        }

        return latent;
    }

    /// <summary>Real ASCENDING schedule, transcribed exactly from the `minimaxmusic.cpp` reference
    /// (`src/pipeline.cpp`'s `sig[i] = 1 - lin[i]` where `lin = linspace(1.0, 1/steps, steps)`), with
    /// an appended terminal `1.0` (clean), length `steps+1`. `sigmas[0] == 0.0` (pure noise).</summary>
    private static float[] BuildSigmaSchedule(int steps)
    {
        var sigmas = new float[steps + 1];
        if (steps == 1)
        {
            sigmas[0] = 0.0f;
        }
        else
        {
            for (int i = 0; i < steps; i++)
            {
                float lin = 1.0f + (1.0f / steps - 1.0f) * i / (steps - 1);
                sigmas[i] = 1.0f - lin;
            }
        }
        sigmas[steps] = 1.0f;
        return sigmas;
    }

    private static float SampleStandardNormal(Random random)
    {
        double u1 = 1.0 - random.NextDouble();
        double u2 = random.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
    }
}
