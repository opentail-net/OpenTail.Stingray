
namespace OpenTail.Stingray.Audio.Primitives;

/// <summary>Box-Muller N(0,1) sampler -- the real VITS-family ONNX graphs' `RandomNormalLike`
/// draws are per-run RNG (not reproducible weights), so production inference just needs valid
/// Gaussian noise, not a specific seed/sequence match (that isolation is covered separately by
/// the golden-noise tests, which feed an exact captured draw instead of using this sampler).
/// Extracted from the original Piper-only implementation so MeloTTS's end-to-end path reuses it.</summary>
public sealed class GaussianRandom
{
    private readonly Random _rng = new();

    public float[] NextArray(int count)
    {
        var result = new float[count];
        int i = 0;
        // Box-Muller produces two independent N(0,1) samples per (log,sqrt,sin,cos) evaluation --
        // use both instead of discarding the sin branch (halves the transcendental-call count).
        for (; i + 1 < count; i += 2)
        {
            double u1 = 1.0 - _rng.NextDouble();
            double u2 = _rng.NextDouble();
            double r = Math.Sqrt(-2.0 * Math.Log(u1));
            double theta = 2.0 * Math.PI * u2;
            result[i] = (float)(r * Math.Cos(theta));
            result[i + 1] = (float)(r * Math.Sin(theta));
        }
        if (i < count)
        {
            double u1 = 1.0 - _rng.NextDouble();
            double u2 = _rng.NextDouble();
            result[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        return result;
    }
}
