
namespace OpenTail.Stingray.Diffusion.Flux2;

/// <summary>
/// 4D Contextual Rotary Position Embedding (RoPE) generator for FLUX.2.
/// Positions tokens in 4D space (t, h, w, l) -- frame/time (for multi-reference-image temporal
/// offsetting), row, column, and a sequence-local index (used by text tokens) -- to allow
/// multi-reference image conditioning without coordinate collision.
/// Reference: examples/flux2/src/flux2/model.py's real `rope`/`apply_rope`/`EmbedND` (confirmed
/// in-repo 2026-09-18; axes_dim=[32,32,32,32], theta=2000 -- NOT FLUX.1's 3-axis/10000 scheme,
/// see docs/087-flux2-implementation-plan.md for the full derivation). The rotation convention
/// itself (interleaved adjacent pairs, `reshape(..., -1, 1, 2)` in `apply_rope`) matches FLUX.1's,
/// so the existing <see cref="InterleavedRoPE"/> helper is reused unchanged -- only the axis count
/// (3 -> 4) and per-axis theta differ.
/// </summary>
public static class Flux2RoPE
{
    /// <summary>
    /// Computes cosine and sine frequency matrices for 4D contextual tokens (t, h, w, l).
    /// Shape: [nTokens * headDim].
    /// </summary>
    /// <param name="positions">Array of (t, h, w, l) integer quadruplets, length = nTokens * 4.</param>
    /// <param name="nTokens">Number of patch/text tokens.</param>
    /// <param name="axesDim">Axial split [dimT, dimH, dimW, dimL], real value [32, 32, 32, 32] (sum must equal headDim).</param>
    /// <param name="theta">Base frequency theta, real value 2000.0 (NOT FLUX.1's 10000.0).</param>
    public static (float[] cos, float[] sin) BuildContextFreqs(
        ReadOnlySpan<int> positions,
        int nTokens,
        int[] axesDim,
        float theta = 2000.0f)
    {
        if (axesDim.Length != 4)
            throw new ArgumentException("FLUX.2 axesDim must contain 4 elements: [dimT, dimH, dimW, dimL].", nameof(axesDim));

        int numAxes = axesDim.Length;
        int headDim = 0;
        for (int a = 0; a < numAxes; a++) headDim += axesDim[a];

        var cos = new float[nTokens * headDim];
        var sin = new float[nTokens * headDim];

        var invFreqs = new float[numAxes][];
        for (int a = 0; a < numAxes; a++)
            invFreqs[a] = InterleavedRoPE.ComputeInvFreqs(axesDim[a], theta);

        for (int i = 0; i < nTokens; i++)
        {
            int outOffset = i * headDim;
            int axisOffset = 0;
            for (int a = 0; a < numAxes; a++)
            {
                int pos = positions[i * numAxes + a];
                int dim = axesDim[a];
                InterleavedRoPE.FillAxisFreqs(
                    cos.AsSpan(outOffset + axisOffset, dim),
                    sin.AsSpan(outOffset + axisOffset, dim),
                    pos, invFreqs[a]);
                axisOffset += dim;
            }
        }

        return (cos, sin);
    }

    /// <summary>
    /// Applies rotary embeddings in-place to Q or K tensor [nTokens, numHeads, headDim].
    /// </summary>
    public static void ApplyRoPE(Span<float> tensor, ReadOnlySpan<float> cos, ReadOnlySpan<float> sin, int nTokens, int numHeads, int headDim)
        => InterleavedRoPE.ApplyRoPE(tensor, cos, sin, nTokens, numHeads, headDim);

    /// <summary>
    /// Compact one-value-per-pair layout <c>[nTokens, headDim/2]</c> that the existing GPU
    /// <c>Flux2DRoPE</c> compute shader expects (<see cref="Core.IVisionOpsBackend.Flux2DRoPE"/>)
    /// -- vs <see cref="BuildContextFreqs"/>'s own CPU-only <c>[nTokens, headDim]</c> layout, which
    /// duplicates each pair's cos/sin value at both `2i` and `2i+1`. The actual per-pair rotation
    /// math is identical between FLUX.1, FLUX.2, and Wan (confirmed 2026-09-19, docs/092) -- only
    /// the frequency TABLE construction (axis count/split/theta) differs -- so FLUX.2's GPU
    /// residency work can dispatch the SAME already-optimized <c>Flux2DRoPE</c> shader FLUX.1 and
    /// Wan already use, needing only this CPU-side table builder, not a new shader.
    /// </summary>
    public static (float[] cos, float[] sin) BuildContextFreqsCompact(
        ReadOnlySpan<int> positions,
        int nTokens,
        int[] axesDim,
        float theta = 2000.0f)
    {
        if (axesDim.Length != 4)
            throw new ArgumentException("FLUX.2 axesDim must contain 4 elements: [dimT, dimH, dimW, dimL].", nameof(axesDim));

        int numAxes = axesDim.Length;
        int headDim = 0;
        for (int a = 0; a < numAxes; a++) headDim += axesDim[a];
        int nPairs = headDim / 2;

        var cos = new float[nTokens * nPairs];
        var sin = new float[nTokens * nPairs];

        var invFreqs = new float[numAxes][];
        for (int a = 0; a < numAxes; a++)
            invFreqs[a] = InterleavedRoPE.ComputeInvFreqs(axesDim[a], theta);

        for (int i = 0; i < nTokens; i++)
        {
            int outOffset = i * nPairs;
            int pairOffset = 0;
            for (int a = 0; a < numAxes; a++)
            {
                int pos = positions[i * numAxes + a];
                var freqs = invFreqs[a];
                int half = freqs.Length;
                for (int j = 0; j < half; j++)
                {
                    float angle = pos * freqs[j];
                    cos[outOffset + pairOffset + j] = MathF.Cos(angle);
                    sin[outOffset + pairOffset + j] = MathF.Sin(angle);
                }
                pairOffset += half;
            }
        }

        return (cos, sin);
    }
}
